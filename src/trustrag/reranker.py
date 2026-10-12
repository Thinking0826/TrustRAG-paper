"""Offline cross-encoder reranking for TrustRAG evidence candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def candidate_union(
    dense_ranks: dict[int, int],
    sparse_ranks: dict[int, int],
    fusion_scores: np.ndarray,
) -> np.ndarray:
    """Keep both candidate lists in full, ordered by fusion score for tie-breaking."""
    if fusion_scores.ndim != 1:
        raise ValueError("fusion_scores must be one-dimensional")
    indices = np.asarray(sorted(set(dense_ranks) | set(sparse_ranks)), dtype=np.int64)
    if len(indices) == 0:
        return indices
    if indices[0] < 0 or indices[-1] >= len(fusion_scores):
        raise ValueError("Candidate index is out of range")
    if not np.isfinite(fusion_scores[indices]).all():
        raise ValueError("Candidate fusion scores must be finite")
    return indices[np.argsort(-fusion_scores[indices], kind="stable")]


def rerank_indices(
    candidate_indices: list[int] | np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Return candidate indices sorted by stable descending reranker score."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if len(candidate_indices) != len(scores):
        raise ValueError("candidate_indices and scores must have equal lengths")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) != len(set(indices.tolist())):
        raise ValueError("candidate_indices must be one-dimensional and unique")
    if not np.isfinite(scores).all():
        raise ValueError("Reranker scores must be finite")
    return indices[np.argsort(-scores, kind="stable")[: min(top_k, len(indices))]]


@dataclass(slots=True)
class LocalReranker:
    """A locally loaded Transformers sequence-classification reranker."""

    tokenizer: Any
    model: Any
    device: str
    max_length: int = 512

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str,
        revision: str | None = None,
        max_length: int = 512,
    ) -> LocalReranker:
        """Load a reranker from a local model directory or offline cache."""
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        source = str(Path(model_path).expanduser())
        tokenizer_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast": True,
        }
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
        }
        if revision:
            tokenizer_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(source, **model_kwargs)
        model.to(device)
        model.eval()
        return cls(tokenizer, model, device, max_length)

    def score_pairs(
        self,
        query: str,
        texts: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Score query-text pairs in batches and return raw ranking logits."""
        if not query.strip():
            raise ValueError("query is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty(0, dtype=np.float32)

        import torch

        scores: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation="only_second",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.shape not in {(len(batch_texts),), (len(batch_texts), 1)}:
                    raise RuntimeError("Expected one relevance logit per query-text pair")
                batch_scores = logits.reshape(-1)
                scores.append(batch_scores.detach().float().cpu().numpy())
        result = np.concatenate(scores).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise RuntimeError("Reranker returned non-finite scores")
        return result
