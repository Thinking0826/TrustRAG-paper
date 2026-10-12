"""Fuse BGE-M3 dense retrieval and BM25 with reciprocal rank fusion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trustrag.bm25 import BM25Index, positive_ranking
from trustrag.chunker import load_jsonl
from trustrag.dense_search import model_load_options, rank_scores, resolve_chunks_path


def rank_positions(ranking: list[int]) -> dict[int, int]:
    """Map document indices to one-based ranks."""
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate document indices")
    return {document_index: rank for rank, document_index in enumerate(ranking, start=1)}


def reciprocal_rank_fusion(
    rankings: list[tuple[list[int], float]],
    document_count: int,
    rrf_k: int = 60,
) -> np.ndarray:
    """Fuse ranked lists into a full score vector."""
    if document_count <= 0:
        raise ValueError("document_count must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    fused_scores = np.zeros(document_count, dtype=np.float32)
    for ranking, weight in rankings:
        if weight < 0:
            raise ValueError("RRF weights must be non-negative")
        positions = rank_positions(ranking)
        if any(index < 0 or index >= document_count for index in positions):
            raise ValueError("Ranking contains an out-of-range document index")
        for document_index, rank in positions.items():
            fused_scores[document_index] += weight / (rrf_k + rank)
    return fused_scores


def hybrid_rank(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    top_k: int,
    candidate_k: int = 30,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """Return hybrid indices, fused scores, and component rank maps."""
    if dense_scores.ndim != 1 or sparse_scores.ndim != 1:
        raise ValueError("Score vectors must be one-dimensional")
    if len(dense_scores) != len(sparse_scores):
        raise ValueError("Dense and sparse score lengths do not match")
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to positive top_k")

    effective_candidate_k = min(candidate_k, len(dense_scores))
    dense_ranking = [
        int(index) for index in rank_scores(dense_scores, effective_candidate_k)
    ]
    sparse_ranking = positive_ranking(sparse_scores, effective_candidate_k)
    fused_scores = reciprocal_rank_fusion(
        [(dense_ranking, dense_weight), (sparse_ranking, sparse_weight)],
        document_count=len(dense_scores),
        rrf_k=rrf_k,
    )
    indices = rank_scores(fused_scores, top_k)
    return (
        indices,
        fused_scores,
        rank_positions(dense_ranking),
        rank_positions(sparse_ranking),
    )


def search(
    manifest_path: Path,
    query: str,
    top_k: int,
    device: str,
    candidate_k: int,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Run offline dense and BM25 retrieval followed by RRF."""
    from sentence_transformers import SentenceTransformer

    manifest_path = manifest_path.expanduser().resolve()
    project_root = Path(os.environ.get("TRUSTRAG_ROOT", Path.cwd())).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks_path = resolve_chunks_path(manifest["chunks_file"], project_root)
    embeddings_path = manifest_path.parent / manifest["embeddings_file"]
    chunks = load_jsonl(chunks_path)
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if len(chunks) != embeddings.shape[0] or len(chunks) != manifest["row_count"]:
        raise RuntimeError("Chunk and embedding row counts do not match")

    model = SentenceTransformer(
        manifest["model_name"],
        **model_load_options(manifest, device),
    )
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)
    dense_scores = np.asarray(embeddings @ query_embedding, dtype=np.float32)
    bm25_index = BM25Index.from_texts([chunk["text"] for chunk in chunks])
    sparse_scores = bm25_index.scores(query)
    indices, fused_scores, dense_ranks, sparse_ranks = hybrid_rank(
        dense_scores,
        sparse_scores,
        top_k,
        candidate_k,
        dense_weight,
        sparse_weight,
        rrf_k,
    )

    results: list[dict[str, Any]] = []
    for rank, row_index in enumerate(indices, start=1):
        index = int(row_index)
        chunk = dict(chunks[index])
        chunk.update(
            {
                "rank": rank,
                "score": float(fused_scores[index]),
                "dense_score": float(dense_scores[index]),
                "bm25_score": float(sparse_scores[index]),
                "dense_rank": dense_ranks.get(index),
                "bm25_rank": sparse_ranks.get(index),
            }
        )
        results.append(chunk)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Dense index manifest JSON")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preview-characters", type=int, default=500)
    return parser


def cli() -> None:
    """Run hybrid search and print citation-aware results."""
    arguments = build_parser().parse_args()
    results = search(
        arguments.manifest,
        arguments.query,
        arguments.top_k,
        arguments.device,
        arguments.candidate_k,
        arguments.dense_weight,
        arguments.sparse_weight,
        arguments.rrf_k,
    )
    for result in results:
        pages = (
            str(result["page_start"])
            if result["page_start"] == result["page_end"]
            else f"{result['page_start']}-{result['page_end']}"
        )
        preview = result["text"][: arguments.preview_characters].replace("\n", " ")
        print(
            f"#{result['rank']} rrf={result['score']:.6f} "
            f"dense_rank={result['dense_rank']} bm25_rank={result['bm25_rank']} "
            f"source={result['source_file']} pages={pages} chunk={result['chunk_id']}"
        )
        print(preview)
        print("-" * 80)


if __name__ == "__main__":
    cli()
