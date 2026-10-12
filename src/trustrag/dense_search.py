"""Search a TrustRAG dense index and print citation-aware results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trustrag.chunker import load_jsonl


def rank_scores(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return stable descending score indices."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]


def resolve_chunks_path(stored_path: str, project_root: Path) -> Path:
    """Resolve a manifest chunk path against the project root."""
    path = Path(stored_path)
    return path if path.is_absolute() else project_root / path


def model_load_options(manifest: dict[str, Any], device: str) -> dict[str, Any]:
    """Build pinned, offline-only model loading options for retrieval."""
    options: dict[str, Any] = {
        "device": device,
        "local_files_only": True,
        "model_kwargs": {"use_safetensors": False},
    }
    if manifest.get("model_revision"):
        options["revision"] = manifest["model_revision"]
    return options


def search(
    manifest_path: Path,
    query: str,
    top_k: int,
    device: str,
) -> list[dict[str, Any]]:
    """Encode a query, score every chunk, and return ranked results."""
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

    scores = np.asarray(embeddings @ query_embedding, dtype=np.float32)
    results: list[dict[str, Any]] = []
    for rank, row_index in enumerate(rank_scores(scores, top_k), start=1):
        chunk = dict(chunks[int(row_index)])
        chunk["rank"] = rank
        chunk["score"] = float(scores[row_index])
        results.append(chunk)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Dense index manifest JSON")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preview-characters", type=int, default=500)
    return parser


def cli() -> None:
    """Run dense search and print results."""
    arguments = build_parser().parse_args()
    results = search(arguments.manifest, arguments.query, arguments.top_k, arguments.device)
    for result in results:
        pages = (
            str(result["page_start"])
            if result["page_start"] == result["page_end"]
            else f"{result['page_start']}-{result['page_end']}"
        )
        preview = result["text"][: arguments.preview_characters].replace("\n", " ")
        print(
            f"#{result['rank']} score={result['score']:.4f} "
            f"source={result['source_file']} pages={pages} chunk={result['chunk_id']}"
        )
        print(preview)
        print("-" * 80)


if __name__ == "__main__":
    cli()
