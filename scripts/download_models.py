"""Download only the four fixed study snapshots; never allocate a GPU."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--models", nargs="+", choices=("bge_m3", "bge_reranker", "qwen7b", "qwen14b_nf4"),
                        default=["bge_m3", "bge_reranker", "qwen7b", "qwen14b_nf4"])
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from trustrag.research_models import model_spec

    for key in args.models:
        spec = model_spec(key)
        path = snapshot_download(spec["model_id"], revision=spec["revision"],
                                 cache_dir=str(args.root / "cache/huggingface/hub"),
                                 allow_patterns=spec.get("allow_patterns", ["*.json", "*.safetensors", "merges.txt", "LICENSE"]),
                                 max_workers=1)
        print(f"Ready: {key}: {path}")


if __name__ == "__main__":
    main()

