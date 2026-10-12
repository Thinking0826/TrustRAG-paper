"""Research-only model extension; frozen engineering benchmark registry is unchanged."""

from __future__ import annotations

import argparse
from pathlib import Path

QWEN14 = {
    "model_id": "Qwen/Qwen2.5-14B-Instruct",
    "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "license": "apache-2.0",
    "quantization": "bnb_nf4",
}


def model_spec(key: str) -> dict:
    from trustrag.benchmark_models import MODELS

    return dict(QWEN14 if key == "qwen14b_nf4" else MODELS[key])


def download(root: Path) -> None:
    from huggingface_hub import snapshot_download

    print("Downloading only the approved Qwen 14B revision; no GPU model load", flush=True)
    path = snapshot_download(
        QWEN14["model_id"],
        revision=QWEN14["revision"],
        cache_dir=str(root / "cache/huggingface/hub"),
        max_workers=1,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "LICENSE"],
    )
    print(f"14B snapshot ready: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    download(args.root)
