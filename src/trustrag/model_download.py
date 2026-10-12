"""Download one pinned Qwen snapshot serially, without loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
HUB_ENDPOINT = "https://huggingface.co"
INDEX_FILE = "model.safetensors.index.json"
WEIGHT_FILES = tuple(f"model-{number:05d}-of-00004.safetensors" for number in range(1, 5))
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    INDEX_FILE,
    *WEIGHT_FILES,
)
DISK_RESERVE_BYTES = 1024**3


def remote_file_sizes(info: Any) -> dict[str, int]:
    """Accept only the intended commit and a complete, fixed file inventory."""
    if info.sha != MODEL_REVISION:
        raise ValueError("The Hub returned a different model revision")
    files = {entry.rfilename: entry for entry in info.siblings or []}
    sizes: dict[str, int] = {}
    for name in REQUIRED_FILES:
        if name not in files:
            raise ValueError(f"Required file is missing from the model repository: {name}")
        size = files[name].size
        if type(size) is not int or size <= 0:
            raise ValueError(f"Remote file size is missing or invalid: {name}")
        sizes[name] = size
    return sizes


def validate_metadata(snapshot: Path) -> None:
    """Inspect small JSON files only; do not deserialize any model tensors."""
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2":
        raise ValueError("Expected model_type=qwen2")
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not isinstance(tokenizer.get("chat_template"), str) or not tokenizer["chat_template"]:
        raise ValueError("Tokenizer chat_template is missing")
    index = json.loads((snapshot / INDEX_FILE).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Weight index has no weight_map")
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise ValueError("Weight index contains invalid shard names")
    if set(weight_map.values()) != set(WEIGHT_FILES):
        raise ValueError("Weight index does not reference the four expected safetensors shards")


def validate_snapshot(snapshot: Path, expected_sizes: dict[str, int]) -> None:
    """Check file completeness by remote size, not by a full weight checksum."""
    if snapshot.name != MODEL_REVISION:
        raise ValueError("Snapshot directory does not match the pinned revision")
    for name in REQUIRED_FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded file is missing: {name}")
        if path.stat().st_size != expected_sizes[name]:
            raise ValueError(f"File size does not match the Hub metadata: {name}")
    validate_metadata(snapshot)


def download_model(cache_dir: Path) -> dict[str, Any]:
    """Use cache-aware HTTP downloads, one file at a time, without force-download."""
    from huggingface_hub import HfApi, hf_hub_download

    cache_dir = cache_dir.expanduser().resolve()
    api = HfApi(endpoint=HUB_ENDPOINT, token=False)
    info = api.model_info(
        MODEL_ID, revision=MODEL_REVISION, files_metadata=True, timeout=30, token=False
    )
    expected_sizes = remote_file_sizes(info)
    snapshot = cache_dir / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REVISION
    required_bytes = sum(
        size
        for name, size in expected_sizes.items()
        if not (snapshot / name).is_file() or (snapshot / name).stat().st_size != size
    )
    existing_parent = cache_dir
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    free_bytes = shutil.disk_usage(existing_parent).free
    if free_bytes < required_bytes + DISK_RESERVE_BYTES:
        raise OSError("Not enough free disk space for the remaining files plus a 1 GiB reserve")

    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Revision: {MODEL_REVISION}", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    print(f"Required files: {len(REQUIRED_FILES)}", flush=True)
    print(f"Total model files: {sum(expected_sizes.values()) / 1024**3:.2f} GiB", flush=True)
    print("Download mode: one file at a time; existing cache reused; no GPU/model load", flush=True)
    for position, name in enumerate(REQUIRED_FILES, start=1):
        print(f"[{position}/{len(REQUIRED_FILES)}] {name}", flush=True)
        downloaded = Path(
            hf_hub_download(
                MODEL_ID,
                filename=name,
                revision=MODEL_REVISION,
                cache_dir=str(cache_dir),
                endpoint=HUB_ENDPOINT,
                token=False,
                force_download=False,
                local_files_only=False,
                etag_timeout=30,
            )
        )
        if downloaded.parent.resolve() != snapshot.resolve() or downloaded.name != name:
            raise ValueError("Download returned an unexpected snapshot path")
        if not downloaded.is_file() or downloaded.stat().st_size != expected_sizes[name]:
            raise ValueError(f"Downloaded file is incomplete or has the wrong size: {name}")
        # Inspect the architecture and complete shard list before fetching large weights.
        if name == INDEX_FILE:
            validate_metadata(snapshot)
    validate_snapshot(snapshot, expected_sizes)
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot": str(snapshot),
        "file_count": len(REQUIRED_FILES),
        "weight_shard_count": len(WEIGHT_FILES),
        "validation": "remote_file_sizes_and_config_index_structure",
        "full_weight_checksum_verified": False,
    }


def cli() -> None:
    """Download only into this project's cache; leave environment packages unchanged."""
    root = Path(os.environ.get("TRUSTRAG_ROOT", Path.cwd())).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "huggingface" / "hub")
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    if cache_dir == root or not cache_dir.is_relative_to(root):
        raise ValueError("Model cache must be a subdirectory of TRUSTRAG_ROOT")
    result = download_model(cache_dir)
    print("Download complete", flush=True)
    print(f"Weight shards: {result['weight_shard_count']}", flush=True)
    print("Validation: file sizes and config/index checked; no full weight checksum", flush=True)
    print(f"Model cache: {result['snapshot']}", flush=True)


if __name__ == "__main__":
    cli()
