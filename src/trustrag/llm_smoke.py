"""Run one bounded, offline Qwen greeting on an idle GPU 0; start no service."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Self

from trustrag.model_download import MODEL_ID, MODEL_REVISION, REQUIRED_FILES, validate_metadata

GIB = 1024**3
MAX_RSS_BYTES = 32 * GIB
MIN_AVAILABLE_BYTES = 64 * GIB
MAX_SECONDS = 300
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 64
GPU_MEMORY_FRACTION = 0.8
GREETING = "请只用一句简短中文回复：你好，TrustRAG 的本地模型推理测试成功。"


def require_environment() -> None:
    """Refuse ambiguous GPU selection or online operation before importing Torch."""
    expected = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run this test inside the server's Ubuntu environment")
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"This smoke test requires {key}={value}")


def read_kib_counter(path: Path, key: str) -> int:
    """Read a Linux proc counter; never substitute zero for unavailable information."""
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == f"{key}:":
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdigit():
                raise RuntimeError(f"Invalid {key} counter in {path}")
            return int(fields[1]) * 1024
    raise RuntimeError(f"Missing {key} counter in {path}")


def guard_violation(rss_bytes: int, elapsed_seconds: float) -> str | None:
    """These sampled guards are not hard OS quotas and can overshoot between samples."""
    if rss_bytes >= MAX_RSS_BYTES:
        return "This Python process reached the 32 GiB RSS guard"
    if elapsed_seconds >= MAX_SECONDS:
        return "This smoke test reached the 300 second runtime guard"
    return None


class ProcessGuard:
    """Sample this process only; terminate only this process if a guard is triggered."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.started = time.monotonic()
        self.peak_rss = 0
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self) -> None:
        while not self.stop_event.wait(0.25):
            try:
                rss = read_kib_counter(Path("/proc/self/status"), "VmRSS")
                self.peak_rss = max(self.peak_rss, rss)
                reason = guard_violation(rss, time.monotonic() - self.started)
            except (OSError, RuntimeError, ValueError) as error:
                reason = f"Cannot monitor this process: {error}"
            if reason:
                os.write(2, f"\n[STOP] {reason}; exiting this process only.\n".encode())
                os._exit(3)

    def __enter__(self) -> Self:
        read_kib_counter(Path("/proc/self/status"), "VmRSS")
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1)


def check_gpu_zero(allow_own_context: bool = False) -> dict[str, int]:
    """Query GPU 0 without changing it. This check is not an exclusive reservation."""

    def query(fields: str, group: str) -> str:
        result = subprocess.run(
            ["nvidia-smi", "-i", "0", f"--query-{group}={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    fields = [
        part.strip()
        for part in query("memory.total,memory.used,memory.free,utilization.gpu", "gpu").split(",")
    ]
    if len(fields) != 4 or any(not value.isdigit() for value in fields):
        raise RuntimeError("Cannot read GPU 0 memory/utilization counters")
    total, used, free, utilization = map(int, fields)
    pids_text = query("pid", "compute-apps")
    lines = pids_text.splitlines() if pids_text else []
    if any(not line.strip().isdigit() for line in lines):
        raise RuntimeError("Cannot read GPU 0 compute process IDs")
    pids = {int(line.strip()) for line in lines}
    allowed = {os.getpid()} if allow_own_context else set()
    if pids - allowed:
        raise RuntimeError("GPU 0 has another compute process; nothing was stopped")
    max_used = 1024 if allow_own_context else 64
    if used > max_used or free < 22 * 1024 or (not allow_own_context and utilization > 0):
        raise RuntimeError("GPU 0 is not sufficiently idle/free; nothing was stopped")
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def check_snapshot(snapshot: Path, root: Path) -> Path:
    """Validate only small metadata and file presence; do not read or hash large weights."""
    snapshot = snapshot.expanduser().resolve()
    root = root.expanduser().resolve()
    if not snapshot.is_relative_to(root) or snapshot.name != MODEL_REVISION:
        raise ValueError("Use the pinned Qwen snapshot inside this project's directory")
    for name in REQUIRED_FILES:
        path = snapshot / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty local model file: {name}")
    validate_metadata(snapshot)
    return snapshot


def generate_greeting(model: Any, tokenizer: Any) -> dict[str, Any]:
    """Use the chat template, batch size one, greedy decoding, and new-token-only output."""
    import torch
    from transformers import GenerationConfig

    messages = [
        {"role": "system", "content": "你是一个简洁的中文助手。"},
        {"role": "user", "content": GREETING},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)
    if encoded["input_ids"].shape[0] != 1:
        raise RuntimeError("Expected exactly one input")
    input_count = encoded["input_ids"].shape[1]
    if not 0 < input_count <= MAX_INPUT_TOKENS:
        raise ValueError("Smoke test input exceeds the 1024 token bound; not truncated")
    encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise RuntimeError("Model has no EOS token configuration")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    # A fresh configuration avoids inheriting the model's sampling defaults.
    config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, generation_config=config)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    generated = output[0, input_count:]
    count = int(generated.numel())
    answer = tokenizer.decode(generated.detach().cpu(), skip_special_tokens=True).strip()
    if not answer or not 0 < count <= MAX_NEW_TOKENS:
        raise RuntimeError("Generation was empty or violated the output bound")
    return {
        "answer": answer,
        "input_tokens": input_count,
        "generated_tokens": count,
        "generation_seconds": round(elapsed, 3),
    }


def run_smoke(snapshot: Path) -> dict[str, Any]:
    """Load on CPU in BF16, then move to the only visible GPU, without Accelerate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This smoke test requires CUDA BF16 support")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    print("Loading Qwen from the local snapshot in BF16...", flush=True)
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=False,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Model weights did not load cleanly: {loading}")
    # Recheck after the CPU load: another user may have started a job in the meantime.
    check_gpu_zero(allow_own_context=True)
    model.to("cuda:0")
    model.eval()
    torch.cuda.synchronize(0)
    load_seconds = time.perf_counter() - started
    result = generate_greeting(model, tokenizer)
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "load_seconds": round(load_seconds, 3),
        **result,
        "peak_torch_allocated_gib": round(torch.cuda.max_memory_allocated(0) / GIB, 3),
        "peak_torch_reserved_gib": round(torch.cuda.max_memory_reserved(0) / GIB, 3),
    }


def cli() -> None:
    root = Path(os.environ.get("TRUSTRAG_ROOT", Path.cwd()))
    default = root / "cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=default / "snapshots" / MODEL_REVISION)
    parser.add_argument("--check-only", action="store_true", help="No Torch import or model load")
    args = parser.parse_args()
    require_environment()
    snapshot = check_snapshot(args.model_dir, root)
    available = read_kib_counter(Path("/proc/meminfo"), "MemAvailable")
    if available < MIN_AVAILABLE_BYTES:
        raise RuntimeError("Less than 64 GiB MemAvailable; no model was loaded")
    gpu = check_gpu_zero()
    print(f"Snapshot: {snapshot}", flush=True)
    print(f"Host MemAvailable: {available / GIB:.1f} GiB (not a container quota)", flush=True)
    print(f"GPU 0 before load: {gpu}", flush=True)
    print("CPU pools: 4 threads per Torch/OMP/BLAS pool; interop threads: 1", flush=True)
    print("Token limits: 1024 input tokens; 64 new tokens; batch size 1", flush=True)
    print("Torch allocator cap: 80% of GPU 0; not a total driver-memory cap", flush=True)
    print("Sampled guards: 32 GiB process RSS / 300 seconds; not hard OS quotas", flush=True)
    print(
        "Cgroup quota not verified: /proc counters and GPU checks do not prove it unlimited",
        flush=True,
    )
    if args.check_only:
        print(
            "Preflight passed; no model loaded and no GPU memory allocated by this script",
            flush=True,
        )
        return
    with ProcessGuard() as guard:
        result = run_smoke(snapshot)
        result["sampled_peak_process_rss_gib"] = round(guard.peak_rss / GIB, 3)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("Smoke test passed; process exiting (this is not a RAG accuracy evaluation)", flush=True)


if __name__ == "__main__":
    cli()
