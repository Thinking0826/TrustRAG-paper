"""Resume the existing development matrix, without rerunning recorded answers."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def run(root: Path, dataset: Path, output: Path) -> None:
    if not (output / "comparison.json").exists():
        print("Waiting for the existing context construction task; no new task started")
        return
    comparison = json.loads((output / "comparison.json").read_text())
    contexts = [
        json.loads(line) for line in (output / "contexts.jsonl").read_text().splitlines() if line
    ]
    for model in comparison["planned_models"]:
        scoped_ids = comparison.get("model_query_ids", {}).get(model)
        allowed = set(scoped_ids) if scoped_ids is not None else None
        if model == "qwen14b_nf4":
            from trustrag.qasper_retrieval import snapshot
            from trustrag.research_models import model_spec

            try:
                path = snapshot(root, model_spec(model))
            except StopIteration:
                print("14B download is pending; no duplicate download or GPU load")
                continue
            if not all(
                (path / f"model-{i:05d}-of-00008.safetensors").is_file() for i in range(1, 9)
            ):
                print("14B shards are still downloading; no duplicate download or GPU load")
                continue
        for budget in (2048, 1024, 4096):
            wanted = {
                (r["query_id"], r["method"], r["budget"])
                for r in contexts
                if r["budget"] == budget and (allowed is None or r["query_id"] in allowed)
            }
            while True:
                answers = output / f"answers.{model}.jsonl"
                recorded = (
                    {
                        (r["query_id"], r["method"], r["budget"])
                        for line in answers.read_text().splitlines()
                        if line and (r := json.loads(line))
                    }
                    if answers.exists()
                    else set()
                )
                if wanted <= recorded:
                    break
                counters = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                )
                available = []
                for line in counters.splitlines():
                    gpu, used, free, utilization = map(int, line.split(","))
                    if used <= 64 and free >= 22000 and utilization == 0:
                        available.append(gpu)
                if not available:
                    print("Both GPUs are occupied; returning without stopping another process")
                    return
                gpu = 1 if 1 in available else available[0]
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "trustrag.qasper_pilot",
                        "generate",
                        str(dataset),
                        "--output",
                        str(output),
                        "--root",
                        str(root),
                        "--model",
                        model,
                        "--gpu",
                        str(gpu),
                        "--budget",
                        str(budget),
                        "--query-limit",
                        "2",
                    ],
                    cwd=root,
                    check=False,
                )
                if result.returncode:
                    print(
                        f"Development batch stopped with exit {result.returncode}; saved records retained"
                    )
                    return
                # The driver can briefly retain a finished child's allocations.
                time.sleep(3)
    print(
        "Available development stages finished; inspect the single comparison.json for completion"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--results", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    dataset = arguments.dataset or root / "data/research/v2/qasper/pilot/dataset.json"
    output = arguments.results or root / "data/research/v2/qasper/development_v1/results"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Existing development generation worker retained; no duplicate started")
        else:
            run(root, dataset, output)
