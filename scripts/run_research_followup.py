"""One serial server workflow: plan, encode controls, select, generate and summarize."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def free_gpu():
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
        gpu, used, free, busy = map(int, line.split(","))
        if used <= 64 and free >= 22000 and busy == 0:
            available.append(gpu)
    if not available:
        print("No idle GPU; existing tasks left untouched", flush=True)
        return None
    return 1 if 1 in available else available[0]


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.exists()
        else []
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/research/v2/qasper/pilot/dataset.json")
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path("data/research/v2/qasper/development_v1/results/retrieval.jsonl"),
    )
    parser.add_argument(
        "--results", type=Path, default=Path("data/research/v2/qasper/development_v2/results")
    )
    parser.add_argument(
        "--reuse", type=Path, default=Path("data/research/v2/qasper/development_v1/results")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    dataset, retrieval, output = (root / p for p in (args.dataset, args.retrieval, args.results))
    reuse = root / args.reuse if args.reuse else None
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".followup.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Existing follow-up supervisor is running; no duplicate task", flush=True)
        return
    wanted = {q["query_id"] for q in json.loads(dataset.read_text())["questions"]}

    def stage(name, gpu=None):
        command = [
            sys.executable,
            "-B",
            "-m",
            "trustrag.qasper_followup",
            name,
            str(dataset),
            "--retrieval",
            str(retrieval),
            "--output",
            str(output),
            "--root",
            str(root),
        ]
        if gpu is not None:
            command += ["--gpu", str(gpu)]
        if name == "context" and reuse:
            command += ["--reuse", str(reuse)]
        subprocess.run(command, cwd=root, check=True)
        if gpu is not None:
            time.sleep(3)

    while not wanted <= {r["query_id"] for r in rows(output / "plans.jsonl")}:
        gpu = free_gpu()
        if gpu is None:
            return
        stage("plan", gpu)
    if not wanted <= {r["query_id"] for r in rows(output / "similarities.jsonl")}:
        gpu = free_gpu()
        if gpu is None:
            return
        stage("similarity", gpu)
    if not (output / "comparison.json").exists():
        stage("context")
    if reuse:
        from trustrag.qasper_followup import reused_answers

        reused_answers(dataset, output, reuse)
    from run_research_development import run

    run(root, dataset, output)
    comparison = json.loads((output / "comparison.json").read_text())
    if comparison["completion"] in ("development_complete", "complete"):
        stage("analyze")
        print(
            "Follow-up matrix complete; evaluate the development result before held-out use",
            flush=True,
        )


if __name__ == "__main__":
    main()
