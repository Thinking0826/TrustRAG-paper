"""Resume the authorized oracle supplement on one idle GPU, keeping old records."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from run_research_followup import free_gpu, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    dataset = root / "data/research/v2/qasper/test_v1/dataset.json"
    output = root / "data/research/v2/qasper/oracle_v1/results"
    output.mkdir(parents=True, exist_ok=True)
    with (root / "data/research/v2/.supplement.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Existing supplement supervisor retained", flush=True)
            return
        if not (output / "comparison.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "trustrag.research_supplement",
                    "prepare-oracle",
                    str(dataset),
                    "--old-results",
                    str(root / "data/research/v2/qasper/test_v1/results"),
                    "--output",
                    str(output),
                    "--root",
                    str(root),
                ],
                cwd=root,
                check=True,
            )
        comparison = json.loads((output / "comparison.json").read_text())
        for model in comparison["planned_models"]:
            wanted = {
                (query_id, method, 2048)
                for query_id in comparison["model_query_ids"][model]
                for method in comparison["methods"]
            }
            while not wanted <= {
                (row["query_id"], row["method"], row["budget"])
                for row in rows(output / f"answers.{model}.jsonl")
            }:
                gpu = free_gpu()
                if gpu is None:
                    return
                completed = subprocess.run(
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
                        "2048",
                        "--query-limit",
                        "20",
                    ],
                    cwd=root,
                    check=False,
                )
                if completed.returncode:
                    print("Supplement batch stopped; saved failures retained", flush=True)
                    return
                time.sleep(3)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "trustrag.research_supplement",
                "analyze-oracle",
                str(dataset),
                "--output",
                str(output),
            ],
            cwd=root,
            check=True,
        )
        print("Oracle supplement complete; old test results unchanged", flush=True)

        archive = root / "data/research/v2/longbench/raw/data.zip"
        external_dataset = root / "data/research/v2/longbench/multifieldqa_en/dataset.json"
        external_output = root / "data/research/v2/longbench/multifieldqa_en/results"
        if not external_dataset.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "trustrag.longbench_supplement",
                    "prepare",
                    str(external_dataset),
                    "--archive",
                    str(archive),
                ],
                cwd=root,
                check=True,
            )
        if not (external_output / "comparison.json").exists():
            gpu = free_gpu()
            if gpu is None:
                return
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "trustrag.longbench_supplement",
                    "context",
                    str(external_dataset),
                    "--output",
                    str(external_output),
                    "--root",
                    str(root),
                    "--gpu",
                    str(gpu),
                ],
                cwd=root,
                check=True,
            )
            time.sleep(3)
        external_comparison = json.loads((external_output / "comparison.json").read_text())
        for model in external_comparison["planned_models"]:
            for budget in external_comparison["budgets"]:
                wanted = {
                    (query_id, method, budget)
                    for query_id in external_comparison["model_query_ids"][model]
                    for method in external_comparison["methods"]
                }
                while not wanted <= {
                    (row["query_id"], row["method"], row["budget"])
                    for row in rows(external_output / f"answers.{model}.jsonl")
                }:
                    gpu = free_gpu()
                    if gpu is None:
                        return
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            "trustrag.longbench_supplement",
                            "generate",
                            str(external_dataset),
                            "--output",
                            str(external_output),
                            "--root",
                            str(root),
                            "--model",
                            model,
                            "--gpu",
                            str(gpu),
                            "--budget",
                            str(budget),
                            "--query-limit",
                            "20",
                        ],
                        cwd=root,
                        check=False,
                    )
                    if completed.returncode:
                        print("Independent validation stopped; saved records retained", flush=True)
                        return
                    time.sleep(3)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "trustrag.longbench_supplement",
                "analyze",
                str(external_dataset),
                "--output",
                str(external_output),
            ],
            cwd=root,
            check=True,
        )
        print("Independent MultiFieldQA-en validation complete", flush=True)


if __name__ == "__main__":
    main()
