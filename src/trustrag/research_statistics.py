"""Paired document-cluster intervals, saved in the existing research summary."""

from __future__ import annotations

import argparse
import fcntl
import json
import random
from collections import defaultdict
from pathlib import Path

SAMPLES = 5000
SEED = 42
METRICS = (
    "answer_token_f1",
    "answer_unit_recall",
    "all_answer_units_matched",
    "evidence_paragraph_f1",
)


def percentile(values: list[float], fraction: float) -> float:
    position = fraction * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_cluster_interval(
    pairs: list[tuple[str, float, float]], samples: int = SAMPLES, seed: int = SEED
) -> dict:
    """Resample papers; keep every paired question from a sampled paper together."""
    groups = defaultdict(list)
    for document, left, right in pairs:
        groups[document].append(float(left) - float(right))
    n = len(pairs)
    totals = [(sum(values), len(values)) for _, values in sorted(groups.items())]
    estimate = {
        "question_count": n,
        "document_count": len(totals),
        "left_mean": sum(left for _, left, _ in pairs) / n,
        "right_mean": sum(right for _, _, right in pairs) / n,
        "difference": sum(total for total, _ in totals) / n,
        "ci95": None,
    }
    if len(totals) < 2:
        estimate["interval_note"] = "Fewer than two independent document clusters"
        return estimate
    generator = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = generator.choices(totals, k=len(totals))
        draws.append(sum(total for total, _ in selected) / sum(count for _, count in selected))
    draws.sort()
    estimate["ci95"] = [percentile(draws, 0.025), percentile(draws, 0.975)]
    return estimate


def method_analysis(dataset: dict, comparison: dict, answers: dict[str, list[dict]]) -> dict:
    """Analyze only complete model matrices; failed recorded responses stay in denominators."""
    documents = {row["query_id"]: row["document_id"] for row in dataset["questions"]}
    analysis = {
        "scope": "exploratory automatic metrics; not human semantic accuracy or confirmatory tests",
        "pair": "closure_greedy minus density_atoms, matched question and budget",
        "estimator": "question-weighted mean paired difference",
        "interval_method": "percentile paired bootstrap with documents resampled as clusters",
        "bootstrap_samples": SAMPLES,
        "seed": SEED,
        "multiple_comparisons": "Exploratory intervals; no significance claims or p-values",
        "failure_policy": "Recorded refusal, rejection and error metrics remain zero; no reruns",
        "models": {},
        "human_evaluation": {
            "status": "pending_actual_independent_reviews",
            "primary_outcome": "complete_correct_and_source_supported_answer",
            "human_labels_created": False,
        },
    }
    recorded_human = comparison.get("paired_analysis", {}).get("human_evaluation")
    if recorded_human:
        analysis["human_evaluation"] = dict(recorded_human)
    for model in comparison["planned_models"]:
        ids = comparison.get("model_query_ids", {}).get(model, list(documents))
        selected_documents = {query: documents[query] for query in ids}
        wanted = {
            (query, method, budget)
            for query in ids
            for method in comparison["methods"]
            for budget in comparison["budgets"]
        }
        rows = {
            (row["query_id"], row["method"], row["budget"]): row for row in answers.get(model, [])
        }
        if not wanted <= rows.keys():
            analysis["models"][model] = {
                "status": "pending_complete_matrix",
                "recorded_conditions": len(wanted & rows.keys()),
                "planned_conditions": len(wanted),
            }
            continue
        budgets = {}
        for budget in comparison["budgets"]:
            metrics = {}
            for metric in METRICS:
                pairs = [
                    (
                        document,
                        rows[(query, "closure_greedy", budget)]["metrics"][metric],
                        rows[(query, "density_atoms", budget)]["metrics"][metric],
                    )
                    for query, document in selected_documents.items()
                ]
                metrics[metric] = paired_cluster_interval(pairs)
            budgets[str(budget)] = metrics
        analysis["models"][model] = {
            "status": "complete_matrix_analyzed",
            "recorded_conditions": len(wanted),
            "budgets": budgets,
        }
    return analysis


def deployment_analysis(dataset: dict, comparison: dict, answers: dict[str, list[dict]]) -> dict:
    """Compare deployments on their declared common questions, not unequal sample means."""
    left, right = "qwen14b_nf4", "qwen7b"
    analysis = {
        "scope": "secondary exploratory automatic metrics; not human semantic accuracy",
        "pair": "qwen14b_nf4 minus qwen7b, matched question, method and budget",
        "configuration_confound": "14B NF4 versus 7B BF16; not a pure parameter-count effect",
        "estimator": "question-weighted mean paired difference",
        "interval_method": "percentile paired bootstrap with documents resampled as clusters",
        "bootstrap_samples": SAMPLES,
        "seed": SEED,
        "failure_policy": "Every recorded refusal, rejection and error remains in the denominator",
        "multiple_comparisons": "Exploratory intervals; no significance claims or p-values",
    }
    documents = {row["query_id"]: row["document_id"] for row in dataset["questions"]}
    scopes = {
        model: comparison.get("model_query_ids", {}).get(model, list(documents))
        for model in (left, right)
    }
    shared = [query for query in documents if query in scopes[left] and query in scopes[right]]
    if not shared:
        return {**analysis, "status": "no_shared_questions"}
    lookup = {}
    for model in (left, right):
        lookup[model] = {
            (row["query_id"], row["method"], row["budget"]): row
            for row in answers.get(model, [])
        }
        wanted = {
            (query, method, budget)
            for query in scopes[model]
            for method in comparison["methods"]
            for budget in comparison["budgets"]
        }
        if not wanted <= lookup[model].keys():
            return {**analysis, "status": "pending_complete_matrices"}
    configurations = {}
    for budget in comparison["budgets"]:
        for method in comparison["methods"]:
            configurations[f"{method}@{budget}"] = {
                metric: paired_cluster_interval(
                    [
                        (
                            documents[query],
                            lookup[left][(query, method, budget)]["metrics"][metric],
                            lookup[right][(query, method, budget)]["metrics"][metric],
                        )
                        for query in shared
                    ]
                )
                for metric in METRICS
            }
    return {
        **analysis,
        "status": "complete_paired_subset_analyzed",
        "question_count": len(shared),
        "document_count": len({documents[query] for query in shared}),
        "configurations": configurations,
    }


def input_activation(contexts: list[dict], comparison: dict) -> dict:
    """Describe saved input changes only; no selection, generation, or score refresh."""
    rows = {(r["query_id"], r["method"], r["budget"]): r for r in contexts}
    budgets = {}
    for budget in comparison["budgets"]:
        pairs = [
            (row, rows[(query, "density_atoms", budget)])
            for (query, method, limit), row in rows.items()
            if method == "closure_greedy" and limit == budget
        ]
        n = len(pairs)
        identical_messages = sum(left["messages"] == right["messages"] for left, right in pairs)
        budgets[str(budget)] = {
            "question_count": n,
            "identical_sources": sum(left["sources"] == right["sources"] for left, right in pairs),
            "identical_messages": identical_messages,
            "changed_messages": n - identical_messages,
            "changed_message_rate": (n - identical_messages) / n if n else None,
        }
    return {
        "scope": "post hoc descriptive audit of stored inputs; not a new confirmatory endpoint",
        "pair": "closure_greedy versus density_atoms, matched question and budget",
        "basis": "direct equality of saved sources and complete rendered messages",
        "budgets": budgets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--input-audit-only", action="store_true")
    arguments = parser.parse_args()
    with (arguments.results / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Generation is still running; no summary changed")
            return
        comparison_path = arguments.results / "comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if arguments.input_audit_only:
            contexts = [
                json.loads(line)
                for line in (arguments.results / "contexts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            comparison["input_activation"] = input_activation(contexts, comparison)
            comparison_path.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print("Saved post hoc input audit in comparison.json; existing scores unchanged")
            return
        answers = {}
        for model in comparison["planned_models"]:
            path = arguments.results / f"answers.{model}.jsonl"
            answers[model] = (
                [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                if path.exists()
                else []
            )
        dataset = json.loads(arguments.dataset.read_text(encoding="utf-8"))
        comparison["paired_analysis"] = method_analysis(dataset, comparison, answers)
        comparison["deployment_analysis"] = deployment_analysis(dataset, comparison, answers)
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for model, result in comparison["paired_analysis"]["models"].items():
            print(f"{model}: {result['status']}")
        print(f"Paired deployments: {comparison['deployment_analysis']['status']}")
        print("Updated the existing comparison.json; no human labels or separate report created")


if __name__ == "__main__":
    main()
