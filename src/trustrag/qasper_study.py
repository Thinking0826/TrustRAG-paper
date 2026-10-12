"""Prepare the prespecified official-test sample, without reading model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustrag.qasper_pilot import prepare


def study_fields(query_ids: list[str]) -> dict:
    """The seed-42 eligibility shuffle already determines all nested random subsets."""
    larger = query_ids[:100]
    return {
        "model_query_ids": {"qwen7b": query_ids, "qwen14b_nf4": larger},
        "study_plan": {
            "study_id": "qasper_test_v1",
            "declared_date": "2026-10-09",
            "eligibility_and_sample_seed": 42,
            "main_budget": 2048,
            "primary_method_pair": ["closure_greedy", "density_atoms"],
            "primary_outcome": "human_adjudicated_complete_correct_source_supported_answer",
            "human_primary_query_ids": larger[:60],
            "human_large_model_query_ids": larger[:30],
            "human_review_methods": ["closure_greedy", "density_atoms"],
            "human_review_budget": 2048,
            "planned_reviewers": 2,
            "human_labels_created": False,
            "automatic_comparisons": "Secondary, exploratory; all four methods and three budgets",
            "model_comparison_scope": "7B BF16 versus 14B NF4 deployment configurations",
            "test_output_policy": "No tuning, answer-alias edits, selection changes or score-driven reruns",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.archive, "test", 300, arguments.output)
    dataset = json.loads(arguments.output.read_text(encoding="utf-8"))
    ids = [row["query_id"] for row in dataset["questions"]]
    dataset.update(study_fields(ids))
    arguments.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Frozen model scopes: 7B {len(ids)} questions; "
        f"14B {len(dataset['model_query_ids']['qwen14b_nf4'])} paired questions"
    )
    print("Human review scopes recorded before outputs; no human labels filled")


if __name__ == "__main__":
    main()
