"""Prepare offline, blinded human review of the prespecified QASPER outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def review_keys(plan: dict) -> list[tuple[str, str, str, int]]:
    """Use the declared IDs, including every recorded failure, without score sampling."""
    budget = plan["human_review_budget"]
    methods = plan["human_review_methods"]
    return [
        (model, query, method, budget)
        for model, ids in (
            ("qwen7b", plan["human_primary_query_ids"]),
            ("qwen14b_nf4", plan["human_large_model_query_ids"]),
        )
        for query in ids
        for method in methods
    ]


def build_packet(dataset: dict, contexts: list[dict], answers: dict[str, list[dict]]) -> tuple:
    """Strip method/model/scores; preserve cited original paragraphs and full source text."""
    plan = dataset["study_plan"]
    keys = review_keys(plan)
    questions = {row["query_id"]: row for row in dataset["questions"]}
    context_lookup = {
        (row["query_id"], row["method"], row["budget"]): row for row in contexts
    }
    answer_lookup = {
        (model, row["query_id"], row["method"], row["budget"]): row
        for model, rows in answers.items()
        for row in rows
    }
    missing = [key for key in keys if key not in answer_lookup]
    if missing:
        raise ValueError(f"Human sample is not ready: {len(missing)} prespecified outputs missing")
    random.Random(42).shuffle(keys)
    cases, private_keys = [], []
    documents = {}
    for number, key in enumerate(keys, 1):
        model, query_id, method, budget = key
        query = questions[query_id]
        document_id = query["document_id"]
        document = dataset["documents"][document_id]
        row = answer_lookup[key]
        context = context_lookup[(query_id, method, budget)]
        original = {source["source_id"]: source for source in document["sources"]}
        case_id = f"R{number:03d}"
        evidence = []
        for source in context["sources"]:
            paragraph_ids = list(dict.fromkeys(span["paragraph_id"] for span in source["spans"]))
            # Resolve against the full dataset, not the compressed prompt alone.
            for paragraph_id in paragraph_ids:
                original[paragraph_id]
            evidence.append(
                {
                    "citation_id": source["citation_id"],
                    "selected_text": source["text"],
                    "original_paragraph_ids": paragraph_ids,
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "document_id": document_id,
                "question": query["question"],
                "outcome": row["status"],
                "displayed_answer": row["raw_output"] if row["status"] == "answered" else None,
                "raw_candidate": row["raw_output"],
                "evidence": evidence,
                "alternative_references": query["references"],
            }
        )
        private_keys.append(
            {
                "case_id": case_id,
                "query_id": query_id,
                "document_id": document_id,
                "model": model,
                "method": method,
                "budget": budget,
                "status": row["status"],
            }
        )
        documents[document_id] = {
            "title": document["title"],
            "abstract": document["abstract"],
            "paragraphs": [
                {
                    "paragraph_id": source["source_id"],
                    "section": source["section_name"],
                    "text": source["text"],
                }
                for source in document["sources"]
            ],
        }
    packet = {
        "schema_version": "qasper_blind_review_v1",
        "study_id": plan["study_id"],
        "review_count": len(cases),
        "cases": cases,
        "documents": documents,
        "human_labels_created": False,
    }
    private = {"study_id": plan["study_id"], "cases": private_keys}
    return packet, private


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_packet(output: Path, packet: dict, private: dict) -> None:
    # A new folder prevents replacing actual reviewer work. No audit/hash subsystem.
    output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name("qasper_review.html").read_text(encoding="utf-8")
    encoded = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__PACKET__", encoded), encoding="utf-8")
    (output / "private_key.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    dataset = json.loads(arguments.dataset.read_text(encoding="utf-8"))
    answers = {}
    for model in ("qwen7b", "qwen14b_nf4"):
        path = arguments.results / f"answers.{model}.jsonl"
        answers[model] = read_rows(path) if path.exists() else []
    try:
        packet, private = build_packet(
            dataset, read_rows(arguments.results / "contexts.jsonl"), answers
        )
    except ValueError as error:
        print(f"{error}; no review packet or human labels created")
        return
    write_packet(arguments.output, packet, private)
    print(f"Prepared {packet['review_count']} blinded responses per actual reviewer")
    print("Send only review.html separately to five reviewers; retain private_key.json yourself")
    print("Each independently exports their judgments before discussion; no human labels prefilled")


if __name__ == "__main__":
    main()
