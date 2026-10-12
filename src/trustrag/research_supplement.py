"""Prespecified evidence-oracle and independent-dataset diagnostic supplements.

Frozen QASPER records are read or copied, never refreshed. Oracle input contains
whole annotated paragraphs, not answer spans. Every run keeps one comparison.json.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from trustrag.bundle_budget import EvidenceAtom, original_pieces
from trustrag.qasper_pilot import (
    PROMPT_PROFILE,
    answer_tokens,
    make_messages,
    reference_coverage,
)
from trustrag.research_statistics import paired_cluster_interval

ORACLE_METHODS = ("density_atoms", "closure_greedy", "oracle_evidence")
BUDGET = 2048


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_json(path: Path, value: dict, *, new: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if new else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def oracle_pieces(query: dict, document: dict) -> list[dict]:
    """Use the first original annotation, independently of scores or output fit."""
    ids = set(query["references"][0]["evidence_paragraph_ids"])
    sources = [source for source in document["sources"] if source["source_id"] in ids]
    if {source["source_id"] for source in sources} != ids:
        raise ValueError("Annotated evidence is missing from the original document")
    known = {
        source["source_id"]: EvidenceAtom(
            source["source_id"], source["source_id"], 0, len(source["text"]), 1.0
        )
        for source in sources
    }
    return [
        {**piece, "citation_id": f"E{index}"}
        for index, piece in enumerate(
            original_pieces(set(known), known, {s["source_id"]: s for s in sources}), 1
        )
    ]


def prepare_oracle(dataset_path: Path, old: Path, output: Path, tokenizer) -> None:
    if (output / "comparison.json").exists():
        print("Existing oracle preparation retained", flush=True)
        return
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries = dataset["questions"]
    ids = {query["query_id"] for query in queries}
    contexts = [
        row
        for row in read_rows(old / "contexts.jsonl")
        if row["query_id"] in ids
        and row["budget"] == BUDGET
        and row["method"] in ORACLE_METHODS[:2]
    ]
    if len(contexts) != len(queries) * 2:
        raise ValueError("The two frozen 2K controls must be complete before reuse")
    started = time.monotonic()
    for query in queries:
        sources = oracle_pieces(query, dataset["documents"][query["document_id"]])
        messages = make_messages(query["question"], sources)
        tokens = len(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        contexts.append(
            {
                "query_id": query["query_id"],
                "document_id": query["document_id"],
                "question": query["question"],
                "method": "oracle_evidence",
                "budget": BUDGET,
                "input_tokens": tokens,
                "sources": sources,
                "messages": messages,
                "selection_prompt_profile": PROMPT_PROFILE,
                "oracle_annotation_index": 0,
                "oracle_fits_budget": tokens <= BUDGET,
                "after_budget": reference_coverage(sources, query["references"]),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    # Interrupted preparation may be repeated, but it never touches generated answers.
    with (output / "contexts.jsonl").open("w", encoding="utf-8") as stream:
        for row in contexts:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    models = ("qwen7b", "qwen14b_nf4")
    scopes = dataset["model_query_ids"]
    for model in models:
        allowed = set(scopes[model])
        selected = [
            row
            for row in read_rows(old / f"answers.{model}.jsonl")
            if row["query_id"] in allowed
            and row["budget"] == BUDGET
            and row["method"] in ORACLE_METHODS[:2]
        ]
        if len(selected) != len(allowed) * 2:
            raise ValueError(f"Incomplete frozen control records for {model}")
        destination = output / f"answers.{model}.jsonl"
        existing = {
            (row["query_id"], row["method"], row["budget"])
            for row in read_rows(destination)
        }
        with destination.open("a", encoding="utf-8") as stream:
            for row in selected:
                if (row["query_id"], row["method"], row["budget"]) not in existing:
                    stream.write(
                        json.dumps({**row, "reused_from": str(old)}, ensure_ascii=False) + "\n"
                    )
    save_json(
        output / "comparison.json",
        {
            "dataset": "QASPER 0.3 evidence-oracle supplement",
            "official_split": "test",
            "completion": "partial",
            "stage": "oracle_contexts_ready",
            "analysis_scope": "posthoc diagnostic of previously evaluated questions",
            "context_record_count": len(contexts),
            "question_count": len(queries),
            "methods": list(ORACLE_METHODS),
            "budgets": [BUDGET],
            "planned_models": list(models),
            "model_query_ids": scopes,
            "generation": {"status": "partial"},
            "selection_seconds": round(time.monotonic() - started, 3),
            "oracle_policy": "First original annotation; whole paragraphs in document order. "
            "No answer text appended. Over-budget evidence is not truncated and is recorded "
            "as an execution error by the unchanged generator.",
            "fitting_oracle_questions": sum(
                row.get("oracle_fits_budget", False) for row in contexts
            ),
            "new_planned_records": sum(len(scopes[model]) for model in models),
            "reused_record_count": 2 * sum(len(scopes[model]) for model in models),
            "prompt_profile": PROMPT_PROFILE,
            "metrics_scope": "Token overlap and literal references, not human semantic accuracy",
        },
    )
    print(
        f"Oracle ready: {len(queries)} questions, "
        f"{sum(row.get('oracle_fits_budget', False) for row in contexts)} fit 2K; "
        f"{2 * sum(len(scopes[model]) for model in models)} controls reused, "
        f"{sum(len(scopes[model]) for model in models)} new records planned",
        flush=True,
    )


def overlap_details(row: dict, references: list[dict]) -> dict:
    """Reference-token precision/recall explain F1 algebra, not semantic correctness."""
    answer = row.get("candidate") or {}
    text = answer.get("answer", "") if row["status"] == "answered" else ""
    tokens = answer_tokens(re.sub(r"\[E\d+\]", "", text))
    alternatives = []
    for reference in references:
        expected = answer_tokens(" ".join(reference["answer_units"]))
        common = sum((Counter(tokens) & Counter(expected)).values())
        precision = common / len(tokens) if tokens else 0.0
        recall = common / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if common else 0.0
        alternatives.append(
            {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "answer_words": len(tokens),
                "reference_words": len(expected),
            }
        )
    return max(alternatives, key=lambda result: result["f1"])


def analyze_oracle(dataset_path: Path, output: Path) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    queries = {q["query_id"]: q for q in dataset["questions"]}
    fitting = {
        r["query_id"]
        for r in read_rows(output / "contexts.jsonl")
        if r["method"] == "oracle_evidence" and r["oracle_fits_budget"]
    }
    analysis = {
        "scope": "posthoc paired diagnostic; oracle is not a deployable selector",
        "primary_population": "Prespecified questions whose whole first annotation fits 2K",
        "intervals": "5,000 paired document-cluster bootstrap draws, seed 42",
        "models": {},
        "human_labels_created": False,
    }
    for model in comparison["planned_models"]:
        rows = {
            (r["query_id"], r["method"]): r
            for r in read_rows(output / f"answers.{model}.jsonl")
        }
        scoped = comparison["model_query_ids"][model]
        wanted = {(q, method) for q in scoped for method in ORACLE_METHODS}
        if not wanted <= rows.keys():
            analysis["models"][model] = {"status": "pending", "saved": len(rows)}
            continue
        ids = [q for q in scoped if q in fitting]
        if not ids:
            analysis["models"][model] = {"status": "no_oracle_fits_2k"}
            continue
        comparisons = {}
        for control in ORACLE_METHODS[:2]:
            comparisons[control] = {
                metric: paired_cluster_interval(
                    [
                        (
                            queries[q]["document_id"],
                            rows[(q, "oracle_evidence")]["metrics"][metric],
                            rows[(q, control)]["metrics"][metric],
                        )
                        for q in ids
                    ]
                )
                for metric in ("answer_token_f1", "all_answer_units_matched")
            }
        expression = {}
        for method in ORACLE_METHODS:
            details = [overlap_details(rows[(q, method)], queries[q]["references"]) for q in ids]
            expression[method] = {
                f"mean_{field}": sum(r[field] for r in details) / len(details)
                for field in ("precision", "recall", "answer_words", "reference_words")
            }
        analysis["models"][model] = {
            "status": "complete",
            "fitting_questions": len(ids),
            "nonfitting_questions": len(scoped) - len(ids),
            "oracle_minus_retrieved": comparisons,
            "overlap_decomposition": expression,
        }
    comparison["oracle_analysis"] = analysis
    save_json(output / "comparison.json", comparison)
    print("Paired oracle analysis saved in comparison.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-oracle", "analyze-oracle"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--old-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.action == "analyze-oracle":
        analyze_oracle(args.dataset, args.output)
        return
    from transformers import AutoTokenizer

    from trustrag.qasper_retrieval import snapshot
    from trustrag.research_models import model_spec

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot(args.root, model_spec("qwen7b")), local_files_only=True
    )
    prepare_oracle(args.dataset, args.old_results, args.output, tokenizer)


if __name__ == "__main__":
    main()
