"""Resumable follow-up: query-conditioned plans, common controls and one summary."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from trustrag.qasper_pilot import (
    PROMPT_PROFILE,
    make_messages,
    reference_coverage,
    summarize_answers,
)
from trustrag.query_closure import (
    BUDGETS,
    METHODS,
    PLANNER_PROFILE,
    atoms_from_record,
    parse_plan,
    planning_messages,
    select,
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def complete_json_object(raw):
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw).lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


def plan_batch(dataset_path, retrieval_path, output, root, gpu, query_limit):
    from trustrag.llm_smoke import ProcessGuard
    from trustrag.qasper_retrieval import idle_gpu, snapshot
    from trustrag.research_models import model_spec

    output.mkdir(parents=True, exist_ok=True)
    path = output / "plans.jsonl"
    done = {r["query_id"] for r in read_rows(path)} if path.exists() else set()
    # Only question text and original candidate units enter the planner.
    questions = json.loads(dataset_path.read_text())["questions"]
    queries = [q for q in questions if q["query_id"] not in done][:query_limit]
    if not queries:
        return
    records = {r["query_id"]: r for r in read_rows(retrieval_path)}
    idle_gpu(gpu)
    with ProcessGuard():
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        class StopAtPlanningObject(StoppingCriteria):
            def __init__(self, tokenizer, prompt_length):
                self.tokenizer = tokenizer
                self.prompt_length = prompt_length

            def __call__(self, input_ids, scores, **kwargs):
                return complete_json_object(
                    self.tokenizer.decode(
                        input_ids[0, self.prompt_length :], skip_special_tokens=True
                    )
                )

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        model_dir = snapshot(root, model_spec("qwen7b"))
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        load_started = time.monotonic()
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda:0")
            .eval()
        )
        load_seconds = time.monotonic() - load_started
        with path.open("a", encoding="utf-8") as stream:
            for query_number, query in enumerate(queries):
                cached = records[query["query_id"]]
                row = {
                    "query_id": query["query_id"],
                    "profile": PLANNER_PROFILE,
                    "model": "qwen7b",
                    "status": "error",
                    "plan": None,
                    "raw_output": "",
                    "input_tokens": 0,
                    "generated_tokens": 0,
                    "planning_seconds": 0.0,
                    "cold_load_seconds": load_seconds if query_number == 0 else 0.0,
                    "peak_torch_reserved_gib": 0.0,
                    "error": None,
                    "physical_gpu": gpu,
                }
                started = time.monotonic()
                try:
                    encoded = tokenizer.apply_chat_template(
                        planning_messages(query["question"], cached),
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                    row["input_tokens"] = int(encoded["input_ids"].shape[1])
                    if row["input_tokens"] > 16384:
                        raise ValueError("Planning input exceeds the fixed 16384-token cap")
                    encoded = {k: v.to("cuda:0") for k, v in encoded.items()}
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            stopping_criteria=StoppingCriteriaList(
                                [StopAtPlanningObject(tokenizer, row["input_tokens"])]
                            ),
                            generation_config=GenerationConfig(
                                max_new_tokens=1536,
                                do_sample=False,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                            ),
                        )
                    new_ids = generated[0, row["input_tokens"] :].detach().cpu()
                    row["generated_tokens"] = int(new_ids.numel())
                    row["raw_output"] = tokenizer.decode(new_ids, skip_special_tokens=True)
                    row["peak_torch_reserved_gib"] = round(
                        torch.cuda.max_memory_reserved() / 1024**3, 3
                    )
                    if not complete_json_object(row["raw_output"]):
                        raise ValueError("Planner output did not finish a JSON object")
                    row["plan"] = parse_plan(row["raw_output"], atoms_from_record(cached))
                    row["status"] = "planned"
                    del generated, encoded
                except (ValueError, TypeError, KeyError, RuntimeError) as error:
                    row["error"] = str(error)
                row["planning_seconds"] = round(time.monotonic() - started, 4)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                torch.cuda.empty_cache()
                print(
                    f"Plan {query['query_id']}: {row['status']}; "
                    f"{row['input_tokens']} input / {row['generated_tokens']} output",
                    flush=True,
                )


def similarities(dataset_path, retrieval_path, output, root, gpu):
    from trustrag.llm_smoke import ProcessGuard
    from trustrag.qasper_retrieval import idle_gpu, snapshot

    path = output / "similarities.jsonl"
    done = {r["query_id"] for r in read_rows(path)} if path.exists() else set()
    wanted = {q["query_id"] for q in json.loads(dataset_path.read_text())["questions"]}
    records = [r for r in read_rows(retrieval_path) if r["query_id"] in wanted - done]
    if not records:
        return
    idle_gpu(gpu)
    with ProcessGuard():
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        from trustrag.benchmark_models import MODELS

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        load_started = time.monotonic()
        encoder = SentenceTransformer(
            str(snapshot(root, MODELS["bge_m3"])),
            device="cuda:0",
            local_files_only=True,
            model_kwargs={"use_safetensors": False, "torch_dtype": torch.float16},
        )
        encoder.max_seq_length = 512
        load_seconds = time.monotonic() - load_started
        with path.open("a", encoding="utf-8") as stream:
            for number, record in enumerate(records, 1):
                texts = {s["source_id"]: s["text"] for s in record["sources"]}
                started = time.monotonic()
                vectors = encoder.encode(
                    [texts[a["source_id"]][a["start"] : a["end"]] for a in record["atoms"]],
                    batch_size=4,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype(np.float32)
                row = {
                    "query_id": record["query_id"],
                    "atom_ids": [a["atom_id"] for a in record["atoms"]],
                    "similarity": np.clip(vectors @ vectors.T, 0, 1).tolist(),
                    "encoding_seconds": round(time.monotonic() - started, 4),
                    "cold_load_seconds": load_seconds if number == 1 else 0.0,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"Similarity {number}/{len(records)}", flush=True)


def build_contexts(dataset_path, retrieval_path, output, root, reuse):
    from transformers import AutoTokenizer

    from trustrag.qasper_retrieval import snapshot
    from trustrag.research_models import model_spec

    dataset = json.loads(dataset_path.read_text())
    retrieval = {r["query_id"]: r for r in read_rows(retrieval_path)}
    plans = {r["query_id"]: r for r in read_rows(output / "plans.jsonl")}
    sims = {r["query_id"]: r for r in read_rows(output / "similarities.jsonl")}
    old = (
        {(r["query_id"], r["method"], r["budget"]): r for r in read_rows(reuse / "contexts.jsonl")}
        if reuse
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot(root, model_spec("qwen7b")), local_files_only=True
    )
    path = output / "contexts.jsonl"
    rows = read_rows(path) if path.exists() else []
    done = {(r["query_id"], r["method"], r["budget"]) for r in rows}
    with path.open("a", encoding="utf-8") as stream:
        for number, query in enumerate(dataset["questions"], 1):
            qid = query["query_id"]
            record, plan, sim = retrieval[qid], plans[qid], sims[qid]
            if sim["atom_ids"] != [a["atom_id"] for a in record["atoms"]]:
                raise ValueError("Similarity uses different evidence units")
            cache = {}

            def render(pieces, question=query["question"]):
                return make_messages(question, pieces)

            def count(messages, cache=cache):
                key = json.dumps(messages, ensure_ascii=False)
                if key not in cache:
                    cache[key] = len(
                        tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True
                        )
                    )
                return cache[key]

            before = reference_coverage(record["sources"], query["references"])
            for budget in BUDGETS:
                for method in METHODS:
                    key = (qid, method, budget)
                    if key in done:
                        continue
                    started = time.monotonic()
                    if key in old:
                        row = {**old[key], "reused_from": str(reuse / "contexts.jsonl")}
                    else:
                        packed = select(
                            record,
                            query["question"],
                            plan,
                            sim["similarity"],
                            method=method,
                            budget=budget,
                            render=render,
                            count=count,
                        )
                        row = {
                            "query_id": qid,
                            "document_id": query["document_id"],
                            "question": query["question"],
                            "method": method,
                            "budget": budget,
                            "selection_prompt_profile": PROMPT_PROFILE,
                            "retrieval": record["retrieval"],
                            "candidate_k": record["candidate_k"],
                            "input_tokens": packed["input_tokens"],
                            "sources": [
                                {**s, "citation_id": f"E{i}"}
                                for i, s in enumerate(packed["sources"], 1)
                            ],
                            "messages": packed["rendered_prompt"],
                            "selected_atom_ids": packed["selected_atom_ids"],
                            "before_budget": before,
                            "after_budget": reference_coverage(
                                packed["sources"], query["references"]
                            ),
                            "planning_status": plan["status"],
                        }
                    row["selection_seconds"] = round(time.monotonic() - started, 4)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    done.add(key)
                    rows.append(row)
            print(f"Contexts {number}/{len(dataset['questions'])}", flush=True)
    comparison = {
        "completion": "partial",
        "stage": "followup_contexts_complete",
        "dataset": "QASPER 0.3",
        "official_split": dataset["official_split"],
        "experiment": "question_conditioned_closure_followup_v1",
        "question_count": len(dataset["questions"]),
        "context_record_count": len(rows),
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "planned_models": dataset.get("planned_models", ["qwen7b"]),
        "selection_prompt_profile": PROMPT_PROFILE,
        "generation": {"status": "pending"},
        "planner": {
            "profile": PLANNER_PROFILE,
            "model": "qwen7b_bf16",
            "planned": sum(p["status"] == "planned" for p in plans.values()),
            "failed": sum(p["status"] != "planned" for p in plans.values()),
            "total_seconds": sum(p["planning_seconds"] for p in plans.values()),
            "cold_load_seconds": sum(p.get("cold_load_seconds", 0) for p in plans.values()),
            "total_input_tokens": sum(p["input_tokens"] for p in plans.values()),
            "total_output_tokens": sum(p["generated_tokens"] for p in plans.values()),
            "failure_policy": "same BGE density fallback, failures retained",
        },
        "control_encoding": {
            "warm_seconds": sum(s["encoding_seconds"] for s in sims.values()),
            "cold_load_seconds": sum(s.get("cold_load_seconds", 0) for s in sims.values()),
        },
        "primary_contrast": "query_closure minus planner_flat at 1024 tokens",
        "budget_selection": {},
    }
    if "model_query_ids" in dataset:
        comparison["model_query_ids"] = dataset["model_query_ids"]
    for budget in BUDGETS:
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method and r["budget"] == budget]
            comparison["budget_selection"][f"{method}@{budget}"] = {
                "question_count": len(selected),
                "mean_input_tokens": sum(r["input_tokens"] for r in selected) / len(selected),
                "all_answer_units_retained_rate": sum(
                    r["after_budget"]["all_answer_units_present"] for r in selected
                )
                / len(selected),
            }
    paired = {(r["query_id"], r["budget"]): r for r in rows if r["method"] == "planner_flat"}
    comparison["changed_contexts_vs_flat"] = {
        str(b): sum(
            r["messages"] != paired[(r["query_id"], b)]["messages"]
            for r in rows
            if r["method"] == "query_closure" and r["budget"] == b
        )
        for b in BUDGETS
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
    )
    if reuse:
        reused_answers(dataset_path, output, reuse)


def reused_answers(dataset_path, output, reuse):
    contexts = {
        (r["query_id"], r["method"], r["budget"]): r for r in read_rows(output / "contexts.jsonl")
    }
    previous = {
        (r["query_id"], r["method"], r["budget"]): r for r in read_rows(reuse / "contexts.jsonl")
    }
    for model in json.loads((output / "comparison.json").read_text())["planned_models"]:
        path = output / f"answers.{model}.jsonl"
        existing = (
            {(r["query_id"], r["method"], r["budget"]) for r in read_rows(path)}
            if path.exists()
            else set()
        )
        with path.open("a", encoding="utf-8") as stream:
            for row in read_rows(reuse / f"answers.{model}.jsonl"):
                key = (row["query_id"], row["method"], row["budget"])
                if key in contexts and key not in existing:
                    if contexts[key]["messages"] != previous[key]["messages"]:
                        raise ValueError("Cannot reuse an answer for changed input")
                    stream.write(
                        json.dumps({**row, "reused_from": str(reuse)}, ensure_ascii=False) + "\n"
                    )
                    existing.add(key)
        summarize_answers(output, model)


def analyze(dataset_path, output):
    from trustrag.research_statistics import METRICS, paired_cluster_interval

    comparison = json.loads((output / "comparison.json").read_text())
    queries = json.loads(dataset_path.read_text())["questions"]
    all_results = {}
    for model in comparison["planned_models"]:
        rows = {
            (r["query_id"], r["method"], r["budget"]): r
            for r in read_rows(output / f"answers.{model}.jsonl")
        }
        ids = set(
            comparison.get("model_query_ids", {}).get(model, [q["query_id"] for q in queries])
        )
        wanted = {(qid, method, budget) for qid in ids for method in METHODS for budget in BUDGETS}
        if not wanted <= rows.keys():
            continue
        contrasts = {}
        for control in METHODS[:-1]:
            contrasts[control] = {
                str(b): {
                    metric: paired_cluster_interval(
                        [
                            (
                                q["document_id"],
                                rows[(q["query_id"], "query_closure", b)]["metrics"][metric],
                                rows[(q["query_id"], control, b)]["metrics"][metric],
                            )
                            for q in queries
                            if q["query_id"] in ids
                        ]
                    )
                    for metric in METRICS
                }
                for b in BUDGETS
            }
        all_results[model] = contrasts
    comparison["followup_analysis"] = {
        "scope": "development" if comparison["official_split"] == "train" else "heldout",
        "models": all_results,
        "primary": "query_closure minus planner_flat at 1024",
        "interval": "5000 paper-cluster bootstrap draws, seed 42",
        "secondary_contrasts": "exploratory, not multiplicity adjusted",
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
    )


def mechanism_diagnostics(plans, contexts, answers):
    """Read saved intervention and cost records without rerunning selection or inference."""
    valid = {p["query_id"]: p for p in plans if p["status"] == "planned"}
    edge_ids = {
        qid for qid, p in valid.items() if any(e["requires"] for e in p["plan"]["evidence"])
    }
    lookup = {(c["query_id"], c["method"], c["budget"]): c for c in contexts}
    responses = {(a["query_id"], a["method"], a["budget"]): a for a in answers}
    budgets = {}
    for budget in sorted({c["budget"] for c in contexts}):
        pairs = [
            (qid, c, lookup[(qid, "query_closure", budget)])
            for (qid, method, b), c in lookup.items()
            if method == "planner_flat" and b == budget
        ]
        valid_tokens = [flat["input_tokens"] for qid, flat, _ in pairs if qid in valid]
        unsatisfied = 0
        for qid, flat, _ in pairs:
            if qid not in edge_ids:
                continue
            chosen = set(flat["selected_atom_ids"])
            if any(
                e["id"] in chosen and any(d not in chosen for d in e["requires"])
                for e in valid[qid]["plan"]["evidence"]
            ):
                unsatisfied += 1
        paired_answers = [
            (responses[(qid, "planner_flat", budget)], responses[(qid, "query_closure", budget)])
            for qid, _, _ in pairs
            if (qid, "planner_flat", budget) in responses
            and (qid, "query_closure", budget) in responses
        ]
        budgets[str(budget)] = {
            "paired_contexts": len(pairs),
            "changed_selected_units": sum(
                flat["selected_atom_ids"] != closed["selected_atom_ids"]
                for _, flat, closed in pairs
            ),
            "changed_messages": sum(
                flat["messages"] != closed["messages"] for _, flat, closed in pairs
            ),
            "flat_contexts_missing_declared_dependencies": unsatisfied,
            "valid_plan_mean_input_tokens": sum(valid_tokens) / len(valid_tokens)
            if valid_tokens
            else None,
            "valid_plan_max_input_tokens": max(valid_tokens, default=None),
            "paired_answers": len(paired_answers),
            "changed_raw_answers": sum(
                a["raw_output"] != b["raw_output"] for a, b in paired_answers
            ),
        }
    new_answers = [a for a in answers if "reused_from" not in a]
    new_contexts = [c for c in contexts if "reused_from" not in c]
    return {
        "scope": "post-development mechanism diagnosis; no new human labels or inference",
        "plan_records": len(plans),
        "valid_plans": len(valid),
        "plans_with_dependencies": len(edge_ids),
        "planning_failure_types": dict(
            Counter(p["error"] for p in plans if p["status"] != "planned")
        ),
        "budgets": budgets,
        "observed_cost": {
            "planning_warm_seconds": sum(p["planning_seconds"] for p in plans),
            "planning_cold_load_seconds": sum(p.get("cold_load_seconds", 0) for p in plans),
            "new_context_selection_seconds": sum(c["selection_seconds"] for c in new_contexts),
            "new_answer_records": len(new_answers),
            "new_answer_warm_generation_seconds": sum(a["generation_seconds"] for a in new_answers),
            "timing_note": "Selection shares a tokenizer cache in fixed method order; per-method times are not independent benchmarks. Generation excludes cold loading.",
        },
        "interpretation": "A zero contrast with identical inputs is an inactive intervention, not a general proof that dependencies cannot help.",
    }


def diagnose(output):
    comparison = json.loads((output / "comparison.json").read_text())
    comparison["mechanism_diagnostics"] = mechanism_diagnostics(
        read_rows(output / "plans.jsonl"),
        read_rows(output / "contexts.jsonl"),
        read_rows(output / "answers.qwen7b.jsonl"),
    )
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "similarity", "context", "analyze", "diagnose"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--query-limit", type=int, default=2)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        plan_batch(args.dataset, args.retrieval, args.output, args.root, args.gpu, args.query_limit)
    elif args.stage == "similarity":
        similarities(args.dataset, args.retrieval, args.output, args.root, args.gpu)
    elif args.stage == "context":
        build_contexts(args.dataset, args.retrieval, args.output, args.root, args.reuse)
    elif args.stage == "analyze":
        analyze(args.dataset, args.output)
    else:
        diagnose(args.output)


if __name__ == "__main__":
    main()
