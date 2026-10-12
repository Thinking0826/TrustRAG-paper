"""Matched vanilla-RAG versus DOS-RAG validation on MultiFieldQA-en."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import string
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

METHODS = ("vanilla_rag", "dos_rag")
BUDGETS = (1024, 2048)
CHUNK_TOKENS = 100
CANDIDATE_K = 40
MAX_NEW_TOKENS = 64
PROMPT_PROFILE = "longbench_multifieldqa_en_official_v1"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: dict, *, new: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if new else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def prepare(archive: Path, output: Path) -> None:
    if output.exists():
        print("Existing MultiFieldQA-en dataset retained", flush=True)
        return
    with zipfile.ZipFile(archive) as zipped:
        rows = [
            json.loads(line)
            for line in zipped.read("data/multifieldqa_en.jsonl").decode("utf-8").splitlines()
            if line
        ]
    if len(rows) != 150:
        raise ValueError("Expected the public LongBench MultiFieldQA-en set of 150 examples")
    questions = [
        {
            "query_id": row["_id"],
            "document_id": row["_id"],
            "question": row["input"],
            "context": row["context"],
            "answers": row["answers"],
            "official_length": row["length"],
        }
        for row in rows
    ]
    write_json(
        output,
        {
            "dataset": "LongBench MultiFieldQA-en",
            "official_split": "test",
            "source": "THUDM/LongBench data/multifieldqa_en.jsonl",
            "citation": "Bai et al., ACL 2024, https://aclanthology.org/2024.acl-long.172/",
            "license_note": "Use follows the public LongBench repository and dataset card.",
            "question_count": len(questions),
            "questions": questions,
            "model_query_ids": {
                "qwen7b": [row["query_id"] for row in questions],
                "qwen14b_nf4": [row["query_id"] for row in questions[:50]],
            },
        },
        new=True,
    )
    print("Prepared all 150 public MultiFieldQA-en examples", flush=True)


def chunk_text(tokenizer, text: str, maximum: int = CHUNK_TOKENS) -> list[dict]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    chunks = []
    for number, start_token in enumerate(range(0, len(offsets), maximum)):
        stop_token = min(start_token + maximum, len(offsets))
        start = 0 if start_token == 0 else offsets[start_token][0]
        end = len(text) if stop_token == len(offsets) else offsets[stop_token][0]
        if text[start:end].strip():
            chunks.append(
                {
                    "chunk_id": f"c{number:04d}",
                    "index": number,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
    if not chunks:
        raise ValueError("A LongBench context produced no nonempty chunks")
    return chunks


def make_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    prompt = (
        "Read the following text and answer briefly.\n\n"
        f"{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\n"
        f"Question: {question}\nAnswer:"
    )
    return [{"role": "user", "content": prompt}]


def selected_pair(tokenizer, question: str, chunks: list[dict], ranking, budget: int) -> dict:
    selected = []
    for raw_index in ranking[:CANDIDATE_K]:
        trial = [*selected, int(raw_index)]
        relevance = [chunks[index] for index in trial]
        document = [chunks[index] for index in sorted(trial)]
        counts = [
            len(
                tokenizer.apply_chat_template(
                    make_messages(question, ordered),
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            for ordered in (relevance, document)
        ]
        if max(counts) > budget:
            break
        selected = trial
    relevance = [chunks[index] for index in selected]
    document = [chunks[index] for index in sorted(selected)]
    return {
        "vanilla_rag": (relevance, make_messages(question, relevance)),
        "dos_rag": (document, make_messages(question, document)),
    }


def idle_gpu(gpu: int) -> None:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    used, free, utilization = (int(value.strip()) for value in line.split(","))
    if used > 64 or free < 22000 or utilization:
        raise RuntimeError(f"GPU {gpu} is occupied; no foreign task will be stopped")
    os.environ.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )


def build_contexts(dataset_path: Path, output: Path, root: Path, gpu: int) -> None:
    if (output / "comparison.json").exists():
        print("Existing MultiFieldQA-en contexts retained", flush=True)
        return
    idle_gpu(gpu)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    from trustrag.benchmark_models import MODELS
    from trustrag.llm_smoke import ProcessGuard
    from trustrag.qasper_retrieval import snapshot
    from trustrag.research_models import model_spec

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot(root, model_spec("qwen7b")), local_files_only=True
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    started = time.monotonic()
    context_path = output / "contexts.jsonl"
    records = read_rows(context_path)
    counts = Counter(row["query_id"] for row in records)
    complete_ids = {query_id for query_id, count in counts.items() if count == 4}
    recorded = {(row["query_id"], row["method"], row["budget"]) for row in records}
    with ProcessGuard():
        encoder = SentenceTransformer(
            str(snapshot(root, MODELS["bge_m3"])),
            device="cuda:0",
            local_files_only=True,
            model_kwargs={"use_safetensors": False, "torch_dtype": torch.float16},
        )
        encoder.max_seq_length = 512
        with context_path.open("a", encoding="utf-8") as stream:
            for number, query in enumerate(dataset["questions"], 1):
                if query["query_id"] in complete_ids:
                    print(
                        f"MultiFieldQA contexts {number}/{len(dataset['questions'])} retained",
                        flush=True,
                    )
                    continue
                chunks = chunk_text(tokenizer, query["context"])
                embeddings = encoder.encode(
                    [query["question"], *[chunk["text"] for chunk in chunks]],
                    batch_size=8,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype(np.float32)
                ranking = np.argsort(-(embeddings[1:] @ embeddings[0]), kind="stable")
                for budget in BUDGETS:
                    pair = selected_pair(tokenizer, query["question"], chunks, ranking, budget)
                    selected_ids = [chunk["chunk_id"] for chunk in pair["vanilla_rag"][0]]
                    for method in METHODS:
                        chosen, messages = pair[method]
                        row = {
                            "query_id": query["query_id"],
                            "document_id": query["document_id"],
                            "question": query["question"],
                            "answers": query["answers"],
                            "method": method,
                            "budget": budget,
                            "prompt_profile": PROMPT_PROFILE,
                            "retrieval": "BGE-M3 dense cosine",
                            "chunk_tokens": CHUNK_TOKENS,
                            "candidate_k": CANDIDATE_K,
                            "selected_chunk_ids": selected_ids,
                            "rendered_chunk_ids": [chunk["chunk_id"] for chunk in chosen],
                            "chunks": chosen,
                            "messages": messages,
                            "input_tokens": len(
                                tokenizer.apply_chat_template(
                                    messages, tokenize=True, add_generation_prompt=True
                                )
                            ),
                        }
                        key = (row["query_id"], row["method"], row["budget"])
                        if key in recorded:
                            continue
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        records.append(row)
                        recorded.add(key)
                    stream.flush()
                print(f"MultiFieldQA contexts {number}/{len(dataset['questions'])}", flush=True)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    lookup = {
        (row["query_id"], row["method"], row["budget"]): row for row in records
    }
    for query in dataset["questions"]:
        for budget in BUDGETS:
            vanilla = lookup[(query["query_id"], "vanilla_rag", budget)]
            dos = lookup[(query["query_id"], "dos_rag", budget)]
            if vanilla["selected_chunk_ids"] != dos["selected_chunk_ids"]:
                raise RuntimeError("The DOS-RAG contrast must keep the selected chunk set fixed")
    write_json(
        output / "comparison.json",
        {
            "dataset": "LongBench MultiFieldQA-en",
            "official_split": "test",
            "completion": "partial",
            "stage": "contexts_complete_generation_pending",
            "question_count": len(dataset["questions"]),
            "context_record_count": len(records),
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "planned_models": ["qwen7b", "qwen14b_nf4"],
            "model_query_ids": dataset["model_query_ids"],
            "retriever": "BAAI/bge-m3 fixed local revision",
            "chunk_tokens": CHUNK_TOKENS,
            "candidate_k": CANDIDATE_K,
            "prompt_profile": PROMPT_PROFILE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "selection_seconds": round(time.monotonic() - started, 2),
            "comparison_scope": "Matched selected chunks; DOS-RAG restores original order",
            "replication_scope": "Core ordering contrast with local BGE-M3 and Qwen; not a numerical reproduction of the published deployment",
            "generation": {"status": "pending"},
        },
        new=True,
    )
    print(f"Independent contexts ready: {len(records)}", flush=True)


def official_f1(prediction: str, answers: list[str]) -> float:
    def tokens(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted = tokens(prediction)
    best = 0.0
    for answer in answers:
        expected = tokens(answer)
        common = sum((Counter(predicted) & Counter(expected)).values())
        if not common:
            score = 0.0
        else:
            precision, recall = common / len(predicted), common / len(expected)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def summarize(output: Path, model: str) -> None:
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    rows = read_rows(output / f"answers.{model}.jsonl")
    planned = len(comparison["model_query_ids"][model]) * len(METHODS) * len(BUDGETS)
    configurations = {}
    for budget in BUDGETS:
        for method in METHODS:
            selected = [r for r in rows if r["budget"] == budget and r["method"] == method]
            if selected:
                configurations[f"{method}@{budget}"] = {
                    "records": len(selected),
                    "mean_official_qa_f1": sum(r["official_qa_f1"] for r in selected)
                    / len(selected),
                    "mean_generation_seconds": sum(r["generation_seconds"] for r in selected)
                    / len(selected),
                    "statuses": dict(Counter(r["status"] for r in selected)),
                }
    comparison["generation"][model] = {
        "completed_records": len(rows),
        "planned_records": planned,
        "configurations": configurations,
    }
    complete = all(
        comparison["generation"].get(key, {}).get("completed_records", 0)
        == len(comparison["model_query_ids"][key]) * len(METHODS) * len(BUDGETS)
        for key in comparison["planned_models"]
    )
    comparison["generation"]["status"] = "complete" if complete else "partial"
    comparison["completion"] = "complete" if complete else "partial"
    comparison["stage"] = "generation_complete" if complete else "generation_in_progress"
    write_json(output / "comparison.json", comparison)


def generate_batch(
    dataset_path: Path,
    output: Path,
    root: Path,
    gpu: int,
    model_key: str,
    budget: int,
    query_limit: int,
) -> None:
    from trustrag.llm_smoke import ProcessGuard
    from trustrag.qasper_retrieval import snapshot
    from trustrag.research_models import model_spec

    idle_gpu(gpu)
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    allowed = set(comparison["model_query_ids"][model_key])
    answer_path = output / f"answers.{model_key}.jsonl"
    existing = read_rows(answer_path)
    completed = {(row["query_id"], row["method"], row["budget"]) for row in existing}
    pending = [
        row
        for row in read_rows(output / "contexts.jsonl")
        if row["budget"] == budget
        and row["query_id"] in allowed
        and (row["query_id"], row["method"], row["budget"]) not in completed
    ]
    chosen_ids = list(dict.fromkeys(row["query_id"] for row in pending))[:query_limit]
    pending = [row for row in pending if row["query_id"] in chosen_ids]
    if not pending:
        print("No missing independent-validation records in this batch", flush=True)
        return
    spec = model_spec(model_key)
    model_dir = snapshot(root, spec)
    with ProcessGuard():
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            GenerationConfig,
        )

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.8, 0)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        options = {}
        if spec.get("quantization") == "bnb_nf4":
            options = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "device_map": {"": 0},
            }
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            **options,
        )
        if not options:
            model.to("cuda:0")
        model.eval()
        with answer_path.open("a", encoding="utf-8") as stream:
            for context in pending:
                row = {
                    "query_id": context["query_id"],
                    "method": context["method"],
                    "budget": budget,
                    "model": model_key,
                    "model_id": spec["model_id"],
                    "revision": spec["revision"],
                    "quantization": spec.get("quantization", "none_bf16"),
                    "physical_gpu": gpu,
                    "prompt_profile": PROMPT_PROFILE,
                    "status": "error",
                    "error": None,
                    "raw_output": "",
                    "input_tokens": 0,
                    "generated_tokens": 0,
                    "generation_seconds": 0.0,
                    "peak_torch_reserved_gib": 0.0,
                    "official_qa_f1": 0.0,
                }
                try:
                    encoded = tokenizer.apply_chat_template(
                        context["messages"],
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                    row["input_tokens"] = int(encoded["input_ids"].shape[1])
                    if row["input_tokens"] > budget:
                        raise ValueError("Rendered prompt exceeds its fixed complete-input budget")
                    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            generation_config=GenerationConfig(
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                            ),
                        )
                    new_ids = generated[0, row["input_tokens"] :].detach().cpu()
                    row["generation_seconds"] = round(time.monotonic() - started, 4)
                    row["generated_tokens"] = int(new_ids.numel())
                    row["raw_output"] = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    row["peak_torch_reserved_gib"] = round(
                        torch.cuda.max_memory_reserved() / 1024**3, 3
                    )
                    row["status"] = (
                        "answered"
                        if new_ids.numel() and int(new_ids[-1]) == tokenizer.eos_token_id
                        else "truncated"
                    )
                    row["official_qa_f1"] = official_f1(
                        row["raw_output"], context["answers"]
                    )
                    del generated, encoded
                except (ValueError, TypeError, KeyError, RuntimeError) as error:
                    row["error"] = str(error)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                torch.cuda.empty_cache()
                print(
                    f"MultiFieldQA {model_key} {budget} {row['query_id']} "
                    f"{row['method']}: {row['status']}",
                    flush=True,
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    summarize(output, model_key)


def analyze(dataset_path: Path, output: Path) -> None:
    from trustrag.research_statistics import paired_cluster_interval

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    documents = {row["query_id"]: row["document_id"] for row in dataset["questions"]}
    analysis = {
        "pair": "dos_rag minus vanilla_rag with identical selected chunks",
        "metric": "official LongBench English QA F1",
        "interval": "5,000 paired document-cluster bootstrap draws, seed 42",
        "models": {},
    }
    for model in comparison["planned_models"]:
        lookup = {
            (row["query_id"], row["method"], row["budget"]): row
            for row in read_rows(output / f"answers.{model}.jsonl")
        }
        ids = comparison["model_query_ids"][model]
        budgets = {}
        for budget in BUDGETS:
            wanted = {(query, method, budget) for query in ids for method in METHODS}
            if not wanted <= lookup.keys():
                budgets[str(budget)] = {"status": "pending"}
                continue
            budgets[str(budget)] = paired_cluster_interval(
                [
                    (
                        documents[query],
                        lookup[(query, "dos_rag", budget)]["official_qa_f1"],
                        lookup[(query, "vanilla_rag", budget)]["official_qa_f1"],
                    )
                    for query in ids
                ]
            )
        analysis["models"][model] = budgets
    comparison["paired_analysis"] = analysis
    write_json(output / "comparison.json", comparison)
    print("Independent paired analysis saved", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "context", "generate", "analyze"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", choices=("qwen7b", "qwen14b_nf4"))
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--query-limit", type=int, default=20)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.archive, args.dataset)
    elif args.action == "context":
        build_contexts(args.dataset, args.output, args.root, args.gpu)
    elif args.action == "generate":
        generate_batch(
            args.dataset,
            args.output,
            args.root,
            args.gpu,
            args.model,
            args.budget,
            args.query_limit,
        )
    else:
        analyze(args.dataset, args.output)


if __name__ == "__main__":
    main()
