"""Fixed BGE hybrid/rerank candidates for the QASPER research pipeline."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path


def idle_gpu(gpu: int) -> None:
    counters = subprocess.check_output(
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
    used, free, utilization = (int(value.strip()) for value in counters.split(","))
    if used > 64 or free < 22000 or utilization:
        raise RuntimeError(f"GPU {gpu} is occupied; no foreign process will be stopped")
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


def snapshot(root: Path, spec: dict) -> Path:
    return next(
        path
        for cache in (root / "cache/huggingface/hub", root / "cache/huggingface")
        if (
            path := cache
            / f"models--{spec['model_id'].replace('/', '--')}"
            / "snapshots"
            / spec["revision"]
        ).is_dir()
    )


def retrieve(dataset_path: Path, output: Path, root: Path, gpu: int, candidate_k: int) -> None:
    """Same label-free candidate pool and relevance scores for every selection method."""
    idle_gpu(gpu)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from trustrag.benchmark_models import MODELS
    from trustrag.bm25 import BM25Index
    from trustrag.hybrid_search import hybrid_rank
    from trustrag.llm_smoke import ProcessGuard
    from trustrag.qasper_pilot import expanded_candidates, structure_atoms
    from trustrag.reranker import LocalReranker, candidate_union

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Candidate records already exist: {output}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    started = time.monotonic()
    queries = dataset["questions"]
    with ProcessGuard():
        encoder = SentenceTransformer(
            str(snapshot(root, MODELS["bge_m3"])),
            device="cuda:0",
            local_files_only=True,
            model_kwargs={"use_safetensors": False, "torch_dtype": torch.float16},
        )
        encoder.max_seq_length = 512
        vectors = {}
        for number, (paper_id, document) in enumerate(dataset["documents"].items(), 1):
            vectors[paper_id] = encoder.encode(
                [source["text"] for source in document["sources"]],
                batch_size=4,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            print(f"Encode papers {number}/{len(dataset['documents'])}", flush=True)
        query_vectors = encoder.encode(
            [query["question"] for query in queries],
            batch_size=4,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    pools = []
    for query, vector in zip(queries, query_vectors, strict=True):
        sources = dataset["documents"][query["document_id"]]["sources"]
        sparse = BM25Index.from_texts([source["text"] for source in sources]).scores(
            query["question"]
        )
        dense = vectors[query["document_id"]] @ vector
        _, fusion, dense_ranks, sparse_ranks = hybrid_rank(dense, sparse, 30, 30, 1, 1, 60)
        pools.append((sources, candidate_union(dense_ranks, sparse_ranks, fusion)))
    del vectors, query_vectors

    with ProcessGuard():
        reranker = LocalReranker.from_pretrained(
            snapshot(root, MODELS["bge_reranker"]),
            "cuda:0",
            max_length=512,
        )
        with output.open("x", encoding="utf-8") as stream:
            for number, (query, (original, pool)) in enumerate(zip(queries, pools, strict=True), 1):
                logits = reranker.score_pairs(
                    query["question"],
                    [original[int(i)]["text"] for i in pool],
                    batch_size=4,
                )
                ranking = [int(pool[i]) for i in np.argsort(-logits, kind="stable")[:candidate_k]]
                sources = expanded_candidates(original, ranking)
                paragraph_logits = reranker.score_pairs(
                    query["question"],
                    [source["text"] for source in sources],
                    batch_size=4,
                )
                atoms, groups = structure_atoms(sources, query["question"])
                lookup = {source["source_id"]: source["text"] for source in sources}
                atom_logits = reranker.score_pairs(
                    query["question"],
                    [lookup[a.source_id][a.start : a.end] for a in atoms],
                    batch_size=4,
                )
                atom_scores = 1 / (1 + np.exp(-np.clip(atom_logits, -30, 30)))
                paragraph_scores = 1 / (1 + np.exp(-np.clip(paragraph_logits, -30, 30)))
                record = {
                    "query_id": query["query_id"],
                    "document_id": query["document_id"],
                    "retrieval": "bge_hybrid_rerank",
                    "candidate_k": candidate_k,
                    "candidate_union_count": len(pool),
                    "sources": sources,
                    "atoms": [
                        asdict(replace(a, utility=float(s)))
                        for a, s in zip(atoms, atom_scores, strict=True)
                    ],
                    "paragraph_utilities": dict(
                        zip(lookup, map(float, paragraph_scores), strict=True)
                    ),
                    "structural_groups": groups,
                    "configuration": {
                        "dense_sparse_candidates_each": 30,
                        "rrf_k": 60,
                        "weights": [1, 1],
                        "encoder_max_tokens": 512,
                        "reranker_max_tokens": 512,
                        "batch_size": 4,
                        "encoder": MODELS["bge_m3"]["revision"],
                        "reranker": MODELS["bge_reranker"]["revision"],
                    },
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"Rerank questions {number}/{len(queries)}", flush=True)
        del reranker
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Candidates ready: {output}; {time.monotonic() - started:.1f} s", flush=True)
