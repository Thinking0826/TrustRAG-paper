"""Restore published study records without model loading or network access."""

from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Preserve any local rerun rather than overwrite it with the published run.
        print(f"Retained existing {path.name}")
        return
    path.write_bytes(content)


def restore(root: Path, archive: Path | None = None) -> None:
    index = json.loads((root / "materials/index.json").read_text(encoding="utf-8"))
    for entry in index["files"]:
        source = root / "materials" / entry["file"]
        data = gzip.decompress(source.read_bytes()) if entry["compressed"] else source.read_bytes()
        write_new(root / entry["target"], data)
    human = root / "data/research/v2/qasper/test_v1/human_review_v2"
    packet = json.loads((human / "review_packet.json").read_text(encoding="utf-8"))
    template = (root / "src/trustrag/qasper_review.html").read_text(encoding="utf-8")
    encoded = json.dumps(packet, ensure_ascii=False).replace("<", "\\u003c")
    write_new(human / "review.html", template.replace("__PACKET__", encoded).encode("utf-8"))

    if archive is not None:
        lb = root / "data/research/v2/longbench/multifieldqa_en"
        original = json.loads((lb / "dataset.json").read_text(encoding="utf-8"))
        with zipfile.ZipFile(archive) as zipped:
            rows = [json.loads(line) for line in zipped.read("data/multifieldqa_en.jsonl").decode("utf-8").splitlines() if line]
        if [row["_id"] for row in rows] != [row["query_id"] for row in original["questions"]]:
            raise ValueError("The supplied archive does not have the published official question order")
        full = dict(original)
        full["questions"] = [{"query_id": row["_id"], "document_id": row["_id"],
                              "question": row["input"], "context": row["context"],
                              "answers": row["answers"], "official_length": row["length"]} for row in rows]
        # Enrichment is permitted only while this is still the text-free manifest.
        if not any("context" in row for row in original["questions"]):
            (lb / "dataset.json").write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        queries = {row["query_id"]: row for row in full["questions"]}
        records = []
        for line in (lb / "results/contexts.coordinates.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            query = queries[row["query_id"]]
            row["question"] = query["question"]
            row["chunks"] = [{**chunk, "text": query["context"][chunk["start"]:chunk["end"]]} for chunk in row["chunks"]]
            context = "\n\n".join(chunk["text"] for chunk in row["chunks"])
            prompt = ("Read the following text and answer briefly.\n\n" + context + "\n\n"
                      "Now, answer the following question based on the above text, only give me the answer "
                      "and do not output any other words.\n\n"
                      f"Question: {query['question']}\nAnswer:")
            row["messages"] = [{"role": "user", "content": prompt}]
            records.append(json.dumps(row, ensure_ascii=False))
        write_new(lb / "results/contexts.jsonl", ("\n".join(records) + "\n").encode("utf-8"))
    print("Study records restored; no answers generated or human judgments changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--longbench-archive", type=Path)
    arguments = parser.parse_args()
    restore(arguments.root.resolve(), arguments.longbench_archive)

