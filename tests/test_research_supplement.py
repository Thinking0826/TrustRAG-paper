"""Small synthetic checks; no actual test responses inform the supplement design."""

import json

from trustrag.research_supplement import oracle_pieces, overlap_details, prepare_oracle


def source(key, text):
    return {
        "source_id": key,
        "document_id": "paper",
        "filename": "paper",
        "text": text,
        "spans": [{"block_id": key, "paragraph_id": key, "start": 0, "end": len(text)}],
    }


def test_oracle_uses_whole_first_annotation_in_document_order():
    document = {
        "sources": [source("p1", "Extra qualification. Alpha is first."), source("p2", "Beta.")]
    }
    query = {
        "references": [
            {"answer_units": ["Alpha"], "evidence_paragraph_ids": ["p1"]},
            {"answer_units": ["Beta"], "evidence_paragraph_ids": ["p2"]},
        ]
    }
    pieces = oracle_pieces(query, document)
    assert len(pieces) == 1
    assert pieces[0]["text"] == "Extra qualification. Alpha is first."
    assert pieces[0]["original_start"] == 0
    assert pieces[0]["citation_id"] == "E1"
    assert "answer_units" not in pieces[0]


def test_preparation_keeps_overlong_evidence_and_reuses_original_answers(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **_):
            return range(2500)

    text = "An original statement containing Alpha."
    reference = {
        "answer_units": ["Alpha"],
        "evidence_paragraph_ids": ["p1"],
        "evidence_texts": [text],
    }
    dataset = {
        "questions": [
            {
                "query_id": "q1",
                "document_id": "paper",
                "question": "Which?",
                "references": [reference],
            }
        ],
        "documents": {"paper": {"sources": [source("p1", text)]}},
        "model_query_ids": {"qwen7b": ["q1"], "qwen14b_nf4": ["q1"]},
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    old = tmp_path / "old"
    old.mkdir()
    controls = [
        {
            "query_id": "q1",
            "method": method,
            "budget": 2048,
            "raw_output": "saved failure",
            "status": "error",
        }
        for method in ("density_atoms", "closure_greedy")
    ]
    serialized = "\n".join(json.dumps(row) for row in controls) + "\n"
    for name in ("contexts.jsonl", "answers.qwen7b.jsonl", "answers.qwen14b_nf4.jsonl"):
        (old / name).write_text(serialized)
    result = tmp_path / "new"
    prepare_oracle(dataset_path, old, result, Tokenizer())
    contexts = [json.loads(line) for line in (result / "contexts.jsonl").read_text().splitlines()]
    oracle = contexts[-1]
    assert not oracle["oracle_fits_budget"]
    assert oracle["sources"][0]["text"] == text
    assert oracle["input_tokens"] == 2500
    assert (old / "answers.qwen7b.jsonl").read_text() == serialized
    prepare_oracle(dataset_path, old, result, Tokenizer())
    assert len((result / "answers.qwen7b.jsonl").read_text().splitlines()) == 2


def test_overlap_decomposition_does_not_create_semantic_labels():
    row = {"status": "answered", "candidate": {"answer": "Alpha and extra words [E1]."}}
    details = overlap_details(row, [{"answer_units": ["Alpha"]}])
    assert details["precision"] == 0.25
    assert details["recall"] == 1.0
    assert details["f1"] == 0.4
    assert "semantic_correct" not in details
