"""Small synthetic checks of blinding, scope, failures and original source resolution."""

import json
import tempfile
import unittest
from pathlib import Path

from trustrag.qasper_review import build_packet, review_keys, write_packet


class QasperReviewTests(unittest.TestCase):
    def fixture(self):
        plan = {
            "study_id": "synthetic",
            "human_primary_query_ids": ["q1", "q2"],
            "human_large_model_query_ids": ["q1"],
            "human_review_methods": ["closure_greedy", "density_atoms"],
            "human_review_budget": 2048,
        }
        dataset = {
            "study_plan": plan,
            "questions": [
                {"query_id": q, "document_id": "doc", "question": q, "references": []}
                for q in ("q1", "q2")
            ],
            "documents": {
                "doc": {
                    "title": "Paper",
                    "abstract": "Abstract",
                    "sources": [
                        {"source_id": "p1", "section_name": "Methods", "text": "Full paragraph"},
                        {"source_id": "p2", "section_name": "Results", "text": "Other paragraph"},
                    ],
                }
            },
        }
        contexts, answers = [], {"qwen7b": [], "qwen14b_nf4": []}
        for model, query, method, budget in review_keys(plan):
            contexts.append(
                {
                    "query_id": query,
                    "method": method,
                    "budget": budget,
                    "sources": [
                        {"citation_id": "E1", "text": "Full", "spans": [{"paragraph_id": "p1"}]}
                    ],
                }
            )
            answers[model].append(
                {
                    "query_id": query,
                    "method": method,
                    "budget": budget,
                    "status": "rejected" if query == "q2" else "answered",
                    "raw_output": "Answer [E1]",
                    "metrics": {"answer_token_f1": 1},
                    "model_id": "hidden",
                }
            )
        return dataset, contexts, answers

    def test_fixed_scope_blinded_and_full_original_available(self):
        dataset, contexts, answers = self.fixture()
        packet, private = build_packet(dataset, contexts, answers)
        self.assertEqual(packet["review_count"], 6)
        self.assertFalse(packet["human_labels_created"])
        content = json.dumps(packet)
        for hidden in ("qwen7b", "qwen14b_nf4", "closure_greedy", "density_atoms", "metrics"):
            self.assertNotIn(hidden, content)
        self.assertEqual(len(private["cases"]), 6)
        self.assertEqual(packet["documents"]["doc"]["paragraphs"][0]["text"], "Full paragraph")
        self.assertEqual(len(packet["documents"]["doc"]["paragraphs"]), 2)
        self.assertEqual(build_packet(dataset, contexts, answers), (packet, private))

    def test_rejected_records_are_retained_without_displayed_success(self):
        packet, _ = build_packet(*self.fixture())
        rejected = [case for case in packet["cases"] if case["outcome"] == "rejected"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(case["displayed_answer"] is None for case in rejected))
        self.assertTrue(all(case["raw_candidate"] for case in rejected))

    def test_missing_prespecified_answers_prevent_packet(self):
        dataset, contexts, answers = self.fixture()
        answers["qwen14b_nf4"].pop()
        with self.assertRaisesRegex(ValueError, "1 prespecified outputs missing"):
            build_packet(dataset, contexts, answers)

    def test_unknown_original_paragraph_does_not_become_fake_page(self):
        dataset, contexts, answers = self.fixture()
        for context in contexts:
            context["sources"][0]["spans"][0]["paragraph_id"] = "missing"
        with self.assertRaises(KeyError):
            build_packet(dataset, contexts, answers)

    def test_offline_html_escapes_script_terminator_and_retains_key_separately(self):
        packet, private = build_packet(*self.fixture())
        packet["cases"][0]["question"] = "</script><script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "packet"
            write_packet(output, packet, private)
            html = (output / "review.html").read_text()
            self.assertNotIn(packet["cases"][0]["question"], html)
            self.assertIn("qwen7b", (output / "private_key.json").read_text())
            self.assertNotIn("qwen7b", html)
            with self.assertRaises(FileExistsError):
                write_packet(output, packet, private)


if __name__ == "__main__":
    unittest.main()
