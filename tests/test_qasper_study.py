"""Only routing and prespecified subset examples; no real test outputs used."""

import unittest

from trustrag.qasper_pilot import model_record_count
from trustrag.qasper_study import study_fields
from trustrag.research_statistics import METRICS, method_analysis


class QasperStudyTests(unittest.TestCase):
    def test_nested_subsets_are_fixed_without_scores(self):
        ids = [f"q{i}" for i in range(300)]
        fields = study_fields(ids)
        self.assertEqual(fields["model_query_ids"]["qwen14b_nf4"], ids[:100])
        self.assertEqual(fields["study_plan"]["human_primary_query_ids"], ids[:60])
        self.assertEqual(fields["study_plan"]["human_large_model_query_ids"], ids[:30])
        self.assertFalse(fields["study_plan"]["human_labels_created"])

    def test_model_counts_and_development_default(self):
        comparison = {
            "context_record_count": 3600,
            "methods": ["a", "b", "c", "d"],
            "budgets": [1024, 2048, 4096],
            **study_fields([f"q{i}" for i in range(300)]),
        }
        self.assertEqual(model_record_count(comparison, "qwen7b"), 3600)
        self.assertEqual(model_record_count(comparison, "qwen14b_nf4"), 1200)
        comparison.pop("model_query_ids")
        self.assertEqual(model_record_count(comparison, "qwen14b_nf4"), 3600)

    def test_subset_analysis_does_not_require_unplanned_questions(self):
        dataset = {
            "questions": [
                {"query_id": "q1", "document_id": "d1"},
                {"query_id": "q2", "document_id": "d2"},
            ]
        }
        comparison = {
            "methods": ["closure_greedy", "density_atoms"],
            "budgets": [2048],
            "planned_models": ["large"],
            "model_query_ids": {"large": ["q1"]},
        }
        rows = [
            {
                "query_id": "q1",
                "method": method,
                "budget": 2048,
                "metrics": dict.fromkeys(METRICS, 1),
            }
            for method in comparison["methods"]
        ]
        result = method_analysis(dataset, comparison, {"large": rows})
        self.assertEqual(result["models"]["large"]["status"], "complete_matrix_analyzed")
        self.assertEqual(result["models"]["large"]["recorded_conditions"], 2)


if __name__ == "__main__":
    unittest.main()
