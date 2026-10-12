"""Small numerical examples for paired document-cluster statistics."""

import unittest

from trustrag.research_statistics import (
    METRICS,
    deployment_analysis,
    input_activation,
    method_analysis,
    paired_cluster_interval,
)


class ResearchStatisticsTests(unittest.TestCase):
    def test_saved_input_audit_distinguishes_changed_and_identical_messages(self):
        contexts = []
        for query in ("q1", "q2"):
            for method in ("closure_greedy", "density_atoms"):
                changed = query == "q2" and method == "closure_greedy"
                contexts.append(
                    {
                        "query_id": query,
                        "method": method,
                        "budget": 1024,
                        "messages": [{"content": "changed" if changed else "same"}],
                        "sources": [{"text": "changed" if changed else "same"}],
                    }
                )
        result = input_activation(contexts, {"budgets": [1024]})
        self.assertIn("post hoc", result["scope"])
        self.assertEqual(
            result["budgets"]["1024"],
            {
                "question_count": 2,
                "identical_sources": 1,
                "identical_messages": 1,
                "changed_messages": 1,
                "changed_message_rate": 0.5,
            },
        )

    def test_saved_input_audit_does_not_change_the_existing_summary(self):
        comparison = {"budgets": [2048], "completion": "complete", "paired_analysis": {"x": 1}}
        self.assertIsNone(input_activation([], comparison)["budgets"]["2048"]["changed_message_rate"])
        self.assertEqual(
            comparison, {"budgets": [2048], "completion": "complete", "paired_analysis": {"x": 1}}
        )

    def test_identical_methods_have_zero_difference(self):
        result = paired_cluster_interval([("a", 0.2, 0.2), ("b", 1, 1)], samples=100)
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["ci95"], [0, 0])

    def test_questions_are_weighted_and_paper_cluster_is_preserved(self):
        pairs = [("a", 1, 0), ("a", 1, 0), ("b", 0, 1)]
        result = paired_cluster_interval(pairs, samples=100)
        self.assertEqual(result["question_count"], 3)
        self.assertEqual(result["document_count"], 2)
        self.assertAlmostEqual(result["difference"], 1 / 3)
        self.assertEqual(result["ci95"], [-1, 1])
        self.assertEqual(result, paired_cluster_interval(pairs, samples=100))

    def test_one_paper_does_not_claim_a_confidence_interval(self):
        result = paired_cluster_interval([("a", 0.2, 0.1), ("a", 0.6, 0.3)])
        self.assertIsNone(result["ci95"])

    def fixture(self):
        dataset = {
            "questions": [
                {"query_id": "q1", "document_id": "d1"},
                {"query_id": "q2", "document_id": "d2"},
            ]
        }
        comparison = {
            "methods": ["closure_greedy", "density_atoms"],
            "budgets": [2048],
            "planned_models": ["qwen7b", "qwen14b_nf4"],
        }
        rows = []
        for query in ("q1", "q2"):
            for method in comparison["methods"]:
                score = 0 if query == "q1" else 1
                rows.append(
                    {
                        "query_id": query,
                        "method": method,
                        "budget": 2048,
                        "status": "error" if not score else "answered",
                        "metrics": dict.fromkeys(METRICS, score),
                    }
                )
        return dataset, comparison, rows

    def test_partial_matrix_has_no_interim_effect_estimate(self):
        dataset, comparison, rows = self.fixture()
        result = method_analysis(dataset, comparison, {"qwen7b": rows[:1]})
        self.assertEqual(result["models"]["qwen7b"]["status"], "pending_complete_matrix")
        self.assertNotIn("budgets", result["models"]["qwen7b"])
        self.assertEqual(result["models"]["qwen14b_nf4"]["recorded_conditions"], 0)

    def test_recorded_failures_remain_in_denominator_and_human_labels_are_not_filled(self):
        dataset, comparison, rows = self.fixture()
        result = method_analysis(dataset, comparison, {"qwen7b": rows})
        score = result["models"]["qwen7b"]["budgets"]["2048"]["answer_token_f1"]
        self.assertEqual(score["left_mean"], 0.5)
        self.assertEqual(score["question_count"], 2)
        self.assertEqual(score["difference"], 0)
        self.assertFalse(result["human_evaluation"]["human_labels_created"])
        self.assertEqual(result["models"]["qwen14b_nf4"]["status"], "pending_complete_matrix")

    def test_deployments_use_only_shared_declared_questions(self):
        dataset, comparison, rows = self.fixture()
        comparison["model_query_ids"] = {"qwen7b": ["q1", "q2"], "qwen14b_nf4": ["q1"]}
        large = [
            {**row, "status": "answered", "metrics": dict.fromkeys(METRICS, 1)}
            for row in rows
            if row["query_id"] == "q1"
        ]
        result = deployment_analysis(dataset, comparison, {"qwen7b": rows, "qwen14b_nf4": large})
        score = result["configurations"]["closure_greedy@2048"]["answer_token_f1"]
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(result["document_count"], 1)
        self.assertEqual(score["right_mean"], 0)
        self.assertEqual(score["difference"], 1)
        self.assertIsNone(score["ci95"])
        self.assertIn("not a pure parameter-count effect", result["configuration_confound"])

    def test_refreshing_automatic_statistics_preserves_recorded_human_progress(self):
        dataset, comparison, rows = self.fixture()
        human = {
            "status": "independent_reviews_complete_adjudication_pending",
            "human_labels_created": True,
            "reviewer_count": 5,
            "adjudicated_primary_result": None,
        }
        comparison["paired_analysis"] = {"human_evaluation": human}
        result = method_analysis(dataset, comparison, {"qwen7b": rows})
        self.assertEqual(result["human_evaluation"], human)
        self.assertIsNot(result["human_evaluation"], human)

    def test_deployments_wait_for_full_planned_matrices(self):
        dataset, comparison, rows = self.fixture()
        result = deployment_analysis(dataset, comparison, {"qwen7b": rows, "qwen14b_nf4": rows[:1]})
        self.assertEqual(result["status"], "pending_complete_matrices")
        self.assertNotIn("configurations", result)

    def test_equal_deployments_keep_recorded_errors_with_zero_scores(self):
        dataset, comparison, rows = self.fixture()
        result = deployment_analysis(dataset, comparison, {"qwen7b": rows, "qwen14b_nf4": rows})
        score = result["configurations"]["closure_greedy@2048"]["answer_token_f1"]
        self.assertEqual(result["status"], "complete_paired_subset_analyzed")
        self.assertEqual(score["question_count"], 2)
        self.assertEqual(score["left_mean"], 0.5)
        self.assertEqual(score["difference"], 0)
        self.assertEqual(score["ci95"], [0, 0])


if __name__ == "__main__":
    unittest.main()
