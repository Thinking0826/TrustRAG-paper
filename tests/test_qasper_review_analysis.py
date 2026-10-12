"""Synthetic examples of multi-rater aggregation and explicit provenance separation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trustrag.qasper_review_analysis import aggregate, finish_adjudication, prepare_adjudication
from trustrag.qasper_simulation import read_packet


class ReviewAnalysisTests(unittest.TestCase):
    def fixture(self, simulated=False):
        yes = {
            "relevant": "yes",
            "factual": "yes",
            "complete": "yes",
            "citations_supported": "yes",
            "unsupported_additions": "no",
            "notes": "Synthetic fixture only",
        }
        packet = {
            "schema_version": "qasper_blind_review_v1",
            "study_id": "synthetic",
            "cases": [
                {"case_id": "R001", "outcome": "answered"},
                {"case_id": "R002", "outcome": "abstained"},
            ],
        }
        private = {
            "cases": [
                {
                    "case_id": "R001",
                    "model": "qwen7b",
                    "query_id": "q1",
                    "document_id": "d1",
                    "method": "closure_greedy",
                },
                {
                    "case_id": "R002",
                    "model": "qwen7b",
                    "query_id": "q1",
                    "document_id": "d1",
                    "method": "density_atoms",
                },
            ]
        }
        reports = [
            {
                "reviewer_id": f"fixture_{i}",
                "schema_version": packet["schema_version"],
                "study_id": "synthetic",
                "independent": True,
                "ratings": {"R001": dict(yes), "R002": dict(yes)},
            }
            for i in range(5)
        ]
        if simulated:
            for r in reports:
                r["record_type"] = "ai_simulated_review"
                r["independent"] = False
                r["human_labels_created"] = False
        return packet, private, reports

    def test_five_exports_are_ten_ratings_not_ten_questions(self):
        result = aggregate(*self.fixture())
        self.assertEqual(result["reviewer_count"], 5)
        self.assertEqual(result["rating_records"], 10)
        self.assertEqual(result["answered_case_count"], 1)
        self.assertEqual(result["nonanswer_case_count"], 1)
        self.assertEqual(result["status"], "awaiting_source_based_adjudication")
        score = result["individual_contrasts"]["fixture_0"]["qwen7b"]
        self.assertEqual(score["question_count"], 1)
        self.assertEqual(score["density_success_count"], 0)
        self.assertEqual(score["paired_difference"], 1)

    def test_simulation_cannot_be_reported_as_human(self):
        packet, private, reports = self.fixture(simulated=True)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            aggregate(packet, private, reports)
        result = aggregate(packet, private, reports, simulated=True)
        self.assertEqual(result["human_participants"], 0)
        self.assertEqual(result["status"], "simulation_workflow_complete")

    def test_uncertainties_remain_unadjudicated(self):
        packet, private, reports = self.fixture()
        reports[0]["ratings"]["R001"]["factual"] = "unsure"
        result = aggregate(packet, private, reports)
        self.assertEqual(result["uncertainty_counts"]["factual"], 1)
        case = result["adjudication_cases"][0]
        self.assertIsNone(case["adjudicated_rating"])
        self.assertEqual(case["uncertain_fields"], ["factual"])

    def test_unreviewed_sample_is_not_success(self):
        packet, private, reports = self.fixture()
        del reports[0]["ratings"]["R001"]
        with self.assertRaisesRegex(ValueError, "not complete"):
            aggregate(packet, private, reports)

    def test_empty_response_uncertainty_is_not_a_source_adjudication_case(self):
        packet, private, reports = self.fixture()
        for report in reports:
            report["ratings"]["R002"]["factual"] = "unsure"
        result = aggregate(packet, private, reports)
        self.assertEqual(result["adjudication_cases"], [])
        self.assertEqual(result["answered_uncertainty_counts"]["factual"], 0)
        self.assertEqual(result["uncertainty_counts"]["factual"], 5)

    def confirmation(self):
        return {
            "schema_version": "qasper_review_confirmation_v1",
            "study_id": "synthetic",
            "human_reviewers": True,
            "independent_reviews": True,
            "reviewer_ids": [f"fixture_{i}" for i in range(5)],
            "statement": "Synthetic test of a coordinator-confirmed legacy export",
        }

    def test_confirmed_legacy_metadata_is_preserved_not_silently_rewritten(self):
        packet, private, reports = self.fixture(simulated=True)
        result = aggregate(packet, private, reports, provenance_confirmation=self.confirmation())
        self.assertEqual(result["human_participants"], 5)
        self.assertEqual(result["provenance"]["basis"], "coordinator_user_confirmation")
        self.assertFalse(result["provenance"]["input_files_modified_by_analysis"])
        self.assertEqual(reports[0]["record_type"], "ai_simulated_review")
        self.assertFalse(reports[0]["independent"])
        self.assertEqual(result["provenance"]["export_metadata"][0]["independent"], False)
        self.assertEqual(result["status"], "awaiting_source_based_adjudication")

    def test_confirmation_does_not_apply_to_another_review_sample(self):
        packet, private, reports = self.fixture(simulated=True)
        confirmation = self.confirmation()
        confirmation["reviewer_ids"][0] = "another_person"
        with self.assertRaisesRegex(ValueError, "must cover"):
            aggregate(packet, private, reports, provenance_confirmation=confirmation)

    def test_confirmation_does_not_replace_the_study_schema_check(self):
        packet, private, reports = self.fixture(simulated=True)
        reports[0]["study_id"] = "another_study"
        with self.assertRaisesRegex(ValueError, "this study"):
            aggregate(packet, private, reports, provenance_confirmation=self.confirmation())

    def test_confirmed_human_reviews_cannot_be_used_as_a_simulation(self):
        with self.assertRaisesRegex(ValueError, "must cover"):
            aggregate(
                *self.fixture(simulated=True),
                simulated=True,
                provenance_confirmation=self.confirmation(),
            )

    def test_adjudication_packet_keeps_sources_and_does_not_prefill_final_ratings(self):
        packet, private, reports = self.fixture()
        packet["review_count"] = 2
        packet["documents"] = {"synthetic": {"title": "Synthetic source"}}
        reports[0]["ratings"]["R001"]["factual"] = "unsure"
        analysis = aggregate(packet, private, reports)
        with TemporaryDirectory() as folder:
            output = Path(folder) / "adjudication.html"
            prepare_adjudication(packet, analysis, output)
            reduced = read_packet(output)
            self.assertEqual(reduced["schema_version"], "qasper_adjudication_v1")
            self.assertEqual(reduced["review_count"], 1)
            self.assertEqual(reduced["cases"], packet["cases"][:1])
            self.assertEqual(reduced["documents"], packet["documents"])
            page = output.read_text()
            self.assertIn("Synthetic fixture only", page)
            self.assertIn("不是新一轮独立盲评", page)
            self.assertIn("ratings:{}", page)

    def adjudication_fixture(self):
        packet, private, reports = self.fixture()
        reports[0]["ratings"]["R001"]["factual"] = "unsure"
        analysis = aggregate(packet, private, reports)
        decision = {
            "schema_version": "qasper_adjudication_v1",
            "study_id": "synthetic_adjudication",
            "reviewer_id": "adjudicator_01",
            "independent": False,
            "human_labels_created": True,
            "record_type": "actual_human_adjudication",
            "ratings": {"R001": dict(reports[1]["ratings"]["R001"])},
        }
        return packet, private, reports, analysis, decision

    def test_adjudication_merges_decisions_and_keeps_pre_adjudication_agreement(self):
        fixture = self.adjudication_fixture()
        result = finish_adjudication(*fixture)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["agreement"], fixture[3]["agreement"])
        self.assertIsNone(fixture[3]["adjudication_cases"][0]["adjudicated_rating"])
        score = result["adjudicated_results"]["method_contrasts"]["qwen7b"]
        self.assertEqual(score["question_count"], 1)
        self.assertEqual(score["closure_success_count"], 1)
        self.assertEqual(score["density_success_count"], 0)
        self.assertEqual(score["difference"], 1)
        self.assertIsNone(score["ci95"])

    def test_retained_uncertainty_is_not_counted_as_verified_success(self):
        fixture = self.adjudication_fixture()
        fixture[4]["ratings"]["R001"]["factual"] = "unsure"
        result = finish_adjudication(*fixture)
        self.assertEqual(result["status"], "human_review_complete_with_retained_uncertainty")
        self.assertTrue(result["adjudication"]["record_complete"])
        self.assertEqual(result["adjudication"]["uncertain_answer_case_ids"], ["R001"])
        self.assertEqual(result["adjudication"]["unresolved_joint_outcome_count"], 1)
        score = result["adjudicated_results"]["method_contrasts"]["qwen7b"]
        self.assertEqual(score["question_count"], 1)
        self.assertEqual(score["closure_success_count"], 0)

    def test_known_failure_is_distinct_from_an_unresolved_joint_outcome(self):
        fixture = self.adjudication_fixture()
        fixture[4]["ratings"]["R001"]["factual"] = "unsure"
        fixture[4]["ratings"]["R001"]["citations_supported"] = "no"
        result = finish_adjudication(*fixture)
        self.assertEqual(result["adjudication"]["unresolved_joint_outcome_count"], 0)
        self.assertEqual(
            result["adjudicated_results"]["case_decisions"][0]["joint_outcome"],
            "verified_non_success",
        )

    def test_missing_adjudication_is_not_filled_from_majority_votes(self):
        fixture = self.adjudication_fixture()
        fixture[4]["ratings"] = {}
        with self.assertRaisesRegex(ValueError, "exactly"):
            finish_adjudication(*fixture)

    def test_legacy_adjudication_requires_record_specific_human_confirmation(self):
        fixture = self.adjudication_fixture()
        fixture[4].update(
            record_type="ai_simulated_adjudication", independent=False, human_labels_created=False
        )
        with self.assertRaisesRegex(ValueError, "Legacy"):
            finish_adjudication(*fixture)
        confirmation = {
            "schema_version": "qasper_adjudication_confirmation_v1",
            "study_id": "synthetic",
            "adjudicator_id": "adjudicator_01",
            "human_adjudication": True,
            "statement": "Synthetic example of coordinator-confirmed legacy metadata",
        }
        result = finish_adjudication(*fixture, confirmation=confirmation)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(fixture[4]["independent"])
        self.assertEqual(result["adjudication"]["provenance_basis"], "coordinator_user_confirmation")

    def test_adjudication_reasons_are_retained_and_required(self):
        fixture = self.adjudication_fixture()
        result = finish_adjudication(*fixture)
        self.assertEqual(
            result["adjudication_cases"][0]["adjudication_reason"], "Synthetic fixture only"
        )
        fixture[4]["ratings"]["R001"]["notes"] = ""
        with self.assertRaisesRegex(ValueError, "reason"):
            finish_adjudication(*fixture)

    def test_human_deployment_contrast_uses_only_shared_questions(self):
        packet, private, reports, _, adjudication = self.adjudication_fixture()
        for case_id, model, question, method in (
            ("R003", "qwen7b", "q2", "closure_greedy"),
            ("R004", "qwen7b", "q2", "density_atoms"),
            ("R005", "qwen14b_nf4", "q1", "closure_greedy"),
            ("R006", "qwen14b_nf4", "q1", "density_atoms"),
        ):
            packet["cases"].append({"case_id": case_id, "outcome": "answered"})
            private["cases"].append(
                {
                    "case_id": case_id,
                    "model": model,
                    "query_id": question,
                    "method": method,
                    "document_id": "d2" if question == "q2" else "d1",
                }
            )
            for report in reports:
                report["ratings"][case_id] = dict(reports[1]["ratings"]["R001"])
        analysis = aggregate(packet, private, reports)
        result = finish_adjudication(packet, private, reports, analysis, adjudication)
        methods = result["adjudicated_results"]["method_contrasts"]
        deployment = result["adjudicated_results"]["deployment_contrasts"]
        self.assertEqual(methods["qwen7b"]["question_count"], 2)
        self.assertEqual(deployment["closure_greedy"]["question_count"], 1)
        self.assertEqual(deployment["closure_greedy"]["difference"], 0)
        self.assertEqual(deployment["density_atoms"]["difference"], 1)


if __name__ == "__main__":
    unittest.main()
