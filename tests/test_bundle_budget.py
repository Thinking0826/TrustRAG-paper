"""Synthetic invariants for a research prototype; not evidence of real RAG gains."""

import unittest
from copy import deepcopy
from dataclasses import replace

from trustrag.bundle_budget import EvidenceAtom, pack_context


class BundleBudgetTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "source_id": "E1",
            "document_id": "doc",
            "filename": "report.pdf",
            "text": "AlphaBetaGammaDelta",
            "spans": [
                {"start": 0, "end": 9, "page_number": 2, "block_id": "b2"},
                {"start": 9, "end": 19, "page_number": 3, "block_id": "b3"},
            ],
        }
        self.atoms = [
            EvidenceAtom("a", "E1", 0, 5, 1),
            EvidenceAtom("b", "E1", 5, 9, 2),
            EvidenceAtom("c", "E1", 9, 14, 1),
            EvidenceAtom("d", "E1", 14, 19, 0.5),
        ]

    def pack(self, atoms=None, budget=30, method="closure_greedy", sources=None, counter=None):
        return pack_context(
            self.atoms if atoms is None else atoms,
            [self.source] if sources is None else sources,
            max_input_tokens=budget,
            render=lambda pieces: {"overhead": 2, "pieces": pieces},
            count_tokens=counter
            or (lambda value: value["overhead"] + sum(len(p["text"]) for p in value["pieces"])),
            method=method,
        )

    def test_preserves_original_ranges_and_physical_pages(self):
        result = self.pack()
        self.assertEqual(result["sources"][0]["text"], self.source["text"])
        self.assertEqual(result["sources"][0]["original_start"], 0)
        self.assertEqual({s["page_number"] for s in result["sources"][0]["spans"]}, {2, 3})
        self.assertTrue(result["dependency_closure_satisfied"])
        self.assertFalse(result["answer_completeness_verified"])
        self.assertFalse(result["semantic_support_verified"])

    def test_tight_budget_cannot_keep_one_member_of_atomic_cycle(self):
        atoms = [replace(self.atoms[0], requires=("b",)), replace(self.atoms[1], requires=("a",))]
        result = self.pack(atoms, budget=7)
        self.assertEqual(result["selected_atom_ids"], [])
        result = self.pack(atoms, budget=11)
        self.assertEqual(result["selected_atom_ids"], ["a", "b"])

    def test_no_dependency_ablation_is_detectably_incomplete(self):
        atoms = [replace(self.atoms[0], requires=("b",)), replace(self.atoms[1], requires=("a",))]
        result = self.pack(atoms, budget=7, method="ranked_atoms")
        self.assertEqual(result["selected_atom_ids"], ["b"])
        self.assertFalse(result["dependency_closure_satisfied"])
        self.assertEqual(result["missing_dependencies"], ["a"])

    def test_transitive_requirements_are_all_retained(self):
        atoms = [
            replace(self.atoms[0], requires=("b",)),
            replace(self.atoms[1], requires=("c",)),
            self.atoms[2],
        ]
        result = self.pack(atoms, budget=16)
        self.assertEqual(result["selected_atom_ids"], ["a", "b", "c"])

    def test_recount_full_prompt_including_nonadditive_source_overhead(self):
        def counter(value):
            return 3 + sum(len(p["text"]) + 6 for p in value["pieces"])

        result = self.pack(budget=13, counter=counter)
        self.assertEqual(result["selected_atom_ids"], ["b"])
        self.assertEqual(result["input_tokens"], 13)
        self.assertGreater(result["token_counter_calls"], 1)

    def test_non_contiguous_removed_text_is_not_silently_joined(self):
        result = self.pack([self.atoms[0], self.atoms[2]])
        self.assertEqual([p["text"] for p in result["sources"]], ["Alpha", "Gamma"])
        self.assertEqual(result["sources"][1]["original_start"], 9)
        self.assertEqual(result["sources"][1]["spans"][0]["start"], 0)

    def test_reproducible_selection_despite_atom_input_order(self):
        self.assertEqual(
            self.pack(budget=12)["selected_atom_ids"],
            self.pack(list(reversed(self.atoms)), budget=12)["selected_atom_ids"],
        )

    def test_inputs_are_not_modified(self):
        before = deepcopy(self.source)
        self.pack(budget=12)
        self.assertEqual(self.source, before)

    def test_unknown_dependency_does_not_produce_false_closure(self):
        with self.assertRaises(ValueError):
            self.pack([replace(self.atoms[0], requires=("missing",))])

    def test_duplicate_and_overlapping_ranges_do_not_earn_double_utility(self):
        for atoms in (
            [self.atoms[0], self.atoms[0]],
            [self.atoms[0], replace(self.atoms[1], start=4)],
        ):
            with self.subTest(atoms=atoms), self.assertRaises(ValueError):
                self.pack(atoms)

    def test_no_positive_utility_and_empty_candidates(self):
        for atoms in ([], [replace(atom, utility=0) for atom in self.atoms]):
            with self.subTest(atoms=atoms):
                result = self.pack(atoms)
                self.assertEqual(result["selected_atom_ids"], [])
                self.assertEqual(result["input_tokens"], 2)

    def test_invalid_relevance_values_and_offsets_are_rejected(self):
        for utility in (float("nan"), float("inf"), -1, True):
            with self.subTest(utility=utility), self.assertRaises(ValueError):
                self.pack([replace(self.atoms[0], utility=utility)])
        for atom in (replace(self.atoms[0], start=-1), replace(self.atoms[0], end=100)):
            with self.subTest(atom=atom), self.assertRaises(ValueError):
                self.pack([atom])
        with self.assertRaises(TypeError):
            self.pack([replace(self.atoms[0], start=True)])

    def test_unknown_source_and_page_provenance_gaps(self):
        with self.assertRaises(ValueError):
            self.pack([replace(self.atoms[0], source_id="unknown")])
        source = deepcopy(self.source)
        source["spans"] = [{"start": 1, "end": 19, "page_number": 2, "block_id": "b"}]
        with self.assertRaises(ValueError):
            self.pack(sources=[source])

    def test_whitespace_separators_may_lack_a_physical_page_span(self):
        source = {
            **self.source,
            "text": "Alpha\n\nBeta",
            "spans": [
                {"start": 0, "end": 5, "page_number": 2, "block_id": "b2"},
                {"start": 7, "end": 11, "page_number": 3, "block_id": "b3"},
            ],
        }
        result = self.pack([EvidenceAtom("a", "E1", 0, 11, 1)], sources=[source])
        self.assertEqual(result["sources"][0]["text"], "Alpha\n\nBeta")

    def test_whitespace_only_candidate_is_not_evidence(self):
        source = {
            **self.source,
            "text": "Alpha   Beta",
            "spans": [{"start": 0, "end": 12, "page_number": 2, "block_id": "b2"}],
        }
        with self.assertRaises(ValueError):
            self.pack([EvidenceAtom("a", "E1", 5, 8, 1)], sources=[source])

    def test_auxiliary_gold_and_scores_are_not_exported_into_evidence(self):
        source = deepcopy(self.source)
        source["gold"] = "not allowed in prompt"
        source["spans"][0]["gold_label"] = "not allowed in prompt"
        result = self.pack(sources=[source])
        self.assertNotIn("gold", str(result["sources"]))

    def test_empty_prompt_over_budget_cannot_claim_a_valid_pack(self):
        with self.assertRaises(ValueError):
            self.pack(budget=1)

    def test_wrong_budget_and_counter_types(self):
        for budget in (0, -1, True, 2.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.pack(budget=budget)
        for counter in (lambda _: True, lambda _: -1, lambda _: 1.5):
            with self.subTest(counter=counter), self.assertRaises(ValueError):
                self.pack(counter=counter)

    def test_methods_and_duplicate_dependencies_are_validated(self):
        with self.assertRaises(ValueError):
            self.pack(method="gold_aware")
        with self.assertRaises(ValueError):
            self.pack([replace(self.atoms[0], requires=("b", "b")), self.atoms[1]])
        with self.assertRaises(TypeError):
            self.pack([replace(self.atoms[0], requires=["b"])])


if __name__ == "__main__":
    unittest.main()
