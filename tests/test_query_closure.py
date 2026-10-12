"""Small CPU tests of source grounding, conditional edges and matched controls."""

import json

import numpy as np
import pytest

from trustrag.bundle_budget import EvidenceAtom
from trustrag.query_closure import (
    diversity_pack,
    parse_plan,
    planned_atoms,
    planning_messages,
    select,
)


def fixture_record():
    texts = ["Method A is our baseline. ", "This variant adds features. ", "Unrelated background. "]
    text = "".join(texts)
    atoms, start = [], 0
    for i, part in enumerate(texts):
        atoms.append(EvidenceAtom(f"u{i}", "p1", start, start + len(part), 0.7))
        start += len(part)
    source = {
        "source_id": "p1",
        "document_id": "d",
        "filename": "d",
        "text": text,
        "section_name": "Method",
        "spans": [{"block_id": "p1", "paragraph_id": "p1", "start": 0, "end": len(text)}],
    }
    return atoms, source


def test_conditional_dependency_is_directional():
    atoms, _ = fixture_record()
    plan = {
        "status": "planned",
        "plan": parse_plan(
            json.dumps({"evidence": [{"id": "u1", "relevance": 2, "requires": ["u0"]}]}), atoms
        ),
    }
    closed, flat = planned_atoms(atoms, plan, True), planned_atoms(atoms, plan, False)
    assert closed[1].requires == ("u0",)
    assert closed[0].requires == ()
    assert [a.utility for a in flat] == [a.utility for a in closed]
    assert all(not a.requires for a in flat)
    assert closed[2].utility == 0


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "not_real", "relevance": 2, "requires": []},
        {"id": "u1", "relevance": 2, "requires": ["not_real"]},
        {"id": "u1", "relevance": True, "requires": []},
        {"id": "u1", "relevance": 2, "requires": ["u1"]},
    ],
)
def test_ungrounded_plans_rejected(entry):
    atoms, _ = fixture_record()
    with pytest.raises(ValueError):
        parse_plan(json.dumps({"evidence": [entry]}), atoms)


def test_failed_plan_uses_same_control_fallback():
    atoms, _ = fixture_record()
    assert planned_atoms(atoms, {"status": "error"}, True) == planned_atoms(
        atoms, {"status": "error"}, False
    )


def test_empty_optional_edges_and_stray_quote_do_not_drop_valid_ids():
    atoms, _ = fixture_record()
    raw = '{"evidence":[{"id":"u1","relevance":2}]}"'
    assert parse_plan(raw, atoms)["evidence"] == [{"id": "u1", "relevance": 2, "requires": []}]


def test_planning_stops_at_complete_object_not_partial_braces():
    from trustrag.qasper_followup import complete_json_object

    assert complete_json_object('```json\n{"evidence": []}')
    assert not complete_json_object('{"evidence": [{"id":"brace } inside string"}')


def test_planning_input_does_not_include_gold():
    atoms, source = fixture_record()
    from dataclasses import asdict

    record = {
        "sources": [source],
        "atoms": [asdict(a) for a in atoms],
        "references": [{"answer_units": ["SECRET GOLD"]}],
    }
    messages = planning_messages("Which method?", record)
    assert "SECRET GOLD" not in json.dumps(messages)
    assert "Which method?" in messages[1]["content"]


def test_query_closure_keeps_antecedent_without_unrelated_sibling():
    from dataclasses import asdict

    atoms, source = fixture_record()
    record = {"sources": [source], "atoms": [asdict(a) for a in atoms]}
    plan = {
        "status": "planned",
        "plan": {"evidence": [{"id": "u1", "relevance": 2, "requires": ["u0"]}]},
    }
    result = select(
        record,
        "Which method?",
        plan,
        np.eye(3),
        method="query_closure",
        budget=100,
        render=lambda pieces: "".join(p["text"] for p in pieces),
        count=len,
    )
    assert result["selected_atom_ids"] == ["u0", "u1"]
    assert result["sources"][0]["text"] == source["text"][: atoms[1].end]


@pytest.mark.parametrize("method", ["semantic_mmr", "coverage_submodular"])
def test_diversity_controls_respect_rendered_budget(method):
    atoms, source = fixture_record()
    result = diversity_pack(
        atoms,
        [source],
        "features",
        np.eye(3),
        budget=45,
        render=lambda pieces: "HEADER:" + "".join(p["text"] for p in pieces),
        count=len,
        method=method,
    )
    assert result["input_tokens"] <= 45
    for piece in result["sources"]:
        assert piece["text"] == source["text"][piece["original_start"] : piece["original_end"]]


@pytest.mark.parametrize("dependency_selected", [True, False])
def test_saved_mechanism_diagnosis_distinguishes_active_and_inactive_edges(dependency_selected):
    from trustrag.qasper_followup import mechanism_diagnostics

    plans = [
        {
            "query_id": "q",
            "status": "planned",
            "planning_seconds": 2,
            "cold_load_seconds": 1,
            "plan": {"evidence": [{"id": "u1", "requires": ["u0"]}]},
        }
    ]
    contexts = [
        {
            "query_id": "q",
            "method": "planner_flat",
            "budget": 1024,
            "selected_atom_ids": ["u0", "u1"] if dependency_selected else ["u1"],
            "messages": "both" if dependency_selected else "one",
            "input_tokens": 200,
            "selection_seconds": 0.1,
        },
        {
            "query_id": "q",
            "method": "query_closure",
            "budget": 1024,
            "selected_atom_ids": ["u0", "u1"],
            "messages": "both",
            "input_tokens": 200,
            "selection_seconds": 0.1,
        },
    ]
    answers = [
        {
            "query_id": "q",
            "method": method,
            "budget": 1024,
            "raw_output": "answer",
            "generation_seconds": 1,
        }
        for method in ("planner_flat", "query_closure")
    ]
    result = mechanism_diagnostics(plans, contexts, answers)
    assert result["plans_with_dependencies"] == 1
    assert result["budgets"]["1024"]["changed_messages"] == int(not dependency_selected)
    assert result["budgets"]["1024"]["flat_contexts_missing_declared_dependencies"] == int(
        not dependency_selected
    )
    assert result["budgets"]["1024"]["changed_raw_answers"] == 0
    assert result["observed_cost"]["new_answer_records"] == 2
