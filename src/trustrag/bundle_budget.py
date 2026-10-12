"""Experimental dependency-closed evidence packing, independent of frozen v1.

The caller must discover structure without gold labels and count the entire rendered
prompt with the actual tokenizer. Closure proves only that declared dependencies
were retained, not that an answer/list is semantically complete. No model is loaded.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceAtom:
    atom_id: str
    source_id: str
    start: int
    end: int
    utility: float
    requires: tuple[str, ...] = ()


def checked_inputs(atoms: list[EvidenceAtom], sources: list[dict]) -> tuple[dict, dict]:
    if not isinstance(atoms, list) or not isinstance(sources, list):
        raise TypeError("Atoms and sources must be lists")
    known_sources = {}
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("Sources must be objects")
        for field in ("source_id", "document_id", "filename", "text"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"Invalid source field: {field}")
        source_id = source["source_id"]
        if source_id in known_sources or len(source["text"]) > 500000:
            raise ValueError("Duplicate source or source too large")
        spans = source.get("spans")
        if not isinstance(spans, list) or not spans:
            raise ValueError("Sources must retain original source spans")
        for span in spans:
            if not isinstance(span, dict):
                raise TypeError("Source spans must be objects")
            start, end, page = (span.get(k) for k in ("start", "end", "page_number"))
            if any(type(v) is not int for v in (start, end)):
                raise TypeError("Source offsets must be integers")
            if not 0 <= start < end <= len(source["text"]):
                raise ValueError("Invalid original source span")
            if page is not None:
                if type(page) is not int or page < 1:
                    raise ValueError("Invalid physical PDF page")
            elif not span.get("paragraph_id"):
                raise ValueError("Source needs a physical page or an original paragraph anchor")
            if not isinstance(span.get("block_id"), str) or not span["block_id"]:
                raise ValueError("Missing original block identifier")
        known_sources[source_id] = source
    known_atoms = {}
    occupied: dict[str, list[tuple[int, int]]] = {}
    for atom in atoms:
        if not isinstance(atom, EvidenceAtom):
            raise TypeError("Packing accepts only explicit EvidenceAtom values")
        if not isinstance(atom.atom_id, str) or not atom.atom_id or atom.atom_id in known_atoms:
            raise ValueError("Duplicate or empty atom_id")
        if atom.source_id not in known_sources:
            raise ValueError("Atom refers to an unknown original source")
        if type(atom.start) is not int or type(atom.end) is not int:
            raise TypeError("Atom offsets must be integers")
        source = known_sources[atom.source_id]
        if not 0 <= atom.start < atom.end <= len(source["text"]):
            raise ValueError("Atom range is outside the original source")
        if not source["text"][atom.start : atom.end].strip():
            raise ValueError("Whitespace-only ranges are not evidence atoms")
        if (
            type(atom.utility) not in (int, float)
            or not math.isfinite(atom.utility)
            or atom.utility < 0
        ):
            raise ValueError("Utility must be a finite, nonnegative relevance value")
        if not isinstance(atom.requires, tuple) or any(
            not isinstance(i, str) for i in atom.requires
        ):
            raise TypeError("Dependencies must be a tuple of atom identifiers")
        if len(set(atom.requires)) != len(atom.requires):
            raise ValueError("Duplicate dependency")
        # Disjoint atoms avoid artificially earning utility twice for one source range.
        for start, end in occupied.get(atom.source_id, []):
            if atom.start < end and atom.end > start:
                raise ValueError("Atoms in one source must not overlap; deduplicate before packing")
        occupied.setdefault(atom.source_id, []).append((atom.start, atom.end))
        coverage = sorted(
            (s["start"], s["end"])
            for s in source["spans"]
            if s["start"] < atom.end and s["end"] > atom.start
        )
        cursor = atom.start
        for start, end in coverage:
            if start > cursor and source["text"][cursor : min(start, atom.end)].strip():
                raise ValueError("Non-whitespace evidence lacks original block/page provenance")
            cursor = max(cursor, end)
        if cursor < atom.end and source["text"][cursor : atom.end].strip():
            raise ValueError("Non-whitespace evidence lacks original block/page provenance")
        known_atoms[atom.atom_id] = atom
    for atom in atoms:
        if any(dependency not in known_atoms for dependency in atom.requires):
            raise ValueError(
                "Unresolved structural dependency; do not treat a partial bundle as complete"
            )
    return known_atoms, known_sources


def dependency_closure(atom_id: str, known: dict[str, EvidenceAtom]) -> set[str]:
    pending, closed = [atom_id], set()
    while pending:
        current = pending.pop()
        if current not in closed:
            closed.add(current)
            pending.extend(known[current].requires)
    return closed


def original_pieces(selected: set[str], known: dict, sources: dict) -> list[dict]:
    """Coalesce only touching ranges: never hide removed original text by joining it."""
    pieces = []
    for source_id, source in sources.items():
        ranges = sorted(
            (known[i].start, known[i].end) for i in selected if known[i].source_id == source_id
        )
        merged: list[list[int]] = []
        for start, end in ranges:
            if merged and start == merged[-1][1]:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        for start, end in merged:
            pieces.append(
                {
                    "source_id": f"{source_id}:{start}-{end}",
                    "original_source_id": source_id,
                    "original_start": start,
                    "original_end": end,
                    "document_id": source["document_id"],
                    "filename": source["filename"],
                    "text": source["text"][start:end],
                    "spans": [
                        {
                            "block_id": span["block_id"],
                            **{
                                key: span[key]
                                for key in ("page_number", "paragraph_id", "section_name")
                                if key in span
                            },
                            "start": max(start, span["start"]) - start,
                            "end": min(end, span["end"]) - start,
                        }
                        for span in source["spans"]
                        if span["start"] < end and span["end"] > start
                    ],
                }
            )
    return pieces


def pack_context(
    atoms: list[EvidenceAtom],
    sources: list[dict],
    *,
    max_input_tokens: int,
    render: Callable[[list[dict]], Any],
    count_tokens: Callable[[Any], int],
    method: str = "closure_greedy",
) -> dict:
    """Experimental greedy heuristic, not an optimal knapsack or risk guarantee.

    `ranked_atoms` is the no-dependency ablation. Both methods re-count the full
    rendered prompt; per-piece token counts cannot substitute for this callback.
    """
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("Input token budget must be a positive integer")
    if method not in {"closure_greedy", "ranked_atoms"}:
        raise ValueError("Unknown experimental selection method")
    known, originals = checked_inputs(atoms, sources)
    counter_calls = 0

    def measured(selected: set[str]) -> tuple[list[dict], Any, int]:
        nonlocal counter_calls
        pieces = original_pieces(selected, known, originals)
        rendered = render(pieces)
        tokens = count_tokens(rendered)
        counter_calls += 1
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Tokenizer callback must return a nonnegative integer")
        return pieces, rendered, tokens

    selected: set[str] = set()
    pieces, rendered, tokens = measured(selected)
    if tokens > max_input_tokens:
        raise ValueError("The empty-evidence prompt already exceeds the input budget")
    closures = {i: dependency_closure(i, known) for i in known}
    while len(selected) < len(known):
        options = []
        for atom_id in sorted(known.keys() - selected):
            additional = (closures[atom_id] if method == "closure_greedy" else {atom_id}) - selected
            gain = sum(known[i].utility for i in additional)
            if gain <= 0:
                continue
            trial_pieces, trial_rendered, trial_tokens = measured(selected | additional)
            if trial_tokens > max_input_tokens:
                continue
            priority = (
                gain / max(1, trial_tokens - tokens)
                if method == "closure_greedy"
                else known[atom_id].utility
            )
            options.append(
                (priority, gain, atom_id, additional, trial_pieces, trial_rendered, trial_tokens)
            )
        if not options:
            break
        # Stable lexical tie-break; no test-label-aware reordering.
        best = min(options, key=lambda o: (-o[0], -o[1], o[2]))
        selected |= best[3]
        pieces, rendered, tokens = best[4:]
    missing = sorted({d for i in selected for d in known[i].requires if d not in selected})
    return {
        "method": method,
        "selected_atom_ids": sorted(selected),
        "omitted_atom_ids": sorted(known.keys() - selected),
        "sources": pieces,
        "rendered_prompt": rendered,
        "input_tokens": tokens,
        "max_input_tokens": max_input_tokens,
        "token_counter_calls": counter_calls,
        "dependency_closure_satisfied": not missing,
        "missing_dependencies": missing,
        "scope": "declared_dependency_closure_and_original_offsets_not_semantic_sufficiency",
        "semantic_support_verified": False,
        "answer_completeness_verified": False,
        "optimizer": "deterministic_greedy_not_globally_optimal",
    }
