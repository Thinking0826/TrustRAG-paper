"""Aggregate independent QASPER ratings; leave source-based adjudication to people."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from trustrag.qasper_simulation import FIELDS, fleiss, joint_success, read_packet
from trustrag.research_statistics import paired_cluster_interval


def aggregate(packet, private, reports, simulated=False, provenance_confirmation=None):
    cases = {c["case_id"]: c for c in packet["cases"]}
    keys = {c["case_id"]: c for c in private["cases"]}
    if set(cases) != set(keys):
        raise ValueError("The coordinator mapping and case IDs differ")
    ids = [r.get("reviewer_id") for r in reports]
    if len(reports) < 2 or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Use at least two distinct reviewer records")
    confirmed = provenance_confirmation is not None
    if confirmed and (
        simulated
        or provenance_confirmation.get("schema_version") != "qasper_review_confirmation_v1"
        or provenance_confirmation.get("study_id") != packet["study_id"]
        or provenance_confirmation.get("human_reviewers") is not True
        or provenance_confirmation.get("independent_reviews") is not True
        or sorted(provenance_confirmation.get("reviewer_ids", [])) != sorted(ids)
        or not provenance_confirmation.get("statement")
    ):
        raise ValueError("The human provenance confirmation must cover these study reviewers")
    for report in reports:
        is_ai = report.get("record_type") == "ai_simulated_review"
        if is_ai != simulated and not confirmed:
            raise ValueError("Simulation and actual-person records cannot be combined")
        if not simulated and (
            report.get("schema_version") != packet["schema_version"]
            or report.get("study_id") != packet["study_id"]
            or (report.get("independent") is not True and not confirmed)
        ):
            raise ValueError("Actual records need this study's schema and independent declaration")
        if set(report.get("ratings", {})) != set(cases):
            raise ValueError(f"{report['reviewer_id']}: the fixed sample is not complete")
        for case, row in report["ratings"].items():
            if any(row.get(f) not in ("yes", "no", "unsure") for f in FIELDS):
                raise ValueError(f"{report['reviewer_id']}/{case}: a judgment is missing")
    answered = [c for c in cases if cases[c]["outcome"] == "answered"]
    result = {
        "record_type": "ai_trial_analysis" if simulated else "actual_review_exports_analysis",
        "study_id": packet["study_id"],
        "reviewer_count": len(reports),
        "case_count": len(cases),
        "rating_records": len(reports) * len(cases),
        "answered_case_count": len(answered),
        "nonanswer_case_count": len(cases) - len(answered),
        "human_participants": 0 if simulated else len(reports),
        "status": "simulation_workflow_complete"
        if simulated
        else "awaiting_source_based_adjudication",
        "scope": "AI rubric trial, not human outcomes"
        if simulated
        else "Independent exported judgments; adjudicated primary outcomes remain pending",
        "agreement": {
            f: fleiss([[r["ratings"][c][f] for r in reports] for c in answered]) for f in FIELDS
        },
        "agreement_scope": "Displayed answers only; nonanswers retain zero joint success in the full denominator",
        "all_record_agreement": {
            f: fleiss([[r["ratings"][c][f] for r in reports] for c in cases]) for f in FIELDS
        },
        "individual_contrasts": {},
        "adjudication_cases": [],
        "nonanswer_coding_disagreements": [],
        "provenance": {
            "basis": "coordinator_user_confirmation"
            if confirmed
            else "simulation_exports"
            if simulated
            else "reviewer_export_declarations",
            "confirmation": provenance_confirmation,
            "input_files_modified_by_analysis": False,
            "export_metadata": [
                {
                    "reviewer_id": r["reviewer_id"],
                    "record_type": r.get("record_type"),
                    "independent": r.get("independent"),
                    "human_labels_created": r.get("human_labels_created"),
                }
                for r in reports
            ],
        },
    }
    grouped = {}
    for c, key in keys.items():
        grouped.setdefault((key["model"], key["query_id"]), {})[key["method"]] = c
    for report in reports:
        comparisons = {}
        for model in sorted({model for model, _ in grouped}):
            pair_ids = [pair for (m, _), pair in grouped.items() if m == model]
            left = sum(
                joint_success(
                    report["ratings"][pair["closure_greedy"]],
                    cases[pair["closure_greedy"]]["outcome"],
                )
                for pair in pair_ids
            )
            right = sum(
                joint_success(
                    report["ratings"][pair["density_atoms"]],
                    cases[pair["density_atoms"]]["outcome"],
                )
                for pair in pair_ids
            )
            comparisons[model] = {
                "question_count": len(pair_ids),
                "closure_success_count": left,
                "density_success_count": right,
                "paired_difference": (left - right) / len(pair_ids),
                "scope": "Individual reviewer judgments, not adjudicated ground truth",
            }
        result["individual_contrasts"][report["reviewer_id"]] = comparisons
    for c in cases:
        ratings = {r["reviewer_id"]: r["ratings"][c] for r in reports}
        disputed = [f for f in FIELDS if len({r[f] for r in ratings.values()}) > 1]
        uncertain = [f for f in FIELDS if any(r[f] == "unsure" for r in ratings.values())]
        if c not in answered:
            if disputed:
                result["nonanswer_coding_disagreements"].append(
                    {"case_id": c, "disputed_fields": disputed, "independent_ratings": ratings}
                )
            continue
        if disputed or uncertain:
            result["adjudication_cases"].append(
                {
                    "case_id": c,
                    "disputed_fields": disputed,
                    "uncertain_fields": uncertain,
                    "independent_ratings": ratings,
                    "adjudicated_rating": None,
                    "adjudication_reason": None,
                }
            )
    result["uncertainty_counts"] = {
        f: sum(r["ratings"][c][f] == "unsure" for r in reports for c in cases) for f in FIELDS
    }
    result["answered_uncertainty_counts"] = {
        f: sum(r["ratings"][c][f] == "unsure" for r in reports for c in answered) for f in FIELDS
    }
    result["decision_policy"] = (
        "Agreement and majority votes are descriptive; no source-based adjudicated labels are manufactured"
    )
    return result


def finish_adjudication(packet, private, reports, analysis, adjudication, confirmation=None):
    """Merge unanimous ratings and actual source decisions, retaining every uncertainty."""
    if analysis["human_participants"] == 0:
        raise ValueError("The human endpoint requires actual independent review records")
    if (
        adjudication.get("schema_version") != "qasper_adjudication_v1"
        or adjudication.get("study_id") != packet["study_id"] + "_adjudication"
        or not adjudication.get("reviewer_id")
    ):
        raise ValueError("The adjudication must belong to this study")
    if confirmation is not None:
        if (
            confirmation.get("schema_version") != "qasper_adjudication_confirmation_v1"
            or confirmation.get("study_id") != packet["study_id"]
            or confirmation.get("adjudicator_id") != adjudication["reviewer_id"]
            or confirmation.get("human_adjudication") is not True
            or not confirmation.get("statement")
        ):
            raise ValueError("The confirmation must cover this human adjudication record")
    elif (
        adjudication.get("record_type") == "ai_simulated_adjudication"
        or adjudication.get("human_labels_created") is False
    ):
        raise ValueError("Legacy adjudication flags require explicit coordinator confirmation")
    queue = {row["case_id"]: row for row in analysis["adjudication_cases"]}
    ratings = adjudication.get("ratings", {})
    if set(ratings) != set(queue):
        raise ValueError("Adjudication must cover exactly the unresolved fixed cases")
    for case_id, rating in ratings.items():
        if any(rating.get(f) not in ("yes", "no", "unsure") for f in FIELDS):
            raise ValueError(f"{case_id}: a final judgment is missing")
        if not rating.get("notes", "").strip():
            raise ValueError(f"{case_id}: the source-based reason is missing")
    result = deepcopy(analysis)
    keys = {row["case_id"]: row for row in private["cases"]}
    decisions = []
    for case in packet["cases"]:
        case_id = case["case_id"]
        if case_id in queue:
            row = ratings[case_id]
            source = "source_based_human_adjudication"
            reason = row["notes"]
        elif case["outcome"] != "answered":
            row = dict.fromkeys(FIELDS, "no")
            row["factual"] = "unsure"
            source = "prespecified_nonanswer_coding"
            reason = "No displayed answer; retained as a non-success in the full denominator"
        else:
            independent = [report["ratings"][case_id] for report in reports]
            if any(len({r[f] for r in independent}) != 1 for f in FIELDS):
                raise ValueError(f"{case_id}: disagreement was omitted from adjudication")
            row = independent[0]
            source = "unanimous_independent_ratings"
            reason = "All independent reviewers agree; original notes remain in their exports"
        successful = joint_success(row, case["outcome"])
        definite_failure = any(row[f] == "no" for f in FIELDS[:4]) or (
            row["unsupported_additions"] == "yes"
        )
        state = (
            "nonanswer"
            if case["outcome"] != "answered"
            else "verified_success"
            if successful
            else "verified_non_success"
            if definite_failure
            else "unresolved"
        )
        decisions.append(
            {
                **keys[case_id],
                "outcome": case["outcome"],
                "rating": {f: row[f] for f in FIELDS},
                "decision_source": source,
                "reason": reason,
                "joint_outcome": state,
                "verified_joint_success": bool(successful),
            }
        )
    for row in result["adjudication_cases"]:
        rating = ratings[row["case_id"]]
        row["adjudicated_rating"] = {f: rating[f] for f in FIELDS}
        row["adjudication_reason"] = rating["notes"]
    grouped = {}
    for row in decisions:
        grouped.setdefault(row["model"], {}).setdefault(row["query_id"], {})[row["method"]] = row
    contrasts = {}
    for model, questions in sorted(grouped.items()):
        pairs = [
            (
                rows["closure_greedy"]["document_id"],
                rows["closure_greedy"]["verified_joint_success"],
                rows["density_atoms"]["verified_joint_success"],
            )
            for _, rows in sorted(questions.items())
        ]
        counts = Counter((bool(left), bool(right)) for _, left, right in pairs)
        contrasts[model] = {
            **paired_cluster_interval(pairs),
            "role": "primary" if model == "qwen7b" else "secondary",
            "closure_success_count": sum(left for _, left, _ in pairs),
            "density_success_count": sum(right for _, _, right in pairs),
            "paired_outcomes": {
                "both_verified_success": counts[(True, True)],
                "closure_only": counts[(True, False)],
                "density_only": counts[(False, True)],
                "neither_verified_success": counts[(False, False)],
            },
            "method_outcomes": {
                method: dict(Counter(rows[method]["joint_outcome"] for rows in questions.values()))
                for method in ("closure_greedy", "density_atoms")
            },
        }
    deployment = {}
    if "qwen7b" in grouped and "qwen14b_nf4" in grouped:
        shared = sorted(set(grouped["qwen7b"]) & set(grouped["qwen14b_nf4"]))
        for method in ("closure_greedy", "density_atoms"):
            deployment[method] = {
                **paired_cluster_interval(
                    [
                        (
                            grouped["qwen7b"][q][method]["document_id"],
                            grouped["qwen14b_nf4"][q][method]["verified_joint_success"],
                            grouped["qwen7b"][q][method]["verified_joint_success"],
                        )
                        for q in shared
                    ]
                ),
                "role": "secondary_paired_deployment",
                "contrast": "14B NF4 minus 7B BF16; not an isolated parameter-count effect",
            }
    uncertain = [
        row["case_id"]
        for row in decisions
        if row["outcome"] == "answered" and "unsure" in row["rating"].values()
    ]
    result["status"] = "human_review_complete_with_retained_uncertainty" if uncertain else "complete"
    result["scope"] = "Independent human ratings plus source adjudication; full fixed sample retained"
    result["adjudication"] = {
        "adjudicator_id": adjudication["reviewer_id"],
        "submitted_case_count": len(ratings),
        "record_complete": True,
        "provenance_basis": "coordinator_user_confirmation"
        if confirmation is not None
        else "adjudicator_export_declaration",
        "confirmation": confirmation,
        "original_export_metadata": {
            k: adjudication.get(k)
            for k in ("record_type", "independent", "human_labels_created")
        },
        "uncertain_answer_case_ids": uncertain,
        "unresolved_joint_outcome_count": sum(r["joint_outcome"] == "unresolved" for r in decisions),
        "final_answered_uncertainty_counts": {
            f: sum(r["rating"][f] == "unsure" for r in decisions if r["outcome"] == "answered")
            for f in FIELDS
        },
    }
    result["adjudicated_results"] = {
        "endpoint": "verified_joint_success",
        "uncertainty_policy": "Uncertain outcomes are retained, not counted as verified success",
        "contrast": "dependency closure minus independent density at 2048 input tokens",
        "bootstrap_samples": 5000,
        "seed": 42,
        "interval_method": "question-weighted paired percentile bootstrap over document clusters",
        "method_contrasts": contrasts,
        "deployment_contrasts": deployment,
        "case_decisions": decisions,
    }
    return result


def prepare_adjudication(packet, analysis, output):
    """Reuse the source viewer for the unresolved cases; leave final judgments blank."""
    queue = {row["case_id"]: row for row in analysis["adjudication_cases"]}
    if not queue:
        return
    cases = [case for case in packet["cases"] if case["case_id"] in queue]
    reduced = {
        **packet,
        "schema_version": "qasper_adjudication_v1",
        "study_id": packet["study_id"] + "_adjudication",
        "review_count": len(cases),
        "cases": cases,
    }
    template = Path(__file__).with_name("qasper_review.html").read_text(encoding="utf-8")
    template = template.replace("科学文档问答 · 独立盲评", "科学文档问答 · 原文分歧裁决")
    template = template.replace("QASPER 独立盲评", "QASPER 原文分歧裁决")
    template = template.replace("填写指南与独立评阅约定", "裁决说明与原文核验规则")
    old_guide = template.split("<summary>裁决说明与原文核验规则</summary>\n<p>", 1)[1]
    old_guide = old_guide.split("</p>", 1)[0]
    template = template.replace(
        old_guide,
        f"五份独立记录已提交。本页只含{len(cases)}条存在分歧或不确定的回答，"
        "不是新一轮独立盲评。先查看下方原始意见，再依据对应完整段落及全文填写最终判断；"
        "不自动采用多数票。备注中记录裁决理由和段落编号，仍无法确定时保留不确定。"
        "模型、方法和自动分数继续隐藏；导出的裁决JSON与原始评分分开保存。",
    )
    template = template.replace("reviewer_01 至 reviewer_05", "adjudicator_01")
    template = template.replace(
        "我独立实际评阅，未查看其他人的记录，未用AI代填",
        "我实际核验原文后作出裁决，理由由本人填写，未用AI代填",
    )
    template = template.replace("<h2>独立判断</h2>", "<h2>最终裁决</h2>")
    template = template.replace("导出个人记录", "导出裁决记录")
    panels = []
    for case in cases:
        row = queue[case["case_id"]]
        parts = [f"<details><summary>{case['case_id']} · 原始独立意见</summary>"]
        for reviewer, rating in row["independent_ratings"].items():
            judgments = "; ".join(f"{f}={rating[f]}" for f in FIELDS)
            parts.append(
                f"<p><b>{html.escape(reviewer)}</b>: {html.escape(judgments)}</p>"
                f"<p class='text'>{html.escape(rating.get('notes', ''))}</p>"
            )
        parts.append("</details>")
        panels.extend(parts)
    template = template.replace(
        "<h2 id=\"case-title\"></h2>",
        f"<details><summary>{len(cases)}项原始独立判断与备注（按盲编号查找）</summary>"
        + "\n".join(panels)
        + "</details><h2 id=\"case-title\"></h2>",
    )
    encoded = json.dumps(reduced, ensure_ascii=False).replace("</", "<\\/")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(template.replace("__PACKET__", encoded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--adjudication-page", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--adjudication-confirmation", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Replace this derived analysis only")
    parser.add_argument(
        "--provenance-confirmation",
        type=Path,
        help="Explicit coordinator confirmation of human-independent origin despite legacy metadata",
    )
    args = parser.parse_args()
    packet = read_packet(args.packet)
    private = json.loads(args.key.read_text(encoding="utf-8"))
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.review]
    result = aggregate(
        packet,
        private,
        reports,
        args.simulated,
        json.loads(args.provenance_confirmation.read_text(encoding="utf-8"))
        if args.provenance_confirmation
        else None,
    )
    if args.adjudication:
        result = finish_adjudication(
            packet,
            private,
            reports,
            result,
            json.loads(args.adjudication.read_text(encoding="utf-8")),
            json.loads(args.adjudication_confirmation.read_text(encoding="utf-8"))
            if args.adjudication_confirmation
            else None,
        )
    with args.output.open("w" if args.refresh else "x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.adjudication_page and not args.simulated:
        prepare_adjudication(packet, result, args.adjudication_page)
    print(
        f"{result['reviewer_count']} reviewers; {result['rating_records']} records; {result['status']}"
    )


if __name__ == "__main__":
    main()
