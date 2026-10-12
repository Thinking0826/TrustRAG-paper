"""Render Fig. 3 directly from the completed study summaries.

The editable SVG and PDF use the existing blue, violet, amber and green palette.
No inference or statistical estimation is performed by this figure script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#387FC8"
PURPLE = "#8951BA"
AMBER = "#EEAA43"
GREEN = "#299C79"
NAVY = "#234D75"
INK = "#20354D"
MUTED = "#475F77"
PALE_BLUE = "#CBE3F9"
PALE_GREEN = "#E5F3ED"


def load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def label(ax, x, y, value, *, size=6.5, color=INK, bold=False, ha="left"):
    return ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        fontsize=size,
        color=color,
        fontweight="bold" if bold else "normal",
        ha=ha,
        va="center",
        clip_on=False,
    )


def heading(ax, letter, title, subtitle, color):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    label(ax, 0.015, 0.965, letter, size=8, bold=True)
    label(ax, 0.075, 0.965, title, size=7.4, bold=True)
    ax.plot([0.075, 0.965], [0.906, 0.906], lw=2.1, color=color, transform=ax.transAxes)
    label(ax, 0.075, 0.843, subtitle, size=6, color=MUTED)


def ribbon(ax, y, text, subtext, color, *, height=0.15):
    ax.add_patch(
        FancyBboxPatch(
            (0.075, y),
            0.89,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            linewidth=0,
            facecolor=color,
            transform=ax.transAxes,
        )
    )
    label(ax, 0.10, y + height * 0.66, text, size=6.35, color="white", bold=True)
    label(ax, 0.10, y + height * 0.25, subtext, size=5.9, color="white")


def clean_axis(ax, ticks, xlabel):
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#A7BBCB")
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="both", length=2, colors=MUTED, labelsize=5.8)
    ax.set_xticks(ticks)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=6, color=MUTED, labelpad=3)


def interval(ax, y, item, color, *, marker="o"):
    center = 100 * item["difference"]
    low, high = [100 * value for value in item["ci95"]]
    ax.errorbar(
        center,
        y,
        xerr=[[center - low], [high - center]],
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markersize=4.6,
        capsize=3,
        elinewidth=1.6,
        capthick=1.1,
        zorder=3,
    )
    return center


def render(root: Path, destination: Path, qa_tools: Path | None = None) -> None:
    main = load(root, "data/research/v2/qasper/test_v1/results/comparison.json")
    oracle = load(root, "data/research/v2/qasper/oracle_v1/results/comparison.json")
    order = load(root, "data/research/v2/longbench/multifieldqa_en/results/comparison.json")
    planner = load(root, "data/research/v2/qasper/development_v2/results/comparison.json")
    human = load(root, "data/research/v2/qasper/test_v1/human_review_v2/analysis.json")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )
    fig, panels = plt.subplots(
        2,
        2,
        figsize=(183 / 25.4, 126 / 25.4),
        gridspec_kw={"left": 0.025, "right": 0.975, "top": 0.977, "bottom": 0.12,
                     "wspace": 0.13, "hspace": 0.12},
    )
    a, b, c, d = panels.flat
    heading(a, "a", "Realized input change", "Main QASPER: density versus closure, n = 300", BLUE)
    heading(b, "b", "Preservation versus answer gain", "Main QASPER: closure minus density, 7B BF16", PURPLE)
    heading(c, "c", "Human-evidence selection headroom", "QASPER evidence oracle at 2K; post hoc diagnostic", GREEN)
    heading(d, "d", "Matched order-only validation", "LongBench: identical selected chunks, two orders", AMBER)

    # a: Counts audit the actual messages, rather than the nominal algorithm.
    bars = a.inset_axes([0.12, 0.45, 0.84, 0.31])
    activation = main["input_activation"]["budgets"]
    for y, budget in zip((2, 1, 0), (1024, 2048, 4096), strict=True):
        item = activation[str(budget)]
        changed, identical = item["changed_messages"], item["identical_messages"]
        bars.barh(y, changed, height=0.44, color=BLUE)
        bars.barh(y, identical, left=changed, height=0.44, color=PALE_BLUE)
        bars.text(-12, y, f"{budget // 1024}K", ha="right", va="center", fontsize=6.2, color=INK)
        bars.text(314, y, f"{changed} / {identical}", va="center", fontsize=6.2, color=INK)
    bars.set(xlim=(0, 420), ylim=(-0.65, 2.55))
    clean_axis(bars, [0, 150, 300], "Matched complete messages")
    label(a, 0.12, 0.775, "Changed", size=6, color=BLUE, bold=True)
    label(a, 0.405, 0.775, "Identical", size=6, color=MUTED)
    mechanism = planner["mechanism_diagnostics"]
    ribbon(a, 0.13, "Planner-flat = query-closure", "50/50 identical pairs at both 1K and 2K", PURPLE)
    maximum = mechanism["budgets"]["1024"]["valid_plan_max_input_tokens"]
    label(a, 0.075, 0.055,
          f'{mechanism["valid_plans"]} valid plans; {mechanism["plans_with_dependencies"]} with edges; '
          f'{maximum} tokens max', size=5.8, color=MUTED)

    # b: Paragraph preservation and answer F1 have distinct scales and roles.
    label(b, 0.075, 0.765, "Full evidence recall", size=6.1, color=MUTED, bold=True)
    label(b, 0.59, 0.765, "Answer F1 difference", size=6.1, color=MUTED, bold=True)
    f1 = b.inset_axes([0.59, 0.355, 0.29, 0.34])
    f1.axvline(0, color="#92A9BB", lw=0.8, linestyle=(0, (2, 2)), zorder=0)
    for y, budget, pos in zip((2, 1, 0), (1024, 2048, 4096), (0.64, 0.525, 0.41), strict=True):
        left = main["budget_selection"][f"density_atoms@{budget}"]["mean_full_evidence_paragraph_recall"]
        right = main["budget_selection"][f"closure_greedy@{budget}"]["mean_full_evidence_paragraph_recall"]
        label(b, 0.075, pos, f"{budget // 1024}K   {left:.3f}  →  {right:.3f}", size=6.3, color=NAVY)
        item = main["paired_analysis"]["models"]["qwen7b"]["budgets"][str(budget)]["answer_token_f1"]
        delta = interval(f1, y, item, PURPLE)
        label(b, 0.975, pos, f"{delta:+.2f}", size=6, ha="right", color=PURPLE)
    f1.set(xlim=(-1.65, 2.35), ylim=(-0.5, 2.5))
    clean_axis(f1, [-1, 0, 1], "F1 points")
    h = human["adjudicated_results"]["method_contrasts"]["qwen7b"]
    low, high = [100 * value for value in h["ci95"]]
    ribbon(b, 0.055, f'Human joint success: {h["closure_success_count"]}/60 versus {h["density_success_count"]}/60',
           f'Difference {100*h["difference"]:+.2f} pp [{low:.2f}, {high:.2f}]; n = 60', NAVY)

    # c: The effect is paired oracle minus the unchanged closure condition.
    c.add_patch(FancyBboxPatch((0.06, 0.035), 0.91, 0.745,
                              boxstyle="round,pad=0.008,rounding_size=0.045",
                              facecolor=PALE_GREEN, edgecolor="none", transform=c.transAxes, zorder=0))
    headroom = c.inset_axes([0.18, 0.295, 0.73, 0.41])
    headroom.set_facecolor("none")
    for y, model, display in ((1, "qwen7b", "7B BF16"), (0, "qwen14b_nf4", "14B NF4")):
        item = oracle["oracle_analysis"]["models"][model]["oracle_minus_retrieved"]["closure_greedy"]["answer_token_f1"]
        delta = interval(headroom, y, item, GREEN, marker="D")
        headroom.text(23.5, y, f"{delta:+.2f}", color=GREEN, fontsize=6.5,
                      fontweight="bold", ha="right", va="center")
        label(c, 0.10, 0.737 if y else 0.52,
              f'{display}: {100*item["right_mean"]:.2f} → {100*item["left_mean"]:.2f}   '
              f'(n = {item["question_count"]})', size=6.4, bold=True, color="#197357")
    headroom.set(xlim=(0, 25), ylim=(-0.5, 1.4))
    clean_axis(headroom, [0, 10, 20], "Oracle minus closure (F1 points)")
    label(c, 0.10, 0.075, "Whole annotated paragraphs; all inputs fit 2K", size=5.9, color="#197357")

    # d: All four deployment-budget contrasts retain their paired uncertainty.
    ordering = d.inset_axes([0.275, 0.37, 0.575, 0.38])
    ordering.axvline(0, color="#92A9BB", lw=0.8, linestyle=(0, (2, 2)), zorder=0)
    contrasts = (("qwen7b", "1024", "7B 1K", BLUE),
                 ("qwen7b", "2048", "7B 2K", BLUE),
                 ("qwen14b_nf4", "1024", "14B 1K", PURPLE),
                 ("qwen14b_nf4", "2048", "14B 2K", PURPLE))
    for y, (model, budget, name, color) in zip((3, 2, 1, 0), contrasts, strict=True):
        item = order["paired_analysis"]["models"][model][budget]
        delta = interval(ordering, y, item, color)
        ordering.text(-9.0, y, name, fontsize=6.2, ha="right", va="center", color=INK)
        ordering.text(10.25, y, f"{delta:+.2f}", fontsize=6, ha="right", va="center", color=color)
    ordering.set(xlim=(-8, 10.5), ylim=(-0.6, 3.65))
    clean_axis(ordering, [-8, 0, 8], "Document minus relevance order (F1 points)")
    label(d, 0.075, 0.795, "7B: n = 150; 14B NF4: n = 50", size=5.9, color=MUTED)
    ribbon(d, 0.055, "No consistent order gain", "All four paired 95% intervals include zero", "#A56A24")

    fig.text(0.043, 0.057, "Different interventions: preserve structure • replace evidence • reorder an identical set",
             fontsize=6.6, fontweight="bold", color=NAVY)
    fig.text(0.043, 0.026, "Points: paired mean differences; whiskers: 95% document-cluster bootstrap intervals (5,000 draws, seed 42).",
             fontsize=5.8, color=MUTED)
    fig.canvas.draw()
    if qa_tools is not None:
        import sys
        sys.path.insert(0, str(qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(qa_tools / "figure03.alignment.json"),
            overlay_svg=str(qa_tools / "figure03.alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(destination.with_suffix(f".{extension}"), dpi=300)
    svg = destination.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--qa-tools", type=Path)
    args = parser.parse_args()
    render(args.root, args.root / "research/v2/figures/figure_03_input_activation", args.qa_tools)
