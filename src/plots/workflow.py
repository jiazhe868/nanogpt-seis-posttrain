"""Pipeline summary diagram for the README banner.

The stages the deployed model went through, each card with the two numbers that changed at that
stage, read from the results files at plot time (never pasted in; PITFALLS F2):
  <think> compliance and acc@5% on the 153 real held-out numerical questions, 6 seeds x 4
  (docs/eval_numerical_real6_final.json), vote@8 for the deployed model.

    python -m src.plots workflow
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.plots import (
    COLOR_CLEAN, COLOR_CONCEPT, COLOR_FORMAT, COLOR_MUTED,
    COLOR_REWARD, DOCS, INK, save, set_publication_style)

ROOT = Path(__file__).resolve().parents[2]
REAL6 = DOCS / "eval_numerical_real6_final.json"


def draw_card(ax, x, y, w, h, stage_num, title, main_takeaway, metric_top, metric_bottom, fill_color, border_color):
    """A stage card: badge, title, what changed, and two metric lines."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fill_color, edgecolor=border_color, linewidth=1.5, zorder=2))
    ax.text(x + w / 2, y + h - 0.28, stage_num, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=INK, alpha=0.75, zorder=3)
    ax.text(x + w / 2, y + h - 0.78, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK, zorder=3, linespacing=1.15)
    ax.text(x + w / 2, y + 1.02, main_takeaway, ha="center", va="center",
            fontsize=8.8, color=INK, alpha=0.90, zorder=3)
    ax.add_patch(FancyBboxPatch(
        (x + 0.12, y + 0.12), w - 0.24, 0.50, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="white", edgecolor=border_color, linewidth=1.0, alpha=0.95, zorder=3))
    ax.text(x + w / 2, y + 0.46, metric_top, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, y + 0.25, metric_bottom, ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=border_color, zorder=4)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
        linewidth=2.0, color=INK, shrinkA=3, shrinkB=3, zorder=1))


def stage_numbers():
    """acc@1, <think> rate and vote@8 per model from the 6-seed real held-out evaluation."""
    sys.path.insert(0, str(ROOT))
    from src.analysis.tier_vote import pooled
    from src.rl.verifiers import NumericalVerifier
    res = json.load(open(REAL6, encoding="utf-8"))["results"]
    tail = NumericalVerifier.ANSWER_TAIL_NUMBERS
    out = {}
    for name, runs in res.items():
        acc1, vote = pooled(runs, 8, tail)
        out[name] = {"acc": st.mean(r["numerical_acc_5pct"] for r in runs),
                     "think": st.mean(r["numerical_think_rate"] for r in runs),
                     "vote8": st.mean(v for v in vote.values() if v == v)}
    return out


def main() -> None:
    set_publication_style()
    if not REAL6.exists():
        print(f"[SKIP] {REAL6.name} not found — workflow not rendered.")
        return
    n = stage_numbers()
    sft, v4, nt2 = n["SFT"], n["v4@300"], n["notation2 (recovered data)"]
    fig, ax = plt.subplots(figsize=(15, 5.4), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.1, 4.9)
    ax.axis("off")
    ax.text(7.5, 4.55, "Post-training pipeline: what each stage changed", ha="center", va="center",
            fontsize=15.0, fontweight="bold", color=INK)
    ax.text(7.5, 4.15, "held-out acc = acc@5% on the 153 real held-out numerical questions, 6 seeds × 4 samples",
            ha="center", va="center", fontsize=9.0, color=INK, alpha=0.75, style="italic")

    y, h, w = 1.30, 2.60, 2.55
    gap = 0.44
    start_x = 0.25
    acc = "held-out acc"
    cards = [
        ("Phase I", "Pretrained base", "continues documents;\nnot yet an assistant",
         "<think>: —", f"{acc}: —", "#F2F4F4", COLOR_MUTED),
        ("Phase II", "SFT", "answers as an assistant,\nreasoning in <think>",
         f"<think>: {sft['think']:.0%}", f"{acc}: {sft['acc']:.1%}", "#EBF5FB", COLOR_CONCEPT),
        ("Phase III", "GRPO\n(composite reward)", "format and termination\nlocked in",
         f"<think>: {v4['think']:.0%}", f"{acc}: {v4['acc']:.1%}", "#FEF9E7", COLOR_REWARD),
        ("Phase IV", "Executed-equation\nSFT", "the model sets up equations,\nan interpreter computes",
         f"<think>: {nt2['think']:.0%}", f"{acc}: {nt2['acc']:.1%}", "#F4ECF7", COLOR_FORMAT),
        ("Deployed", "+ vote@8", "returns the answer most\nof 8 samples agree on",
         "configs/deploy.yaml", f"{acc}: {nt2['vote8']:.1%}", "#E8F8F5", COLOR_CLEAN),
    ]
    for i, (stage, title, takeaway, top, bottom, fill, border) in enumerate(cards):
        x = start_x + i * (w + gap)
        draw_card(ax, x, y, w, h, stage, title, takeaway, top, bottom, fill, border)
        if i < len(cards) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)

    ax.text(7.5, 0.45,
            "Phase IV data: 105k teacher-written, execution-checked solutions in paper notation (local Qwen3.8-27B).\n"
            "RL on executed answers lifts the problems it practises on (+37 pp) and their close variants (§5.18).",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.03", fc="#EAF2F8", ec=COLOR_CONCEPT, lw=1.2))
    fig.tight_layout()
    save(fig, "workflow")


if __name__ == "__main__":
    main()
