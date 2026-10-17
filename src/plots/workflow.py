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
    ax.text(x + w / 2, y + h - 0.26, stage_num, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=INK, alpha=0.75, zorder=3)
    ax.text(x + w / 2, y + h - 0.62, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + 1.02, main_takeaway, ha="center", va="center",
            fontsize=9.5, color=INK, alpha=0.85, zorder=3)
    ax.text(x + w / 2, y + 0.58, metric_top, ha="center", va="center",
            fontsize=9.5, color=INK, alpha=0.8, zorder=4)
    ax.text(x + w / 2, y + 0.30, metric_bottom, ha="center", va="center",
            fontsize=11, fontweight="bold", color=border_color, zorder=4)


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
    fig, ax = plt.subplots(figsize=(15, 3.2), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0.95, 3.45)
    ax.axis("off")
    y, h, w = 1.05, 2.25, 2.55
    gap = 0.44
    start_x = 0.25
    cards = [
        ("Phase I", "Pretrained base", "continues documents", "", "", "#F4F6F6", COLOR_MUTED),
        ("Phase II", "SFT", "answers questions", f"format {sft['think']:.0%}", f"accuracy {sft['acc']:.1%}",
         "#EBF5FB", COLOR_CONCEPT),
        ("Phase III", "GRPO", "well-formed answers", f"format {v4['think']:.0%}", f"accuracy {v4['acc']:.1%}",
         "#FEF9E7", COLOR_REWARD),
        ("Phase IV", "Executed equations", "interpreter computes", f"format {nt2['think']:.0%}",
         f"accuracy {nt2['acc']:.1%}", "#F4ECF7", COLOR_FORMAT),
        ("Deployed", "+ vote@8", "majority of 8 answers", "", f"accuracy {nt2['vote8']:.1%}", "#E8F8F5", COLOR_CLEAN),
    ]
    for i, (stage, title, takeaway, top, bottom, fill, border) in enumerate(cards):
        x = start_x + i * (w + gap)
        draw_card(ax, x, y, w, h, stage, title, takeaway, top, bottom, fill, border)
        if i < len(cards) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    fig.tight_layout()
    save(fig, "workflow")
    fig_stage_outputs()


STAGES_JSON = DOCS / "stage_outputs.json"


def fig_stage_outputs():
    """One held-out question answered by each phase (README §1.3), drawn as cards: the question on
    top, then each phase's own output and a one-line verdict. Texts come from docs/stage_outputs.json."""
    import textwrap
    if not STAGES_JSON.exists():
        print(f"[SKIP] {STAGES_JSON.name} not found — stage_outputs not rendered.")
        return
    d = json.load(open(STAGES_JSON, encoding="utf-8"))
    fig = plt.figure(figsize=(14, 7.9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.9)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.25, 6.62), 13.5, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#F4F6F6", edgecolor=COLOR_MUTED, lw=1.2))
    ax.text(0.5, 7.55, "Question", fontsize=11, fontweight="bold", color=INK, va="top")
    ax.text(13.5, 7.55, f"expected: {d['target']}", fontsize=10, color="#566573", va="top", ha="right")
    ax.text(0.5, 7.22, "\n".join(textwrap.wrap(d["prompt"], 140)), fontsize=10, color=INK, va="top", linespacing=1.35)
    colors = [COLOR_MUTED, COLOR_CONCEPT, COLOR_REWARD, COLOR_FORMAT]
    fills = ["#F4F6F6", "#EBF5FB", "#FEF9E7", "#F4ECF7"]
    bw, bh = 6.68, 3.1
    boxes = [(0.25, 3.33), (7.07, 3.33), (0.25, 0.1), (7.07, 0.1)]
    for st, c, fc, (bx, by) in zip(d["stages"], colors, fills, boxes):
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=fc, edgecolor=c, lw=1.6))
        ax.text(bx + 0.22, by + bh - 0.16, f"{st['phase']}  ·  {st['model']}", fontsize=11, fontweight="bold",
                color=INK, va="top")
        lines = []
        for para in st["text"].split("\n"):
            lines += textwrap.wrap(para, 68) or [""]
        ax.text(bx + 0.22, by + bh - 0.55, "\n".join(lines[:12]), fontsize=9.3, family="monospace", color=INK,
                va="top", linespacing=1.25)
        mark = "✓" if st["ok"] else "✗"
        ax.text(bx + 0.22, by + 0.16, f"{mark}  {st['verdict']}", fontsize=10.5, fontweight="bold",
                color=COLOR_CLEAN if st["ok"] else "#C0392B", va="bottom")
    save(fig, "stage_outputs")


if __name__ == "__main__":
    main()
