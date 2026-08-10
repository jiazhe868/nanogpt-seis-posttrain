"""Pipeline summary diagram for the README banner.

Highlights the 4-stage progression (Base -> SFT -> RLVR -> RAG Grounding)
and the first-order empirical takeaway.

    python -m src.plots workflow
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.plots import (
    COLOR_CLEAN, COLOR_CONCEPT, COLOR_FORMAT, COLOR_MUTED,
    COLOR_REWARD, COLOR_TRAP, INK, save, set_publication_style)


def draw_card(ax, x, y, w, h, stage_num, title, main_takeaway, format_acc, math_acc, fill_color, border_color):
    """Draw a clean, elegant summary stage card with dual metrics."""
    # Outer Card Box
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fill_color, edgecolor=border_color, linewidth=1.5, zorder=2))
    
    # Stage Badge
    ax.text(x + w / 2, y + h - 0.28, stage_num, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=INK, alpha=0.75, zorder=3)
    
    # Card Title
    ax.text(x + w / 2, y + h - 0.60, title, ha="center", va="center",
            fontsize=12.0, fontweight="bold", color=INK, zorder=3)
    
    # Main Behavior Takeaway
    ax.text(x + w / 2, y + 0.92, main_takeaway, ha="center", va="center",
            fontsize=9.2, color=INK, alpha=0.90, zorder=3)
    
    # Key Metrics Pill Box
    ax.add_patch(FancyBboxPatch(
        (x + 0.15, y + 0.12), w - 0.30, 0.48, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="white", edgecolor=border_color, linewidth=1.0, alpha=0.95, zorder=3))
    
    # Dual Metrics inside Pill Box
    ax.text(x + w / 2, y + 0.43, f"Format Acc:  {format_acc}", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, y + 0.23, f"Math Acc@5%:  {math_acc}", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=border_color, zorder=4)


def arrow(ax, x0, y0, x1, y1):
    """Draw connecting flow arrow."""
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
        linewidth=2.0, color=INK, shrinkA=3, shrinkB=3, zorder=1))


def main() -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(14, 5.3), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.1, 4.8)
    ax.axis("off")

    # Header Title
    ax.text(7.0, 4.40, "LLM Post-Training Pipeline & Empirical Evolution",
            ha="center", va="center", fontsize=15.0, fontweight="bold", color=INK)

    # 4 Main Stages Layout
    y, h, w = 1.35, 2.50, 2.85
    gap = 0.52
    start_x = 0.30

    cards = [
        ("Phase I", "Pretrained Base", "Raw Text Continuation\n(Non-Instructional)", "0.0%", "0.0%", "#F2F4F4", COLOR_MUTED),
        ("Phase II", "Supervised SFT", "Learned ChatML Structure\n& Response Format", "61.4%", "3.6%", "#EBF5FB", COLOR_CONCEPT),
        ("Phase III", "Verifier RL", "Enforced Strict Format\n& Response Termination", "97.8%", "11.4%", "#FEF9E7", COLOR_REWARD),
        ("Phase IV", "RAG Grounding", "Open-Book Context\n& Calculation Precision", "98.2%", "44.4%", "#E8F8F5", COLOR_CLEAN),
    ]

    for i, (stage, title, takeaway, fmt_acc, math_acc, fill, border) in enumerate(cards):
        x = start_x + i * (w + gap)
        draw_card(ax, x, y, w, h, stage, title, takeaway, fmt_acc, math_acc, fill, border)
        
        # Connect with arrows between cards
        if i < len(cards) - 1:
            arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)

    # First-Order Takeaway Banner at Bottom
    ax.text(7.0, 0.40, "Key Takeaway: Verifier RL guarantees format compliance (97.8%), while RAG grounding enables open-book arithmetic calculation (44.4% Acc)",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.03", fc="#FDEDEC",
                      ec=COLOR_TRAP, lw=1.3))

    fig.tight_layout()
    save(fig, "workflow")


if __name__ == "__main__":
    main()
