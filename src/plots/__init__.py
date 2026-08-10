"""Figure generation for Phase II and Phase III, plus the shared house style.

Every figure is regenerated from the run logs and results files in
checkpoints/ and docs/, never from numbers pasted into the plotting code -- a
defect this project shipped once (PITFALLS F2).

    python -m src.plots            # all
    python -m src.plots sft grpo   # a subset
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
DOCS = ROOT / "docs"

# palette
COLOR_REWARD = "#008080"    # deep teal   - primary series
COLOR_CONCEPT = "#2980B9"   # royal blue  - secondary series
COLOR_FORMAT = "#8E44AD"    # deep purple - format / think-rate
COLOR_CLEAN = "#2ECC71"     # emerald     - termination / stop rate
COLOR_KL = "#E67E22"        # amber       - KL / drift
COLOR_LOSS = "#E74C3C"      # coral       - loss / entropy
COLOR_SFT = "#7F8C8D"       # grey        - baseline model
COLOR_TRAP = "#C0392B"      # red         - what the broken measurement said
COLOR_FIX = "#148F77"       # green       - what the corrected measurement says
COLOR_MUTED = "#95A5A6"
INK = "#2C3E50"


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#BDC3C7",
        "axes.linewidth": 1.2,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FAFAFA",
    })


def rolling_mean(values, window: int = 15):
    """Trailing moving average with an EXPANDING window over the first points.

    Back-filling the leading window-1 samples with a constant (the original
    behaviour) drew a flat line across the opening of every panel, which reads
    as "nothing happened for 20 steps" when those steps simply were not
    plotted. An expanding mean shows the real early trajectory.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values
    w = min(window, n)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty(n, dtype=float)
    head = min(w - 1, n)
    idx = np.arange(head)
    out[:head] = cumsum[1:head + 1] / (idx + 1)
    if n >= w:
        out[w - 1:] = (cumsum[w:] - cumsum[:-w]) / float(w)
    return out


def nice_ylim(*arrays, pad: float = 0.12, floor=None, ceil=None):
    """Data-driven y-limits.

    Hardcoded limits tuned to one run silently clip the next one; every axis
    that can carry unknown data should call this instead.
    """
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-6)
    lo, hi = lo - pad * span, hi + pad * span
    if floor is not None:
        lo = min(lo, floor)
    if ceil is not None:
        hi = max(hi, ceil)
    return (lo, hi)


def panel(ax, title: str, tag: str = "") -> None:
    """Bold panel title plus a small provenance tag ('[log: grpo_v3]')."""
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=14 if tag else 8, color=INK)
    if tag:
        ax.text(0.5, 1.015, tag, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, color="#7F8C8D", style="italic")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend(ax, **kw):
    kw.setdefault("frameon", True)
    kw.setdefault("facecolor", "white")
    kw.setdefault("edgecolor", "#BDC3C7")
    kw.setdefault("fontsize", 9)
    return ax.legend(**kw)


def save(fig, name: str) -> None:
    """Write assets/<name>.png and .pdf, and mirror the png into docs/."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    png = ASSETS / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    import shutil
    shutil.copy(png, DOCS / f"{name}.png")
    print(f"  saved assets/{name}.png + .pdf")
