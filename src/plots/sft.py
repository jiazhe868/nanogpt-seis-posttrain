"""Phase II figures.

    assets/sft_training_dynamics  train / val loss vs step, with the LR panel
    assets/sft_data_scaling       held-out test loss vs corpus size, plus the
                                  per-source breakdown the aggregate hides

Both are measured on the hash-pinned held-out TEST split (src.sft.build_corpus
split --test-frac), never on a per-run validation set: each corpus rebuild used
to redraw its own val split, which made cross-version losses incomparable.

    python -m src.plots sft
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from src.plots import ASSETS, DOCS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

BLUE   = "#2a78d6"
RED    = "#e34948"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
INK    = "#0b0b0b"
INK2   = "#52514e"
GRID   = "#e6e6e3"
AXIS   = "#D3D3D3"


def apply_pretrain_style():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save_figure(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}")


# ────────────────────────────────────────────────────────────────────────
# Figure 1: SFT training dynamics (2-panel: loss + LR)
# ────────────────────────────────────────────────────────────────────────

def fig_training_dynamics():
    log_csv = ROOT / "checkpoints" / "runs" / "sft_lr2e-4_14k.csv"
    steps, train_loss, val_loss, lrs = [], [], [], []

    if log_csv.exists():
        with open(log_csv) as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                steps.append(step)
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                lrs.append(float(row["lr"]))

    if not steps:
        print("[SKIP] sft_log.csv not found — skipping training dynamics figure")
        return

    best_k = min(range(len(val_loss)), key=lambda j: val_loss[j])
    best_step, best_val = steps[best_k], val_loss[best_k]

    apply_pretrain_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    # ── panel 1: loss ──
    ax1.plot(steps, train_loss, color=BLUE, lw=2, marker="o", ms=4, label="train")
    ax1.plot(steps, val_loss,   color=RED,  lw=2, marker="o", ms=4, label="val")
    ax1.scatter([best_step], [best_val], s=70, facecolor="none",
                edgecolor=RED, lw=2, zorder=5)
    # Place the callout away from the curves. The minimum now falls early
    # (epoch 2 on a document-disjoint split), so the old fixed left-offset
    # pushed the label on top of both series.
    early = best_step < 0.4 * steps[-1]
    ax1.annotate(
        f"best val {best_val:.4f}\n(ppl {math.exp(best_val):.1f}) @ step {best_step}",
        xy=(best_step, best_val),
        xytext=(best_step + (0.18 if early else -0.22) * steps[-1],
                best_val - 0.45),
        ha="left" if early else "right", fontsize=9, color=INK2,
        arrowprops=dict(arrowstyle="-", color=INK2, lw=1),
    )
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_ylim(bottom=0.6)
    # The gap after the minimum is real in the likelihood sense (held-out NLL
    # rises 0.51 nats from step 80 to 336, t=+33) but NOT in the behavioural
    # sense: the step-336 checkpoint generates equally accurate answers and a
    # significantly higher rate of well-formed <think> blocks. Cross-entropy
    # penalises drifting away from one reference wording, which is not the same
    # as getting worse. Labelled accordingly rather than just "overfitting".
    ax1.axvspan(best_step, steps[-1], color=RED, alpha=0.05, zorder=0)
    ax1.text(best_step + (steps[-1] - best_step) / 2, ax1.get_ylim()[1] * 0.985,
             "held-out likelihood degrades here\n"
             "(but generation quality does not — see PITFALLS E1)",
             ha="center", va="top", fontsize=8.5, color=INK2, alpha=0.9)
    ax1.set_title("nanoGPT-Seis-Posttrain — SFT training dynamics (113M, 2×A30, 14.1k pairs)",
                   loc="left", fontsize=13)
    ax1.grid(True, color=GRID, lw=1)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="upper right")
    # direct-label series end
    ax1.text(steps[-1], train_loss[-1] - 0.06, "train",
             color=BLUE, fontsize=9, va="top", ha="right")

    # ── panel 2: learning rate ──
    ax2.plot(steps, lrs, color=VIOLET, lw=2)
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("SFT optimisation step")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, color=GRID, lw=1)
    ax2.set_axisbelow(True)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    save_figure(fig, str(ASSETS / "sft_training_dynamics"))
    plt.close()
    print("saved assets/sft_training_dynamics.{png,pdf}")


# ────────────────────────────────────────────────────────────────────────
# Figure 2: SFT data scaling law — val loss vs clean dataset size
# ────────────────────────────────────────────────────────────────────────

def fig_data_scaling():
    """Two panels.

    (a) Held-out test loss vs corpus size, with the log-linear fit through the
        three points whose training mixture was comparable. The 14.1k point is
        plotted but excluded from the fit because its mixture changed (it adds
        a historical-earthquake corpus), and it is scored here on the
        paper-only test set that predates that corpus.

    (b) The per-source breakdown, which is the part the aggregate hides: the
        added data is transformative on its own domain and roughly neutral on
        the original one.
    """
    # Measured on data/eval/heldout_subsets.jsonl (281 paper-derived pairs held out
    # from every model below), scored UNPACKED: each pair in isolation.
    #
    # Packed scoring concatenates pairs into 4096-token blocks under a plain
    # causal mask, so a pair's loss depends on whatever precedes it in its
    # block, and long pairs dominate the token-weighted mean. At the ~0.01-nat
    # effect sizes here that artefact flipped the sign: packed scoring made the
    # 14.1k model look worse than 8.5k on paper (2.1390 vs 2.1312) while
    # unpacked paired scoring shows it consistently better (-0.010 nats,
    # 95% CI [-0.015, -0.005]) on the same file.
    # COMPOSITION-MATCHED series: every point is a stratified chunk-level
    # subsample of the same 14.1k corpus, holding the mixture at 46% paper /
    # 25% fundamentals / 28% earthquakes. An earlier version varied size and
    # mixture together (4.2k was 100% paper, 8.5k was 76/24), which measured
    # the two effects superimposed and understated the slope by ~28%.
    fit_x = np.array([2060, 4069, 8034, 14068], dtype=float)
    fit_y = np.array([2.5005, 2.4347, 2.3654, 2.2700])
    off_x = off_y = None

    lx = np.log2(fit_x)
    b, a = np.polyfit(lx, fit_y, 1)          # y = a + b*log2(x), b negative

    apply_pretrain_style()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})

    # ── panel (a): scaling ──
    grid = np.logspace(np.log10(1800), np.log10(24000), 200)
    ax.plot(grid, a + b * np.log2(grid), color=INK2, lw=1.2, ls="--", zorder=2,
            label=f"log-linear fit  ({b:+.3f} nats / doubling)")
    ax.plot(fit_x, fit_y, color=BLUE, lw=2, marker="o", ms=7, zorder=3,
            label="fixed mixture (46% paper / 25% fund. / 28% eq.)")

    for k, (x, y) in enumerate(zip(fit_x, fit_y)):
        # last point sits below the fit line, so label it underneath
        dy = -16 if k == len(fit_x) - 1 else 10
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, dy),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("training pairs")
    ax.set_ylabel("held-out test loss")
    ax.set_title("(a) SFT data scaling — fixed mixture, unpacked scoring",
                 loc="left", fontsize=12.5)
    ax.set_xticks([2060, 4069, 8034, 14068])
    ax.set_xticklabels(["2.1k", "4.1k", "8.0k", "14.1k"])
    ax.set_ylim(2.20, 2.58)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    # ── panel (b): the same scaling step, decomposed by domain ──
    # An earlier version of this panel compared the 8.5k model (76% paper, no
    # earthquake data at all) against 14.1k, which conflated mixture with size
    # and made the earthquake bar near-tautological. Both models here are the
    # SAME fixed 46/25/28 mixture, so this decomposes the panel-(a) scaling
    # step by domain rather than measuring the effect of adding a corpus.
    labels = ["paper\n(205 ids)", "earthquakes\n(6 unseen events)",
              "fundamentals\n(4 unseen topics)"]
    before = [2.4105, 1.8951, 2.5941]      # 8.0k, fixed mixture
    after  = [2.3136, 1.7413, 2.4689]      # 14.1k, fixed mixture
    diffs  = [-0.0969, -0.1538, -0.1252]
    cis    = [(-0.1033, -0.0908), (-0.1728, -0.1353), (-0.1371, -0.1133)]
    idx = np.arange(len(labels))
    w = 0.36

    ax2.bar(idx - w / 2, before, w, color=AXIS, edgecolor=INK2, lw=0.8,
            label="8.0k corpus")
    ax2.bar(idx + w / 2, after, w, color=BLUE, edgecolor=INK2, lw=0.8,
            label="14.1k corpus")
    for i, (lo, hi) in enumerate(zip(before, after)):
        for off, v in ((-w / 2, lo), (w / 2, hi)):
            ax2.text(i + off, v + 0.015, f"{v:.3f}", ha="center",
                     fontsize=8.5, color=INK2)
    # Deltas go ABOVE the taller bar: placing them at the base ran them behind
    # the bars themselves.
    for i, (d, (clo, chi)) in enumerate(zip(diffs, cis)):
        ax2.annotate(f"{d:+.3f}  [{clo:.3f}, {chi:.3f}]",
                     xy=(i, max(before[i], after[i]) + 0.10), ha="center",
                     va="bottom", fontsize=8.5, color=BLUE, fontweight="bold")

    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("held-out test loss")
    ax2.set_ylim(1.35, 3.00)
    ax2.set_title("(b) The same scaling step, by domain  (paired, 95% CI)",
                  loc="left", fontsize=12.5)
    ax2.grid(True, axis="y", color=GRID, lw=1)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    save_figure(fig, str(ASSETS / "sft_data_scaling"))
    plt.close()
    print("saved assets/sft_data_scaling.{png,pdf}")


# ────────────────────────────────────────────────────────────────────────

def main():
    ASSETS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    fig_training_dynamics()
    fig_data_scaling()


if __name__ == "__main__":
    main()
