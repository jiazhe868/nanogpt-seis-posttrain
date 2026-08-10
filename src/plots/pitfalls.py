"""Figures illustrating the methodological pitfalls hit in this project.

Each figure contrasts THE TRAP (what the broken measurement said) with THE FIX
(what the corrected measurement says). Together with PITFALLS.md they are the
retrospective record of how the headline numbers in this repo went from
untrustworthy to defensible.

Provenance rules followed here:
  * Anything that can be recomputed from a surviving log or results file IS
    recomputed at plot time -- see load_grpo_log() and the docs/*.json
    readers. Those panels carry a "[log]" tag in their subtitle.
  * Historical measurements whose run artifacts are not reloadable (v1-era
    evaluations, decoder sweeps, unpacked paired scores) are declared as module
    constants. Those panels carry a "[historical measurement]" tag.

That distinction matters: a previous version of the GRPO benchmark figure
hardcoded numbers with no provenance and presented them as a fresh measurement,
which is one of the pitfalls documented here (see B1). Constants are
fine; unlabelled constants are not.

Usage:
  python -m src.plots pitfalls
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plots import (  # noqa: E402
    ASSETS, COLOR_CLEAN, COLOR_CONCEPT, COLOR_FIX, COLOR_FORMAT, COLOR_KL,
    COLOR_LOSS, COLOR_MUTED, COLOR_REWARD, COLOR_SFT, COLOR_TRAP, DOCS,
    legend, nice_ylim, panel, rolling_mean, save, set_publication_style)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))




# ---------------------------------------------------------------------------
# Historical constants: measurements whose run artifacts are no longer reloadable.
# ---------------------------------------------------------------------------

# pair-level split left every val pair with train siblings.
LEAK_PAIR_LEVEL_FRAC = 1.00      # 864/864 val pairs shared a source chunk
LEAK_MEAN_SIBLINGS = 1.88
LEAK_CHUNK_LEVEL_FRAC = 0.00

# my own generator reintroduced the same leak via md5(topic::variant).
GEN_LEAK_BEFORE = 1.00           # 100% of test identities also in train
GEN_LEAK_AFTER = 0.00
EQ_GAIN_APPARENT = -0.784        # nats, pre-fix
EQ_GAIN_TRUE = -0.476            # nats, post-fix

# decoder sweep: fraction of repeated 5-grams, 4 prompts x 250 tokens.
DECODE_TEMPS = [0.2, 0.5, 0.7, 0.9]
DECODE_RPS = [1.05, 1.15, 1.25]
DECODE_REPEAT = np.array([
    [0.562, 0.162, 0.000],
    [0.105, 0.005, 0.000],
    [0.039, 0.002, 0.000],
    [0.007, 0.003, 0.000],
])
# the sign flip. num acc @1%, SFT vs GRPO, at two repetition penalties.
DECODE_SFT_ACC_RP105, DECODE_GRPO_ACC_RP105 = 0.077, 0.085   # GRPO "ahead"
DECODE_SFT_ACC_RP120, DECODE_GRPO_ACC_RP120 = 0.154, 0.120   # GRPO behind

# packed vs unpacked, two nested paper-only test files (14.1k - 8.5k).
PACKED_DELTAS = {"paper_common\n(281 pairs)": +0.0078, "paper\n(356 pairs, superset)": -0.0105}
UNPACKED_DELTAS = {"paper_common\n(281 pairs)": -0.0100, "paper\n(356 pairs, superset)": -0.0092}
UNPACKED_CIS = {"paper_common\n(281 pairs)": (-0.0154, -0.0045),
                "paper\n(356 pairs, superset)": (-0.0137, -0.0045)}
PACKED_ABS, UNPACKED_ABS = 2.1312, 2.2799   # same file, same model

# confounded and composition-matched scaling series.
CONFOUNDED_N = np.array([2100, 4216, 8507])
CONFOUNDED_LOSS = np.array([2.2590, 2.1869, 2.1312])
MATCHED_N = np.array([2060, 4069, 8034, 14068])
MATCHED_LOSS = np.array([2.5005, 2.4347, 2.3654, 2.2700])
# mixture drift of the confounded series (paper / fundamentals / eq).
MIXTURE = {"4.2k": (1.00, 0.00, 0.00), "8.5k": (0.76, 0.24, 0.00), "14.1k": (0.46, 0.25, 0.28)}

# per-domain paired lift, fixed 46/25/28 mixture, 8.0k -> 14.1k.
PER_DOMAIN = {
    "paper":        (-0.0969, (-0.1033, -0.0908), -30.1),
    "earthquakes":  (-0.1538, (-0.1728, -0.1353), -16.1),
    "fundamentals": (-0.1252, (-0.1371, -0.1133), -19.8),
}

# likelihood vs behaviour divergence, SFT best(step80) vs last(step336).
BEST_LAST_LOSS = (2.2700, 2.7798)          # paired diff +0.5098, t=+33.5
BEST_LAST_THINK = (0.614, 0.741)           # z=+2.47, significant

# share of the v1 "+590%" that was format compliance.
V1_FORMAT_SHARE = 0.73


# ---------------------------------------------------------------------------
# Loaders for surviving artifacts
# ---------------------------------------------------------------------------

def load_grpo_log(run: str) -> dict:
    """Load a GRPO run log from checkpoints/runs/<run>.csv."""
    path = ROOT / "checkpoints" / "runs" / f"{run}.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for k in rows[0]:
        out[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return out






def _bars(ax, labels, trap, fix, trap_lbl, fix_lbl, fmt="{:.1%}", ylabel=""):
    x = np.arange(len(labels))
    w = 0.36
    r1 = ax.bar(x - w / 2, trap, w, label=trap_lbl, color=COLOR_TRAP,
                edgecolor="#2C3E50", linewidth=1.1)
    r2 = ax.bar(x + w / 2, fix, w, label=fix_lbl, color=COLOR_FIX,
                edgecolor="#2C3E50", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    for rects in (r1, r2):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(fmt.format(h), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold", color="#2C3E50")
    return r1, r2


# ---------------------------------------------------------------------------
# A1 -- train/test leakage, twice
# ---------------------------------------------------------------------------

def fig_leakage():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    ax = axes[0]
    _bars(ax, ["val pairs sharing a\nsource chunk with train"],
          [LEAK_PAIR_LEVEL_FRAC], [LEAK_CHUNK_LEVEL_FRAC],
          "pair-level split (v1)", "chunk-level split (fix)", ylabel="fraction of val set")
    ax.set_ylim(0, 1.18)
    panel(ax, "(a) The original split leaked every val pair", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper right")
    ax.text(0.30, 0.52, f"mean {LEAK_MEAN_SIBLINGS:.2f} sibling\nQ&A pairs in train",
            ha="left", fontsize=9, color=COLOR_TRAP, fontweight="bold")

    ax = axes[1]
    _bars(ax, ["held-out identities\nalso seen in training"],
          [GEN_LEAK_BEFORE], [GEN_LEAK_AFTER],
          "md5(topic::variant)", "md5(identity) (fix)", ylabel="fraction of test identities")
    ax.set_ylim(0, 1.18)
    panel(ax, "(b) ...then I reintroduced it in my own generator", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper right")

    ax = axes[2]
    bars = ax.bar([0, 1], [abs(EQ_GAIN_APPARENT), abs(EQ_GAIN_TRUE)],
                  0.5, color=[COLOR_TRAP, COLOR_FIX], edgecolor="#2C3E50", linewidth=1.1)
    leaked = abs(EQ_GAIN_APPARENT) - abs(EQ_GAIN_TRUE)
    ax.bar([0], [leaked], 0.5, bottom=[abs(EQ_GAIN_TRUE)], color="none",
           edgecolor="white", linewidth=1.6, hatch="///",
           label=f"leakage artefact ({leaked / abs(EQ_GAIN_APPARENT):.0%} of the\napparent gain)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before leak fix", "after leak fix"], fontsize=9)
    ax.set_ylabel("earthquake-subset gain (nats, |Δ|)", fontsize=10)
    for b, v in zip(bars, [EQ_GAIN_APPARENT, EQ_GAIN_TRUE]):
        ax.annotate(f"{v:+.3f}", xy=(b.get_x() + b.get_width() / 2, abs(v)),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9.5, fontweight="bold", color="#2C3E50")
    ax.set_ylim(0, 0.95)
    panel(ax, "(c) The leak inflated a real effect by 39%", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper right")

    fig.suptitle("A1 — Train/test leakage: the same bug, made twice",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_1_data_leakage")


# ---------------------------------------------------------------------------
# B1 -- circular evaluation
# ---------------------------------------------------------------------------

def fig_circular():
    v3 = load_grpo_log("grpo")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    ax = axes[0]
    ax.plot(v3["step"], rolling_mean(v3["mean_reward"], 20),
            color=COLOR_TRAP, linewidth=2.4, label="composite training reward")
    ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.1)
    ax.set_xlabel("GRPO step", fontsize=10)
    ax.set_ylabel("reward (20-step rolling mean)", fontsize=10)
    panel(ax, "(a) The reward goes up — necessarily", "[log: grpo]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="lower right")
    ax.text(0.04, 0.94, "this IS the optimisation objective;\nreporting it as a result is circular",
            transform=ax.transAxes, va="top", fontsize=8.8, color=COLOR_TRAP,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#FDEDEC", ec=COLOR_TRAP, lw=1))

    ax = axes[1]
    res = json.load(open(DOCS / "eval_rl_results.json", encoding="utf-8"))
    sft, grpo = res["SFT"], res["GRPO @100 (selected)"]
    labels = ["num acc @5%", "<think> rate", "prompt-echo\n(lower better)"]
    keys = ["numerical_acc_5pct", "numerical_think_rate", "echo_rate"]
    _bars(ax, labels, [sft[k] for k in keys], [grpo[k] for k in keys],
          "SFT", "GRPO", ylabel="rate")
    ax.set_ylim(0, 1.22)
    for i, note in enumerate(["z=+0.16\nn.s.", "z=+6.55\nsignificant", "z=−2.94\nsignificant"]):
        col = COLOR_MUTED if "n.s." in note else COLOR_FIX
        ax.text(i, 1.10, note, ha="center", fontsize=8.5, fontweight="bold", color=col)
    panel(ax, "(b) Reward-independent metrics disagree", "[log: eval_rl_results.json]")
    # centre-left: the only region free of both the bars and the z-score labels
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5,
              loc="center left")

    ax = axes[2]
    ax.bar([0], [V1_FORMAT_SHARE], 0.5, color=COLOR_FORMAT, edgecolor="#2C3E50",
           linewidth=1.1, label="`<think>` format compliance")
    ax.bar([0], [1 - V1_FORMAT_SHARE], 0.5, bottom=[V1_FORMAT_SHARE], color=COLOR_MUTED,
           edgecolor="#2C3E50", linewidth=1.1, label="everything else")
    ax.set_xticks([0])
    ax.set_xticklabels(['the v1 "+590% improvement"'], fontsize=9.5)
    ax.set_ylabel("share of the reported reward gain", fontsize=10)
    ax.set_ylim(0, 1.32)
    ax.annotate(f"{V1_FORMAT_SHARE:.0%}", xy=(0, V1_FORMAT_SHARE / 2), ha="center", va="center",
                fontsize=15, fontweight="bold", color="white")
    panel(ax, "(c) What the headline number actually measured", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper right")

    fig.suptitle("B1 — Circular evaluation: scoring the model with its own training reward",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_2_circular_evaluation")


# ---------------------------------------------------------------------------
# B2 -- packed evaluation and unpaired tests
# ---------------------------------------------------------------------------

def fig_protocol():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)
    labels = list(PACKED_DELTAS)

    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.36
    pk = [PACKED_DELTAS[k] for k in labels]
    up = [UNPACKED_DELTAS[k] for k in labels]
    err = np.array([[UNPACKED_DELTAS[k] - UNPACKED_CIS[k][0] for k in labels],
                    [UNPACKED_CIS[k][1] - UNPACKED_DELTAS[k] for k in labels]])
    ax.bar(x - w / 2, pk, w, color=COLOR_TRAP, edgecolor="#2C3E50", linewidth=1.1,
           label="packed (block-scored)")
    ax.bar(x + w / 2, up, w, yerr=err, capsize=4, color=COLOR_FIX, edgecolor="#2C3E50",
           linewidth=1.1, label="unpacked + paired (fix)")
    ax.axhline(0, color="#2C3E50", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("measured Δloss, 14.1k − 8.5k (nats)", fontsize=10)
    for i, v in enumerate(pk):
        ax.annotate(f"{v:+.4f}", xy=(i - w / 2, v), xytext=(0, 4 if v > 0 else -13),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold", color=COLOR_TRAP)
    ax.text(0.5, 0.93, "packed: OPPOSITE SIGNS\non nested test sets",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, fontweight="bold",
            color=COLOR_TRAP, bbox=dict(boxstyle="round,pad=0.35", fc="#FDEDEC",
                                        ec=COLOR_TRAP, lw=1))
    panel(ax, "(a) Packing flipped the conclusion", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="lower left")

    ax = axes[1]
    bars = ax.bar([0, 1], [PACKED_ABS, UNPACKED_ABS], 0.5,
                  color=[COLOR_TRAP, COLOR_FIX], edgecolor="#2C3E50", linewidth=1.1)
    for b, v in zip(bars, [PACKED_ABS, UNPACKED_ABS]):
        ax.annotate(f"{v:.4f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color="#2C3E50")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["packed", "unpacked"], fontsize=9.5)
    ax.set_ylim(2.0, 2.42)
    ax.set_ylabel("test loss, same file & model (nats)", fontsize=10)
    # draw the bias arrow in the gap between the bars so it clears both labels
    ax.annotate("", xy=(0.5, PACKED_ABS), xytext=(0.5, UNPACKED_ABS),
                arrowprops=dict(arrowstyle="<->", color=COLOR_KL, lw=1.8))
    ax.text(0.56, (PACKED_ABS + UNPACKED_ABS) / 2,
            f"{UNPACKED_ABS - PACKED_ABS:.3f} nats\noptimistic bias",
            fontsize=9, color=COLOR_KL, fontweight="bold", va="center", ha="center")
    panel(ax, "(b) Packing makes every model look better", "[historical measurement]")

    ax = axes[2]
    effect = abs(np.mean(up))
    ax.bar([0], [effect], 0.45, color=COLOR_CONCEPT, edgecolor="#2C3E50",
           linewidth=1.1, label="true effect being measured")
    ax.bar([1], [abs(UNPACKED_ABS - PACKED_ABS)], 0.45, color=COLOR_TRAP,
           edgecolor="#2C3E50", linewidth=1.1, label="packing bias")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["corpus-doubling\neffect (paper)", "measurement\nartefact"], fontsize=9)
    ax.set_ylabel("magnitude (nats)", fontsize=10)
    ax.set_yscale("log")
    for i, v in enumerate([effect, abs(UNPACKED_ABS - PACKED_ABS)]):
        ax.annotate(f"{v:.4f}", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9.5, fontweight="bold", color="#2C3E50")
    ax.set_ylim(top=abs(UNPACKED_ABS - PACKED_ABS) * 6)
    ax.text(0.5, 0.42, f"artefact is\n{abs(UNPACKED_ABS - PACKED_ABS) / effect:.0f}× the effect",
            transform=ax.transAxes, ha="center", va="center", fontsize=10, fontweight="bold",
            color=COLOR_TRAP, bbox=dict(boxstyle="round,pad=0.4", fc="#FDEDEC",
                                        ec=COLOR_TRAP, lw=1.2))
    panel(ax, "(c) The artefact dwarfed the signal", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper left")

    fig.suptitle("B2 — Evaluation protocol: packed loss and unpaired tests cannot resolve small effects",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_3_evaluation_protocol")


# ---------------------------------------------------------------------------
# B3 -- decoding parameters larger than the training effect
# ---------------------------------------------------------------------------

def fig_decoding():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    ax = axes[0]
    im = ax.imshow(DECODE_REPEAT, cmap="Reds", aspect="auto", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(DECODE_RPS)))
    ax.set_xticklabels([f"{r:.2f}" for r in DECODE_RPS], fontsize=9.5)
    ax.set_yticks(range(len(DECODE_TEMPS)))
    ax.set_yticklabels([f"{t:.1f}" for t in DECODE_TEMPS], fontsize=9.5)
    ax.set_xlabel("repetition penalty", fontsize=10)
    ax.set_ylabel("temperature", fontsize=10)
    for i in range(DECODE_REPEAT.shape[0]):
        for j in range(DECODE_REPEAT.shape[1]):
            v = DECODE_REPEAT[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="white" if v > 0.3 else "#2C3E50")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="repeated 5-gram fraction")
    ax.set_title("(a) Degeneration is a decoder property", fontsize=11.5,
                 fontweight="bold", pad=14, color="#2C3E50")
    ax.text(0.5, 1.015, "[historical measurement]", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.5, color="#7F8C8D", style="italic")

    ax = axes[1]
    x = np.arange(2)
    w = 0.36
    sft = [DECODE_SFT_ACC_RP105, DECODE_SFT_ACC_RP120]
    grpo = [DECODE_GRPO_ACC_RP105, DECODE_GRPO_ACC_RP120]
    ax.bar(x - w / 2, sft, w, label="SFT", color=COLOR_SFT, edgecolor="#2C3E50", linewidth=1.1)
    ax.bar(x + w / 2, grpo, w, label="GRPO", color=COLOR_REWARD, edgecolor="#2C3E50", linewidth=1.1)
    for i in range(2):
        for off, v in ((-w / 2, sft[i]), (w / 2, grpo[i])):
            ax.annotate(f"{v:.1%}", xy=(i + off, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold", color="#2C3E50")
    ax.set_xticks(x)
    ax.set_xticklabels(["rp = 1.05\n(degenerate)", "rp = 1.20\n(deployment)"], fontsize=9.5)
    ax.set_ylabel("numerical accuracy @1%", fontsize=10)
    ax.set_ylim(0, 0.225)
    ax.annotate("GRPO ahead", xy=(0, 0.115), ha="center", fontsize=9,
                fontweight="bold", color=COLOR_TRAP)
    ax.annotate("GRPO behind", xy=(1, 0.185), ha="center", fontsize=9,
                fontweight="bold", color=COLOR_FIX)
    panel(ax, "(b) The decoder flipped the sign of the result", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper left")

    ax = axes[2]
    train_effect = abs(DECODE_GRPO_ACC_RP120 - DECODE_SFT_ACC_RP120)
    decode_effect = abs(DECODE_SFT_ACC_RP120 - DECODE_SFT_ACC_RP105)
    ax.bar([0], [train_effect], 0.45, color=COLOR_CONCEPT, edgecolor="#2C3E50",
           linewidth=1.1, label="SFT→GRPO training effect")
    ax.bar([1], [decode_effect], 0.45, color=COLOR_TRAP, edgecolor="#2C3E50",
           linewidth=1.1, label="rp 1.05→1.20 decoding effect")
    for i, v in enumerate([train_effect, decode_effect]):
        ax.annotate(f"{v:.1%} pts", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["what we wanted\nto measure", "what the decoder\ndid to us"], fontsize=9)
    ax.set_ylabel("|Δ| numerical accuracy @1%", fontsize=10)
    ax.set_ylim(0, 0.105)
    # sits in the gap between the two bars, clear of the legend
    ax.text(0.5, 0.46, f"decoding effect is\n{decode_effect / train_effect:.1f}× the\ntraining effect",
            transform=ax.transAxes, ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=COLOR_TRAP, bbox=dict(boxstyle="round,pad=0.35", fc="#FDEDEC",
                                        ec=COLOR_TRAP, lw=1.2))
    panel(ax, "(c) Pin the decoder before comparing models", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper left")

    fig.suptitle("B3 — At 113M, a decoding mistake is larger than the training effect",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_4_decoding_artifacts")


# ---------------------------------------------------------------------------
# D1 -- silent failures: dead reward, padding-driven KL, frozen policy
# ---------------------------------------------------------------------------

def fig_silent():
    v1, v3 = load_grpo_log("grpo_v1_dead_verifier"), load_grpo_log("grpo")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    ax = axes[0]
    ax.plot(v1["step"], v1["num_reward"], color=COLOR_TRAP, linewidth=2.6,
            label=f"v1: dead verifier ({int((v1['num_reward'] != 0).sum())}/{len(v1['step'])} steps nonzero)")
    ax.plot(v3["step"], rolling_mean(v3["num_reward"], 20), color=COLOR_FIX,
            linewidth=2.2, label=f"v3: live ({int((v3['num_reward'] != 0).sum())}/{len(v3['step'])} nonzero)")
    ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.1)
    ax.set_xlabel("GRPO step", fontsize=10)
    ax.set_ylabel("numerical sub-reward", fontsize=10)
    panel(ax, "(a) A reward that was never wired up", "[log: grpo_v1 / grpo]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8, loc="lower left")
    ax.text(0.5, 0.55, "exactly 0.0000 for all 300 steps\n(missing `import math` upstream)",
            transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold", color=COLOR_TRAP,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDEDEC", ec=COLOR_TRAP, lw=1))

    ax = axes[1]
    ax.semilogy(v1["step"], np.maximum(v1["kl_div"], 1e-4), color=COLOR_TRAP,
                linewidth=1.6, label="v1: padding tokens in loss/KL")
    ax.semilogy(v3["step"], np.maximum(v3["kl_div"], 1e-4), color=COLOR_FIX,
                linewidth=1.6, label="v3: padding masked out")
    ax.axhline(0.05, color="#E74C3C", linestyle="--", linewidth=1.2, label="KL target 0.05")
    peak = int(np.argmax(v1["kl_div"]))
    ax.annotate(f"KL = {v1['kl_div'][peak]:.2f}\nat step {int(v1['step'][peak])}",
                xy=(v1["step"][peak], v1["kl_div"][peak]),
                xytext=(v1["step"][peak] - 150, 2.2),
                arrowprops=dict(facecolor=COLOR_TRAP, shrink=0.05, width=1.2, headwidth=6),
                fontsize=9, fontweight="bold", color=COLOR_TRAP)
    ax.set_xlabel("GRPO step", fontsize=10)
    ax.set_ylabel("token KL (log scale)", fontsize=10)
    panel(ax, "(b) The v1 'learning' was padding gradients", "[log: grpo_v1 / grpo]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8, loc="lower left")

    ax = axes[2]
    ax.plot(v3["step"], v3["kl_div"], color=COLOR_KL, alpha=0.35, linewidth=1.0)
    ax.plot(v3["step"], rolling_mean(v3["kl_div"], 20), color=COLOR_KL,
            linewidth=2.4, label="KL (rolling)")
    ax.axhline(0.05, color="#E74C3C", linestyle="--", linewidth=1.1, label="target 0.05")
    ax.axhline(0.10, color="#E74C3C", linestyle=":", linewidth=1.1, label="deadband edge (2×)")
    ax.set_xlabel("GRPO step", fontsize=10)
    ax.set_ylabel("token KL", fontsize=10, color=COLOR_KL)
    axb = ax.twinx()
    axb.plot(v3["step"], v3["kl_coeff"], color=COLOR_CLEAN, linewidth=2.4, label="β (adaptive)")
    axb.set_ylabel("KL coefficient β", fontsize=10, color=COLOR_CLEAN)
    axb.spines["top"].set_visible(False)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = axb.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, frameon=True, facecolor="white", edgecolor="#BDC3C7",
              fontsize=8, loc="upper left")
    panel(ax, "(c) With padding fixed, the controller works", "[log: grpo]")

    fig.suptitle("D1 — Silent failures: code that runs, logs cleanly, and measures nothing",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_5_silent_failures")


# ---------------------------------------------------------------------------
# E1 -- checkpoint selection on the wrong criterion
# ---------------------------------------------------------------------------

def fig_selection():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    ax = axes[0]
    x = np.arange(2)
    w = 0.5
    ax.bar(x, BEST_LAST_LOSS, w, color=[COLOR_FIX, COLOR_TRAP],
           edgecolor="#2C3E50", linewidth=1.1)
    for i, v in enumerate(BEST_LAST_LOSS):
        ax.annotate(f"{v:.4f}", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_xticks(x)
    ax.set_xticklabels(["best\n(step 80)", "last\n(step 336)"], fontsize=9.5)
    ax.set_ylabel("held-out loss (nats)", fontsize=10)
    ax.set_ylim(0, 3.35)
    axb = ax.twinx()
    axb.plot(x, [t * 100 for t in BEST_LAST_THINK], "o-", color=COLOR_FORMAT,
             linewidth=2.6, markersize=10, label="<think> rate (right axis)")
    axb.set_ylabel("<think> rate (%)", fontsize=10, color=COLOR_FORMAT)
    axb.set_ylim(0, 100)
    axb.spines["top"].set_visible(False)
    for i, t in enumerate(BEST_LAST_THINK):
        axb.annotate(f"{t:.1%}", xy=(i, t * 100), xytext=(0, -18),
                     textcoords="offset points", ha="center", fontsize=9.5,
                     fontweight="bold", color=COLOR_FORMAT)
    panel(ax, "(a) Likelihood and behaviour disagree", "[historical measurement]")
    axb.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper center")
    ax.text(0.5, 0.055, "+0.51 nats WORSE (t=+33)\nyet +12.7 pp BETTER formatted (z=+2.47)",
            transform=ax.transAxes, ha="center", fontsize=8.8, fontweight="bold",
            color="#2C3E50", bbox=dict(boxstyle="round,pad=0.35", fc="#FEF9E7",
                                       ec="#B7950B", lw=1))

    ax = axes[1]
    sel = json.load(open(DOCS / "checkpoint_selection.json", encoding="utf-8"))
    names, acc, think = [], [], []
    for path, m in sel["results"].items():
        names.append(Path(path).stem.replace("sft_ckpt_", ""))
        acc.append(m["numerical_acc_5pct"])
        think.append(m["numerical_think_rate"])
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, acc, w, label="num acc @5% (primary)", color=COLOR_CONCEPT,
           edgecolor="#2C3E50", linewidth=1.1)
    ax.bar(x + w / 2, think, w, label="<think> rate (tie-break)", color=COLOR_FORMAT,
           edgecolor="#2C3E50", linewidth=1.1)
    for i in range(len(names)):
        ax.annotate(f"{acc[i]:.1%}", xy=(i - w / 2, acc[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{think[i]:.1%}", xy=(i + w / 2, think[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    win = Path(sel["selected"]).stem.replace("sft_ckpt_", "")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{'← SELECTED' if n == win else ''}" for n in names], fontsize=8.5)
    ax.set_ylabel("rate", fontsize=10)
    # headroom so the note clears the tallest bar instead of sitting on top of it
    ax.set_ylim(0, 1.45)
    ax.axhspan(min(acc), max(acc), color=COLOR_MUTED, alpha=0.18)
    ax.text(0.5, 0.68, "primary metric spread (2 pp) is within\nnoise at n=166 → tie-break decides",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color="#566573",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COLOR_MUTED, lw=1))
    panel(ax, "(b) Val loss would have picked the worst one", "[log: checkpoint_selection.json]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8, loc="upper left")

    ax = axes[2]
    v3 = load_grpo_log("grpo")
    roll = rolling_mean(v3["mean_reward"], 20)
    ax.plot(v3["step"], roll, color=COLOR_REWARD, linewidth=2.4, label="reward (rolling)")
    peak = int(np.argmax(roll))
    ax.axvline(v3["step"][peak], color=COLOR_FIX, linestyle="--", linewidth=1.6,
               label=f"peak at step {int(v3['step'][peak])}")
    ax.axvline(200, color=COLOR_TRAP, linestyle="--", linewidth=1.6, label="final step 200")
    ax.set_xlabel("GRPO step", fontsize=10)
    ax.set_ylabel("mean reward", fontsize=10)
    panel(ax, "(c) The last GRPO step is not the best one", "[log: grpo]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8, loc="lower right")
    ax.text(0.5, 0.95, "step 100: <think> 98.2%, echo 11.1%\nstep 200: <think> 91.6%, echo 16.4%",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6, fontweight="bold",
            color="#2C3E50", bbox=dict(boxstyle="round,pad=0.35", fc="#EAFAF1",
                                       ec=COLOR_FIX, lw=1))

    fig.suptitle("E1 — Selecting checkpoints on validation loss picks the wrong model",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_6_checkpoint_selection")


# ---------------------------------------------------------------------------
# F1 -- confounded scaling law
# ---------------------------------------------------------------------------

def fig_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    ax = axes[0]
    for N, L, c, lbl in ((CONFOUNDED_N, CONFOUNDED_LOSS, COLOR_TRAP, "confounded (mixture varies with size)"),
                         (MATCHED_N, MATCHED_LOSS, COLOR_FIX, "composition-matched 46/25/28 (fix)")):
        lg = np.log2(N)
        a, b = np.polyfit(lg, L, 1)
        xs = np.linspace(lg.min() - 0.3, lg.max() + 0.3, 50)
        ax.plot(2 ** xs, a * xs + b, "--", color=c, linewidth=1.8, alpha=0.85)
        ax.plot(N, L, "o", color=c, markersize=9, markeredgecolor="#2C3E50",
                markeredgewidth=1.0, label=f"{lbl}\nslope {a:.4f} nats/doubling")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training pairs (log₂)", fontsize=10)
    ax.set_ylabel("held-out test loss (nats)", fontsize=10)
    panel(ax, "(a) The confounded fit understated data value", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8, loc="upper right")
    sc = (0.0813 - 0.0633) / 0.0633
    ax.text(0.03, 0.06, f"clean slope is {sc:.0%} steeper", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=COLOR_FIX)

    ax = axes[1]
    pts = list(MIXTURE)
    paper = [MIXTURE[p][0] for p in pts]
    fund = [MIXTURE[p][1] for p in pts]
    eq = [MIXTURE[p][2] for p in pts]
    ax.bar(pts, paper, 0.55, label="paper", color=COLOR_CONCEPT, edgecolor="#2C3E50", linewidth=1.1)
    ax.bar(pts, fund, 0.55, bottom=paper, label="fundamentals", color=COLOR_FORMAT,
           edgecolor="#2C3E50", linewidth=1.1)
    ax.bar(pts, eq, 0.55, bottom=np.array(paper) + np.array(fund), label="earthquakes",
           color=COLOR_CLEAN, edgecolor="#2C3E50", linewidth=1.1)
    ax.set_ylabel("share of corpus", fontsize=10)
    ax.set_ylim(0, 1.28)
    ax.set_xlabel("scaling-series point", fontsize=10)
    panel(ax, "(b) Size and mixture varied together", "[historical measurement]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5,
              loc="upper center", ncol=3)

    ax = axes[2]
    doms = list(PER_DOMAIN)
    vals = [PER_DOMAIN[d][0] for d in doms]
    err = np.array([[PER_DOMAIN[d][0] - PER_DOMAIN[d][1][0] for d in doms],
                    [PER_DOMAIN[d][1][1] - PER_DOMAIN[d][0] for d in doms]])
    bars = ax.bar(doms, vals, 0.55, yerr=err, capsize=5,
                  color=[COLOR_CONCEPT, COLOR_CLEAN, COLOR_FORMAT],
                  edgecolor="#2C3E50", linewidth=1.1)
    ax.axhline(0, color="#2C3E50", linewidth=1.2)
    # bars grow downward from 0, so label at the bar midpoint rather than
    # offsetting below the value (which put white text on white background)
    for b, d in zip(bars, doms):
        v, _, t = PER_DOMAIN[d]
        ax.annotate(f"{v:+.4f}\nt={t:.0f}", xy=(b.get_x() + b.get_width() / 2, v / 2),
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.set_ylabel("paired Δloss, 8.0k → 14.1k (nats)", fontsize=10)
    ax.set_ylim(bottom=-0.215)
    panel(ax, "(c) At fixed mixture, every domain lifts", "[historical measurement]")
    ax.text(0.5, 0.07, "the old panel's 'fundamentals flat, n.s.'\nwas a mixture artefact",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color="#566573",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLOR_MUTED, lw=1))

    fig.suptitle("F1 — A scaling law measured while the data mixture also changed",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_7_confounded_scaling")


FIGURES = {
    1: ("Train/test leakage", fig_leakage),
    2: ("Circular evaluation", fig_circular),
    3: ("Evaluation protocol", fig_protocol),
    4: ("Decoding artifacts", fig_decoding),
    5: ("Silent failures", fig_silent),
    6: ("Checkpoint selection", fig_selection),
    7: ("Confounded scaling", fig_scaling),
}


def main():
    ap = argparse.ArgumentParser(description="Render the pitfall retrospective figures.")
    ap.add_argument("--only", type=int, action="append", choices=sorted(FIGURES),
                    help="render only these figure numbers (repeatable)")
    args = ap.parse_args([])

    set_publication_style()
    wanted = args.only or sorted(FIGURES)
    print("=" * 78)
    print("Rendering pitfall figures")
    print("=" * 78)
    for n in wanted:
        title, fn = FIGURES[n]
        print(f"[{n}] {title}")
        fn()
    print("done.")




# ---------------------------------------------------------------------------
# C1 / D2 -- the metric's floor exceeded the model
# ---------------------------------------------------------------------------

# Measured by src.analysis.validate_measurement floor on the 166 held-out numerical items.
SPRAY_K = [1, 2, 3, 5, 8]
SPRAY_ACC5_OLD = [0.020, 0.040, 0.058, 0.093, 0.144]   # min-over-first-8 rule
OLD_SFT, OLD_GRPO = 0.127, 0.133      # as previously reported
NEW_SFT, NEW_GRPO = 0.096, 0.102      # re-measured, tail-3 rule
OLD_FLOOR, NEW_FLOOR = 0.143, 0.058
OLD_CEIL, NEW_CEIL = 0.681, 0.861
# src.analysis.validate_measurement hacking, mean over 166 items, before vs after fix
PROBE_BEFORE = {"reference (genuine)": 0.418, "verbose filler + correct answer": 0.557}
PROBE_AFTER = {"reference (genuine)": 0.768, "verbose filler + correct answer": 0.768}
ECHO_ON_GENUINE_BEFORE, ECHO_ON_GENUINE_AFTER = -0.428, -0.006


def fig_metric_validity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=300)

    # (a) the gameability curve that made the floor exceed the model
    ax = axes[0]
    ax.plot(SPRAY_K, [v * 100 for v in SPRAY_ACC5_OLD], "o-", color=COLOR_TRAP,
            linewidth=2.4, markersize=8, markeredgecolor="#2C3E50",
            label="knowledge-free number spray")
    ax.axhline(OLD_GRPO * 100, color=COLOR_CONCEPT, linestyle="--", linewidth=2.0,
               label=f"the real model ({OLD_GRPO:.1%})")
    ax.fill_between([SPRAY_K[0], SPRAY_K[-1]], OLD_GRPO * 100, 15.5,
                    color=COLOR_TRAP, alpha=0.10)
    # placed in the empty lower-right; an arrow here would cross the curve
    ax.text(5.9, 3.4, "the spray OVERTAKES\nthe model at k = 8", ha="center",
            fontsize=9.2, fontweight="bold", color=COLOR_TRAP,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FDEDEC", ec=COLOR_TRAP, lw=1.2))
    ax.set_xlabel("numbers emitted in the answer (k)", fontsize=10)
    ax.set_ylabel("accuracy @5% (%)", fontsize=10)
    ax.set_ylim(0, 15.5)
    panel(ax, "(a) Every extra number was a lottery ticket", "[validate_measurement floor]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper left")

    # (b) floor / model / ceiling, before and after the scoring fix
    ax = axes[1]
    x = np.arange(2)
    w = 0.26
    ax.bar(x - w, [OLD_FLOOR, NEW_FLOOR], w, label="floor (blind spray)",
           color=COLOR_TRAP, edgecolor="#2C3E50", linewidth=1.1)
    ax.bar(x, [OLD_GRPO, NEW_GRPO], w, label="measured model",
           color=COLOR_CONCEPT, edgecolor="#2C3E50", linewidth=1.1)
    ax.bar(x + w, [OLD_CEIL, NEW_CEIL], w, label="ceiling (ground truth)",
           color=COLOR_FIX, edgecolor="#2C3E50", linewidth=1.1)
    for i, (f, m, c) in enumerate(((OLD_FLOOR, OLD_GRPO, OLD_CEIL),
                                   (NEW_FLOOR, NEW_GRPO, NEW_CEIL))):
        for off, v in ((-w, f), (0, m), (w, c)):
            ax.annotate(f"{v:.1%}", xy=(i + off, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        fontweight="bold", color="#2C3E50")
    ax.set_xticks(x)
    ax.set_xticklabels(["min over first 8\n(as reported)", "last 3\n(corrected)"], fontsize=9.5)
    ax.set_ylabel("accuracy @5%", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.text(0, 0.30, "floor ABOVE\nthe model", ha="center", fontsize=9,
            fontweight="bold", color=COLOR_TRAP,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec=COLOR_TRAP, lw=1))
    ax.text(1, 0.30, "model clears\nthe floor", ha="center", fontsize=9,
            fontweight="bold", color=COLOR_FIX,
            bbox=dict(boxstyle="round,pad=0.3", fc="#EAFAF1", ec=COLOR_FIX, lw=1))
    panel(ax, "(b) The fix restored dynamic range", "[validate_measurement floor]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8, loc="upper left")

    # (c) the reward-hacking channel, closed
    ax = axes[2]
    labels = ["genuine\nreference", "verbose filler\n+ correct answer"]
    before = [PROBE_BEFORE["reference (genuine)"], PROBE_BEFORE["verbose filler + correct answer"]]
    after = [PROBE_AFTER["reference (genuine)"], PROBE_AFTER["verbose filler + correct answer"]]
    _bars(ax, labels, before, after, "before fix", "after fix",
          fmt="{:+.3f}", ylabel="composite reward (mean over 166 items)")
    ax.set_ylim(0, 1.05)
    # bracket the two RED (before-fix) bars in the gap between the groups
    ax.annotate("", xy=(0.5, PROBE_BEFORE["reference (genuine)"]),
                xytext=(0.5, PROBE_BEFORE["verbose filler + correct answer"]),
                arrowprops=dict(arrowstyle="<->", color=COLOR_TRAP, lw=1.8))
    ax.text(0.5, 0.487, "+0.139\npadding PAID", fontsize=8.8, color=COLOR_TRAP,
            fontweight="bold", va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.text(0.5, 0.11, f"echo penalty on the GROUND TRUTH:\n"
                       f"{ECHO_ON_GENUINE_BEFORE:+.3f}  →  {ECHO_ON_GENUINE_AFTER:+.3f}",
            transform=ax.transAxes, ha="center", fontsize=8.8, fontweight="bold",
            color="#2C3E50", bbox=dict(boxstyle="round,pad=0.35", fc="#FEF9E7",
                                       ec="#B7950B", lw=1))
    panel(ax, "(c) Padding out-scored the truth, until fixed", "[validate_measurement hacking]")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=8.5, loc="upper left")

    fig.suptitle("C1 / D2 — A metric whose floor exceeded the model it was grading",
                 fontsize=13.5, fontweight="bold", color="#2C3E50", y=1.04)
    fig.tight_layout()
    save(fig, "pitfall_8_metric_validity")


FIGURES[8] = ("Metric validity", fig_metric_validity)


if __name__ == "__main__":
    main()
