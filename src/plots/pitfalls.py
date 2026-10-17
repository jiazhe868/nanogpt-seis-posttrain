"""Figures illustrating the methodological pitfalls hit in this project.

Each figure contrasts THE TRAP (what the broken measurement said) with THE FIX
(what the corrected measurement says). Together with PITFALLS.md they are the
retrospective record of how the headline numbers in this repo went from
untrustworthy to defensible.

Provenance rules followed here:
  * Anything that can be recomputed from a surviving log or results file IS
    recomputed at plot time -- see load_grpo_log() and the docs/*.json
    readers (including the z statistics in B1).
  * Historical measurements whose run artifacts are not reloadable (v1-era
    evaluations, decoder sweeps, unpacked paired scores) are declared below as
    named module constants, each with a comment saying what it measured.

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
from matplotlib.patches import Patch

from src.plots import (  # noqa: E402
    COLOR_CLEAN, COLOR_CONCEPT, COLOR_FIX, COLOR_FORMAT, COLOR_KL, COLOR_LOSS,
    COLOR_MUTED, COLOR_REWARD, COLOR_SFT, COLOR_TRAP, DOCS, INK,
    legend, panel, rolling_mean, save, set_publication_style)

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
PACKED_DELTAS = {"paper_common\n281 pairs": +0.0078, "paper\n356 pairs": -0.0105}
UNPACKED_DELTAS = {"paper_common\n281 pairs": -0.0100, "paper\n356 pairs": -0.0092}
UNPACKED_CIS = {"paper_common\n281 pairs": (-0.0154, -0.0045),
                "paper\n356 pairs": (-0.0137, -0.0045)}
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


# ---------------------------------------------------------------------------
# Loaders and drawing helpers
# ---------------------------------------------------------------------------

FIGSIZE = (15, 5.0)
EDGE = dict(edgecolor=INK, linewidth=1.0)


def load_grpo_log(run: str) -> dict:
    """Load a GRPO run log from checkpoints/runs/<run>.csv."""
    path = ROOT / "checkpoints" / "runs" / f"{run}.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for k in rows[0]:
        out[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return out


def z_two_proportions(p1: float, p2: float, n1: int, n2: int) -> float:
    """Pooled two-proportion z statistic for p2 - p1."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    return (p2 - p1) / se if se else 0.0


def bar_labels(ax, rects, fmt, values=None, below=None, fontsize=12, color=INK):
    """Value label above each positive bar, under each negative one (or under `below`)."""
    values = [r.get_height() for r in rects] if values is None else values
    for k, (r, v) in enumerate(zip(rects, values)):
        up = r.get_height() >= 0 and below is None
        y = r.get_y() + r.get_height() if below is None else below[k]
        ax.annotate(fmt.format(v).replace("-", "\u2212"), xy=(r.get_x() + r.get_width() / 2, y),
                    xytext=(0, 3 if up else -3),
                    textcoords="offset points", ha="center", va="bottom" if up else "top",
                    fontsize=fontsize, fontweight="bold", color=color)


def finish(fig, title: str, name: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.94))  # lay out the panels first, then put the title in the strip above
    fig.suptitle(title, fontsize=17, fontweight="bold", color=INK, y=0.995)
    save(fig, name)


# ---------------------------------------------------------------------------
# A1 -- train/test leakage, twice
# ---------------------------------------------------------------------------

def fig_leakage():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)

    for ax, (before, after), ticks, ylabel, title in (
            (axes[0], (LEAK_PAIR_LEVEL_FRAC, LEAK_CHUNK_LEVEL_FRAC), ["pair-level split", "chunk-level split"],
             "val pairs with a train sibling (%)", "(a) The original split leaked"),
            (axes[1], (GEN_LEAK_BEFORE, GEN_LEAK_AFTER), ["md5(topic::variant)", "md5(identity)"],
             "test identities seen in training (%)", "(b) My generator leaked again")):
        r = ax.bar([0, 1], [100 * before, 100 * after], 0.6, color=[COLOR_TRAP, COLOR_FIX], **EDGE)
        bar_labels(ax, r, "{:.0f}%")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ticks)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 115)
        panel(ax, title)

    ax = axes[2]
    apparent, true = abs(EQ_GAIN_APPARENT), abs(EQ_GAIN_TRUE)
    r = ax.bar([0, 1], [apparent, true], 0.6, color=[COLOR_TRAP, COLOR_FIX], **EDGE)
    ax.bar([0], [apparent - true], 0.6, bottom=[true], color="none", edgecolor="white", linewidth=1.5,
           hatch="///")
    bar_labels(ax, r, "{:.3f}")
    ax.text(0.27, (apparent + true) / 2, f"{(apparent - true) / apparent:.0%} came\nfrom the leak", ha="left",
            va="center", fontsize=12.5, fontweight="bold", color=COLOR_TRAP)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before fix", "after fix"])
    ax.set_ylabel("loss drop on earthquakes (nats)")
    ax.set_ylim(0, 0.92)
    panel(ax, "(c) The leak inflated the gain")

    finish(fig, "A1 — Train/test leakage, made twice", "pitfall_1_data_leakage")


# ---------------------------------------------------------------------------
# B1 -- circular evaluation
# ---------------------------------------------------------------------------

def fig_circular():
    v3 = load_grpo_log("grpo")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)

    ax = axes[0]
    ax.plot(v3["step"], rolling_mean(v3["mean_reward"], 20), color=COLOR_TRAP, linewidth=2.4)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("training reward (rolling mean)")
    ax.set_ylim(-0.9, 0.05)
    ax.text(0.04, 0.95, "the objective itself,\nnot a result", transform=ax.transAxes, va="top",
            fontsize=12.5, fontweight="bold", color=COLOR_TRAP)
    panel(ax, "(a) The reward rises by design")

    ax = axes[1]
    res = json.load(open(DOCS / "eval_rl_results.json", encoding="utf-8"))
    sft, grpo = res["SFT"], res["GRPO @100 (selected)"]
    keys = ["numerical_acc_5pct", "numerical_think_rate"]
    x, w = np.arange(len(keys)), 0.4
    r1 = ax.bar(x - w / 2, [100 * sft[k] for k in keys], w, color=COLOR_SFT, label="SFT", **EDGE)
    r2 = ax.bar(x + w / 2, [100 * grpo[k] for k in keys], w, color=COLOR_REWARD, label="GRPO", **EDGE)
    for r in (r1, r2):
        bar_labels(ax, r, "{:.1f}")
    for i, k in enumerate(keys):
        z = z_two_proportions(sft[k], grpo[k], sft["n_numerical"], grpo["n_numerical"])
        sig = abs(z) >= 1.96
        ax.text(i, 111, f"z = {z:+.1f}" if sig else "n.s.", ha="center", fontsize=12.5, fontweight="bold",
                color=COLOR_FIX if sig else COLOR_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(["accuracy @5%", "<think> rate"])
    ax.set_ylabel("held-out rate (%)")
    ax.set_ylim(0, 122)
    legend(ax, loc="center left")
    panel(ax, "(b) Held-out metrics disagree")

    ax = axes[2]
    ax.bar([0], [100 * V1_FORMAT_SHARE], 1.0, color=COLOR_FORMAT, **EDGE)
    ax.bar([0], [100 * (1 - V1_FORMAT_SHARE)], 1.0, bottom=[100 * V1_FORMAT_SHARE], color=COLOR_MUTED, **EDGE)
    ax.text(0, 50 * V1_FORMAT_SHARE, f"<think> format\n{V1_FORMAT_SHARE:.0%}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0, 100 * V1_FORMAT_SHARE + 50 * (1 - V1_FORMAT_SHARE), f"other {1 - V1_FORMAT_SHARE:.0%}",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.set_xlim(-0.75, 0.75)
    ax.set_xticks([0])
    ax.set_xticklabels(['v1 "+590%" reward gain'])
    ax.set_ylabel("share of the gain (%)")
    ax.set_ylim(0, 100)
    panel(ax, "(c) What +590% measured")

    finish(fig, "B1 — Circular evaluation: scoring with the training reward", "pitfall_2_circular_evaluation")


# ---------------------------------------------------------------------------
# B2 -- packed evaluation and unpaired tests
# ---------------------------------------------------------------------------

def fig_protocol():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)
    labels = list(PACKED_DELTAS)
    pk = [PACKED_DELTAS[k] for k in labels]
    up = [UNPACKED_DELTAS[k] for k in labels]

    ax = axes[0]
    x, w = np.arange(len(labels)), 0.4
    err = np.array([[UNPACKED_DELTAS[k] - UNPACKED_CIS[k][0] for k in labels],
                    [UNPACKED_CIS[k][1] - UNPACKED_DELTAS[k] for k in labels]])
    r = ax.bar(x - w / 2, pk, w, color=COLOR_TRAP, label="packed", **EDGE)
    ax.bar(x + w / 2, up, w, yerr=err, capsize=4, color=COLOR_FIX, label="unpacked, paired", **EDGE)
    bar_labels(ax, r, "{:+.4f}", color=COLOR_TRAP)
    ax.axhline(0, color=INK, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ loss, 14.1k − 8.5k (nats)")
    ax.set_ylim(-0.019, 0.014)
    legend(ax, loc="upper right")
    panel(ax, "(a) Packing flipped the sign")

    ax = axes[1]
    r = ax.bar([0, 1], [PACKED_ABS, UNPACKED_ABS], 0.6, color=[COLOR_TRAP, COLOR_FIX], **EDGE)
    bar_labels(ax, r, "{:.3f}")
    half = r[0].get_width() / 2
    ax.plot([half, 0.5], [PACKED_ABS] * 2, ":", color=COLOR_KL, linewidth=1.4)
    ax.plot([0.5, 1 - half], [UNPACKED_ABS] * 2, ":", color=COLOR_KL, linewidth=1.4)
    ax.annotate("", xy=(0.5, PACKED_ABS), xytext=(0.5, UNPACKED_ABS),
                arrowprops=dict(arrowstyle="<->", color=COLOR_KL, lw=1.8))
    ax.text(0.5, (PACKED_ABS + UNPACKED_ABS) / 2, f"{UNPACKED_ABS - PACKED_ABS:.3f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=COLOR_KL,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["packed", "unpacked"])
    ax.set_ylim(2.0, 2.36)
    ax.set_ylabel("loss, same model and file (nats)")
    panel(ax, "(b) Packing lowers the loss")

    ax = axes[2]
    effect, bias = abs(np.mean(up)), abs(UNPACKED_ABS - PACKED_ABS)
    r = ax.bar([0, 1], [effect, bias], 0.6, color=[COLOR_CONCEPT, COLOR_TRAP], **EDGE)
    bar_labels(ax, r, "{:.4f}")
    ax.set_yscale("log")
    ax.set_ylim(0.004, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2× data effect", "packing bias"])
    ax.set_ylabel("size (nats, log scale)")
    panel(ax, f"(c) The bias is {bias / effect:.0f}× the effect")

    finish(fig, "B2 — Packed, unpaired evaluation hides small effects", "pitfall_3_evaluation_protocol")


# ---------------------------------------------------------------------------
# B3 -- decoding parameters larger than the training effect
# ---------------------------------------------------------------------------

def fig_decoding():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)

    ax = axes[0]
    im = ax.imshow(DECODE_REPEAT, cmap="Reds", aspect="auto", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(DECODE_RPS)))
    ax.set_xticklabels([f"{r:.2f}" for r in DECODE_RPS])
    ax.set_yticks(range(len(DECODE_TEMPS)))
    ax.set_yticklabels([f"{t:.1f}" for t in DECODE_TEMPS])
    ax.set_xlabel("repetition penalty")
    ax.set_ylabel("temperature")
    for i in range(DECODE_REPEAT.shape[0]):
        for j in range(DECODE_REPEAT.shape[1]):
            v = DECODE_REPEAT[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white" if v > 0.3 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("repeated 5-grams")
    ax.set_title("(a) Repetition vs decoder", fontsize=15, fontweight="bold", pad=10, color=INK)

    ax = axes[1]
    x, w = np.arange(2), 0.4
    sft = [100 * DECODE_SFT_ACC_RP105, 100 * DECODE_SFT_ACC_RP120]
    grpo = [100 * DECODE_GRPO_ACC_RP105, 100 * DECODE_GRPO_ACC_RP120]
    r1 = ax.bar(x - w / 2, sft, w, label="SFT", color=COLOR_SFT, **EDGE)
    r2 = ax.bar(x + w / 2, grpo, w, label="GRPO", color=COLOR_REWARD, **EDGE)
    for r in (r1, r2):
        bar_labels(ax, r, "{:.1f}")
    ax.text(0, 12.5, "GRPO ahead", ha="center", fontsize=12.5, fontweight="bold", color=COLOR_TRAP)
    ax.text(1, 19.5, "GRPO behind", ha="center", fontsize=12.5, fontweight="bold", color=COLOR_FIX)
    ax.set_xticks(x)
    ax.set_xticklabels(["rp = 1.05\n(degenerate)", "rp = 1.20\n(deployed)"])
    ax.set_ylabel("accuracy @1% (%)")
    ax.set_ylim(0, 25)
    legend(ax, loc="upper left")
    panel(ax, "(b) The decoder flips the sign")

    ax = axes[2]
    train_effect = 100 * abs(DECODE_GRPO_ACC_RP120 - DECODE_SFT_ACC_RP120)
    decode_effect = 100 * abs(DECODE_SFT_ACC_RP120 - DECODE_SFT_ACC_RP105)
    r = ax.bar([0, 1], [train_effect, decode_effect], 0.6, color=[COLOR_CONCEPT, COLOR_TRAP], **EDGE)
    bar_labels(ax, r, "{:.1f} pp")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SFT → GRPO", "rp 1.05 → 1.20"])
    ax.set_ylabel("|Δ accuracy @1%| (pp)")
    ax.set_ylim(0, 9.5)
    panel(ax, f"(c) Decoder effect {decode_effect / train_effect:.1f}× larger")

    finish(fig, "B3 — At 113M, decoding can outweigh training", "pitfall_4_decoding_artifacts")


# ---------------------------------------------------------------------------
# D1 -- silent failures: dead reward, padding-driven KL, frozen policy
# ---------------------------------------------------------------------------

def fig_silent():
    v1, v3 = load_grpo_log("grpo_v1_dead_verifier"), load_grpo_log("grpo")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)

    ax = axes[0]
    live = rolling_mean(v3["num_reward"], 20)
    ax.plot(v1["step"], v1["num_reward"], color=COLOR_TRAP, linewidth=2.6)
    ax.plot(v3["step"], live, color=COLOR_FIX, linewidth=2.2)
    always_zero = not np.any(v1["num_reward"] != 0)
    ax.text(v1["step"][-1], -0.03, "v1: always 0" if always_zero else "v1", ha="right", va="top",
            fontsize=12.5, fontweight="bold", color=COLOR_TRAP)
    ax.text(v3["step"][-1] + 8, live[-1], "v3: live", ha="left", va="center",
            fontsize=12.5, fontweight="bold", color=COLOR_FIX)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("numerical sub-reward")
    ax.set_ylim(-0.5, 0.05)
    panel(ax, "(a) A reward never wired up")

    ax = axes[1]
    ax.semilogy(v1["step"], np.maximum(v1["kl_div"], 1e-4), color=COLOR_TRAP, linewidth=1.4,
                label="v1: padding in KL")
    ax.semilogy(v3["step"], np.maximum(v3["kl_div"], 1e-4), color=COLOR_FIX, linewidth=1.8,
                label="v3: padding masked")
    ax.axhline(0.05, color=COLOR_LOSS, linestyle="--", linewidth=1.2, label="target 0.05")
    peak = int(np.argmax(v1["kl_div"]))
    ax.annotate(f"peak {v1['kl_div'][peak]:.1f}", xy=(v1["step"][peak], v1["kl_div"][peak]), xytext=(-6, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=12.5, fontweight="bold",
                color=COLOR_TRAP)
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("token KL (log scale)")
    legend(ax, loc="upper left")
    panel(ax, "(b) v1 KL came from padding")

    ax = axes[2]
    ax.plot(v3["step"], v3["kl_div"], color=COLOR_KL, alpha=0.3, linewidth=1.0)
    ax.plot(v3["step"], rolling_mean(v3["kl_div"], 20), color=COLOR_KL, linewidth=2.4)
    ax.axhline(0.05, color=COLOR_LOSS, linestyle="--", linewidth=1.2)
    ax.text(v3["step"][-1], 0.053, "KL target", ha="right", va="bottom", fontsize=12, color=COLOR_LOSS)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("token KL", color=COLOR_KL)
    panel(ax, "(c) Fixed: β pulls KL to target")
    axb = ax.twinx()
    axb.plot(v3["step"], v3["kl_coeff"], color=COLOR_CLEAN, linewidth=2.4)
    axb.set_ylabel("KL coefficient β", color=COLOR_CLEAN)
    axb.spines["top"].set_visible(False)
    axb.grid(False)

    finish(fig, "D1 — Silent failures: code that runs and measures nothing", "pitfall_5_silent_failures")


# ---------------------------------------------------------------------------
# E1 -- checkpoint selection on the wrong criterion
# ---------------------------------------------------------------------------

def fig_selection():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)
    colors = [COLOR_SFT, COLOR_CONCEPT]

    # (a) loss on the left axis, <think> rate on the right axis, same two checkpoints
    ax = axes[0]
    w = 0.4
    r = ax.bar([-w / 2, w / 2], BEST_LAST_LOSS, w, color=colors, **EDGE)
    bar_labels(ax, r, "{:.2f}")
    ax.set_ylabel("held-out loss (nats)")
    ax.set_ylim(0, 4.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["loss\n(left axis)", "<think> rate\n(right axis)"])
    legend(ax, loc="upper left", handles=[Patch(facecolor=c, **EDGE) for c in colors],
           labels=["step 80 (best loss)", "step 336 (last)"])
    panel(ax, "(a) Loss worse, format better")
    axb = ax.twinx()
    r = axb.bar([1 - w / 2, 1 + w / 2], [100 * t for t in BEST_LAST_THINK], w, color=colors, **EDGE)
    bar_labels(axb, r, "{:.0f}%")
    axb.set_ylabel("<think> rate (%)")
    axb.set_ylim(0, 115)
    axb.spines["top"].set_visible(False)
    axb.grid(False)

    ax = axes[1]
    sel = json.load(open(DOCS / "checkpoint_selection.json", encoding="utf-8"))
    names, acc, think = [], [], []
    for path, m in sel["results"].items():
        names.append(Path(path).stem.replace("sft_ckpt_", ""))
        acc.append(100 * m["numerical_acc_5pct"])
        think.append(100 * m["numerical_think_rate"])
    win = Path(sel["selected"]).stem.replace("sft_ckpt_", "")
    x, w = np.arange(len(names)), 0.4
    r1 = ax.bar(x - w / 2, acc, w, label="accuracy @5%", color=COLOR_CONCEPT, **EDGE)
    r2 = ax.bar(x + w / 2, think, w, label="<think> rate", color=COLOR_FORMAT, **EDGE)
    for r in (r1, r2):
        bar_labels(ax, r, "{:.1f}", fontsize=11.5)
    ticks = [{"best": "lowest\nval loss"}.get(n, n.replace("epoch", "epoch ")) for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels([t + ("\n(selected)" if n == win else "") for t, n in zip(ticks, names)])
    ax.set_ylabel("held-out rate (%)")
    ax.set_ylim(0, 100)
    legend(ax, loc="upper left")
    panel(ax, "(b) Val loss picks the worst one")

    ax = axes[2]
    v3 = load_grpo_log("grpo")
    ax.plot(v3["step"], rolling_mean(v3["mean_reward"], 20), color=COLOR_REWARD, linewidth=2.4)
    res = json.load(open(DOCS / "eval_rl_results.json", encoding="utf-8"))
    for step, key, col in ((100, "GRPO @100 (selected)", COLOR_FIX), (200, "GRPO @200 (final)", COLOR_TRAP)):
        ax.axvline(step, color=col, linestyle="--", linewidth=1.6)
        ax.text(step - 4, -0.03, f"step {step}\n<think> {res[key]['numerical_think_rate']:.0%}", ha="right",
                va="top", fontsize=12, fontweight="bold", color=col)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("training reward (rolling mean)")
    ax.set_ylim(-0.9, 0.0)
    panel(ax, "(c) The last step is not best")

    finish(fig, "E1 — Validation loss picks the wrong checkpoint", "pitfall_6_checkpoint_selection")


# ---------------------------------------------------------------------------
# F1 -- confounded scaling law
# ---------------------------------------------------------------------------

def fig_scaling():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)

    ax = axes[0]
    slope = {}
    for N, L, c, name in ((CONFOUNDED_N, CONFOUNDED_LOSS, COLOR_TRAP, "drifting mixture"),
                          (MATCHED_N, MATCHED_LOSS, COLOR_FIX, "fixed mixture")):
        lg = np.log2(N)
        a, b = np.polyfit(lg, L, 1)
        slope[name] = a
        xs = np.linspace(lg.min() - 0.3, lg.max() + 0.3, 50)
        ax.plot(2 ** xs, a * xs + b, "--", color=c, linewidth=1.8, alpha=0.85)
        ax.plot(N, L, "o", color=c, markersize=9, markeredgecolor=INK, markeredgewidth=1.0,
                label=f"{name}: {a:+.3f}".replace("-", "\u2212"))
    ax.set_xscale("log", base=2)
    ax.set_xticks([2048, 4096, 8192, 16384])
    ax.set_xticklabels(["2k", "4k", "8k", "16k"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("training pairs (log scale)")
    ax.set_ylabel("held-out loss (nats)")
    ax.set_ylim(2.08, 2.66)
    legend(ax, loc="upper right")
    panel(ax, f"(a) Fixed mixture: {slope['fixed mixture'] / slope['drifting mixture'] - 1:.0%} steeper")

    ax = axes[1]
    pts = list(MIXTURE)
    bottom = np.zeros(len(pts))
    for k, (name, col) in enumerate((("paper", COLOR_CONCEPT), ("fundamentals", COLOR_FORMAT),
                                     ("earthquakes", COLOR_CLEAN))):
        share = 100 * np.array([MIXTURE[p][k] for p in pts])
        ax.bar(pts, share, 0.6, bottom=bottom, label=name, color=col, **EDGE)
        bottom += share
    ax.set_ylabel("share of corpus (%)")
    ax.set_xlabel("corpus size (pairs)")
    ax.set_ylim(0, 135)
    ax.set_yticks([0, 25, 50, 75, 100])
    legend(ax, loc="upper center", ncol=2)
    panel(ax, "(b) Size and mixture co-varied")

    ax = axes[2]
    doms = list(PER_DOMAIN)
    vals = np.array([PER_DOMAIN[d][0] for d in doms])
    lo = np.array([PER_DOMAIN[d][1][0] for d in doms])
    hi = np.array([PER_DOMAIN[d][1][1] for d in doms])
    cols = {"paper": COLOR_CONCEPT, "earthquakes": COLOR_CLEAN, "fundamentals": COLOR_FORMAT}
    r = ax.bar(doms, vals, 0.6, yerr=np.array([vals - lo, hi - vals]), capsize=5,
               color=[cols[d] for d in doms], **EDGE)
    bar_labels(ax, r, "{:+.3f}", below=lo)
    ax.axhline(0, color=INK, linewidth=1.2)
    ax.set_ylabel("Δ loss, 8.0k → 14.1k (nats)")
    ax.set_ylim(-0.215, 0.01)
    ax.set_yticks([0, -0.05, -0.10, -0.15, -0.20])
    panel(ax, "(c) Every domain improves")

    finish(fig, "F1 — A scaling law confounded by the data mixture", "pitfall_7_confounded_scaling")


# ---------------------------------------------------------------------------
# C1 / D2 -- the metric's floor exceeded the model
# ---------------------------------------------------------------------------

def fig_metric_validity():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=300)

    # (a) the gameability curve that made the floor exceed the model
    ax = axes[0]
    ax.plot(SPRAY_K, [100 * v for v in SPRAY_ACC5_OLD], "o-", color=COLOR_TRAP, linewidth=2.4,
            markersize=8, markeredgecolor=INK, label="blind number spray")
    ax.axhline(100 * OLD_GRPO, color=COLOR_CONCEPT, linestyle="--", linewidth=2.0,
               label=f"the model ({OLD_GRPO:.1%})")
    ax.set_xlabel("numbers in the answer (k)")
    ax.set_ylabel("accuracy @5% (%)")
    ax.set_ylim(0, 20)
    legend(ax, loc="upper left")
    panel(ax, "(a) Guessing beats the model")

    # (b) floor / model / ceiling, before and after the scoring fix
    ax = axes[1]
    x, w = np.arange(2), 0.27
    for off, vals, name, col in ((-w, (OLD_FLOOR, NEW_FLOOR), "floor (blind spray)", COLOR_TRAP),
                                 (0, (OLD_GRPO, NEW_GRPO), "model", COLOR_CONCEPT),
                                 (w, (OLD_CEIL, NEW_CEIL), "ceiling (ground truth)", COLOR_FIX)):
        r = ax.bar(x + off, [100 * v for v in vals], w, label=name, color=col, **EDGE)
        bar_labels(ax, r, "{:.1f}", fontsize=11.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["as reported:\nfloor > model", "corrected:\nmodel > floor"])
    ax.set_ylabel("accuracy @5% (%)")
    ax.set_ylim(0, 112)
    legend(ax, loc="upper left")
    panel(ax, "(b) The fix restored the range")

    # (c) the reward-hacking channel, closed
    ax = axes[2]
    keys = ["reference (genuine)", "verbose filler + correct answer"]
    x, w = np.arange(2), 0.4
    genuine = [PROBE_BEFORE[keys[0]], PROBE_AFTER[keys[0]]]
    filler = [PROBE_BEFORE[keys[1]], PROBE_AFTER[keys[1]]]
    r1 = ax.bar(x - w / 2, genuine, w, label="genuine answer", color=COLOR_FIX, **EDGE)
    r2 = ax.bar(x + w / 2, filler, w, label="filler + correct answer", color=COLOR_TRAP, **EDGE)
    for r in (r1, r2):
        bar_labels(ax, r, "{:.2f}")
    ax.annotate(f"{filler[0] - genuine[0]:+.2f}", xy=(w / 2, filler[0]), xytext=(0, 22), textcoords="offset points",
                ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=COLOR_TRAP)
    ax.set_xticks(x)
    ax.set_xticklabels(["before fix", "after fix"])
    ax.set_ylabel("composite reward")
    ax.set_ylim(0, 1.1)
    legend(ax, loc="upper left")
    panel(ax, "(c) Padding paid, until fixed")

    finish(fig, "C1 / D2 — A metric whose floor beat the model", "pitfall_8_metric_validity")


FIGURES = {
    1: ("Train/test leakage", fig_leakage),
    2: ("Circular evaluation", fig_circular),
    3: ("Evaluation protocol", fig_protocol),
    4: ("Decoding artifacts", fig_decoding),
    5: ("Silent failures", fig_silent),
    6: ("Checkpoint selection", fig_selection),
    7: ("Confounded scaling", fig_scaling),
    8: ("Metric validity", fig_metric_validity),
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


if __name__ == "__main__":
    main()
