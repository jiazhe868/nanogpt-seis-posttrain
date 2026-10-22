"""Phase III figures.

    assets/grpo_training_dynamics    4 panels for the current (v4) run: training +
                                     held-out reward, reward components, KL with
                                     beta and entropy, completion length and stop rate
    assets/grpo_training_dynamics_v3 the same 4 panels for the shipped v3 run
    assets/grpo_benchmark_comparison every phase on the held-out numerical questions, new problems
                                     on the fine-tuning papers, key concepts in conceptual answers
                                     and <think> format (95% CIs)
    assets/general_model_comparison  the deployed model against general Qwen models of every size, one
                                     executed answer each; key concepts at equal length
                                     (docs/eval_general_models.json)
    assets/grpo_stability_v3_vs_v4   the shipped v3 run against the v4 trainer:
                                     reward (train + held-out), KL and beta,
                                     termination, reward opportunity
    assets/grpo_beta_intervention    v3 restarted from its step-100 weights with
                                     only beta changed (0.192 vs 0.035)
    assets/grpo_training_dynamics_v5 the 4 panels for v5 (TRL num_iterations=2, clip-higher,
                                     TIS), x-axis in optimizer updates
    assets/grpo_speed_v4b_vs_v5      held-out objective against wall time and against
                                     updates, KL, and where each run's time went (+ MFU)
    assets/numerical_diagnosis       oracle prefill ladder (how much of a correct derivation
                                     the model needs) and binding transfer across question
                                     wordings (docs/numerical_diagnosis.json)
    assets/rl_transfer_diagnosis     where GRPO's gain goes: trained problems vs other problems
                                     from the same source chunks vs unseen chunks; program
                                     validity vs setup; KL and log-prob shifts per set
                                     (docs/rl_transfer_diagnosis.json)
    assets/numerical_track           real held-out accuracy at every stage of the numerical track
                                     (6 seeds x 4, 95% CIs, vote@8 markers, teacher ceiling)
    assets/rl_curriculum             simpler questions as stepping stones: the learnable band
                                     for original vs simple questions, and never-solved problems
                                     lifted by practising their first step

Both read the run log and results file at plot time and warn when their
provenance disagrees.

    python -m src.plots grpo
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
    COLOR_LOSS, COLOR_MUTED, COLOR_REWARD, COLOR_SFT, COLOR_TRAP, DOCS, INK,
    legend, nice_ylim, panel, rolling_mean, save, set_publication_style)

ROOT = Path(__file__).resolve().parents[2]
LOG_CSV = ROOT / "checkpoints" / "runs" / "grpo.csv"
V4_CSV = ROOT / "checkpoints" / "runs" / "grpo_v4.csv"
V4_EVAL_CSV = ROOT / "checkpoints" / "runs" / "grpo_v4_eval.csv"
INTERVENTION = {beta: ROOT / "checkpoints" / "runs" / f"grpo_v3_intervention_beta{tag}.csv"
                for beta, tag in (("0.192", "0192"), ("0.035", "0035"))}
EVAL_JSON = ROOT / "docs" / "eval_rl_results.json"          # archived k=1 run (pitfall figures read it)
BENCH_JSON = ROOT / "docs" / "eval_rl_v2labels_k4.json"   # k=4, per-item, corrected (v2) labels
RUNS = ROOT / "checkpoints" / "runs"
# v4b: the v4 algorithm on corrected labels with the CUDA-graph decoder; v5: + TRL rollout reuse
SPEED_RUNS = {  # name: (training log, held-out log, optimizer updates per rollout batch)
    "v4b": (RUNS / "grpo_v4b.csv", RUNS / "grpo_v4b_eval.csv", 1),
    "v5": (RUNS / "grpo_v5.csv", RUNS / "grpo_v5_eval.csv", 2),
}
# MFU accounting (README §5.15): 2N FLOPs per token forward, A30 dense bf16 peak per GPU.
N_PARAMS, PEAK_FLOPS, N_GPUS = 113.3e6, 165e12, 2
MEAN_PROMPT_TOKENS = 106      # ChatML prompt length, mean over data/rl/train.jsonl
DIAG_JSON = ROOT / "docs" / "numerical_diagnosis.json"
TRANSFER_JSON = ROOT / "docs" / "rl_transfer_diagnosis.json"





def load_grpo_logs(log_csv: Path = LOG_CSV):
    """Load GRPO training metrics from CSV into a dict of numpy arrays.

    `stop_rate` only exists in v3+ logs (it is the TerminationVerifier
    diagnostic); it comes back as all-NaN for older runs and the panel that
    uses it degrades gracefully.
    """
    cols = ["step", "mean_reward", "num_reward", "concept_reward", "format_reward",
            "clean_reward", "policy_loss", "kl_div", "kl_coeff", "entropy",
            "mean_comp_tokens", "stop_rate"]
    data = {c: [] for c in cols}

    with open(log_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("step"):
                continue
            for c in cols:
                data[c].append(float(row[c]) if row.get(c) not in (None, "") else float("nan"))

    return {c: np.array(v) for c, v in data.items()}




def fig_grpo_training_dynamics(log_csv: Path = V4_CSV, eval_csv: Path | None = V4_EVAL_CSV,
                               name: str = "grpo_training_dynamics", updates_per_step: int = 1):
    """Generate 4-panel figure evaluating GRPO training convergence.

    With `eval_csv` (v4+ runs), panel (a) also carries the held-out objective —
    the curve to read for convergence, since the per-step training reward is a
    batch sample (PITFALLS F3). The KL-target line is drawn only when beta
    actually moved; a fixed-beta run is labelled as such instead.

    `updates_per_step` > 1 (v5: TRL num_iterations = 2) puts the x-axis in optimizer updates,
    so runs that reuse rollouts line up with v4's one-update-per-batch axis. Rolling windows
    stay 20 rollout batches.
    """
    if not log_csv.exists():
        print(f"[SKIP] {log_csv.name} not found — {name} not rendered.")
        return
    logs = load_grpo_logs(log_csv)
    held_out = _read_csv(eval_csv) if eval_csv is not None and eval_csv.exists() else None
    tag = f"[log: {log_csv.name}" + (f", {eval_csv.name}]" if held_out is not None else "]")
    mu = max(1, int(updates_per_step))
    steps = logs["step"] * mu
    xlabel = "GRPO Optimization Step" if mu == 1 else f"Optimizer Update ({mu} per rollout batch)"
    win_label = "20-Step" if mu == 1 else "20-Batch"
    mean_rewards = logs["mean_reward"]
    num_rewards = logs["num_reward"]
    concept_rewards = logs["concept_reward"]
    format_rewards = logs["format_reward"]
    kl_divs = logs["kl_div"]
    entropies = logs["entropy"]
    comp_tokens = logs["mean_comp_tokens"]
    stop_rates = logs["stop_rate"]
    has_stop = bool(np.isfinite(stop_rates).any())

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

    # -------------------------------------------------------------------------
    # Subplot (a): Total Mean Reward & Rolling Average Trend
    # -------------------------------------------------------------------------
    ax1.plot(steps, mean_rewards, color=COLOR_REWARD, alpha=0.35, linewidth=1.0, label="per step")
    rolling_r = rolling_mean(mean_rewards, window=20)
    ax1.plot(steps, rolling_r, color=COLOR_REWARD, linewidth=2.5, label=f"{win_label} mean")
    if held_out is not None:
        ax1.plot(held_out["step"] * mu, held_out["mean_reward"], "o-", color=INK, linewidth=1.6, ms=3.5,
                 label="held-out")
    ax1.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.2)

    ax1.set_title("(a) Reward", fontsize=15, fontweight="bold", pad=14, color="#2C3E50")
    ax1.set_xlabel(xlabel, fontsize=14.2, color="#2C3E50")
    ax1.set_ylabel("Mean Reward $R(y)$", fontsize=14.2, color="#2C3E50")
    ax1.set_ylim(*nice_ylim(mean_rewards, rolling_r,
                            held_out["mean_reward"] if held_out is not None else [], floor=0.0, ceil=0.0))
    ax1.grid(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=12.2, loc="lower right")

    # Annotate peak rolling reward
    peak_idx = int(np.argmax(rolling_r))

    # -------------------------------------------------------------------------
    # Subplot (b): Sub-Reward Breakdown
    # -------------------------------------------------------------------------
    r_num = rolling_mean(num_rewards, 20)
    r_con = rolling_mean(concept_rewards, 20)
    r_fmt = rolling_mean(format_rewards, 20)
    ax2.plot(steps, r_num, color=COLOR_LOSS, linewidth=2.2, label="numerical")
    ax2.plot(steps, r_con, color=COLOR_CONCEPT, linewidth=2.2, label="concept")
    ax2.plot(steps, r_fmt, color=COLOR_FORMAT, linewidth=2.2, label="format")
    ax2.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.2)

    ax2.set_title("(b) Reward components", fontsize=15, fontweight="bold", pad=14, color="#2C3E50")
    ax2.set_xlabel(xlabel, fontsize=14.2, color="#2C3E50")
    ax2.set_ylabel("Reward Sub-score", fontsize=14.2, color="#2C3E50")
    ax2.set_ylim(*nice_ylim(r_num, r_con, r_fmt, floor=0.0, ceil=0.0))
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=12.2, loc="center",
               bbox_to_anchor=(0.5, 0.42), ncol=3)

    # -------------------------------------------------------------------------
    # Subplot (c): Reference Model KL Divergence Anchor
    # -------------------------------------------------------------------------
    kl_roll = rolling_mean(kl_divs, 20)
    ax3.plot(steps, kl_divs, color=COLOR_KL, alpha=0.35, linewidth=1.0, label="KL per token")
    ax3.plot(steps, kl_roll, color=COLOR_KL, linewidth=2.5, label=f"KL, {win_label} mean")
    betas = logs["kl_coeff"][np.isfinite(logs["kl_coeff"])]
    if betas.size and np.ptp(betas) > 1e-9:
        ax3.axhline(0.05, color="#E74C3C", linestyle="--", linewidth=1.2,
                    label=f"Adaptive-$\\beta$ KL Target (0.05); $\\beta$ {betas.min():.3f}–{betas.max():.3f}")
    elif betas.size:
        ax3.plot([], [], " ", label=f"Fixed $\\beta$ = {betas[0]:.3f} (no controller)")

    ax3.set_title("(c) KL and entropy", fontsize=15, fontweight="bold", pad=14, color="#2C3E50")
    ax3.set_xlabel(xlabel, fontsize=14.2, color="#2C3E50")
    ax3.set_ylabel("KL Divergence $D_{KL}(\\pi_\\theta \\parallel \\pi_{ref})$", fontsize=14.2, color=COLOR_KL)
    ax3.set_ylim(*nice_ylim(kl_divs, floor=0.0, ceil=0.06))
    ax3.grid(True)
    ax3.spines["top"].set_visible(False)

    # Entropy shares this panel: KL measures drift from the reference, entropy
    # measures collapse of the policy itself. Reading them together separates
    # "moved somewhere new" from "stopped exploring".
    ax3b = ax3.twinx()
    ax3b.plot(steps, rolling_mean(entropies, 20), color=COLOR_LOSS,
              linewidth=2.2, label="entropy (nats)")
    # At least a 0.5-nat span: autoscaling a +-0.07-nat wobble to full height
    # makes a flat entropy curve read as instability.
    ent_roll = rolling_mean(entropies, 20)
    mid, half = (np.nanmax(ent_roll) + np.nanmin(ent_roll)) / 2, max(0.25, 0.6 * np.nanmax(ent_roll) - 0.6 * np.nanmin(ent_roll))
    ax3b.set_ylim(mid - half, mid + half)
    ax3b.set_ylabel("Policy Entropy (nats)", fontsize=14.2, color=COLOR_LOSS)
    ax3b.spines["top"].set_visible(False)

    l3a, lab3a = ax3.get_legend_handles_labels()
    l3b, lab3b = ax3b.get_legend_handles_labels()
    ax3.legend(l3a + l3b, lab3a + lab3b, frameon=True, facecolor="white", edgecolor="#BDC3C7",
               fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    # -------------------------------------------------------------------------
    # Subplot (d): Termination behaviour — completion length & stop rate.
    # stop_rate is the fraction of rollouts that emitted <|im_end|> instead of
    # being cut off at the token budget. It is the metric the V3
    # TerminationVerifier was added to move, so it gets its own axis rather
    # than being inferred from length.
    # -------------------------------------------------------------------------
    len_roll = rolling_mean(comp_tokens, 20)
    ax4.plot(steps, len_roll, color=COLOR_CONCEPT, linewidth=2.2, label="length (tokens)")
    ax4.set_title("(d) Length and stop rate", fontsize=15, fontweight="bold", pad=14, color="#2C3E50")
    ax4.set_xlabel(xlabel, fontsize=14.2, color="#2C3E50")
    ax4.set_ylabel("Mean Completion Tokens", fontsize=14.2, color=COLOR_CONCEPT)
    ax4.grid(True)
    ax4.spines["top"].set_visible(False)

    ax4b = ax4.twinx()
    if has_stop:
        ax4b.plot(steps, rolling_mean(stop_rates, 20), color=COLOR_CLEAN,
                  linewidth=2.2, label="stop rate")
        ax4b.set_ylim(0, 1.02)
    else:
        ax4b.text(0.5, 0.5, "stop_rate not logged in this run",
                  transform=ax4b.transAxes, ha="center", va="center",
                  fontsize=12.2, color="#7F8C8D", style="italic")
    ax4b.set_ylabel("Rollout Stop Rate", fontsize=14.2, color=COLOR_CLEAN)
    ax4b.spines["top"].set_visible(False)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, frameon=True, facecolor="white", edgecolor="#BDC3C7",
               fontsize=12.2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)


    plt.tight_layout()
    save(fig, name)


def _boot_ci(values, rng, n_boot=2000):
    """95% percentile bootstrap CI of a mean over prompts."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


HEADLINE_JSON = DOCS / "eval_numerical_real6_final.json"
INDIST_JSON = DOCS / "eval_indist.json"
CEILINGS_JSON = DOCS / "eval_ceilings.json"
GENERAL_JSON = DOCS / "eval_general_models.json"
HEADLINE = [  # (key in HEADLINE_JSON, label, color, use vote@8, key in INDIST_JSON, concept rows in CEILINGS_JSON)
    ("SFT", "SFT", COLOR_SFT, False, "SFT", "sft"),
    ("v4@300", "+ GRPO", COLOR_KL, False, "v4@300", "grpo_v4"),
    ("notation2 (recovered data)", "executed equations", COLOR_CONCEPT, False, "notation2", "notation2_prose"),
    ("notation2 + GRPO v14 @75 (corpus, dev-selected)", "+ GRPO on the corpus", COLOR_REWARD, False, "v14-75",
     "deployed_prose"),
    ("notation2 + GRPO v14 @75 (corpus, dev-selected)", "+ vote@8 (deployed)", COLOR_FORMAT, True, "v14-75",
     "deployed_routed"),
]


def _concept_items(rows, key="coverage"):
    """Per-question key-concept coverage, averaged over that question's samples."""
    per = {}
    for r in rows:
        per.setdefault(r["id"], []).append(r[key])
    return {i: {"coverage": float(np.mean(v))} for i, v in per.items()}


def _coverage(text, key_concepts):
    text = text.lower()
    return sum(k.lower() in text for k in key_concepts) / len(key_concepts)


def _headline_items(runs, tol_vote=None):
    """Per-question acc@5%, acc@1% and <think> rate over all seeds; or vote@8 at a tolerance."""
    sys.path.insert(0, str(ROOT))
    from src.rl.self_consistency import select_consensus
    from src.rl.verifiers import NumericalVerifier
    tail = NumericalVerifier.ANSWER_TAIL_NUMBERS
    per = {}
    for r in runs:
        for x in r["per_item"]:
            if x["type"] != "numerical":
                continue
            d = per.setdefault(x["id"], {"acc5": [], "acc1": [], "think": [], "target": x["target"], "tails": []})
            d["acc5"].append(x["acc5"]); d["acc1"].append(x["acc1"]); d["think"].append(x["think"])
            d["tails"].extend(a[-tail:] for a in x.get("answer_numbers", []))
    out = {}
    for i, d in per.items():
        row = {k: float(np.mean(d[k])) for k in ("acc5", "acc1", "think")}
        if tol_vote is not None:
            t = d["target"]
            def hit(a, tol):
                den = abs(t) if abs(t) > 1e-7 else 1.0
                return bool(a) and any(abs(v - t) / den <= tol for v in a)
            subs = [d["tails"][j:j + 8] for j in range(0, len(d["tails"]) - 7, 8)]
            row["acc5"] = float(np.mean([hit(sb[select_consensus(sb)], 0.05) for sb in subs]))
            row["acc1"] = float(np.mean([hit(sb[select_consensus(sb)], 0.01) for sb in subs]))
        out[i] = row
    return out


def _in_prompt_fn():
    """test.jsonl by id, and a test for held-out numerical questions whose answer is already in the question."""
    sys.path.insert(0, str(ROOT))
    from src.analysis.tier_vote import close
    from src.rl.verifiers import NumericalVerifier
    nv = NumericalVerifier()
    test = {it["id"]: it for it in map(json.loads, open(ROOT / "data" / "rl" / "test.jsonl", encoding="utf-8"))}

    def in_prompt(i):
        it = test[i]
        t = float(it["final_numerical_value"])
        pn = nv.extract_numbers(it["prompt"])
        return bool(pn) and any(close(p, t) for p in pn)
    return in_prompt, test


def _stats(values, rng):
    """Mean and 95% bootstrap whiskers (distances below and above) of per-question values, in percent."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = _boot_ci(v, rng)
    m = v.mean()
    return 100 * m, 100 * (m - lo), 100 * (hi - m)


def _series(groups, srcs, rng):
    vals, err = [], [[], []]
    for (_, metric, keep), src in zip(groups, srcs):
        m, lo, hi = _stats([src[i][metric] for i in src if keep(i)], rng)
        vals.append(m); err[0].append(lo); err[1].append(hi)
    return vals, err


def _label_color(color):
    """The series colour for a value label, or ink when the colour is too light to read on white."""
    r, g, b = (int(color.lstrip("#")[k:k + 2], 16) / 255 for k in (0, 2, 4))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.6 else color


def _bar_groups(ax, series, idx, names, pad):
    """Grouped bars for groups `idx` of every series, value labels above the whiskers."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(idx))
    for k, (label, color, vals, err, bold) in enumerate(series):
        off = (k - (n - 1) / 2) * width
        v = [vals[g] for g in idx]
        e = [[err[0][g] for g in idx], [err[1][g] for g in idx]]
        ax.bar(x + off, v, width, color=color, edgecolor=INK, lw=1.6 if bold else 0.9, label=label,
               yerr=e, error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=2.5))
        for xi, vv, h in zip(x + off, v, e[1]):
            if np.isfinite(vv):                            # plotted in percent, so bar labels need no % sign
                ax.text(xi, vv + h + pad, f"{vv:.0f}" if vv >= 50 else f"{vv:.1f}",
                        ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=_label_color(color))
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13.5, fontweight="bold", color=INK)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_grpo_benchmark_comparison():
    """Headline comparison by phase (README §1.2).

    Numerical: the 153 real held-out questions, 6 seeds x 4 samples (docs/eval_numerical_real6_final.json),
    as acc@5% and answers within 1% (the 134 that must be computed are in the README table); and acc@5% on 300
    new teacher problems about the fine-tuning papers (docs/eval_indist.json). Conceptual: the share of each
    of the 641 non-numerical held-out prompts' key concepts named in the visible answer, 2 samples, prose
    decoding (docs/eval_ceilings.json; the deployed chat routes these questions to prose). Then <think>
    compliance. Whiskers are 95% bootstrap CIs over questions. The deployed configuration (executed-equation
    student + GRPO on the corpus pool + vote@8) is drawn in purple. Contamination and prompt-echo are 0%.
    """
    if not HEADLINE_JSON.exists():
        print(f"[SKIP] {HEADLINE_JSON.name} not found — headline comparison not rendered.")
        return
    _, test = _in_prompt_fn()
    res = json.load(open(HEADLINE_JSON, encoding="utf-8"))["results"]
    indist = json.load(open(INDIST_JSON, encoding="utf-8"))["results"] if INDIST_JSON.exists() else {}
    concept = json.load(open(CEILINGS_JSON, encoding="utf-8"))["concept"] if CEILINGS_JSON.exists() else {}
    conceptual = [it for it in test.values() if it["type"] != "numerical" and it.get("key_concepts")]

    groups = [("within 5%", "acc5", lambda i: True),
              ("within 1%", "acc1", lambda i: True),
              ("<think> format", "think", lambda i: True),
              ("within 5%", "acc5", lambda i: True),                     # the in-distribution set
              ("key concepts\nnamed", "coverage", lambda i: True)]       # the conceptual questions
    rng = np.random.default_rng(20261017)
    series = []
    for key, label, color, vote, ikey, ckey in HEADLINE:
        if key not in res:
            continue
        rows = _headline_items(res[key], 0.05 if vote else None)
        irows = _headline_items(indist[ikey], 0.05 if vote else None) if ikey in indist else {}
        crows = _concept_items(concept[ckey]) if ckey in concept else {}
        series.append((label, color, *_series(groups, (rows, rows, rows, irows, crows), rng), vote))
    fig, (ax, axm, axc, ax2) = plt.subplots(1, 4, figsize=(16.5, 6.0), dpi=300,
                                            gridspec_kw={"width_ratios": [2.3, 1.0, 1.35, 0.9]})
    for panel_ax, idx, pad in ((ax, [0, 1], 0.6), (axm, [3], 0.6), (axc, [4], 0.6), (ax2, [2], 1.2)):
        _bar_groups(panel_ax, series, idx, [groups[g][0] for g in idx], pad)
    ax.set_ylim(0, 50)
    ax.set_ylabel("accuracy (%)", fontsize=14.2, color=INK)
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=11.9, loc="upper center", ncol=2)
    ax.set_title("Held-out questions (153)", fontsize=15.6, fontweight="bold", pad=10, color=INK)
    axm.set_ylim(0, 50)
    axm.set_title("New problems on\nfine-tuning papers (300)", fontsize=15.6, fontweight="bold", pad=10, color=INK)
    floor = 100 * np.mean([_coverage(it["prompt"], it["key_concepts"]) for it in conceptual])
    axc.axhline(floor, color=INK, ls="--", lw=1.2, zorder=3)
    axc.text(-0.48, 49.3, f"reference answers: 100\nquestion text alone: {floor:.1f} (dashed)",
             fontsize=10.5, color=INK, va="top")
    axc.set_xlim(-0.5, 0.5)
    axc.set_ylim(0, 50)
    axc.set_ylabel("key concepts in the answer (%)", fontsize=14.2, color=INK)
    axc.set_title(f"Conceptual\nquestions ({len(conceptual)})", fontsize=15.6, fontweight="bold", pad=10, color=INK)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("answers (%)", fontsize=14.2, color=INK)
    ax2.set_title("Answer format", fontsize=15.6, fontweight="bold", pad=10, color=INK)
    plt.tight_layout()
    png_path = ASSETS / "grpo_benchmark_comparison.png"
    pdf_path = ASSETS / "grpo_benchmark_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path.relative_to(ROOT)} and {pdf_path.relative_to(ROOT)}")


def fig_general_models():
    """The deployed 113M model against general Qwen models of every size, scored the same way (README §1.4).

    (a) The 153 real held-out questions: one executed answer per question, within 1% of the label (ours: its
    executed paper-notation equations; general models: one Python program each), split into question types the
    model has practised (nearest training problem with word-Jaccard >= 0.4) and the rest. (b) Key concepts named
    in the 641 conceptual answers when every answer is cut to the length of ours on the same prompt.
    Data: docs/eval_general_models.json (scripts/experiments/general_models_compare.py). Whiskers: 95% CIs.
    """
    if not GENERAL_JSON.exists():
        print(f"[SKIP] {GENERAL_JSON.name} not found — general-model comparison not rendered.")
        return
    d = json.load(open(GENERAL_JSON, encoding="utf-8"))
    t, c = d["table"], d["concepts"]
    general = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.8-27B"]
    greys = ["#D5DBDB", "#AAB7B8", "#839192", "#5D6D7E", "#2E4053"]
    groups = ["all", "practised types", "other question types"]
    names = [f"all questions\n({t['all']['n']})", f"practised types\n({t['practised types']['n']})",
             f"other types\n({t['other question types']['n']})"]

    def whiskers(rows):
        return [[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]]
    ours = [t[g]["ours"]["within1_one_sample"] for g in groups]
    series = [("ours, 113M", COLOR_FORMAT, [r[0] for r in ours], whiskers(ours), True)]
    for name, col in zip(general, greys):
        rows = [t[g][name]["within1"] for g in groups]
        series.append((name + (" (teacher)" if "27B" in name else ""), col, [r[0] for r in rows], whiskers(rows), False))
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300, gridspec_kw={"width_ratios": [2.6, 1.3]})
    _bar_groups(ax, series, [0, 1, 2], names, 1.0)
    ax.set_ylim(0, 75)
    ax.set_ylabel("answers within 1% (%)", fontsize=14.2, color=INK)
    ax.set_title("Held-out questions, one executed answer each", fontsize=15.6, fontweight="bold", pad=10, color=INK)
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=11.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=3)
    cseries = [("ours, 113M", COLOR_FORMAT, [c["ours"]["same_length"][0]], whiskers([c["ours"]["same_length"]]), True)]
    for name, col in zip(general[:4], greys):
        if name in c:
            cseries.append((name, col, [c[name]["same_length"][0]], whiskers([c[name]["same_length"]]), False))
    _bar_groups(axc, cseries, [0], [f"answers cut to our length\n({c['ours']['words']:.0f} words)"], 0.6)
    axc.set_ylim(0, 50)
    axc.set_ylabel("key concepts named (%)", fontsize=14.2, color=INK)
    axc.set_title("Domain vocabulary (641)", fontsize=15.6, fontweight="bold", pad=10, color=INK)
    plt.tight_layout()
    png_path = ASSETS / "general_model_comparison.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(ASSETS / "general_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path.relative_to(ROOT)}")


def _read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("step")]
    cols = rows[0].keys() if rows else []
    return {c: np.array([float(r[c]) if r.get(c) not in (None, "") else np.nan for r in rows])
            for c in cols}


def ema(values, alpha: float = 0.1):
    """Bias-corrected exponential moving average (no flat back-fill at the start)."""
    out, m, w = np.empty(len(values)), 0.0, 0.0
    for i, v in enumerate(np.asarray(values, dtype=float)):
        m, w = (1 - alpha) * m + alpha * v, (1 - alpha) * w + alpha
        out[i] = m / w
    return out


def fig_grpo_stability_v3_vs_v4():
    """Same SFT start, same reward, same lr: the v3 trainer against the v4 one.

    v3 logged ONE prompt per step from rank 0, so its raw reward is plotted only
    as a 20-step rolling mean. v4 logs the mean over all 256 rollouts of the
    global batch, plus the held-out objective (expected reward under the
    sampling policy on the 275-prompt validation split, fixed seed).
    """
    if not (V4_CSV.exists() and V4_EVAL_CSV.exists()):
        print(f"[SKIP] {V4_CSV.name} / {V4_EVAL_CSV.name} not found — run the v4 config first.")
        return
    v3, v4, ev = _read_csv(LOG_CSV), _read_csv(V4_CSV), _read_csv(V4_EVAL_CSV)
    c2, c3 = COLOR_SFT, COLOR_REWARD
    n2 = "v3: 1 prompt/GPU, bang-bang β"
    n3 = f"v4: {int(v4['n_rollouts'][0]) // 8} prompts, fixed β"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=300)
    (a1, a2, a3), (a4, a5, a6) = axes

    # (a) training reward
    a1.plot(v3["step"], v3["mean_reward"], color=c2, alpha=0.18, lw=0.8)
    a1.plot(v3["step"], rolling_mean(v3["mean_reward"], 20), color=c2, lw=2.2, label="v3")
    a1.plot(v4["step"], v4["mean_reward"], color=c3, alpha=0.25, lw=0.8)
    a1.plot(v4["step"], ema(v4["mean_reward"]), color=c3, lw=2.4, label="v4")
    a1.set_xlabel("step"); a1.set_ylabel("training reward")
    a1.set_ylim(-1.05, 0.15)
    sd3 = float(np.std(v3["mean_reward"][50:] - rolling_mean(v3["mean_reward"], 20)[50:]))
    sd4 = float(np.std(v4["mean_reward"][50:] - ema(v4["mean_reward"])[50:]))
    panel(a1, "(a) Training reward")
    legend(a1, loc="lower right", fontsize=11.5)

    # (b) held-out objective
    a2.plot(ev["step"], ev["mean_reward"], "o-", color=c3, lw=2.2, ms=4, label="all prompts")
    a2.plot(ev["step"], ev["reward_numerical"], "s--", color=COLOR_LOSS, lw=1.4, ms=3, label="numerical items")
    a2.plot(ev["step"], ev["reward_conceptual"], "^--", color=COLOR_CONCEPT, lw=1.4, ms=3, label="conceptual")
    a2.set_xlabel("step"); a2.set_ylabel("held-out reward")
    panel(a2, "(b) v4 held-out reward")
    legend(a2, loc="lower right", fontsize=11)

    # (c) KL and beta
    a3.plot(v3["step"], rolling_mean(v3["kl_div"], 20), color=c2, lw=2.2, label="v3 KL")
    a3.plot(v4["step"], v4["kl_div"], color=c3, alpha=0.3, lw=0.8)
    a3.plot(v4["step"], ema(v4["kl_div"]), color=c3, lw=2.4, label="v4 KL")
    a3.set_xlabel("step"); a3.set_ylabel("KL per token")
    a3.set_ylim(0, None)
    a3b = a3.twinx()
    a3b.plot(v3["step"], v3["kl_coeff"], color=c2, lw=1.4, ls=":", label="v3 β")
    a3b.plot(v4["step"], v4["kl_coeff"], color=c3, lw=1.4, ls=":", label="v4 β")
    a3b.set_yscale("log"); a3b.set_ylabel("β (dotted)")
    a3b.spines["top"].set_visible(False)
    panel(a3, "(c) Drift from the SFT reference")
    h1, l1 = a3.get_legend_handles_labels(); h2, l2 = a3b.get_legend_handles_labels()
    legend(a3, handles=h1 + h2, labels=l1 + l2, loc="lower right", fontsize=11)

    # (d) termination
    a4.plot(v3["step"], rolling_mean(v3["stop_rate"], 20), color=c2, lw=2.2, label="v3")
    a4.plot(v4["step"], ema(v4["stop_rate"]), color=c3, lw=2.4, label="v4")
    a4.plot(ev["step"], ev["stop_rate"], "o", color=c3, mfc="white", ms=4, label="v4 held-out")
    a4.set_xlabel("step"); a4.set_ylabel("stop rate")
    a4.set_ylim(0, 1.02)
    panel(a4, "(d) Termination")
    legend(a4, loc="lower right", fontsize=11)

    # (e) format + numerical accuracy on held-out
    a5.plot(ev["step"], ev["think_rate"], "o-", color=COLOR_FORMAT, lw=2.0, ms=4, label="<think> format")
    a5.plot(ev["step"], ev["num_acc1"], "s-", color=COLOR_LOSS, lw=2.0, ms=4, label="numerical accuracy")
    a5.set_xlabel("step"); a5.set_ylabel("held-out rate")
    a5.set_ylim(0, 1.02)
    panel(a5, "(e) v4: format vs correctness")
    legend(a5, loc="center right", fontsize=11)

    # (f) reward opportunity + entropy
    a6.plot(v4["step"], ema(v4["zero_var_frac"]), color=COLOR_KL, lw=2.2, label="no-gradient groups")
    a6.plot(v4["step"], ema(v4["std_r_num"]), color=COLOR_LOSS, lw=1.6, label="std of numerical reward")
    a6.plot(v4["step"], ema(v4["std_r_format"]), color=COLOR_FORMAT, lw=1.6, label="std of format reward")
    a6.plot(v4["step"], ema(v4["std_r_term"]), color=COLOR_CLEAN, lw=1.6, label="std of stop reward")
    a6.set_xlabel("step"); a6.set_ylabel("fraction / std")
    a6.set_ylim(0, None)
    a6b = a6.twinx()
    a6b.plot(v4["step"], ema(v4["entropy"]), color=COLOR_MUTED, lw=1.4, ls="--", label="entropy (nats)")
    a6b.set_ylabel("entropy (dashed)")
    a6b.spines["top"].set_visible(False)
    panel(a6, "(f) v4: where gradient comes from")
    h1, l1 = a6.get_legend_handles_labels(); h2, l2 = a6b.get_legend_handles_labels()
    a6.set_ylim(0, 0.26)
    legend(a6, handles=h1 + h2, labels=l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=11)

    plt.tight_layout()
    save(fig, "grpo_stability_v3_vs_v4")


def fig_grpo_beta_intervention():
    """Was the post-step-100 regression caused by the beta jump?

    Both arms restart from the shipped v3 step-100 weights with the v3
    algorithm and the same prompt order and seed; only beta differs. 0.192 is
    what the bang-bang controller set at step 100 and then held; 0.035 is its
    mean over steps 81-100. Held-out numbers are the expected reward under the
    sampling policy on the 275 validation prompts (fixed seed).
    """
    paths = {b: (p, p.with_name(p.stem + "_eval.csv")) for b, p in INTERVENTION.items()}
    if not all(t.exists() and e.exists() for t, e in paths.values()):
        print("[SKIP] intervention logs not found — run configs/rl_grpo_intervention_beta*.yaml first.")
        return
    v3 = _read_csv(LOG_CSV)
    arms = {b: (_read_csv(t), _read_csv(e)) for b, (t, e) in paths.items()}
    colors = {"0.192": COLOR_TRAP, "0.035": COLOR_FIX}
    m = v3["step"] >= 100

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.6), dpi=300)
    for b, (tr, ev) in arms.items():
        x = 100 + ev["step"]
        a1.plot(x, ev["mean_reward"], "o-", color=colors[b], lw=2.0, ms=4, label=f"β = {b}")
        a2.plot(x, ev["stop_rate"], "o-", color=colors[b], lw=2.0, ms=4, label=f"β = {b}")
        a3.plot(100 + tr["step"], ema(tr["kl_div"], 0.2), color=colors[b], lw=2.0, label=f"β = {b}")
    a2.plot(v3["step"][m], rolling_mean(v3["stop_rate"], 20)[m], color=COLOR_SFT, lw=1.8, ls="--",
            label="original v3 run")
    a3.plot(v3["step"][m], rolling_mean(v3["kl_div"], 20)[m], color=COLOR_SFT, lw=1.8, ls="--",
            label="original v3 run")
    a1.set_ylabel("held-out reward")
    a2.set_ylabel("stop rate"); a2.set_ylim(0, 1.02)
    a3.set_ylabel("KL per token"); a3.set_ylim(0, None)
    for ax, title in ((a1, "(a) Held-out objective"), (a2, "(b) Termination"), (a3, "(c) KL to the SFT reference")):
        ax.set_xlabel("v3 step (restarted at 100)")
        panel(ax, title)
        legend(ax, loc="lower right" if ax is a1 else "lower left", fontsize=11)
    plt.tight_layout()
    save(fig, "grpo_beta_intervention")


def _mfu(log, mu: int):
    """(generation, update, whole-step) MFU of one run, from its logged timings."""
    rows, comp = np.nanmean(log["n_rollouts"]), np.nanmean(log["mean_comp_tokens"])
    gen, step = np.nanmean(log["gen_sec"]), np.nanmean(log["step_sec"])
    upd = np.nanmean(log["update_sec"]) if "update_sec" in log and np.isfinite(log["update_sec"]).any() else step - gen
    f = 2 * N_PARAMS
    g_fl = rows * comp * f
    seq = rows * (MEAN_PROMPT_TOKENS + comp)
    u_fl = seq * 4 * f + (mu - 1) * seq * 3 * f       # ref fwd + policy fwd/bwd, then fwd/bwd per reuse
    peak = PEAK_FLOPS * N_GPUS
    return g_fl / gen / peak, u_fl / upd / peak, (g_fl + u_fl) / step / peak


def fig_grpo_speed_v4b_vs_v5():
    """Does reusing each rollout batch (TRL num_iterations=2) buy speed without costing quality?

    v4b and v5 share labels, reward, decoder, lr and β; v5 adds two optimizer updates per
    rollout batch with the PPO ratio clipped to [0.8, 1.28] and truncated importance sampling.
    v4 (old labels, eager decoder) appears only in the time breakdown: its reward is on the v1
    labels and is not comparable.
    """
    if not all(t.exists() and e.exists() for t, e, _ in SPEED_RUNS.values()):
        print("[SKIP] v4b / v5 logs not found — run configs/rl_grpo_v4b.yaml and rl_grpo_v5.yaml first.")
        return
    runs = {n: (_read_csv(t), _read_csv(e), mu) for n, (t, e, mu) in SPEED_RUNS.items()}
    colors = {"v4": COLOR_SFT, "v4b": COLOR_CONCEPT, "v5": COLOR_REWARD}
    labels = {"v4b": "CUDA-graph decoding", "v5": "+ two updates per batch"}

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    for n, (tr, ev, mu) in runs.items():
        wall = np.interp(ev["step"], np.r_[0, tr["step"]], np.r_[0, tr["elapsed_sec"]]) / 60
        a1.plot(wall, ev["mean_reward"], "o-", color=colors[n], lw=2.0, ms=3.5, label=labels[n])
        a2.plot(ev["step"] * mu, ev["mean_reward"], "o-", color=colors[n], lw=2.0, ms=3.5, label=labels[n])
        a3.plot(tr["step"] * mu, tr["kl_div"], color=colors[n], alpha=0.25, lw=0.8)
        a3.plot(tr["step"] * mu, ema(tr["kl_div"]), color=colors[n], lw=2.2, label=labels[n])
        last = ev["step"] * mu > (ev["step"][-1] * mu - 100)
        a2.hlines(np.mean(ev["mean_reward"][last]), (ev["step"][-1] * mu) - 100, ev["step"][-1] * mu,
                  color=colors[n], lw=1.0, ls="--")
        a1.annotate(f"{wall[-1]:.1f} min", xy=(wall[-1], ev["mean_reward"][-1]), xytext=(4, -12),
                    textcoords="offset points", fontsize=11.5, color=colors[n], fontweight="bold")
    a1.set_xlabel("wall-clock minutes on 2× A30")
    a2.set_xlabel("optimizer update")
    a3.set_xlabel("optimizer update")
    for ax in (a1, a2):
        ax.set_ylabel("held-out reward")
        ax.axhline(0, color="#7F8C8D", ls=":", lw=1.0)
    a3.set_ylabel("KL per token")
    a3.set_ylim(0, None)
    panel(a1, "(a) Held-out reward vs time")
    panel(a2, "(b) Held-out reward vs updates")
    panel(a3, "(c) KL from the SFT model")
    legend(a1, loc="lower right", fontsize=11.5)
    legend(a2, loc="lower right", fontsize=11.5)
    legend(a3, loc="upper left", fontsize=11.5)

    # (d) where 300 updates' worth of time went
    rows = [("v4", _read_csv(V4_CSV), 1)] if V4_CSV.exists() else []
    rows += [(n, tr, mu) for n, (tr, _, mu) in runs.items()]
    names = []
    for i, (n, tr, mu) in enumerate(rows):
        gen = np.nansum(tr["gen_sec"]) / 60
        upd = (np.nansum(tr["update_sec"]) if "update_sec" in tr and np.isfinite(tr["update_sec"]).any()
               else np.nansum(tr["step_sec"] - tr["gen_sec"])) / 60
        wall = tr["elapsed_sec"][-1] / 60
        other = max(wall - gen - upd, 0.0)
        a4.barh(i, gen, color=colors[n], edgecolor=INK, lw=0.8, label="generation" if i == 0 else None)
        a4.barh(i, upd, left=gen, color=colors[n], alpha=0.45, edgecolor=INK, lw=0.8,
                hatch="//", label="policy update" if i == 0 else None)
        a4.barh(i, other, left=gen + upd, color="white", edgecolor=INK, lw=0.8,
                label="evals and checkpoints" if i == 0 else None)
        mg, mu_, ms = _mfu(tr, mu)
        a4.text(wall + 1.5, i, f"{wall:.0f} min, MFU {ms:.1%}",
                va="center", fontsize=11.5, color=INK)
        names.append({"v4": "original", "v4b": "CUDA graph", "v5": "+ 2 updates"}[n])
    a4.set_yticks(range(len(rows)))
    a4.set_yticklabels(names)
    a4.invert_yaxis()
    a4.set_xlabel("minutes for 300 updates")
    a4.set_xlim(0, max(r[1]["elapsed_sec"][-1] for r in rows) / 60 * 1.45)
    panel(a4, "(d) Where the time went")
    a4.grid(True, axis="x")
    legend(a4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=11.5)
    plt.tight_layout()
    save(fig, "grpo_speed_v4b_vs_v5")


def fig_numerical_diagnosis():
    """Where numerical answers fail, and what training fixes (audit §15.9).

    (a) Oracle ladder: the reference derivation is prefilled up to a fraction f and the model
        finishes it. Even at f = 1 (only the final answer left) it restates the result 39-53% of
        the time against an 86% instrument ceiling; the repetition penalty costs ~14 pp there.
    (b) Binding transfer: counterfactual and word-problem SFT raise accuracy on question
        wordings close to training (fresh numbers in trained questions; unseen phrasings of the
        generated families) by 15-66 pp, and on new real question types by nothing.
    """
    if not DIAG_JSON.exists():
        print(f"[SKIP] {DIAG_JSON.name} not found — numerical_diagnosis not rendered.")
        return
    d = json.load(open(DIAG_JSON, encoding="utf-8"))
    prov = d["_provenance"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4), dpi=300, gridspec_kw={"width_ratios": [1, 1.35]})
    styles = {"SFT (rp 1.2)": (COLOR_SFT, "-"), "SFT (rp 1.0)": (COLOR_SFT, "--"),
              "SFT+calc (calculator)": (COLOR_FORMAT, "-"), "GRPO v4": (COLOR_KL, "-")}
    names = {"SFT (rp 1.2)": "SFT", "SFT (rp 1.0)": "SFT, no repetition penalty",
             "SFT+calc (calculator)": "SFT + calculator", "GRPO v4": "first GRPO"}
    for name, rows in d["ladder"].items():
        xs = sorted(rows, key=float)
        c, ls = styles.get(name, (COLOR_MUTED, "-"))
        a1.plot([float(x) for x in xs], [rows[x]["must_compute"] for x in xs], marker="o", ms=4, lw=2.0,
                color=c, ls=ls, label=names.get(name, name))
    a1.axhline(prov["instrument_ceiling"]["must_compute"], color=COLOR_FIX, ls=":", lw=1.4,
               label=f"reference answers ({prov['instrument_ceiling']['must_compute']:.0%})")
    a1.axhline(prov["chance_floor_must_compute"], color=COLOR_TRAP, ls=":", lw=1.4,
               label=f"ignoring the question ({prov['chance_floor_must_compute']:.1%})")
    a1.set_xlabel("fraction of the reference derivation prefilled")
    a1.set_ylabel("accuracy, questions that need computation")
    a1.set_ylim(0, 1.0)
    panel(a1, "(a) Finishing a correct derivation")
    legend(a1, loc="upper left", fontsize=11)

    sets = [("cf-trained-types", "fresh numbers,\\ntrained questions"),
            ("wp-heldout-phrasing", "generated problems,\\nunseen phrasings"),
            ("cf-heldout-types", "fresh numbers,\\nheld-out questions"),
            ("real-heldout", "real held-out\\nquestions")]
    models = [("SFT", "SFT", COLOR_SFT), ("v4", "first GRPO", COLOR_KL),
              ("cf-e4 (counterfactual SFT)", "counterfactual SFT", COLOR_CONCEPT),
              ("wp2", "+ word problems", COLOR_REWARD), ("v7", "+ GRPO", COLOR_FORMAT)]
    models = [m for m in models if m[0] in d["transfer"]]
    x = np.arange(len(sets))
    w = 0.8 / len(models)
    for i, (key, label, c) in enumerate(models):
        vals = [d["transfer"][key].get(s, {}).get("mean", np.nan) for s, _ in sets]
        xi = x + (i - (len(models) - 1) / 2) * w
        a2.bar(xi, vals, w, color=c, edgecolor=INK, lw=0.8, label=label)
        for xx, v in zip(xi, vals):
            if np.isfinite(v):
                a2.text(xx, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=11, color=c, fontweight="bold")
    a2.set_xticks(x)
    a2.set_xticklabels([lab.replace("\\n", "\n") for _, lab in sets], fontsize=12.2)
    a2.set_ylabel("accuracy")
    a2.set_ylim(0, 0.8)
    panel(a2, "(b) Transfer across wording")
    legend(a2, loc="upper right", fontsize=11)
    plt.tight_layout()
    save(fig, "numerical_diagnosis")


# Real held-out accuracy at each stage of the numerical track (README §5.18): (results file in docs/,
# model key, label, kind). Every row was scored with the same 6 seeds x 4 samples on the 153 questions.
TRACK = [
    ("eval_numerical_real6_final.json", "SFT", "SFT\n(prose)", "sft"),
    ("eval_numerical_real6_final.json", "v4@300", "+ first GRPO\n(format reward)", "rl"),
    ("eval_numerical_real6.json", "pot-realcf", "executed\nPython", "sft"),
    ("eval_numerical_real6_decoder.json", "pot-v3 (rp1.0)", "+ real-style\nrewrites", "sft"),
    ("eval_numerical_real6_final.json", "notation", "paper\nnotation", "sft"),
    ("eval_numerical_real6_final.json", "notation + GRPO v11 @100", "paper notation\n+ GRPO", "rl"),
    ("eval_numerical_real6_final.json", "notation2 (recovered data)", "paper notation,\nmore data", "sft"),
    ("eval_numerical_real6_final.json", "notation2 + GRPO v12 @75 (dev-selected)", "+ GRPO,\ntraining papers", "rl"),
    ("eval_numerical_real6_final.json", "notation2 + GRPO v14 @75 (corpus, dev-selected)", "+ GRPO, whole\ncorpus (deployed)", "deploy"),
]
TEACHER_JSONL = DOCS / "teacher_27b_heldout_blind.jsonl"


def fig_numerical_track():
    """Real held-out accuracy at every stage of the numerical track (README §5.18).

    Bars: acc@5% (mean over 6 seeds x 4 samples) with 95% bootstrap CIs over the 153 questions;
    diamonds: vote@8 over executed or stated answers; dashed line: the 27B teacher solving blind.
    """
    sys.path.insert(0, str(ROOT))
    from src.analysis.tier_vote import pooled
    from src.rl.verifiers import NumericalVerifier
    rows, rng = [], np.random.default_rng(0)
    for fname, key, label, kind in TRACK:
        f = DOCS / fname
        if not f.exists():
            continue
        res = json.load(open(f, encoding="utf-8"))["results"]
        if key not in res:
            continue
        acc1, vote = pooled(res[key], 8, NumericalVerifier.ANSWER_TAIL_NUMBERS)
        a = np.array(list(acc1.values()))
        boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
        v = [x for x in vote.values() if x == x]
        rows.append((label, kind, a.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), float(np.mean(v))))
    if not rows:
        print("[SKIP] no 6-seed results — numerical_track not rendered.")
        return
    teacher = None
    if TEACHER_JSONL.exists():
        T = [json.loads(l) for l in open(TEACHER_JSONL, encoding="utf-8")]
        def close(x, y):
            d = abs(y) if abs(y) > 1e-7 else 1.0
            return abs(x - y) / d <= 0.05
        ok = 0
        for r in T:
            try:
                ok += close(float(r["answer2"]), float(r["answer"]))
            except (TypeError, ValueError):
                pass
        teacher = ok / len(T)
    colors = {"sft": COLOR_CONCEPT, "rl": COLOR_REWARD, "deploy": COLOR_FORMAT}
    fig, ax = plt.subplots(figsize=(15, 5.8), dpi=300)
    x = np.arange(len(rows))
    for i, (label, kind, m, lo, hi, v8) in enumerate(rows):
        ax.bar(i, m, 0.62, color=colors[kind], edgecolor=INK, lw=0.8, alpha=0.95 if kind != "deploy" else 1.0)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=4, lw=1.2)
        ax.scatter(i, v8, marker="D", s=46, color="white", edgecolor=colors[kind], lw=1.8, zorder=5)
        ax.text(i, max(hi, v8) + 0.012, f"{m:.1%}", ha="center", va="bottom", fontsize=12.2, fontweight="bold", color=INK)
    if teacher is not None:
        ax.axhline(teacher, color=COLOR_TRAP, ls="--", lw=1.4)
        ax.text(len(rows) - 0.5, teacher + 0.008, f"27B teacher {teacher:.1%}", ha="right", va="bottom",
                fontsize=12.2, color=COLOR_TRAP, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=12.2)
    ax.set_ylabel("accuracy (within 5%)")
    ax.set_ylim(0, 0.56)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=COLOR_CONCEPT, label="SFT stage"), Patch(color=COLOR_REWARD, label="+ GRPO"),
               Patch(color=COLOR_FORMAT, label="deployed model"),
               Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=INK, label="vote@8")]
    ax.legend(handles=handles, loc="upper left", fontsize=12.2, frameon=True)
    panel(ax, "Held-out accuracy at each step")
    plt.tight_layout()
    save(fig, "numerical_track")


def fig_rl_transfer():
    """Why GRPO's gain stays near the problems it trains on (audit §15.17).

    (a) GRPO on 576 problems (set A) against the SFT start: the trained problems, problems from
        92 other papers (B600; the stratum the start solves sometimes, and all), real held-out,
        and B600 after GRPO on other problems from B600's own papers.
    (b) The gain on a test question against its word-Jaccard similarity to the nearest trained
        question: the same curve within A and across papers.
    (c) Accuracy = P(program runs) x P(correct | it runs), plain sampling at T = 1. The program
        part transfers a little; the setup part (relation, constants, units) does not.
    (d) The policy moved as much on unseen prompts as on trained ones (KL per token) and pushed
        failing samples down there too, but the freed mass went to other wrong setups.
    """
    if not TRANSFER_JSON.exists():
        print(f"[SKIP] {TRANSFER_JSON.name} not found — rl_transfer_diagnosis not rendered.")
        return
    d = json.load(open(TRANSFER_JSON, encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    (a1, a2), (a3, a4) = axes
    t = d["transfer"]["grpo-lo-75"]
    bars = [("trained\nproblems", t["A-learnable"]), ("other papers,\nsometimes solved", t["B600 sometimes-solved"]),
            ("other\npapers", t["B600"]), ("real\nheld-out", t["real"])]
    if d.get("bsame"):
        m = max(d["bsame"], key=lambda k: d["bsame"][k]["B600"]["acc"])
        bars.append(("other papers,\nRL on same papers", d["bsame"][m]["B600"]))
    x = np.arange(len(bars))
    a1.bar(x - 0.2, [v["base_acc"] for _, v in bars], 0.4, color=COLOR_SFT, edgecolor=INK, lw=0.8, label="before")
    a1.bar(x + 0.2, [v["acc"] for _, v in bars], 0.4, color=[COLOR_REWARD] * 4 + [COLOR_FORMAT] * (len(bars) - 4),
           edgecolor=INK, lw=0.8, label="after GRPO")
    for i, (_, v) in enumerate(bars):
        a1.text(x[i] + 0.2, v["acc"] + 0.015, f"{100 * v['diff']:+.0f}", ha="center", va="bottom",
                fontsize=12.2, color=INK, fontweight="bold")
    a1.set_xticks(x)
    a1.set_xticklabels([lab for lab, _ in bars], fontsize=11.5)
    a1.set_ylabel("accuracy")
    a1.set_ylim(0, 1.0)
    panel(a1, "(a) Trained problems vs other papers")
    legend(a1, loc="upper right", fontsize=11)

    styles = [(COLOR_REWARD, "o", "-"), (COLOR_KL, "s", "--"), (COLOR_FORMAT, "^", ":")]
    centers = {}
    short = {0: "same papers, never solved", 1: "other papers", 2: "other papers, RL on same papers"}
    for k, ((name, rows), (c, mk, ls)) in enumerate(zip(d["similarity"].items(), styles)):
        name = short.get(k, name)
        xs = [i + (k - 1) * 0.08 for i in range(len(rows))]
        for xx, r in zip(xs, rows):
            centers[xx] = r["bucket"]
        ys = [100 * r["diff"] for r in rows]
        err = [[100 * (r["diff"] - r["ci"][0]) for r in rows], [100 * (r["ci"][1] - r["diff"]) for r in rows]]
        a2.errorbar(xs, ys, yerr=err, color=c, marker=mk, ls=ls, lw=2, ms=6, capsize=3, label=name)
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xticks(range(4))
    a2.set_xticklabels(["< 0.2", "0.2–0.35", "0.35–0.5", "≥ 0.5"])
    a2.set_xlabel("similarity to the nearest trained question")
    a2.set_ylabel("gain (percentage points)")
    panel(a2, "(b) Gain vs similarity to a trained problem")
    legend(a2, loc="upper left", fontsize=11)

    dec = d["decomposition_T1"]
    names = [s for s in ("A-learnable", "B600", "real") if s in dec]
    xs = np.arange(len(names))
    for j, (key, lab, c) in enumerate((("runs", "program runs", COLOR_FORMAT),
                                       ("acc_given_runs", "correct | runs", COLOR_CONCEPT))):
        b = [dec[s]["base"][key] for s in names]
        g = [dec[s]["grpo"][key] for s in names]
        off = (j - 0.5) * 0.4
        a3.bar(xs + off, b, 0.36, color="white", edgecolor=c, lw=1.6, hatch="//", label=f"{lab}: SFT")
        a3.bar(xs + off, g, 0.18, color=c, edgecolor=INK, lw=0.6, label=f"{lab}: GRPO")
        for xx, bb, gg in zip(xs + off, b, g):
            a3.text(xx, max(bb, gg) + 0.015, f"{bb:.0%}→{gg:.0%}", ha="center", va="bottom", fontsize=11, color=c, fontweight="bold")
    a3.set_xticks(xs)
    a3.set_xticklabels(["trained (A)", "other papers (B600)", "real"], fontsize=12.2)
    a3.set_ylim(0, 1.05)
    a3.set_ylabel("fraction of samples")
    panel(a3, "(c) Valid program vs right setup")
    legend(a3, loc="upper right", fontsize=11)

    groups = ["correct", "runs, wrong value", "crashes"]
    w = 0.25
    for j, (gname, c) in enumerate(zip(groups, [COLOR_CLEAN, COLOR_KL, COLOR_LOSS])):
        a4.bar(xs + (j - 1) * w, [dec[s]["dlogp_per_token"][gname] for s in names], w, color=c, edgecolor=INK, lw=0.7,
               label=f"SFT samples: {gname}")
    for i, s in enumerate(names):
        a4.text(xs[i], 0.006, f"KL {dec[s]['grpo']['kl_per_token']:.2f}", ha="center", va="bottom", fontsize=11, color=INK)
    a4.axhline(0, color=INK, lw=0.8)
    a4.set_xticks(xs)
    a4.set_xticklabels(["trained (A)", "other papers (B600)", "real"], fontsize=12.2)
    a4.set_ylabel("change in log-probability per token")
    a4.set_ylim(-0.16, 0.03)
    panel(a4, "(d) How the policy changed")
    legend(a4, loc="lower right", fontsize=11)
    plt.tight_layout()
    save(fig, "rl_transfer_diagnosis")

def fig_rl_curriculum():
    """Simpler questions as stepping stones (audit §15.17, README §5.19).

    (a) Where questions sit for RL: never solved (no gradient), learnable (1-7 of 8), always
        solved, for the original set-A problems and their one-step simplifications.
    (b) Never-solved original problems after GRPO on the simple questions: those whose own first
        step was practised rise ~7x; the rest move a little.
    """
    if not TRANSFER_JSON.exists():
        return
    d = json.load(open(TRANSFER_JSON, encoding="utf-8")).get("simple_questions")
    if not d:
        print("[SKIP] no simple_questions section — rl_curriculum not rendered.")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300, gridspec_kw={"width_ratios": [1, 1.1]})
    lab = [("original_A", "original questions"), ("simple_A", "one-step versions")]
    bottoms = np.zeros(2)
    for key, name, c in (("never", "never solved", COLOR_LOSS), ("learnable", "learnable (1–7 of 8)", COLOR_REWARD),
                         ("always", "always solved", COLOR_SFT)):
        vals = np.array([d["learnability"][k][key] / d["learnability"][k]["n"] for k, _ in lab])
        a1.bar([0, 1], vals, 0.55, bottom=bottoms, color=c, edgecolor=INK, lw=0.8, label=name)
        for i, v in enumerate(vals):
            a1.text(i, bottoms[i] + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=13.5, color="white", fontweight="bold")
        bottoms += vals
    a1.set_xticks([0, 1])
    a1.set_xticklabels([n for _, n in lab])
    a1.set_ylabel("share of problems")
    a1.set_ylim(0, 1.22)
    a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    panel(a1, "(a) Where problems sit for RL")
    legend(a1, loc="upper center", fontsize=11, ncol=3)

    c = d["curriculum_A0_shared"]
    rows = [("first step practised", c["first step in simple pool"]), ("first step not practised", c["first step not in pool"])]
    x = np.arange(len(rows))
    a2.bar(x - 0.2, [r["base_acc"] for _, r in rows], 0.4, color=COLOR_SFT, edgecolor=INK, lw=0.8, label="before")
    a2.bar(x + 0.2, [r["acc"] for _, r in rows], 0.4, color=COLOR_REWARD, edgecolor=INK, lw=0.8, label="after GRPO on simple questions")
    for i, (_, r) in enumerate(rows):
        a2.text(x[i] + 0.2, r["acc"] + 0.003, f"{100 * r['diff']:+.1f} pp",
                ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    a2.set_xticks(x)
    a2.set_xticklabels([n for n, _ in rows])
    a2.set_ylabel("accuracy on the full problem")
    a2.set_ylim(0, 0.135)
    panel(a2, "(b) Never-solved problems")
    legend(a2, loc="upper right", fontsize=11)
    plt.tight_layout()
    save(fig, "rl_curriculum")


def main():
    global LOG_CSV, EVAL_JSON
    ap = argparse.ArgumentParser(description="Render Stage 3 GRPO figures from real run logs.")
    ap.add_argument("--log", type=str, default=str(LOG_CSV),
                    help="GRPO training log CSV (default: the v3 run)")
    ap.add_argument("--eval-json", type=str, default=str(EVAL_JSON),
                    help="held-out eval results from src.rl.eval_rl")
    args = ap.parse_args([])
    log_csv = Path(args.log)
    eval_json = Path(args.eval_json)

    print("=" * 80)
    print("Rendering Phase III figures")
    print(f"  current run  : {V4_CSV}  (+ {V4_EVAL_CSV.name})")
    print(f"  v3 run       : {log_csv}")
    print(f"  eval results : {eval_json}")
    print("=" * 80)

    if not LOG_CSV.exists():
        raise SystemExit(f"[ERROR] training log not found: {LOG_CSV}")

    set_publication_style()
    fig_grpo_training_dynamics()                                          # v4: the current trainer
    fig_grpo_training_dynamics(LOG_CSV, None, "grpo_training_dynamics_v3")  # the shipped v3 run
    fig_grpo_benchmark_comparison()
    fig_general_models()
    fig_grpo_stability_v3_vs_v4()
    fig_grpo_beta_intervention()
    v5_log, v5_eval, v5_mu = SPEED_RUNS["v5"]
    fig_grpo_training_dynamics(v5_log, v5_eval, "grpo_training_dynamics_v5", updates_per_step=v5_mu)
    fig_grpo_speed_v4b_vs_v5()
    fig_numerical_diagnosis()
    fig_numerical_track()
    fig_rl_transfer()
    fig_rl_curriculum()

    # Sync PNGs to docs/ (the benchmark chart is absent if eval has not run yet)
    import shutil
    for name in ("grpo_benchmark_comparison.png",):
        src = ASSETS / name
        if src.exists():
            shutil.copy(src, DOCS / name)
        else:
            print(f"[SKIP] {name} was not rendered — not synced to docs/")
    print("[SUCCESS] GRPO figures rendered and synced to docs/")


if __name__ == "__main__":
    main()
