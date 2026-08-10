"""Phase III figures.

    assets/grpo_training_dynamics    4 panels: reward components, KL vs beta,
                                     entropy, completion length and stop rate
    assets/grpo_benchmark_comparison SFT vs RL on the held-out metrics

Both read the run log and results file at plot time and warn when their
provenance disagrees.

    python -m src.plots grpo
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plots import (  # noqa: E402
    ASSETS, COLOR_CLEAN, COLOR_CONCEPT, COLOR_FIX, COLOR_FORMAT, COLOR_KL,
    COLOR_LOSS, COLOR_MUTED, COLOR_REWARD, COLOR_SFT, COLOR_TRAP, DOCS,
    legend, nice_ylim, panel, rolling_mean, save, set_publication_style)

ROOT = Path(__file__).resolve().parents[2]
LOG_CSV = ROOT / "checkpoints" / "runs" / "grpo.csv"
EVAL_JSON = ROOT / "docs" / "eval_rl_results.json"





def load_grpo_logs():
    """Load GRPO training metrics from CSV into a dict of numpy arrays.

    `stop_rate` only exists in v3+ logs (it is the TerminationVerifier
    diagnostic); it comes back as all-NaN for older runs and the panel that
    uses it degrades gracefully.
    """
    cols = ["step", "mean_reward", "num_reward", "concept_reward", "format_reward",
            "clean_reward", "policy_loss", "kl_div", "kl_coeff", "entropy",
            "mean_comp_tokens", "stop_rate"]
    data = {c: [] for c in cols}

    with open(LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("step"):
                continue
            for c in cols:
                data[c].append(float(row[c]) if row.get(c) not in (None, "") else float("nan"))

    return {c: np.array(v) for c, v in data.items()}




def fig_grpo_training_dynamics():
    """Generate 4-panel figure evaluating GRPO training convergence."""
    logs = load_grpo_logs()
    steps = logs["step"]
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
    ax1.plot(steps, mean_rewards, color=COLOR_REWARD, alpha=0.35, linewidth=1.0, label="Step Mean Reward")
    rolling_r = rolling_mean(mean_rewards, window=20)
    ax1.plot(steps, rolling_r, color=COLOR_REWARD, linewidth=2.5, label="20-Step Rolling Average")
    ax1.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.2)

    ax1.set_title("(a) Total GRPO Mean Reward Curve", fontsize=12, fontweight="bold", pad=8, color="#2C3E50")
    ax1.set_xlabel("GRPO Optimization Step", fontsize=10.5, color="#2C3E50")
    ax1.set_ylabel("Mean Reward $R(y)$", fontsize=10.5, color="#2C3E50")
    ax1.set_ylim(*nice_ylim(mean_rewards, rolling_r, floor=0.0, ceil=0.0))
    ax1.grid(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=9, loc="lower right")

    # Annotate peak rolling reward
    peak_idx = int(np.argmax(rolling_r))
    lo1, hi1 = ax1.get_ylim()
    # keep the callout inside the axes regardless of where the peak lands
    dx = -0.30 * (steps[-1] - steps[0]) if peak_idx > len(steps) / 2 else 0.15 * (steps[-1] - steps[0])
    ax1.annotate(
        f"Peak Avg Reward: {rolling_r[peak_idx]:+.3f}",
        xy=(steps[peak_idx], rolling_r[peak_idx]),
        xytext=(steps[peak_idx] + dx, min(rolling_r[peak_idx] + 0.18 * (hi1 - lo1), hi1 - 0.08 * (hi1 - lo1))),
        arrowprops=dict(facecolor=COLOR_REWARD, shrink=0.08, width=1.2, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_REWARD,
    )

    # -------------------------------------------------------------------------
    # Subplot (b): Sub-Reward Breakdown
    # -------------------------------------------------------------------------
    r_num = rolling_mean(num_rewards, 20)
    r_con = rolling_mean(concept_rewards, 20)
    r_fmt = rolling_mean(format_rewards, 20)
    ax2.plot(steps, r_num, color=COLOR_LOSS, linewidth=2.2, label="Numerical ($R_{num}$)")
    ax2.plot(steps, r_con, color=COLOR_CONCEPT, linewidth=2.2, label="Concept Reward ($R_{concept}$)")
    ax2.plot(steps, r_fmt, color=COLOR_FORMAT, linewidth=2.2, label="Format Compliance ($R_{format}$)")
    ax2.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.2)

    ax2.set_title("(b) Verifier Sub-Reward Breakdown", fontsize=12, fontweight="bold", pad=8, color="#2C3E50")
    ax2.set_xlabel("GRPO Optimization Step", fontsize=10.5, color="#2C3E50")
    ax2.set_ylabel("Reward Sub-score", fontsize=10.5, color="#2C3E50")
    ax2.set_ylim(*nice_ylim(r_num, r_con, r_fmt, floor=0.0, ceil=0.0))
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=9, loc="lower right")

    # -------------------------------------------------------------------------
    # Subplot (c): Reference Model KL Divergence Anchor
    # -------------------------------------------------------------------------
    kl_roll = rolling_mean(kl_divs, 20)
    ax3.plot(steps, kl_divs, color=COLOR_KL, alpha=0.35, linewidth=1.0, label="Token KL ($D_{KL}$)")
    ax3.plot(steps, kl_roll, color=COLOR_KL, linewidth=2.5, label="20-Step Rolling Mean")
    ax3.axhline(0.05, color="#E74C3C", linestyle="--", linewidth=1.2, label="Adaptive-$\\beta$ KL Target (0.05)")

    ax3.set_title("(c) Policy Drift from the SFT Reference", fontsize=12, fontweight="bold", pad=8, color="#2C3E50")
    ax3.set_xlabel("GRPO Optimization Step", fontsize=10.5, color="#2C3E50")
    ax3.set_ylabel("KL Divergence $D_{KL}(\\pi_\\theta \\parallel \\pi_{ref})$", fontsize=10.5, color=COLOR_KL)
    ax3.set_ylim(*nice_ylim(kl_divs, floor=0.0, ceil=0.06))
    ax3.grid(True)
    ax3.spines["top"].set_visible(False)

    # Entropy shares this panel: KL measures drift from the reference, entropy
    # measures collapse of the policy itself. Reading them together separates
    # "moved somewhere new" from "stopped exploring".
    ax3b = ax3.twinx()
    ax3b.plot(steps, rolling_mean(entropies, 20), color=COLOR_LOSS,
              linewidth=2.2, label="Policy Entropy (nats)")
    ax3b.set_ylabel("Policy Entropy (nats)", fontsize=10.5, color=COLOR_LOSS)
    ax3b.spines["top"].set_visible(False)

    l3a, lab3a = ax3.get_legend_handles_labels()
    l3b, lab3b = ax3b.get_legend_handles_labels()
    ax3.legend(l3a + l3b, lab3a + lab3b, frameon=True, facecolor="white",
               edgecolor="#BDC3C7", fontsize=8.5, loc="upper left")

    # -------------------------------------------------------------------------
    # Subplot (d): Termination behaviour — completion length & stop rate.
    # stop_rate is the fraction of rollouts that emitted <|im_end|> instead of
    # being cut off at the token budget. It is the metric the V3
    # TerminationVerifier was added to move, so it gets its own axis rather
    # than being inferred from length.
    # -------------------------------------------------------------------------
    len_roll = rolling_mean(comp_tokens, 20)
    ax4.plot(steps, len_roll, color=COLOR_CONCEPT, linewidth=2.2, label="Completion Length (tokens)")
    ax4.set_title("(d) Termination Behaviour", fontsize=12, fontweight="bold", pad=8, color="#2C3E50")
    ax4.set_xlabel("GRPO Optimization Step", fontsize=10.5, color="#2C3E50")
    ax4.set_ylabel("Mean Completion Tokens", fontsize=10.5, color=COLOR_CONCEPT)
    ax4.grid(True)
    ax4.spines["top"].set_visible(False)

    ax4b = ax4.twinx()
    if has_stop:
        ax4b.plot(steps, rolling_mean(stop_rates, 20), color=COLOR_CLEAN,
                  linewidth=2.2, label="Stop Rate (emitted $<|im\\_end|>$)")
        ax4b.set_ylim(0, 1.02)
    else:
        ax4b.text(0.5, 0.5, "stop_rate not logged in this run",
                  transform=ax4b.transAxes, ha="center", va="center",
                  fontsize=9, color="#7F8C8D", style="italic")
    ax4b.set_ylabel("Rollout Stop Rate", fontsize=10.5, color=COLOR_CLEAN)
    ax4b.spines["top"].set_visible(False)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, frameon=True, facecolor="white",
               edgecolor="#BDC3C7", fontsize=9, loc="lower right")

    plt.tight_layout()

    ASSETS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    png_path = ASSETS / "grpo_training_dynamics.png"
    pdf_path = ASSETS / "grpo_training_dynamics.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {png_path.relative_to(ROOT)} and {pdf_path.relative_to(ROOT)}")


def fig_grpo_benchmark_comparison():
    """SFT vs GRPO on the held-out metrics from src.rl.eval_rl.

    The V1 chart plotted hardcoded values of the training reward itself — a
    circular benchmark. This one reads docs/eval_rl_results.json produced by
    `python -m src.rl.eval_rl` on the chunk-held-out validation split.
    """
    if not EVAL_JSON.exists():
        print(f"[SKIP] {EVAL_JSON} not found — run `python -m src.rl.eval_rl` first.")
        return
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    # Any number of model series may be present (e.g. SFT + several GRPO
    # checkpoints). Keep SFT first; preserve file order for the rest.
    series = [k for k in results if not k.startswith("_")]
    series.sort(key=lambda k: (0 if k.upper().startswith("SFT") else 1))
    if len(series) < 2:
        print(f"[SKIP] eval results incomplete ({series}) — need at least two model series.")
        return

    prov = results.get("_provenance")
    if prov is None:
        print("[WARN] eval results carry no provenance — they predate the provenance field "
              "and may describe an OLDER run than the training log being plotted. "
              "Re-run `python -m src.rl.eval_rl` to be sure.")
    else:
        print(f"[INFO] eval provenance: SFT={prov.get('sft_ckpt')}  GRPO={prov.get('grpo_ckpt')}")
        # Logs live at checkpoints/runs/<run>.csv and weights at
        # checkpoints/<run>/, so the log STEM is what must appear in the
        # evaluated checkpoint path -- not the parent dir, which is now
        # always "runs".
        run_dir = LOG_CSV.stem
        if run_dir not in str(prov.get("grpo_ckpt", "")):
            print(f"[WARN] MISMATCH: plotting training log from '{run_dir}' but the benchmark "
                  f"numbers come from '{prov.get('grpo_ckpt')}'. The two panels describe "
                  f"different runs — re-run src.rl.eval_rl against '{run_dir}'.")

    metrics = [
        ("Numerical\naccuracy @1%", "numerical_acc_1pct"),
        ("Numerical\naccuracy @5%", "numerical_acc_5pct"),
        ("<think> rate\n(numerical)", "numerical_think_rate"),
        ("Contamination\nrate (lower=better)", "contamination_rate"),
        ("Prompt-echo\nrate (lower=better)", "echo_rate"),
    ]
    scores = {name: [results[name][k] for _, k in metrics] for name in series}
    palette = [COLOR_SFT, COLOR_REWARD, COLOR_CONCEPT, COLOR_FORMAT, COLOR_CLEAN]

    x = np.arange(len(metrics))
    n = len(series)
    width = min(0.35, 0.8 / n)

    fig, ax = plt.subplots(figsize=(11 + 1.2 * (n - 2), 5.5), dpi=300)

    all_rects = []
    for i, name in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        color = palette[i % len(palette)]
        rects = ax.bar(x + offset, scores[name], width, label=name,
                       color=color, edgecolor="#2C3E50", linewidth=1.2)
        all_rects.append((rects, color))

    ax.axhline(0, color="#2C3E50", linewidth=1.0)
    ax.set_ylabel("Rate", fontsize=11, fontweight="bold", color="#2C3E50")
    ax.set_title("Reward-Independent Evaluation on the Hash-Pinned Held-Out Set",
                 fontsize=13, fontweight="bold", pad=12, color="#2C3E50")
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in metrics], fontsize=9.5, fontweight="bold", color="#2C3E50")
    ax.set_ylim(0, max(max(v) for v in scores.values()) * 1.25 + 0.05)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=10, loc="upper right")

    for rects, color in all_rects:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1%}",
                        xy=(rect.get_x() + rect.get_width() / 2, height + 0.005),
                        xytext=(0, 0), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8 if n > 2 else 9, fontweight="bold", color=color)

    plt.tight_layout()

    png_path = ASSETS / "grpo_benchmark_comparison.png"
    pdf_path = ASSETS / "grpo_benchmark_comparison.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {png_path.relative_to(ROOT)} and {pdf_path.relative_to(ROOT)}")


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
    print(f"  training log : {log_csv}")
    print(f"  eval results : {eval_json}")
    print("=" * 80)

    if not LOG_CSV.exists():
        raise SystemExit(f"[ERROR] training log not found: {LOG_CSV}")

    set_publication_style()
    fig_grpo_training_dynamics()
    fig_grpo_benchmark_comparison()

    # Sync PNGs to docs/ (the benchmark chart is absent if eval has not run yet)
    import shutil
    for name in ("grpo_training_dynamics.png", "grpo_benchmark_comparison.png"):
        src = ASSETS / name
        if src.exists():
            shutil.copy(src, DOCS / name)
        else:
            print(f"[SKIP] {name} was not rendered — not synced to docs/")
    print("[SUCCESS] GRPO figures rendered and synced to docs/")


if __name__ == "__main__":
    main()
