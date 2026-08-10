"""Measure the measurement, before believing any model comparison.

    floor    what a knowledge-free policy scores, and what the ground truth
             scores. A metric is only informative between the two.
    hacking  hand-built degenerate answers scored by the real reward; anything
             ranking at or above a genuine answer is an exploitable channel.
    group    one real G=8 rollout group, scored end to end, showing which
             reward components actually produced gradient.

This project got the first two wrong: PITFALLS C1 (the floor exceeded the
model) and D2 (padding diluted the echo penalty).

    python -m src.analysis.validate_measurement floor
    python -m src.analysis.validate_measurement hacking
    python -m src.analysis.validate_measurement group --markdown
    python -m src.analysis.validate_measurement all      # floor + hacking (CPU only)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.verifiers import CompositeRewardEngine, NumericalVerifier  # noqa: E402

TOL1, TOL5 = 0.01, 0.05

# Corrected tail-rule measurements from docs/eval_rl_results.json; override on
# the command line when re-measuring.
DEFAULT_GRPO_ACC5 = 0.102
DEFAULT_SFT_ACC5 = 0.096

DEFAULT_TEST = str(ROOT / "data" / "rl" / "test.jsonl")

REWARD_COMPONENTS = ["r_num", "r_concept", "r_format", "r_echo", "r_clean", "r_length", "r_term"]


def load_numerical_items(path: str) -> tuple[list[dict], int]:
    """Held-out items with a parseable target, plus the total prompt count."""
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    numerical = [it for it in items
                 if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
    return numerical, len(items)


def min_relative_error(numbers, target, tail: int):
    """Replicate eval_rl's scoring exactly: min rel-error over the answer tail."""
    if not numbers:
        return None
    denom = abs(target) if abs(target) > 1e-7 else 1.0
    return min(abs(v - target) / denom for v in numbers[-tail:])


# ===========================================================================
# floor / ceiling
# ===========================================================================


def run_floor_ceiling(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Floor/ceiling validity check for numerical accuracy.")
    parser.add_argument("--test", default=DEFAULT_TEST)
    parser.add_argument("--trials", type=int, default=400,
                        help="Monte-Carlo trials per spray width")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grpo-acc5", type=float, default=DEFAULT_GRPO_ACC5,
                        help="measured acc@5%% to compare against the floor")
    parser.add_argument("--sft-acc5", type=float, default=DEFAULT_SFT_ACC5)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    verifier = NumericalVerifier()
    tail = verifier.ANSWER_TAIL_NUMBERS

    numerical, n_prompts = load_numerical_items(args.test)
    targets = [float(it["final_numerical_value"]) for it in numerical]
    n = len(numerical)

    print("=" * 76)
    print("METRIC VALIDITY — numerical accuracy")
    print("=" * 76)
    print(f"held-out numerical items: {n}   (of {n_prompts} total prompts)")
    print(f"eval scores min-relative-error over the LAST {tail} numbers")
    log_targets = np.log10(np.abs(targets) + 1e-30)
    print(f"target magnitudes: 10^{log_targets.min():.1f} .. 10^{log_targets.max():.1f}")

    # ---- ceiling: score the reference answers themselves --------------------
    hits1 = hits5 = parsed = 0
    for item, target in zip(numerical, targets):
        reference = (item.get("reference_final_answer") or item.get("reference_response")
                     or item.get("response") or "")
        rel_err = min_relative_error(verifier.extract_numbers(str(reference)), target, tail)
        if rel_err is None:
            continue
        parsed += 1
        hits1 += rel_err <= TOL1
        hits5 += rel_err <= TOL5
    print("\n--- CEILING (reference answers, should be ~100%) ---")
    if parsed:
        print(f"  parseable      {parsed}/{n}  ({parsed / n:.1%})")
        print(f"  acc @1%        {hits1 / n:.1%}")
        print(f"  acc @5%        {hits5 / n:.1%}")
    else:
        print("  [no reference-answer field in this test file — ceiling not measurable here]")

    # ---- why a blind spray can win: is the target distribution clustered? ---
    print("\n--- TARGET DISTRIBUTION (explains how gameable the metric is) ---")
    n_distinct = len(set(targets))
    print(f"  distinct target values: {n_distinct}/{n} ({n_distinct / n:.1%})")
    target_array = np.array(targets)
    n_near_another = 0
    for target in targets:
        gaps = np.abs(target_array - target) / (abs(target) if abs(target) > 1e-7 else 1.0)
        n_near_another += int((gaps <= TOL5).sum() > 1)   # >1 because it matches itself
    print(f"  targets within 5% of another target: {n_near_another}/{n} ({n_near_another / n:.1%})")
    print("  most common values: "
          + ", ".join(f"{v:g}×{c}" for v, c in Counter(targets).most_common(5)))

    # ---- floor: degenerate baselines ---------------------------------------
    print("\n--- FLOOR (policies with zero knowledge) ---")
    for constant in (1.0, 10.0):
        h1 = sum(min_relative_error([constant], t, tail) <= TOL1 for t in targets)
        h5 = sum(min_relative_error([constant], t, tail) <= TOL5 for t in targets)
        print(f"  constant '{constant:g}'            acc@1% {h1 / n:5.1%}   acc@5% {h5 / n:5.1%}")

    # Spray: k numbers drawn from the empirical distribution of targets. This is
    # the strongest realistic degenerate policy — it knows the answer
    # distribution (learnable from training data) but nothing about the question.
    print(f"\n  number spray, k draws from the empirical target distribution"
          f" ({args.trials} trials):")
    print(f"  {'k':>3} {'acc@1%':>9} {'acc@5%':>9}")
    spray_accuracy = {}
    for k in (1, 2, 3, 5, 8):
        hits_at_1 = hits_at_5 = 0
        for _ in range(args.trials):
            for target in targets:
                draws = [rng.choice(targets) for _ in range(k)]
                rel_err = min_relative_error(draws, target, tail)
                hits_at_1 += rel_err <= TOL1
                hits_at_5 += rel_err <= TOL5
        total = args.trials * n
        spray_accuracy[k] = (hits_at_1 / total, hits_at_5 / total)
        print(f"  {k:>3} {hits_at_1 / total:>8.1%} {hits_at_5 / total:>8.1%}")

    # ---- which scoring rule gives the widest usable range? ------------------
    # min-over-8 is what makes spraying pay. Scoring only the answer tail should
    # keep the ceiling (the reference states its answer at the end) while
    # collapsing the floor.
    print("\n--- ALTERNATIVE SCORING RULES (ceiling vs floor trade-off) ---")
    reference_number_counts = []
    rules = {
        "min over first 8 (OLD, broken)": lambda ns: ns[:8],
        "last 3 (current)": lambda ns: ns[-3:],
        "last number only": lambda ns: ns[-1:],
        "min over first 2": lambda ns: ns[:2],
    }
    print(f"  {'rule':<28} {'ceiling@5%':>11} {'floor@5%':>10} {'range':>8}")
    for name, pick in rules.items():
        ceiling_hits = 0
        for item, target in zip(numerical, targets):
            reference = (item.get("reference_final_answer")
                         or item.get("reference_response") or "")
            numbers = verifier.extract_numbers(str(reference))
            if name.startswith("min over first 8"):
                reference_number_counts.append(len(numbers))
            if numbers:
                denom = abs(target) if abs(target) > 1e-7 else 1.0
                ceiling_hits += min(abs(v - target) / denom for v in pick(numbers)) <= TOL5
        floor_hits = 0
        for _ in range(args.trials):
            for target in targets:
                draws = [rng.choice(targets) for _ in range(8)]
                denom = abs(target) if abs(target) > 1e-7 else 1.0
                floor_hits += min(abs(v - target) / denom for v in pick(draws)) <= TOL5
        ceiling, floor = ceiling_hits / n, floor_hits / (args.trials * n)
        print(f"  {name:<28} {ceiling:>10.1%} {floor:>10.1%} {ceiling - floor:>7.1%}")

    print(f"\n  numbers per reference answer: median {int(np.median(reference_number_counts))}, "
          f"mean {np.mean(reference_number_counts):.1f}, "
          f"p90 {int(np.percentile(reference_number_counts, 90))}")

    # ---- verdict ------------------------------------------------------------
    floor = spray_accuracy[tail][1]     # adaptive adversary: k = tail width
    ceiling = hits5 / n if parsed else float("nan")
    print("\n--- VERDICT ---")
    print(f"  floor   (knowledge-free spray, k={tail})  {floor:.1%}")
    print(f"  model   SFT {args.sft_acc5:.1%} / GRPO {args.grpo_acc5:.1%}")
    print(f"  ceiling (reference answers)          {ceiling:.1%}")

    if args.grpo_acc5 <= floor:
        print("\n  -> FAIL: the model is at or below the floor. This metric cannot")
        print("     demonstrate any capability; do not report it as one.")
    else:
        span = ceiling - floor
        position = (args.grpo_acc5 - floor) / span if span > 0 else float("nan")
        print(f"\n  -> model sits {args.grpo_acc5 - floor:+.1%} pts above the floor, "
              f"{position:.1%} of the way through the")
        print(f"     usable range [{floor:.1%}, {ceiling:.1%}].")
        if position < 0.15:
            print("     Real but weak: the effect is small relative to the range the")
            print("     metric can resolve, so treat small deltas here with suspicion.")
    return 0


# ===========================================================================
# reward hacking
# ===========================================================================


def build_probes(item: dict) -> dict:
    """Adversarial completions for one numerical item."""
    target = item["final_numerical_value"]
    concepts = item.get("key_concepts", [])[:8]
    reference_think = item.get("reference_think", "") or "We apply the standard relation."
    reference_answer = item.get("reference_final_answer", "") or f"The answer is {target}."

    keywords = " ".join(concepts)
    # numbers spanning the plausible range, none of them derived
    spray = ", ".join(f"{v:g}" for v in [1, 2, 7.1, 10, 100, 1000, 1e5, 3e10])

    return {
        "reference (genuine)":
            f"<think>{reference_think}</think>{reference_answer}",
        "keyword stuffing in think":
            f"<think>{keywords} {keywords} {keywords}</think>"
            "The answer follows from the analysis above.",
        "number spray, no reasoning":
            f"<think>Consider the values.</think>The result may be {spray}.",
        "spray + keywords (combined)":
            f"<think>{keywords}</think>Possible values: {spray}. {keywords}.",
        "empty think block":
            "<think></think>The seismic parameters are computed from the standard relation.",
        "think tag, no answer":
            f"<think>{reference_think}</think>",
        "correct answer, no think":
            reference_answer,
        "verbose filler + correct answer":
            f"<think>{reference_think}</think>{reference_answer} "
            + ("Further analysis confirms this. " * 8),
    }


def run_hacking_probe(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Adversarial probe of the composite reward.")
    parser.add_argument("--test", default=DEFAULT_TEST)
    args = parser.parse_args(argv)

    engine = CompositeRewardEngine()
    numerical, _ = load_numerical_items(args.test)

    names = list(build_probes(numerical[0]))
    totals = {name: [] for name in names}
    components = {name: {} for name in names}

    for item in numerical:
        for name, text in build_probes(item).items():
            scores = engine.score(text, item)
            totals[name].append(scores["total_reward"])
            for key, value in scores.items():
                if key != "total_reward":
                    components[name].setdefault(key, []).append(value)

    n = len(numerical)
    reference_mean = sum(totals["reference (genuine)"]) / n

    print("=" * 88)
    print(f"ADVERSARIAL REWARD PROBE — {n} held-out numerical items")
    print("=" * 88)
    print("A probe scoring >= the genuine reference is an exploitable channel.\n")
    component_names = sorted({k for c in components.values() for k in c})
    header = (f"{'probe':<32} {'total':>8} {'vs ref':>9}  "
              + " ".join(f"{s[:9]:>10}" for s in component_names))
    print(header)
    print("-" * len(header))

    exploits = []
    for name in sorted(names, key=lambda x: -sum(totals[x]) / n):
        mean_reward = sum(totals[name]) / n
        delta = mean_reward - reference_mean
        # An exploit is a strategy that PAYS: strictly above the genuine answer
        # by a usable margin. Tying is fine — the reward is then indifferent to
        # the trick, so no gradient pushes the policy toward it.
        flag = ""
        if name != "reference (genuine)" and delta > 0.02:
            flag = "  <== EXPLOIT"
            exploits.append((name, delta))
        elif name != "reference (genuine)" and delta > -0.02:
            flag = "  (neutral — no gradient)"
        row = f"{name:<32} {mean_reward:>8.3f} {delta:>+9.3f}  "
        row += " ".join(f"{sum(components[name].get(s, [0])) / n:>10.3f}"
                        for s in component_names)
        print(row + flag)

    print("\n" + "=" * 88)
    if exploits:
        print(f"{len(exploits)} exploitable channel(s) found:")
        for name, delta in exploits:
            print(f"  - {name}: {delta:+.3f} vs the genuine answer")
        print("\nRL optimises this reward directly; any channel here is a gradient")
        print("it can climb without getting better at the task.")
    else:
        print("No probe matched the genuine reference. The reward ranks all")
        print("hand-built degenerate strategies strictly below a real answer.")
    return 0


# ===========================================================================
# one worked rollout group
# ===========================================================================
# The group-relative update is easy to state and easy to misread. This samples
# a real group from a real checkpoint, scores every completion with the real
# verifiers, and prints the arithmetic the trainer actually sees.
#
# Reading one group makes several failures self-evident:
#   * if every completion scores the same, std=0 -> every advantage is 0 -> the
#     group contributes NO gradient, however large the reward
#   * a dead verifier is a constant column, which cancels in the group baseline
#     exactly like any other constant
#   * only sampled tokens belong in the mask; forced-EOS padding does not


def run_worked_group(argv=None) -> int:
    import torch

    from src.sft.chat import SFTChatEngine

    parser = argparse.ArgumentParser(description="Score one real G=8 rollout group.")
    parser.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "grpo" / "selected.pt"))
    parser.add_argument("--prompts", default=str(ROOT / "data" / "rl" / "train.jsonl"))
    parser.add_argument("--index", type=int, default=3, help="which prompt in the pool")
    parser.add_argument("--group-size", "-g", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="rollout temperature (train_grpo samples on-policy at 1.0)")
    parser.add_argument("--top-k", type=int, default=16388,
                        help="full vocab = unrestricted, closest to the trainer's top_p=1.0")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = parser.parse_args(argv)

    numerical, _ = load_numerical_items(args.prompts)
    item = numerical[args.index % len(numerical)]

    engine = SFTChatEngine(checkpoint_path=Path(args.ckpt))
    reward = CompositeRewardEngine()

    print(f"\nPROMPT  {item['prompt'][:150]}...")
    print(f"TARGET  {item['final_numerical_value']}\n")

    rows = []
    for i in range(args.group_size):
        torch.manual_seed(args.seed + i)
        text, stats = engine.generate_single_prompt(
            user_query=item["prompt"], max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, repetition_penalty=1.0)
        rows.append({"i": i, "text": text, "tokens": int(stats["total_tokens"]),
                     **reward.score(text, item)})

    rewards = torch.tensor([r["total_reward"] for r in rows], dtype=torch.float32)
    mean, std = rewards.mean(), rewards.std()
    advantages = (rewards - mean) / (std + 1e-8)

    header = ["#"] + [c.replace("r_", "") for c in REWARD_COMPONENTS] + ["total", "adv", "effect"]
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
    for row, advantage in zip(rows, advantages.tolist()):
        effect = ("reinforced" if advantage > 0.05
                  else "suppressed" if advantage < -0.05 else "neutral")
        cells = ([str(row["i"])]
                 + [f"{row.get(c, 0.0):+.2f}" for c in REWARD_COMPONENTS]
                 + [f"{row['total_reward']:+.3f}", f"{advantage:+.2f}", effect])
        print(("| " + " | ".join(cells) + " |") if args.markdown else "  ".join(cells))

    print(f"\ngroup mean R = {mean:+.4f}   std = {std:.4f}")
    if std < 1e-6:
        print("  std = 0 -> every advantage is 0 -> THIS GROUP PRODUCES NO GRADIENT.")
        print("  A high but uniform reward teaches nothing; only within-group spread does.")
    else:
        print(f"  advantages sum to {advantages.sum():+.2e} (they must, by construction) — the")
        print("  update raises the log-prob of above-average completions and lowers the rest.")

    print("\nper-component within-group spread (zero spread = no contribution):")
    for component in REWARD_COMPONENTS:
        values = torch.tensor([r.get(component, 0.0) for r in rows], dtype=torch.float32)
        flag = "" if values.std() > 1e-6 else "   <- constant, cancels in the baseline"
        print(f"  {component:<12} mean {values.mean():+.3f}  std {values.std():.3f}{flag}")
    return 0


# ===========================================================================


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    rest = sys.argv[2:]
    if command not in ("floor", "hacking", "group", "all"):
        print(__doc__)
        return 2
    if command in ("floor", "all"):
        run_floor_ceiling(rest if command == "floor" else [])
    if command in ("hacking", "all"):
        run_hacking_probe(rest if command == "hacking" else [])
    if command == "group":
        run_worked_group(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
