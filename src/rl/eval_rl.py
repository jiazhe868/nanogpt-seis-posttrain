"""Compare SFT against the RL policy on held-out prompts.

Reports numerical accuracy at 1%/5%, <think> compliance, contamination, echo,
and completion length; the composite training reward comes last and clearly
labelled, because a metric that appears in the loss cannot also be the
benchmark (PITFALLS B1).

Stores PER-ITEM outcomes, which is what makes pairing possible: McNemar on
discordant pairs, paired bootstrap CIs, pass@k, and an explicit MDE line so a
null result is not mistaken for no effect. Decoding is the pinned deployment
config with a per-prompt seed.

    python -m src.rl.eval_rl --sft-ckpt checkpoints/sft/selected.pt \
        --grpo-ckpt checkpoints/grpo/selected.pt -k 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sft.chat import SFTChatEngine
from src.rl.verifiers import CompositeRewardEngine, NumericalVerifier, CleanLanguageVerifier, EchoPenaltyVerifier

VAL_FILE = ROOT / "data" / "rl" / "val.jsonl"


def evaluate_checkpoint(
    ckpt_path: Path,
    val_items: List[Dict[str, Any]],
    max_new_tokens: int = 384,
    temperature: float = 0.3,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
    seed: int = 1234,
    num_samples: int = 1,
) -> Dict[str, Any]:
    print(f"\n[EVAL] Loading checkpoint -> {ckpt_path}", flush=True)
    engine = SFTChatEngine(checkpoint_path=ckpt_path)

    num_verifier = NumericalVerifier()
    clean_verifier = CleanLanguageVerifier()
    echo_verifier = EchoPenaltyVerifier()
    reward_engine = CompositeRewardEngine()

    n_num = n_num_acc1 = n_num_acc5 = n_num_fmt = 0
    n_all = n_contaminated = n_echo = 0
    total_tokens = 0
    rewards: List[float] = []
    samples: List[Dict[str, str]] = []
    # Per-item outcomes. Without these no paired test (McNemar) and no bootstrap
    # is possible after the fact -- the earlier version stored only 3 sample
    # completions, which made every comparison retroactively unpairable.
    per_item: List[Dict] = []

    for idx, item in enumerate(val_items):
        if (idx + 1) % 20 == 0:
            print(f"  • prompt {idx + 1}/{len(val_items)}", flush=True)

        # Decode at the deployment configuration (chat.py defaults), seeded per
        # prompt for reproducibility. Neither greedy (top_k=1) nor a weak
        # repetition penalty is usable here: at temp 0.2 / rp 1.05 this model
        # emits 56% verbatim-repeated 5-grams, so those settings measure a
        # decoding artifact rather than the weights. rp>=1.2 removes it.
        is_num = (item["type"] == "numerical"
                  and item.get("final_numerical_value") is not None)
        target = float(item["final_numerical_value"]) if is_num else None

        # k completions per prompt. With k=1 this reduces exactly to the old
        # behaviour; with k>1 each prompt yields a RATE in [0,1] instead of a
        # single Bernoulli draw, cutting per-prompt variance ~1/k.
        hits1, hits5, thinks, contam, echoes, rws, toks = [], [], [], [], [], [], []
        first_completion = None
        for s in range(num_samples):
            torch.manual_seed(seed + idx * 1000 + s)
            completion, stats = engine.generate_single_prompt(
                user_query=item["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            if first_completion is None:
                first_completion = completion
            toks.append(int(stats["total_tokens"]))
            contam.append(clean_verifier(completion) < 0)
            echoes.append(echo_verifier(completion, item["prompt"]) < 0)
            rws.append(reward_engine.score(completion, item)["total_reward"])

            if is_num:
                thinks.append(
                    re.search(r"<think>.*?</think>", completion, flags=re.DOTALL) is not None)
                answer_text = re.sub(r"^.*?</think>", "", completion, flags=re.DOTALL)
                numbers = num_verifier.extract_numbers(answer_text)
                if numbers:
                    denom = abs(target) if abs(target) > 1e-7 else 1.0
                    # Score the stated answer (tail), matching NumericalVerifier.
                    # min-over-first-8 was gameable by verbosity: a question-blind
                    # 8-number spray scored 14.3%, above the model itself.
                    err = min(abs(v - target) / denom
                              for v in numbers[-num_verifier.ANSWER_TAIL_NUMBERS:])
                    hits1.append(err <= 0.01)
                    hits5.append(err <= 0.05)
                else:
                    hits1.append(False)
                    hits5.append(False)

        def rate(xs):
            return (sum(xs) / len(xs)) if xs else 0.0

        n_all += 1
        total_tokens += sum(toks) / len(toks)
        n_contaminated += rate(contam)
        n_echo += rate(echoes)
        rewards.append(sum(rws) / len(rws))

        rec = {"id": item.get("id", idx), "type": item["type"],
               "contaminated": rate(contam), "echo": rate(echoes),
               "reward": sum(rws) / len(rws), "tokens": sum(toks) / len(toks)}
        if is_num:
            n_num += 1
            n_num_fmt += rate(thinks)
            n_num_acc1 += rate(hits1)
            n_num_acc5 += rate(hits5)
            rec.update({"acc1": rate(hits1), "acc5": rate(hits5),
                        "think": rate(thinks),
                        # pass@k: did ANY sample land? separates "can sometimes
                        # do it" from "reliably does it"
                        "pass_at_k_1pct": float(any(hits1)),
                        "pass_at_k_5pct": float(any(hits5))})
        per_item.append(rec)

        if idx < 3:
            samples.append({"prompt": item["prompt"][:120],
                            "completion": (first_completion or "")[:400]})

    return {
        "numerical_acc_1pct": n_num_acc1 / max(1, n_num),
        "numerical_acc_5pct": n_num_acc5 / max(1, n_num),
        "numerical_think_rate": n_num_fmt / max(1, n_num),
        "contamination_rate": n_contaminated / max(1, n_all),
        "echo_rate": n_echo / max(1, n_all),
        "mean_completion_tokens": total_tokens / max(1, n_all),
        "composite_training_reward": sum(rewards) / max(1, len(rewards)),
        "n_numerical": n_num,
        "n_total": n_all,
        "num_samples": num_samples,
        "pass_at_k_1pct": (sum(r.get("pass_at_k_1pct", 0.0) for r in per_item)
                           / max(1, n_num)),
        "pass_at_k_5pct": (sum(r.get("pass_at_k_5pct", 0.0) for r in per_item)
                           / max(1, n_num)),
        "per_item": per_item,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Reward-independent SFT vs GRPO evaluation")
    # Defaults track the current pipeline: the behaviour-selected SFT checkpoint
    # (src.sft.checkpoints select) and the V3 GRPO run trained from it.
    parser.add_argument("--sft-ckpt", type=str, default=str(ROOT / "checkpoints" / "sft" / "selected.pt"))
    parser.add_argument("--grpo-ckpt", type=str, default=str(ROOT / "checkpoints" / "grpo" / "grpo_ckpt_step_200.pt"))
    parser.add_argument("--val-file", type=str, default=str(VAL_FILE))
    parser.add_argument("--num-conceptual", type=int, default=60,
                        help="Non-numerical val prompts to include (all numerical prompts are always used)")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.3, help="Deployment default (chat.py)")
    parser.add_argument("--top-k", type=int, default=40, help="Deployment default (chat.py)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Deployment default (chat.py)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-samples", "-k", type=int, default=1,
                        help="completions per prompt; k>1 cuts per-prompt variance ~1/k "
                             "and enables pass@k (cost scales linearly)")
    parser.add_argument("--out-json", type=str, default=str(ROOT / "docs" / "eval_rl_results.json"))
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE III EVALUATION — held-out, chunk-disjoint")
    print("=" * 80)

    with open(args.val_file, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    numerical = [it for it in all_items if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
    conceptual = [it for it in all_items if it["type"] != "numerical"][: args.num_conceptual]
    val_items = numerical + conceptual
    print(f"[INFO] Eval prompts: {len(numerical)} numerical + {len(conceptual)} conceptual "
          f"= {len(val_items)} (val file has {len(all_items)})")

    results = {}
    for name, ckpt in (("SFT", Path(args.sft_ckpt)), ("GRPO", Path(args.grpo_ckpt))):
        if not ckpt.exists():
            print(f"[WARN] {name} checkpoint not found: {ckpt} — skipping")
            continue
        results[name] = evaluate_checkpoint(
            ckpt, val_items,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            num_samples=args.num_samples,
        )

    print("\n" + "=" * 80)
    print("RESULTS (held-out same-construct metrics; the training reward is last)")
    print("=" * 80)
    metric_rows = [
        ("Numerical accuracy @1%", "numerical_acc_1pct", "{:.1%}"),
        ("Numerical accuracy @5%", "numerical_acc_5pct", "{:.1%}"),
        ("<think> rate (numerical)", "numerical_think_rate", "{:.1%}"),
        ("Contamination rate", "contamination_rate", "{:.1%}"),
        ("Prompt-echo rate", "echo_rate", "{:.1%}"),
        ("Mean completion tokens", "mean_completion_tokens", "{:.0f}"),
        ("[ref] composite training reward", "composite_training_reward", "{:+.4f}"),
    ]
    header = f"{'Metric':<34}" + "".join(f" | {name:<12}" for name in results)
    print(header)
    print("-" * len(header))
    for label, key, fmt in metric_rows:
        row = f"{label:<34}"
        for name in results:
            row += f" | {fmt.format(results[name][key]):<12}"
        print(row)
    print("=" * 80)

    # PAIRED significance testing. Both models answer the SAME prompts, so the
    # unpaired two-proportion z-test used previously threw away most of the
    # available power -- the same "always pair" lesson already learned for loss
    # comparison (audit §9) but never applied here.
    #
    # Also reports the minimum detectable effect, because at n=166 an unpaired
    # test could only resolve an 11-point difference: a null result there says
    # "underpowered", not "no effect", and must not be read as the latter.
    if {"SFT", "GRPO"} <= results.keys():
        import math
        A, B = results["SFT"], results["GRPO"]
        by_id_a = {r["id"]: r for r in A.get("per_item", [])}
        by_id_b = {r["id"]: r for r in B.get("per_item", [])}
        shared = [i for i in by_id_a if i in by_id_b]

        print("\nSignificance of SFT → GRPO deltas "
              "(paired bootstrap + McNemar, α=0.05):")
        paired_metrics = [
            ("Numerical accuracy @1%", "acc1", True),
            ("Numerical accuracy @5%", "acc5", True),
            ("<think> rate (numerical)", "think", True),
            ("Contamination rate", "contaminated", False),
            ("Prompt-echo rate", "echo", False),
        ]
        rng = random.Random(20260731)
        for label, key, num_only in paired_metrics:
            pairs = [(by_id_a[i].get(key), by_id_b[i].get(key)) for i in shared
                     if (by_id_a[i].get(key) is not None
                         and by_id_b[i].get(key) is not None
                         and (not num_only or by_id_a[i]["type"] == "numerical"))]
            if not pairs:
                continue
            n = len(pairs)
            diffs = [b - a for a, b in pairs]
            mean_d = sum(diffs) / n
            # paired bootstrap CI over per-prompt differences
            boots = []
            for _ in range(2000):
                s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                boots.append(s)
            boots.sort()
            lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
            sig = "significant" if (lo > 0 or hi < 0) else "n.s."
            # McNemar on the discordant pairs (only meaningful for 0/1 outcomes)
            disc = [(a, b) for a, b in pairs if a != b]
            b01 = sum(1 for a, b in disc if b > a)
            b10 = sum(1 for a, b in disc if b < a)
            mc = (f"  McNemar {b01}↑/{b10}↓"
                  if (b01 + b10) else "  McNemar n/a (0 discordant)")
            print(f"  {label:<26} n={n:<4} Δ={mean_d:+.4f} "
                  f"95%CI[{lo:+.4f},{hi:+.4f}] {sig:<12}{mc}")

        # Power: what could this design have detected?
        n_num = A["n_numerical"]
        p0 = A["numerical_acc_5pct"]
        za, zb = 1.959964, 0.8416212
        lo_d, hi_d = 0.0, 0.9
        for _ in range(120):
            d = (lo_d + hi_d) / 2
            p2 = min(p0 + d, 0.999)
            pbar = (p0 + p2) / 2
            need = (za * math.sqrt(2 * pbar * (1 - pbar) / n_num)
                    + zb * math.sqrt((p0 * (1 - p0) + p2 * (1 - p2)) / n_num))
            lo_d, hi_d = (d, hi_d) if need > d else (lo_d, d)
        print(f"\n  [power] at n={n_num}, k={A.get('num_samples', 1)} sample(s)/prompt, "
              f"baseline {p0:.1%}: MDE ≈ {(lo_d + hi_d) / 2:.1%} at 80% power.")
        print("          A null result below that threshold means UNDERPOWERED, "
              "not 'no effect'.")
        if A.get("num_samples", 1) > 1:
            print(f"  [pass@k] any-sample-correct @5%: "
                  f"SFT {A['pass_at_k_5pct']:.1%} → GRPO {B['pass_at_k_5pct']:.1%} "
                  f"(capability present vs reliably expressed)")

    # Record provenance alongside the numbers. Without it a results file from an
    # earlier run is indistinguishable from a current one, and the plotting
    # script will happily render stale checkpoints as if they were the new ones.
    results["_provenance"] = {
        "sft_ckpt": str(args.sft_ckpt),
        "grpo_ckpt": str(args.grpo_ckpt),
        "val_file": str(args.val_file),
        "n_numerical": len(numerical),
        "n_conceptual": len(conceptual),
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "num_samples": args.num_samples,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Full results (incl. sample completions) -> {out_path}")


if __name__ == "__main__":
    main()
