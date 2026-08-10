"""Where does the model's measured accuracy actually come from?

One aggregate number cannot distinguish "competent but unreliable" from "scoring
by accident", and those imply opposite next steps. Three probes, each answering
a different question about the same 166 held-out numerical items:

  tier      -- split by whether the answer must be COMPUTED or is already
               present in the prompt, and by derivation depth. Result: all of
               the measured accuracy is retrieval; must-compute items sit at or
               below their chance floor at every depth.

  oracle    -- extract the arithmetic the model wrote in its own <think>,
               execute it, and see whether that recovers the answer. This is
               the ceiling on what a calculator tool could buy, measured before
               building one. Result: below chance -- the expressions are noise,
               so the gap is formula recall, not arithmetic.

  prompting -- can a system prompt reproduce what GRPO installed? Result: no
               (72.3% -> 73.5% <think> vs GRPO's 98.2%), so that gain is real
               and attributable to RL.

  generations -- the same held-out prompts through base, SFT, and RL, printed
               side by side. Qualitative, but it shows what the percentages
               above actually look like.

Every tier reports its OWN chance floor: target clustering differs by tier, so
a single global floor would flatter some tiers and penalise others.

Usage:
  python -m src.analysis.diagnose_capability tier
  python -m src.analysis.diagnose_capability oracle    --ckpt <path> -k 3
  python -m src.analysis.diagnose_capability prompting --limit 166
  python -m src.analysis.diagnose_capability generations
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.rl.verifiers import NumericalVerifier  # noqa: E402
from src.rl.verifiers import safe_eval_number  # noqa: E402
from src.sft.chat import SFTChatEngine, DEFAULT_SYSTEM_PROMPT  # noqa: E402

TOL = 0.05


def spray_floor(targets_in_tier, all_targets, verifier, rng, trials=400):
    """Chance accuracy for this tier: k blind draws from the global answer
    distribution, scored by the same tail rule the model is scored by."""
    k = verifier.ANSWER_TAIL_NUMBERS
    hit = 0
    for _ in range(trials):
        for t in targets_in_tier:
            den = abs(t) if abs(t) > 1e-7 else 1.0
            draws = [rng.choice(all_targets) for _ in range(k)]
            if min(abs(v - t) / den for v in draws) <= TOL:
                hit += 1
    return hit / (trials * max(1, len(targets_in_tier)))


def paired_ci(diffs, rng, n_boot=2000):
    if not diffs:
        return 0.0, 0.0, 0.0
    n = len(diffs)
    mean_d = sum(diffs) / n
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return mean_d, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def report(name, tiers, items_by_id, sft, grpo, all_targets, verifier, rng,
           num_samples: int = 1):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    hdr = (f"{'tier':<26}{'n':>4}{'floor':>8}{'SFT':>8}{'GRPO':>8}"
           f"{'Δ (paired)':>13}{'95% CI':>20}{'pass@4':>9}{'p@4 flr':>9}")
    print(hdr)
    print("-" * len(hdr))
    for label, ids in tiers:
        ids = [i for i in ids if i in sft and i in grpo]
        if not ids:
            continue
        tg = [float(items_by_id[i]["final_numerical_value"]) for i in ids]
        fl = spray_floor(tg, all_targets, verifier, rng)
        s = float(np.mean([sft[i]["acc5"] for i in ids]))
        g = float(np.mean([grpo[i]["acc5"] for i in ids]))
        pk = float(np.mean([grpo[i].get("pass_at_k_5pct", 0.0) for i in ids]))
        # pass@k has its OWN, much higher floor: k independent blind draws.
        # Comparing pass@4 against the single-shot floor would flatter the model.
        pk_fl = 1.0 - (1.0 - fl) ** num_samples
        d, lo, hi = paired_ci([grpo[i]["acc5"] - sft[i]["acc5"] for i in ids], rng)
        star = "*" if (lo > 0 or hi < 0) else " "
        mark = "" if pk >= pk_fl else "  <- below chance"
        print(f"{label:<26}{len(ids):>4}{fl:>8.1%}{s:>8.1%}{g:>8.1%}"
              f"{d:>+12.1%}{star}  [{lo:+.1%},{hi:+.1%}]{pk:>9.1%}{pk_fl:>9.1%}{mark}")


def run_tier(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier the numerical benchmark by difficulty.")
    ap.add_argument("--results", type=str,
                    default=str(ROOT / "docs" / "eval_rl_paired_k4.json"))
    ap.add_argument("--test", type=str, default=str(ROOT / "data" / "rl" / "test.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", type=str,
                    default=str(ROOT / "docs" / "tiered_benchmark.json"))
    args = ap.parse_args(argv)

    res = json.load(open(args.results, encoding="utf-8"))
    if not ({"SFT", "GRPO"} <= res.keys()):
        print(f"[ERROR] need SFT and GRPO in {args.results}", file=sys.stderr)
        return 2
    sft = {r["id"]: r for r in res["SFT"]["per_item"] if "acc5" in r}
    grpo = {r["id"]: r for r in res["GRPO"]["per_item"] if "acc5" in r}
    k = res["SFT"].get("num_samples", 1)

    items = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    by_id = {it.get("id", i): it for i, it in enumerate(items)}
    verifier = NumericalVerifier()
    rng = random.Random(args.seed)

    shared = [i for i in sft if i in grpo and i in by_id]
    all_targets = [float(by_id[i]["final_numerical_value"]) for i in shared]
    print(f"[INFO] {len(shared)} numerical items, k={k} samples/prompt each")

    # ---- cut A: does the item require arithmetic at all? -------------------
    retr, comp = [], []
    for i in shared:
        t = float(by_id[i]["final_numerical_value"])
        den = abs(t) if abs(t) > 1e-7 else 1.0
        pn = verifier.extract_numbers(by_id[i]["prompt"])
        (retr if (pn and min(abs(p - t) / den for p in pn) <= TOL) else comp).append(i)
    report("CUT A — does the answer need to be COMPUTED, or is it already in the prompt?",
           [("retrievable from prompt", retr), ("must be computed", comp)],
           by_id, sft, grpo, all_targets, verifier, rng, k)

    # ---- cut B: derivation depth (quartile bins) ----------------------------
    depth = {i: str(by_id[i].get("reference_think") or "").count("=") for i in shared}
    qs = np.percentile([depth[i] for i in shared], [25, 50, 75])
    edges = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)]
    names = [f"depth <={qs[0]:.0f} steps (Q1)", f"depth {qs[0]:.0f}-{qs[1]:.0f} (Q2)",
             f"depth {qs[1]:.0f}-{qs[2]:.0f} (Q3)", f"depth >{qs[2]:.0f} (Q4, deepest)"]
    tiers_b = [(nm, [i for i in shared if lo < depth[i] <= hi])
               for nm, (lo, hi) in zip(names, edges)]
    report("CUT B — accuracy vs derivation depth ('=' steps in the reference)",
           tiers_b, by_id, sft, grpo, all_targets, verifier, rng, k)

    # ---- cut C: depth, restricted to items that actually require computing --
    tiers_c = [(nm, [i for i in ids if i in set(comp)]) for nm, ids in tiers_b]
    report("CUT C — depth, EXCLUDING items whose answer is already in the prompt",
           tiers_c, by_id, sft, grpo, all_targets, verifier, rng, k)

    out = {"n": len(shared), "num_samples": k,
           "retrievable": len(retr), "must_compute": len(comp),
           "depth_quartiles": [float(x) for x in qs]}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), indent=2)
    print(f"\n[INFO] * = paired bootstrap CI excludes zero.  summary -> {args.out_json}")
    return 0











# Right-hand side of an '=' -- now allowing letters, because real derivations
# interleave units ("10,000 m * 5,000 m") and those must be stripped, not
# treated as a parse failure. Validated against ground truth in _sanity().
RHS = re.compile(r"=\s*([^=\n;,]{3,120})")

_KEEP = {"log", "log10", "log2", "exp", "sqrt", "abs", "sin", "cos", "tan",
         "asin", "acos", "atan", "sinh", "cosh", "tanh", "radians", "degrees",
         "floor", "ceil", "pow", "round", "min", "max", "pi", "e", "tau"}
_SCI = re.compile(r"(?<=\d)[eE](?=[+-]?\d)")
_ALPHA = re.compile(r"[A-Za-zµμΩ°Å_]+")


def normalise(text: str, verifier: NumericalVerifier) -> str:
    """'3×10^10' -> '3e10', '^' -> '**', strip thousands separators."""
    t = verifier.SCI_NOTATION.sub(lambda m: f"{m.group(1)}e{m.group(2)}", text)
    t = t.replace("×", "*").replace("·", "*").replace("^", "**")
    t = t.replace("−", "-").replace("–", "-")
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)


def strip_units(expr: str) -> str:
    """Drop unit tokens ('m', 'N', 'Pa', 'km') while protecting the 'e' of
    scientific notation and whitelisted function/constant names."""
    protected = _SCI.sub("\x00", expr)          # 5e7 -> 5\x007
    protected = _ALPHA.sub(lambda m: m.group(0) if m.group(0) in _KEEP else " ",
                           protected)
    out = protected.replace("\x00", "e")
    # a stripped unit can leave "5  7" (juxtaposition) -- that is not arithmetic
    if re.search(r"\d\s+\d", out):
        return ""
    return re.sub(r"\s+", " ", out).strip(" .:")


def expressions(think: str, verifier: NumericalVerifier):
    """Every evaluable arithmetic expression the model wrote, in order."""
    out = []
    for m in RHS.finditer(normalise(think, verifier)):
        raw = strip_units(m.group(1))
        if not raw or not re.search(r"[\+\-\*/]", raw):
            continue                      # a bare number, not a calculation
        # trailing prose can leave dangling operators; trim to a parseable core
        for cand in (raw, raw.rstrip("*/+-( "), raw.split(" ")[0]):
            v = safe_eval_number(cand)
            if v is not None:
                out.append((cand, v))
                break
    return out


def close(v, t):
    return abs(v - t) / (abs(t) if abs(t) > 1e-7 else 1.0) <= TOL


def run_oracle(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 0 gate: can the model state the right calculation?")
    ap.add_argument("--ckpt", type=str,
                    default=str(ROOT / "checkpoints" / "grpo" / "selected.pt"))
    ap.add_argument("--test", type=str, default=str(ROOT / "data" / "rl" / "test.jsonl"))
    ap.add_argument("--num-samples", "-k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args(argv)

    verifier = NumericalVerifier()
    rng = random.Random(11)

    items = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    num = [x for x in items
           if x["type"] == "numerical" and x.get("final_numerical_value") is not None]
    # the tier that matters: answer NOT already present in the prompt
    must = []
    for it in num:
        t = float(it["final_numerical_value"])
        den = abs(t) if abs(t) > 1e-7 else 1.0
        pn = verifier.extract_numbers(it["prompt"])
        if not (pn and min(abs(p - t) / den for p in pn) <= TOL):
            must.append(it)
    all_targets = [float(x["final_numerical_value"]) for x in num]

    engine = SFTChatEngine(checkpoint_path=Path(args.ckpt))
    print(f"[INFO] {Path(args.ckpt).name} | {len(must)} must-compute items | "
          f"k={args.num_samples} | device {engine.device}", flush=True)

    stated, last_ok, any_ok, n_expr, no_expr = [], [], [], [], 0
    for idx, it in enumerate(must):
        if (idx + 1) % 20 == 0:
            print(f"  • {idx + 1}/{len(must)}", flush=True)
        tgt = float(it["final_numerical_value"])
        s_hit, l_hit, a_hit = [], [], []
        for s in range(args.num_samples):
            torch.manual_seed(args.seed + idx * 1000 + s)
            comp, _ = engine.generate_single_prompt(
                user_query=it["prompt"], max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty)

            # what the model actually ANSWERED (tail rule, as eval_rl scores it)
            ans = re.sub(r"^.*?</think>", "", comp, flags=re.DOTALL)
            ns = verifier.extract_numbers(ans)
            s_hit.append(bool(ns) and close(
                min(ns[-verifier.ANSWER_TAIL_NUMBERS:],
                    key=lambda v: abs(v - tgt) / (abs(tgt) if abs(tgt) > 1e-7 else 1.0)), tgt))

            # what executing its OWN arithmetic would have produced
            think = comp.split("</think>")[0] if "</think>" in comp else comp
            ex = expressions(think, verifier)
            n_expr.append(len(ex))
            if not ex:
                l_hit.append(False)
                a_hit.append(False)
            else:
                l_hit.append(close(ex[-1][1], tgt))
                a_hit.append(any(close(v, tgt) for _, v in ex))
        stated.append(np.mean(s_hit))
        last_ok.append(np.mean(l_hit))
        any_ok.append(np.mean(a_hit))
        if not n_expr or n_expr[-1] == 0:
            no_expr += 1

    n = len(must)
    med_expr = float(np.median(n_expr)) if n_expr else 0.0

    # INSTRUMENT CEILING. Run the identical extractor over the REFERENCE
    # derivations for the same items. Those provably contain the correct
    # arithmetic, so whatever the extractor recovers there is the most this
    # measurement can see. A trained tool-use model would emit one designated
    # <python> block and not need extraction at all, so this ceiling is a
    # property of the probe, not of the plan.
    ref_last, ref_any = [], []
    for it in must:
        t = float(it["final_numerical_value"])
        ex = expressions(str(it.get("reference_think") or ""), verifier)
        ref_last.append(bool(ex) and close(ex[-1][1], t))
        ref_any.append(any(close(v, t) for _, v in ex))
    ceil_last, ceil_any = float(np.mean(ref_last)), float(np.mean(ref_any))

    def floor(k, trials=400):
        hit = 0
        for _ in range(trials):
            for it in must:
                t = float(it["final_numerical_value"])
                den = abs(t) if abs(t) > 1e-7 else 1.0
                if min(abs(rng.choice(all_targets) - t) / den
                       for _ in range(max(1, k))) <= TOL:
                    hit += 1
        return hit / (trials * n)

    f1 = floor(1)
    fk = floor(int(round(med_expr)) or 1)

    print("\n" + "=" * 78)
    print("PHASE 0 — ORACLE EXECUTION on must-compute items")
    print("=" * 78)
    print(f"  items {n}   samples/item {args.num_samples}   "
          f"median expressions per answer {med_expr:.0f}")
    print(f"  answers containing no evaluable expression at all: "
          f"{100 * np.mean([e == 0 for e in n_expr]):.1f}% of samples")
    print()
    print(f"  {'measure':<44}{'acc@5%':>9}{'floor':>9}")
    print("  " + "-" * 62)
    print(f"  {'model stated answer (status quo)':<44}{np.mean(stated):>8.1%}{f1:>9.1%}")
    print(f"  {'EXECUTE last expression (the intervention)':<44}"
          f"{np.mean(last_ok):>8.1%}{f1:>9.1%}")
    print(f"  {'execute ANY expression (upper bound)':<44}"
          f"{np.mean(any_ok):>8.1%}{fk:>9.1%}")
    print()
    print(f"  {'INSTRUMENT CEILING (same extractor on ground truth)':<44}")
    print(f"  {'  reference, last expression':<44}{ceil_last:>8.1%}")
    print(f"  {'  reference, any expression':<44}{ceil_any:>8.1%}")
    print("  (the extractor cannot see more than this; a trained tool-use model")
    print("   emits one designated <python> block and needs no extraction)")

    gain = np.mean(last_ok) - np.mean(stated)
    frac = np.mean(any_ok) / ceil_any if ceil_any > 0 else float("nan")
    print("\n" + "=" * 78)
    print(f"GATE: last-expression accuracy = {np.mean(last_ok):.1%} "
          f"(floor {f1:.1%}), change vs stated answer {gain:+.1%}")
    print(f"      model reaches {frac:.0%} of the instrument ceiling "
          f"({np.mean(any_ok):.1%} of {ceil_any:.1%}, any-expression)")
    if np.mean(last_ok) >= 0.15:
        print("  -> PROCEED. Executing the model's own arithmetic clears the 15% bar;")
        print("     the formulas are there and only the arithmetic is missing.")
    elif np.mean(last_ok) >= 0.08:
        print("  -> MARGINAL. Above floor but below the 15% bar; tool use would help")
        print("     a little. Reconsider cost/benefit before building.")
    else:
        print("  -> STOP. The model's own expressions do not evaluate to the answer,")
        print("     so a calculator has nothing correct to execute. The gap is")
        print("     formula recall, not arithmetic; tool use will not fix it.")
    print("=" * 78)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"ckpt": args.ckpt, "n": n, "k": args.num_samples,
                   "stated": float(np.mean(stated)),
                   "last_expression": float(np.mean(last_ok)),
                   "any_expression": float(np.mean(any_ok)),
                   "floor_1": f1, "floor_k": fk,
                   "ceiling_last": ceil_last, "ceiling_any": ceil_any,
                   "median_expressions": med_expr},
                  open(args.out_json, "w", encoding="utf-8"), indent=2)
        print(f"[INFO] -> {args.out_json}")
    return 0










# Each variant is a hypothesis about how cheaply the GRPO behaviour is obtained.
PROMPTS = {
    "default (what SFT eval used)": DEFAULT_SYSTEM_PROMPT,
    "explicit think instruction": (
        DEFAULT_SYSTEM_PROMPT
        + " Always reason step by step inside <think> and </think> tags, then "
          "give the final answer after the closing tag."
    ),
    "think + terse answer": (
        DEFAULT_SYSTEM_PROMPT
        + " Always reason inside <think> and </think> tags first. After the "
          "closing tag, state the final answer in one or two sentences, ending "
          "with the numerical result."
    ),
}

# GRPO @100 reference values, docs/eval_rl_results.json (corrected scoring).
GRPO_THINK, GRPO_ACC5, GRPO_TOKENS = 0.982, 0.102, 251
SFT_THINK, SFT_ACC5, SFT_TOKENS = 0.729, 0.096, 308


def run_prompting(argv=None) -> int:
    ap = argparse.ArgumentParser(description="System-prompt control for the GRPO <think> gain.")
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "sft" / "selected.pt"))
    ap.add_argument("--test", type=str, default=str(ROOT / "data" / "rl" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=60, help="numerical prompts to use")
    ap.add_argument("--num-samples", "-k", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-json", type=str,
                    default=str(ROOT / "docs" / "prompted_baseline.json"))
    args = ap.parse_args(argv)

    items = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    num = [it for it in items
           if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
    num = num[: args.limit]

    verifier = NumericalVerifier()
    engine = SFTChatEngine(checkpoint_path=Path(args.ckpt))
    print(f"[INFO] {Path(args.ckpt).name} on {len(num)} numerical prompts, "
          f"k={args.num_samples}, device {engine.device}")

    out = {}
    for name, sys_prompt in PROMPTS.items():
        think = acc5 = toks = 0.0
        for idx, it in enumerate(num):
            tgt = float(it["final_numerical_value"])
            t_hits, a_hits, t_toks = [], [], []
            for s in range(args.num_samples):
                torch.manual_seed(args.seed + idx * 1000 + s)
                comp, stats = engine.generate_single_prompt(
                    user_query=it["prompt"],
                    system_prompt=sys_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
                t_hits.append(
                    re.search(r"<think>.*?</think>", comp, flags=re.DOTALL) is not None)
                ans = re.sub(r"^.*?</think>", "", comp, flags=re.DOTALL)
                ns = verifier.extract_numbers(ans)
                if ns:
                    den = abs(tgt) if abs(tgt) > 1e-7 else 1.0
                    err = min(abs(v - tgt) / den
                              for v in ns[-verifier.ANSWER_TAIL_NUMBERS:])
                    a_hits.append(err <= 0.05)
                else:
                    a_hits.append(False)
                t_toks.append(int(stats["total_tokens"]))
            think += sum(t_hits) / len(t_hits)
            acc5 += sum(a_hits) / len(a_hits)
            toks += sum(t_toks) / len(t_toks)
        n = len(num)
        out[name] = {"think_rate": think / n, "acc5": acc5 / n,
                     "mean_tokens": toks / n, "n": n}
        print(f"  {name:<32} <think> {think/n:6.1%}   acc@5% {acc5/n:6.1%}   "
              f"tokens {toks/n:5.0f}")

    print("\n" + "=" * 78)
    print("PROMPTED BASELINE vs GRPO")
    print("=" * 78)
    print(f"{'variant':<34}{'<think>':>10}{'acc@5%':>10}{'tokens':>9}")
    print("-" * 78)
    print(f"{'SFT + default prompt':<34}{SFT_THINK:>9.1%}{SFT_ACC5:>10.1%}{SFT_TOKENS:>9.0f}")
    for name, m in out.items():
        if name.startswith("default"):
            continue
        print(f"{'SFT + ' + name:<34}{m['think_rate']:>9.1%}{m['acc5']:>10.1%}"
              f"{m['mean_tokens']:>9.0f}")
    print(f"{'GRPO @100 (200 steps, ~90 min)':<34}{GRPO_THINK:>9.1%}"
          f"{GRPO_ACC5:>10.1%}{GRPO_TOKENS:>9.0f}")
    print("-" * 78)

    best = max((m["think_rate"] for k, m in out.items() if not k.startswith("default")),
               default=0.0)
    print()
    if best >= GRPO_THINK - 0.05:
        print("VERDICT: a system prompt reproduces GRPO's <think> gain.")
        print("  The RL stage's only significant effect was obtainable for free.")
    elif best > SFT_THINK + 0.05:
        print(f"VERDICT: prompting closes part of the gap "
              f"({SFT_THINK:.1%} -> {best:.1%} vs GRPO {GRPO_THINK:.1%}).")
        print("  GRPO retains a real but partial advantage on this metric.")
    else:
        print(f"VERDICT: prompting does NOT reproduce the gain "
              f"(best {best:.1%} vs GRPO {GRPO_THINK:.1%}).")
        print("  The <think> improvement is attributable to the RL stage.")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.ckpt, "n": len(num),
                   "num_samples": args.num_samples,
                   "grpo_reference": {"think_rate": GRPO_THINK, "acc5": GRPO_ACC5},
                   "variants": out}, f, indent=2)
    print(f"\n[INFO] -> {args.out_json}")
    return 0




def run_generations(argv=None) -> int:
    """Same prompts through base, SFT, and RL, side by side."""
    from src.sft.chat import SFTChatEngine

    ap = argparse.ArgumentParser(description="Base vs SFT vs RL on the same prompts.")
    ap.add_argument("--base", default=str(ROOT.parent / "nanogpt_seis" / "checkpoints" / "ckpt.pt"))
    ap.add_argument("--sft", default=str(ROOT / "checkpoints" / "sft" / "selected.pt"))
    ap.add_argument("--grpo", default=str(ROOT / "checkpoints" / "grpo" / "selected.pt"))
    ap.add_argument("--test", default=str(ROOT / "data" / "rl" / "test.jsonl"))
    ap.add_argument("--index", type=int, action="append", default=[],
                    help="held-out prompt index (repeatable; default 0 and 3)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-json", default=str(ROOT / "docs" / "example_generations.json"))
    args = ap.parse_args(argv)

    items = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    numerical = [it for it in items
                 if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
    chosen = [numerical[i % len(numerical)] for i in (args.index or [0, 3])]

    out = []
    for stage, ckpt in (("base", args.base), ("sft", args.sft), ("grpo", args.grpo)):
        if not Path(ckpt).exists():
            print(f"[skip] {stage}: {ckpt} not found")
            continue
        engine = SFTChatEngine(checkpoint_path=Path(ckpt))
        for item in chosen:
            torch.manual_seed(args.seed)
            if stage == "base":
                # The base has never seen <|im_start|>; its ChatML rows are random.
                # Query it the only way a raw LM can be queried: as a continuation.
                tokens = engine.tokenizer.encode(item["prompt"] + "\n").ids
                text = "".join(
                    chunk["text_chunk"] for chunk in engine.stream_response(
                        tokens, max_new_tokens=args.max_new_tokens,
                        temperature=0.3, top_k=40, repetition_penalty=1.2))
            else:
                text, _ = engine.generate_single_prompt(
                    user_query=item["prompt"], max_new_tokens=args.max_new_tokens,
                    temperature=0.3, top_k=40, repetition_penalty=1.2)
            out.append({"stage": stage, "prompt": item["prompt"],
                        "target": item["final_numerical_value"], "text": text.strip()})
            print(f"\n{'=' * 78}\n[{stage}]  target={item['final_numerical_value']}")
            print(f"PROMPT: {item['prompt'][:160]}")
            print(f"{'-' * 78}\n{text.strip()[:1200]}")
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[info] {len(out)} generations -> {args.out_json}")
    return 0


_COMMANDS = {"tier": run_tier, "oracle": run_oracle, "prompting": run_prompting,
             "generations": run_generations}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in _COMMANDS:
        print(__doc__)
        return 2
    return _COMMANDS[cmd](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
