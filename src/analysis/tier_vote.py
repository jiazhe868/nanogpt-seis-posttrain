"""acc@1 and self-consistency vote@k by tier, from stored per-sample answers (README §5.14).

Reads an eval JSON whose models were each run over several seeds ({"results": {name: [run,
...]}}, as written for docs/eval_calc_k4.json and docs/eval_vote_pooled8.json), pools every
sample per prompt, and reports, per tier (answer already in the prompt / must be computed):
acc@1 (mean single-sample hit rate), vote@k over disjoint k-subsets of the pooled samples, and
paired bootstrap CIs for vote@k - acc@1 and for acc@1 against a baseline model.

    python -m src.analysis.tier_vote docs/eval_vote_pooled8.json --baseline SFT
    python -m src.analysis.tier_vote docs/eval_calc_k4.json --baseline SFT+calc
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rl.self_consistency import select_consensus  # noqa: E402
from src.rl.verifiers import NumericalVerifier  # noqa: E402

TOL = 0.05


def close(a: float, b: float) -> bool:
    d = abs(b) if abs(b) > 1e-7 else 1.0
    return abs(a - b) / d <= TOL


def hit(tail, target) -> bool:
    return bool(tail) and any(close(v, target) for v in tail)


def pooled(runs, k: int, tail_n: int):
    """per-prompt acc@1 and vote@k, pooling all runs (seeds) of one model."""
    pool, single = {}, {}
    for run in runs:
        for x in run["per_item"]:
            if x["type"] != "numerical" or "answer_numbers" not in x:
                continue
            tails = [a[-tail_n:] for a in x["answer_numbers"]]
            pool.setdefault(x["id"], (x["target"], []))[1].extend(tails)
            single.setdefault(x["id"], []).append(st.mean(hit(a, x["target"]) for a in tails))
    acc1 = {i: st.mean(v) for i, v in single.items()}
    vote = {}
    for i, (t, T) in pool.items():
        subsets = [T[j:j + k] for j in range(0, len(T) - k + 1, k)]
        vote[i] = st.mean(hit(s[select_consensus(s)], t) for s in subsets) if subsets else float("nan")
    return acc1, vote


def ci(diffs, rng, n_boot=2000):
    bs = sorted(st.mean(diffs[rng.randrange(len(diffs))] for _ in diffs) for _ in range(n_boot))
    return bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results")
    ap.add_argument("--baseline", default="SFT")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--test", default=str(ROOT / "data" / "rl" / "test.jsonl"))
    args = ap.parse_args()

    res = json.load(open(args.results, encoding="utf-8"))["results"]
    items = {it["id"]: it for it in map(json.loads, open(args.test, encoding="utf-8"))}
    nv = NumericalVerifier()

    def retrievable(i) -> bool:          # same cut as diagnose_capability tier (cut A)
        it = items[i]
        t = float(it["final_numerical_value"])
        pn = nv.extract_numbers(it["prompt"])
        return bool(pn) and any(close(p, t) for p in pn)

    tiers = [("all", lambda i: True), ("in prompt", retrievable), ("must-compute", lambda i: not retrievable(i))]
    rng = random.Random(0)
    base = pooled(res[args.baseline], args.k, nv.ANSWER_TAIL_NUMBERS)[0]
    print(f"{'model':<12}{'tier':<14}{'n':>4}{'acc@1':>8}{f'vote@{args.k}':>9}"
          f"{f'  vote@{args.k} - acc@1':<26}{f'acc@1 - {args.baseline}'}")
    for name, runs in res.items():
        acc1, vote = pooled(runs, args.k, nv.ANSWER_TAIL_NUMBERS)
        for tier, keep in tiers:
            ids = [i for i in acc1 if keep(i)]
            dv = [vote[i] - acc1[i] for i in ids]
            db = [acc1[i] - base[i] for i in ids if i in base]
            (vl, vh), (bl, bh) = ci(dv, rng), ci(db, rng)
            print(f"{name:<12}{tier:<14}{len(ids):>4}{st.mean(acc1[i] for i in ids):>8.3f}"
                  f"{st.mean(vote[i] for i in ids):>9.3f}  {st.mean(dv):+.3f} [{vl:+.3f}, {vh:+.3f}]"
                  f"       {st.mean(db):+.3f} [{bl:+.3f}, {bh:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
