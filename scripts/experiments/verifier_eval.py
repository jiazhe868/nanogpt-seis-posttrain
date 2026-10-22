"""Score labelled samples with a verifier and compare selection rules over k samples.

    python scripts/experiments/verifier_eval.py <verifier.pt> <out.json> <samples.jsonl> [<samples.jsonl> ...]
    python scripts/experiments/verifier_eval.py --report <out.json> [ids.json]   # recompute the table only

Per question and temperature, the samples are cut into disjoint groups of k = 8; each rule picks one
sample per group, scored like a single sample. Rules: one sample, vote@8 (src/rl/self_consistency.py),
best and weighted (src/rl/verifier.py). Differences against vote@8 are paired over questions,
95% bootstrap CIs. With ids.json (a list of ids), the table is restricted to those questions.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl.self_consistency import select_consensus  # noqa: E402
from src.rl.verifier import select  # noqa: E402

K = 8
RULES = ("one sample", "vote@8", "best@8", "weighted@8")


def per_question(samples):
    by = {}
    for s in samples:
        by.setdefault((s["T"], s["id"]), []).append(s)
    out = {}
    for (T, pid), ss in by.items():
        acc = {r: [] for r in RULES}
        for g in range(0, len(ss) - K + 1, K):
            grp = ss[g:g + K]
            tails, sc, ok = [x["tail"] for x in grp], [x["score"] for x in grp], [x["correct"] for x in grp]
            acc["one sample"].append(np.mean(ok))
            acc["vote@8"].append(ok[select_consensus(tails)])
            acc["best@8"].append(ok[select(tails, sc, "best")])
            acc["weighted@8"].append(ok[select(tails, sc, "weighted")])
        out.setdefault(T, {})[pid] = {r: float(np.mean(v)) for r, v in acc.items()}
    return out


def report(res, ids=None):
    rng = np.random.default_rng(0)
    table = {}
    for T, qs in sorted(res.items()):
        keys = [i for i in qs if ids is None or str(i) in ids]
        row = {}
        for r in RULES:
            v = np.array([qs[i][r] for i in keys])
            d = v - np.array([qs[i]["vote@8"] for i in keys])
            b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            row[r] = {"acc": 100 * v.mean(), "vs_vote": 100 * d.mean(),
                      "ci": [100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)]}
        table[str(T)] = row
        print(f"T {T}  ({len(keys)} questions)")
        for r, x in row.items():
            print(f"   {r:11s} {x['acc']:5.1f}   vs vote@8 {x['vs_vote']:+5.1f} [{x['ci'][0]:+5.1f}, {x['ci'][1]:+5.1f}]")
    return table


def main():
    if sys.argv[1] == "--report":
        d = json.load(open(sys.argv[2]))
        ids = {str(i) for i in json.load(open(sys.argv[3]))} if len(sys.argv) > 3 else None
        res = {float(T): qs for T, qs in d["per_question"].items()}
        report(res, ids)
        return
    import torch
    from src.rl.train_grpo import load_sft_model
    from src.rl.verifier import score
    from src.sft.dataset import load_bpe_tokenizer
    ckpt, out = sys.argv[1], sys.argv[2]
    samples = [json.loads(l) for p in sys.argv[3:] for l in open(p, encoding="utf-8")]
    dev = torch.device("cuda")
    model = load_sft_model(ROOT / ckpt, dev).eval()
    tok = load_bpe_tokenizer(ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json")
    for s, v in zip(samples, score(model, tok, [(s["q"], s["completion"]) for s in samples], dev)):
        s["score"] = v
    res = per_question(samples)
    table = report(res)
    json.dump({"verifier": ckpt, "inputs": sys.argv[3:], "table": table,
               "per_question": {str(T): qs for T, qs in res.items()},
               "scores": [{"id": s["id"], "T": s["T"], "score": s["score"], "correct": s["correct"]} for s in samples]},
              open(out, "w"))


if __name__ == "__main__":
    main()
