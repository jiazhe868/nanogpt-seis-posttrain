#!/usr/bin/env python3
"""Build a broad RL pool: many distinct problems the student has never trained on, each meant to be
seen about once, plus first-step "stepping stones" for problems it never solves (README §5.19).

Why: GRPO's gains travel by similarity to the practised problems (audit §15.17), so the pool must be
broad to reach held-out questions; a problem the student never solves gives no gradient, but practising
its first step lifts the full problem ~7x.

    candidates  -> RL-schema candidates, deduplicated against the SFT data, each other and held-out chunks
    teacher     -> convert tools/gen_pot_problems.py generate/rewrite output to RL-schema candidates
    never       -> never-solved candidates (0/8 after tools/filter_learnable_prompts.py) that have a program,
                   for tools/../gen_simple-style first-step questions
    build       -> the pool: learnable rows (1-7 of 8) from all filtered files, at most --per-template per
                   number-masked template, shuffled; reports how close the real held-out questions are to it

Strict protocol: everything derives from SFT-train seeds; test chunks never enter the pool.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data/splits/train_pot_notation2.jsonl"
WORD = re.compile(r"[a-zA-Z]{3,}")


def words(q: str) -> frozenset:
    return frozenset(WORD.findall(q.lower()))


def template(q: str) -> str:
    return re.sub(r"\d[\d,.]*", "#", q.strip())


class NearestNeighbour:
    """Max word-Jaccard of a question against a reference set, via an inverted index on rarer words."""

    def __init__(self, texts):
        self.sets = [words(t) for t in texts]
        df = collections.Counter(w for s in self.sets for w in s)
        self.inv = collections.defaultdict(list)
        for j, s in enumerate(self.sets):
            for w in s:
                if df[w] < 0.05 * len(self.sets):
                    self.inv[w].append(j)

    def best(self, q: str, top: int = 300) -> float:
        w = words(q)
        cand = collections.Counter(j for x in w for j in self.inv.get(x, ()))
        return max((len(w & self.sets[j]) / max(len(w | self.sets[j]), 1) for j, _ in cand.most_common(top)), default=0.0)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def excluded_chunks(paths):
    out = set()
    for p in paths:
        for r in load(p):
            if r.get("chunk_id") not in (None, "None"):
                out.add(r["chunk_id"])
    return out


def dedupe(rows, ref_nn, exclude, seen_prompts, max_sim):
    kept, stats = [], collections.Counter()
    for r in rows:
        q = r["prompt"].strip()
        if q in seen_prompts:
            stats["duplicate"] += 1
            continue
        if r.get("chunk_id") in exclude:
            stats["held-out chunk"] += 1
            continue
        if ref_nn.best(q) >= max_sim:
            stats["near-duplicate of an SFT question"] += 1
            continue
        seen_prompts.add(q)
        kept.append(r)
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["candidates", "teacher", "never", "build"])
    ap.add_argument("--inp", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-chunks", nargs="*", default=["data/rl/test.jsonl", "data/rl/transfer_B.jsonl"],
                    help="files whose chunks may not enter the pool (test; B kept as an untouched check)")
    ap.add_argument("--also-dedupe", nargs="*", default=[], help="earlier candidate files to deduplicate against")
    ap.add_argument("--max-sim", type=float, default=0.8)
    ap.add_argument("--per-template", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix", default="v13_")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.mode in ("candidates", "teacher"):
        rows = []
        for p in args.inp:
            for k, r in enumerate(load(p)):
                if args.mode == "teacher":          # gen_pot_problems output -> RL schema
                    try:
                        ans = float(r["answer"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    r = {"id": f"{args.prefix}{Path(p).stem}_{k}", "prompt": r["question"], "type": "numerical",
                         "final_numerical_value": ans, "chunk_id": r.get("seed_chunk"), "key_concepts": [],
                         "program": r["program"], "unit": r.get("unit", ""), "source": r.get("twist", "teacher")}
                rows.append(r)
        sft_nn = NearestNeighbour([json.loads(l)["instruction"] for l in open(SFT, encoding="utf-8")])
        seen = {json.loads(l)["prompt"].strip() for p in args.also_dedupe for l in open(p, encoding="utf-8")}
        kept, stats = dedupe(rows, sft_nn, excluded_chunks(args.exclude_chunks), seen, args.max_sim)
        with open(args.out, "w", encoding="utf-8") as g:
            g.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
        print(f"{len(rows)} rows -> {len(kept)} candidates -> {args.out} | dropped: {dict(stats)}")
        return 0

    if args.mode == "never":
        rows = [r for p in args.inp for r in load(p) if r.get("passes") == 0 and r.get("program")]
        rng.shuffle(rows)
        rows = rows[:args.limit] if args.limit else rows
        with open(args.out, "w", encoding="utf-8") as g:
            g.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{len(rows)} never-solved candidates with a program -> {args.out}")
        return 0

    # build
    rows = [r for p in args.inp for r in load(p) if 0 < r.get("passes", 0) < r.get("k", 8)]
    rng.shuffle(rows)
    per, pool, seen = collections.Counter(), [], set()
    for r in rows:
        t = template(r["prompt"])
        if per[t] >= args.per_template or r["prompt"].strip() in seen:
            continue
        per[t] += 1
        seen.add(r["prompt"].strip())
        pool.append(r)
    with open(args.out, "w", encoding="utf-8") as g:
        g.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in pool)
    by_src = collections.Counter(Path(p).stem for p in args.inp for r in load(p) if 0 < r.get("passes", 0) < r.get("k", 8))
    print(f"pool: {len(pool)} learnable problems ({len(rows)} before the template cap) -> {args.out} | by file: {dict(by_src)}")
    # coverage of the real held-out questions, and the gain the similarity law predicts (audit §15.17)
    real = [r for r in load(ROOT / "data/rl/test.jsonl") if r.get("type") == "numerical"]
    nn = NearestNeighbour([r["prompt"] for r in pool])
    sims = sorted(nn.best(r["prompt"]) for r in real)
    gain = {(0, .2): 0.0, (.2, .35): 0.02, (.35, .5): 0.10, (.5, 1.01): 0.17}   # measured per-bin gains (pp/100)
    pred = sum(next(g for (lo, hi), g in gain.items() if lo <= s < hi) for s in sims) / len(sims)
    print(f"real held-out vs pool: median nearest-neighbour Jaccard {sims[len(sims) // 2]:.2f}, "
          f">= 0.35: {sum(s >= .35 for s in sims) / len(sims):.0%}, >= 0.5: {sum(s >= .5 for s in sims) / len(sims):.0%}; "
          f"similarity-law prediction for the held-out gain: about +{100 * pred:.1f} pp (upper end: neighbours must be learnable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
