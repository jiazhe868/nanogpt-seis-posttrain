#!/usr/bin/env python3
"""Exercises written from source-paper excerpts, for the corpus evaluation protocol (README §5.18.8,
audit §15.18).

Under the strict protocol, RL problems come only from SFT-training seeds, so the held-out questions are
about papers the model never practised on. Under the corpus protocol the model may practise on every
paper in its corpus, including the papers the held-out questions were drawn from, but never on the
held-out questions themselves. This has the local teacher write K exercises per excerpt with executable
solution programs, keeps those whose program runs, and drops any exercise that shares a 13-gram with a
held-out question or whose word-Jaccard with one reaches --max-sim (no near-copies of test questions).
Output rows have the gen_pot_problems schema, ready for tools/build_rl_pool.py teacher.

    VLLM_USE_FLASHINFER_SAMPLER=0 python tools/gen_chunk_problems.py --chunks chunks.jsonl \\
        --ids data/rl/test.jsonl data/rl/val.jsonl --per-chunk 8 --out data/sources/pot/chunk_heldout_27b.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_rl_pool import NearestNeighbour  # noqa: E402
from tools.gen_pot_problems import RULES, chat, heldout_ngrams, load_llm, ngrams, parse_json  # noqa: E402
from src.rl.pot import run_program  # noqa: E402

PROMPT = """Here is an excerpt from a geoscience paper:
<excerpt>
{text}
</excerpt>

Write {k} quantitative exercises that a student who has read this paper should be able to solve.
Requirements for each exercise:
- built on the relations, parameter values, units and situations described in the excerpt; vary the numbers and scenarios rather than restating a result from the excerpt;
- fully self-contained: state every number needed (except universal constants such as pi or g = 9.81 m/s^2); never refer to "the study", "the paper", "the excerpt" or "the text";
- the answer must be uniquely determined: state any empirical formula or simplifying rule explicitly in the question;
- realistic values and units; 1 to 4 computation steps;
- ask for exactly ONE final numerical quantity and say which unit it should be in.
If the excerpt has no quantitative content (for example a reference list), return [].

{rules}

Return ONLY a JSON list, no other text:
[{{"question": "...", "program": "...", "unit": "..."}}]"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunks", required=True, help="JSONL of {chunk_id, title, text}")
    ap.add_argument("--ids", nargs="+", required=True, help="files whose chunk_ids select the excerpts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-chunk", type=int, default=8)
    ap.add_argument("--max-sim", type=float, default=0.5, help="drop exercises this close to a held-out question")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    want = set()
    for p in a.ids:
        for line in open(p, encoding="utf-8"):
            c = json.loads(line).get("chunk_id")
            if c and c != "None":
                want.add(c)
    chunks = [r for r in map(json.loads, open(a.chunks, encoding="utf-8")) if r["chunk_id"] in want]
    args = argparse.Namespace(model=a.model, tp=a.tp, quantization=None, gpu_util=0.92, max_len=8192, max_tokens=3000,
                              temperature=0.8, eager=False, max_num_seqs=64, max_batched_tokens=4096, seed=a.seed)
    llm, sp = load_llm(args)
    texts = chat(llm, sp, [PROMPT.format(text=c["text"][:7000], k=a.per_chunk, rules=RULES) for c in chunks])
    held = heldout_ngrams()
    heldq = [json.loads(l)["prompt"] for f in ("data/rl/test.jsonl", "data/rl/val.jsonl")
             for l in open(ROOT / f, encoding="utf-8")]
    nn = NearestNeighbour(heldq)
    stats, kept = collections.Counter(), 0
    with open(a.out, "w", encoding="utf-8") as g:
        for c, t in zip(chunks, texts):
            items = parse_json(t)
            if not isinstance(items, list):
                stats["unparsable"] += 1
                continue
            for it in items:
                if not isinstance(it, dict) or not it.get("question") or not it.get("program"):
                    stats["malformed"] += 1
                    continue
                ans, _ = run_program(it["program"])
                if ans is None:
                    stats["program failed"] += 1
                    continue
                if ngrams(it["question"]) & held:
                    stats["13-gram overlap with a held-out question"] += 1
                    continue
                if nn.best(it["question"]) >= a.max_sim:
                    stats["near-copy of a held-out question"] += 1
                    continue
                g.write(json.dumps({"question": it["question"], "program": it["program"], "answer": ans,
                                    "unit": it.get("unit", ""), "seed_chunk": c["chunk_id"], "twist": "corpus"},
                                   ensure_ascii=False) + "\n")
                kept += 1
    print(f"{len(chunks)} excerpts -> {kept} exercises -> {a.out} | dropped: {dict(stats)}")
    return 0


if __name__ == "__main__":   # vLLM spawns TP workers that re-import this file
    raise SystemExit(main())
