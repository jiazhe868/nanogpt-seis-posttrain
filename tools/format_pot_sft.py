#!/usr/bin/env python3
"""Teacher program-of-thought rows -> student SFT pairs.

    <think>
    b_value = 1.0  # dimensionless
    ...
    answer = freq_m7
    </think>Answer: 0.005 events per year.

At inference the decoder (src.rl.rollout, pot=True) executes the program and forces
"Answer: <value>"; the student writes the program and the unit. Rows are kept only if the
program re-executes to the stored answer, and (for generated rows, with --require-verified) the
independent re-solve agreed.

    python tools/format_pot_sft.py --out data/splits/train_pot.jsonl data/sources/pot/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.pot import run_program  # noqa: E402


def close(a, b, tol=1e-6):
    d = abs(b) if abs(b) > 1e-12 else 1.0
    return abs(a - b) / d <= tol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-verified", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--notation", action="store_true", help="programs are in paper notation (src.rl.pot)")
    args = ap.parse_args()
    n = kept = 0
    seen = set()
    with open(args.out, "w", encoding="utf-8") as g:
        for path in args.inputs:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                n += 1
                if args.require_verified and r.get("source") != "convert" and not r.get("verified"):
                    continue
                prog = r["program"].strip()
                a, _ = run_program(prog, notation=args.notation)
                if a is None or not math.isfinite(a) or not close(a, r["answer"]):
                    continue
                q = r["question"].strip()
                if q in seen:
                    continue
                seen.add(q)
                unit = (r.get("unit") or "").strip().rstrip(".")
                unit = "" if unit.lower() in ("", "dimensionless", "none", "-", "unitless") else " " + unit
                g.write(json.dumps({"instruction": q, "response": f"<think>\n{prog}\n</think>Answer: {a:.4g}{unit}.",
                                    "type": "numerical", "chunk_id": r.get("seed_chunk") or r.get("chunk_id"),
                                    "source": r.get("source", "generated"), "target": a}, ensure_ascii=False) + "\n")
                kept += 1
                if args.limit and kept >= args.limit:
                    break
    print(f"{n} rows -> {kept} SFT pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
