#!/usr/bin/env python3
"""Counterfactual variants of REAL questions from their program-of-thought solutions.

The scaling run (audit §15.10) showed teacher-written exercises move accuracy on more exercises
of the same kind (3k -> 30k: 4% -> 21%) far more than on the real held-out questions (must-compute
2% -> 4%): real questions are long, multi-part and phrased like their source papers. The 2,334
real training questions converted to programs (tools/gen_pot_problems.py convert) give clean,
executable solutions in that style: every given value is a named assignment copied from the
question. Perturbing those values in both the question and the program, then re-executing,
yields new real-style questions with exact labels -- no teacher needed. Compared with
src.sft.counterfactual (which parses the prose derivation), the program makes the dependency
structure explicit, so far more questions qualify.

A value is perturbed only if its literal occurs exactly once in the question in a plain or
thousands-separated form; constants the question does not state (e.g. an assumed rigidity) and
values written in scientific notation are left alone.

    python tools/pot_counterfactuals.py data/sources/pot/convert_27b.jsonl \\
        data/sources/pot/convert_cf.jsonl --variants 6
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.pot import THOUSANDS, run_program  # noqa: E402

# formula constants stay fixed: changing "Mw = (2/3) log10(M0) - 10.7" to "- 13.1" is internally
# consistent but contradicts standard seismology the model should keep
PROTECT_NAME = re.compile(r"const|coef|offset|intercept|slope|exponent", re.I)
PROTECT_COMMENT = re.compile(r"relationship|relation\b|formula|equation|empirical|constant|coefficient", re.I)
# a given value: `name = 12.5  # unit` (Python) or `μ = 3.2 [Pa]` (paper notation, --notation)
ASSIGN = re.compile(r"^(\s*)([^\W\d]\w*)(\s*=\s*)([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)(\s*(?:#.*|\[[^\]]*\])?)$")
NOTATION = False


def forms(v: float, literal: str):
    """Ways the question may write the value of a program literal."""
    out = {literal}
    try:
        x = float(literal.replace(",", ""))
    except ValueError:
        return out
    if x.is_integer() and "e" not in literal.lower():
        xi = int(x)
        out |= {str(xi), f"{xi:,}"}
    else:
        out.add(f"{x:g}")
    return {f for f in out if f}


def perturb(literal: str, rng: random.Random) -> str:
    x = float(literal.replace(",", ""))
    dec = len(literal.split(".")[1]) if "." in literal and "e" not in literal.lower() else 0
    for _ in range(20):
        y = round(x * math.exp(rng.uniform(-0.5, 0.5)), dec)
        if y != x and (y > 0 or x <= 0):
            if dec == 0:
                y = int(y)
                return f"{y:,}" if "," in literal else str(y)
            return f"{y:.{dec}f}"
    return literal


def in_formula(q: str, f: str) -> bool:
    """Is the literal inside a formula written in the question (after '=' in a span with log/^/exp)?"""
    m = re.search(rf"(?<![\w.,]){re.escape(f)}(?![\w]|[.,]\d)", q)
    if not m:
        return False
    left = q[max(0, m.start() - 60):m.start()]
    seg = re.split(r"[.;:]\s|\n", left)[-1]
    return "=" in seg and bool(re.search(r"log|\^|\*\*|exp|sqrt|×\s*10", seg))


def variant(q: str, prog: str, rng: random.Random):
    lines = prog.split("\n")
    edits, qnew = [], q
    for k, line in enumerate(lines):
        m = ASSIGN.match(line)
        if not m:
            continue
        lit = m.group(4)
        if PROTECT_NAME.search(m.group(2)) or PROTECT_COMMENT.search(m.group(5) or ""):
            continue
        hits = [f for f in forms(0, lit)
                if len(re.findall(rf"(?<![\w.,]){re.escape(f)}(?![\w]|[.,]\d)", q)) == 1]
        if len(hits) != 1:
            continue                                   # absent from the question, or ambiguous
        f = hits[0]
        if in_formula(q, f):
            continue
        new = perturb(f, rng)
        if new == f:
            continue
        qnew = re.sub(rf"(?<![\w.,]){re.escape(f)}(?![\w]|[.,]\d)", new, qnew, count=1)
        lines[k] = f"{m.group(1)}{m.group(2)}{m.group(3)}{new}{m.group(5)}"
        edits.append((f, new))
    if not edits:
        return None
    p2 = "\n".join(lines)
    a, _ = run_program(p2, notation=NOTATION)
    if a is None or not math.isfinite(a):
        return None
    return qnew, p2, a, edits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notation", action="store_true", help="programs are in paper notation (src.rl.pot)")
    args = ap.parse_args()
    global NOTATION
    NOTATION = args.notation
    rng = random.Random(args.seed)
    n = usable = out_n = 0
    with open(args.dst, "w", encoding="utf-8") as g:
        for line in open(args.src, encoding="utf-8"):
            r = json.loads(line)
            n += 1
            a0, _ = run_program(r["program"], notation=NOTATION)
            if a0 is None:
                continue
            seen = set()
            for _ in range(args.variants * 3):
                if len(seen) >= args.variants:
                    break
                v = variant(r["question"], r["program"], rng)
                if v is None:
                    continue
                q2, p2, a2, edits = v
                if q2 in seen or abs(a2 - a0) <= 1e-9 * max(abs(a0), 1.0):
                    continue                           # the answer must depend on the edit
                seen.add(q2)
                g.write(json.dumps({"question": q2, "program": p2, "answer": a2, "unit": r.get("unit", ""),
                                    "chunk_id": r.get("chunk_id"), "source": "convert_cf", "verified": True,
                                    "edits": edits}, ensure_ascii=False) + "\n")
                out_n += 1
            usable += bool(seen)
    print(f"{n} converted real questions | {usable} with a perturbable given value | {out_n} variants -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
