"""Rewrite the arithmetic in reference solutions into calculator calls.

Drill-only training taught the calculator format (86% on held-out drills with the calculator,
5% without) but not the habit: on real questions only 12% of completions used it, and they
fell back to mental arithmetic after the first step, because the replayed paper data still
demonstrates mental arithmetic. This converts that paper data to the same convention.

A chain step `EXPR = VALUE` (or `≈ VALUE`) is rewritten to `<<EXPR'>> = fmt(v)` only when
EXPR is pure arithmetic (numbers, + − × · / ^, parentheses, log10 / ln / sqrt / exp, π) and
`safe_eval_number(EXPR')` reproduces VALUE within 2% — so a reference's wrong arithmetic is
left as written, never blessed by a calculator call.

    python -m src.sft.calculator_format data/splits/train.jsonl data/splits/train_calcfmt.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from typing import Optional, Tuple

from src.rl.verifiers import safe_eval_number

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
# characters an arithmetic expression (in the references' notation) may contain
EXPR_CHARS = set("0123456789.eE+-−*/×·^() πlogsqrtexpn")
SCI = re.compile(r"(\d(?:\.\d+)?)\s*[×x·*]\s*10\s*\^\s*\(?\s*([-+−]?\d+)\s*\)?")


def to_python(expr: str) -> Optional[str]:
    """Reference notation -> an expression safe_eval_number accepts, or None."""
    e = expr.strip().replace("−", "-").replace("π", "pi")
    e = SCI.sub(lambda m: f"{m.group(1)}e{m.group(2).replace('−', '-')}", e)
    e = e.replace("×", "*").replace("·", "*").replace("^", "**")
    e = re.sub(r"\bln\(", "log(", e)
    if not re.search(r"[*/+]|\*\*|(?<=[\d)])\s*-\s*(?=[\d(])|log|sqrt|exp", e):
        return None                       # a bare number, not a computation
    if not re.match(r"^[\d(.]|^(?:log10|log|sqrt|exp)\(", e):
        return None
    return e


def convert_text(text: str) -> Tuple[str, int]:
    """(rewritten text, number of calls). Scans each '=' / '≈' and the arithmetic before it."""
    out, calls, pos = [], 0, 0
    for m in re.finditer(rf"\s*[=≈]\s*({NUM})", text):
        start = m.start()
        lo = start
        while lo > pos and text[lo - 1] in EXPR_CHARS:
            lo -= 1
        raw = text[lo:start]
        # trim to the expression proper: drop leading words/operators/spaces
        k = 0
        while k < len(raw) and not (raw[k].isdigit() or raw[k] == "(" or raw[k:k + 3] in ("log", "sqr", "exp")):
            k += 1
        expr = raw[k:].rstrip()
        py = to_python(expr) if expr else None
        if py is None:
            continue
        val = safe_eval_number(py)
        try:
            stated = float(m.group(1))
        except ValueError:
            continue
        if val is None or abs(val - stated) > 0.02 * max(abs(stated), 1e-12):
            continue
        a = lo + k
        out.append(text[pos:a])
        out.append(f"<<{py}>> = {val:.4g}")
        pos = m.end()
        calls += 1
    out.append(text[pos:])
    return "".join(out), calls


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    n = n_num = n_conv = calls = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if r.get("type") == "numerical":
                n_num += 1
                new, c = convert_text(r["response"])
                if c:
                    r["response"] = new
                    r["calculator_calls"] = c
                    n_conv += 1
                    calls += c
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n} pairs | numerical {n_num} | converted {n_conv} ({n_conv / max(n_num, 1):.0%}) | calls {calls} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
