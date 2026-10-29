"""Keep the teaching rewrites (tools/gen_rewrites.py) that are complete, self-contained and consistent.

    python -I scripts/midtrain/filter_rewrites.py work/midtrain/rewrites/rw_*.jsonl --out work/midtrain/rewrites/kept.jsonl

A rewrite is dropped when it
  * was cut off by the token limit,
  * refers to "the excerpt", "the passage" or "the text above" (it is meant to stand alone),
  * (textbook style) states numbers the excerpt does not contain: more than a quarter of its distinct values with two
    or more significant digits, results after "=" excepted,
  * (worked and exam styles) contains a line "<arithmetic> = <number>" whose arithmetic is off by more than 2%.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data/sources/real_style/pool.jsonl"
META = re.compile(r"\b(?:the|this|above|given|provided) (?:excerpt|passage|text above|provided text|source text)\b", re.I)
NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:[eE]|×\s*10\^?|x\s*10\^?)\s*\{?[-−]?\d+\}?)?")
EXPR = re.compile(r"((?:[-−(]?\s*\d[\d.,]*\s*(?:[×x*/+\-−^]\s*)?[)]*\s*){2,})=\s*(-?[\d.,]+(?:\s*(?:[eE]|×\s*10\^?)\s*[-−]?\d+)?)")


def value(s):
    s = s.replace(",", "").replace("−", "-").replace(" ", "").replace("{", "").replace("}", "")
    m = re.match(r"^(-?[\d.]+)(?:(?:[eE]|×10\^?|x10\^?)(-?\d+))?$", s)
    if not m:
        return None
    try:
        return float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1)
    except (ValueError, OverflowError):
        return None


def numbers(text):
    out = set()
    for m in NUM.finditer(text):
        v = value(m.group(0))
        if v is not None and math.isfinite(v):
            out.add(v)
    return out


def significant(v):
    digits = re.sub(r"[^0-9]", "", f"{abs(v):.6g}".split("e")[0]).strip("0")
    return len(digits) >= 2


def grounded(text, source_nums):
    results = set()
    for line in text.splitlines():
        if "=" in line or "≈" in line:
            results |= numbers(re.split(r"[=≈]", line)[-1])
    vals = [v for v in numbers(text) if significant(v) and v not in results]
    if len(vals) < 4:
        return True
    near = lambda v: any(abs(v - s) <= 0.005 * max(abs(v), abs(s)) for s in source_nums)
    return sum(not near(v) for v in vals) <= 0.25 * len(vals)


def arithmetic_ok(text):
    for m in EXPR.finditer(text):
        lhs = m.group(1).replace("×", "*").replace("x", "*").replace("^", "**").replace("−", "-").replace(",", "")
        rhs = value(m.group(2))
        if rhs is None or not re.fullmatch(r"[\d.*/+\-() ]+", lhs) or "**" in lhs:   # no powers: 9**9**9 never ends
            continue
        try:
            got = eval(lhs, {"__builtins__": {}})                       # digits and operators only (checked above)
        except Exception:
            continue
        if isinstance(got, (int, float)) and abs(got - rhs) > 0.02 * max(abs(got), abs(rhs), 1e-12):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for p in a.inputs for r in map(json.loads, open(p, encoding="utf-8"))]
    need = {r["chunk_id"] for r in rows if r["style"] == "textbook"}
    src = {}
    for line in open(POOL, encoding="utf-8"):
        c = json.loads(line)
        if c["chunk_id"] in need:
            src[c["chunk_id"]] = numbers(c["text"])
    why = Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            t = r["text"].strip()
            if r["finish"] == "length":
                why["truncated"] += 1
            elif META.search(t):
                why["refers to the excerpt"] += 1
            elif r["style"] == "textbook" and not grounded(t, src.get(r["chunk_id"], set())):
                why["values not in the excerpt"] += 1
            elif r["style"] != "textbook" and not arithmetic_ok(t):
                why["arithmetic off"] += 1
            else:
                why["kept"] += 1
                f.write(json.dumps({"chunk_id": r["chunk_id"], "style": r["style"], "text": t}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rewrites: " + ", ".join(f"{k} {v}" for k, v in why.most_common()), file=sys.stderr)


if __name__ == "__main__":
    main()
