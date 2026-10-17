"""Counterfactual numeric augmentation: re-instantiate real questions with new numbers.

Diagnosis (audit §15.9): the SFT model scores 5% on numerical questions it was TRAINED on, the
same as on held-out ones, and 3-5% on questions that need one operation on two given numbers.
Its derivations are not bound to the question's values. Training on the same 3,705 references
again cannot fix that: the answer can be memorised without reading the numbers.

This module extracts each reference derivation as a program -- its verified `EXPR = VALUE`
steps, with every operand traced to a value given in the question, an earlier step, or a
constant -- then perturbs the given values, re-evaluates every step exactly, and rewrites the
question, the reasoning and the final answer with the new numbers. The question's wording,
context and distractors stay real; only the numbers change, so the answer can no longer be
recalled, only computed from the question. Arithmetic is written as calculator calls
(`<<expr>> = value`, src.rl.rollout) and the final answer opens by stating the result.

A variant is kept only if every number in the new text is accounted for: rewritten, or provably
independent of the changed values. Anything ambiguous is dropped, not guessed.

    python -m src.sft.counterfactual data/splits/train.jsonl data/splits/train_cf.jsonl --variants 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

from src.rl.numerical_target import derive_numerical_target, given_values, normalize_scientific_notation
from src.rl.verifiers import safe_eval_number
from src.sft.calculator_format import EXPR_CHARS, NUM, to_python

LIT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e[-+]?\d+)?")
# a plain number as written in text: 1,000,000 / 2.8 / 50000 / 0.0536 (no scientific notation)
SCI = re.compile(r"\d+(?:\.\d+)?[eE][-+]?\d+")
TEXT_NUM = re.compile(r"(?<![\w.^/])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w^])")


def close(a: float, b: float, tol: float) -> bool:
    d = abs(b) if abs(b) > 1e-12 else 1.0
    return abs(a - b) / d <= tol


def fmt_like(v: float, template: str) -> str:
    """Format v with the template's decimals and thousands separators."""
    dec = len(template.split(".")[1]) if "." in template else 0
    s = f"{v:,.{dec}f}" if "," in template else f"{v:.{dec}f}"
    return s


def fmt_result(v: float) -> str:
    return f"{v:.4g}"


def find_steps(text: str) -> List[Dict]:
    """Verified computation steps `EXPR = VALUE` (same rule as src.sft.calculator_format)."""
    out = []
    for m in re.finditer(rf"\s*[=≈]\s*({NUM})", text):
        lo = m.start()
        while lo > 0 and text[lo - 1] in EXPR_CHARS:
            lo -= 1
        raw = text[lo:m.start()]
        k = 0
        while k < len(raw) and not (raw[k].isdigit() or raw[k] == "(" or raw[k:k + 3] in ("log", "sqr", "exp")):
            k += 1
        expr = raw[k:].rstrip()
        py = to_python(expr) if expr else None
        if py is None:
            continue
        v = safe_eval_number(py)
        try:
            stated = float(m.group(1))
        except ValueError:
            continue
        if v is None or not close(v, stated, 0.02):
            continue
        out.append({"py": py, "value": v, "start": lo + k, "end": m.end()})
    return out


class Program:
    """Steps of one reference derivation, with operand provenance."""

    def __init__(self, prompt: str, think: str, target: float):
        self.prompt, self.think, self.target = prompt, think, target
        self.given = sorted(set(given_values(prompt)))
        self.steps = find_steps(think)
        known: List[float] = []
        for s in self.steps:
            s["ops"] = []
            for m in LIT.finditer(s["py"]):
                o = float(m.group())
                if any(close(o, g, 1e-3) for g in self.given):
                    kind, ref = "given", min(self.given, key=lambda g: abs(g - o))
                elif any(close(o, k, 5e-3) for k in known):
                    kind, ref = "step", max(i for i, k in enumerate(known) if close(o, k, 5e-3))
                else:
                    kind, ref = "const", o
                s["ops"].append((m.span(), kind, ref))
            known.append(s["value"])
        hits = [i for i, s in enumerate(self.steps) if close(s["value"], target, 0.02)]
        self.target_step = hits[-1] if hits else None

    @property
    def usable(self) -> bool:
        if self.target_step is None:
            return False
        return any(kind != "const" for _, kind, _ in self.steps[self.target_step]["ops"])

    def run(self, new_given: Dict[float, float], forms: Optional[Dict[float, str]] = None
            ) -> Optional[List[Tuple[str, float]]]:
        """Re-evaluate every step with substituted given values -> [(new_py, new_value)].

        Operands are written the way the text shows them -- a given value in the question's
        own form (`forms`), an earlier result as its 4-significant-figure statement -- so that
        binding is a verbatim copy, and later steps compute from the rounded value as written."""
        vals: List[float] = []
        out = []
        for s in self.steps:
            py, shift = s["py"], 0
            for (a, b), kind, ref in s["ops"]:
                if kind == "given":
                    nv = new_given.get(ref, ref)
                    rep = fmt_like(nv, forms[ref]) if forms and ref in forms else repr(nv)
                elif kind == "step":
                    rep = fmt_result(vals[ref])
                else:
                    continue
                py = py[:a + shift] + rep + py[b + shift:]
                shift += len(rep) - (b - a)
            v = safe_eval_number(re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", py))
            if v is None or not math.isfinite(v):
                return None
            vals.append(float(fmt_result(v)))
            out.append((py, v))
        return out


def perturb(v: float, template: str, rng: random.Random) -> float:
    """A new value of the same kind: same sign, decimals and magnitude band, never equal."""
    dec = len(template.split(".")[1]) if "." in template else 0
    for _ in range(20):
        w = v * math.exp(rng.uniform(-0.6, 0.6))
        w = round(w, dec)
        if w != v and w > 0 and (dec > 0 or w >= 1):
            return w
    return v


def rewrite_numbers(text: str, mapping: List[Tuple[float, float]], protect: List[Tuple[int, int]] = ()) -> Optional[str]:
    """Replace every plain number in `text` that matches an old value (2%) with the new value in
    the same format. Returns None if a number matches two different changed values."""
    out, pos = [], 0
    for m in TEXT_NUM.finditer(text):
        if any(a <= m.start() < b for a, b in protect):
            continue
        s = m.group(0)
        try:
            x = float(s.replace(",", ""))
        except ValueError:
            continue
        cands = {round(new, 12) for old, new in mapping if close(x, old, 0.02) and abs(old) > 0}
        if not cands:
            continue
        if len(cands) > 1:
            return None
        out.append(text[pos:m.start()])
        out.append(fmt_like(cands.pop(), s))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def make_variant(rec: Dict, prog: Program, rng: random.Random) -> Optional[Dict]:
    prompt, think = prog.prompt, prog.think
    final = rec["response"].split("</think>", 1)[1]
    # given values the program actually uses, with their text form in the question
    used = sorted({ref for s in prog.steps for _, kind, ref in s["ops"] if kind == "given"})
    forms = {}
    for m in TEXT_NUM.finditer(prompt):
        x = float(m.group(0).replace(",", ""))
        for g in used:
            if close(x, g, 1e-9):
                forms.setdefault(g, m.group(0))
    if set(forms) != set(used):
        return None                       # a used value is written in a form we do not rewrite
    # constants must not collide with a changed value (it would be rewritten in prose)
    consts = [ref for s in prog.steps for _, kind, ref in s["ops"] if kind == "const"]
    if any(close(c, g, 0.02) for c in consts for g in used):
        return None
    new_given = {g: perturb(g, forms[g], rng) for g in used}
    if all(new_given[g] == g for g in used):
        return None
    run = prog.run(new_given, forms)
    if run is None:
        return None
    new_target = run[prog.target_step][1]
    if not math.isfinite(new_target) or new_target == 0:
        return None
    old_vals = [(g, new_given[g]) for g in used] + [(s["value"], v) for s, (_, v) in zip(prog.steps, run)
                                                     if not close(s["value"], v, 1e-9)]
    # values in scientific notation are not rewritten: if one depends on a changed value it
    # would go stale, so the variant is dropped
    for txt in (think, final):
        for m in SCI.finditer(normalize_scientific_notation(txt)):
            x = float(m.group())
            if any(close(x, old, 0.02) for old, _ in old_vals):
                return None
    # 1. question
    new_prompt = rewrite_numbers(prompt, [(g, new_given[g]) for g in used])
    # 2. reasoning: each verified step becomes a calculator call with the new operands
    parts, pos = [], 0
    for s, (py, v) in zip(prog.steps, run):
        parts.append(think[pos:s["start"]])
        parts.append(f"<<{py}>> = {fmt_result(v)}")
        pos = s["end"]
    parts.append(think[pos:])
    # prose between steps: rewrite restated values; step spans are already exact
    spans, cur = [], 0
    rebuilt = ""
    for chunk_i, chunk in enumerate(parts):
        if chunk_i % 2 == 1:
            rebuilt += chunk
        else:
            r = rewrite_numbers(chunk, old_vals)
            if r is None:
                return None
            rebuilt += r
    new_think = rebuilt
    new_final = rewrite_numbers(final, old_vals)
    if new_prompt is None or new_final is None:
        return None
    # every value in the new final answer that matched an old changed value was rewritten; the
    # answer must now state the new target
    if not any(close(x, new_target, 0.02) for x in given_values(new_final) + [float(t.replace(",", "")) for t in
                                                                           re.findall(r"\d[\d,]*\.?\d*", new_final)]):
        return None
    # the repo's own label rule must recover the new target from the rewritten text
    t2, _ = derive_numerical_target(new_prompt, new_think, new_final)
    if t2 is None or not close(t2, new_target, 0.02):
        return None
    answer = f"Answer: {fmt_result(new_target)}. "
    return {
        "instruction": new_prompt,
        "response": f"<think>{new_think}</think>{answer}{new_final.lstrip()}",
        "type": "numerical",
        "chunk_id": rec.get("chunk_id"),
        "counterfactual_of": rec.get("chunk_id"),
        "target": new_target,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-key", default="instruction")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = n_prog = n_out = 0
    with open(args.src, encoding="utf-8") as f, open(args.dst, "w", encoding="utf-8") as g:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "numerical":
                continue
            if args.prompt_key != "instruction":
                rec = {"instruction": rec[args.prompt_key], "chunk_id": rec.get("chunk_id"),
                       "response": f"<think>{rec.get('reference_think', '')}</think>{rec.get('reference_final_answer', '')}"}
            if "</think>" not in rec["response"]:
                continue
            n += 1
            think, final = rec["response"].split("</think>", 1)
            think = think.replace("<think>", "")
            t, _ = derive_numerical_target(rec["instruction"], think, final)
            if t is None:
                continue
            prog = Program(rec["instruction"], think, t)
            if not prog.usable:
                continue
            n_prog += 1
            seen = set()
            for _ in range(args.variants * 3):
                if len(seen) >= args.variants:
                    break
                v = make_variant(rec, prog, rng)
                if v is None or v["instruction"] in seen:
                    continue
                seen.add(v["instruction"])
                g.write(json.dumps(v, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"{n} numerical pairs | {n_prog} with a usable program | {n_out} counterfactual variants -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
