#!/usr/bin/env python3
"""Program-of-thought training data from a local open-weight teacher (vLLM, offline).

Why (audit §15.9): binding a question's values to a relation is learnable by the 113M model near
its training wordings but does not transfer to new question types; ~1,250 question types are far
too few. Small models that do well on numerical word problems were trained on very many problems
with executable solutions. This script makes that data on this server, with an Apache-2.0 model
(Qwen3.8-27B-FP8 / Qwen3.5-9B), and checks every label by execution (src.rl.pot.run_program).

Modes (all write JSONL):
  notation  rewrite existing program solutions in paper notation (SYMBOL = expression [unit], short
            symbols, ^ powers); kept only if they execute (src.rl.pot, notation=True) to the same answer.
  rewrite   new questions in the style of each real numerical training question (same multi-part
            structure and kind of computation, new scenario and numbers), with program solutions.
  generate  new self-contained exercises seeded by TRAINING questions (data/splits/train.jsonl),
            each with a program solution; kept only if the program runs.
  solve     re-solve exercises from the question alone; a problem is "verified" when the two
            independently executed answers agree within 1%.
  convert   the real SFT numerical questions rewritten as programs, kept only if the program
            reproduces the repo's label (src.rl.numerical_target).
  audit     held-out questions: is all the information needed in the question? (+ program)

Decontamination: seeds come only from the SFT train split, whose source chunks are disjoint from
every held-out split, and any output sharing a 13-word sequence with a held-out prompt is dropped
(8-word windows also caught shared formula strings such as "Mw = (2/3) log10(M0) - 10.7").

    python tools/gen_pot_problems.py generate --model Qwen/Qwen3.8-27B-FP8 --tp 2 \\
        --n-seeds 300 --per-seed 3 --out data/sources/pot_pilot_27b.jsonl
"""

from __future__ import annotations

import argparse
import os

# vLLM's FlashInfer sampler JIT-compiles with the system nvcc (11.5 here, below FlashInfer's 12);
# the PyTorch sampler is used instead. Any other JIT gets the CUDA 12.5 toolkit.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
if os.path.isdir("/usr/local/cuda-12.5"):
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.5")
    os.environ["PATH"] = "/usr/local/cuda-12.5/bin:" + os.environ.get("PATH", "")
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.pot import extract_program, run_program  # noqa: E402

HELDOUT_FILES = ["data/rl/test.jsonl", "data/rl/val.jsonl", "data/splits/val.jsonl", "data/splits/test.jsonl"]
TWISTS = [
    "a field observation after an earthquake", "an engineering design check", "a seismic hazard assessment",
    "a laboratory rock-mechanics measurement", "an earthquake early-warning scenario", "a comparison of two events",
    "an inverse problem: solve for an input quantity given the result", "a unit conversion between SI and other units",
    "a historical earthquake", "a seismic monitoring network", "a tsunami or coastal impact", "aftershock statistics",
    "fault mechanics and slip", "wave propagation and travel times", "ground motion and site amplification",
    "earthquake recurrence and probability", "induced seismicity", "building or bridge response",
    "loss and casualty estimation", "geodesy and crustal deformation",
]
RULES = """Rules for the solution program:
- one assignment per line, descriptive snake_case names, numbers copied exactly as written in the question;
- a short comment with the unit after each given value;
- only arithmetic (+ - * / **) and math functions log10, log, exp, sqrt, sin, cos, tan, asin, acos, atan, radians, degrees, abs, min, max, and the constants pi and e;
- no imports, loops, conditionals, lists or prints;
- the last line must be `answer = ...` (the single requested quantity, in the unit you state);
  if the question asks for several quantities, `answer` is the LAST quantity it asks for."""
GEN_PROMPT = """Here is an exercise from a seismology and earthquake-engineering course:
<example>
{seed}
</example>

Write {k} NEW exercises on related but clearly different situations, framed as {twist}.
Requirements for each exercise:
- about seismology, earthquake physics or earthquake engineering quantities (not generic logistics or finance);
- fully self-contained: state every number needed (except universal constants such as pi or g = 9.81 m/s^2); never refer to "the study", "the paper", "the text" or "the data above";
- the answer must be uniquely determined: use standard, well-established relations, and state any simplifying rule or empirical formula explicitly in the question rather than leaving it implied;
- realistic values and units; {steps} computation step(s);
- ask for exactly ONE final numerical quantity and say which unit it should be in.

{rules}

Return ONLY a JSON list, no other text:
[{{"question": "...", "program": "...", "unit": "..."}}]"""
SOLVE_PROMPT = """Solve this exercise with a short Python program.

<exercise>
{q}
</exercise>

{rules}

Return ONLY the program inside a ```python block."""
CONVERT_PROMPT = """Here is an exercise and a reference solution written in prose.

<exercise>
{q}
</exercise>
<reference_solution>
{ref}
</reference_solution>

Rewrite the computation the exercise asks for as a short Python program whose `answer` is the final
quantity the reference solution concludes. If the exercise asks for several quantities, compute the
one the reference states last as its result.

{rules}

Return ONLY the program inside a ```python block."""
REWRITE_PROMPT = """Here is a quantitative question from a seismology and earthquake-engineering exam, with the
final part of its model answer.

<question>
{q}
</question>
<model_answer>
{ref}
</model_answer>

Write {k} NEW questions in the same style and at the same difficulty: the same kind of context (a named
earthquake, site, structure, network or study setting -- invent new ones), the same multi-part structure and
phrasing conventions, and the same kind of computation, but a different scenario and different numbers.
Requirements for each question:
- every number needed is stated in the question (or is a standard constant); never "from the study" or "the text";
- state any simplifying rule or empirical formula explicitly so that the answer is uniquely determined;
- it may ask for several quantities (and for brief discussion) like the original, but the program's `answer`
  must be the LAST quantity the question asks for.

{rules}

Return ONLY a JSON list, no other text:
[{{"question": "...", "program": "...", "unit": "..."}}]"""
NOTATION_PROMPT = """Rewrite the worked solution below in the notation of a seismology paper.

<question>
{q}
</question>
<solution>
{prog}
</solution>

Rules:
- one equation per line, in the form: SYMBOL = expression [unit]
- SYMBOL is a short standard physics symbol (for example Mw, M0, A, L, W, D, μ, v, t, R, f, E, σ, Δσ, ρ),
  optionally with a subscript after an underscore (v_p, t_s, R_hyp, A_km2); at most 8 characters;
- write each given number exactly as it appears in the question;
- write powers with ^ (10^19, 10^(1.5*Mw + 9.1)) and multiplication with *; use log10 for base-10 and ln for
  natural logarithms, never a bare log;
- put the unit in square brackets at the end of every line ([m], [N m], [km/s], [-] if dimensionless);
- no comments, no code, no prose; keep exactly the same computation; the last line is the requested quantity.

Return ONLY the equations."""
NOTATION_LINE = re.compile(r"^\s*([^\W\d]\w*)\s*=\s*\S.*$")
NOTATION_FORBIDDEN = re.compile(r"\bimport\b|\bprint\b|\bdef\b|#|[\"']|\bmath\.")
TRAILING_ANSWER = re.compile(r"^\s*answer\s*=\s*[^\W\d]\w*\s*(\[[^\]]*\])?\s*$")


def to_notation(text: str):
    """Teacher output -> (equations, reason). Equations only if every line is SYMBOL = expr [unit]
    with a symbol of at most 8 characters and nothing code-like."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]
    if lines and TRAILING_ANSWER.match(lines[-1]):
        lines = lines[:-1]
    if not lines:
        return None, "empty"
    for l in lines:
        mm = NOTATION_LINE.match(l)
        if not mm or NOTATION_FORBIDDEN.search(l):
            return None, "bad_format"
        if len(mm.group(1)) > 8:
            return None, "long_symbol"
    return "\n".join(lines), "ok"


AUDIT_PROMPT = """Here is an exam question and the answer key's value.

<question>
{q}
</question>
<answer_key>{t}</answer_key>

Can a strong student compute the answer key's value from the information in the question plus standard
textbook seismology (well-known formulas and constants)? Answer with a JSON object:
{{"self_contained": true/false, "missing": "what information is missing, or empty", "program": "a Python program computing the value if self_contained, else empty"}}
{rules}
Return ONLY the JSON object."""


def ngrams(text: str, n: int = 13):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def heldout_ngrams():
    out = set()
    for f in HELDOUT_FILES:
        p = ROOT / f
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            out |= ngrams(r.get("prompt") or r.get("instruction") or "")
    return out


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def close(a, b, tol):
    d = abs(b) if abs(b) > 1e-12 else 1.0
    return abs(a - b) / d <= tol


def load_llm(args):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_util, quantization=args.quantization,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=args.seed, enforce_eager=args.eager,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_batched_tokens)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
    return llm, sp


def chat(llm, sp, prompts, think=False):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": think}, use_tqdm=True)
    texts = [o.outputs[0].text for o in outs]
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[gen] {len(prompts)} prompts, {ntok} tokens in {time.time() - t0:.0f}s "
          f"({ntok / max(time.time() - t0, 1e-6):.0f} tok/s)", flush=True)
    return texts


def numerical_seeds(n, seed, pool="numerical"):
    """Training questions to seed from. pool="all" also uses conceptual questions as topic seeds:
    the teacher writes numerical exercises about their subject, which widens topic coverage from
    the 3.7k numerical questions to all 14k training questions."""
    sys.path.insert(0, str(ROOT))
    from src.rl.numerical_target import derive_numerical_target
    recs = []
    for r in map(json.loads, open(ROOT / "data/splits/train.jsonl", encoding="utf-8")):
        if pool == "all" and r["type"] != "numerical":
            recs.append((r, None))
            continue
        if r["type"] != "numerical" or "</think>" not in r["response"]:
            continue
        th, fin = r["response"].split("</think>", 1)
        t, _ = derive_numerical_target(r["instruction"], th.replace("<think>", ""), fin)
        recs.append((r, t))
    random.Random(seed).shuffle(recs)
    return recs[:n] if n else recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["generate", "rewrite", "solve", "convert", "audit", "notation"])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 to quantize a bf16 model at load time")
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--eager", action="store_true")
    # Qwen3.5/3.8 interleave linear-attention (GDN) layers whose recurrent state is allocated per
    # sequence: 256 concurrent sequences do not fit next to 27B-FP8 weights on 24 GB GPUs.
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-batched-tokens", type=int, default=4096)
    ap.add_argument("--think", action="store_true", help="enable the teacher's thinking mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=300)
    ap.add_argument("--per-seed", type=int, default=3)
    ap.add_argument("--seed-pool", choices=["numerical", "all"], default="numerical")
    ap.add_argument("--inp", help="solve / notation: JSONL of problems with programs")
    ap.add_argument("--limit", type=int, default=0, help="notation: only the first N rows")
    ap.add_argument("--chunk", type=int, default=8000, help="notation: prompts per vLLM call (saved after each)")
    ap.add_argument("--hint", default="", help="notation: an extra instruction appended to the prompt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    llm, sp = load_llm(args)
    rules = RULES
    out = open(args.out, "w", encoding="utf-8")
    stats = {}

    if args.mode in ("generate", "rewrite"):
        held = heldout_ngrams()
        seeds = numerical_seeds(args.n_seeds, args.seed, args.seed_pool if args.mode == "generate" else "numerical")
        prompts, meta = [], []
        for r, _ in seeds:
            if args.mode == "rewrite":
                fin = r["response"].split("</think>", 1)[-1].strip()
                prompts.append(REWRITE_PROMPT.format(q=r["instruction"], ref=fin[:1200], k=args.per_seed, rules=rules))
                meta.append({"seed_chunk": r.get("chunk_id"), "twist": "rewrite", "steps": "as original"})
                continue
            twist = rng.choice(TWISTS)
            steps = rng.choice(["1", "2", "2 to 3", "3 to 4"])
            prompts.append(GEN_PROMPT.format(seed=r["instruction"], k=args.per_seed, twist=twist, steps=steps, rules=rules))
            meta.append({"seed_chunk": r.get("chunk_id"), "twist": twist, "steps": steps})
        texts = chat(llm, sp, prompts, think=args.think)
        for m, t in zip(meta, texts):
            items = parse_json(t)
            if not isinstance(items, list):
                stats["unparsable"] = stats.get("unparsable", 0) + 1
                continue
            for it in items:
                if not isinstance(it, dict) or not it.get("question") or not it.get("program"):
                    stats["malformed"] = stats.get("malformed", 0) + 1
                    continue
                ans, env = run_program(it["program"])
                if ans is None:
                    stats["program_failed"] = stats.get("program_failed", 0) + 1
                    continue
                if ngrams(it["question"]) & held:
                    stats["contaminated"] = stats.get("contaminated", 0) + 1
                    continue
                stats["kept"] = stats.get("kept", 0) + 1
                out.write(json.dumps({"question": it["question"], "program": extract_program(it["program"]),
                                      "unit": it.get("unit", ""), "answer": ans, **m}, ensure_ascii=False) + "\n")

    elif args.mode == "notation":
        rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
        if args.limit:
            rows = rows[:args.limit]
        for c0 in range(0, len(rows), args.chunk):
            part = rows[c0:c0 + args.chunk]
            texts = chat(llm, sp, [NOTATION_PROMPT.format(q=r["question"], prog=r["program"]) +
                                   (("\n\n" + args.hint) if args.hint else "") for r in part], think=args.think)
            for r, t in zip(part, texts):
                eqs, why = to_notation(t)
                if eqs is None:
                    stats[why] = stats.get(why, 0) + 1
                    continue
                a, _ = run_program(eqs, notation=True)
                if a is None:
                    stats["does_not_run"] = stats.get("does_not_run", 0) + 1
                    continue
                if not close(a, r["answer"], 1e-3):
                    stats["different_answer"] = stats.get("different_answer", 0) + 1
                    continue
                stats["kept"] = stats.get("kept", 0) + 1
                out.write(json.dumps({**{k: v for k, v in r.items() if k not in ("program", "answer")},
                                      "program": eqs, "answer": a, "program_python": r["program"],
                                      "verified": True}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[notation] {min(c0 + args.chunk, len(rows))}/{len(rows)} rows | {json.dumps(stats)}", flush=True)

    elif args.mode == "solve":
        rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
        texts = chat(llm, sp, [SOLVE_PROMPT.format(q=r["question"], rules=rules) for r in rows], think=args.think)
        for r, t in zip(rows, texts):
            a2, _ = run_program(t)
            r["program2"] = extract_program(t)
            r["answer2"] = a2
            r["verified"] = a2 is not None and close(a2, r["answer"], 0.01)
            stats["verified" if r["verified"] else "disagree"] = stats.get("verified" if r["verified"] else "disagree", 0) + 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    elif args.mode == "convert":
        seeds = [(r, t) for r, t in numerical_seeds(args.n_seeds, args.seed) if t is not None]
        texts = chat(llm, sp, [CONVERT_PROMPT.format(q=r["instruction"], ref=r["response"][:6000], rules=rules)
                               for r, _ in seeds], think=args.think)
        for (r, t), txt in zip(seeds, texts):
            a, _ = run_program(txt)
            ok = a is not None and close(a, t, 0.02)
            stats["matches_label" if ok else "rejected"] = stats.get("matches_label" if ok else "rejected", 0) + 1
            if ok:
                out.write(json.dumps({"question": r["instruction"], "program": extract_program(txt), "answer": a,
                                      "label": t, "chunk_id": r.get("chunk_id"), "source": "convert"},
                                     ensure_ascii=False) + "\n")

    else:  # audit
        items = [json.loads(l) for l in open(ROOT / "data/rl/test.jsonl", encoding="utf-8")]
        items = [it for it in items if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
        texts = chat(llm, sp, [AUDIT_PROMPT.format(q=it["prompt"], t=it["final_numerical_value"], rules=rules)
                               for it in items], think=args.think)
        for it, t in zip(items, texts):
            j = parse_json(t) if t else None
            j = j if isinstance(j, dict) else {}
            a, _ = run_program(j.get("program", "")) if j.get("program") else (None, {})
            rec = {"id": it["id"], "self_contained": bool(j.get("self_contained")), "missing": j.get("missing", ""),
                   "program_answer": a, "label": it["final_numerical_value"],
                   "program_matches_label": a is not None and close(a, float(it["final_numerical_value"]), 0.05)}
            k = ("self_contained" if rec["self_contained"] else "needs_context")
            stats[k] = stats.get(k, 0) + 1
            stats[k + "_and_matches"] = stats.get(k + "_and_matches", 0) + rec["program_matches_label"]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
