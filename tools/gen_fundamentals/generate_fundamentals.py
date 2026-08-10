#!/usr/bin/env python3
"""
generate_fundamentals.py — Standalone generator for a seismology fundamentals
SFT dataset. Portable: Python 3.8+ standard library only, no pip installs, no
imports from the nanoGPT-Seis repo. Copy this directory anywhere and run it.

WHY THIS EXISTS
    The paper-derived SFT corpus teaches specific studies, not the subject. A
    coverage audit of 4,434 cleaned pairs found 0.3% mentioning P-waves with
    "compressional", 0.2% mentioning deep-focus earthquakes at all. The model's
    answer quality tracked that distribution exactly. This script generates the
    missing textbook layer from an explicit syllabus (curriculum.py).

OUTPUT
    JSONL, one line per topic-variant, in the same schema the repo's cleaner
    and splitter already consume:

        {"chunk_id": "...", "source": "fundamentals",
         "pair_count": 6,
         "pairs": [{"type": "conceptual", "instruction": "...", "response": "..."}, ...]}

    One topic == one chunk_id ACROSS ALL VARIANTS, so the repo's chunk-level
    splitter keeps every question about a topic on the same side and cannot
    leak. (An earlier version keyed chunk_id on topic+variant, which leaked.)

QUICK START
    export TEACHER_API_KEY="..."
    python3 generate_fundamentals.py --dry-run                 # no API calls
    python3 generate_fundamentals.py --variants 2 --workers 4  # small pilot
    python3 generate_fundamentals.py --variants 12 --workers 8 # ~2,400 pairs

    # any OpenAI-compatible endpoint
    python3 generate_fundamentals.py --api-base https://api.openai.com/v1 --model gpt-4o-mini
    python3 generate_fundamentals.py --api-base http://localhost:8000/v1 --model my-local-model

RESUMABLE
    Output is appended and every completed (topic, variant) is recorded. Re-run
    the same command after an interruption and it skips finished work. Use
    --overwrite to start clean.

VALIDATE / MERGE (no API key needed)
    python3 generate_fundamentals.py --validate-only --out fundamentals.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum import CURRICULUM, summary, total_pairs  # noqa: E402

# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a seismology professor writing training material for a compact \
domain assistant. You produce accurate, self-contained question-and-answer pairs that teach \
fundamentals.

ABSOLUTE REQUIREMENTS — a violation makes the sample unusable:

1. STANDALONE. The reader has NO document in front of them. Never write "the text", "the study", \
"the paper", "the authors", "as described above", "Table 2", "Figure 3", or any citation such as \
"(Smith et al., 2019)". Write settled textbook knowledge in your own words.

2. CORRECT. Every number, formula and physical claim must be right. State the unit convention \
whenever a formula depends on it (for example Mw = (2/3)log10(M0) - 6.07 for M0 in N*m, versus \
-10.7 for dyne*cm). Prefer being brief and correct over long and vague.

3. DISTINCT SYMBOLS. Every symbol in a formula must be defined and must denote one thing. Never \
write degenerate expressions such as "W / W", "M = M - M", or reuse one letter for two quantities.

4. SELF-CONSISTENT ARITHMETIC. In numerical questions, put the derivation inside \
<think>...</think>, carry units through every step, and end AFTER the closing tag with a short \
final answer sentence containing the computed value. Use <think> exactly once. Do not write \
"</think>" anywhere else.

5. LENGTH. Answers should be 120-260 words. Complete, not padded. Conversational scientific prose, \
not bullet lists.

Return ONLY a JSON object, no prose around it, of the form:
{"pairs": [{"type": "conceptual|numerical|comparison", "instruction": "...", "response": "..."}]}
"""

TYPE_GUIDE = {
    "conceptual": (
        "Ask a 'why' or 'how' question about the mechanism. The answer explains the physics in "
        "plain scientific prose. NO <think> block and no arithmetic."
    ),
    "numerical": (
        "Pose a concrete calculation with all input values GIVEN IN THE QUESTION ITSELF (never "
        "referring to an external table). The answer opens with a <think> block showing the "
        "derivation step by step with units, closes it with </think>, then states one short final "
        "sentence containing the computed value with units."
    ),
    "comparison": (
        "Ask the reader to contrast two things that are commonly confused. The answer states the "
        "distinction crisply and gives the discriminating physical reason. NO <think> block."
    ),
}


def build_user_prompt(topic: Dict[str, Any], variant: int) -> str:
    mix = ", ".join(f"{n} of type '{k}'" for k, n in topic["mix"].items() if n > 0)
    covers = "\n".join(f"  - {c}" for c in topic["must_cover"])
    guides = "\n".join(
        f"  {k}: {TYPE_GUIDE[k]}" for k, n in topic["mix"].items() if n > 0
    )
    total = sum(topic["mix"].values())

    angle = VARIANT_ANGLES[variant % len(VARIANT_ANGLES)]
    notes = f"\nSPECIAL NOTE: {topic['notes']}\n" if topic.get("notes") else ""

    return f"""TOPIC: {topic['topic']}
SUBJECT AREA: {topic['section']}

Generate exactly {total} question-answer pairs: {mix}.

The set must collectively cover these points:
{covers}
{notes}
QUESTION TYPES:
{guides}

FRAMING FOR THIS BATCH (use it to keep questions distinct from other batches on
the same topic — vary the wording and entry point, never the underlying facts):
{angle}

Return the JSON object now."""


# Distinct framings so repeated variants on one topic do not duplicate wording.
VARIANT_ANGLES = [
    "Ask as a student encountering the topic for the first time.",
    "Ask as a graduate student probing a subtlety or common misconception.",
    "Frame around a concrete real-world observation an analyst would face.",
    "Frame around interpreting data or a measurement.",
    "Ask 'what would happen if' about varying one physical parameter.",
    "Ask why a frequent oversimplification is wrong, and what is actually true.",
    "Frame around how the quantity is estimated in practice, and its uncertainty.",
    "Ask about the physical limits: when does the standard model break down?",
    "Frame around orders of magnitude and typical values.",
    "Ask how this concept connects to a neighbouring subfield.",
    "Frame as a practical engineering or hazard-assessment decision.",
    "Ask the reader to reason from first principles rather than recall a formula.",
    "Ask what observation would distinguish two competing explanations.",
    "Frame around a historical earthquake and what it taught the field.",
    "Ask about the assumptions behind the standard treatment and when they fail.",
    "Frame around instrumentation: what is actually measured versus what is inferred.",
    "Ask how the answer differs between oceanic and continental settings.",
    "Frame around a common exam-style misconception and correct it.",
    "Ask about the time and length scales involved and why they matter.",
    "Frame around what a practitioner would report, and to whom.",
]
# NOTE: angles are cycled as variant % len(VARIANT_ANGLES). With only 12 of
# them, --variants 13+ reused angle 0 and produced near-duplicates of variant 0.
# Measured novelty decay on the first 12 variants (fraction of word-types a
# variant contributes that no earlier variant did):
#     v1 55%  v3 33%  v5 30%  v7 26%  v9 21%  v11 12%
# The lever with better returns is MORE TOPICS, not more variants per topic.


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class ApiError(Exception):
    pass


def call_llm_api(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    max_tokens: int = 4000,
    temperature: float = 0.8,
    max_retries: int = 5,
    timeout: int = 180,
) -> str:
    """POST to an OpenAI-compatible /chat/completions endpoint. Stdlib only."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (408, 409, 429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt + random.uniform(0, 1.5), 45))
                continue
            # 400/401/403/404 will not fix themselves
            raise ApiError(last_err)
        except Exception as e:  # timeouts, connection resets, malformed JSON
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20))

    raise ApiError(f"exhausted {max_retries} retries — {last_err}")


# --------------------------------------------------------------------------
# Parsing and local validation
# --------------------------------------------------------------------------

CONTAMINATION = re.compile(
    r"\b(?:the|this|that)\s+(?:text|passage|excerpt|document|paper|article|study|studies)\b"
    r"|\bthe\s+(?:authors?|researchers)\b"
    r"|\bet\s+al\b"
    r"|\b(?:fig(?:ure)?|table|section|appendix)s?\s+\(?\d"
    r"|\bas\s+(?:described|shown|mentioned|stated|discussed)\s+(?:in|above|earlier|previously)\b",
    re.IGNORECASE,
)
DEGENERATE = re.compile(
    r"\b([A-Za-z_]\w*)\s*/\s*\1\b"          # W / W
    r"|\b([A-Za-z_]\w*)\s*=\s*\2\s*-\s*\2\b"  # M = M - M
)
VALID_TYPES = {"conceptual", "numerical", "comparison"}


def extract_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)  # first ... last brace
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def check_pair(p: Dict[str, Any]) -> Optional[str]:
    """Return a rejection reason, or None if the pair is acceptable."""
    if not isinstance(p, dict):
        return "not an object"
    inst = (p.get("instruction") or "").strip()
    resp = (p.get("response") or "").strip()
    ptype = (p.get("type") or "").strip().lower()

    if ptype not in VALID_TYPES:
        return f"bad type {ptype!r}"
    if len(inst) < 15:
        return "instruction too short"
    if len(resp) < 80:
        return "response too short"
    if CONTAMINATION.search(inst):
        return "phantom-source reference in instruction"
    if CONTAMINATION.search(resp):
        return "phantom-source reference in response"
    if DEGENERATE.search(resp):
        return "degenerate expression (X/X or X=X-X)"

    n_open, n_close = resp.count("<think>"), resp.count("</think>")
    if ptype == "numerical":
        if n_open != 1 or n_close != 1:
            return f"numerical needs exactly one <think> pair (got {n_open}/{n_close})"
        if resp.find("<think>") > resp.find("</think>"):
            return "think tags out of order"
        tail = resp.split("</think>", 1)[1].strip()
        if len(tail) < 20:
            return "no final answer after </think>"
        if not re.search(r"\d", tail):
            return "final answer contains no number"
    else:
        if n_open or n_close:
            return f"{ptype} must not contain think tags"
    return None


def parse_response(raw: str, stats: Dict[str, int]) -> List[Dict[str, str]]:
    obj = extract_json(raw)
    if obj is None:
        stats["unparseable"] += 1
        return []
    pairs = obj.get("pairs") if isinstance(obj, dict) else None
    if not isinstance(pairs, list):
        stats["no_pairs_key"] += 1
        return []

    kept = []
    for p in pairs:
        reason = check_pair(p)
        if reason:
            stats[f"reject:{reason.split('(')[0].strip()}"] += 1
            continue
        kept.append({
            "type": p["type"].strip().lower(),
            "instruction": p["instruction"].strip(),
            "response": p["response"].strip(),
        })
    return kept


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def chunk_id_for(topic: str, variant: int = 0) -> str:
    """Group key for the train/test splitter — TOPIC only, not variant.

    Including the variant here was a real leak: it made every variant its own
    chunk, so variant 3 of a topic could sit in test while variants 0-2 and
    4-11 sat in train. Measured on a live run, 100% of test topics also
    appeared in training under another variant, turning held-out perplexity
    into a memorisation score. Resume keys on (topic, variant) separately via
    resume_key(), so per-batch scheduling still works.
    """
    return hashlib.md5(topic.encode("utf-8")).hexdigest()


def resume_key(topic: str, variant: int) -> str:
    return f"{chunk_id_for(topic)}::v{variant}"


def load_done(out_path: Path) -> set:
    """(chunk_id, variant) pairs already present, so a re-run resumes."""
    done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(f"{rec['chunk_id']}::v{rec.get('variant', 0)}")
                except Exception:
                    continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a seismology fundamentals SFT dataset (portable, stdlib only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--out", default="fundamentals.jsonl", help="output JSONL path")
    ap.add_argument("--variants", type=int, default=1,
                    help="independent batches per topic; total pairs ~= variants x %d" % total_pairs())
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    ap.add_argument("--api-key", default="", help="or set TEACHER_API_KEY / OPENAI_API_KEY")
    ap.add_argument("--api-base", default=os.getenv("TEACHER_API_BASE", ""),
                    help="OpenAI-compatible base URL (must end in /v1 for most providers)")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", ""),
                    help="teacher model name exposed by --api-base")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit-topics", type=int, default=0, help="only the first N topics (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="truncate the output instead of resuming")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and the plan, call nothing")
    ap.add_argument("--validate-only", action="store_true", help="re-check an existing output file")
    args = ap.parse_args()

    out_path = Path(args.out)

    # ---- validate-only -----------------------------------------------------
    if args.validate_only:
        return validate_file(out_path)

    topics = CURRICULUM[: args.limit_topics] if args.limit_topics else CURRICULUM
    jobs = [(t, v) for v in range(args.variants) for t in topics]

    # ---- dry run -----------------------------------------------------------
    if args.dry_run:
        print(summary())
        print(f"\nplan: {len(topics)} topics x {args.variants} variants = {len(jobs)} API calls")
        print(f"      ~{sum(sum(t['mix'].values()) for t in topics) * args.variants} pairs requested")
        print(f"      endpoint: {args.api_base}  model: {args.model}")
        print("\n" + "=" * 72)
        print("EXAMPLE PROMPT (topic 1, variant 0)")
        print("=" * 72)
        print(build_user_prompt(topics[0], 0))
        return 0

    api_key = args.api_key or os.getenv("TEACHER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key and "localhost" not in args.api_base and "127.0.0.1" not in args.api_base:
        print("[ERROR] No API key. Pass --api-key or set TEACHER_API_KEY / OPENAI_API_KEY.",
              file=sys.stderr)
        return 2

    if not args.api_base or not args.model:
        print("[ERROR] No teacher endpoint. Pass --api-base and --model, or set "
              "TEACHER_API_BASE / TEACHER_MODEL. Any OpenAI-compatible endpoint works.",
              file=sys.stderr)
        return 2

    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = load_done(out_path)
    if done:
        print(f"[resume] {len(done)} chunks already in {out_path}; skipping those")
    jobs = [(t, v) for (t, v) in jobs if resume_key(t["topic"], v) not in done]
    if not jobs:
        print("[done] nothing left to generate")
        return validate_file(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    stats: Dict[str, int] = {}
    counters = {"ok": 0, "fail": 0, "pairs": 0}

    def bump(key: str, n: int = 1) -> None:
        stats[key] = stats.get(key, 0) + n

    def work(job) -> None:
        topic, variant = job
        cid = chunk_id_for(topic["topic"])
        try:
            raw = call_llm_api(
                SYSTEM_PROMPT, build_user_prompt(topic, variant),
                api_key=api_key, api_base=args.api_base, model=args.model,
                max_tokens=args.max_tokens, temperature=args.temperature,
                max_retries=args.max_retries, timeout=args.timeout,
            )
        except ApiError as e:
            with write_lock:
                counters["fail"] += 1
                print(f"  [FAIL] {topic['topic'][:44]:<44} v{variant}  {e}", flush=True)
            return

        from collections import defaultdict
        d: Dict[str, int] = defaultdict(int)
        pairs = parse_response(raw, d)

        with write_lock:
            for k, v in d.items():
                bump(k, v)
            if not pairs:
                counters["fail"] += 1
                print(f"  [EMPTY] {topic['topic'][:43]:<43} v{variant}  (all pairs rejected)", flush=True)
                return
            rec = {
                "chunk_id": cid,
                "source": "fundamentals",
                "topic": topic["topic"],
                "section": topic["section"],
                "variant": variant,
                "pair_count": len(pairs),
                "pairs": pairs,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            counters["ok"] += 1
            counters["pairs"] += len(pairs)
            n = counters["ok"] + counters["fail"]
            print(f"  [{n:>4}/{len(jobs)}] {topic['topic'][:43]:<43} v{variant}  "
                  f"+{len(pairs)} pairs (total {counters['pairs']})", flush=True)

    print(f"generating {len(jobs)} chunks -> {out_path}  "
          f"({args.workers} workers, model={args.model})")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    dt = time.time() - t0

    print(f"\ndone in {dt/60:.1f} min: {counters['ok']} chunks ok, {counters['fail']} failed, "
          f"{counters['pairs']} pairs written")
    if stats:
        print("rejections / parse issues:")
        for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<52} {v}")
    return validate_file(out_path)


def validate_file(out_path: Path) -> int:
    """Re-read the output and report a quality summary. No API needed."""
    if not out_path.exists():
        print(f"[ERROR] {out_path} not found", file=sys.stderr)
        return 2

    from collections import Counter
    n_chunks = 0
    types = Counter()
    sections = Counter()
    bad = 0
    lens: List[int] = []
    seen_inst = set()
    dupes = 0

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_chunks += 1
            sections[rec.get("section", "?")] += 1
            for p in rec["pairs"]:
                reason = check_pair(p)
                if reason:
                    bad += 1
                    continue
                types[p["type"]] += 1
                lens.append(len(p["response"].split()))
                key = p["instruction"].strip().lower()
                if key in seen_inst:
                    dupes += 1
                seen_inst.add(key)

    total = sum(types.values())
    print("\n" + "=" * 66)
    print(f"VALIDATION: {out_path}")
    print("=" * 66)
    print(f"chunks (topic-variants) : {n_chunks}")
    print(f"usable pairs            : {total}")
    print(f"failing re-validation   : {bad}")
    print(f"duplicate instructions  : {dupes}")
    if lens:
        lens.sort()
        print(f"response words          : mean {sum(lens)/len(lens):.0f} | "
              f"median {lens[len(lens)//2]} | p95 {lens[int(0.95*len(lens))]}")
    print("by type    : " + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
    print("by section : " + ", ".join(f"{k}={v}" for k, v in sorted(sections.items())))
    print("=" * 66)
    print("\nNext, on the training machine:")
    print(f"  cp {out_path.name} <repo>/data/")
    print("  # merge with the paper-derived corpus and rebuild the splits:")
    print("  python -m src.sft.build_corpus clean --input data/sources/paper_raw.jsonl \\")
    print("      --output data/sources/paper.jsonl")
    print(f"  cat data/{out_path.name} >> data/sources/paper.jsonl")
    print("  python -m src.sft.build_corpus split --input data/sources/paper.jsonl \\")
    print("      --output-dir data/processed_v2")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
