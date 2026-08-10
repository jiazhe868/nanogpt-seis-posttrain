#!/usr/bin/env python3
"""
generate_earthquakes.py — Historical-earthquake SFT data generator.

Portable: Python 3.8+ standard library only. Companion to
generate_fundamentals.py, sharing its API plumbing and output schema.

WHAT THIS ADDS
    generate_fundamentals.py teaches settled physics. This teaches the
    empirical record: what specific earthquakes did, why they did it, what
    they destroyed, and what the field learned. Together they cover the two
    halves of domain competence — principles and cases.

THE HALLUCINATION PROBLEM, AND THE GUARD
    Historical events are far riskier to generate than fundamentals. Asked
    about the 1985 Michoacan earthquake, an LLM will fluently invent a
    magnitude, shift the date by a year, or quote a death toll off by an order
    of magnitude. Wrong specifics read exactly like right ones.

    Three defences:
      1. Every event in earthquakes.py carries VERIFIED seed facts, injected
         as the only permitted source of specifics.
      2. A FACT-CONTRADICTION VALIDATOR parses magnitudes and years out of each
         generated answer and rejects the pair if they disagree with the seed
         (Mw tolerance 0.25; any 4-digit year near a magnitude claim must match).
      3. The prompt forbids inventing casualty figures and requires disputed
         numbers to be stated as ranges.

    A pair that fails validation is dropped, not repaired. Cheaper to discard
    than to teach the model a confident falsehood.

USAGE
    export TEACHER_API_KEY="..."
    python3 generate_earthquakes.py --dry-run
    python3 generate_earthquakes.py --variants 1 --workers 4      # pilot
    python3 generate_earthquakes.py --variants 12 --workers 8     # ~3,000 pairs

    python3 generate_earthquakes.py --validate-only --out earthquakes_sft.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earthquakes import EARTHQUAKES, summary, total_pairs  # noqa: E402
from generate_fundamentals import (  # noqa: E402  (shared, already tested)
    ApiError,
    call_llm_api,
    extract_json,
    CONTAMINATION,
    DEGENERATE,
)

# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a seismologist writing training material about specific historical \
earthquakes for a compact domain assistant.

You will be given VERIFIED FACTS about one earthquake. Those facts, plus standard textbook \
seismology, are the ONLY sources you may use for specifics.

ABSOLUTE REQUIREMENTS — a violation makes the sample unusable:

1. NEVER CONTRADICT THE GIVEN FACTS. The magnitude, date, depth, mechanism and setting supplied \
to you are correct. Reproduce them accurately or omit them. Do not round a magnitude to a \
different value, do not shift the year, do not relocate the event.

2. NEVER INVENT SPECIFICS. Do not add casualty figures, economic losses, rupture lengths, slip \
amounts, or peak accelerations that were not supplied. If a number was not given, either omit it \
or describe it qualitatively ("severe damage", "widespread collapse"). Inventing a plausible \
number is the worst possible failure here.

3. PRESERVE STATED UNCERTAINTY. Where the facts give a range or say estimates vary, keep that \
uncertainty. Do not collapse "roughly 100,000 to over 300,000 deaths" into a single confident \
figure.

4. STANDALONE. The reader has no document. Never write "the text", "the study", "the provided \
facts", "as listed above", or any citation. Write as an expert recalling the event.

5. EXPLAIN, DON'T LIST. Answers are connected scientific prose, 120-260 words, not bullet points. \
For mechanism questions, give the physics. For impact questions, give the causal chain from \
ground motion to damage.

6. NUMERICAL QUESTIONS. Put the derivation in <think>...</think>, use the event's real parameters, \
carry units, and state the final value in one sentence AFTER the closing tag. Use <think> exactly \
once and never write "</think>" elsewhere. Non-numerical answers must contain no think tags.

QUESTION TYPES:
  factual      — what happened: parameters, setting, sequence of events.
  mechanism    — the physics: why this rupture style, this depth, this behaviour.
  impact       — damage and casualties, and the causal chain that produced them.
  significance — what the event changed in the science or in engineering practice.
  numerical    — a calculation grounded in the event's real parameters.

Return ONLY a JSON object:
{"pairs": [{"type": "factual|mechanism|impact|significance|numerical",
            "instruction": "...", "response": "..."}]}
"""

VARIANT_ANGLES = [
    "Ask as a student who has heard of the event but knows no details.",
    "Ask why this event behaved differently from what was expected at the time.",
    "Frame around comparison with another earthquake of similar magnitude.",
    "Ask what this event revealed that was not known before it.",
    "Frame around the engineering or policy response that followed.",
    "Ask about the tectonic setting and why this location produces such events.",
    "Frame around what the instrumental records showed.",
    "Ask what would happen if a similar event struck the same place today.",
    "Frame around the secondary hazards rather than the shaking itself.",
    "Ask about the time scales: foreshocks, rupture duration, aftershocks, recovery.",
    "Frame around why the casualty figure was as high or as low as it was.",
    "Ask which assumptions in hazard assessment this event invalidated.",
    "Frame around what a seismologist saw in the first hours after it happened.",
    "Ask how this event fits into the longer earthquake history of the region.",
    "Frame around the difference between the magnitude and the actual consequences.",
    "Ask what is still debated or uncertain about this earthquake.",
]


def build_user_prompt(eq: Dict[str, Any], variant: int) -> str:
    mix = ", ".join(f"{n} of type '{k}'" for k, n in eq["mix"].items() if n > 0)
    total = sum(eq["mix"].values())
    facts = "\n".join(f"  - {f}" for f in eq["facts"])
    angle = VARIANT_ANGLES[variant % len(VARIANT_ANGLES)]

    return f"""EARTHQUAKE: {eq['name']}

VERIFIED FACTS (the only permitted source of specifics):
  - Date (UTC): {eq['date']}
  - Moment magnitude: Mw {eq['mw']}
  - Hypocentral depth: approximately {eq['depth_km']} km
  - Region: {eq['region']}
  - Faulting mechanism: {eq['mechanism']}
  - Tectonic setting: {eq['setting']}
{facts}

SCIENTIFIC SIGNIFICANCE:
  {eq['significance']}

Generate exactly {total} question-answer pairs: {mix}.

FRAMING FOR THIS BATCH (vary the entry point, never the facts):
{angle}

Return the JSON object now."""


# --------------------------------------------------------------------------
# Fact-contradiction validation
# --------------------------------------------------------------------------

# "Mw 7.8", "M 9.0", "magnitude 6.2", "Mw of 8.3", and also "Mw 10.5".
# The two-digit branch matters: an earlier one-digit pattern parsed a
# fabricated "10.5" as "0.5", which was then discarded as implausible and let
# the fabrication through. Capture it, then reject it as out of range.
MAG_RE = re.compile(
    r"\b(?:Mw|M_?w|magnitude)\s*(?:of\s*)?[=:]?\s*(\d{1,2}(?:\.\d+)?)\b",
    re.IGNORECASE,
)
# No earthquake has Mw below ~3 in this catalogue or above ~9.6. A value
# outside this band attached to a named event is fabricated, not a rounding
# difference.
MAG_PLAUSIBLE = (2.5, 9.9)
# 1500-2029. The earlier 1800-2029 range silently skipped the 1755 Lisbon
# earthquake, so a fabricated "1758" passed the year check unexamined.
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")
MAG_TOL = 0.25  # Mw 7.8 vs "about 7.9" is fine; 7.8 vs 8.6 is not


# Words that carry no identifying information, so they must not be used to
# decide whether a stray number refers to THIS event. Taking name.split()[1]
# naively yielded anchors like "Great" (1964 Great Alaska), "Sea" (2013 Sea of
# Okhotsk) and "San" (1906 San Francisco) — all of which appear in unrelated
# sentences and would misfire the contradiction checks in both directions.
_GENERIC = {
    "great", "earthquake", "earthquakes", "sequence", "doublet", "tsunami",
    "and", "the", "of", "sea", "islands", "island", "border", "region",
    "eastern", "western", "northern", "southern", "central", "off", "coast",
    "summit", "collapse", "aftermath", "comparison", "event", "induced",
    "deep", "focus", "deep-focus", "reservoir-induced", "good", "friday",
}


def name_anchors(eq: Dict[str, Any]) -> List[str]:
    """Distinctive tokens identifying this event, best first.

    Falls back to a two-word phrase when no single token is distinctive
    (e.g. "Rat Islands"), and finally to the whole name minus the leading year.
    """
    raw = re.sub(r"^\d{4}(?:-\d{4})?\s+", "", eq["name"])          # drop leading year
    raw = re.sub(r"[(),:]", " ", raw)
    tokens = [t for t in raw.split() if t]
    good = [t for t in tokens if len(t) >= 4 and t.lower() not in _GENERIC]
    if good:
        return good
    if len(tokens) >= 2:
        return [" ".join(tokens[:2])]
    return [raw.strip()] if raw.strip() else [eq["name"]]


def _mentions_event(text: str, anchors: List[str]) -> bool:
    return any(re.search(re.escape(a), text, re.IGNORECASE) for a in anchors)


def seed_mw(eq: Dict[str, Any]) -> Optional[float]:
    v = eq["mw"]
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).lstrip("~"))
    except ValueError:
        return None


def check_pair(p: Dict[str, Any], eq: Dict[str, Any]) -> Optional[str]:
    """Return a rejection reason, or None if acceptable."""
    if not isinstance(p, dict):
        return "not an object"
    inst = (p.get("instruction") or "").strip()
    resp = (p.get("response") or "").strip()
    ptype = (p.get("type") or "").strip().lower()

    if ptype not in eq["mix"] and ptype not in {"factual", "mechanism", "impact",
                                                "significance", "numerical"}:
        return f"bad type {ptype!r}"
    if len(inst) < 15:
        return "instruction too short"
    if len(resp) < 80:
        return "response too short"
    if CONTAMINATION.search(inst) or CONTAMINATION.search(resp):
        return "phantom-source reference"
    if DEGENERATE.search(resp):
        return "degenerate expression"

    blob = inst + " " + resp

    # --- magnitude contradiction -----------------------------------------
    target = seed_mw(eq)
    if target is not None:
        anchors = name_anchors(eq)
        for m in MAG_RE.findall(blob):
            try:
                v = float(m)
            except ValueError:
                continue
            if not (MAG_PLAUSIBLE[0] <= v <= MAG_PLAUSIBLE[1]):
                # Physically impossible magnitude. Reject if it is attached to
                # this event; ignore otherwise (it is some other quantity).
                if any(re.search(rf"{re.escape(a)}[^.]{{0,80}}?{re.escape(m)}"
                                 rf"|{re.escape(m)}[^.]{{0,80}}?{re.escape(a)}",
                                 blob, re.IGNORECASE) for a in anchors):
                    return f"implausible magnitude ({m})"
                continue
            if abs(v - target) > MAG_TOL:
                # A second, clearly-labelled event may legitimately be compared.
                # Only reject when the wrong value is attached to THIS event.
                near = any(re.search(
                    rf"{re.escape(a)}[^.]{{0,80}}?{re.escape(m)}"
                    rf"|{re.escape(m)}[^.]{{0,80}}?{re.escape(a)}",
                    blob, re.IGNORECASE) for a in anchors)
                if near:
                    return f"magnitude contradiction ({m} vs seed {target})"

    # --- year contradiction ----------------------------------------------
    # Reject only when a NON-seed year sits right next to THIS event's name, in
    # either word order ("Tohoku-oki earthquake of 2013" and "the 2013
    # Tohoku-oki earthquake" must both fail). A distant year is usually a
    # legitimate reference to another earthquake or to a code revision, so a
    # tight window keeps comparisons like "...comparable to the 1960 Chilean
    # event..." acceptable.
    seed_year = eq["date"][:4]
    anchors_y = name_anchors(eq)
    WINDOW = 35
    for m in YEAR_RE.finditer(blob):
        year = m.group(1)
        if year == seed_year:
            continue
        lo = max(0, m.start() - WINDOW)
        hi = min(len(blob), m.end() + WINDOW)
        context = blob[lo:hi]
        if _mentions_event(context, anchors_y) and seed_year not in context:
            return f"year contradiction ({year} beside '{anchors_y[0]}', seed {seed_year})"

    # --- think-tag hygiene ------------------------------------------------
    n_open, n_close = resp.count("<think>"), resp.count("</think>")
    if ptype == "numerical":
        if n_open != 1 or n_close != 1:
            return f"numerical needs exactly one <think> pair (got {n_open}/{n_close})"
        if resp.find("<think>") > resp.find("</think>"):
            return "think tags out of order"
        tail = resp.split("</think>", 1)[1].strip()
        if len(tail) < 20 or not re.search(r"\d", tail):
            return "no numeric final answer after </think>"
    else:
        if n_open or n_close:
            return f"{ptype} must not contain think tags"
    return None


def parse_response(raw: str, eq: Dict[str, Any], stats: Dict[str, int]) -> List[Dict[str, str]]:
    obj = extract_json(raw)
    if obj is None:
        stats["unparseable"] = stats.get("unparseable", 0) + 1
        return []
    pairs = obj.get("pairs") if isinstance(obj, dict) else None
    if not isinstance(pairs, list):
        stats["no_pairs_key"] = stats.get("no_pairs_key", 0) + 1
        return []

    kept = []
    for p in pairs:
        reason = check_pair(p, eq)
        if reason:
            key = "reject:" + reason.split("(")[0].strip()
            stats[key] = stats.get(key, 0) + 1
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

def chunk_id_for(name: str, variant: int = 0) -> str:
    """Group key for the splitter — EVENT only, not variant. See the note in
    generate_fundamentals.chunk_id_for: including the variant leaked 100% of
    test events into training under a different variant."""
    return hashlib.md5(f"eq::{name}".encode("utf-8")).hexdigest()


def resume_key(name: str, variant: int) -> str:
    return f"{chunk_id_for(name)}::v{variant}"


def load_done(out_path: Path) -> set:
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
        description="Generate historical-earthquake SFT data (portable, stdlib only).")
    ap.add_argument("--out", default="earthquakes_sft.jsonl")
    ap.add_argument("--variants", type=int, default=1,
                    help=f"batches per earthquake; total pairs ~= variants x {total_pairs()}")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls (not more data)")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-base", default=os.getenv("TEACHER_API_BASE", ""),
                    help="OpenAI-compatible base URL, e.g. https://<host>/v1")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", ""),
                    help="teacher model name exposed by --api-base")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="lower than the fundamentals default: facts, not creativity")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit-events", type=int, default=0, help="first N events only (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.validate_only:
        return validate_file(out_path)

    events = EARTHQUAKES[: args.limit_events] if args.limit_events else EARTHQUAKES
    jobs = [(e, v) for v in range(args.variants) for e in events]

    if args.dry_run:
        print(summary())
        print(f"\nplan: {len(events)} earthquakes x {args.variants} variants = {len(jobs)} API calls")
        print(f"      ~{sum(sum(e['mix'].values()) for e in events) * args.variants} pairs requested")
        print(f"      endpoint: {args.api_base}  model: {args.model}")
        print("\n" + "=" * 72 + "\nEXAMPLE PROMPT (event 1, variant 0)\n" + "=" * 72)
        print(build_user_prompt(events[0], 0))
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
    jobs = [(e, v) for (e, v) in jobs if resume_key(e["name"], v) not in done]
    if not jobs:
        print("[done] nothing left to generate")
        return validate_file(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stats: Dict[str, int] = {}
    counters = {"ok": 0, "fail": 0, "pairs": 0}

    def work(job) -> None:
        eq, variant = job
        try:
            raw = call_llm_api(
                SYSTEM_PROMPT, build_user_prompt(eq, variant),
                api_key=api_key, api_base=args.api_base, model=args.model,
                max_tokens=args.max_tokens, temperature=args.temperature,
                max_retries=args.max_retries, timeout=args.timeout,
            )
        except ApiError as e:
            with lock:
                counters["fail"] += 1
                print(f"  [FAIL] {eq['name'][:44]:<44} v{variant}  {e}", flush=True)
            return

        local: Dict[str, int] = {}
        pairs = parse_response(raw, eq, local)
        with lock:
            for k, v in local.items():
                stats[k] = stats.get(k, 0) + v
            if not pairs:
                counters["fail"] += 1
                print(f"  [EMPTY] {eq['name'][:43]:<43} v{variant}  (all pairs rejected)", flush=True)
                return
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "chunk_id": chunk_id_for(eq["name"]),
                    "source": "historical_earthquake",
                    "event": eq["name"],
                    "mw": eq["mw"],
                    "variant": variant,
                    "pair_count": len(pairs),
                    "pairs": pairs,
                }, ensure_ascii=False) + "\n")
                f.flush()
            counters["ok"] += 1
            counters["pairs"] += len(pairs)
            n = counters["ok"] + counters["fail"]
            print(f"  [{n:>4}/{len(jobs)}] {eq['name'][:43]:<43} v{variant}  "
                  f"+{len(pairs)} pairs (total {counters['pairs']})", flush=True)

    print(f"generating {len(jobs)} chunks -> {out_path} ({args.workers} workers, model={args.model})")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))

    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {counters['ok']} chunks ok, "
          f"{counters['fail']} failed, {counters['pairs']} pairs written")
    if stats:
        print("rejections / parse issues:")
        for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<52} {v}")
    return validate_file(out_path)


def validate_file(out_path: Path) -> int:
    if not out_path.exists():
        print(f"[ERROR] {out_path} not found", file=sys.stderr)
        return 2
    from collections import Counter
    by_name = {e["name"]: e for e in EARTHQUAKES}
    n_chunks = 0
    types: Counter = Counter()
    events: Counter = Counter()
    bad = 0
    lens: List[int] = []
    seen = set()
    dupes = 0

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_chunks += 1
            eq = by_name.get(rec.get("event", ""))
            events[rec.get("event", "?")] += rec["pair_count"]
            for p in rec["pairs"]:
                if eq and check_pair(p, eq):
                    bad += 1
                    continue
                types[p["type"]] += 1
                lens.append(len(p["response"].split()))
                k = p["instruction"].strip().lower()
                if k in seen:
                    dupes += 1
                seen.add(k)

    total = sum(types.values())
    print("\n" + "=" * 66)
    print(f"VALIDATION: {out_path}")
    print("=" * 66)
    print(f"chunks (event-variants) : {n_chunks}")
    print(f"usable pairs            : {total}")
    print(f"failing re-validation   : {bad}")
    print(f"duplicate instructions  : {dupes}")
    print(f"distinct events covered : {len(events)} / {len(EARTHQUAKES)}")
    if lens:
        lens.sort()
        print(f"response words          : mean {sum(lens)/len(lens):.0f} | "
              f"median {lens[len(lens)//2]} | p95 {lens[int(0.95*len(lens))]}")
    print("by type : " + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
    print("=" * 66)
    print("\nMerge on the training machine:")
    print(f"  cp {out_path.name} <repo>/data/")
    print("  python -m src.sft.build_corpus merge --cleaned data/sft_paper_clean.jsonl \\")
    print("      --rewritten data/sources/paper_rewritten.jsonl --fundamentals data/sources/fundamentals.jsonl \\")
    print(f"      --extra data/{out_path.name} --output data/sft_dataset_v5.jsonl")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
