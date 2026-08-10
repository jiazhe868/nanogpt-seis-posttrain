#!/usr/bin/env python3
"""Build the SFT corpus: clean -> merge -> split.

Three stages of one pipeline, so they live in one module:

    clean   drop pairs that cite a source the model cannot see, and pairs
            outside the seismology domain
    merge   combine the data sources into one chunk-format file, deduplicating
            on the instruction and grouping by source chunk
    split   emit ChatML train/val/test, split at the level of the source chunk

Why each stage is shaped the way it is: README section 4.3 (data lineage) and
PITFALLS A1 (leakage).

    python -m src.sft.build_corpus clean
    python -m src.sft.build_corpus merge
    python -m src.sft.build_corpus split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

# ===========================================================================
# stage 1: clean
# ===========================================================================
# A small closed-book model must never be trained to reference a source it
# cannot see. Only grammar-preserving rewrites are attempted; anything that
# still points at a phantom source afterwards is dropped rather than mangled,
# and newlines are preserved so the model does not learn to emit walls of text.

# "(Smith et al., 2016)", "(e.g., Nof & Allen, 2016; Jones 2018a)" — removing a
# parenthetical citation never breaks the sentence around it.
paren_citation = re.compile(
    r"\s*\((?:see\s+)?(?:e\.g\.,?\s*)?[^()]{0,90}?(?:19|20)\d{2}[a-z]?(?:[;,][^()]{0,90}?)?\)"
)

source_words = (r"(?:text|passage|excerpt|document|manuscript|study|studies|paper|"
                r"article|research|authors?|researchers)")
report_verbs = (
    r"(?:states?|indicates?|shows?|notes?|reports?|suggests?|found|finds?|"
    r"demonstrates?|reveals?|concludes?|explains?|mentions?|highlights?|"
    r"presents?|describes?|discusses?|observes?|argues?|proposes?|estimates?)"
)

# Sentence-initial "According to the text, X" -> "X"
leadin_prefix = re.compile(
    rf"^(?:Based on|According to|As (?:described|stated|mentioned|noted|discussed) in)"
    rf" (?:the|this) (?:provided |given )?{source_words},?\s*",
    re.IGNORECASE,
)
# "The study found that X" -> "X". Requires "that": deleting up to and
# including it leaves a complete clause.
leadin_verb_that = re.compile(
    rf"(^|(?<=[.!?]\s))(?:The|This) (?:provided |given )?{source_words} {report_verbs} that\s+"
)

# Any remaining match means DROP.
contamination_patterns = [
    # unconditional phantom-source nouns
    re.compile(r"\b(?:the|this|that) (?:text|passage|excerpt|document|manuscript|paper|article)\b",
               re.IGNORECASE),
    # "the study" — but not "the study of earthquakes" (a field, not a paper)
    re.compile(r"\b(?:the|this|that) stud(?:y|ies)\b(?!\s+of\b)", re.IGNORECASE),
    re.compile(r"\bthe (?:authors?|researchers)\b", re.IGNORECASE),
    re.compile(r"\bet al\b", re.IGNORECASE),
    # "Table 2" / "Fig. 3" — numbered references to invisible artifacts. Two
    # forms, so list enumerations like "equation. 2)" do not match.
    re.compile(r"\b(?:fig(?:ure)?|table|section|chapter|appendix)s?\s+\(?\d", re.IGNORECASE),
    re.compile(r"\b(?:Fig|Tab|Eq|Sect?)s?\.\s*\(?\d"),
    # "the figure/table" (unnumbered), excluding the geologic "water table"
    re.compile(r"\b(?:the|this) (?:figure|table)\b(?<!water table)", re.IGNORECASE),
    # reporting verbs pointing at a source. "given by"/"provided by" are
    # excluded: standard mathematical phrasing.
    re.compile(r"\b(?:as|are|is|was|were) (?:described|mentioned|stated|noted|discussed|"
               r"presented|listed|reported|outlined|summarized) (?:in|above|earlier|previously)\b",
               re.IGNORECASE),
    re.compile(r"\b(?:in|of|from) (?:the|this) (?:current|present) (?:work|analysis|investigation)\b",
               re.IGNORECASE),
]

# "the provided data" is contamination only when the data is NOT actually in
# the instruction (checked in clean_pair).
data_reference = re.compile(
    r"\bthe (?:provided|given|above|following) "
    r"(?:data|information|values|parameters|context|percentages|numbers)\b",
    re.IGNORECASE,
)
number_token = re.compile(r"\d")

# The pretraining crawl pulled in unrelated open-access papers, so ~15% of the
# generated Q&A is off-domain (operating-system history, medical negligence,
# brewing microbiology). A 113M model has no capacity to spare on topics it
# will never be asked about.
domain_pattern = re.compile(
    r"\b(?:seismic|seismolog|seismo|earthquake|aftershock|foreshock|mainshock|"
    r"fault|rupture|tectonic|magnitude|epicent|hypocent|subduction|megathrust|"
    r"tsunami|ground motion|ground-motion|shaking|PGA|PGV|accelerogram|"
    r"liquefaction|landslide|volcan|geolog|geophys|geodetic|crust|mantle|"
    r"lithosphe|asthenosphe|slab|slip rate|stress drop|shear modulus|"
    r"moment tensor|focal mechanism|P-wave|S-wave|surface wave|Rayleigh|Love wave|"
    r"body wave|attenuation|site effect|response spectr|spectral acceleration|"
    r"base isolation|seismic hazard|seismic risk|seismic design|retrofit|"
    r"structural (?:damage|response|analysis|dynamics)|"
    r"soil|borehole|InSAR|GNSS|GPS displacement|strain rate|"
    r"Omori|Gutenberg-?Richter|b-value|Mw|M0|Vp|Vs|Poisson.s ratio|"
    r"tremor|slow slip|creep|paleoseismic|geomorph|erosion|uplift|"
    r"rock mechanics|fracture mechanics|elastic|viscoelastic|plasticity|"
    r"finite element|damping|natural frequency|modal analysis|"
    r"masonry|reinforced concrete|steel frame|bridge|foundation|pile|"
    r"disaster|evacuation|emergency response|early warning|resilience)\b",
    re.IGNORECASE,
)


def is_in_domain(instruction: str, response: str) -> bool:
    return bool(domain_pattern.search(instruction) or domain_pattern.search(response))


def capitalize_first(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
        if not ch.isspace():
            break
    return text


def apply_safe_rewrites(text: str) -> str:
    text = paren_citation.sub("", text)
    rewritten = leadin_prefix.sub("", text)
    if rewritten != text:
        rewritten = capitalize_first(rewritten)
    text = leadin_verb_that.sub(lambda m: m.group(1), rewritten)
    if text != rewritten:
        text = capitalize_first(text)
    # Tidy spaces introduced by removals, but never touch newlines.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:)])", r"\1", text)
    return text.strip()


def is_contaminated(text: str) -> bool:
    return any(p.search(text) for p in contamination_patterns)


def think_tags_valid(response: str) -> bool:
    n_open = response.count("<think>")
    n_close = response.count("</think>")
    if n_open != n_close or n_open > 1:
        return False
    if n_open == 1:
        if not response.lstrip().startswith("<think>"):
            return False
        answer = response.split("</think>", 1)[1].strip()
        if len(answer) < 20:  # no real answer after the reasoning block
            return False
    return True


def clean_pair(pair: dict, stats: Counter, domain_filter: bool = True) -> Optional[dict]:
    instruction = (pair.get("instruction") or "").strip()
    response = (pair.get("response") or "").strip()
    pair_type = pair.get("type", "general")

    if not instruction or not response or len(instruction) < 10 or len(response) < 20:
        stats["drop_empty_or_short"] += 1
        return None

    if domain_filter and not is_in_domain(instruction, response):
        stats["drop_off_domain"] += 1
        return None

    clean_instruction = apply_safe_rewrites(instruction)
    clean_response = apply_safe_rewrites(response)

    if not think_tags_valid(clean_response):
        stats["drop_bad_think_tags"] += 1
        return None
    if is_contaminated(clean_instruction):
        stats["drop_contaminated_instruction"] += 1
        return None
    if is_contaminated(clean_response):
        stats["drop_contaminated_response"] += 1
        return None

    # "the given values" is fine when the instruction actually carries the
    # numbers; phantom otherwise.
    instruction_has_data = len(number_token.findall(clean_instruction)) >= 2
    if not instruction_has_data and (data_reference.search(clean_instruction)
                                     or data_reference.search(clean_response)):
        stats["drop_phantom_data_reference"] += 1
        return None

    if len(clean_instruction) < 10 or len(clean_response) < 20:
        stats["drop_empty_after_clean"] += 1
        return None

    if clean_instruction != instruction or clean_response != response:
        stats["rewritten"] += 1
    stats["kept"] += 1
    return {"instruction": clean_instruction, "response": clean_response, "type": pair_type}


def run_clean(args: argparse.Namespace) -> None:
    stats: Counter = Counter()
    n_chunks_in = n_chunks_out = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_chunks_in += 1
            kept_pairs = []
            for pair in record.get("pairs", []):
                stats["pairs_in"] += 1
                cleaned = clean_pair(pair, stats, domain_filter=not args.no_domain_filter)
                if cleaned is not None:
                    kept_pairs.append(cleaned)
            if kept_pairs:
                n_chunks_out += 1
                fout.write(json.dumps({
                    "chunk_id": record.get("chunk_id", ""),
                    "source": record.get("source", ""),
                    "pair_count": len(kept_pairs),
                    "pairs": kept_pairs,
                }, ensure_ascii=False) + "\n")

    kept_pct = 100 * stats["kept"] / max(1, stats["pairs_in"])
    print(f"[clean] chunks: {n_chunks_in} in -> {n_chunks_out} with >=1 clean pair")
    print(f"[clean] pairs in: {stats['pairs_in']}")
    print(f"[clean] kept:     {stats['kept']} ({kept_pct:.1f}%), of which rewritten "
          f"{stats['rewritten']}")
    for key in sorted(stats):
        if key.startswith("drop_"):
            print(f"[clean] {key}: {stats[key]}")


# ===========================================================================
# stage 2: merge
# ===========================================================================
# Not `cat`, for three reasons:
#   dedup    the cleaner and the rewriter use different contamination regexes,
#            so `cat` would keep both a pair and its rewrite
#   grouping rewritten pairs carry their ORIGINAL chunk_id and must land in the
#            same record as their siblings, or the splitter can put siblings on
#            opposite sides of the boundary -- the leak the split exists to stop
#   sweep    one final contamination pass over everything that survives

merge_contamination = re.compile(
    r"\b(?:the|this|that)\s+(?:text|passage|excerpt|document|paper|article)\b"
    r"|\b(?:the|this|that)\s+stud(?:y|ies)\b(?!\s+of\b)"
    r"|\bthe\s+(?:authors?|researchers)\b"
    r"|\bet\s+al\b"
    r"|\b(?:fig(?:ure)?|table|section|appendix)s?\s+\(?\d"
    r"|\b(?:Fig|Tab|Eq|Sect?)s?\.\s*\(?\d",
    re.IGNORECASE,
)


def instruction_key(text: str) -> str:
    """Normalized instruction, used to detect duplicates across sources."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def load_chunk_records(path: Path, label: str, stats: Counter) -> List[Dict]:
    if not path or not path.exists():
        print(f"[skip] {label}: {path} not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    n_pairs = sum(len(r.get("pairs", [])) for r in records)
    stats[f"{label}_chunks"] = len(records)
    stats[f"{label}_pairs"] = n_pairs
    print(f"[load] {label:<14} {len(records):>5} chunks, {n_pairs:>6} pairs  ({path})")
    return records


def run_merge(args: argparse.Namespace) -> None:
    stats: Counter = Counter()
    sources = [
        ("cleaned", load_chunk_records(Path(args.cleaned), "cleaned", stats)),
        ("rewritten", load_chunk_records(Path(args.rewritten), "rewritten", stats)),
        ("fundamentals", load_chunk_records(Path(args.fundamentals), "fundamentals", stats)),
    ]
    for i, extra in enumerate(args.extra):
        label = Path(extra).stem or f"extra{i}"
        sources.append((label, load_chunk_records(Path(extra), label, stats)))

    # chunk_id -> merged record; insertion order preserved for reproducibility
    merged: "OrderedDict[str, Dict]" = OrderedDict()
    seen_instructions = set()
    dropped: Counter = Counter()

    for label, records in sources:
        for record in records:
            chunk_id = record.get("chunk_id") or ""
            if not chunk_id:
                dropped["missing_chunk_id"] += 1
                continue
            for pair in record.get("pairs", []):
                instruction = (pair.get("instruction") or "").strip()
                response = (pair.get("response") or "").strip()
                if not instruction or not response or len(instruction) < 10 or len(response) < 20:
                    dropped["too_short"] += 1
                    continue

                key = instruction_key(instruction)
                if key in seen_instructions:
                    dropped[f"duplicate_from_{label}"] += 1
                    continue
                if not args.keep_contaminated and (merge_contamination.search(instruction)
                                                   or merge_contamination.search(response)):
                    dropped[f"contaminated_from_{label}"] += 1
                    continue

                seen_instructions.add(key)
                entry = merged.setdefault(chunk_id, {
                    "chunk_id": chunk_id,
                    "source": record.get("source", label),
                    "pair_count": 0,
                    "pairs": [],
                })
                entry["pairs"].append({
                    "type": pair.get("type", "conceptual"),
                    "instruction": instruction,
                    "response": response,
                })
                stats[f"kept_{label}"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for record in merged.values():
            record["pair_count"] = len(record["pairs"])
            total_pairs += record["pair_count"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 62)
    print(f"MERGED -> {out_path}")
    print("=" * 62)
    for label, _ in sources:
        if stats[f"kept_{label}"]:
            print(f"  kept from {label:<14} {stats[f'kept_{label}']:>6}")
    print(f"  {'TOTAL pairs':<25} {total_pairs:>6}")
    print(f"  {'unique chunks':<25} {len(merged):>6}")
    if dropped:
        print("\n  dropped:")
        for key, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"    {key:<38} {count:>6}")
    print("=" * 62)
    print("\nNext:  python -m src.sft.build_corpus split")


# ===========================================================================
# stage 3: split
# ===========================================================================


def format_chatml(instruction: str, response: str) -> str:
    return (f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{response.strip()}<|im_end|>")


def run_split(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input):
        raise SystemExit(f"[error] input file {args.input} does not exist")
    os.makedirs(args.output_dir, exist_ok=True)

    # Group pairs by source chunk so the split can be made at the level of the
    # generating document. A pair-level split leaks: each chunk yields ~3 Q&A
    # pairs about the same excerpt, so val questions would have train siblings
    # covering the same facts and val loss would measure memorization.
    chunks: Dict[str, List[Dict]] = {}
    skipped = 0

    print(f"[split] reading {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            chunk_id = record.get("chunk_id") or f"line{line_no}"
            for pair in record.get("pairs", []):
                instruction = pair.get("instruction", "").strip()
                response = pair.get("response", "").strip()
                if not instruction or not response or len(instruction) < 10 or len(response) < 20:
                    skipped += 1
                    continue
                chunks.setdefault(chunk_id, []).append({
                    "instruction": instruction,
                    "response": response,
                    "type": pair.get("type", "general"),
                    "chunk_id": chunk_id,
                    "chatml": format_chatml(instruction, response),
                })

    n_pairs = sum(len(v) for v in chunks.values())
    print(f"[split] {n_pairs} pairs from {len(chunks)} chunks ({skipped} skipped/invalid)")

    chunk_ids = sorted(chunks.keys())

    # Test split first, assigned by hashing chunk_id rather than by shuffling.
    # A shuffled test set is re-drawn every time the corpus changes, so two
    # model versions end up with different held-out data and their losses are
    # not comparable. Hashing pins a chunk to a split forever, so a chunk held
    # out today stays held out after the corpus grows.
    test_ids = set()
    if args.test_frac > 0:
        bucket = max(1, int(round(1.0 / args.test_frac)))
        for chunk_id in chunk_ids:
            digest = hashlib.md5(f"seistest::{chunk_id}".encode("utf-8")).hexdigest()[:8]
            if int(digest, 16) % bucket == 0:
                test_ids.add(chunk_id)

    # Chunk-level val split over what remains: shuffle chunk ids, assign whole
    # chunks to val until the target pair fraction is reached.
    remaining = [c for c in chunk_ids if c not in test_ids]
    random.seed(args.seed)
    random.shuffle(remaining)

    target_val_pairs = int(n_pairs * args.val_frac)
    val_data: List[Dict] = []
    train_data: List[Dict] = []
    for chunk_id in remaining:
        if len(val_data) < target_val_pairs:
            val_data.extend(chunks[chunk_id])
        else:
            train_data.extend(chunks[chunk_id])
    test_data = [pair for chunk_id in sorted(test_ids) for pair in chunks[chunk_id]]

    # The invariant the whole design exists to protect. Assert it on every
    # build: a leak is a property of the generating process, so fixing the
    # splitter once does not immunize the next data source (PITFALLS A1).
    train_chunks = {p["chunk_id"] for p in train_data}
    val_chunks = {p["chunk_id"] for p in val_data}
    assert train_chunks.isdisjoint(val_chunks), "train and val share a source chunk"
    assert train_chunks.isdisjoint(test_ids), "train and test share a source chunk"
    assert val_chunks.isdisjoint(test_ids), "val and test share a source chunk"

    random.shuffle(train_data)
    random.shuffle(val_data)

    for name, rows in (("train", train_data), ("val", val_data), ("test", test_data)):
        if not rows:
            continue
        path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        n_chunks = len({p["chunk_id"] for p in rows})
        print(f"[split] {name:<5} {len(rows):>6} pairs  {n_chunks:>5} chunks  -> {path}")
    print("[split] invariant: train/val/test chunk sets are pairwise disjoint -> OK")


# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    p_clean = sub.add_parser("clean", help="drop phantom-source and off-domain pairs")
    p_clean.add_argument("--input", default="data/sources/paper_raw.jsonl")
    p_clean.add_argument("--output", default="data/sources/paper.jsonl")
    p_clean.add_argument("--no-domain-filter", action="store_true",
                         help="keep off-domain pairs (default: drop them)")
    p_clean.set_defaults(func=run_clean)

    p_merge = sub.add_parser("merge", help="combine sources, dedup, group by chunk")
    p_merge.add_argument("--cleaned", default="data/sources/paper.jsonl")
    p_merge.add_argument("--rewritten", default="data/sources/paper_rewritten.jsonl")
    p_merge.add_argument("--fundamentals", default="data/sources/fundamentals.jsonl")
    p_merge.add_argument("--extra", action="append",
                         default=["data/sources/earthquakes.jsonl"],
                         help="additional chunk-format JSONL to merge (repeatable)")
    p_merge.add_argument("--output", default="data/sources/merged.jsonl")
    p_merge.add_argument("--keep-contaminated", action="store_true",
                         help="skip the final contamination sweep (not recommended)")
    p_merge.set_defaults(func=run_merge)

    p_split = sub.add_parser("split", help="emit ChatML train/val/test, split by chunk")
    p_split.add_argument("--input", default="data/sources/merged.jsonl")
    p_split.add_argument("--output-dir", default="data/splits")
    p_split.add_argument("--val-frac", type=float, default=0.05)
    p_split.add_argument("--test-frac", type=float, default=0.05,
                         help="held-out fraction, assigned by chunk_id hash so the same "
                              "chunks stay held out as the corpus grows (0 disables)")
    p_split.add_argument("--seed", type=int, default=42)
    p_split.set_defaults(func=run_split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
