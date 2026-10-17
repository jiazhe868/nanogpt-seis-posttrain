"""Turn the SFT splits into RL prompts the verifiers can actually score.

Labels each numerical item with the value its reference derivation concludes
(src.rl.numerical_target — including scientific notation, "1.8 x 10^19 N·m" ->
1.8e19) and extracts key concepts, excluding words already in the prompt so
echoing the question cannot earn concept coverage.

Ends with a hard schema self-check. Without it, a build that fails silently
leaves the trainer reading a stale file whose fields the verifiers cannot find
— which is how one run spent 300 steps with a dead reward (PITFALLS D1).

    python -m src.rl.prepare_rl_dataset                      # build train/val from data/splits
    python -m src.rl.prepare_rl_dataset relabel data/rl/*.jsonl   # re-derive labels in place

`relabel` rewrites only the numerical labels of existing RL files: it keeps the
previous value as `final_numerical_value_v1` and records how each new one was
found in `label_source`, so every changed label stays auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.rl.numerical_target import (  # noqa: F401  (re-exported for older imports)
    BARE_POWER, SCI_NOTATION, derive_numerical_target, normalize_scientific_notation)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "splits"
RL_DIR = ROOT / "data" / "rl"

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
    "these", "those", "it", "its", "they", "them", "their", "which", "what", "where",
    "when", "how", "why", "can", "could", "would", "should", "also", "thus", "such",
    "using", "given", "value", "values", "approximately", "therefore", "result",
    "results", "resulting", "calculated", "calculate", "formula", "final", "answer",
}

def legacy_extract_final_numerical_value(text: str) -> Optional[float]:
    """The v1 labelling rule, kept only to audit old labels: the last regex match of an
    answer marker / '[=≈~]' / 'is' in the final-answer text. About half of the numerical
    labels it produced were wrong (see src.rl.numerical_target); do not use it for new data."""
    text = normalize_scientific_notation(text)
    # Strip thousands separators inside numbers ("10,000" -> "10000")
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)

    patterns = [
        # Explicit final-answer markers first
        r"(?:final answer|answer)[:\s]*(?:is\s*)?(?:[A-Za-z_0-9(){}\\]*\s*[=≈~]\s*)?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"(?:equals?|[=≈~]|is approximately|is about|is)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:km|m|mm|cm|m/s|km/s|mm/yr|GPa|MPa|Pa|N|N[·*]?m|dyn|TECU|min|Hz|s|g|%)\b",
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
    ]

    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            # Last match: final-answer convention
            try:
                val = float(matches[-1])
                if not (math.isnan(val) or math.isinf(val)):
                    return val
            except ValueError:
                pass
    return None


def extract_key_concepts(text: str, prompt: str, max_concepts: int = 8) -> List[str]:
    """Significant answer terms, excluding prompt words and prefix duplicates."""
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    prompt_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", prompt.lower()))
    words = re.findall(r"\b[a-zA-Z]{4,}\b", clean_text.lower())

    freq: Dict[str, int] = {}
    for w in words:
        if w not in STOP_WORDS and w not in prompt_words:
            freq[w] = freq.get(w, 0) + 1

    ranked = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)]
    concepts: List[str] = []
    for w in ranked:
        # skip prefix-duplicates of an already-selected concept
        if any(w.startswith(c[:5]) or c.startswith(w[:5]) for c in concepts):
            continue
        concepts.append(w)
        if len(concepts) >= max_concepts:
            break
    return concepts


def sanitize_think_tags(text: str) -> str:
    """Fix broken unclosed or mismatched <think> tags."""
    has_start = "<think>" in text
    has_end = "</think>" in text

    if has_start and not has_end:
        text = text + "\n</think>"
    elif has_end and not has_start:
        text = "<think>\n" + text
    return text


def process_sft_file(input_path: Path, output_path: Path) -> int:
    """Process SFT JSONL file into the RL prompt dataset format."""
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    n_numerical = 0
    n_with_target = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)

            instruction = item.get("instruction", "")
            response = sanitize_think_tags(item.get("response", ""))
            task_type = item.get("type", "conceptual")

            think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
            if think_match:
                think_block = think_match.group(1).strip()
                final_answer = response[think_match.end():].strip()
            else:
                think_block = ""
                final_answer = response.strip()

            final_num, label_source = (derive_numerical_target(instruction, think_block, final_answer)
                                       if task_type == "numerical" else (None, None))
            key_concepts = extract_key_concepts(final_answer, instruction)

            if task_type == "numerical":
                n_numerical += 1
                if final_num is not None:
                    n_with_target += 1

            rl_item = {
                "id": count,
                "prompt": instruction,
                "type": task_type,
                "chunk_id": item.get("chunk_id", ""),
                "reference_response": response,
                "reference_final_answer": final_answer,
                "reference_think": think_block,
                "final_numerical_value": final_num,
                "label_source": label_source,
                "key_concepts": key_concepts,
            }

            f_out.write(json.dumps(rl_item, ensure_ascii=False) + "\n")
            count += 1

    print(f"[INFO] {input_path.name}: {count} items -> {output_path}")
    if n_numerical:
        print(f"[INFO]   numerical items: {n_numerical}, with extracted target: "
              f"{n_with_target} ({100 * n_with_target / n_numerical:.1f}%)")

    # Hard schema self-check: the exact failure mode of the V2 run (verifier
    # reading a key the dataset never contained) must be impossible to repeat.
    with open(output_path, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    assert all("final_numerical_value" in r and "key_concepts" in r for r in recs), \
        "schema check failed: missing verifier target fields"
    num_targets = sum(1 for r in recs if r["type"] == "numerical" and r["final_numerical_value"] is not None)
    if n_numerical > 0:
        assert num_targets > 0.8 * n_numerical, \
            f"schema check failed: only {num_targets}/{n_numerical} numerical items have targets"
    return count


def relabel_file(path: Path) -> Dict[str, int]:
    """Re-derive the numerical labels of an existing RL file in place (atomic rewrite)."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    stats = {"numerical": 0, "unchanged": 0, "changed": 0, "now_unlabelled": 0, "newly_labelled": 0}
    for r in rows:
        if r.get("type") != "numerical":
            continue
        stats["numerical"] += 1
        old = r.get("final_numerical_value_v1", r.get("final_numerical_value"))
        new, source = derive_numerical_target(r["prompt"], r.get("reference_think", ""),
                                              r.get("reference_final_answer", ""))
        r["final_numerical_value_v1"] = old
        r["final_numerical_value"] = new
        r["label_source"] = source
        if new is None:
            stats["unchanged" if old is None else "now_unlabelled"] += 1
        elif old is None:
            stats["newly_labelled"] += 1
        else:
            stats["unchanged" if abs(new - old) <= 0.01 * max(abs(old), 1e-12) else "changed"] += 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build Stage 3 RL prompt datasets, or relabel existing ones.")
    parser.add_argument("command", nargs="?", default="build", choices=["build", "relabel"])
    parser.add_argument("files", nargs="*", help="relabel: RL jsonl files to rewrite in place")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--out-dir", type=str, default=str(RL_DIR))
    args = parser.parse_args()

    if args.command == "relabel":
        for f in args.files or [str(RL_DIR / f"{s}.jsonl") for s in ("train", "val", "test")]:
            st = relabel_file(Path(f))
            n = max(st["numerical"], 1)
            print(f"[relabel] {f}: {st['numerical']} numerical | unchanged {st['unchanged']} ({st['unchanged']/n:.0%}) | "
                  f"changed {st['changed']} ({st['changed']/n:.0%}) | now unlabelled {st['now_unlabelled']} | "
                  f"newly labelled {st['newly_labelled']}")
        return

    processed = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    print("=" * 80)
    print("🚀 Preparing Stage 3 RL Prompt Datasets (V3)")
    print("=" * 80)

    train_count = process_sft_file(processed / "train.jsonl", out_dir / "train.jsonl")
    val_count = process_sft_file(processed / "val.jsonl", out_dir / "val.jsonl")

    print(f"\n[SUCCESS] RL datasets ready: {train_count} train / {val_count} val -> {out_dir}")


if __name__ == "__main__":
    main()
