"""Turn the SFT splits into RL prompts the verifiers can actually score.

Extracts each item's final numerical value (including scientific notation:
"1.8 x 10^19 N·m" -> 1.8e19) and its key concepts, excluding words already in
the prompt so echoing the question cannot earn concept coverage.

Ends with a hard schema self-check. Without it, a build that fails silently
leaves the trainer reading a stale file whose fields the verifiers cannot find
— which is how one run spent 300 steps with a dead reward (PITFALLS D1).

    python -m src.rl.prepare_rl_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# "1.8 × 10^19", "1.8 x 10**19", "1.8·10^19" -> "1.8e19"
SCI_NOTATION = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*[×x·*]\s*10\s*(?:\^|\*\*)?\s*[({\[]?([-+]?\d+)[)}\]]?"
)
# Bare "10^19" (no mantissa) -> "1e19"
BARE_POWER = re.compile(r"(?<![\d.eE])10\s*(?:\^|\*\*)\s*[({\[]?([-+]?\d+)[)}\]]?")


def normalize_scientific_notation(text: str) -> str:
    text = SCI_NOTATION.sub(lambda m: f"{m.group(1)}e{m.group(2)}", text)
    text = BARE_POWER.sub(lambda m: f"1e{m.group(1)}", text)
    return text


def extract_final_numerical_value(text: str) -> Optional[float]:
    """Extract the primary final numerical answer from post-<think> text."""
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

            final_num = extract_final_numerical_value(final_answer) if task_type == "numerical" else None
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


def main():
    parser = argparse.ArgumentParser(description="Build Stage 3 RL prompt datasets (V3).")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--out-dir", type=str, default=str(RL_DIR))
    args = parser.parse_args()

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
