"""The split contract: which data may influence which decision.

A pre-declared selection rule prevents cherry-picking a checkpoint after seeing
the numbers. It does NOT remove selection bias: if the same prompts choose the
checkpoint and produce the reported number, that number is a development
result, not a held-out one.

This project shipped exactly that mistake. checkpoint selection and `eval_rl`
both defaulted to the full 808-prompt held-out pool, so the reported
"GRPO @100 = 10.2%" was measured on prompts that had already been used to pick
step 100 over step 200.

The contract, keyed by source chunk so siblings never straddle the boundary:

    train        SFT / RL parameter updates.
    val          training-time monitoring only (early stopping, loss curves).
    dev          hyperparameters, decoder, thresholds, CHECKPOINT SELECTION,
                 and metric calibration (floor/ceiling, echo thresholds).
    test         reported once. Never used to change code, thresholds,
                 checkpoints, prompts, or decoding.

`dev` and `test` partition the held-out pool by a hash of `chunk_id`, so the
assignment is stable across corpus rebuilds -- the same property that makes the
held-out pool itself stable (see src.sft.build_corpus split).

Usage:
    from src.sft.splits import assign_split, load_split
    items_dev  = load_split("data/rl/test.jsonl", "dev")
    items_test = load_split("data/rl/test.jsonl", "test")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

# Fraction of the held-out pool reserved as the frozen final test set.
TEST_FRACTION = 0.5
_SALT = "seis-final-test::"


def assign_split(chunk_id: str) -> str:
    """'dev' or 'test' for a source chunk. Deterministic, corpus-independent."""
    digest = hashlib.md5(f"{_SALT}{chunk_id}".encode("utf-8")).hexdigest()[:8]
    return "test" if (int(digest, 16) % 1000) / 1000.0 < TEST_FRACTION else "dev"


def split_items(items: Iterable[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {"dev": [], "test": []}
    for it in items:
        out[assign_split(str(it.get("chunk_id", it.get("id", ""))))].append(it)
    return out


def load_split(path: str | Path, which: str) -> List[Dict]:
    """Load one side of the held-out pool. `which` in {'dev','test','all'}."""
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if which == "all":
        return items
    if which not in ("dev", "test"):
        raise ValueError(f"which must be dev/test/all, got {which!r}")
    return split_items(items)[which]


def describe(path: str | Path) -> str:
    parts = split_items(json.loads(l) for l in open(path, encoding="utf-8") if l.strip())
    lines = [f"split contract for {path}"]
    for name, rows in parts.items():
        n_num = sum(1 for r in rows
                    if r.get("type") == "numerical"
                    and r.get("final_numerical_value") is not None)
        chunks = len({r.get("chunk_id") for r in rows})
        lines.append(f"  {name:<5} {len(rows):>4} prompts  {n_num:>4} numerical  "
                     f"{chunks:>4} chunks")
    # the invariant that makes the contract meaningful
    dev_c = {r.get("chunk_id") for r in parts["dev"]}
    test_c = {r.get("chunk_id") for r in parts["test"]}
    assert dev_c.isdisjoint(test_c), "dev and test share a source chunk"
    lines.append("  invariant: dev_chunks.isdisjoint(test_chunks) -> OK")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(describe(sys.argv[1] if len(sys.argv) > 1 else "data/rl/test.jsonl"))
