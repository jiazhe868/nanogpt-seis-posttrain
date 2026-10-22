"""The clean dev set (README §3.1): the labelled numerical prompts of the SFT validation split.

data/rl/val_numerical_dev.jsonl came from an older split and 74 of its 96 prompts are in the Phase II
training split. data/splits/val.jsonl is chunk-disjoint from train and test; its numerical prompts get the
test set's label rule (src.rl.prepare_rl_dataset.process_sft_file). Ids are offset by 200000.

    python scripts/experiments/build_dev_clean.py      # writes data/rl/dev_clean.jsonl
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl.prepare_rl_dataset import process_sft_file  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    tmp = Path(d) / "val_rl.jsonl"
    process_sft_file(ROOT / "data/splits/val.jsonl", tmp)
    rows = [json.loads(l) for l in open(tmp, encoding="utf-8")]
rows = [dict(r, id=200000 + r["id"]) for r in rows if r["type"] == "numerical" and r["final_numerical_value"] is not None]
with open(ROOT / "data/rl/dev_clean.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} labelled numerical dev prompts -> data/rl/dev_clean.jsonl")
