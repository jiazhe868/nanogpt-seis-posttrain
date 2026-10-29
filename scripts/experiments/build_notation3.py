"""Phase IV mixtures from round 3 on: train_pot_notation2 + the real-style data from tools/gen_real_style.py.

    python scripts/experiments/build_notation3.py data/splits/train_pot_notation3.jsonl pilot pilot2 wA
    python scripts/experiments/build_notation3.py data/splits/train_pot_notation4.jsonl pilot pilot2 wA wB_sft

For each tag, data/sources/real_style/notation_<tag>.jsonl (verified problems in paper notation) and
con_<tag>.jsonl (conceptual pairs) are read. Problems are deduplicated by question, given 3 counterfactual
variants each (tools/pot_counterfactuals.py, new numbers, same formula, exact labels), and formatted as SFT
rows (tools/format_pot_sft.py); conceptual pairs are added as prose rows. The notation2 rows are kept as
they are.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/sources/real_style"
KEYS = ("instruction", "response", "type", "chunk_id")


def main():
    out, tags = Path(sys.argv[1]), sys.argv[2:]
    stem = out.stem.replace("train_pot_", "")           # e.g. notation3 -> data/sources/real_style/notation3_*.jsonl
    probs, seen = [], set()
    for t in tags:
        for r in map(json.loads, open(D / f"notation_{t}.jsonl", encoding="utf-8")):
            q = r["question"].strip()
            if q not in seen:
                seen.add(q); probs.append(r)
    base = D / f"{stem}_problems.jsonl"
    with open(base, "w", encoding="utf-8") as f:
        for r in probs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    py = sys.executable
    subprocess.run([py, "tools/pot_counterfactuals.py", str(base), str(D / f"{stem}_cf.jsonl"), "--variants", "3",
                    "--seed", "3", "--notation"], cwd=ROOT, check=True)
    for name in (f"{stem}_problems", f"{stem}_cf"):
        subprocess.run([py, "tools/format_pot_sft.py", "--notation", "--out", str(D / f"sft_{name}.jsonl"),
                        str(D / f"{name}.jsonl")], cwd=ROOT, check=True)
    rows = [json.loads(l) for l in open(ROOT / "data/splits/train_pot_notation2.jsonl", encoding="utf-8")]
    n2 = len(rows)
    new_num = [{k: r[k] for k in KEYS} for name in (f"{stem}_problems", f"{stem}_cf")
               for r in map(json.loads, open(D / f"sft_{name}.jsonl", encoding="utf-8"))]
    con = [{"instruction": r["instruction"], "response": r["response"], "type": r.get("type", "conceptual"),
            "chunk_id": r["chunk_id"]}
           for t in tags for r in map(json.loads, open(D / f"con_{t}.jsonl", encoding="utf-8"))
           if r.get("instruction") and r.get("response")]
    rows += new_num + con
    random.Random(7).shuffle(rows)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out}: {len(rows)} rows = notation2 {n2} + real-style numerical {len(new_num)} "
          f"({len(probs)} problems with counterfactual variants) + conceptual {len(con)}")


if __name__ == "__main__":
    main()
