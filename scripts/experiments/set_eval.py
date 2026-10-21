"""Multi-seed evaluation of one checkpoint on any numerical problem file (README §1.2 in-distribution panel).

usage: set_eval.py out.json items.jsonl ckpt seeds decoder name
    decoder: prose      (T 0.3, top-k 40, rp 1.2: the Phase II/III models)
             notation   (executed paper-notation equations, T 0.3, top-k 40, rp 1.0: Phase IV and later)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.rl.eval_rl import evaluate_checkpoint  # noqa: E402

out, items_path, ck, seeds, decoder, name = sys.argv[1:7]
items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
items = [it for it in items if it.get("type") == "numerical" and it.get("final_numerical_value") is not None]
kw = dict(pot=True, pot_notation=True, repetition_penalty=1.0) if decoder == "notation" else dict(repetition_penalty=1.2)
runs = [evaluate_checkpoint(Path(ck), items, num_samples=4, seed=int(s), batched=True, batch_prompts=16, **kw)
        for s in seeds.split(",")]
json.dump({"ckpt": ck, "items": items_path, "decoder": decoder, "seeds": seeds, "results": {name: runs}}, open(out, "w"))
print(name, [round(r["numerical_acc_5pct"], 4) for r in runs])
