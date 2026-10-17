"""6-seed (or any) real held-out evaluation of one program-format checkpoint.
usage: real6_eval.py out.json ckpt seeds rp notation(0|1) name"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.rl.eval_rl import evaluate_checkpoint
out, ck, seeds, rp, notation, name = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")], \
    float(sys.argv[4]), sys.argv[5] == "1", sys.argv[6]
items = [json.loads(l) for l in open("data/rl/test.jsonl")]
num = [it for it in items if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
runs = [evaluate_checkpoint(Path(ck), num, num_samples=4, seed=s, batched=True, pot=True, pot_notation=notation,
                            repetition_penalty=rp, batch_prompts=16) for s in seeds]
json.dump({"ckpt": ck, "rp": rp, "notation": notation, "results": {name: runs}}, open(out, "w"))
