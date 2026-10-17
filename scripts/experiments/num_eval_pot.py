"""acc@5 of checkpoints on several numerical sets, paired by prompt. Model suffixes: ':calc'
(calculator decoding) or ':pot' (program-of-thought execution). usage:
  num_eval_pot.py out.json "name=ckpt[:calc|:pot],..." seeds set1=path,set2=path,..."""
import json, sys, random, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.rl.eval_rl import evaluate_checkpoint
out = sys.argv[1]
models = {}
for x in sys.argv[2].split(","):
    n, ck = x.split("=", 1)
    mode = ck.rsplit(":", 1)[1] if ck.endswith((":calc", ":pot", ":potnorp", ":potnot", ":potnot12")) else ""
    models[n] = (ck[:-len(mode) - 1] if mode else ck, mode)
seeds = [int(s) for s in sys.argv[3].split(",")]
SETS = dict(kv.split("=", 1) for kv in sys.argv[4].split(","))
def load(p):
    its = [json.loads(l) for l in open(p)]
    return [it for it in its if it["type"] == "numerical" and it.get("final_numerical_value") not in (None, "None")]
res = {}
for n, (ck, mode) in models.items():
    res[n] = {}
    for sname, p in SETS.items():
        runs = [evaluate_checkpoint(Path(ck), load(p), num_samples=4, seed=s, batched=True, batch_prompts=16,
                                    calculator=(mode == "calc"), pot=mode.startswith("pot"),
                                    pot_notation=mode.startswith("potnot"),
                                    repetition_penalty=1.0 if mode in ("potnorp", "potnot") else 1.2) for s in seeds]
        acc = {}
        for r in runs:
            for x in r["per_item"]:
                acc.setdefault(str(x["id"]), []).append(x["acc5"])
        res[n][sname] = {i: st.mean(v) for i, v in acc.items()}
json.dump(res, open(out, "w"))
rng = random.Random(0); names = list(res); base = names[0]
print(f"\n{'model':<16}" + "".join(f"{s:>26}" for s in SETS))
for n in names:
    row = f"{n:<16}"
    for s in SETS:
        P, B = res[n][s], res[base][s]; m = st.mean(P.values())
        if n == base: row += f"{m:>12.3f}{'':>14}"
        else:
            d = [P[i] - B[i] for i in P if i in B]
            bs = sorted(st.mean(d[rng.randrange(len(d))] for _ in d) for _ in range(2000))
            row += f"{m:>12.3f} {st.mean(d):+.3f}[{bs[50]:+.2f},{bs[1949]:+.2f}]"
    print(row)
