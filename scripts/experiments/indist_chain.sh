#!/bin/bash
# In-distribution set (README §1.2): new teacher problems seeded by SFT-train questions (fine-tuning papers),
# never in any SFT or RL training set, labels verified by a second independent solution, 300 sampled with
# no solve-rate filter; then every phase model scored with 6 seeds x 4.
set -u
S=${WORK:-work/experiments}; mkdir -p $S data/eval
PY=${PY:-python}; VP=${VLLM_PY:-python}
export VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/indist_progress.log; }

log "1 generate (27B teacher, seeds from data/splits/train.jsonl)"
$VP tools/gen_pot_problems.py generate --n-seeds 500 --per-seed 2 --seed-pool all --seed 9001 --out $S/indist_gen.jsonl > $S/indist_gen.log 2>&1
log "2 independent re-solve"
$VP tools/gen_pot_problems.py solve --inp $S/indist_gen.jsonl --out $S/indist_solved.jsonl > $S/indist_solve.log 2>&1
log "3 verified only, deduplicated against every SFT and RL training set, 300 sampled"
$PY - <<'PYEOF' | tee -a $S/indist_progress.log
import glob, json, random
from tools.build_rl_pool import NearestNeighbour
S = "work/experiments"
rows = [json.loads(l) for l in open(f"{S}/indist_solved.jsonl")]
ver = [r for r in rows if r.get("verified")]
train = [json.loads(l)["instruction"] for f in ("data/splits/train_pot_notation2.jsonl", "data/splits/train.jsonl") for l in open(f)]
pools = [json.loads(l).get("prompt", "") for f in glob.glob("data/rl/train*.jsonl") + glob.glob("data/rl/transfer_*.jsonl")
         + glob.glob("data/rl/v1*_*.jsonl") for l in open(f)]
nn = NearestNeighbour(train + pools)
seen, keep = set(), []
for r in ver:
    q = r["question"].strip()
    if q in seen or nn.best(q) >= 0.8:
        continue
    seen.add(q); keep.append(r)
random.Random(0).shuffle(keep)
out = [{"id": 900000 + k, "prompt": r["question"], "type": "numerical", "final_numerical_value": float(r["answer"]),
        "unit": r.get("unit", ""), "seed_chunk": r.get("seed_chunk"), "source": "indist"} for k, r in enumerate(keep[:300])]
with open("data/eval/indist_300.jsonl", "w") as g:
    g.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
print(f"  generated {len(rows)}, verified {len(ver)}, after dedupe {len(keep)}, sampled {len(out)} -> data/eval/indist_300.jsonl")
PYEOF
log "4 score every phase, 6 seeds x 4"
V15=checkpoints/grpo_v15_repeat/grpo_ckpt_step_135.pt   # end of the repeated-practice run (dev kept the deployed model)
SEEDS=31,32,33,34,35,36
( export CUDA_VISIBLE_DEVICES=0
  $PY scripts/experiments/set_eval.py $S/indist_sft.json data/eval/indist_300.jsonl checkpoints/sft/selected.pt $SEEDS prose SFT > $S/indist_sft.log 2>&1
  $PY scripts/experiments/set_eval.py $S/indist_v4.json data/eval/indist_300.jsonl checkpoints/grpo/selected.pt $SEEDS prose "v4@300" > $S/indist_v4.log 2>&1
  $PY scripts/experiments/set_eval.py $S/indist_n2.json data/eval/indist_300.jsonl checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt $SEEDS notation notation2 > $S/indist_n2.log 2>&1 ) &
( export CUDA_VISIBLE_DEVICES=1
  $PY scripts/experiments/set_eval.py $S/indist_v14.json data/eval/indist_300.jsonl checkpoints/grpo_v14_corpus/grpo_ckpt_step_75.pt $SEEDS notation v14-75 > $S/indist_v14.log 2>&1
  $PY scripts/experiments/set_eval.py $S/indist_v15.json data/eval/indist_300.jsonl $V15 $SEEDS notation v15-135 > $S/indist_v15.log 2>&1 ) &
wait
for f in sft v4 n2 v14 v15; do tail -n 1 $S/indist_$f.log | tee -a $S/indist_progress.log; done
log "done"
