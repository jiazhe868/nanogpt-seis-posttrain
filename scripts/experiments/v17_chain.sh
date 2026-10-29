#!/bin/bash
# GRPO v17 (Phase V) from the round-3 SFT model, strict protocol: candidates = the real-style problems of
# tools/gen_real_style.py + the real-question rewrites (data/rl/v13_filtered_existing.jsonl), re-filtered to those
# the model solves 1-7 of 8; one pass; with and without SFT replay (ptx); clean-dev selection; real held-out.
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; mkdir -p $S
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
BASE=checkpoints/sft_pot_notation3/sft_ckpt_epoch4.pt
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/v17_progress.log; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # SFT replay + rollout CUDA graphs fragment memory
export ROLLOUT_MAX_GRAPHS=1                                 # long real-style prompts: one cached decoder graph
if [ ! -s data/rl/train_v17_pool.jsonl ]; then
log "candidates"
$PY - <<'PYEOF' | tee -a $S/v17_progress.log
import json
rows = []
for k, r in enumerate(map(json.loads, open('data/sources/real_style/notation3_problems.jsonl'))):
    rows.append({"id": f"rs_{k}", "prompt": r["question"], "type": "numerical", "final_numerical_value": float(r["answer"]),
                 "key_concepts": [], "program": r.get("program_python", ""), "chunk_id": r.get("chunk_id"), "source": "realstyle"})
for r in map(json.loads, open('data/rl/v13_filtered_existing.jsonl')):
    rows.append({k: v for k, v in r.items() if k not in ("passes", "k")})
for h in range(2):
    with open(f'data/rl/v17_cand_{h}.jsonl', 'w') as f:
        for r in rows[h::2]: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  {len(rows)} candidates")
PYEOF
log "learnability filter with the round-3 model (8 samples, T 0.8)"
for h in 0 1; do
  CUDA_VISIBLE_DEVICES=$h $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/v17_cand_$h.jsonl \
    --out data/rl/v17_filtered_$h.jsonl --seed $((41 + h)) --batch-prompts 12 > $S/v17_filter_$h.log 2>&1 &
done
wait
$PY tools/build_rl_pool.py build --inp data/rl/v17_filtered_0.jsonl data/rl/v17_filtered_1.jsonl --out data/rl/train_v17_pool.jsonl 2>&1 | tail -2 | tee -a $S/v17_progress.log
fi
N=$(wc -l < data/rl/train_v17_pool.jsonl); STEPS=$(( (N + 63) / 64 )); SAVE=$(( (STEPS + 5) / 6 ))
for arm in ptx plain; do
  sed -i "s/^  max_steps: .*/  max_steps: $STEPS/; s/^  save_every: .*/  save_every: $SAVE/" configs/rl_grpo_v17_$arm.yaml
  log "GRPO v17 $arm: $N problems, $STEPS steps (one pass), checkpoints every $SAVE"
  mkdir -p checkpoints/grpo_v17_$arm
  $TR --standalone --nproc_per_node=2 --master_port 29717 -m src.rl.train_grpo --config configs/rl_grpo_v17_$arm.yaml \
    > checkpoints/grpo_v17_$arm/train.log 2>&1
  grep -q "training complete" checkpoints/grpo_v17_$arm/train.log || { log "FAILED: $arm"; exit 1; }
done
log "clean-dev selection"
M0="n3=$BASE:potnot"; M1=""; i=0
for arm in ptx plain; do
  for s in $(ls checkpoints/grpo_v17_$arm/grpo_ckpt_step_*.pt | sed 's/.*step_\([0-9]*\)\.pt/\1/' | sort -n); do
    m="$arm-$s=checkpoints/grpo_v17_$arm/grpo_ckpt_step_$s.pt:potnot"
    if [ $((i % 2)) = 0 ]; then M1="${M1:+$M1,}$m"; else M0="$M0,$m"; fi; i=$((i+1))
  done
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/v17_dev_0.json "$M0" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/v17_dev_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/v17_dev_1.json "$M1" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/v17_dev_1.log 2>&1 &
wait
B=$($PY - <<PYEOF
import json
r = {**json.load(open('$S/v17_dev_0.json')), **json.load(open('$S/v17_dev_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
with open('$S/v17_progress.log', 'a') as f:
    for m in sorted(acc): f.write(f"  dev {m}: {100 * acc[m]:.1f}\n")
best = max(acc, key=lambda m: (round(acc[m], 4), m == 'n3'))
json.dump({'best': best, 'dev': acc}, open('$S/v17_best.json', 'w'))
print(best)
PYEOF
)
if [ "$B" = "n3" ]; then C=$BASE; else C=checkpoints/grpo_v17_${B%%-*}/grpo_ckpt_step_${B#*-}.pt; fi
log "selected $B ($C): real held-out, 6 x 4"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/real6_eval.py $S/v17_real6.json $C 31,32,33,34,35,36 1.0 1 "v17 $B" > $S/v17_real6.log 2>&1
log "done"
