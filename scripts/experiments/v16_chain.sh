#!/bin/bash
# GRPO v16: is the reward too sparse? Control (correctness) vs step reward on the same pool, from the deployed
# model; dev selection (val_numerical_dev, 4 x 4) per arm, then the 6-seed real held-out set.
set -u
S=${WORK:-work/experiments}; mkdir -p $S
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
DEP=checkpoints/grpo_v14_corpus/grpo_ckpt_step_75.pt
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/v16_progress.log; }

log "pool: the v15 candidates (re-filtered for the deployed model) it solves 0-7 of 8, with a teacher program"
$PY -c "
import json
rows = [json.loads(l) for l in open('data/rl/v15_filtered_corpus.jsonl')]
with open('data/rl/train_v16_pool.jsonl', 'w') as f:
    for r in rows:
        if r['passes'] < r['k'] and r.get('program'): f.write(json.dumps(r) + '\\n')"

for arm in ctrl step; do
  log "GRPO v16 $arm (2 GPUs)"
  mkdir -p checkpoints/grpo_v16_$arm
  $TR --standalone --nproc_per_node=2 --master_port 29716 -m src.rl.train_grpo --config configs/rl_grpo_v16_$arm.yaml \
    > checkpoints/grpo_v16_$arm/train.log 2>&1
  grep -q "training complete" checkpoints/grpo_v16_$arm/train.log || { log "FAILED: $arm did not complete"; exit 1; }
done

log "dev selection, both arms"
M0="deployed=$DEP:potnot"; M1=""; i=0
for arm in ctrl step; do
  for s in $(ls checkpoints/grpo_v16_$arm/grpo_ckpt_step_*.pt | sed 's/.*step_\([0-9]*\)\.pt/\1/' | sort -n); do
    m="$arm-$s=checkpoints/grpo_v16_$arm/grpo_ckpt_step_$s.pt:potnot"
    if [ $((i % 2)) = 0 ]; then M1="${M1:+$M1,}$m"; else M0="$M0,$m"; fi; i=$((i+1))
  done
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/v16_dev_0.json "$M0" 31,32,33,34 "dev=data/rl/val_numerical_dev.jsonl" > $S/v16_dev_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/v16_dev_1.json "$M1" 31,32,33,34 "dev=data/rl/val_numerical_dev.jsonl" > $S/v16_dev_1.log 2>&1 &
wait
$PY - <<PYEOF | tee -a $S/v16_progress.log
import json
r = {**json.load(open('$S/v16_dev_0.json')), **json.load(open('$S/v16_dev_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
for m in sorted(acc): print(f'  dev {m}: {acc[m]:.4f}')
best = {}
for arm in ('ctrl', 'step'):
    c = {m: a for m, a in acc.items() if m.startswith(arm)}
    best[arm] = max(c, key=lambda m: (round(c[m], 4), -int(m.split('-')[1])))
json.dump({'best': best, 'dev': acc}, open('$S/v16_best.json', 'w')); print('  selected:', best)
PYEOF

log "real held-out, 6 seeds x 4: each arm's dev-selected checkpoint"
for arm in ctrl step; do
  B=$($PY -c "import json; print(json.load(open('$S/v16_best.json'))['best']['$arm'])")
  g=$([ $arm = ctrl ] && echo 0 || echo 1)
  CUDA_VISIBLE_DEVICES=$g $PY scripts/experiments/real6_eval.py $S/real6_v16_$arm.json checkpoints/grpo_v16_$arm/grpo_ckpt_step_${B#*-}.pt \
    31,32,33,34,35,36 1.0 1 "v16 $arm ($B)" > $S/real6_v16_$arm.log 2>&1 &
done
wait
log "done"
