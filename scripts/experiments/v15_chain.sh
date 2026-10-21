#!/bin/bash
# GRPO v15: repeated practice (~8 visits per problem) on the corpus problems, from the deployed v14 checkpoint.
# Re-filter for the deployed model -> pool -> GRPO -> dev selection -> 6-seed real held-out -> 64-sample ceiling.
set -u
S=${WORK:-work/experiments}; mkdir -p $S
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
DEP=checkpoints/grpo_v14_corpus/grpo_ckpt_step_75.pt
VISITS=${VISITS:-8}
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/v15_progress.log; }

log "1 re-filter the corpus exercises and their stepping stones for the deployed model (8 samples, T 0.8)"
cat data/rl/v14_filtered_corpus.jsonl data/rl/v14_filtered_stones.jsonl > $S/v15_cand.jsonl
$PY -c "
import json; R = [l for l in open('$S/v15_cand.jsonl')]
for p in (0, 1): open('$S/v15_in_%d.jsonl' % p, 'w').writelines(R[p::2])"
( CUDA_VISIBLE_DEVICES=0 $PY tools/filter_learnable_prompts.py --ckpt $DEP --notation --inp $S/v15_in_0.jsonl --out $S/v15_out_0.jsonl --seed 21 --batch-prompts 12 > $S/v15_filter_0.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $PY tools/filter_learnable_prompts.py --ckpt $DEP --notation --inp $S/v15_in_1.jsonl --out $S/v15_out_1.jsonl --seed 22 --batch-prompts 12 > $S/v15_filter_1.log 2>&1 ) &
wait
cat $S/v15_out_0.jsonl $S/v15_out_1.jsonl > data/rl/v15_filtered_corpus.jsonl
echo "  $(tail -1 $S/v15_filter_0.log) || $(tail -1 $S/v15_filter_1.log)" | tee -a $S/v15_progress.log

log "2 pool"
$PY tools/build_rl_pool.py build --inp data/rl/v15_filtered_corpus.jsonl --out data/rl/train_v15_repeat.jsonl 2>&1 | tee -a $S/v15_progress.log
N=$(wc -l < data/rl/train_v15_repeat.jsonl); STEPS=$(( (N * VISITS + 63) / 64 ))
sed -i "s/^  max_steps: .*/  max_steps: $STEPS/" configs/rl_grpo_v15_repeat.yaml

log "3 GRPO v15: $N problems x $VISITS visits = $STEPS steps, 2 GPUs"
mkdir -p checkpoints/grpo_v15_repeat
$TR --standalone --nproc_per_node=2 --master_port 29715 -m src.rl.train_grpo --config configs/rl_grpo_v15_repeat.yaml > checkpoints/grpo_v15_repeat/train.log 2>&1
grep -q "training complete" checkpoints/grpo_v15_repeat/train.log || { log "FAILED: GRPO did not complete"; exit 1; }

log "4 dev selection: val_numerical_dev (validation chunks; their excerpts were practised too), 4 seeds x 4"
M0="deployed=$DEP:potnot"; M1=""; i=0
for s in $(ls checkpoints/grpo_v15_repeat/grpo_ckpt_step_*.pt | sed 's/.*step_\([0-9]*\)\.pt/\1/' | sort -n); do
  if [ $((i % 2)) = 0 ]; then M1="${M1:+$M1,}v15-$s=checkpoints/grpo_v15_repeat/grpo_ckpt_step_$s.pt:potnot"
  else M0="$M0,v15-$s=checkpoints/grpo_v15_repeat/grpo_ckpt_step_$s.pt:potnot"; fi; i=$((i+1))
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/v15_dev_0.json "$M0" 31,32,33,34 "dev=data/rl/val_numerical_dev.jsonl" > $S/v15_dev_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/v15_dev_1.json "$M1" 31,32,33,34 "dev=data/rl/val_numerical_dev.jsonl" > $S/v15_dev_1.log 2>&1 &
wait
$PY - <<PYEOF | tee -a $S/v15_progress.log
import json
r = {**json.load(open('$S/v15_dev_0.json')), **json.load(open('$S/v15_dev_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
for m in sorted(acc, key=lambda m: int(m.split('-')[1]) if '-' in m else 0): print(f'  dev {m}: {acc[m]:.4f}')
best = max(acc, key=lambda m: (round(acc[m], 4), -int(m.split('-')[1]) if '-' in m else 0))
json.dump({'v15': best, 'dev': acc}, open('$S/v15_best.json', 'w')); print('  selected:', best)
PYEOF

B=$($PY -c "import json; print(json.load(open('$S/v15_best.json'))['v15'])")
if [ "$B" = "deployed" ]; then C=$DEP; else C=checkpoints/grpo_v15_repeat/grpo_ckpt_step_${B#*-}.pt; fi
L=$(ls checkpoints/grpo_v15_repeat/grpo_ckpt_step_*.pt | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
log "5 real held-out, 6 seeds x 4: selected $B ($C) and the last checkpoint ($L)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/real6_eval.py $S/real6_v15_sel.json $C 31,32,33,34,35,36 1.0 1 "v15 selected ($B)" > $S/real6_v15_sel.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/real6_eval.py $S/real6_v15_last.json $L 31,32,33,34,35,36 1.0 1 "v15 last" > $S/real6_v15_last.log 2>&1 &
wait
log "6 sampling ceiling of the selected checkpoint (64 samples, T 0.8)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/ceiling_check.py numerical $C v15 0.8 64 > $S/ceil_num_v15.log 2>&1
log "done"
