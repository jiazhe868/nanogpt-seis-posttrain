#!/bin/bash
# Follow-up to scripts/experiments/round6_chain.sh before deploying GRPO v20: the plain arm again with seed 43 for
# three passes (how much of one run's result is luck?), then the 153 real held-out questions and the 300 new problems
# for both seeds' final checkpoints. scripts/experiments/build_eval_rounds.py pools both seeds against round 5's two
# seeds (v19 plain step 51 and v19b plain step 51; the learning rate is constant after warmup, so step 51 of the
# six-pass v19b run is a three-pass replicate).
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/round6_rl_progress.log; }

STEPS=$(sed -n 's/^  max_steps: //p' configs/rl_grpo_v20_plain.yaml)
sed -e "s|three passes (scripts/experiments/round6_chain.sh)|three passes, seed 43 (scripts/experiments/round6_rl_more.sh)|" \
    -e "s|grpo_v20_plain|grpo_v20b_plain|g; s|^  seed: .*|  seed: 43|" configs/rl_grpo_v20_plain.yaml > configs/rl_grpo_v20b_plain.yaml
log "GRPO v20b plain: seed 43, three passes ($STEPS steps)"
mkdir -p checkpoints/grpo_v20b_plain
$TR --standalone --nproc_per_node=2 --master_port 29722 -m src.rl.train_grpo --config configs/rl_grpo_v20b_plain.yaml \
  > checkpoints/grpo_v20b_plain/train.log 2>&1 < /dev/null
grep -q "training complete" checkpoints/grpo_v20b_plain/train.log || { log "FAILED: v20b plain"; exit 1; }

for arm in v20 v20b; do
  C=checkpoints/grpo_${arm}_plain/grpo_ckpt_step_$STEPS.pt
  log "$arm plain step $STEPS: real held-out (6 x 4) and in-distribution (300 x 4)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/real6_eval.py $S/${arm}_plain_real6.json $C 31,32,33,34,35,36 1.0 1 \
    "$arm plain $STEPS" > $S/${arm}_plain_real6.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/set_eval.py $S/${arm}_plain_indist.json data/eval/indist_300.jsonl $C \
    0,1,2,3 notation ${arm}_plain > $S/${arm}_plain_indist.log 2>&1 < /dev/null &
  wait
  $PY - <<PYEOF | tee -a $S/round6_rl_progress.log
import json, statistics as st
a = next(iter(json.load(open('$S/${arm}_plain_real6.json'))['results'].values()))
b = next(iter(json.load(open('$S/${arm}_plain_indist.json'))['results'].values()))
print(f"  $arm plain $STEPS: held-out {100 * st.mean(x['numerical_acc_5pct'] for x in a):.1f}, "
      f"in-distribution {100 * st.mean(x['numerical_acc_5pct'] for x in b):.1f}")
PYEOF
done

log "done"
