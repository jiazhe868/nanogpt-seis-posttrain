#!/bin/bash
# Follow-up to scripts/experiments/round5_rl_chain.sh: the 153 real held-out questions and the 300 new problems for
# both arms' final checkpoints, then the plain arm again with another seed for six passes (how much does one RL
# run's luck move it?). The learning rate is constant after warmup, so its step 51 is a three-pass replicate, scored
# the same way.
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/round5_rl_progress.log; }

for arm in plain ptx; do
  C=checkpoints/grpo_v19_$arm/grpo_ckpt_step_51.pt
  log "v19 $arm step 51: real held-out (6 x 4) and in-distribution (300 x 4)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/real6_eval.py $S/v19_${arm}_real6.json $C 31,32,33,34,35,36 1.0 1 "v19 $arm 51" \
    > $S/v19_${arm}_real6.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/set_eval.py $S/v19_${arm}_indist.json data/eval/indist_300.jsonl $C 0,1,2,3 notation \
    v19_$arm > $S/v19_${arm}_indist.log 2>&1 < /dev/null &
  wait
  $PY - <<PYEOF | tee -a $S/round5_rl_progress.log
import json, statistics as st
a = next(iter(json.load(open('$S/v19_${arm}_real6.json'))['results'].values()))
b = next(iter(json.load(open('$S/v19_${arm}_indist.json'))['results'].values()))
print(f"  v19 $arm 51: held-out {100 * st.mean(x['numerical_acc_5pct'] for x in a):.1f}, "
      f"in-distribution {100 * st.mean(x['numerical_acc_5pct'] for x in b):.1f}")
PYEOF
done

sed -e "s|three passes|six passes, seed 43|; s|grpo_v19_plain|grpo_v19b_plain|g" \
    -e "s|^  max_steps: .*|  max_steps: 102|; s|^  seed: .*|  seed: 43|" configs/rl_grpo_v19_plain.yaml > configs/rl_grpo_v19b_plain.yaml
log "GRPO v19b plain: seed 43, six passes (102 steps)"
mkdir -p checkpoints/grpo_v19b_plain
$TR --standalone --nproc_per_node=2 --master_port 29720 -m src.rl.train_grpo --config configs/rl_grpo_v19b_plain.yaml \
  > checkpoints/grpo_v19b_plain/train.log 2>&1 < /dev/null
grep -q "training complete" checkpoints/grpo_v19b_plain/train.log || { log "FAILED: v19b plain"; exit 1; }

C=checkpoints/grpo_v19b_plain/grpo_ckpt_step_51.pt
log "v19b plain step 51: real held-out (6 x 4) and in-distribution (300 x 4)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/real6_eval.py $S/v19b_plain_real6.json $C 31,32,33,34,35,36 1.0 1 "v19b plain 51" \
  > $S/v19b_plain_real6.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/set_eval.py $S/v19b_plain_indist.json data/eval/indist_300.jsonl $C 0,1,2,3 notation \
  v19b_plain > $S/v19b_plain_indist.log 2>&1 < /dev/null &
wait
log "done: follow-up"
