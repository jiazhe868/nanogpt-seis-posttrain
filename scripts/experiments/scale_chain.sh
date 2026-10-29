#!/bin/bash
# Is the 113M model still data-limited? Phase IV SFT on 25% / 50% / 100% of the teacher solutions (all replay
# kept; same start, 4 epochs, same schedule), scored on the real held-out set (6 x 4), the 300 new problems on
# the fine-tuning papers (4 x 4) and the clean dev set (4 x 4).
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; mkdir -p $S
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/scale_progress.log; }
for tag in frac50 frac25; do
  log "SFT $tag"
  $TR --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_pot_notation2_$tag.yaml --snapshot-epochs \
    > $S/sft_$tag.log 2>&1
  grep -q "Training Complete" $S/sft_$tag.log || { log "FAILED: $tag"; exit 1; }
done
log "evaluation"
ev() {  # name checkpoint gpu
  export CUDA_VISIBLE_DEVICES=$3
  $PY scripts/experiments/real6_eval.py $S/scale_real6_$1.json $2 31,32,33,34,35,36 1.0 1 "$1" > $S/scale_real6_$1.log 2>&1
  $PY scripts/experiments/set_eval.py $S/scale_indist_$1.json data/eval/indist_300.jsonl $2 0,1,2,3 notation $1 > $S/scale_indist_$1.log 2>&1
  $PY scripts/experiments/num_eval_pot.py $S/scale_dev_$1.json "$1=$2:potnot" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/scale_dev_$1.log 2>&1
}
ev frac25 checkpoints/sft_pot_notation2_frac25/sft_ckpt_epoch4.pt 0 &
ev frac50 checkpoints/sft_pot_notation2_frac50/sft_ckpt_epoch4.pt 1 &
wait
ev full checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt 0
log "done"
