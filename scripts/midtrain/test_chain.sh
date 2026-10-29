#!/bin/bash
# Continued-pretraining test: 1.57B tokens of the math/STEM mixture on the 113M base (configs/midtrain/test1p6b.yaml),
# then the fine-tuning recipe and evaluations of the pretraining-token control, so the result sits next to
# 113M_3150M (held-out 21.2, in-distribution 40.8).
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; mkdir -p $S
TR=${TORCHRUN:-torchrun}
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/ctrl_tokens_progress.log; }

if [ ! -f checkpoints/midtrain/test1p6b/ckpt.pt ]; then
  R=""; [ -f checkpoints/midtrain/test1p6b/ckpt_last.pt ] && R="--resume"
  log "113M_mid1p6B: continued pretraining, 4,000 iterations of the math/STEM mixture $R"
  $TR --standalone --nproc_per_node=2 scripts/midtrain/train_mid.py --config configs/midtrain/test1p6b.yaml $R \
    >> work/midtrain/test1p6b.log 2>&1 < /dev/null
  [ -f checkpoints/midtrain/test1p6b/ckpt.pt ] || { log "FAILED: continued pretraining"; exit 1; }
  grep "^\[eval\]" work/midtrain/test1p6b.log | tail -1 | tee -a $S/ctrl_tokens_progress.log
fi
RUNS="113M_mid1p6B checkpoints/midtrain/test1p6b/ckpt.pt 2.0e-4 2.0e-5 4096 4" \
  bash scripts/experiments/pretrain_tokens_control.sh
