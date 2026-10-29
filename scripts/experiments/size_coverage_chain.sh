#!/bin/bash
# Model size at matched pretraining tokens (README §5.21): the scaling sweep's smaller muP models get the fine-tuning
# recipe and evaluations of the pretraining-token control (scripts/experiments/pretrain_tokens_control.sh: the 153 real
# held-out questions, 6 seeds x 4, and the 300 new problems). Pairs: 44M vs 122M at ~0.7B tokens, 24M vs 75M at
# ~1.2B, 3M and 9M vs our 113M at ~3.2B.
#   AFTER=<pid file of a GPU job to wait for> bash scripts/experiments/size_coverage_chain.sh
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; SW=../nanogpt_seis/checkpoints/scaling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/size_coverage_progress.log; }

if [ -n "${AFTER:-}" ]; then
  while kill -0 $(cat $AFTER) 2>/dev/null; do sleep 60; done
fi
log "model size: fine-tune 5 sweep models (log: $S/ctrl_tokens_progress.log)"
RUNS="l_678M $SW/C1.8e17_l/ckpt.pt 8.0e-4 8.0e-5 1024 16
xl_1190M $SW/C5.4e17_xl/ckpt.pt 8.0e-4 8.0e-5 1024 16
m_1240M $SW/C1.8e17_m/ckpt.pt 8.0e-4 8.0e-5 1024 16
s_3180M $SW/C1.8e17_s/ckpt.pt 8.0e-4 8.0e-5 1024 16
xs_3180M $SW/C6e16_xs/ckpt.pt 8.0e-4 8.0e-5 1024 16" bash scripts/experiments/pretrain_tokens_control.sh \
  || { log "FAILED: size runs"; exit 1; }
log "done"
