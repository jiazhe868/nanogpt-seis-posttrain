#!/bin/bash
S=${WORK:-work/experiments}; mkdir -p $S
cd "$(dirname "$0")/../.."
VP=${VLLM_PY:-python}          # interpreter with vLLM, for the 27B teacher
PY=${PY:-python}
TR=${TORCHRUN:-torchrun}
step() { echo "[$(date '+%m-%d %H:%M')] $1" >> $S/pilot_progress.log; }
step "0 same-chunk GRPO checkpoints on B600 and real (deployment decoding)"
GB=checkpoints/grpo_transfer_Bsame
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/bsame_eval_0.json "bsame-50=$GB/grpo_ckpt_step_50.pt:potnot,bsame-75=$GB/grpo_ckpt_step_75.pt:potnot" 31,32 "B600=data/rl/transfer_B600.jsonl,real=data/rl/test.jsonl" > $S/bsame_eval_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/bsame_eval_1.json "bsame-100=$GB/grpo_ckpt_step_100.pt:potnot,bsame-150=$GB/grpo_ckpt_step_150.pt:potnot" 31,32 "B600=data/rl/transfer_B600.jsonl,real=data/rl/test.jsonl" > $S/bsame_eval_1.log 2>&1 &
wait
step "1 teacher restatements of A and B chunks"
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} $VP scripts/experiments/gen_notes.py --chunks $S/chunks_AB_real.jsonl \
    --ids data/rl/transfer_A.jsonl data/rl/transfer_B600.jsonl --out $S/notes_AB.jsonl > $S/gen_notes.log 2>&1
[ -s $S/notes_AB.jsonl ] || { step "FAILED: no notes"; exit 1; }
$PY scripts/experiments/notes_sft.py $S/notes_AB.jsonl >> $S/pilot_progress.log 2>&1
step "2 continued SFT: K (notes on B chunks) and C (notes on A chunks)"
for arm in K_notesB C_notesA; do
  mkdir -p checkpoints/pilot_$arm
  $TR --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/pilot_$arm.yaml --snapshot-epochs > checkpoints/pilot_$arm/train.log 2>&1
  grep -q "Training Complete" checkpoints/pilot_$arm/train.log || { step "FAILED: SFT $arm"; exit 1; }
done
step "3 evaluation"
BASE=checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt
M="notation2=$BASE:potnot,K=checkpoints/pilot_K_notesB/sft_ckpt_epoch3.pt:potnot,C=checkpoints/pilot_C_notesA/sft_ckpt_epoch3.pt:potnot"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/pilot_eval_b.json "$M" 31,32 "B-rec=data/rl/pilot_Brec.jsonl,real=data/rl/test.jsonl" > $S/pilot_eval_b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/pilot_eval_a.json "$M" 31,32 "A-rec=data/rl/pilot_Arec400.jsonl" > $S/pilot_eval_a.log 2>&1 &
wait
step "done"
