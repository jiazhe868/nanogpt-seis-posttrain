#!/bin/bash
# curriculum check: does GRPO on the SIMPLE versions of A lift the ORIGINAL (harder) A problems?
S=${WORK:-work/experiments}; mkdir -p $S
cd "$(dirname "$0")/../.."
PY=${PY:-python}
GS=checkpoints/grpo_simpleA
SETS="A-learnable=data/rl/transfer_A_learnable.jsonl,A0-shared=data/rl/transfer_A_zero_shared.jsonl"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/simple_followup_0.json "simpleA-75=$GS/grpo_ckpt_step_75.pt:potnot" 31,32 "$SETS" > $S/simple_followup_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/simple_followup_1.json "simpleA-150=$GS/grpo_ckpt_step_150.pt:potnot" 31,32 "$SETS" > $S/simple_followup_1.log 2>&1 &
wait
echo FOLLOWUP_DONE
