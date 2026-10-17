#!/bin/bash
S=${WORK:-work/experiments}; mkdir -p $S
cd "$(dirname "$0")/../.."
PY=${PY:-python}
TR=${TORCHRUN:-torchrun}
export CUDA_VISIBLE_DEVICES=1
BASE=checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt
$PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/transfer_Bother.jsonl --out $S/learnBother.jsonl --seed 3 --batch-prompts 12 > $S/learnBother.log 2>&1
$PY - >> $S/learnBother.log 2>&1 <<PYEOF
import json
R = [json.loads(l) for l in open('$S/learnBother.jsonl')]
keep = [r for r in R if 0 < r['passes'] < r['k']]
open('data/rl/transfer_Bother_learnable.jsonl', 'w').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in keep)
print('B-other learnable', len(keep), 'of', len(R))
PYEOF
mkdir -p checkpoints/grpo_transfer_Bsame
$TR --standalone --nproc_per_node=1 --master_port 29703 -m src.rl.train_grpo --config configs/rl_grpo_transfer_Bsame.yaml > checkpoints/grpo_transfer_Bsame/train.log 2>&1
echo "BSAME_DONE $(grep -c 'training complete' checkpoints/grpo_transfer_Bsame/train.log)" > $S/bsame_done.txt
