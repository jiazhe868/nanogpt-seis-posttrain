#!/bin/bash
S=${WORK:-work/experiments}; mkdir -p $S
cd "$(dirname "$0")/../.."
VP=${VLLM_PY:-python}          # interpreter with vLLM, for the 27B teacher
PY=${PY:-python}
TR=${TORCHRUN:-torchrun}
step() { echo "[$(date '+%m-%d %H:%M')] $1" >> $S/simple_progress.log; }
until grep -q -E "done|FAILED" $S/pilot_progress.log 2>/dev/null; do sleep 30; done
step "1 teacher: simple versions of A and B600"
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} $VP tools/gen_stepping_stones.py --inp data/rl/transfer_A.jsonl data/rl/transfer_B600.jsonl \
    --out $S/simple_AB.jsonl > $S/gen_simple.log 2>&1
[ -s $S/simple_AB.jsonl ] || { step "FAILED: no simple questions"; exit 1; }
$PY - >> $S/simple_progress.log 2>&1 <<PYEOF
import json
R = [json.loads(l) for l in open('$S/simple_AB.jsonl')]
A = [r for r in R if str(r['orig_id']).startswith('trA')]; B = [r for r in R if str(r['orig_id']).startswith('trB')]
for name, rows in (('simple_A', A), ('simple_B600', B)):
    open(f'data/rl/{name}.jsonl', 'w').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
for p in (0, 1): open(f'data/rl/simple_A_{p}.jsonl', 'w').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in A[p::2])
print('simple A', len(A), '| simple B600', len(B))
PYEOF
step "2 learnability filter on simple A"
BASE=checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt
( CUDA_VISIBLE_DEVICES=0 $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/simple_A_0.jsonl --out $S/learnSA_0.jsonl --seed 1 --batch-prompts 12 > $S/learnSA_0.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/simple_A_1.jsonl --out $S/learnSA_1.jsonl --seed 2 --batch-prompts 12 > $S/learnSA_1.log 2>&1 ) &
wait
$PY - >> $S/simple_progress.log 2>&1 <<PYEOF
import json, collections
R = [json.loads(l) for f in ('$S/learnSA_0.jsonl', '$S/learnSA_1.jsonl') for l in open(f)]
keep = [r for r in R if 0 < r['passes'] < r['k']]
open('data/rl/simple_A_learnable.jsonl', 'w').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in keep)
h = collections.Counter(r['passes'] for r in R)
print('simple A pass histogram', sorted(h.items()), '| learnable', len(keep), 'of', len(R), '| mean pass', round(sum(r['passes'] for r in R) / (8 * len(R)), 3))
PYEOF
step "3 GRPO on simple A (GPU0) | baselines on simple sets (GPU1)"
mkdir -p checkpoints/grpo_simpleA
( CUDA_VISIBLE_DEVICES=0 $TR --standalone --nproc_per_node=1 --master_port 29704 -m src.rl.train_grpo --config configs/rl_grpo_simpleA.yaml > checkpoints/grpo_simpleA/train.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/simple_eval_base.json "notation2=$BASE:potnot,grpo-lo-75=checkpoints/grpo_transfer_lo/grpo_ckpt_step_75.pt:potnot,bsame-150=checkpoints/grpo_transfer_Bsame/grpo_ckpt_step_150.pt:potnot" 31,32 "simple-B600=data/rl/simple_B600.jsonl,simple-A-learnable=data/rl/simple_A_learnable.jsonl" > $S/simple_eval_base.log 2>&1 ) &
wait
step "4 evaluate GRPO on simple A"
GS=checkpoints/grpo_simpleA
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/simple_eval_0.json "simpleA-75=$GS/grpo_ckpt_step_75.pt:potnot" 31,32 "simple-B600=data/rl/simple_B600.jsonl,simple-A-learnable=data/rl/simple_A_learnable.jsonl,B600=data/rl/transfer_B600.jsonl,real=data/rl/test.jsonl" > $S/simple_eval_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/simple_eval_1.json "simpleA-150=$GS/grpo_ckpt_step_150.pt:potnot" 31,32 "simple-B600=data/rl/simple_B600.jsonl,simple-A-learnable=data/rl/simple_A_learnable.jsonl,B600=data/rl/transfer_B600.jsonl,real=data/rl/test.jsonl" > $S/simple_eval_1.log 2>&1 &
wait
step "done"
