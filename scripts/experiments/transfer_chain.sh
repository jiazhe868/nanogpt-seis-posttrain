#!/bin/bash
# Transfer diagnostic: GRPO vs RFT vs teacher-SFT trained on set A (unseen templates), tested on seed-disjoint set B and real held-out.
S=${WORK:-work/experiments}; mkdir -p $S
cd "$(dirname "$0")/../.."
VP=${VLLM_PY:-python}          # interpreter with vLLM, for the 27B teacher
PY=${PY:-python}
TR=${TORCHRUN:-torchrun}
step() { echo "[$(date '+%m-%d %H:%M')] $1" >> $S/transfer_progress.log; }
BASE=checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt
step "1 learnability filter on A"
$PY -c "
import json
R=[json.loads(l) for l in open('data/rl/transfer_A.jsonl')]
for p in (0,1):
    open(f'data/rl/transfer_A_{p}.jsonl','w').writelines(json.dumps(r, ensure_ascii=False)+'\n' for r in R[p::2])"
( CUDA_VISIBLE_DEVICES=0 $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/transfer_A_0.jsonl --out $S/learnA_0.jsonl --seed 1 --batch-prompts 12 > $S/learnA_0.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/transfer_A_1.jsonl --out $S/learnA_1.jsonl --seed 2 --batch-prompts 12 > $S/learnA_1.log 2>&1 ) &
wait
$PY - > $S/transfer_A.log 2>&1 <<PYEOF
import json, collections
R = [json.loads(l) for f in ('$S/learnA_0.jsonl', '$S/learnA_1.jsonl') for l in open(f)]
keep = [r for r in R if 0 < r['passes'] < r['k']]
open('data/rl/transfer_A_learnable.jsonl', 'w').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in keep)
h = collections.Counter(r['passes'] for r in R)
print('A pass histogram', sorted(h.items()), '| learnable', len(keep), 'of', len(R), '| mean pass', sum(r['passes'] for r in R) / (8 * len(R)))
PYEOF
step "2 GRPO on A: lr 5e-6 (GPU0) and 1.5e-5 (GPU1)"
mkdir -p checkpoints/grpo_transfer_lo checkpoints/grpo_transfer_hi
( CUDA_VISIBLE_DEVICES=0 $TR --standalone --nproc_per_node=1 --master_port 29701 -m src.rl.train_grpo --config configs/rl_grpo_transfer_lo.yaml > checkpoints/grpo_transfer_lo/train.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $TR --standalone --nproc_per_node=1 --master_port 29702 -m src.rl.train_grpo --config configs/rl_grpo_transfer_hi.yaml > checkpoints/grpo_transfer_hi/train.log 2>&1 ) &
wait
step "2 done (lo: $(grep -c 'training complete' checkpoints/grpo_transfer_lo/train.log), hi: $(grep -c 'training complete' checkpoints/grpo_transfer_hi/train.log))"
step "3 teacher notation solutions for A"
$PY -c "
import json
R=[json.loads(l) for l in open('data/rl/transfer_A.jsonl')]
open('data/rl/transfer_A_notin.jsonl','w').writelines(json.dumps({'question':r['prompt'],'program':r['program'],'answer':r['final_numerical_value'],'unit':r.get('unit',''),'chunk_id':r['chunk_id'],'group':'A'}, ensure_ascii=False)+'\n' for r in R)"
$VP tools/gen_pot_problems.py notation --model Qwen/Qwen3.8-27B-FP8 --tp 2 --temperature 0.2 --max-tokens 1200 \
    --inp data/rl/transfer_A_notin.jsonl --out data/rl/transfer_A_notation.jsonl > $S/transfer_teacher.log 2>&1
$PY tools/format_pot_sft.py --notation --out data/splits/transfer_sftA.jsonl data/rl/transfer_A_notation.jsonl >> $S/transfer_A.log 2>&1
step "4 RFT sampling on A (16 per problem, keep up to 2 correct)"
$PY -c "
import json
R=[json.loads(l) for l in open('data/rl/transfer_A.jsonl')]
for p in (0,1): open(f'data/rl/transfer_A_{p}.jsonl','w').writelines(json.dumps(r, ensure_ascii=False)+'\n' for r in R[p::2])"
( CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/rft_sample.py $BASE data/rl/transfer_A_0.jsonl $S/rftA_0.jsonl 16 0.8 2 > $S/rftA_0.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/rft_sample.py $BASE data/rl/transfer_A_1.jsonl $S/rftA_1.jsonl 16 0.8 2 > $S/rftA_1.log 2>&1 ) &
wait
cat $S/rftA_0.jsonl $S/rftA_1.jsonl > data/splits/transfer_rftA.jsonl
echo "RFT: $(tail -1 $S/rftA_0.log) || $(tail -1 $S/rftA_1.log)" >> $S/transfer_A.log
echo "teacher SFT pairs: $(wc -l < data/splits/transfer_sftA.jsonl) | RFT pairs: $(wc -l < data/splits/transfer_rftA.jsonl)" >> $S/transfer_A.log
step "5 SFT on RFT data and on teacher data"
mkdir -p checkpoints/transfer_rftA checkpoints/transfer_sftA
$TR --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/transfer_rftA.yaml --snapshot-epochs > checkpoints/transfer_rftA/train.log 2>&1
$TR --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/transfer_sftA.yaml --snapshot-epochs > checkpoints/transfer_sftA/train.log 2>&1
step "6 evaluation on B (seed-disjoint) and real held-out"
SETS="B600=data/rl/transfer_B600.jsonl,A-learnable=data/rl/transfer_A_learnable.jsonl,real=data/rl/test.jsonl"
GL=checkpoints/grpo_transfer_lo; GH=checkpoints/grpo_transfer_hi
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/transfer_eval_a.json "notation2=$BASE:potnot,grpo-lo-75=$GL/grpo_ckpt_step_75.pt:potnot,grpo-lo-150=$GL/grpo_ckpt_step_150.pt:potnot,rftA=checkpoints/transfer_rftA/sft_ckpt_epoch2.pt:potnot" 31,32 "$SETS" > $S/transfer_eval_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/transfer_eval_b.json "notation2=$BASE:potnot,grpo-hi-75=$GH/grpo_ckpt_step_75.pt:potnot,grpo-hi-150=$GH/grpo_ckpt_step_150.pt:potnot,sftA=checkpoints/transfer_sftA/sft_ckpt_epoch2.pt:potnot" 31,32 "$SETS" > $S/transfer_eval_b.log 2>&1 &
wait
step "done"
