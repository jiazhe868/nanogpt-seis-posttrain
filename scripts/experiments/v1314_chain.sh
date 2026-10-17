#!/bin/bash
# GRPO v13 (strict protocol) and v14 (corpus protocol): broad unseen pools + stepping stones, one pass each, dev-selected.
S=${WORK:-work/experiments}; mkdir -p $S
cd "$(dirname "$0")/../.."
VP=${VLLM_PY:-python}          # interpreter with vLLM, for the 27B teacher
PY=${PY:-python}
TR=${TORCHRUN:-torchrun}
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
step() { echo "[$(date '+%m-%d %H:%M')] $1" >> $S/v1314_progress.log; }
BASE=checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt
filter2() {  # $1 input, $2 output (rows + passes), $3 tag
  $PY -c "
import json; R=[l for l in open('$1')]
for p in (0, 1): open('$S/$3_in_%d.jsonl' % p, 'w').writelines(R[p::2])"
  ( CUDA_VISIBLE_DEVICES=0 $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp $S/$3_in_0.jsonl --out $S/$3_out_0.jsonl --seed 11 --batch-prompts 12 > $S/$3_0.log 2>&1 ) &
  ( CUDA_VISIBLE_DEVICES=1 $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp $S/$3_in_1.jsonl --out $S/$3_out_1.jsonl --seed 12 --batch-prompts 12 > $S/$3_1.log 2>&1 ) &
  wait
  cat $S/$3_out_0.jsonl $S/$3_out_1.jsonl > $2
  echo "  $3: $(tail -1 $S/$3_0.log) || $(tail -1 $S/$3_1.log)" >> $S/v1314_progress.log
}
grpo() {  # $1 config, $2 output dir, $3 train file, $4 port
  N=$(wc -l < $3); STEPS=$(( (N + 63) / 64 ))
  sed -i "s/^  max_steps: .*/  max_steps: $STEPS/" $1
  step "GRPO $(basename $2): $N prompts, $STEPS steps (one pass), 2 GPUs"
  mkdir -p $2
  $TR --standalone --nproc_per_node=2 --master_port $4 -m src.rl.train_grpo --config $1 > $2/train.log 2>&1
  grep -q "training complete" $2/train.log || step "WARNING: $(basename $2) did not complete"
}
[ -s data/rl/v13_filtered_existing.jsonl ] || { step "FAILED: missing data/rl/v13_filtered_existing.jsonl"; exit 1; }
step "1 teacher: strict pool (rewrite round 3 + exercises from all training topics) and corpus pool (exercises from held-out papers)"
$VP tools/gen_pot_problems.py rewrite --n-seeds 0 --per-seed 3 --seed 3 --out data/sources/pot/rewrite3_27b.jsonl > $S/v13_rewrite3.log 2>&1
$VP tools/gen_pot_problems.py generate --seed-pool all --n-seeds 3000 --per-seed 3 --seed 5 --out data/sources/pot/gen3_27b.jsonl > $S/v13_gen3.log 2>&1
$VP tools/gen_chunk_problems.py --chunks $S/chunks_heldout.jsonl --ids data/rl/test.jsonl data/rl/val.jsonl --per-chunk 8 \
    --out data/sources/pot/chunk_heldout_27b.jsonl > $S/v14_chunks.log 2>&1
grep "excerpts ->" $S/v14_chunks.log >> $S/v1314_progress.log
$PY tools/build_rl_pool.py teacher --inp data/sources/pot/rewrite3_27b.jsonl data/sources/pot/gen3_27b.jsonl \
    --also-dedupe data/rl/v13_cand_existing.jsonl --prefix v13t_ --out data/rl/v13_cand_teacher.jsonl >> $S/v1314_progress.log 2>&1
$PY tools/build_rl_pool.py teacher --inp data/sources/pot/chunk_heldout_27b.jsonl --exclude-chunks data/rl/transfer_B.jsonl \
    --also-dedupe data/rl/v13_cand_existing.jsonl data/rl/v13_cand_teacher.jsonl --prefix v14c_ --out data/rl/v14_cand_corpus.jsonl >> $S/v1314_progress.log 2>&1
cat data/rl/v13_cand_teacher.jsonl data/rl/v14_cand_corpus.jsonl > $S/cand_new.jsonl
[ -s $S/cand_new.jsonl ] || { step "FAILED: no new candidates"; exit 1; }
filter2 $S/cand_new.jsonl $S/filtered_new.jsonl fnew
grep '"id": "v13t_' $S/filtered_new.jsonl > data/rl/v13_filtered_teacher.jsonl
grep '"id": "v14c_' $S/filtered_new.jsonl > data/rl/v14_filtered_corpus.jsonl
step "2 stepping stones for never-solved problems (strict and corpus)"
$PY tools/build_rl_pool.py never --inp data/rl/v13_filtered_teacher.jsonl --limit 6000 --out $S/never_strict.jsonl >> $S/v1314_progress.log 2>&1
$PY tools/build_rl_pool.py never --inp data/rl/v14_filtered_corpus.jsonl --out $S/never_corpus.jsonl >> $S/v1314_progress.log 2>&1
cat $S/never_strict.jsonl $S/never_corpus.jsonl > $S/never_all.jsonl
$VP tools/gen_stepping_stones.py --inp $S/never_all.jsonl --out $S/stones_all.jsonl --prefix st_ > $S/v1314_stones.log 2>&1
grep "problems ->" $S/v1314_stones.log >> $S/v1314_progress.log
filter2 $S/stones_all.jsonl $S/filtered_stones.jsonl fstone
grep '"orig_id": "v13t_' $S/filtered_stones.jsonl > data/rl/v13_filtered_stones.jsonl
grep '"orig_id": "v14c_' $S/filtered_stones.jsonl > data/rl/v14_filtered_stones.jsonl
step "3 pools"
$PY tools/build_rl_pool.py build --inp data/rl/v13_filtered_existing.jsonl data/rl/v13_filtered_teacher.jsonl data/rl/v13_filtered_stones.jsonl \
    data/rl/transfer_A_learnable.jsonl data/rl/simple_A_learnable.jsonl --out data/rl/train_v13_broad.jsonl >> $S/v1314_progress.log 2>&1
$PY tools/build_rl_pool.py build --inp data/rl/v13_filtered_existing.jsonl data/rl/v13_filtered_teacher.jsonl data/rl/v13_filtered_stones.jsonl \
    data/rl/transfer_A_learnable.jsonl data/rl/simple_A_learnable.jsonl data/rl/v14_filtered_corpus.jsonl data/rl/v14_filtered_stones.jsonl \
    --out data/rl/train_v14_corpus.jsonl >> $S/v1314_progress.log 2>&1
cat data/rl/val_numerical_dev.jsonl data/rl/transfer_B600.jsonl > data/rl/v13_dev.jsonl
step "4 GRPO"
grpo configs/rl_grpo_v13_broad.yaml checkpoints/grpo_v13_broad data/rl/train_v13_broad.jsonl 29705
grpo configs/rl_grpo_v14_corpus.yaml checkpoints/grpo_v14_corpus data/rl/train_v14_corpus.jsonl 29706
step "5 dev selection (deployment decoding, 2 seeds x 4)"
M0="notation2=$BASE:potnot"; M1=""; i=0
for G in checkpoints/grpo_v13_broad checkpoints/grpo_v14_corpus; do
  tag=$(basename $G | cut -d_ -f2)
  for s in $(ls $G/grpo_ckpt_step_*.pt 2>/dev/null | sed 's/.*step_\([0-9]*\)\.pt/\1/' | sort -n); do
    if [ $((i % 2)) = 0 ]; then M0="$M0,$tag-$s=$G/grpo_ckpt_step_$s.pt:potnot"; else M1="${M1:+$M1,}$tag-$s=$G/grpo_ckpt_step_$s.pt:potnot"; fi; i=$((i+1))
  done
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/v1314_dev_0.json "$M0" 31,32 "dev=data/rl/v13_dev.jsonl" > $S/v1314_dev_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/v1314_dev_1.json "$M1" 31,32 "dev=data/rl/v13_dev.jsonl" > $S/v1314_dev_1.log 2>&1 &
wait
$PY - >> $S/v1314_progress.log <<PYEOF
import json
r = {**json.load(open('$S/v1314_dev_0.json')), **json.load(open('$S/v1314_dev_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
for m in sorted(acc, key=lambda m: (m.split('-')[0], int(m.split('-')[1]) if '-' in m else 0)): print(f'  dev {m}: {acc[m]:.4f}')
best = {}
for run in ('v13', 'v14'):
    cands = {m: a for m, a in acc.items() if m.startswith(run + '-')}
    cands['notation2'] = acc['notation2']
    best[run] = max(cands, key=lambda m: (round(cands[m], 4), -int(m.split('-')[1]) if '-' in m else 0))
json.dump(best, open('$S/v1314_best.json', 'w')); print('  selected:', best)
PYEOF
step "6 real held-out, 6 seeds x 4, for each selected checkpoint"
for run in v13 v14; do
  B=$($PY -c "import json; print(json.load(open('$S/v1314_best.json'))['$run'])")
  if [ "$B" = "notation2" ]; then C=$BASE; else d=checkpoints/grpo_$([ $run = v13 ] && echo v13_broad || echo v14_corpus); C=$d/grpo_ckpt_step_${B#*-}.pt; fi
  echo "$run $B $C" >> $S/v1314_progress.log
  [ $run = v13 ] && GPU=0 || GPU=1
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/experiments/real6_eval.py $S/real6_$run.json $C 31,32,33,34,35,36 1.0 1 "$run selected ($B)" > $S/real6_$run.log 2>&1 &
done
wait
step "done"
