#!/bin/bash
# Phase IV round 3: build the mixture, SFT from the Phase II checkpoint (4 epochs, every epoch kept), choose the
# epoch on the clean dev set, then score the chosen one on the held-out set, the 300 new problems on the
# fine-tuning papers and the conceptual dev prompts.     TAGS="pilot pilot2 wA" bash scripts/experiments/notation3_chain.sh
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; mkdir -p $S
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}; TAGS=${TAGS:-"pilot pilot2 wA"}
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/notation3_progress.log; }
log "mixture from $TAGS"
$PY scripts/experiments/build_notation3.py data/splits/train_pot_notation3.jsonl $TAGS 2>&1 | tail -1 | tee -a $S/notation3_progress.log
log "SFT (2 GPUs)"
$TR --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_pot_notation3.yaml --snapshot-epochs > $S/sft_notation3.log 2>&1
grep -q "Training Complete" $S/sft_notation3.log || { log "FAILED: SFT"; exit 1; }
log "epoch selection on the clean dev set"
M0=""; M1=""
for e in 1 2 3 4; do
  m="n3-e$e=checkpoints/sft_pot_notation3/sft_ckpt_epoch$e.pt:potnot"
  if [ $((e % 2)) = 1 ]; then M0="${M0:+$M0,}$m"; else M1="${M1:+$M1,}$m"; fi
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/n3_dev_0.json "$M0" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/n3_dev_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/n3_dev_1.json "$M1" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/n3_dev_1.log 2>&1 &
wait
E=$($PY -c "
import json
r = {**json.load(open('$S/n3_dev_0.json')), **json.load(open('$S/n3_dev_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
for m in sorted(acc): print(m, round(100 * acc[m], 1), file=open('$S/notation3_progress.log', 'a'))
print(max(acc, key=lambda m: (round(acc[m], 4), -int(m[-1])))[-1])")
C=checkpoints/sft_pot_notation3/sft_ckpt_epoch$E.pt
log "selected epoch $E; held-out, new problems and conceptual dev"
( export CUDA_VISIBLE_DEVICES=0
  $PY scripts/experiments/real6_eval.py $S/n3_real6.json $C 31,32,33,34,35,36 1.0 1 "notation3 e$E" > $S/n3_real6.log 2>&1
  $PY scripts/experiments/set_eval.py $S/n3_indist.json data/eval/indist_300.jsonl $C 0,1,2,3 notation n3 > $S/n3_indist.log 2>&1 ) &
( export CUDA_VISIBLE_DEVICES=1
  WORK=$S $PY scripts/experiments/ceiling_check.py concept $C n3_devcon prose work/experiments/dev_clean_all.jsonl > $S/n3_concept.log 2>&1
  WORK=$S $PY scripts/experiments/ceiling_check.py concept checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt n2_devcon prose \
    work/experiments/dev_clean_all.jsonl > $S/n2_concept.log 2>&1 ) &
wait
log "done"
