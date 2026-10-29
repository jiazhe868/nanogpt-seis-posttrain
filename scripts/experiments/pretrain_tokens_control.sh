#!/bin/bash
# Does more pretraining help the fine-tuned model? One fine-tuning recipe (the round-3 mixture, single stage from the
# base, 4 epochs, epoch chosen on the clean dev set) applied to the scaling sweep's 2xl checkpoints (122M params,
# 82M / 246M / 738M pretraining tokens, muP, 1,024 context) and to our 113M base (3.15B tokens, 4,096 context).
# Peak LR is the same fraction (2/3) of each model's pretraining peak as the shipped SFT; tokens per step match.
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; mkdir -p $S configs/ctrl_tokens
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/ctrl_tokens_progress.log; }
SW=../nanogpt_seis/checkpoints/scaling

# name  base checkpoint  peak LR  min LR  seq len  micro-batch  [training file  data-order seed]
# (RUNS="..." in the environment evaluates other bases or mixtures; a second data seed repeats a run with new luck;
#  PICK=last evaluates epoch 4 instead of the dev set's choice)
RUNS=${RUNS:-"2xl_82M $SW/C6e16_2xl/ckpt.pt 8.0e-4 8.0e-5 1024 16
2xl_246M $SW/C1.8e17_2xl/ckpt.pt 8.0e-4 8.0e-5 1024 16
2xl_738M $SW/C5.4e17_2xl/ckpt.pt 8.0e-4 8.0e-5 1024 16
113M_3150M ../nanogpt_seis/checkpoints/ckpt.pt 2.0e-4 2.0e-5 4096 4"}

while read NAME CK LR MIN SEQ MB TRAIN DSEED; do
  [ -f checkpoints/ctrl_tokens/$NAME/DONE ] && { log "$NAME: already done"; continue; }
  cat > configs/ctrl_tokens/$NAME.yaml <<EOF
# Pretraining-token control ($NAME), scripts/experiments/pretrain_tokens_control.sh.
train:
  ckpt_path: "$CK"
  tokenizer_path: "../nanogpt_seis/data/tokenized/tokenizer.json"
  train_path: "${TRAIN:-data/splits/train_pot_notation3.jsonl}"
  val_path: "data/splits/val.jsonl"
  out_dir: "checkpoints/ctrl_tokens/$NAME"
  epochs: 4
  batch_size: $MB
  grad_accum: 4
  learning_rate: $LR
  min_lr: $MIN
  weight_decay: 0.01
  warmup_ratio: 0.05
  max_seq_len: $SEQ
  pack_sequences: true
  seed: 42
  data_seed: ${DSEED:-0}
EOF
  log "$NAME: SFT (2 GPUs, peak LR $LR, $SEQ-token windows)"
  $TR --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/ctrl_tokens/$NAME.yaml --snapshot-epochs \
    > $S/ctrl_sft_$NAME.log 2>&1 < /dev/null
  grep -q "Training Complete" $S/ctrl_sft_$NAME.log || { log "FAILED: $NAME SFT"; exit 1; }
  grep -E "Step [0-9]+/[0-9]+ " $S/ctrl_sft_$NAME.log | tail -1 | cut -c1-120 | tee -a $S/ctrl_tokens_progress.log

  O=checkpoints/ctrl_tokens/$NAME
  CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/ctrl_dev_${NAME}_0.json \
    "e1=$O/sft_ckpt_epoch1.pt:potnot,e3=$O/sft_ckpt_epoch3.pt:potnot" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" \
    > $S/ctrl_dev_${NAME}_0.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/ctrl_dev_${NAME}_1.json \
    "e2=$O/sft_ckpt_epoch2.pt:potnot,e4=$O/sft_ckpt_epoch4.pt:potnot" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" \
    > $S/ctrl_dev_${NAME}_1.log 2>&1 < /dev/null &
  wait
  E=$($PY -c "
import json
r = {**json.load(open('$S/ctrl_dev_${NAME}_0.json')), **json.load(open('$S/ctrl_dev_${NAME}_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
print('  dev ' + ', '.join(f'{m} {100 * acc[m]:.1f}' for m in sorted(acc)), file=open('$S/ctrl_tokens_progress.log', 'a'))
print(max(acc, key=lambda m: (round(acc[m], 4), -int(m[-1])))[-1])")
  [ "${PICK:-dev}" = last ] && E=4     # PICK=last: the dev set cannot tell epochs 2-4 apart (2026-10-26)
  C=$O/sft_ckpt_epoch$E.pt
  log "$NAME: epoch $E; real held-out (6 x 4) and in-distribution (300 x 4)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/real6_eval.py $S/ctrl_real6_$NAME.json $C 31,32,33,34,35,36 1.0 1 "$NAME e$E" \
    > $S/ctrl_real6_$NAME.log 2>&1 < /dev/null &
  CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/set_eval.py $S/ctrl_indist_$NAME.json data/eval/indist_300.jsonl $C 0,1,2,3 notation $NAME \
    > $S/ctrl_indist_$NAME.log 2>&1 < /dev/null &
  wait
  $PY - <<PYEOF | tee -a $S/ctrl_tokens_progress.log
import json, statistics as st
a = json.load(open('$S/ctrl_real6_$NAME.json'))['results']
b = json.load(open('$S/ctrl_indist_$NAME.json'))['results']
ra = next(iter(a.values())); rb = next(iter(b.values()))
print(f"  $NAME: held-out {100 * st.mean(x['numerical_acc_5pct'] for x in ra):.1f}, "
      f"in-distribution {100 * st.mean(x['numerical_acc_5pct'] for x in rb):.1f}")
PYEOF
  echo $E > $O/DONE
done <<< "$RUNS"
log "done"
