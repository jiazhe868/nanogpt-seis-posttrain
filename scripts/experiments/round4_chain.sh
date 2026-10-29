#!/bin/bash
# Round 4. Window B is split by paper: 70% joins the SFT mixture, 30% stays out of SFT as a fresh RL pool
# (v17 showed RL on problems already in SFT does not reach new questions). Phase IV SFT on the round-3 data +
# window B's SFT share; epoch chosen on the clean dev set; GRPO v18 on the fresh pool with and without SFT
# replay; clean-dev selection over the SFT model and every RL checkpoint; one real held-out evaluation.
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; D=data/sources/real_style; mkdir -p $S
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/round4_progress.log; }

log "split window B by paper (30% of papers -> fresh RL pool)"
$PY - <<PYEOF | tee -a $S/round4_progress.log
import hashlib, json
rl = lambda d: int(hashlib.md5(str(d).encode()).hexdigest(), 16) % 10 < 3
num = [json.loads(l) for l in open('$D/notation_wB.jsonl')]
con = [json.loads(l) for l in open('$D/con_wB.jsonl')]
for name, rows in (('notation', num), ('con', con)):
    with open(f'$D/{name}_wB_sft.jsonl', 'w') as f, open(f'$D/{name}_wB_rl.jsonl', 'w') as g:
        for r in rows: (g if rl(r.get('doc_id')) else f).write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  problems: {sum(not rl(r.get('doc_id')) for r in num)} SFT, {sum(rl(r.get('doc_id')) for r in num)} RL; "
      f"conceptual pairs: {sum(not rl(r.get('doc_id')) for r in con)} SFT")
PYEOF
log "mixture"
$PY scripts/experiments/build_notation3.py data/splits/train_pot_notation4.jsonl pilot pilot2 wA wB_sft 2>&1 | tail -1 | tee -a $S/round4_progress.log
sed -e 's|^# Phase IV round 3 (2026-10-23, audit §15.28).*|# Phase IV round 4 (README §5.20): the round-3 mixture plus 70% (by paper) of window B, whose other 30% is|' \
    -e 's|^# pairs written by the local teacher.*|# the fresh RL pool of GRPO v18 (scripts/experiments/round4_chain.sh).|' \
    -e 's|train_pot_notation3.jsonl|train_pot_notation4.jsonl|; s|checkpoints/sft_pot_notation3|checkpoints/sft_pot_notation4|' \
    configs/sft_pot_notation3.yaml > configs/sft_pot_notation4.yaml
log "SFT (2 GPUs)"
$TR --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_pot_notation4.yaml --snapshot-epochs > $S/sft_notation4.log 2>&1
grep -q "Training Complete" $S/sft_notation4.log || { log "FAILED: SFT"; exit 1; }

log "epoch selection on the clean dev set"
M0="n4-e1=checkpoints/sft_pot_notation4/sft_ckpt_epoch1.pt:potnot,n4-e3=checkpoints/sft_pot_notation4/sft_ckpt_epoch3.pt:potnot"
M1="n4-e2=checkpoints/sft_pot_notation4/sft_ckpt_epoch2.pt:potnot,n4-e4=checkpoints/sft_pot_notation4/sft_ckpt_epoch4.pt:potnot"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/n4_dev_0.json "$M0" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/n4_dev_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/n4_dev_1.json "$M1" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/n4_dev_1.log 2>&1 &
wait
E=$($PY -c "
import json
r = {**json.load(open('$S/n4_dev_0.json')), **json.load(open('$S/n4_dev_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
for m in sorted(acc): print(f'  dev {m}: {100 * acc[m]:.1f}', file=open('$S/round4_progress.log', 'a'))
print(max(acc, key=lambda m: (round(acc[m], 4), -int(m[-1])))[-1])")
BASE=checkpoints/sft_pot_notation4/sft_ckpt_epoch$E.pt
log "selected epoch $E; fresh RL pool"
$PY - <<PYEOF | tee -a $S/round4_progress.log
import json
rows = []
for k, r in enumerate(map(json.loads, open('$D/notation_wB_rl.jsonl'))):
    rows.append({"id": f"rsB_{k}", "prompt": r["question"], "type": "numerical", "final_numerical_value": float(r["answer"]),
                 "key_concepts": [], "program": r.get("program_python", ""), "chunk_id": r.get("chunk_id"), "source": "realstyle"})
for h in range(2):
    with open(f'data/rl/v18_cand_{h}.jsonl', 'w') as f:
        for r in rows[h::2]: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  {len(rows)} fresh candidates")
PYEOF
for h in 0 1; do
  CUDA_VISIBLE_DEVICES=$h $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/v18_cand_$h.jsonl \
    --out data/rl/v18_filtered_$h.jsonl --seed $((51 + h)) --batch-prompts 12 > $S/v18_filter_$h.log 2>&1 &
done
wait
$PY tools/build_rl_pool.py build --inp data/rl/v18_filtered_0.jsonl data/rl/v18_filtered_1.jsonl --out data/rl/train_v18_pool.jsonl 2>&1 | tail -1 | tee -a $S/round4_progress.log
N=$(wc -l < data/rl/train_v18_pool.jsonl); STEPS=$(( (N + 63) / 64 )); SAVE=$(( (STEPS + 5) / 6 ))
for arm in plain ptx; do
  sed -e "s|GRPO v17 ($arm), Phase V from the round-3 SFT model|GRPO v18 ($arm), Phase V from the round-4 SFT model on a fresh pool|; s|audit §15.28|README §5.22|" \
      -e "s|base_checkpoint: checkpoints/sft_pot_notation3/sft_ckpt_epoch4.pt|base_checkpoint: $BASE|" \
      -e "s|train_v17_pool|train_v18_pool|; s|train_pot_notation3.jsonl|train_pot_notation4.jsonl|" \
      -e "s|grpo_v17_$arm|grpo_v18_$arm|g; s|^  max_steps: .*|  max_steps: $STEPS|; s|^  save_every: .*|  save_every: $SAVE|" \
      configs/rl_grpo_v17_$arm.yaml > configs/rl_grpo_v18_$arm.yaml
  log "GRPO v18 $arm: $N fresh problems, $STEPS steps"
  mkdir -p checkpoints/grpo_v18_$arm
  $TR --standalone --nproc_per_node=2 --master_port 29718 -m src.rl.train_grpo --config configs/rl_grpo_v18_$arm.yaml \
    > checkpoints/grpo_v18_$arm/train.log 2>&1
  grep -q "training complete" checkpoints/grpo_v18_$arm/train.log || { log "FAILED: v18 $arm"; exit 1; }
done

log "clean-dev selection"
M0="n4=$BASE:potnot"; M1=""; i=0
for arm in plain ptx; do
  for s in $(ls checkpoints/grpo_v18_$arm/grpo_ckpt_step_*.pt | sed 's/.*step_\([0-9]*\)\.pt/\1/' | sort -n); do
    m="$arm-$s=checkpoints/grpo_v18_$arm/grpo_ckpt_step_$s.pt:potnot"
    if [ $((i % 2)) = 0 ]; then M1="${M1:+$M1,}$m"; else M0="$M0,$m"; fi; i=$((i+1))
  done
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/num_eval_pot.py $S/v18_dev_0.json "$M0" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/v18_dev_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/num_eval_pot.py $S/v18_dev_1.json "$M1" 31,32,33,34 "dev=data/rl/dev_clean.jsonl" > $S/v18_dev_1.log 2>&1 &
wait
B=$($PY - <<PYEOF
import json
r = {**json.load(open('$S/v18_dev_0.json')), **json.load(open('$S/v18_dev_1.json'))}
acc = {m: sum(v['dev'].values()) / len(v['dev']) for m, v in r.items()}
with open('$S/round4_progress.log', 'a') as f:
    for m in sorted(acc): f.write(f"  dev {m}: {100 * acc[m]:.1f}\n")
best = max(acc, key=lambda m: (round(acc[m], 4), m == 'n4'))
json.dump({'best': best, 'dev': acc}, open('$S/v18_best.json', 'w'))
print(best)
PYEOF
)
if [ "$B" = "n4" ]; then C=$BASE; else C=checkpoints/grpo_v18_${B%%-*}/grpo_ckpt_step_${B#*-}.pt; fi
log "selected $B ($C): real held-out (6 x 4) and the 300 new problems"
CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/real6_eval.py $S/r4_real6.json $C 31,32,33,34,35,36 1.0 1 "round 4 $B" > $S/r4_real6.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/set_eval.py $S/r4_indist.json data/eval/indist_300.jsonl $C 0,1,2,3 notation r4 > $S/r4_indist.log 2>&1 &
wait
log "done"
