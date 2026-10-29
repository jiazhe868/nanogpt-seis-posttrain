#!/bin/bash
# Round 6. Window D of real-style data (excerpts 30,700..40,700, scripts/experiments/real_style_chain.sh) is split by
# paper like windows B and C: 70% joins the SFT mixture, 30% stays out. Single-stage fine-tunes of the mid-trained base
# in two data orders (the control recipe of scripts/experiments/pretrain_tokens_control.sh, last epoch), then plain GRPO
# for three passes on the held-back problems of windows C and D, which no fine-tune has seen. The fine-tunes are scored
# by pretrain_tokens_control.sh, the GRPO checkpoints by scripts/experiments/round6_rl_more.sh.
#   AFTER=<pid file of the window D job> bash scripts/experiments/round6_chain.sh
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; D=data/sources/real_style; C=checkpoints/ctrl_tokens
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/round6_progress.log; }

if [ -n "${AFTER:-}" ]; then
  while kill -0 $(cat $AFTER) 2>/dev/null; do sleep 60; done
fi
grep -q "wD: done" $S/real_style_progress.log && [ -s $D/notation_wD.jsonl ] || { log "FAILED: window D did not finish"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1

log "split window D by paper (30% of papers -> fresh RL pool)"
$PY - <<PYEOF | tee -a $S/round6_progress.log
import hashlib, json
rl = lambda d: int(hashlib.md5(str(d).encode()).hexdigest(), 16) % 10 < 3
num = [json.loads(l) for l in open('$D/notation_wD.jsonl')]
con = [json.loads(l) for l in open('$D/con_wD.jsonl')]
for name, rows in (('notation', num), ('con', con)):
    with open(f'$D/{name}_wD_sft.jsonl', 'w') as f, open(f'$D/{name}_wD_rl.jsonl', 'w') as g:
        for r in rows: (g if rl(r.get('doc_id')) else f).write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  problems: {sum(not rl(r.get('doc_id')) for r in num)} SFT, {sum(rl(r.get('doc_id')) for r in num)} RL; "
      f"conceptual pairs: {sum(not rl(r.get('doc_id')) for r in con)} SFT")
PYEOF
log "mixture"
$PY scripts/experiments/build_notation3.py data/splits/train_pot_notation6.jsonl pilot pilot2 wA wB wC_sft wD_sft 2>&1 | tail -1 \
  | tee -a $S/round6_progress.log

log "fine-tune the mid-trained base in two data orders (log: $S/ctrl_tokens_progress.log)"
M=checkpoints/midtrain/test1p6b/ckpt.pt; N6=data/splits/train_pot_notation6.jsonl
PICK=last RUNS="113M_mid1p6B_n6 $M 2.0e-4 2.0e-5 4096 4 $N6 0
113M_mid1p6B_n6_d1 $M 2.0e-4 2.0e-5 4096 4 $N6 1" bash scripts/experiments/pretrain_tokens_control.sh \
  || { log "FAILED: fine-tuning"; exit 1; }
NAME=113M_mid1p6B_n6; BASE=$C/$NAME/sft_ckpt_epoch4.pt
log "fresh RL pool: the held-back problems of windows C and D, filtered to problems the round-6 model solves sometimes"
$PY - <<PYEOF | tee -a $S/round6_progress.log
import json
rows = []
for w in ("C", "D"):
    for k, r in enumerate(map(json.loads, open(f'$D/notation_w{w}_rl.jsonl'))):
        rows.append({"id": f"rs{w}_{k}", "prompt": r["question"], "type": "numerical", "final_numerical_value": float(r["answer"]),
                     "key_concepts": [], "program": r.get("program_python", ""), "chunk_id": r.get("chunk_id"), "source": "realstyle"})
for h in range(2):
    with open(f'data/rl/v20_cand_{h}.jsonl', 'w') as f:
        for r in rows[h::2]: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  {len(rows)} fresh candidates")
PYEOF
for h in 0 1; do
  CUDA_VISIBLE_DEVICES=$h $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/v20_cand_$h.jsonl \
    --out data/rl/v20_filtered_$h.jsonl --seed $((71 + h)) --batch-prompts 12 > $S/v20_filter_$h.log 2>&1 &
done
wait
$PY tools/build_rl_pool.py build --inp data/rl/v20_filtered_0.jsonl data/rl/v20_filtered_1.jsonl \
  --out data/rl/train_v20_pool.jsonl 2>&1 | tail -1 | tee -a $S/round6_progress.log
N=$(wc -l < data/rl/train_v20_pool.jsonl); PASS=$(( (N + 63) / 64 )); STEPS=$((3 * PASS))
sed -e "s|GRPO v19 (plain), Phase V from the round-5 model on a fresh pool, three passes (scripts/experiments/round5_rl_chain.sh)|GRPO v20 (plain), Phase V from the round-6 model on the held-back problems of windows C and D, three passes (scripts/experiments/round6_chain.sh)|" \
    -e "s|base_checkpoint: .*|base_checkpoint: $BASE|; s|train_v19_pool|train_v20_pool|; s|train_pot_notation5.jsonl|train_pot_notation6.jsonl|" \
    -e "s|grpo_v19_plain|grpo_v20_plain|g; s|^  max_steps: .*|  max_steps: $STEPS|; s|^  save_every: .*|  save_every: $PASS|" \
    configs/rl_grpo_v19_plain.yaml > configs/rl_grpo_v20_plain.yaml
log "GRPO v20 plain: $N fresh problems, $STEPS steps (3 passes)"
mkdir -p checkpoints/grpo_v20_plain
$TR --standalone --nproc_per_node=2 --master_port 29721 -m src.rl.train_grpo --config configs/rl_grpo_v20_plain.yaml \
  > checkpoints/grpo_v20_plain/train.log 2>&1 < /dev/null
grep -q "training complete" checkpoints/grpo_v20_plain/train.log || { log "FAILED: v20 plain"; exit 1; }

log "done"
