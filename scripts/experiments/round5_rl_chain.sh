#!/bin/bash
# Round 5, Phase V: does GRPO on fresh problems add to the best fine-tune? The 30% of window C kept out of SFT
# (scripts/experiments/round5_chain.sh) is filtered to the problems the round-5 model (mid-trained base, data order 0,
# epoch 4) solves sometimes in 8 tries; GRPO v19 with and without SFT replay for three passes over that pool. The final
# checkpoints are scored by scripts/experiments/round5_rl_more.sh.
#   AFTER=<pid file of a GPU job to wait for> bash scripts/experiments/round5_rl_chain.sh
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; D=data/sources/real_style
PY=${PY:-python}; TR=${TORCHRUN:-torchrun}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/round5_rl_progress.log; }
NAME=113M_mid1p6B_n5; BASE=checkpoints/ctrl_tokens/$NAME/sft_ckpt_epoch4.pt

if [ -n "${AFTER:-}" ]; then
  while kill -0 $(cat $AFTER) 2>/dev/null; do sleep 60; done
fi
log "fresh RL pool: window C's RL share, filtered to problems the round-5 model solves sometimes"
$PY - <<PYEOF | tee -a $S/round5_rl_progress.log
import json
rows = []
for k, r in enumerate(map(json.loads, open('$D/notation_wC_rl.jsonl'))):
    rows.append({"id": f"rsC_{k}", "prompt": r["question"], "type": "numerical", "final_numerical_value": float(r["answer"]),
                 "key_concepts": [], "program": r.get("program_python", ""), "chunk_id": r.get("chunk_id"), "source": "realstyle"})
for h in range(2):
    with open(f'data/rl/v19_cand_{h}.jsonl', 'w') as f:
        for r in rows[h::2]: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  {len(rows)} fresh candidates")
PYEOF
for h in 0 1; do
  CUDA_VISIBLE_DEVICES=$h $PY tools/filter_learnable_prompts.py --ckpt $BASE --notation --inp data/rl/v19_cand_$h.jsonl \
    --out data/rl/v19_filtered_$h.jsonl --seed $((61 + h)) --batch-prompts 12 > $S/v19_filter_$h.log 2>&1 &
done
wait
$PY tools/build_rl_pool.py build --inp data/rl/v19_filtered_0.jsonl data/rl/v19_filtered_1.jsonl \
  --out data/rl/train_v19_pool.jsonl 2>&1 | tail -1 | tee -a $S/round5_rl_progress.log
N=$(wc -l < data/rl/train_v19_pool.jsonl); PASS=$(( (N + 63) / 64 )); STEPS=$((3 * PASS))
for arm in plain ptx; do
  sed -e "s|GRPO v18 ($arm), Phase V from the round-4 SFT model on a fresh pool, strict protocol (README §5.22)|GRPO v19 ($arm), Phase V from the round-5 model on a fresh pool, three passes (scripts/experiments/round5_rl_chain.sh)|" \
      -e "s|base_checkpoint: checkpoints/sft_pot_notation4/sft_ckpt_epoch3.pt|base_checkpoint: $BASE|" \
      -e "s|train_v18_pool|train_v19_pool|; s|train_pot_notation4.jsonl|train_pot_notation5.jsonl|" \
      -e "s|grpo_v18_$arm|grpo_v19_$arm|g; s|^  max_steps: .*|  max_steps: $STEPS|; s|^  save_every: .*|  save_every: $PASS|" \
      configs/rl_grpo_v18_$arm.yaml > configs/rl_grpo_v19_$arm.yaml
  log "GRPO v19 $arm: $N fresh problems, $STEPS steps (3 passes)"
  mkdir -p checkpoints/grpo_v19_$arm
  $TR --standalone --nproc_per_node=2 --master_port 29719 -m src.rl.train_grpo --config configs/rl_grpo_v19_$arm.yaml \
    > checkpoints/grpo_v19_$arm/train.log 2>&1 < /dev/null
  grep -q "training complete" checkpoints/grpo_v19_$arm/train.log || { log "FAILED: v19 $arm"; exit 1; }
done

log "done"
