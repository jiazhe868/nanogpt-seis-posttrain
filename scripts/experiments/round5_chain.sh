#!/bin/bash
# Round 5. Window C of real-style data (excerpts 20,700..30,700, scripts/experiments/real_style_chain.sh) is split by
# paper like window B: 70% joins the SFT mixture, 30% stays out as a fresh RL pool. The mixture also takes window B's
# former RL share (GRPO v18 on it added only +0.5). Single-stage fine-tunes of the original and the mid-trained base
# in two data orders (the control recipe of scripts/experiments/pretrain_tokens_control.sh, last epoch), each scored there
# on the 153 real held-out questions and the 300 new problems.
#   AFTER=<pid file of the window C job> bash scripts/experiments/round5_chain.sh
set -u
cd "$(dirname "$0")/../.."
S=${WORK:-work/experiments}; D=data/sources/real_style
PY=${PY:-python}
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/round5_progress.log; }

if [ -n "${AFTER:-}" ]; then
  while kill -0 $(cat $AFTER) 2>/dev/null; do sleep 60; done
fi
grep -q "wC: done" $S/real_style_progress.log && [ -s $D/notation_wC.jsonl ] || { log "FAILED: window C did not finish"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ROLLOUT_MAX_GRAPHS=1

log "split window C by paper (30% of papers -> fresh RL pool)"
$PY - <<PYEOF | tee -a $S/round5_progress.log
import hashlib, json
rl = lambda d: int(hashlib.md5(str(d).encode()).hexdigest(), 16) % 10 < 3
num = [json.loads(l) for l in open('$D/notation_wC.jsonl')]
con = [json.loads(l) for l in open('$D/con_wC.jsonl')]
for name, rows in (('notation', num), ('con', con)):
    with open(f'$D/{name}_wC_sft.jsonl', 'w') as f, open(f'$D/{name}_wC_rl.jsonl', 'w') as g:
        for r in rows: (g if rl(r.get('doc_id')) else f).write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  problems: {sum(not rl(r.get('doc_id')) for r in num)} SFT, {sum(rl(r.get('doc_id')) for r in num)} RL; "
      f"conceptual pairs: {sum(not rl(r.get('doc_id')) for r in con)} SFT")
PYEOF
log "mixture"
$PY scripts/experiments/build_notation3.py data/splits/train_pot_notation5.jsonl pilot pilot2 wA wB wC_sft 2>&1 | tail -1 \
  | tee -a $S/round5_progress.log

log "fine-tune 4 models (log: $S/ctrl_tokens_progress.log)"
B=../nanogpt_seis/checkpoints/ckpt.pt; M=checkpoints/midtrain/test1p6b/ckpt.pt; N5=data/splits/train_pot_notation5.jsonl
PICK=last RUNS="113M_mid1p6B_n5 $M 2.0e-4 2.0e-5 4096 4 $N5 0
113M_3150M_n5 $B 2.0e-4 2.0e-5 4096 4 $N5 0
113M_mid1p6B_n5_d1 $M 2.0e-4 2.0e-5 4096 4 $N5 1
113M_3150M_n5_d1 $B 2.0e-4 2.0e-5 4096 4 $N5 1" bash scripts/experiments/pretrain_tokens_control.sh \
  || { log "FAILED: fine-tuning"; exit 1; }

log "done"
