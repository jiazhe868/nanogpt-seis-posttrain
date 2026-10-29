#!/bin/bash
# Real-style data round: Phase II-style pairs from unused corpus excerpts (tools/gen_real_style.py) -> labels by
# the test set's rule -> Python programs kept if they reproduce the label or agree with a blind re-solve ->
# paper notation (tools/gen_pot_problems.py).
#   START=0 COUNT=400 TAG=pilot bash scripts/experiments/real_style_chain.sh
set -u
cd "$(dirname "$0")/../.."
D=data/sources/real_style; S=${WORK:-work/experiments}; mkdir -p $D $S
PY=${PY:-python}; VPY=${VPY:-python}
START=${START:-0}; COUNT=${COUNT:-400}; TAG=${TAG:-pilot}
export VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $S/real_style_progress.log; }
log "$TAG: generate pairs for excerpts $START..$((START + COUNT))"
$VPY tools/gen_real_style.py generate --pool $D/pool.jsonl --start $START --count $COUNT --batch 2000 --max-num-seqs ${SEQS:-64} \
  --out $D/pairs_$TAG.jsonl > $S/rs_gen_$TAG.log 2>&1 || { log "FAILED: generate"; exit 1; }
tail -1 $S/rs_gen_$TAG.log | cut -c1-300 | tee -a $S/real_style_progress.log
$PY - <<PYEOF | tee -a $S/real_style_progress.log
import json
n = c = 0
with open('$D/num_$TAG.jsonl', 'w') as f, open('$D/con_$TAG.jsonl', 'w') as g:
    for l in open('$D/pairs_$TAG.jsonl'):
        r = json.loads(l)
        for p in r['pairs']:
            row = {**p, 'chunk_id': r['chunk_id'], 'doc_id': r['doc_id'], 'group': 'realstyle'}
            if p['type'] == 'numerical':
                f.write(json.dumps(row, ensure_ascii=False) + '\n'); n += 1
            else:
                g.write(json.dumps(row, ensure_ascii=False) + '\n'); c += 1
print(f'  numerical {n}, other {c}')
PYEOF
log "$TAG: convert the derivations to programs (kept if they reproduce the label, or for a blind re-solve)"
$VPY tools/gen_pot_problems.py convert --inp $D/num_$TAG.jsonl --keep-all --temperature 0.2 --max-tokens 1500 --chunk 8000 \
  --out $D/convert_$TAG.jsonl > $S/rs_convert_$TAG.log 2>&1 || { log "FAILED: convert"; exit 1; }
grep '^\[convert\]' $S/rs_convert_$TAG.log | tail -1 | tee -a $S/real_style_progress.log
log "$TAG: blind re-solve from the question alone, for programs that miss their label"
$PY -c "
import json
rows = [json.loads(l) for l in open('$D/convert_$TAG.jsonl')]
with open('$D/unmatched_$TAG.jsonl', 'w') as f:
    for r in rows:
        if not r.get('matches_label'): f.write(json.dumps(r, ensure_ascii=False) + '\n')"
$VPY tools/gen_pot_problems.py solve --inp $D/unmatched_$TAG.jsonl --temperature 0.3 --max-tokens 1500 --chunk 8000 \
  --out $D/solve_$TAG.jsonl > $S/rs_solve_$TAG.log 2>&1 || { log "FAILED: solve"; exit 1; }
$PY - <<PYEOF | tee -a $S/real_style_progress.log
import json
lab = [r for r in map(json.loads, open('$D/convert_$TAG.jsonl')) if r.get('matches_label')]
ver = [r for r in map(json.loads, open('$D/solve_$TAG.jsonl')) if r.get('verified')]
with open('$D/verified_$TAG.jsonl', 'w') as f:
    for r in lab + ver: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'  verified {len(lab) + len(ver)} programs (reproduce the label {len(lab)}, agree with a blind re-solve {len(ver)})')
PYEOF
log "$TAG: paper notation"
$VPY tools/gen_pot_problems.py notation --inp $D/verified_$TAG.jsonl --temperature 0.2 --max-tokens 1200 --chunk 8000 \
  --out $D/notation_$TAG.jsonl > $S/rs_notation_$TAG.log 2>&1 || { log "FAILED: notation"; exit 1; }
log "$TAG: done, $(wc -l < $D/notation_$TAG.jsonl) problems in paper notation"
