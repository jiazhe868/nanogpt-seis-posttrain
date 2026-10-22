#!/bin/bash
# Outcome verifier for the deployed model (src/rl/verifier.py):
#   1. labelled training samples on ~10.5k strict-protocol problems (no test or dev chunks), both GPUs
#   2. train the verifier (GPU 0) while sampling 32 answers per dev and test question at T 0.3 and 0.8 (GPU 1)
#   3. score dev with each epoch; the test set is scored only for the configuration selected on dev
#   bash scripts/experiments/verifier_chain.sh > work/experiments/verifier/chain.log 2>&1
set -u
cd "$(dirname "$0")/../.."
V=work/experiments/verifier
PY=${PY:-python}
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "problems: strict pools without any test or dev chunk or prompt (all real-style rewrites, the teacher"
log "  problems the deployed model sometimes solves, 1,500 others); dev = data/rl/v13_dev.jsonl"
mkdir -p $V
$PY - <<'EOF'
import json, random
L = lambda p: [json.loads(l) for l in open(p)]
held = L('data/rl/test.jsonl') + L('data/rl/v13_dev.jsonl') + L('data/rl/val.jsonl') + L('data/rl/val_numerical_dev.jsonl')
bad_c = {x['chunk_id'] for x in held}; bad_q = {x['prompt'].strip().lower() for x in held}
ok = lambda x: x['chunk_id'] not in bad_c and x['prompt'].strip().lower() not in bad_q and x.get('final_numerical_value') is not None
ex = [x for x in L('data/rl/v13_filtered_existing.jsonl') if ok(x)]
te = [x for x in L('data/rl/v13_filtered_teacher.jsonl') if ok(x)]
rng = random.Random(0)
mixed = [x for x in te if 0 < x['passes'] < x['k']]; rest = [x for x in te if not 0 < x['passes'] < x['k']]
seen, uniq = set(), []
for x in ex + mixed + rng.sample(rest, 1500):
    if x['id'] not in seen: seen.add(x['id']); uniq.append(x)
rng.shuffle(uniq)
keys = ('id', 'prompt', 'type', 'final_numerical_value', 'chunk_id')
for h in range(2):
    with open(f'work/experiments/verifier/train_problems_{h}.jsonl', 'w') as f:
        for x in uniq[h::2]: f.write(json.dumps({k: x[k] for k in keys}) + '\n')
with open('work/experiments/verifier/dev_problems.jsonl', 'w') as f:
    for x in L('data/rl/v13_dev.jsonl'):
        if x.get('final_numerical_value') is not None: f.write(json.dumps({k: x[k] for k in keys}) + '\n')
print(len(uniq), 'training problems')
EOF

log "training samples"
for h in 0 1; do
  ( export CUDA_VISIBLE_DEVICES=$h
    $PY scripts/experiments/verifier_samples.py $V/train_problems_$h.jsonl $V/train_T08_$h.jsonl 8 0.8 $h > $V/samples_$h.log 2>&1
    $PY scripts/experiments/verifier_samples.py $V/train_problems_$h.jsonl $V/train_T03_$h.jsonl 4 0.3 $((10 + h)) >> $V/samples_$h.log 2>&1 ) &
done
wait

log "train verifier (GPU 0) | dev and test samples (GPU 1)"
( export CUDA_VISIBLE_DEVICES=0
  $PY scripts/experiments/verifier_train.py checkpoints/verifier $V/train_T08_0.jsonl $V/train_T08_1.jsonl \
    $V/train_T03_0.jsonl $V/train_T03_1.jsonl > $V/train.log 2>&1 ) &
( export CUDA_VISIBLE_DEVICES=1
  for T in 0.3 0.8; do
    $PY scripts/experiments/verifier_samples.py $V/dev_problems.jsonl $V/dev_T$T.jsonl 32 $T 21 > $V/dev_samples.log 2>&1
    $PY scripts/experiments/verifier_samples.py data/rl/test.jsonl $V/test_T$T.jsonl 32 $T 31 > $V/test_samples.log 2>&1
  done ) &
wait

log "dev scoring"
for e in 1 2; do
  ( export CUDA_VISIBLE_DEVICES=$((e - 1))
    $PY scripts/experiments/verifier_eval.py checkpoints/verifier/verifier_epoch$e.pt $V/dev_eval_epoch$e.json \
      $V/dev_T0.3.jsonl $V/dev_T0.8.jsonl > $V/dev_eval_epoch$e.log 2>&1 ) &
done
wait
log "verifier chain done"
