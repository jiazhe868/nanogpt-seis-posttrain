"""Collect the RL transfer diagnosis (audit §15.17) into docs/rl_transfer_diagnosis.json."""
import json, random, sys
S = sys.argv[1]
ea, eb = json.load(open(f'{S}/transfer_eval_a.json')), json.load(open(f'{S}/transfer_eval_b.json'))
es = json.load(open(f'{S}/transfer_eval_sameseed.json'))
ev = {}
for src in (ea, eb, es):
    for m, sets in src.items():
        for s, items in sets.items():
            ev.setdefault(m, {}).setdefault(s, items)
passes = {r['id']: r['passes'] for r in map(json.loads, open(f'{S}/learnB600_base.jsonl'))}
for m in ev:
    if 'B600' in ev[m]:
        ev[m]['B600 sometimes-solved'] = {i: v for i, v in ev[m]['B600'].items() if 0 < passes.get(i, 0) < 8}
        ev[m]['B600 never-solved'] = {i: v for i, v in ev[m]['B600'].items() if passes.get(i, 0) == 0}
def stat(m, s, base='notation2'):
    a, b = ev[base][s], ev[m][s]
    ids = sorted(a)
    rng = random.Random(0)
    d = sorted(sum(b[i] - a[i] for i in smp) / len(smp) for smp in ([rng.choice(ids) for _ in ids] for _ in range(2000)))
    return {'n': len(ids), 'acc': sum(b[i] for i in ids) / len(ids), 'base_acc': sum(a[i] for i in ids) / len(ids),
            'diff': sum(b[i] - a[i] for i in ids) / len(ids), 'ci': [d[50], d[1949]]}
order = ['A-learnable', 'A0-shared', 'A0-unshared', 'A8-solved', 'B600 sometimes-solved', 'B600 never-solved', 'B600', 'real']
models = ['grpo-lo-75', 'grpo-lo-150', 'rftA', 'sftA', 'grpo-hi-75']
out = {'_provenance': {'decoding': 'deployment: paper-notation PoT, T=0.3, top-k 40, rp 1.0; 2 seeds x 4 samples per item',
                       'base': 'checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt',
                       'trained_on': 'data/rl/transfer_A_learnable.jsonl (576 problems, 75 chunks)',
                       'files': ['transfer_eval_a.json', 'transfer_eval_b.json', 'transfer_eval_sameseed.json', 'locality_lo75.json']},
       'transfer': {m: {s: stat(m, s) for s in order if s in ev.get(m, {}) and s in ev['notation2']} for m in models if m in ev}}
L = json.load(open(f'{S}/locality_lo75.json'))
dec = {}
for s in ('A-learnable', 'B600', 'real'):
    dec[s] = {}
    for samp in ('base', 'grpo'):
        x = [r for r in L if r['set'] == s and r['sampler'] == samp]
        runs = [r for r in x if r['why'] == 'ran']
        dec[s][samp] = {'acc': sum(r['correct'] for r in x) / len(x), 'runs': len(runs) / len(x),
                        'acc_given_runs': sum(r['correct'] for r in runs) / len(runs),
                        'kl_per_token': sum(r['kl'] for r in x) / sum(r['ntok'] for r in x)}
    b = [r for r in L if r['set'] == s and r['sampler'] == 'base']
    grp = {'correct': [r for r in b if r['correct']], 'runs, wrong value': [r for r in b if r['why'] == 'ran' and not r['correct']],
           'crashes': [r for r in b if r['why'] != 'ran']}
    dec[s]['dlogp_per_token'] = {k: sum(r['dlogp'] for r in y) / sum(r['ntok'] for r in y) for k, y in grp.items()}
out['decomposition_T1'] = dec
tax = {}
for f in ('tax_transfer_base.txt', 'tax_transfer_grpo.txt'):
    for line in open(f'{S}/{f}'):
        name, rest = line.split(None, 1)
        st, d = rest.split(': ', 1)
        tax.setdefault(name, {})[st.replace('transfer_', '').replace('.jsonl', '')] = json.loads(d)
out['failure_taxonomy_T03'] = tax
# gain vs word-Jaccard similarity to the nearest trained question (any chunk)
import os
BUCKETS = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01)]
def by_sim(m, s, nnfile, base='notation2'):
    nn = json.load(open(f'{S}/{nnfile}')); rows = []
    for lo, hi in BUCKETS:
        ids = [i for i in ev[base][s] if lo <= nn.get(i, 0) < hi]
        if len(ids) < 10: continue
        a, b = ev[base][s], ev[m][s]; rng = random.Random(0)
        d = sorted(sum(b[i] - a[i] for i in smp) / len(smp) for smp in ([rng.choice(ids) for _ in ids] for _ in range(2000)))
        rows.append({'bucket': [lo, min(hi, 1.0)], 'n': len(ids), 'base_acc': sum(a[i] for i in ids) / len(ids),
                     'acc': sum(b[i] for i in ids) / len(ids), 'diff': sum(b[i] - a[i] for i in ids) / len(ids), 'ci': [d[50], d[1949]]})
    return rows
sim = {'A never solved, GRPO on A @75': by_sim('grpo-lo-75', 'A0-shared', 'nn_A0shared_A.json'),
       'B600, GRPO on A @75 (other papers)': by_sim('grpo-lo-75', 'B600', 'nn_B600_A.json')}
if os.path.exists(f'{S}/bsame_eval_0.json'):
    for f in ('bsame_eval_0.json', 'bsame_eval_1.json'):
        for m, sets in json.load(open(f'{S}/{f}')).items():
            for st, items in sets.items(): ev.setdefault(m, {})[st] = items
    best = max((m for m in ev if m.startswith('bsame-')), key=lambda m: sum(ev[m]['B600'].values()))
    sim[f'B600, GRPO on B-other @{best.split("-")[1]} (same papers)'] = by_sim(best, 'B600', 'nn_B600_Botherlearn.json')
    out['bsame'] = {m: {st: stat(m, st) for st in ('B600', 'real')} for m in ev if m.startswith('bsame-')}
out['similarity'] = sim
out['_provenance']['similarity'] = 'word-Jaccard (content words of 3+ letters) between a test question and its most similar trained question'
json.dump(out, open('docs/rl_transfer_diagnosis.json', 'w'), indent=1)
for m, sets in out['transfer'].items():
    print(m); [print(f"   {s:24s} n={v['n']:4d} {v['base_acc']:.3f} -> {v['acc']:.3f}  {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]") for s, v in sets.items()]

# simpler questions (curriculum) and the reading pilot
import os
if os.path.exists(f'{S}/simple_followup_1.json'):
    for f in ('simple_eval_base.json', 'simple_eval_0.json', 'simple_eval_1.json', 'simple_followup_0.json', 'simple_followup_1.json',
              'pilot_eval_a.json', 'pilot_eval_b.json'):
        for m, sets in json.load(open(f'{S}/{f}')).items():
            for st_, items in sets.items(): ev.setdefault(m, {})[st_] = items
    def stat_ids(m, s, ids, base='notation2'):
        a, b = ev[base][s], ev[m][s]; rng = random.Random(0)
        d = sorted(sum(b[i] - a[i] for i in smp) / len(smp) for smp in ([rng.choice(ids) for _ in ids] for _ in range(2000)))
        return {'n': len(ids), 'base_acc': sum(a[i] for i in ids) / len(ids), 'acc': sum(b[i] for i in ids) / len(ids),
                'diff': sum(b[i] - a[i] for i in ids) / len(ids), 'ci': [d[50], d[1949]]}
    trained = {json.loads(l)['orig_id'] for l in open('data/rl/simple_A_learnable.jsonl')}
    a0 = list(ev['notation2']['A0-shared'])
    out['simple_questions'] = {
        '_provenance': 'simple = one early quantity (1-2 computed steps) of each original solution; label from the original program; '
                       'GRPO transfer-lo recipe on 499 learnable simple-A questions (configs/rl_grpo_simpleA.yaml)',
        'results': {m: {s: stat_ids(m, s, list(ev['notation2'][s])) for s in ('simple-A-learnable', 'simple-B600', 'B600', 'real', 'A-learnable', 'A0-shared')
                        if s in ev.get(m, {}) and s in ev['notation2']} for m in ('simpleA-75', 'simpleA-150', 'grpo-lo-75', 'bsame-150')},
        'curriculum_A0_shared': {'first step in simple pool': stat_ids('simpleA-75', 'A0-shared', [i for i in a0 if i in trained]),
                                 'first step not in pool': stat_ids('simpleA-75', 'A0-shared', [i for i in a0 if i not in trained])},
        'learnability': {'simple_A': {'never': 790, 'learnable': 499, 'always': 243, 'n': 1532, 'mean_pass': 0.296},
                         'original_A': {'never': 1272, 'learnable': 576, 'always': 179, 'n': 2027, 'mean_pass': 0.207}}}
    out['reading_pilot'] = {
        '_provenance': '8 teacher restatements per source chunk; K = notes on B chunks, C = notes on A chunks; 658 notes + 3,000 replay, 3 epochs, lr 5e-5',
        'results': {m: {s: stat_ids(m, s, list(ev['notation2'][s])) for s in ('B-rec', 'A-rec', 'real')} for m in ('K', 'C')}}
    json.dump(out, open('docs/rl_transfer_diagnosis.json', 'w'), indent=1)
    print('simple + pilot sections added')
