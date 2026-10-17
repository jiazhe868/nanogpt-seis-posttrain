"""How hard are the questions, and does RL gain more on easier ones?  Difficulty = number of computed steps in
the teacher's solution program (assignments whose right side is not a bare number), plus question length."""
import json, ast, re, random, collections, statistics as st
S = __import__('os').environ.get('WORK', 'work/experiments')
def steps(prog):
    n = 0
    for line in (prog or '').split('\n'):
        line = re.sub(r'#.*', '', line).strip()
        if '=' not in line or line.startswith(('import', 'from')): continue
        lhs, rhs = line.split('=', 1)
        rhs = rhs.strip()
        if re.fullmatch(r'[-+]?\d[\d_,]*\.?\d*(?:[eE][-+]?\d+)?', rhs): continue        # a given value
        if re.fullmatch(r'[A-Za-z_]\w*', rhs) and lhs.strip() == 'answer': continue     # answer = x alias
        n += 1
    return n
def bucket(k): return '1' if k <= 1 else ('2' if k == 2 else ('3-4' if k <= 4 else '5+'))
ORDER = ['1', '2', '3-4', '5+']
def table(title, items, ev, base, models, key='id'):
    print(f'\n== {title}')
    by = collections.defaultdict(list)
    for it in items:
        if str(it[key]) in ev[base]: by[bucket(steps(it['program']))].append(str(it[key]))
    print(f"   {'steps':6s} {'n':>4s} {'base':>6s} " + ' '.join(f'{m:>22s}' for m in models))
    for b in ORDER:
        ids = by.get(b, [])
        if len(ids) < 8: continue
        a = sum(ev[base][i] for i in ids) / len(ids)
        cells = []
        for m in models:
            x = sum(ev[m][i] for i in ids) / len(ids); rng = random.Random(0)
            d = sorted(sum(ev[m][i] - ev[base][i] for i in s) / len(s) for s in ([rng.choice(ids) for _ in ids] for _ in range(1000)))
            cells.append(f"{100 * (x - a):+5.1f} [{100 * d[25]:+5.1f},{100 * d[974]:+5.1f}]")
        print(f"   {b:6s} {len(ids):4d} {100 * a:5.1f}% " + ' '.join(f'{c:>22s}' for c in cells))
ea = json.load(open(f'{S}/transfer_eval_a.json')); ev = {m: dict(ea[m]) for m in ea}
for f in ('bsame_eval_0.json', 'bsame_eval_1.json'):
    for m, sets in json.load(open(f'{S}/{f}')).items(): ev[m] = sets
A = [json.loads(l) for l in open('data/rl/transfer_A_learnable.jsonl')]
B = [json.loads(l) for l in open('data/rl/transfer_B600.jsonl')]
evA = {m: ev[m]['A-learnable'] for m in ('notation2', 'grpo-lo-75') if 'A-learnable' in ev[m]}
table('A-learnable: the trained problems (in-distribution)', A, evA, 'notation2', ['grpo-lo-75'])
evB = {m: ev[m]['B600'] for m in ('notation2', 'grpo-lo-75', 'bsame-150')}
table('B600: other papers (GRPO on A) and same papers (GRPO on B-other)', B, evB, 'notation2', ['grpo-lo-75', 'bsame-150'])
# real held-out: teacher programs + per-item results over 6 seeds
T = {str(json.loads(l)['id']): json.loads(l) for l in open('docs/teacher_27b_heldout_blind.jsonl')}
R = json.load(open('docs/eval_numerical_real6_final.json'))['results']
def per_item(name):
    acc = collections.defaultdict(list)
    for seed in R[name]:
        for it in seed['per_item']:
            if it['type'] == 'numerical': acc[str(it['id'])].append(it['acc5'])
    return {i: sum(v) / len(v) for i, v in acc.items()}
names = {'notation': 'notation', 'v11': 'notation + GRPO v11 @100', 'notation2': 'notation2 (recovered data)', 'v12': 'notation2 + GRPO v12 @75 (dev-selected)'}
evR = {k: per_item(v) for k, v in names.items()}
items = [{'id': i, 'program': T[i]['program2']} for i in evR['notation'] if i in T]
table('real held-out: GRPO v11 over notation', items, {k: evR[k] for k in ('notation', 'v11')}, 'notation', ['v11'])
table('real held-out: GRPO v12 over notation2', items, {k: evR[k] for k in ('notation2', 'v12')}, 'notation2', ['v12'])
for name, its in (('A-learnable', A), ('B600', B), ('real (teacher programs)', [{'program': T[i]['program2']} for i in T])):
    c = collections.Counter(bucket(steps(it['program'])) for it in its)
    print(f"{name:24s} step mix: " + ', '.join(f"{b}: {c[b] / len(its):.0%}" for b in ORDER) + f" | median steps {st.median(steps(it['program']) for it in its)}")
