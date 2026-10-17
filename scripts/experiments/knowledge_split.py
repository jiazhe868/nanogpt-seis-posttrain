"""Split a set by whether the teacher's program needs a number the question does not state (a constant or
formula coefficient the solver must know), then compare per-item accuracy of models on each part."""
import json, re, sys, random
sys.path.insert(0, '.')
from tools.pot_counterfactuals import ASSIGN, forms
def literals(prog):
    out = []
    for line in prog.split('\n'):
        m = ASSIGN.match(line)
        if m: out.append(m.group(4))
        else:   # numbers inside expressions, e.g. "Mw = (2/3)*log10(M0) - 10.7"
            rhs = line.split('=', 1)[1] if '=' in line else ''
            rhs = re.sub(r'#.*', '', rhs)
            out += [x for x in re.findall(r'(?<![\w.])\d+\.?\d*(?:[eE][-+]?\d+)?', rhs) if x not in ('2', '3', '1', '0', '10', '100', '1000', '0.5', '60', '24', '3600', '365', '365.25', '1e3', '1e6', '1e9', '1e-3', '1e-6')]
    return out
def stated(q, lit):
    qn = q.replace(',', '')
    try: x = float(lit.replace(',', ''))
    except ValueError: return True
    for f in forms(0, lit) | {f"{x:g}", f"{x:.0f}" if x.is_integer() else f"{x:g}"}:
        if re.search(rf"(?<![\w.]){re.escape(f.replace(',', ''))}(?![\w]|\.\d)", qn): return True
    m = re.match(r'^(\d+(?:\.\d+)?)[eE]([-+]?\d+)$', f"{x:.6g}")
    if m and re.search(rf"{re.escape(m.group(1))}\s*[×x\*]\s*10\^?\s*{int(m.group(2))}", q): return True
    return False
items = {r['id']: r for r in map(json.loads, open(sys.argv[1]))}
ev = json.load(open(sys.argv[2])); setname = sys.argv[3]; models = sys.argv[4].split(',')
need = {i: any(not stated(r['prompt'], l) for l in literals(r['program'])) for i, r in items.items()}
for lab, sel in (('all constants stated in question', False), ('needs an unstated constant', True)):
    ids = [i for i in items if need[i] == sel and i in ev[models[0]][setname]]
    accs = {m: sum(ev[m][setname][i] for i in ids) / len(ids) for m in models}
    rng = random.Random(0); d = []
    for _ in range(2000):
        s = [rng.choice(ids) for _ in ids]
        d.append(sum(ev[models[1]][setname][i] - ev[models[0]][setname][i] for i in s) / len(s))
    d.sort()
    print(f"{lab:34s} n={len(ids):3d} | " + ' '.join(f"{m} {a:.3f}" for m, a in accs.items()) +
          f" | {models[1]} - {models[0]}: {accs[models[1]] - accs[models[0]]:+.3f} [{d[50]:+.3f}, {d[1949]:+.3f}]")
