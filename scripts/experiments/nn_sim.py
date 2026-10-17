"""Nearest-neighbour word-Jaccard of each test question to a training pool (any chunk), via an inverted index."""
import json, re, sys, collections
def words(q): return frozenset(re.findall(r"[a-zA-Z]{3,}", q.lower()))
def load(p, key):
    R = [json.loads(l) for l in open(p)]
    return [r for r in R if r.get('type', 'numerical') == 'numerical'], key
test, tk = load(sys.argv[1], 'prompt'); pool = [json.loads(l) for l in open(sys.argv[2])]
pk = 'prompt' if 'prompt' in pool[0] else 'instruction'
P = [words(r[pk]) for r in pool]
df = collections.Counter(w for p in P for w in p)
inv = collections.defaultdict(list)
for j, p in enumerate(P):
    for w in p:
        if df[w] < 0.05 * len(P): inv[w].append(j)
out = {}
for r in test:
    w = words(r[tk]); cand = collections.Counter(j for x in w for j in inv.get(x, ()))
    best = 0.0
    for j, _ in cand.most_common(300):
        best = max(best, len(w & P[j]) / len(w | P[j]))
    out[r['id']] = best
json.dump(out, open(sys.argv[3], 'w'))
v = sorted(out.values()); n = len(v)
print(f"{sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]} ({len(P)}): median NN Jaccard {v[n // 2]:.2f}, p75 {v[int(n * .75)]:.2f}, p90 {v[int(n * .9)]:.2f}, share >= 0.35: {sum(x >= 0.35 for x in v) / n:.2f}, >= 0.5: {sum(x >= 0.5 for x in v) / n:.2f}")
