"""For each test problem, the highest word-Jaccard similarity to a training problem from the same chunk."""
import json, re, sys, collections
def words(q): return set(re.findall(r"[a-zA-Z]{3,}", q.lower()))
test = [json.loads(l) for l in open(sys.argv[1])]; train = [json.loads(l) for l in open(sys.argv[2])]
by = collections.defaultdict(list)
for r in train: by[r['chunk_id']].append(words(r['prompt']))
out = {}
for r in test:
    w = words(r['prompt'])
    out[r['id']] = max((len(w & v) / max(len(w | v), 1) for v in by.get(r['chunk_id'], [])), default=0.0)
json.dump(out, open(sys.argv[3], 'w'))
v = sorted(out.values()); n = len(v)
print(f"{sys.argv[1]} vs {sys.argv[2]}: median {v[n // 2]:.2f}, p90 {v[int(n * .9)]:.2f}, share >= 0.5: {sum(x >= 0.5 for x in v) / n:.2f}, with no same-chunk training problem: {sum(x == 0 for x in v) / n:.2f}")
