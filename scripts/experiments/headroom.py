"""RL headroom: pass@k, maj@k and answer diversity at the RL sampling temperature (paper-notation students)."""
import json, sys, re, math, statistics as st, collections, torch
sys.path.insert(0, '.')
from pathlib import Path
from src.rl.train_grpo import load_sft_model, build_prompt_ids
from src.rl.rollout import generate, decode_completion
from src.rl.pot import run_program
from src.sft.train_sft import load_bpe_tokenizer
ck, K, T, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
tok = load_bpe_tokenizer(Path('.').resolve().parent / 'nanogpt_seis' / 'data' / 'tokenized' / 'tokenizer.json')
model = load_sft_model(Path(ck), torch.device('cuda'))
def close(a, b, t=0.05):
    d = abs(b) if abs(b) > 1e-7 else 1.0; return abs(a - b) / d <= t
def load(p, n):
    R = [json.loads(l) for l in open(p)]
    return [r for r in R if r['type'] == 'numerical' and r.get('final_numerical_value') not in (None, 'None')][:n]
sets = {'real held-out': load('data/rl/test.jsonl', 153), 'held-out rewrites': load('data/eval/pot_rewrite_heldout.jsonl', 300),
        'v12 RL pool': load('data/rl/train_v12_learnable.jsonl', 300)}
def pass_at_k(n, c, k):            # unbiased estimator (Chen et al. 2021)
    if n - c < k: return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
res = {}
g = torch.Generator(device='cuda').manual_seed(0)
for name, items in sets.items():
    rows = []
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        comps, _ = generate(model, [build_prompt_ids(tok, it['prompt']) for it in chunk], K, 384, temperature=T, top_k=None,
                            stop_ids=(16385, 0), generator=g, pot=True, pot_notation=True, tokenizer=tok)
        for j, it in enumerate(chunk):
            t = float(it['final_numerical_value']); answers = []
            for c in comps[j * K:(j + 1) * K]:
                txt = decode_completion(tok, c, (16385, 0))
                body = txt[txt.find('<think>') + 7:txt.find('</think>')] if '<think>' in txt and '</think>' in txt else ''
                a, _ = run_program(body, notation=True) if body else (None, {})
                answers.append(a)
            corr = [a is not None and close(a, t) for a in answers]
            runs = [a is not None for a in answers]
            # distinct answers (to 3 significant figures) among programs that run
            distinct = len({f"{a:.3g}" for a in answers if a is not None})
            # majority answer among runnable programs (clustered at 5%)
            maj = False
            ok = [a for a in answers if a is not None]
            if ok:
                best = max(ok, key=lambda x: sum(close(y, x) for y in ok))
                maj = close(best, t)
            rows.append({'c': sum(corr), 'runs': sum(runs), 'distinct': distinct, 'maj': maj})
    n = K
    r = {f'pass@{k}': st.mean(pass_at_k(n, x['c'], k) for x in rows) for k in (1, 4, 8, 32) if k <= n}
    r.update({f'maj@{K}': st.mean(x['maj'] for x in rows), 'run rate': st.mean(x['runs'] / n for x in rows),
              'distinct answers / question': st.mean(x['distinct'] for x in rows),
              'questions never solved': st.mean(x['c'] == 0 for x in rows), 'questions always solved': st.mean(x['c'] == n for x in rows)})
    res[name] = r
    print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(res, open(out, 'w'))
