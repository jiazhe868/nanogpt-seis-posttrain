"""Failure taxonomy on an RL-schema file whose rows carry the teacher's Python program. Deployment decoder (notation)."""
import json, sys, math, collections, torch
sys.path.insert(0, '.')
from pathlib import Path
from src.rl.train_grpo import load_sft_model, build_prompt_ids
from src.rl.rollout import generate, decode_completion
from src.rl.pot import run_program
from src.sft.train_sft import load_bpe_tokenizer
ck, path, n, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tok = load_bpe_tokenizer(Path('.').resolve().parent / 'nanogpt_seis' / 'data' / 'tokenized' / 'tokenizer.json')
model = load_sft_model(Path(ck), torch.device('cuda'))
prog = {r['prompt']: r['program'] for r in map(json.loads, open('data/rl/transfer_A.jsonl'))}
prog.update({r['prompt']: r['program'] for r in map(json.loads, open('data/rl/transfer_B.jsonl'))})
items = [json.loads(l) for l in open(path)][:n]
def close(a, b, t=0.05):
    d = abs(b) if abs(b) > 1e-12 else 1.0; return abs(a - b) / d <= t
cats = collections.Counter(); g = torch.Generator(device='cuda').manual_seed(0)
for i in range(0, len(items), 16):
    chunk = items[i:i + 16]
    comps, _ = generate(model, [build_prompt_ids(tok, it['prompt']) for it in chunk], k, 384, temperature=0.3, top_k=40,
                        repetition_penalty=1.0, stop_ids=(16385, 0), generator=g, pot=True, pot_notation=True, tokenizer=tok)
    for j, it in enumerate(chunk):
        t = float(it['final_numerical_value']); _, tenv = run_program(prog.get(it['prompt'], ''))
        for c in comps[j * k:(j + 1) * k]:
            txt = decode_completion(tok, c, (16385, 0))
            if '<think>' not in txt or '</think>' not in txt: cats['no closed working'] += 1; continue
            a, env = run_program(txt[txt.find('<think>') + 7:txt.find('</think>')], notation=True)
            if a is None: cats['crash'] += 1; continue
            if close(a, t): cats['correct'] += 1; continue
            r = a / t if t else float('inf')
            if t and a and r > 0 and any(close(r, 10 ** e, 0.05) for e in range(-12, 13) if e): cats['units (power of ten)'] += 1
            elif t and any(close(abs(r), f, 0.03) or close(abs(r), 1 / f, 0.03) for f in (2, 3, 1.5, math.pi, 2 * math.pi, 9.81, 60, 3600, 24, 365.25, 1e3)): cats['common factor'] += 1
            elif any(close(a, v) for kk, v in tenv.items() if kk != 'answer'): cats['teacher intermediate'] += 1
            elif any(close(v, t) for v in env.values()): cats['computed target, returned other'] += 1
            else: cats['wrong setup (other)'] += 1
tot = sum(cats.values())
print(json.dumps({c: round(cats[c] / tot, 3) for c in ('correct', 'crash', 'units (power of ten)', 'common factor', 'teacher intermediate', 'computed target, returned other', 'wrong setup (other)', 'no closed working')}))
