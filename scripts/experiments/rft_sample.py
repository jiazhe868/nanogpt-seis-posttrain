"""Rejection sampling for RFT: k samples per problem at T, keep up to `keep` distinct correct programs."""
import json, sys, torch
sys.path.insert(0, '.')
from pathlib import Path
from src.rl.train_grpo import load_sft_model, build_prompt_ids
from src.rl.rollout import generate, decode_completion
from src.rl.pot import run_program
from src.sft.train_sft import load_bpe_tokenizer
ck, inp, out, k, T, keep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
tok = load_bpe_tokenizer(Path('.').resolve().parent / 'nanogpt_seis' / 'data' / 'tokenized' / 'tokenizer.json')
model = load_sft_model(Path(ck), torch.device('cuda'))
items = [json.loads(l) for l in open(inp)]
def close(a, b, t=0.02):
    d = abs(b) if abs(b) > 1e-7 else 1.0; return abs(a - b) / d <= t
g = torch.Generator(device='cuda').manual_seed(0); n_out = solved = 0
with open(out, 'w') as f:
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        comps, _ = generate(model, [build_prompt_ids(tok, it['prompt']) for it in chunk], k, 384, temperature=T, top_k=None,
                            stop_ids=(16385, 0), generator=g, pot=True, pot_notation=True, tokenizer=tok)
        for j, it in enumerate(chunk):
            t = float(it['final_numerical_value']); got = []
            for c in comps[j * k:(j + 1) * k]:
                txt = decode_completion(tok, c, (16385, 0))
                if '<think>' not in txt or '</think>' not in txt: continue
                body = txt[txt.find('<think>') + 7:txt.find('</think>')].strip()
                a, _ = run_program(body, notation=True)
                if a is not None and close(a, t) and body not in got: got.append(body)
            solved += bool(got)
            unit = (it.get('unit') or '').strip().rstrip('.')
            unit = '' if unit.lower() in ('', 'dimensionless', 'none', '-', 'unitless') else ' ' + unit
            for body in got[:keep]:
                a, _ = run_program(body, notation=True)
                f.write(json.dumps({'instruction': it['prompt'], 'response': f"<think>\n{body}\n</think>Answer: {a:.4g}{unit}.",
                                    'type': 'numerical', 'chunk_id': it.get('chunk_id')}, ensure_ascii=False) + '\n'); n_out += 1
print(f"{len(items)} problems | solved at least once in {k}: {solved} | RFT pairs: {n_out}")
