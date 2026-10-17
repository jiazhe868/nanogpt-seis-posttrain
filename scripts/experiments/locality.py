"""Where did GRPO change the policy? Exact per-token KL(GRPO || base) along samples, and the log-prob
shift of correct vs incorrect samples, on trained (A-learnable), unseen-seed (B600) and real prompts.
Also records why programs crash. Plain sampling at T=1, paper-notation PoT decoder."""
import json, sys, math, random, ast, torch
sys.path.insert(0, '.')
from pathlib import Path
from src.rl.train_grpo import load_sft_model, build_prompt_ids, full_logits
from src.rl.rollout import generate_rollouts, decode_completion
from src.rl import pot
from src.sft.train_sft import load_bpe_tokenizer
base_ck, grpo_ck, out_path, sets_arg, n, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
STOP = (16385, 0)
dev = torch.device('cuda')
tok = load_bpe_tokenizer(Path('.').resolve().parent / 'nanogpt_seis' / 'data' / 'tokenized' / 'tokenizer.json')
M = {'base': load_sft_model(Path(base_ck), dev).eval(), 'grpo': load_sft_model(Path(grpo_ck), dev).eval()}
def close(a, b, t=0.05):
    d = abs(b) if abs(b) > 1e-12 else 1.0; return abs(a - b) / d <= t
def crash_reason(body):
    src = pot.MATH_PREFIX.sub("", pot.MATH_IMPORT.sub("", pot.THOUSANDS.sub("", pot.extract_program(body))))
    src = pot.normalize_notation(src)
    if src is None: return 'notation rejected'
    if not src.strip(): return 'empty'
    try: tree = ast.parse(src)
    except (SyntaxError, ValueError): return 'syntax'
    env = {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name): node, name = st.value, st.targets[0].id
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str): continue
        else: return 'non-assignment line'
        try: v = float(pot._eval(node, env))
        except pot.ProgramError as e:
            m = str(e)
            if m.startswith('undefined name'):
                return 'undefined: misspelt' if pot._resolve(m.split()[-1], env) else 'undefined: never defined'
            return 'disallowed call' if 'call' in m else ('division by zero' if 'division' in m else 'other construct')
        except (ArithmeticError, ValueError, OverflowError, TypeError): return 'math domain'
        if not math.isfinite(v): return 'non-finite'
        env[name] = v
    return 'limits'
recs = []
for sname, path in (kv.split('=') for kv in sets_arg.split(',')):
    items = [json.loads(l) for l in open(path)]
    items = [r for r in items if r.get('type', 'numerical') == 'numerical' and r.get('final_numerical_value') is not None]
    random.Random(0).shuffle(items); items = items[:n]
    for sampler in ('base', 'grpo'):
        g = torch.Generator(device='cuda').manual_seed(1)
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            pids = [build_prompt_ids(tok, it['prompt']) for it in chunk]
            comps, term, forced = generate_rollouts(M[sampler], pids, k, 384, temperature=1.0, generator=g,
                                                    tokenizer=tok, pot=True, pot_notation=True)
            for j, it in enumerate(chunk):
                for r in range(k):
                    c, f = comps[j * k + r], forced[j * k + r]
                    L = next((t + 1 for t, x in enumerate(c) if x in STOP), len(c))
                    c, f = c[:L], f[:L]
                    if not c: continue
                    P = len(pids[j]); ids = torch.tensor([pids[j] + c], device=dev)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        lb = full_logits(M['base'], ids, P - 1).float().log_softmax(-1)[0, :L]
                        lg = full_logits(M['grpo'], ids, P - 1).float().log_softmax(-1)[0, :L]
                    tgt = torch.tensor(c, device=dev)[:, None]
                    keep = torch.tensor([1.0 - x for x in f], device=dev)
                    d = ((lg.gather(1, tgt) - lb.gather(1, tgt))[:, 0] * keep)
                    kl = ((lg.exp() * (lg - lb)).sum(-1) * keep)
                    txt = decode_completion(tok, c, STOP)
                    ok, why = False, 'no closed working'
                    if '<think>' in txt and '</think>' in txt:
                        body = txt[txt.find('<think>') + 7:txt.find('</think>')]
                        a, _ = pot.run_program(body, notation=True)
                        if a is None: why = crash_reason(body)
                        else: ok, why = close(a, float(it['final_numerical_value'])), 'ran'
                    recs.append({'set': sname, 'sampler': sampler, 'id': it.get('id'), 'correct': ok, 'why': why,
                                 'ntok': float(keep.sum()), 'dlogp': float(d.sum()), 'kl': float(kl.sum()),
                                 'kl_first8': float(kl[:8].sum())})
        print(sname, sampler, 'done', flush=True)
json.dump(recs, open(out_path, 'w'))
import collections
def mean(x): return sum(x) / max(len(x), 1)
for sname in dict.fromkeys(r['set'] for r in recs):
    R = [r for r in recs if r['set'] == sname]
    b = [r for r in R if r['sampler'] == 'base']; g = [r for r in R if r['sampler'] == 'grpo']
    print(f"\n== {sname}: acc@T=1 base {mean([r['correct'] for r in b]):.3f}  grpo {mean([r['correct'] for r in g]):.3f}")
    print(f"   KL(grpo||base) on grpo samples: {mean([r['kl'] for r in g]):.3f} nats/sequence, "
          f"{sum(r['kl'] for r in g) / sum(r['ntok'] for r in g):.4f} nats/token | on base samples: {mean([r['kl'] for r in b]):.3f}/seq")
    for lab, sel in (('correct', True), ('incorrect', False)):
        x = [r for r in b if r['correct'] == sel]
        if x: print(f"   base samples {lab:9s} (n={len(x)}): mean log p_grpo - log p_base = {mean([r['dlogp'] for r in x]):+.3f} nats/seq")
    for s, x in (('base', b), ('grpo', g)):
        c = collections.Counter(r['why'] for r in x if r['why'] not in ('ran',))
        print(f"   crash/fail reasons ({s} samples, share of all samples):", {kk: round(v / len(x), 3) for kk, v in c.most_common()})
