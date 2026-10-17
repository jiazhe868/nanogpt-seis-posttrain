#!/usr/bin/env python3
"""First-step "stepping stone" questions for problems the student never solves (README §5.18.9).

A problem the student never solves gives GRPO no gradient (all 8 samples score 0). For each such problem
this picks one early quantity of the teacher's solution program (dependency depth 1-2, every given value
it needs stated verbatim in the question) and has the local teacher write a short self-contained question
asking only for it. The label is that quantity from executing the original program, so it is exact;
questions that drop a needed given value are discarded. In the transfer study, practising a never-solved
problem's first step lifted the full problem from 1.5% to 10.8% (audit §15.17).

    VLLM_USE_FLASHINFER_SAMPLER=0 python tools/gen_stepping_stones.py --inp never_solved.jsonl --out stones.jsonl
"""
import json, sys, ast, math, re, argparse, os
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from tools.gen_pot_problems import load_llm, chat
from tools.pot_counterfactuals import forms
from src.rl.pot import run_program
PROMPT = """Here is a geoscience problem and a Python solution.

Problem:
{q}

Solution:
{prog}

Write a NEW, shorter, self-contained question that asks ONLY for the quantity `{var}`, which the solution computes on this line:
    {line}
Requirements:
- Keep the same physical scenario and style, but state only the given values needed to compute `{var}`, with their units: {givens}.
- Ask for exactly one number and say which unit the answer should be in{unit_hint}.
- If the original problem states a formula that this step needs, keep it; otherwise do not add one. Never reveal the answer.
Return only the question text."""
def plan(prog):
    a, env = run_program(prog)
    if a is None: return None
    depth, given, src, deps, cands = {}, {}, {}, {}, []
    for line in prog.split('\n'):
        code, _, comment = line.partition('#')
        code = code.strip()
        try: node = ast.parse(code).body[0] if code else None
        except SyntaxError: continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)): continue
        name = node.targets[0].id
        refs = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)} & set(depth)
        if not refs:
            depth[name] = 0; given[name] = (code.split('=', 1)[1].strip(), comment.strip()); deps[name] = set(); continue
        depth[name] = 1 + max(depth[r] for r in refs)
        deps[name] = set().union(*({r} | deps[r] for r in refs))
        src[name] = (line.strip(), comment.strip())
        ops = sum(isinstance(n, (ast.BinOp, ast.Call)) for n in ast.walk(node.value))
        v = env.get(name)
        if v is None or not math.isfinite(v) or v == 0 or name == 'answer': continue
        if 1 <= depth[name] <= 2 and ops >= 1:
            cands.append((depth[name], ops, name))
    if not cands: return None
    _, _, var = max(cands)
    g = [k for k in deps[var] if depth[k] == 0]
    mids = [src[k][0] for k in src if k in deps[var]]
    return {'var': var, 'value': env[var], 'line': src[var][0], 'unit': src[var][1], 'mids': mids,
            'givens': [(k, given[k][0], given[k][1]) for k in g]}
def stated(q, lit):
    try: x = float(lit.replace(',', ''))
    except ValueError: return True
    qn = q.replace(',', '')
    cands = {f.replace(',', '') for f in forms(0, lit)} | {f'{x:g}'}
    if any(re.search(rf'(?<![\w.]){re.escape(f)}(?![\w]|\.\d)', qn) for f in cands): return True
    m = re.match(r'^(\d+(?:\.\d+)?)e([-+]?\d+)$', f'{x:.6g}')
    return bool(m and re.search(rf'{re.escape(m.group(1))}\s*[×x\*]\s*10\^?\s*\(?[-−]?{abs(int(m.group(2)))}', q))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', nargs='+', required=True); ap.add_argument('--out', required=True); ap.add_argument('--prefix', default='s')
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.inp for l in open(p)]
    jobs = []
    for r in rows:
        p = plan(r['program'])
        if p and all(stated(r['prompt'], lit) for _, lit, _ in p['givens']):
            jobs.append((r, p))
    args = argparse.Namespace(model='Qwen/Qwen3.8-27B-FP8', tp=2, quantization=None, gpu_util=0.92, max_len=8192, max_tokens=400,
                              temperature=0.3, eager=False, max_num_seqs=64, max_batched_tokens=4096, seed=0)
    llm, sp = load_llm(args)
    prompts = [PROMPT.format(q=r['prompt'], prog=r['program'], var=p['var'], line=p['line'],
                             givens='; '.join(f"{k} = {v} {u}".strip() for k, v, u in p['givens']) or 'none',
                             unit_hint=f" ({p['unit']})" if p['unit'] else '') for r, p in jobs]
    texts = chat(llm, sp, prompts)
    kept = 0
    with open(a.out, 'w') as g:
        for (r, p), t in zip(jobs, texts):
            q = t.strip().split('</think>')[-1].strip().strip('"')
            if len(q) < 30 or not all(stated(q, lit) for _, lit, _ in p['givens']): continue
            prog = '\n'.join([f"{k} = {v}  # {u}" for k, v, u in p['givens']] + p.get('mids', []) + [p['line'], f"answer = {p['var']}"])
            g.write(json.dumps({'id': f"{a.prefix}{r['id']}", 'orig_id': r['id'], 'prompt': q, 'type': 'numerical',
                                'final_numerical_value': p['value'], 'chunk_id': r['chunk_id'], 'key_concepts': [],
                                'program': prog, 'unit': p['unit'], 'orig_steps_var': p['var']}, ensure_ascii=False) + '\n')
            kept += 1
    print(f"{len(rows)} problems -> {len(jobs)} with an early step whose givens are all stated -> {kept} simple questions -> {a.out}")
if __name__ == '__main__':
    main()
