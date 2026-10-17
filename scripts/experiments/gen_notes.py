"""Knowledge-augmentation pilot: the local teacher restates each source chunk in several styles
(no questions, no worked examples), so its content appears in many phrasings instead of one."""
import json, sys, argparse, os, random
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from tools.gen_pot_problems import load_llm, chat
STYLES = [
    "a concise textbook section that explains the physical ideas and every quantitative relationship",
    "lecture notes: one short paragraph per relationship, giving the equation, the meaning and unit of each symbol, and the values or conditions stated in the excerpt",
    "an encyclopedia-style summary written for a geophysics student",
    "a paraphrase of the methods and results as a different author would write them in a review article",
    "an explanation for an engineer who needs to apply these relationships, saying which units each equation expects",
    "a summary that starts from the main result and then explains how it was obtained",
    "notes that connect each quantity to its physical meaning and to the other quantities it depends on",
    "a plain-language explanation followed by the precise equations and parameter values",
]
PROMPT = """Below is an excerpt from a geoscience paper. Restate its scientific content as {style}.

Rules:
- Keep every quantitative fact: equations, parameter values, units, and the conditions under which they hold.
- Write equations in paper notation (for example M0 = μ·A·D, or Mw = (2/3)·log10(M0) − 6.07 with M0 in N·m) and define every symbol with its unit.
- Use only what the excerpt says: do not add facts, and do not pose questions, exercises or worked examples.
- Plain text without markdown headings.

Excerpt:
---
{text}
---"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', required=True); ap.add_argument('--ids', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3.8-27B-FP8'); ap.add_argument('--tp', type=int, default=2)
    a = ap.parse_args()
    want = {json.loads(l)['chunk_id'] for p in a.ids for l in open(p)}
    C = [r for r in map(json.loads, open(a.chunks)) if r['chunk_id'] in want]
    args = argparse.Namespace(model=a.model, tp=a.tp, quantization=None, gpu_util=0.92, max_len=8192, max_tokens=1400,
                              temperature=0.7, eager=False, max_num_seqs=64, max_batched_tokens=4096, seed=0)
    llm, sp = load_llm(args)
    jobs = [(c, s) for c in C for s in range(len(STYLES))]
    texts = chat(llm, sp, [PROMPT.format(style=STYLES[s], text=c['text'][:9000]) for c, s in jobs])
    with open(a.out, 'w') as g:
        for (c, s), t in zip(jobs, texts):
            t = t.strip()
            if '</think>' in t: t = t.split('</think>', 1)[1].strip()
            if len(t) < 300: continue
            g.write(json.dumps({'chunk_id': c['chunk_id'], 'title': c['title'], 'style': s, 'note': t}, ensure_ascii=False) + '\n')
    print(len(C), 'chunks ->', len(jobs), 'notes ->', a.out)


if __name__ == '__main__':   # vLLM spawns TP workers that re-import this file
    main()
