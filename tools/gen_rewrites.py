"""Teaching rewrites of the seismology excerpts for continued pretraining (scripts/midtrain/).

    python tools/gen_rewrites.py --start 0 --count 64000 --out work/midtrain/rewrites/rw_0.jsonl

Each excerpt of the decontaminated pool (data/sources/real_style/pool.jsonl: test and validation papers and held-out
earthquakes already removed) is rewritten in two of three styles, rotating by position: a textbook section, lecture
notes with worked numerical examples, and exam questions with model answers. The teacher is Qwen3.5-4B without
thinking. scripts/midtrain/filter_rewrites.py later drops truncated outputs and those whose given values are not in
the excerpt.
"""
import argparse
import json
import os
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

SYSTEM = ("You are an expert seismologist and science writer. You turn research excerpts into clear teaching material "
          "for graduate students in geophysics. Use only facts, relations and values that the excerpt states, or "
          "standard textbook seismology; never invent results, data, place names or citations. Do not mention the "
          "excerpt, the paper or its authors: write a self-contained text. Write equations in plain text or LaTeX and "
          "define every symbol with its unit.")
STYLES = {
    "textbook": "Rewrite the excerpt below as a section of a seismology textbook. Explain the physical ideas step by "
                "step. State each relation the excerpt uses as an equation, define every symbol and unit, and explain "
                "why the relation holds and when it applies. About 500-800 words.",
    "worked": "Write lecture notes with worked examples based on the excerpt below. Pick the two or three most "
              "important quantitative relations in it, or standard relations it relies on. For each one, state the "
              "relation with its symbols and units, then solve one numerical example step by step, using values from "
              "the excerpt where it gives them. Show every arithmetic step and the unit of the result. About 400-700 "
              "words.",
    "exam": "Write 4 to 6 exam questions with complete model answers based on the excerpt below. At least two must be "
            "quantitative: give every needed value in the question, then compute the answer step by step with units. "
            "The others ask why or how. Format each as 'Question: ...' followed by 'Answer: ...'.",
}
ORDER = list(STYLES)
USER = "{instruction}\n\nExcerpt:\n\"\"\"\n{text}\n\"\"\""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/sources/real_style/pool.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--styles-per-excerpt", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4000, help="requests per vLLM call (results saved after each)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    reqs = []
    with open(a.pool, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k >= a.start + a.count:
                break
            if k >= a.start:
                c = json.loads(line)
                for j in range(a.styles_per_excerpt):
                    reqs.append((c, ORDER[(k + j) % len(ORDER)]))
    done = set()
    if os.path.exists(a.out):
        done = {(r["chunk_id"], r["style"]) for r in map(json.loads, open(a.out, encoding="utf-8"))}
    todo = [(c, s) for c, s in reqs if (c["chunk_id"], s) not in done]
    print(f"{len(todo)} rewrites to do ({len(done)} done)", flush=True)
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.92, seed=a.seed,
              limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=8192)
    tok = llm.get_tokenizer()

    def fit(text, n=4500):
        ids = tok.encode(text, add_special_tokens=False)
        return text if len(ids) <= n else tok.decode(ids[:n])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as out:
        for b in range(0, len(todo), a.batch):
            batch = todo[b:b + a.batch]
            t0 = time.time()
            msgs = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER.format(instruction=STYLES[s], text=fit(c["text"]))}]
                    for c, s in batch]
            sps = [SamplingParams(temperature=0.7, top_p=0.9, max_tokens=a.max_tokens, seed=a.seed + b + k)
                   for k in range(len(batch))]
            res = llm.chat(msgs, sps, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            ntok = trunc = 0
            for (c, s), r in zip(batch, res):
                o = r.outputs[0]
                ntok += len(o.token_ids); trunc += o.finish_reason == "length"
                out.write(json.dumps({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "style": s, "text": o.text,
                                      "finish": o.finish_reason, "n_tokens": len(o.token_ids)},
                                     ensure_ascii=False) + "\n")
            out.flush()
            dt = time.time() - t0
            print(f"[rw] {b + len(batch)}/{len(todo)} rewrites, {ntok} tokens in {dt:.0f}s ({ntok / dt:.0f} tok/s), "
                  f"truncated {trunc}", flush=True)


if __name__ == "__main__":
    main()
