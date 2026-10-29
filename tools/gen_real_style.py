#!/usr/bin/env python3
"""Real-style questions from the rest of the pretraining corpus, written by the local teacher.

The held-out questions were written by the Phase II generator (src/sft/generate_sft_data.py) from paper
excerpts, and the model answers a new question well when it has practised a similar one. Phase II used
about 5,000 of the corpus's ~138,000 domain excerpts. This tool applies the same generator prompt and
chunking to the others, asking for 3 numerical pairs and 1 conceptual pair per excerpt, with shorter answers,
in a plain-text output format (JSON with LaTeX and nested <think> tags failed for 30% of excerpts).

    pool      chunk every domain document of ../nanogpt_seis/data/processed/train.jsonl as Phase II did;
              drop every document that contains an excerpt of the test or validation split (paper-level
              hold-out) and every excerpt that names a held-out historical earthquake (its year and place);
              excerpts Phase II already used go last
    generate  excerpts [--start, --start + --count) of the pool -> pairs; numerical pairs get the test set's
              label rule (src.rl.numerical_target); near-copies of held-out questions are dropped

    python tools/gen_real_style.py pool --out data/sources/real_style/pool.jsonl
    VLLM_USE_FLASHINFER_SAMPLER=0 python tools/gen_real_style.py generate --pool data/sources/real_style/pool.jsonl \\
        --start 0 --count 30000 --out data/sources/real_style/pairs_000.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sft.generate_sft_data import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, chunk_text  # noqa: E402

CORPUS = ROOT.parent / "nanogpt_seis" / "data" / "processed" / "train.jsonl"
DOMAIN_SOURCES = {"fulltext", "arxiv", "eartharxiv", "openalex", "substack", "journals", "preprints"}
KEYWORDS = {"earthquake", "seismic", "seismology", "fault", "tectonic", "tsunami", "seismometer", "epicenter",
            "hypocenter", "subduction", "lithosphere", "attenuation", "ground motion", "shear wave", "p-wave",
            "s-wave", "magnitude"}
HELD_OUT = ("data/rl/test.jsonl", "data/splits/val.jsonl")          # their documents are never used
PHASE2 = ("data/splits/train.jsonl",)                                # excerpts Phase II already used

# The Phase II prompt with two changes: the pair mix (3 numerical + 1 conceptual) and shorter answers.
SYSTEM = SYSTEM_PROMPT.replace(
    "1. Generate EXACTLY 2 or 3 distinct, high-rigor instruction-response pairs based on the text.",
    "1. Generate EXACTLY 4 distinct, high-rigor instruction-response pairs based on the text: 3 of Type B and "
    "1 of Type A or Type C.").replace(
    "5. Response Depth & Thoroughness: Do NOT produce superficial 1-sentence answers. Provide complete, academic, "
    "multi-paragraph explanations (up to 1,500-2,000 tokens) that thoroughly cover the scientific reasoning.",
    "5. Response Depth: Type B responses give the complete derivation inside <think>...</think>, then the final "
    "answer with its unit (about 300-600 tokens). Type A and C responses are complete, academic explanations of "
    "about 300-500 words.")
SYSTEM = SYSTEM.replace("3. Diverse & Deep Question Taxonomy:",
                        "3. Diverse & Deep Question Taxonomy (exactly 3 of the 4 pairs are Type B):")
# The teacher's own reasoning tags are <think>...</think>, and inside its answer they can end the turn early;
# LaTeX also breaks JSON. So the derivation goes in <derivation> tags and the pairs in a plain-text format.
SYSTEM = SYSTEM.replace("<think>...</think>", "<derivation>...</derivation>")
SYSTEM = SYSTEM[:SYSTEM.index("OUTPUT FORMAT:")] + """OUTPUT FORMAT:
Plain text only (no JSON, no markdown fences). Write the 4 pairs like this, numerical pairs first:
=== PAIR 1 | numerical ===
QUESTION:
<the user's question>
RESPONSE:
<derivation>step-by-step derivation (under 400 tokens)</derivation>
Final answer: <value and unit>
=== PAIR 2 | numerical ===
...
=== PAIR 4 | conceptual ===
QUESTION:
<the user's question>
RESPONSE:
<expert answer>"""
assert "exactly 3 of the 4" in SYSTEM and "<think>" not in SYSTEM, "the Phase II prompt changed: update the replacements"
# The Phase II user turn asks for "2 or 3 pairs" as JSON, which overrides the mix and the format above.
USER = USER_PROMPT_TEMPLATE.split("Generate 2 or 3")[0] + (
    "Generate the 4 grounded, high-rigor instruction-response pairs (3 numerical, 1 conceptual) based on the above "
    "text, in the plain-text format described in the instructions.")
assert USER.startswith("TEXT EXCERPT:") and "{text_chunk}" in USER
PAIR = re.compile(r"^=== PAIR \d+ \| (\w+) ===\s*$", flags=re.MULTILINE)


def parse_pairs(text):
    """Pairs from the plain-text format; a numerical response is re-wrapped as <think>derivation</think>final."""
    parts = PAIR.split(text or "")
    pairs = []
    for typ, body in zip(parts[1::2], parts[2::2]):
        m = re.search(r"QUESTION:\s*(.*?)\s*RESPONSE:\s*(.*)", body, flags=re.DOTALL)
        if not m:
            continue
        q, resp = m.group(1).strip(), m.group(2).strip()
        d = re.search(r"<derivation>(.*?)</derivation>", resp, flags=re.DOTALL)
        if d:
            resp = f"<think>{d.group(1).strip()}</think>{resp[d.end():].strip()}"
        pairs.append({"type": typ.lower(), "instruction": q, "response": resp})
    return pairs


def chunk_id(doc_id, c_idx, text):
    return hashlib.md5(f"{doc_id}_{c_idx}_{text[:50]}".encode()).hexdigest()


def corpus_chunks():
    """(doc_id, source, title, chunk_id, text) with the filters and id rule of the Phase II generator."""
    for i, line in enumerate(open(CORPUS, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = doc.get("source", "processed")
        text = doc.get("text", doc.get("full_text", ""))
        if not text or len(text) < 500 or src in ("general", "fineweb_edu", "wiki_general"):
            continue
        if src not in DOMAIN_SOURCES and not any(k in text[:2000].lower() for k in KEYWORDS):
            continue
        did = doc.get("id", i)
        for c_idx, c in enumerate(chunk_text(text, chunk_size=8000)):
            yield did, src, doc.get("title", ""), chunk_id(did, c_idx, c), c


def held_out_events(held):
    """(year, place) of the historical earthquakes behind held-out questions (data/sources/earthquakes.jsonl)."""
    out = set()
    for line in open(ROOT / "data/sources/earthquakes.jsonl", encoding="utf-8"):
        r = json.loads(line)
        m = re.match(r"(\d{4})\s+(.+?)\s+earthquake", r.get("event", ""))
        if r.get("chunk_id") in held and m:           # e.g. "1975 Kalapana, Hawaii": Kalapana, Hawaii (over-matching is safe)
            for part in re.split(r"[,-]", m.group(2)):
                word = (re.sub(r"[^\w ]", "", part).split() or [""])[0]
                if len(word) >= 3:
                    out.add((m.group(1), word))
    return out


def build_pool(out):
    ids = lambda paths: {json.loads(l).get("chunk_id") for p in paths for l in open(ROOT / p, encoding="utf-8")}
    held, used = ids(HELD_OUT), ids(PHASE2)
    events = held_out_events(held)
    rows = list(corpus_chunks())
    bad_docs = {d for d, _, _, c, _ in rows if c in held}
    found = sum(c in held for *_, c, _ in rows)
    names = lambda x: any(y in x and place in x for y, place in events)
    keep = [r for r in rows if r[0] not in bad_docs and not names(r[4])]
    print(f"held-out events {sorted(events)}: {sum(names(r[4]) for r in rows if r[0] not in bad_docs)} excerpts dropped")
    random.Random(2026).shuffle(keep)
    keep.sort(key=lambda r: r[3] in used)                    # stable: unused excerpts first, shuffled
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for d, s, t, c, x in keep:
            f.write(json.dumps({"chunk_id": c, "doc_id": d, "source": s, "title": t, "used_by_phase2": c in used,
                                "text": x}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} domain excerpts; held-out excerpts found {found} of {len(held)} -> {len(bad_docs)} documents "
          f"dropped; pool {len(keep)} ({sum(r[3] in used for r in keep)} already used by Phase II) -> {out}")


def generate(a):
    from vllm import LLM, SamplingParams
    from src.rl.numerical_target import derive_numerical_target
    from tools.build_rl_pool import NearestNeighbour
    from tools.gen_pot_problems import heldout_ngrams, ngrams
    pool = []
    with open(a.pool, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k >= a.start + a.count:
                break
            if k >= a.start:
                pool.append(json.loads(line))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["chunk_id"] for l in open(a.out, encoding="utf-8")}
    todo = [c for c in pool if c["chunk_id"] not in done]
    held = [json.loads(l)["prompt"] for p in ("data/rl/test.jsonl", "data/rl/dev_clean.jsonl", "data/eval/heldout_large.jsonl")
            if (ROOT / p).exists() for l in open(ROOT / p, encoding="utf-8")]
    grams, nn = heldout_ngrams() | {g for q in held for g in ngrams(q)}, NearestNeighbour(held)
    print(f"{len(todo)} excerpts to do ({len(done)} done)", flush=True)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=8192, gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=a.seed, max_num_seqs=a.max_num_seqs,
              max_num_batched_tokens=4096)
    tok = llm.get_tokenizer()

    def fit(text, n=4000):   # a single long paragraph can make an excerpt longer than the context
        ids = tok.encode(text, add_special_tokens=False)
        return text if len(ids) <= n else tok.decode(ids[:n])
    stats = {}
    with open(a.out, "a", encoding="utf-8") as out:
        for b in range(0, len(todo), a.batch):
            batch = todo[b:b + a.batch]
            t0 = time.time()
            msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(text_chunk=fit(c["text"]))}]
                    for c in batch]
            # each request gets the rest of the context as its output budget
            budgets = [min(a.max_tokens, 8192 - 32 - len(tok.encode(tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))) for m in msgs]
            sps = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max(512, n), seed=a.seed + b + k)  # Phase II: T 0.7
                   for k, n in enumerate(budgets)]
            res = llm.chat(msgs, sps, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            ntok = 0
            for c, r in zip(batch, res):
                ntok += len(r.outputs[0].token_ids)
                pairs = parse_pairs(r.outputs[0].text) or []
                kept = []
                for p in pairs:
                    q, resp, typ = (p.get("instruction") or "").strip(), (p.get("response") or "").strip(), p.get("type", "")
                    if not q or not resp:
                        continue
                    if ngrams(q) & grams or nn.best(q) >= 0.5:
                        stats["near_heldout"] = stats.get("near_heldout", 0) + 1
                        continue
                    row = {"type": typ, "instruction": q, "response": resp}
                    if typ == "numerical":
                        m = re.search(r"<think>(.*?)</think>", resp, flags=re.DOTALL)
                        think, final = (m.group(1), resp[m.end():]) if m else ("", resp)
                        row["label"], row["label_source"] = derive_numerical_target(q, think, final)
                        stats["numerical"] = stats.get("numerical", 0) + 1
                        stats["labelled"] = stats.get("labelled", 0) + (row["label"] is not None)
                    else:
                        stats["other"] = stats.get("other", 0) + 1
                    kept.append(row)
                stats["excerpts"] = stats.get("excerpts", 0) + 1
                stats["unparsed"] = stats.get("unparsed", 0) + (not pairs)
                stats["truncated"] = stats.get("truncated", 0) + (r.outputs[0].finish_reason == "length")
                rec = {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "source": c["source"], "pairs": kept}
                if not pairs:
                    rec["raw"] = r.outputs[0].text[:4000]
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[gen] {b + len(batch)}/{len(todo)} excerpts, {ntok} tokens in {time.time() - t0:.0f}s "
                  f"({ntok / max(1e-9, time.time() - t0):.0f} tok/s) | {json.dumps(stats)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["pool", "generate"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/sources/real_style/pool.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2000, help="excerpts per vLLM call (results saved after each)")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=5000, help="upper bound; each request also stays within the context")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=64, help="concurrent sequences (the GDN state is per sequence)")
    a = ap.parse_args()
    if a.mode == "pool":
        build_pool(a.out)
    else:
        generate(a)


if __name__ == "__main__":
    main()
