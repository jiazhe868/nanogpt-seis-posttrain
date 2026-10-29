"""Select, decontaminate and tokenize the continued-pretraining sources into uint16 token streams.

    python -I scripts/midtrain/build_mix.py finemath fineweb_edu cosmopedia replay domain [--workers 40]

Each source becomes work/midtrain/tokenized/<name>.bin (documents separated by <|endoftext|>, id 0, the format of
../nanogpt_seis/src/tokenizer/encode.py), with its counts in work/midtrain/tokenized/meta.json.

finemath     FineMath-4+, whole files in order up to --finemath-tokens (default 4B of our tokens)
fineweb_edu  FineWeb-Edu 10BT sample, int_score >= 4
cosmopedia   Cosmopedia v2 rows on STEM topics: BISAC SCIENCE / MATHEMATICS / TECHNOLOGY & ENGINEERING / NATURE, or
             seeded from auto_math_text, stanford, openstax, khanacademy
replay       the original pretraining corpus (../nanogpt_seis/data/processed/train.jsonl) minus held-out papers
domain       its seismology documents (the rule of tools/gen_real_style.py) minus held-out papers, for upweighting
rewrites     the teaching rewrites kept by scripts/midtrain/filter_rewrites.py (work/midtrain/rewrites/kept.jsonl)

Held-out papers are the documents holding a test or validation excerpt (tools/gen_real_style.py). A public document
that shares a 13-word sequence with one of them, or with a held-out question, is dropped; sequences found in more
than one held-out paper (licence text, boilerplate) are ignored.
"""
import argparse
import glob
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.gen_real_style import CORPUS, DOMAIN_SOURCES, HELD_OUT, KEYWORDS, chunk_id  # noqa: E402
from src.sft.generate_sft_data import chunk_text  # noqa: E402

RAW, OUT = ROOT / "work/midtrain/raw", ROOT / "work/midtrain/tokenized"
REWRITES = ROOT / "work/midtrain/rewrites/kept.jsonl"
TOKENIZER = ROOT.parent / "nanogpt_seis/data/tokenized/tokenizer.json"
EOT, N = 0, 13
STEM = {"SCIENCE", "MATHEMATICS", "TECHNOLOGY & ENGINEERING", "NATURE"}
STEM_SEEDS = {"auto_math_text", "stanford", "openstax", "khanacademy"}
CATEGORY = re.compile(r'as part of "([^"]+)"')
WORD = re.compile(r"\w+")
HELD_GRAMS = set()          # filled in the parent before the workers fork


def grams(text):
    w = WORD.findall(text.lower())
    return {hash(tuple(w[i:i + N])) for i in range(len(w) - N + 1)}


def corpus_docs(lo=0, hi=None):
    """(id, source, text) of the corpus lines lo..hi, with the id rule of tools/gen_real_style.py."""
    for i, line in enumerate(open(CORPUS, encoding="utf-8")):
        if i < lo or (hi is not None and i >= hi):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = doc.get("text", doc.get("full_text", ""))
        if text:
            yield doc.get("id", i), doc.get("source", "processed"), text


def is_domain(src, text):
    """The pool rule of tools/gen_real_style.py, applied to whole documents."""
    if len(text) < 500 or src in ("general", "fineweb_edu", "wiki_general"):
        return False
    return src in DOMAIN_SOURCES or any(k in text[:2000].lower() for k in KEYWORDS)


def held_out():
    """Ids of the held-out papers and the 13-gram hashes of their text and of the held-out questions."""
    held = {json.loads(l).get("chunk_id") for p in HELD_OUT for l in open(ROOT / p, encoding="utf-8")}
    bad, texts = set(), {}
    for did, src, text in corpus_docs():
        if is_domain(src, text) and any(chunk_id(did, c, x) in held
                                        for c, x in enumerate(chunk_text(text, chunk_size=8000))):
            bad.add(did); texts[did] = text
    seen, dup = set(), set()
    for text in texts.values():
        g = grams(text)
        dup |= g & seen; seen |= g
    qs = [json.loads(l).get("prompt") or json.loads(l).get("instruction") or ""
          for p in HELD_OUT for l in open(ROOT / p, encoding="utf-8")]
    for q in qs:
        seen |= grams(q)
    return bad, seen - dup


def encode_texts(tok, texts):
    out = []
    for i in range(0, len(texts), 512):
        for e in tok.encode_batch(texts[i:i + 512]):
            out.extend(e.ids); out.append(EOT)
    return np.asarray(out, dtype=np.uint16)


def rows(job):
    """Texts of one input part after the source filter: (texts, rows read)."""
    name, path, lo, hi = job
    if name == "finemath":
        t = pq.read_table(path, columns=["text"]).column("text").to_pylist()
        return t, len(t)
    if name == "fineweb_edu":
        d = pq.read_table(path, columns=["text", "int_score"]).to_pydict()
        return [x for x, s in zip(d["text"], d["int_score"]) if s >= 4], len(d["text"])
    if name == "cosmopedia":
        d = pq.read_table(path, columns=["text", "prompt", "seed_data"]).to_pydict()
        keep = []
        for x, p, s in zip(d["text"], d["prompt"], d["seed_data"]):
            m = CATEGORY.search(p)
            if s in STEM_SEEDS or (m and m.group(1).split(" / ")[0] in STEM):
                keep.append(x)
        return keep, len(d["text"])
    if name == "rewrites":
        with open(path, encoding="utf-8") as f:
            t = [json.loads(l)["text"] for k, l in enumerate(f) if lo <= k < hi]
        return t, len(t)
    docs = list(corpus_docs(lo, hi))
    return [x for did, src, x in docs if did not in BAD and (name == "replay" or is_domain(src, x))], len(docs)


def work(job):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"          # one core per worker
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    texts, n_read = rows(job)
    public = job[0] in ("finemath", "fineweb_edu", "cosmopedia")
    kept = [x for x in texts if x.strip() and not (public and grams(x) & HELD_GRAMS)]
    arr = encode_texts(tok, kept)
    part = OUT / "parts" / job[0] / f"{job[2]:06d}_{Path(job[1]).stem}.bin"
    part.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(part)
    return job[0], n_read, len(texts), len(texts) - len(kept), len(kept), int(arr.size)


BAD = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--finemath-tokens", type=float, default=4e9)
    a = ap.parse_args()
    global BAD, HELD_GRAMS
    t0 = time.time()
    BAD, HELD_GRAMS = held_out()
    print(f"held-out papers {len(BAD)}, 13-gram hashes {len(HELD_GRAMS):,} ({time.time() - t0:.0f} s)", flush=True)

    jobs = []
    for name in a.sources:
        if name == "finemath":
            files = sorted(glob.glob(str(RAW / "finemath/finemath-4plus/*.parquet")))
            chars = [sum(pq.read_table(f, columns=["char_count"]).column("char_count").to_pylist()) for f in files[:4]]
            k = int(np.ceil(a.finemath_tokens * 2.99 / (np.mean(chars) or 1)))     # 2.99 bytes per our token
            jobs += [(name, f, i, 0) for i, f in enumerate(files[:k])]
        elif name == "fineweb_edu":
            jobs += [(name, f, i, 0) for i, f in enumerate(sorted(glob.glob(str(RAW / "fineweb_edu/sample/10BT/*.parquet"))))]
        elif name == "cosmopedia":
            jobs += [(name, f, i, 0) for i, f in enumerate(sorted(glob.glob(str(RAW / "cosmopedia/cosmopedia-v2/*.parquet"))))]
        else:
            path = REWRITES if name == "rewrites" else CORPUS
            n = sum(1 for _ in open(path, encoding="utf-8")); step = -(-n // a.workers)
            jobs += [(name, str(path), lo, lo + step) for lo in range(0, n, step)]

    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    stats = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for name, n_read, n_sel, n_drop, n_kept, n_tok in pool.imap_unordered(work, jobs):
            s = stats.setdefault(name, [0, 0, 0, 0, 0])
            for i, v in enumerate((n_read, n_sel, n_drop, n_kept, n_tok)):
                s[i] += v
            print(f"  {name}: {s[4] / 1e9:.2f}B tokens so far ({time.time() - t0:.0f} s)", flush=True)
    for name in a.sources:
        parts = sorted((OUT / "parts" / name).glob("*.bin"))
        with open(OUT / f"{name}.bin", "wb") as f:
            for p in parts:
                f.write(p.read_bytes()); p.unlink()
        n_read, n_sel, n_drop, n_kept, n_tok = stats[name]
        meta[name] = {"rows_read": n_read, "selected": n_sel, "dropped_heldout_overlap": n_drop, "documents": n_kept,
                      "tokens": n_tok}
        print(f"{name}: {n_kept:,} documents ({n_drop:,} dropped for held-out overlap), {n_tok / 1e9:.3f}B tokens")
    meta["held_out_papers"] = len(BAD)
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
