"""How far each metric is from its ceiling.

    numerical <ckpt> <name> [T] [n]   n samples per real held-out numerical question (executed paper-notation
                                      decoding at temperature T, no top-k, rp 1.0): per-question solve counts,
                                      for pass@k curves and the share of questions never solved
    concept <ckpt> <name> <decoder> [items limit]   2 samples per non-numerical prompt (default: the 641 held-out), as `prose`
                                      (T 0.3, top-k 40, rp 1.2) or `deployed` (configs/deploy.yaml decoding);
                                      key-concept coverage of the visible answer, as KeyConceptVerifier matches it

Writes work/experiments/ceiling_<kind>_<name>.json. Run from the repository root on one GPU.
"""
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl.rollout import decode_completion, generate  # noqa: E402
from src.rl.train_grpo import load_sft_model  # noqa: E402
from src.rl.verifiers import KeyConceptVerifier, NumericalVerifier  # noqa: E402
from src.sft.chat import format_chatml_token_stream  # noqa: E402
from src.sft.dataset import DEFAULT_SYSTEM_PROMPT, IM_END_ID, load_bpe_tokenizer  # noqa: E402

WORK = Path(os.environ.get("WORK", ROOT / "work" / "experiments"))
TOK = ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json"


def setup(ckpt):
    dev = torch.device("cuda")
    return load_sft_model(ROOT / ckpt, dev).eval(), load_bpe_tokenizer(TOK), dev


def visible(text):
    return re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)


def numerical(ckpt, name, T=0.8, n=64, per_call=2):
    model, tok, dev = setup(ckpt)
    nv = NumericalVerifier()
    items = [json.loads(l) for l in open(ROOT / "data" / "rl" / "test.jsonl", encoding="utf-8")]
    items = [it for it in items if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
    gen = torch.Generator(device=dev).manual_seed(7)
    out = {}
    for i in range(0, len(items), per_call):
        chunk = items[i:i + per_call]
        ids = [format_chatml_token_stream(tok, [], it["prompt"], DEFAULT_SYSTEM_PROMPT) for it in chunk]
        comps, _ = generate(model, ids, n, 384, temperature=T, top_k=None, repetition_penalty=1.0,
                            stop_ids=(IM_END_ID, 0), generator=gen, tokenizer=tok, pot=True, pot_notation=True)
        for j, it in enumerate(chunk):
            t = float(it["final_numerical_value"]); den = abs(t) if abs(t) > 1e-7 else 1.0
            c = 0
            for comp in comps[j * n:(j + 1) * n]:
                nums = nv.extract_numbers(visible(decode_completion(tok, comp, (IM_END_ID, 0))))
                c += bool(nums) and min(abs(v - t) / den for v in nums[-nv.ANSWER_TAIL_NUMBERS:]) <= 0.05
            out[it["id"]] = c
        print(f"{i + len(chunk)}/{len(items)}", flush=True)
    json.dump({"ckpt": ckpt, "temperature": T, "n": n, "correct": out}, open(WORK / f"ceiling_numerical_{name}.json", "w"))


def concept(ckpt, name, decoder, items_path=None, n=2, per_call=32, limit=0):
    model, tok, dev = setup(ckpt)
    kv = KeyConceptVerifier()
    dec = {"prose": dict(temperature=0.3, top_k=40, repetition_penalty=1.2, pot=False, pot_notation=False),
           "deployed": dict(temperature=0.3, top_k=40, repetition_penalty=1.0, pot=True, pot_notation=True)}[decoder]
    items = [json.loads(l) for l in open(items_path or ROOT / "data" / "rl" / "test.jsonl", encoding="utf-8")]
    items = [it for it in items if it["type"] != "numerical" and it.get("key_concepts")]
    if limit:
        import random
        random.Random(0).shuffle(items); items = items[:limit]
    gen = torch.Generator(device=dev).manual_seed(7)
    rows = []
    for i in range(0, len(items), per_call):
        chunk = items[i:i + per_call]
        ids = [format_chatml_token_stream(tok, [], it["prompt"], DEFAULT_SYSTEM_PROMPT) for it in chunk]
        comps, _ = generate(model, ids, n, 384, stop_ids=(IM_END_ID, 0), generator=gen, tokenizer=tok, **dec)
        for j, it in enumerate(chunk):
            for comp in comps[j * n:(j + 1) * n]:
                text = decode_completion(tok, comp, (IM_END_ID, 0))
                vis = visible(text).lower()
                rows.append({"id": it["id"], "type": it["type"],
                             "coverage": sum(k.lower() in vis for k in it["key_concepts"]) / len(it["key_concepts"]),
                             "verifier": kv(text, it["key_concepts"]), "words": len(vis.split()),
                             "think": "</think>" in text, "example": text[:400] if len(rows) < 6 else None})
        print(f"{i + len(chunk)}/{len(items)}", flush=True)
    json.dump({"ckpt": ckpt, "decoder": decoder, "rows": rows}, open(WORK / f"ceiling_concept_{name}.json", "w"))


if __name__ == "__main__":
    kind, ckpt, name = sys.argv[1:4]
    if kind == "numerical":
        numerical(ckpt, name, *(float(a) if k == 0 else int(a) for k, a in enumerate(sys.argv[4:6])))
    else:
        concept(ckpt, name, sys.argv[4], *(sys.argv[5:6]), limit=int(sys.argv[6]) if len(sys.argv) > 6 else 0)
