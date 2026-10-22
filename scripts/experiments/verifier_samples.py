"""Labelled samples of the deployed model, for training and evaluating the verifier (src/rl/verifier.py).

    python scripts/experiments/verifier_samples.py <items.jsonl> <out.jsonl> <n per item> <T> [seed] [ckpt]

Decoding as served: executed paper notation, top-k 40, rp 1.0, 384 new tokens. Each output line has the
item id, the question, the completion, its scored numbers (`tail`, the last 3 numbers after </think>)
and `correct` (a tail number within 5% of the label: the src/rl/eval_rl.py rule).
"""
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl.rollout import decode_completion, generate  # noqa: E402
from src.rl.train_grpo import load_sft_model  # noqa: E402
from src.rl.verifiers import NumericalVerifier  # noqa: E402
from src.sft.chat import format_chatml_token_stream  # noqa: E402
from src.sft.dataset import DEFAULT_SYSTEM_PROMPT, IM_END_ID, load_bpe_tokenizer  # noqa: E402

DEPLOYED = "checkpoints/grpo_v14_corpus/grpo_ckpt_step_75.pt"
TOK = ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json"


def label(nv, completion, target):
    nums = nv.extract_numbers(re.sub(r"^.*?</think>", "", completion, flags=re.DOTALL))
    tail = nums[-nv.ANSWER_TAIL_NUMBERS:]
    den = abs(target) if abs(target) > 1e-7 else 1.0
    return tail, bool(tail) and min(abs(v - target) / den for v in tail) <= 0.05


def main():
    items_path, out_path, n, T = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    ckpt = sys.argv[6] if len(sys.argv) > 6 else DEPLOYED
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    items = [it for it in items if it.get("type", "numerical") == "numerical" and it.get("final_numerical_value") is not None]
    dev = torch.device("cuda")
    model = load_sft_model(ROOT / ckpt, dev).eval()
    tok = load_bpe_tokenizer(TOK)
    nv = NumericalVerifier()
    gen = torch.Generator(device=dev).manual_seed(seed)
    per_call = max(1, 128 // n)
    with open(out_path, "w", encoding="utf-8") as f:
        for k in range(0, len(items), per_call):
            chunk = items[k:k + per_call]
            ids = [format_chatml_token_stream(tok, [], it["prompt"], DEFAULT_SYSTEM_PROMPT) for it in chunk]
            comps, _ = generate(model, ids, n, 384, temperature=T, top_k=40, repetition_penalty=1.0,
                                stop_ids=(IM_END_ID, 0), generator=gen, tokenizer=tok, pot=True, pot_notation=True)
            for j, it in enumerate(chunk):
                for c in comps[j * n:(j + 1) * n]:
                    text = decode_completion(tok, c, (IM_END_ID, 0))
                    tail, ok = label(nv, text, float(it["final_numerical_value"]))
                    f.write(json.dumps({"id": it["id"], "T": T, "q": it["prompt"], "completion": text,
                                        "tail": tail, "correct": ok}) + "\n")
            if (k // per_call) % 20 == 0:
                print(f"{k + len(chunk)}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
