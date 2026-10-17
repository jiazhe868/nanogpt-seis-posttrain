#!/usr/bin/env python3
"""Keep the RL prompts the current policy can learn from: solved sometimes, not always.

A GRPO group with all-correct or all-wrong rollouts has zero advantage and no gradient; in v8,
25-50% of groups were like that, and every prompt was seen once (audit §15.12). This samples
each candidate `k` times with the policy at the RL temperature, scores it with the v9 reward
(executed answer correct; program must run), and keeps prompts with 0 < passes < k -- the
DAPO "dynamic sampling" idea, done once up front.

    python tools/filter_learnable_prompts.py --ckpt checkpoints/sft_pot_v2/sft_ckpt_epoch4.pt \\
        --inp data/rl/candidates.jsonl --out data/rl/learnable.jsonl --k 8 --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.rollout import generate_rollouts  # noqa: E402
from src.rl.train_grpo import build_prompt_ids, load_sft_model, score_rollouts  # noqa: E402
from src.rl.verifiers import CompositeRewardEngine  # noqa: E402
from src.sft.train_sft import load_bpe_tokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch-prompts", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notation", action="store_true", help="the policy writes paper notation (pot_notation)")
    args = ap.parse_args()
    tok = load_bpe_tokenizer(ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json")
    model = load_sft_model(Path(args.ckpt), torch.device("cuda"))
    items = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    eng = CompositeRewardEngine()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    hist = [0] * (args.k + 1)
    with open(args.out, "w", encoding="utf-8") as g:
        for i in range(0, len(items), args.batch_prompts):
            chunk = items[i:i + args.batch_prompts]
            ids = [build_prompt_ids(tok, it["prompt"]) for it in chunk]
            comps, term, _ = generate_rollouts(model, ids, args.k, args.max_new_tokens, args.temperature, 1.0,
                                               generator=gen, tokenizer=tok, pot=True, pot_notation=args.notation)
            groups = score_rollouts(eng, tok, chunk, comps, term, args.k, "correctness", True, args.notation)
            for it, gr in zip(chunk, groups):
                passes = sum(s["total_reward"] >= 0.5 for s in gr)
                hist[passes] += 1
                g.write(json.dumps({**it, "passes": passes, "k": args.k}, ensure_ascii=False) + "\n")
            if (i // args.batch_prompts) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(items)} prompts | pass histogram {hist}", flush=True)
    print(f"pass-count histogram (0..{args.k}): {hist}; learnable (1..{args.k - 1}): {sum(hist[1:args.k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
