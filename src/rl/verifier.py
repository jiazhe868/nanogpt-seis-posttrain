"""Outcome verifier for numerical answers: pick the best of k samples instead of the most common one.

A second copy of the 113M model reads the question, one sampled answer and a follow-up user turn,
"Is this answer correct? Reply yes or no."; its score is logit("yes") - logit("no") for the next
token. It is trained on the deployed model's own samples, each labelled by executing it against the
problem's answer (scripts/experiments/verifier_*.py).

Selection over k samples:
  best      the sample with the highest score
  weighted  answers that agree within 5% form a cluster; the cluster with the largest summed
            sigmoid(score) wins, and its highest-scoring sample is returned
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch

from src.rl.self_consistency import agree
from src.sft.chat import format_chatml_token_stream

VERIFY_TURN = "Is this answer correct? Reply yes or no."
MAX_COMPLETION_TOKENS = 600


def verifier_ids(tokenizer, question: str, completion: str) -> List[int]:
    """ChatML tokens: system, the question, the sampled answer, the verify turn, assistant header."""
    ids = tokenizer.encode(completion.strip(), add_special_tokens=False).ids
    if len(ids) > MAX_COMPLETION_TOKENS:          # keep the end, where the executed answer is
        completion = tokenizer.decode(ids[-MAX_COMPLETION_TOKENS:])
    return format_chatml_token_stream(tokenizer, [(question, completion)], VERIFY_TURN)


def yes_no_ids(tokenizer) -> Tuple[int, int]:
    yes, no = tokenizer.encode("yes").ids, tokenizer.encode("no").ids
    assert len(yes) == 1 and len(no) == 1, "the verdict tokens must be single tokens"
    return yes[0], no[0]


def last_logits(model, idx: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """Next-token logits at position `last[b]` of each right-padded row, without the full-sequence
    output projection (the GPT forward of src/model/gqa_gpt.py up to the final norm)."""
    T = idx.shape[1]
    x = model.drop(model.tok_emb(idx))
    cos, sin = model.rope_cos[:, :, :T], model.rope_sin[:, :, :T]
    for blk in model.blocks:
        x, _ = blk(x, cos, sin)
    x = model.norm(x)[torch.arange(idx.shape[0], device=idx.device), last]
    logits = model.lm_head(x)
    if model.cfg.mup:
        logits = logits * model.output_mult.to(logits.dtype)
    return logits


def batch_scores(model, rows: Sequence[List[int]], yes_id: int, no_id: int, device) -> torch.Tensor:
    """logit(yes) - logit(no) for a batch of token rows (differentiable)."""
    T = max(len(r) for r in rows)
    idx = torch.zeros((len(rows), T), dtype=torch.long, device=device)
    for b, r in enumerate(rows):
        idx[b, :len(r)] = torch.tensor(r, device=device)
    last = torch.tensor([len(r) - 1 for r in rows], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = last_logits(model, idx, last).float()
    return logits[:, yes_id] - logits[:, no_id]


@torch.no_grad()
def score(model, tokenizer, pairs: Sequence[Tuple[str, str]], device, batch: int = 32) -> List[float]:
    """Verifier scores for (question, completion) pairs, longest first to limit padding."""
    yes_id, no_id = yes_no_ids(tokenizer)
    rows = [verifier_ids(tokenizer, q, c) for q, c in pairs]
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
    out = [0.0] * len(rows)
    for k in range(0, len(order), batch):
        chunk = order[k:k + batch]
        for i, s in zip(chunk, batch_scores(model, [rows[i] for i in chunk], yes_id, no_id, device).tolist()):
            out[i] = s
    return out


def select(tails: Sequence[Sequence[float]], scores: Sequence[float], rule: str = "weighted", tol: float = 0.05) -> int:
    """Index of the chosen sample among k (no access to the target)."""
    if rule == "best":
        return max(range(len(scores)), key=lambda i: (scores[i], -i))
    p = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    mass = [sum(p[j] for j in range(len(tails)) if j == i or agree(tails[i], tails[j], tol)) for i in range(len(tails))]
    return max(range(len(tails)), key=lambda i: (mass[i], scores[i], -i))
