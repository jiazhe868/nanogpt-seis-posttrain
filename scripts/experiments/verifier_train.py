"""Train the outcome verifier (src/rl/verifier.py) on labelled samples from verifier_samples.py.

    python scripts/experiments/verifier_train.py <out_dir> <samples.jsonl> [<samples.jsonl> ...]

Starts from the deployed checkpoint. Binary cross-entropy on logit(yes) - logit(no) (VERIFIER_LOSS=pairwise
adds a ranking term between correct and wrong samples of the same problem); 5% of the problems
(by id hash) are held out to monitor loss, AUC, and how often the top-scored sample of a problem is
correct. Saves verifier_epoch{N}.pt in the SFT checkpoint format.
"""
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl.train_grpo import load_sft_model  # noqa: E402
from src.rl.verifier import batch_scores, verifier_ids, yes_no_ids  # noqa: E402
from src.sft.dataset import load_bpe_tokenizer  # noqa: E402

INIT = os.environ.get("VERIFIER_INIT", "checkpoints/grpo_v14_corpus/grpo_ckpt_step_75.pt")
TOK = ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json"
LR, EPOCHS, BATCH, WARMUP = float(os.environ.get("VERIFIER_LR", 1e-5)), int(os.environ.get("VERIFIER_EPOCHS", 2)), 16, 200
# "bce" scores each sample alone; "pairwise" adds a within-problem ranking term (correct above wrong
# samples of the same problem), since selection only compares samples of one question
LOSS = os.environ.get("VERIFIER_LOSS", "bce")


def held_out(pid):
    return int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % 20 == 0


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]; neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    allv = sorted((s, y) for s, y in zip(scores, labels))
    rank_sum = sum(r + 1 for r, (_, y) in enumerate(allv) if y)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def evaluate(model, val, yes_id, no_id, dev):
    if not val:
        return {}
    model.eval()
    scores = []
    for k in range(0, len(val), 64):
        scores += batch_scores(model, [r[0] for r in val[k:k + 64]], yes_id, no_id, dev).tolist()
    model.train()
    labels = [r[1] for r in val]
    loss = F.binary_cross_entropy_with_logits(torch.tensor(scores), torch.tensor(labels, dtype=torch.float)).item()
    by = {}
    for (_, y, pid), s in zip(val, scores):
        by.setdefault(pid, []).append((s, y))
    mixed = [v for v in by.values() if 0 < sum(y for _, y in v) < len(v)]
    top = sum(max(v)[1] for v in mixed) / max(1, len(mixed))
    base = sum(sum(y for _, y in v) / len(v) for v in mixed) / max(1, len(mixed))
    return {"loss": loss, "auc": auc(scores, labels), "top1_mixed": top, "random_mixed": base, "n_mixed": len(mixed)}


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    tok = load_bpe_tokenizer(TOK)
    yes_id, no_id = yes_no_ids(tok)
    rows = []
    for p in sys.argv[2:]:
        for line in open(p, encoding="utf-8"):
            s = json.loads(line)
            rows.append((verifier_ids(tok, s["q"], s["completion"]), float(s["correct"]), s["id"]))
    train = [r for r in rows if not held_out(r[2])]
    val = [r for r in rows if held_out(r[2])]
    print(f"{len(train)} train samples ({100 * sum(r[1] for r in train) / len(train):.1f}% correct), {len(val)} val", flush=True)
    dev = torch.device("cuda")
    model = load_sft_model(ROOT / INIT, dev).train()
    cfg = torch.load(ROOT / INIT, map_location="cpu", weights_only=False)["model_cfg"]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    steps = EPOCHS * (len(train) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    print("step 0", evaluate(model, val, yes_id, no_id, dev), flush=True)
    step, rng = 0, random.Random(0)
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(train)
        batches = []
        if LOSS == "pairwise":   # 4 problems x 4 of their samples per batch
            by = {}
            for r in train:
                by.setdefault(r[2], []).append(r)
            groups = [g[k:k + 4] for g in by.values() for k in range(0, len(g), 4)]
            rng.shuffle(groups)
            batches = [sum(groups[k:k + BATCH // 4], []) for k in range(0, len(groups), BATCH // 4)]
        else:                    # length-bucketed: sort within windows of 50 batches, then shuffle
            for w in range(0, len(train), BATCH * 50):
                win = sorted(train[w:w + BATCH * 50], key=lambda r: len(r[0]))
                batches += [win[b:b + BATCH] for b in range(0, len(win), BATCH)]
            rng.shuffle(batches)
        for b in batches:
            s = batch_scores(model, [r[0] for r in b], yes_id, no_id, dev)
            y = torch.tensor([r[1] for r in b], device=dev)
            loss = F.binary_cross_entropy_with_logits(s, y)
            if LOSS == "pairwise":
                pid = [r[2] for r in b]
                terms = []
                for p in set(pid):
                    m = torch.tensor([x == p for x in pid], device=dev)
                    pos, neg = s[m & (y > 0.5)], s[m & (y < 0.5)]
                    if len(pos) and len(neg):
                        terms.append(F.softplus(neg[None, :] - pos[:, None]).mean())
                if terms:
                    loss = loss + torch.stack(terms).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 1000 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f}", evaluate(model, val, yes_id, no_id, dev), flush=True)
        torch.save({"model": model.state_dict(), "model_cfg": cfg, "verifier": {"init": INIT, "lr": LR, "loss": LOSS, "epoch": epoch}},
                   out_dir / f"verifier_epoch{epoch}.pt")
        print(f"epoch {epoch} saved", evaluate(model, val, yes_id, no_id, dev), flush=True)


if __name__ == "__main__":
    main()
