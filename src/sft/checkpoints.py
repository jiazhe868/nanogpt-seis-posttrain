#!/usr/bin/env python3
"""Everything you do to a checkpoint after training it.

    select   rank candidates on measured BEHAVIOUR, not validation loss, and
             copy the winner to a stable path
    compare  score checkpoints on one shared held-out set, per-pair and paired
    strip    drop AdamW state from archived checkpoints (1.36 GB -> 453 MB)

Why behaviour and not loss: README section 4.7 and PITFALLS E1. Why per-pair
and paired: PITFALLS B2.

    python -m src.sft.checkpoints select --ckpt-dir checkpoints/sft \\
        --link-best checkpoints/sft/selected.pt
    python -m src.sft.checkpoints compare --test data/eval/heldout_subsets.jsonl \\
        --unpacked --ckpt sft=checkpoints/sft/selected.pt \\
        --ckpt grpo=checkpoints/grpo/selected.pt
    python -m src.sft.checkpoints strip checkpoints/ --apply --skip-last
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NANOGPT_SEIS_ROOT = ROOT.parent / "nanogpt_seis"

# ===========================================================================
# select
# ===========================================================================
# The selection rule is declared here, in source, so it cannot be tuned after
# seeing the table.
#   primary   numerical accuracy at 5% tolerance -- the only available scored
#             correctness proxy. It covers only the numerical subset; conceptual
#             factuality (the majority of held-out use) has no metric at all.
#             5% rather than 1% because it is less noisy at n~170.
#   tie-break <think> rate on numerical prompts (does the model show its work),
#             consulted only within TIE_EPS of the leader.
#   gates     disqualify outright rather than trading off against the primary.
PRIMARY = "numerical_acc_5pct"   # the only available scored correctness proxy
TIE_BREAK = "numerical_think_rate"
TIE_EPS = 0.02          # 2 percentage points ~ noise at this sample size

GATES = [
    # (metric, comparison, threshold, description)
    ("contamination_rate", "<=", 0.02, "phantom-source lead-ins in <=2% of answers"),
    ("echo_rate", "<=", 0.40, "prompt regurgitation in <=40% of answers"),
    ("mean_completion_tokens", ">=", 40, "answers not collapsed to near-empty"),
]


def failed_gates(metrics: Dict[str, float]) -> List[str]:
    """Descriptions of the gates this checkpoint fails (empty means it passes)."""
    failures = []
    for key, comparison, threshold, description in GATES:
        value = metrics.get(key, 0.0)
        ok = (value <= threshold) if comparison == "<=" else (value >= threshold)
        if not ok:
            failures.append(f"{description} (got {value:.3f})")
    return failures


def run_select(args: argparse.Namespace) -> int:
    from src.rl.eval_rl import evaluate_checkpoint

    candidates: List[Path] = [Path(c) for c in args.ckpt]
    if args.ckpt_dir:
        candidates += sorted(Path(args.ckpt_dir).glob("sft_ckpt_*.pt"))

    seen, ordered = set(), []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            ordered.append(path)
        else:
            print(f"[skip] missing: {path}")
    if not ordered:
        print("[error] no candidate checkpoints found", file=sys.stderr)
        return 2

    with open(args.test, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    numerical = [it for it in all_items
                 if it["type"] == "numerical" and it.get("final_numerical_value") is not None]
    conceptual = [it for it in all_items if it["type"] != "numerical"][: args.num_conceptual]
    items = numerical + conceptual
    print(f"[info] {len(ordered)} candidates | held-out set: {len(numerical)} numerical "
          f"+ {len(conceptual)} conceptual = {len(items)} prompts")
    print(f"[info] primary = {PRIMARY}, tie-break = {TIE_BREAK} (eps {TIE_EPS})")

    results = {}
    for path in ordered:
        print(f"\n[eval] {path}")
        results[str(path)] = evaluate_checkpoint(
            path, items, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty, seed=args.seed)

    print("\n" + "=" * 100)
    print("CHECKPOINT SELECTION — measured behaviour on held-out data")
    print("=" * 100)
    header = (f"{'checkpoint':<34} {'num@5%':>8} {'num@1%':>8} {'think':>8} "
              f"{'contam':>8} {'echo':>8} {'len':>6}  gate")
    print(header)
    print("-" * len(header))
    ranked = []
    for path, metrics in results.items():
        failures = failed_gates(metrics)
        print(f"{Path(path).name[:33]:<34} "
              f"{metrics['numerical_acc_5pct']:>7.1%} {metrics['numerical_acc_1pct']:>8.1%} "
              f"{metrics['numerical_think_rate']:>8.1%} {metrics['contamination_rate']:>8.1%} "
              f"{metrics['echo_rate']:>8.1%} {metrics['mean_completion_tokens']:>6.0f}  "
              f"{'PASS' if not failures else 'FAIL: ' + failures[0]}")
        if not failures:
            ranked.append((path, metrics))

    if not ranked:
        print("\n[error] every candidate failed a gate — inspect the table above.")
        return 1

    ranked.sort(key=lambda kv: -kv[1][PRIMARY])
    leader = ranked[0][1][PRIMARY]
    tied = [kv for kv in ranked if leader - kv[1][PRIMARY] <= TIE_EPS]
    tied.sort(key=lambda kv: -kv[1][TIE_BREAK])
    winner_path, winner_metrics = tied[0]

    print("=" * 100)
    print(f"SELECTED: {winner_path}")
    print(f"  {PRIMARY} = {winner_metrics[PRIMARY]:.1%}"
          + (f"  (tied within {TIE_EPS:.0%} with {len(tied) - 1} other(s); "
             f"broke tie on {TIE_BREAK} = {winner_metrics[TIE_BREAK]:.1%})" if len(tied) > 1 else ""))
    print("  Validation loss was deliberately NOT used. To override the automatic pick,")
    print("  the full table above is the evidence.")
    print("=" * 100)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"selected": winner_path, "primary": PRIMARY,
                   "tie_break": TIE_BREAK, "results": results}, f, indent=2)
    print(f"[info] full results -> {out_path}")

    if args.link_best:
        shutil.copy(winner_path, args.link_best)
        print(f"[info] copied winner -> {args.link_best}")
    return 0


# ===========================================================================
# compare
# ===========================================================================
# Packed evaluation concatenates pairs into 4096-token blocks under a plain
# causal mask, so every pair is conditioned on whatever happened to precede it
# in its block. Change the file, change the packing, change the number -- two
# nested test sets gave opposite signs for the same two models. Scoring each
# pair in isolation removes that coupling and makes a PAIRED test possible, so
# per-pair difficulty cancels.


@torch.no_grad()
def per_pair_losses(model, jsonl_path, tokenizer, device, amp, max_seq_len=4096,
                    subset=None) -> List[float]:
    """Mean per-token NLL for each pair, scored alone (no packing)."""
    import torch.nn.functional as F

    from src.sft.dataset import IGNORE_INDEX, encode_chatml_sample

    losses = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if subset and subset not in record.get("subsets", []):
                continue
            ids, labels = encode_chatml_sample(
                tokenizer, record["instruction"], record["response"])
            ids, labels = ids[:max_seq_len], labels[:max_seq_len]
            inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            if (targets != IGNORE_INDEX).sum() == 0:
                continue
            with amp:
                logits, _ = model(inputs, targets, ignore_index=IGNORE_INDEX)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="mean")
            losses.append(loss.item())
    return losses


def paired_report(name_a: str, losses_a: List[float],
                  name_b: str, losses_b: List[float]) -> None:
    """Paired comparison over per-pair losses, with a bootstrap CI."""
    n = len(losses_a)
    diffs = [b - a for a, b in zip(losses_a, losses_b)]
    mean_diff = sum(diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_err = (variance / n) ** 0.5
    t_stat = mean_diff / std_err if std_err > 0 else 0.0

    rng = random.Random(0)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    low, high = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    print(f"\nPaired over {n} pairs, unpacked (each pair scored in isolation):")
    print(f"  mean {name_a} = {sum(losses_a) / n:.4f}   mean {name_b} = {sum(losses_b) / n:.4f}")
    print(f"  mean paired diff ({name_b} - {name_a}) = {mean_diff:+.4f} nats")
    print(f"  95% bootstrap CI [{low:+.4f}, {high:+.4f}]   t = {t_stat:+.2f}")
    print(f"  -> {'significant' if (low > 0 or high < 0) else 'NOT significant (CI spans zero)'}")


def run_compare(args: argparse.Namespace) -> int:
    from torch.utils.data import DataLoader

    from src.sft.dataset import (IGNORE_INDEX, IM_END_ID, SFTDataCollator, SFTDataset,
                                 load_bpe_tokenizer)
    from src.sft.train_sft import evaluate_validation_loss, load_pretrained_sft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_bpe_tokenizer(args.tokenizer_path)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
           else torch.autocast("cpu", enabled=False))

    def load(path: Path):
        return load_pretrained_sft_model(checkpoint_path=path, device=device,
                                         target_vocab_size=16388,
                                         target_block_size=args.max_seq_len).to(device)

    if args.unpacked:
        curves: List[Tuple[str, List[float]]] = []
        for spec in args.ckpt:
            label, _, path = spec.partition("=")
            if not Path(path).exists():
                print(f"{label:<14} MISSING")
                continue
            model = load(Path(path))
            curves.append((label, per_pair_losses(model, args.test, tokenizer, device, amp,
                                                  args.max_seq_len, subset=args.subset)))
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"shared test set: {args.test}  (per-pair, unpacked)")
        for label, losses in curves:
            mean_loss = sum(losses) / len(losses)
            print(f"  {label:<14} mean {mean_loss:.4f}   ppl {math.exp(mean_loss):.2f}")
        if len(curves) == 2:
            paired_report(curves[0][0], curves[0][1], curves[1][0], curves[1][1])
        return 0

    dataset = SFTDataset(jsonl_path=args.test, tokenizer=tokenizer,
                         max_seq_len=args.max_seq_len, pack_sequences=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=SFTDataCollator(pad_token_id=IM_END_ID,
                                                   ignore_index=IGNORE_INDEX))
    print(f"shared test set: {args.test}  ({len(dataset)} packed blocks)")
    print("=" * 54)
    print(f"{'model':<14} {'test loss':>10} {'ppl':>9}")
    print("-" * 54)
    results = []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        if not Path(path).exists():
            print(f"{label:<14} {'MISSING':>10}")
            continue
        model = load(Path(path))
        loss = evaluate_validation_loss(model, loader, device, amp)
        results.append((label, loss))
        print(f"{label:<14} {loss:>10.4f} {math.exp(loss):>9.2f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print("=" * 54)
    if len(results) == 2:
        (label_a, loss_a), (label_b, loss_b) = results
        delta = loss_b - loss_a
        print(f"\n{label_b} vs {label_a}: {delta:+.4f} nats "
              f"({'better' if delta < 0 else 'worse'} for {label_b})")
        print("Measured on identical held-out chunks. Note that packed scoring is unfit")
        print("for small effects — use --unpacked for anything under ~0.15 nats.")
    return 0


# ===========================================================================
# strip
# ===========================================================================
# Training checkpoints embed optimizer.state_dict(), which for a 113M model is
# roughly twice the size of the weights (two moment buffers per parameter).
# Inference, evaluation, and RL initialization only read the 'model' key, so
# archived checkpoints carry ~2/3 dead weight. Only *_last.pt needs optimizer
# state, and only to resume that exact run.


def human_bytes(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def strip_one(path: Path, apply: bool, backup: bool) -> Tuple[int, int]:
    """Returns (bytes_before, bytes_after) — estimated when apply is False."""
    before = path.stat().st_size
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  [skip] {path.name}: unreadable ({exc})")
        return before, before

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        print(f"  [skip] {path.name}: no 'model' key")
        return before, before
    if "optimizer" not in checkpoint:
        print(f"  [ok]   {path.name}: already stripped ({human_bytes(before)})")
        return before, before

    if not apply:
        # Optimizer state is ~2 fp32 moment tensors per trainable parameter.
        n_params = sum(v.numel() for v in checkpoint["model"].values() if hasattr(v, "numel"))
        estimated = max(before - n_params * 4 * 2, 0)
        print(f"  [dry]  {path.name}: {human_bytes(before)} -> ~{human_bytes(estimated)}")
        return before, estimated

    slim = {k: v for k, v in checkpoint.items() if k != "optimizer"}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(slim, tmp)
    if backup:
        os.replace(path, path.with_suffix(path.suffix + ".bak"))
    os.replace(tmp, path)
    after = path.stat().st_size
    print(f"  [done] {path.name}: {human_bytes(before)} -> {human_bytes(after)}")
    return before, after


def run_strip(args: argparse.Namespace) -> int:
    root = Path(args.root)
    targets = sorted(root.rglob("*.pt")) if root.is_dir() else [root]
    if args.skip_last:
        targets = [p for p in targets if not p.name.endswith("_last.pt")]
    if not targets:
        print("No .pt checkpoints found.")
        return 0

    print(f"{'Applying' if args.apply else 'Dry run over'} {len(targets)} checkpoint(s) "
          f"under {root}\n")
    total_before = total_after = 0
    for path in targets:
        before, after = strip_one(path, apply=args.apply, backup=not args.no_backup)
        total_before += before
        total_after += after

    print(f"\nTotal: {human_bytes(total_before)} -> {human_bytes(total_after)} "
          f"(saves {human_bytes(total_before - total_after)})")
    if not args.apply:
        print("Dry run only — re-run with --apply to rewrite.")
    return 0


# ===========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    p_select = sub.add_parser("select", help="rank candidates on measured behaviour")
    p_select.add_argument("--ckpt", action="append", default=[],
                          help="explicit checkpoint path (repeatable)")
    p_select.add_argument("--ckpt-dir", default=None, help="directory to scan for sft_ckpt_*.pt")
    p_select.add_argument("--test", default=str(ROOT / "data" / "rl" / "test.jsonl"),
                          help="held-out set in RL schema (see src.rl.prepare_rl_dataset)")
    p_select.add_argument("--num-conceptual", type=int, default=60,
                          help="non-numerical prompts; these feed only the contamination and "
                               "echo gates, the primary metric uses the numerical subset")
    p_select.add_argument("--max-new-tokens", type=int, default=384)
    p_select.add_argument("--temperature", type=float, default=0.3)
    p_select.add_argument("--top-k", type=int, default=40)
    p_select.add_argument("--repetition-penalty", type=float, default=1.2)
    p_select.add_argument("--seed", type=int, default=1234)
    p_select.add_argument("--link-best", default=None, help="copy the winner to this path")
    p_select.add_argument("--out-json", default=str(ROOT / "docs" / "checkpoint_selection.json"))
    p_select.set_defaults(func=run_select)

    p_compare = sub.add_parser("compare", help="score checkpoints on one shared test set")
    p_compare.add_argument("--test", required=True, help="shared held-out test JSONL")
    p_compare.add_argument("--subset", default=None,
                           help="restrict to one tag in data/eval/heldout_subsets.jsonl")
    p_compare.add_argument("--ckpt", action="append", required=True, help="label=path (repeatable)")
    p_compare.add_argument("--tokenizer-path",
                           default=str(NANOGPT_SEIS_ROOT / "data" / "tokenized" / "tokenizer.json"))
    p_compare.add_argument("--batch-size", type=int, default=4)
    p_compare.add_argument("--max-seq-len", type=int, default=4096)
    p_compare.add_argument("--unpacked", action="store_true",
                           help="score each pair in isolation and run a paired test "
                                "(removes the sequence-packing confound)")
    p_compare.set_defaults(func=run_compare)

    p_strip = sub.add_parser("strip", help="drop optimizer state from archived checkpoints")
    p_strip.add_argument("root", help="checkpoint file or directory to scan")
    p_strip.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    p_strip.add_argument("--no-backup", action="store_true", help="do not keep .bak copies")
    p_strip.add_argument("--skip-last", action="store_true", help="leave *_last.pt resumable")
    p_strip.set_defaults(func=run_strip)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
