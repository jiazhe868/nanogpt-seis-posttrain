#!/usr/bin/env python3
"""Phase II: supervised fine-tuning of the 113M pretrained base.

Loads the pretrained weights, resizes the vocabulary 16,384 -> 16,388 for the
ChatML special tokens, and trains on assistant tokens only. bf16 autocast,
AdamW, gradient clipping, linear warmup then cosine decay, DDP or single GPU.

    torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft \
        --config configs/sft_base.yaml
    python -m src.sft.train_sft --epochs 3 --batch-size 4 --grad-accum 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Resolve repository paths and append parent codes directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODES_ROOT = REPO_ROOT.parent
NANOGPT_SEIS_ROOT = CODES_ROOT / "nanogpt_seis"

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

# Import GPT model from nanogpt_seis codebase
from nanogpt_seis.src.model.gqa_gpt import GPT, GPTConfig
from src.sft.dataset import (
    IGNORE_INDEX,
    IM_END_ID,
    IM_START_ID,
    SFTDataCollator,
    SFTDataset,
    load_bpe_tokenizer,
)


def setup_distributed() -> Tuple[int, int, int, bool]:
    """
    Initializes PyTorch Distributed Data Parallel (DDP) if launched via torchrun.

    Returns:
        Tuple of (rank, local_rank, world_size, is_ddp_enabled).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        is_ddp = True
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_ddp = False

    return rank, local_rank, world_size, is_ddp


def get_learning_rate(
    current_step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Calculates scheduled learning rate with linear warmup and cosine decay.

    Args:
        current_step: Current training step index (0-indexed).
        total_steps: Total training steps.
        warmup_steps: Number of warmup steps.
        peak_lr: Maximum target learning rate.
        min_lr: Minimum learning rate floor.

    Returns:
        Scheduled learning rate value.
    """
    if current_step < warmup_steps:
        return peak_lr * (current_step + 1) / max(1, warmup_steps)
    if current_step >= total_steps:
        return min_lr
    decay_ratio = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def resize_model_embeddings(model: GPT, new_vocab_size: int = 16388) -> None:
    """
    Resizes model embedding layer and linear LM head to accommodate ChatML special tokens.
    Preserves pretrained weights for tokens 0..old_vocab_size-1 and maintains weight tying.

    Args:
        model: Pretrained GPT model instance.
        new_vocab_size: Target extended vocabulary size (default: 16388).
    """
    old_vocab_size, d_model = model.tok_emb.weight.shape
    if new_vocab_size <= old_vocab_size:
        return

    # Create new combined weight tensor
    new_weight = torch.empty(
        (new_vocab_size, d_model),
        device=model.tok_emb.weight.device,
        dtype=model.tok_emb.weight.dtype,
    )
    with torch.no_grad():
        new_weight[:old_vocab_size] = model.tok_emb.weight.data
        new_weight[old_vocab_size:].normal_(mean=0.0, std=0.02)

    # Wrap in Parameter so optimizer tracks it
    new_param = nn.Parameter(new_weight)

    # Create new Embedding and Linear modules using the SAME shared Parameter
    new_tok_emb = nn.Embedding(new_vocab_size, d_model)
    new_tok_emb.weight = new_param

    new_lm_head = nn.Linear(d_model, new_vocab_size, bias=False)
    new_lm_head.weight = new_param

    model.tok_emb = new_tok_emb
    model.lm_head = new_lm_head
    model.cfg.vocab_size = new_vocab_size


def load_pretrained_sft_model(
    checkpoint_path: Path,
    device: str,
    target_vocab_size: int = 16388,
    target_block_size: int = 4096,
) -> GPT:
    """
    Loads pretrained weights from checkpoint and resizes vocabulary for SFT.

    Args:
        checkpoint_path: Absolute or relative path to pretrained .pt checkpoint.
        device: Target device string ('cuda' or 'cpu').
        target_vocab_size: Target extended vocabulary size.
        target_block_size: Context length capacity (default: 4096).

    Returns:
        Initialized and resized GPT model instance.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_config = checkpoint["model_cfg"]

    config = GPTConfig(
        vocab_size=raw_config.get("vocab_size", 16384),
        block_size=target_block_size,
        n_layer=raw_config.get("n_layer", 16),
        n_head=raw_config.get("n_head", 12),
        n_kv_head=raw_config.get("n_kv_head", 4),
        d_model=raw_config.get("d_model", 768),
        ffn_multiple_of=raw_config.get("ffn_multiple_of", 256),
        rope_theta=raw_config.get("rope_theta", 10000.0),
        dropout=raw_config.get("dropout", 0.0),
    )

    model = GPT(config)
    model.load_state_dict(checkpoint["model"])

    # Resize embedding for ChatML special tokens
    resize_model_embeddings(model, new_vocab_size=target_vocab_size)
    return model


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    amp_context: torch.cuda.amp.autocast,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """
    Evaluates validation cross-entropy loss over unmasked response tokens.

    Args:
        model: Model instance (PyTorch Module or DDP wrapper).
        val_loader: Validation DataLoader.
        device: Torch device string.
        amp_context: Autocast context for mixed precision.
        ignore_index: Label value to ignore (-100).

    Returns:
        Token-weighted mean validation loss, identical on every DDP rank.
    """
    model.eval()
    # Sum of per-token losses and token count, so batches with few unmasked
    # tokens (e.g. padded packing tails) are weighted by their actual size
    # instead of counting as one full batch.
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    token_count = torch.zeros((), dtype=torch.float64, device=device)

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        inputs = input_ids[:, :-1]
        targets = labels[:, 1:]

        with amp_context:
            logits, _ = model(inputs, targets, ignore_index=ignore_index)

        flat_logits = logits.float().view(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        loss_sum += F.cross_entropy(
            flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum"
        ).double()
        token_count += (flat_targets != ignore_index).sum().double()

    # Each rank sees a different DistributedSampler shard; reduce so every rank
    # (and the logged number) reflects the full validation set.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    model.train()
    if token_count.item() == 0:
        return float("inf")
    return (loss_sum / token_count).item()


def train_sft():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for nanoGPT-Seis")
    parser.add_argument("--config", type=str, default="", help="Path to YAML configuration file (e.g. configs/sft_base.yaml)")
    parser.add_argument("--ckpt-path", type=str, default=str(NANOGPT_SEIS_ROOT / "checkpoints" / "ckpt.pt"), help="Pretrained checkpoint path")
    parser.add_argument("--tokenizer-path", type=str, default=str(NANOGPT_SEIS_ROOT / "data" / "tokenized" / "tokenizer.json"), help="BPE tokenizer path")
    parser.add_argument("--train-path", type=str, default=str(REPO_ROOT / "data" / "splits" / "train.jsonl"), help="Formatted train JSONL path")
    parser.add_argument("--val-path", type=str, default=str(REPO_ROOT / "data" / "splits" / "val.jsonl"), help="Formatted val JSONL path")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "checkpoints"), help="Output directory for SFT checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak SFT learning rate")
    parser.add_argument("--min-lr", type=float, default=2e-5, help="Minimum learning rate floor")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio of total steps")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum context length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate --ckpt-path on --val-path and exit")
    parser.add_argument("--snapshot-epochs", action="store_true",
                        help="Save a weights-only checkpoint at the end of every epoch. "
                             "Validation loss is a poor selection criterion here (a later "
                             "checkpoint with 0.51 nats WORSE held-out loss generated "
                             "significantly better-formatted answers), so keep candidates "
                             "and choose with `src.sft.checkpoints select` instead.")
    args = parser.parse_args()

    # Load YAML config if provided and override default values
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
            train_cfg = cfg_dict.get("train", {})
            for key, val in train_cfg.items():
                if hasattr(args, key):
                    setattr(args, key, val)

    # 1. Setup Distributed Environment
    rank, local_rank, world_size, is_ddp = setup_distributed()
    is_master = (rank == 0)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed + rank)

    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"================================================================================")
        print(f"🚀 Starting Supervised Fine-Tuning (SFT) for nanoGPT-Seis")
        print(f"   World Size: {world_size} GPU(s) | Device: {device} | DDP: {is_ddp}")
        print(f"================================================================================")

    # 2. Load Tokenizer & Datasets
    tokenizer = load_bpe_tokenizer(args.tokenizer_path)

    if args.eval_only:
        val_dataset = SFTDataset(
            jsonl_path=args.val_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            pack_sequences=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=SFTDataCollator(pad_token_id=IM_END_ID, ignore_index=IGNORE_INDEX),
            num_workers=2,
            pin_memory=True,
        )
        model = load_pretrained_sft_model(
            checkpoint_path=Path(args.ckpt_path),
            device=device,
            target_vocab_size=16388,
            target_block_size=args.max_seq_len,
        ).to(device)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if "cuda" in device
            else torch.autocast("cpu", enabled=False)
        )
        val_loss = evaluate_validation_loss(model, val_loader, device, amp_context)
        print(f"[EVAL-ONLY] ckpt={args.ckpt_path}")
        print(f"[EVAL-ONLY] val={args.val_path} ({len(val_dataset)} packed blocks)")
        print(f"[EVAL-ONLY] val_loss={val_loss:.4f} | ppl={math.exp(val_loss):.2f}")
        if is_ddp:
            dist.destroy_process_group()
        return

    train_dataset = SFTDataset(
        jsonl_path=args.train_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        pack_sequences=True,
    )
    val_dataset = SFTDataset(
        jsonl_path=args.val_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        pack_sequences=True,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    data_collator = SFTDataCollator(pad_token_id=IM_END_ID, ignore_index=IGNORE_INDEX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Load Pretrained Model & Extend Vocabulary
    model = load_pretrained_sft_model(
        checkpoint_path=Path(args.ckpt_path),
        device=device,
        target_vocab_size=16388,
        target_block_size=args.max_seq_len,
    ).to(device)

    raw_model = model
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module

    if is_master:
        print(f"[INFO] Model parameters: {raw_model.num_params():,} total")
        print(f"[INFO] Train samples (packed): {len(train_dataset)} | Val samples (packed): {len(val_dataset)}")

    # 4. Optimizer & LR Schedule Setup
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if "cuda" in device else "cpu",
    )

    total_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_training_steps = total_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    if is_master:
        print(f"[INFO] Total SFT training steps: {total_training_steps} (Warmup: {warmup_steps} steps)")

    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if "cuda" in device
        else torch.autocast("cpu", enabled=False)
    )

    # Logging CSV setup
    log_csv_path = os.path.join(args.out_dir, "sft_log.csv")
    if is_master:
        with open(log_csv_path, "w", encoding="utf-8") as f:
            f.write("step,epoch,train_loss,val_loss,lr,elapsed_sec\n")

    # 5. Training Loop
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]

            # Require gradient sync only on final accumulation micro-batch
            is_accumulating = ((step_idx + 1) % args.grad_accum != 0) and ((step_idx + 1) != len(train_loader))
            if is_ddp:
                model.require_backward_grad_sync = not is_accumulating

            # The last accumulation group of an epoch may hold fewer than
            # grad_accum micro-batches; divide by the group's true size so the
            # final optimizer step is not systematically undersized.
            group_start = (step_idx // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)

            with amp_context:
                logits, loss = model(inputs, targets, ignore_index=IGNORE_INDEX)
                loss = loss / group_size

            accumulated_loss += loss.item() * group_size
            loss.backward()

            # Execute optimizer step after grad_accum micro-batches
            if not is_accumulating:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)

                # Set scheduled learning rate
                current_lr = get_learning_rate(
                    current_step=global_step,
                    total_steps=total_training_steps,
                    warmup_steps=warmup_steps,
                    peak_lr=args.learning_rate,
                    min_lr=args.min_lr,
                )
                for param_group in optimizer.param_groups:
                    lr_mult = param_group.get("lr_mult", 1.0)
                    param_group["lr"] = current_lr * lr_mult

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                avg_train_loss = accumulated_loss / group_size
                accumulated_loss = 0.0

                # Logging & evaluation at interval
                if global_step % 10 == 0 or global_step == total_training_steps:
                    val_loss = evaluate_validation_loss(model, val_loader, device, amp_context)
                    elapsed = time.time() - start_time

                    if is_master:
                        print(
                            f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_training_steps} | "
                            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
                        )
                        with open(log_csv_path, "a", encoding="utf-8") as f:
                            f.write(f"{global_step},{epoch},{avg_train_loss:.4f},{val_loss:.4f},{current_lr:.6e},{elapsed:.1f}\n")

                        # Save best checkpoint (weights only — optimizer state
                        # lives in sft_ckpt_last.pt for resume; keeping it out
                        # of every best snapshot saves ~0.9 GB per file)
                        if math.isfinite(val_loss) and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_checkpoint_path = os.path.join(args.out_dir, "sft_ckpt_best.pt")
                            torch.save(
                                {
                                    "model": raw_model.state_dict(),
                                    "model_cfg": raw_model.cfg.__dict__,
                                    "global_step": global_step,
                                    "epoch": epoch,
                                    "best_val_loss": best_val_loss,
                                },
                                best_checkpoint_path,
                            )

        # Per-epoch snapshot for behaviour-based selection later
        if is_master and args.snapshot_epochs:
            snap = os.path.join(args.out_dir, f"sft_ckpt_epoch{epoch}.pt")
            torch.save({
                "model": raw_model.state_dict(),
                "model_cfg": raw_model.cfg.__dict__,
                "global_step": global_step,
                "epoch": epoch,
            }, snap)
            print(f"[SNAPSHOT] epoch {epoch} -> {snap}")

    # Save final checkpoint
    if is_master:
        last_checkpoint_path = os.path.join(args.out_dir, "sft_ckpt_last.pt")
        torch.save(
            {
                "model": raw_model.state_dict(),
                "model_cfg": raw_model.cfg.__dict__,
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": args.epochs,
                "best_val_loss": best_val_loss,
            },
            last_checkpoint_path,
        )
        print(f"================================================================================")
        print(f"🎉 SFT Training Complete!")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Best Checkpoint: {os.path.join(args.out_dir, 'sft_ckpt_best.pt')}")
        print(f"================================================================================")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    train_sft()
