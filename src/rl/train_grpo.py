"""Phase III: group-relative REINFORCE with KL regularization.

For each prompt: sample G completions, score them with src.rl.verifiers,
standardize the rewards within the group, take one policy-gradient step.

    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    per_token  = -(logp * advantage - beta * kl) * sampled_mask

The group mean replaces PPO's learned critic. There is NO importance ratio and
NO clipped surrogate, so this is not canonical GRPO — see README section 5.3.
KL uses the k3 estimator against the frozen SFT reference, with an adaptive
beta and a hard early stop on runaway. Masks cover only sampled tokens:
including forced-EOS padding puts gradient on tokens the policy never chose
(PITFALLS D1). 2-GPU DDP, bf16.

    torchrun --standalone --nproc_per_node=2 -m src.rl.train_grpo \
        --config configs/rl_grpo.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sft.dataset import DEFAULT_SYSTEM_PROMPT, IM_END_ID, load_bpe_tokenizer
from src.sft.chat import format_chatml_token_stream
from src.sft.train_sft import GPT, load_pretrained_sft_model
from src.rl.verifiers import CompositeRewardEngine


def setup_ddp() -> Tuple[int, int, int]:
    """Initialize PyTorch DDP distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def load_sft_model(ckpt_path: Path, device: torch.device) -> GPT:
    """Load Stage 2 SFT model checkpoint."""
    model = load_pretrained_sft_model(
        checkpoint_path=ckpt_path,
        device=str(device),
        target_vocab_size=16388,
        target_block_size=4096,
    )
    model.to(device)
    return model


def build_prompt_ids(tokenizer: Any, prompt_text: str, device: torch.device) -> torch.Tensor:
    """ChatML prompt with the SAME system turn and encoding as SFT training."""
    ids = format_chatml_token_stream(
        tokenizer=tokenizer,
        history=[],
        new_user_message=prompt_text,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def generate_rollouts(
    model: GPT,
    prompt_ids: torch.Tensor,
    tokenizer: Any,
    group_size: int = 8,
    max_new_tokens: int = 384,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = IM_END_ID,
    block_size: int = 4096,
) -> Tuple[torch.Tensor, List[str], List[int], List[bool]]:
    """Generate G rollout completions for a single prompt.

    Returns (sequences, decoded texts, completion lengths, terminated flags).
    Completion length counts the tokens the policy actually sampled,
    including its terminating EOS; lockstep padding beyond that is excluded
    from the loss by the caller.
    """
    model.eval()
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]

    input_ids = prompt_ids.repeat(group_size, 1)
    generated_ids = input_ids.clone()
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    comp_lens = torch.zeros(group_size, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if finished.all():
            break

        ctx = generated_ids[:, -block_size:] if generated_ids.shape[1] > block_size else generated_ids
        logits, _ = model(ctx)
        next_token_logits = logits.squeeze(1) / max(temperature, 1e-6)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for i in range(group_size):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                next_token_logits[i, indices_to_remove] = -float("Inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        newly_finished = (~finished) & (next_tokens.squeeze(-1) == eos_token_id)
        comp_lens = comp_lens + (~finished).long()  # counts this token for unfinished rows
        finished = finished | newly_finished
        # Lockstep padding for rows already finished (excluded from loss)
        next_tokens = torch.where(finished.unsqueeze(-1) & ~newly_finished.unsqueeze(-1),
                                  torch.tensor(eos_token_id, device=device), next_tokens)

        generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

    text_completions = []
    for i in range(group_size):
        comp_ids = generated_ids[i, prompt_len:].tolist()
        if eos_token_id in comp_ids:
            comp_ids = comp_ids[:comp_ids.index(eos_token_id)]
        text_completions.append(tokenizer.decode(comp_ids))

    return generated_ids, text_completions, comp_lens.tolist(), finished.tolist()


def compute_grpo_advantages(rewards: List[float]) -> torch.Tensor:
    """Compute standardized group advantages A_i = (R_i - mean) / (std + eps)."""
    r_tensor = torch.tensor(rewards, dtype=torch.float32)
    if len(rewards) <= 1:
        return torch.zeros_like(r_tensor)

    mean_r = r_tensor.mean()
    std_r = r_tensor.std(unbiased=False)

    if std_r < 1e-8:
        return torch.zeros_like(r_tensor)
    return (r_tensor - mean_r) / (std_r + 1e-8)


def train_grpo(config_path: Path) -> None:
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print("=" * 80)
        print("🚀 Starting Stage 3 GRPO Training for nanoGPT-Seis (V2)")
        print(f"   World Size: {world_size} GPU(s) | Device: {device}")
        print("=" * 80)

    # Load policy and reference models
    base_ckpt = ROOT / config["model"]["base_checkpoint"]
    policy_model = load_sft_model(base_ckpt, device)
    ref_model = load_sft_model(base_ckpt, device)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if world_size > 1:
        policy_model = DDP(policy_model, device_ids=[local_rank])
    raw_policy = policy_model.module if hasattr(policy_model, "module") else policy_model

    grpo_cfg = config["grpo"]
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(grpo_cfg["learning_rate"]),
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # Load RL prompt dataset and assert it matches the verifier schema.
    prompt_file = ROOT / config["data"]["train_file"]
    with open(prompt_file, "r", encoding="utf-8") as f:
        rl_prompts = [json.loads(line) for line in f if line.strip()]

    missing = [k for k in ("prompt", "type", "final_numerical_value", "key_concepts")
               if any(k not in item for item in rl_prompts)]
    assert not missing, f"RL dataset schema mismatch, missing fields: {missing} — regenerate with src.rl.prepare_rl_dataset"
    n_num = sum(1 for it in rl_prompts if it["type"] == "numerical")
    n_num_target = sum(1 for it in rl_prompts if it["type"] == "numerical" and it["final_numerical_value"] is not None)
    assert n_num == 0 or n_num_target > 0.5 * n_num, \
        f"NumericalVerifier would be mostly dead: {n_num_target}/{n_num} numerical items have targets"
    if is_main:
        print(f"[INFO] RL prompts: {len(rl_prompts)} total | numerical with target: {n_num_target}/{n_num}")

    tok_path = ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json"
    tokenizer = load_bpe_tokenizer(tok_path)
    reward_engine = CompositeRewardEngine()

    output_dir = ROOT / config["save"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv = ROOT / config["save"]["log_file"]

    if is_main:
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "mean_reward", "num_reward", "concept_reward", "format_reward",
                "clean_reward", "policy_loss", "kl_div", "kl_coeff", "entropy",
                "mean_comp_tokens", "stop_rate", "lr", "elapsed_sec",
            ])

    start_time = time.time()
    max_steps = grpo_cfg.get("max_steps", 300)
    group_size = grpo_cfg.get("group_size", 8)
    max_new_toks = grpo_cfg.get("max_new_tokens", 384)
    temperature = grpo_cfg.get("temperature", 1.0)
    top_p = grpo_cfg.get("top_p", 1.0)
    kl_coeff = float(grpo_cfg.get("kl_coeff", 0.04))
    kl_target = float(grpo_cfg.get("kl_target", 0.05))
    kl_stop = float(grpo_cfg.get("kl_stop", 0.5))
    kl_window: deque = deque(maxlen=20)

    # Without-replacement prompt schedule, disjoint across ranks per pass.
    rng = random.Random(int(grpo_cfg.get("seed", 42)))
    order: List[int] = []

    stop_flag = torch.zeros(1, device=device)

    for step in range(1, max_steps + 1):
        if not order:
            order = list(range(len(rl_prompts)))
            rng.shuffle(order)
        idx = order.pop()
        # offset by rank so the two ranks never train the same prompt in a step
        item = rl_prompts[(idx + rank) % len(rl_prompts)]
        prompt_ids = build_prompt_ids(tokenizer, item["prompt"], device)

        # 1. Generate G rollouts
        generated_seqs, completions, comp_lens, terminated = generate_rollouts(
            raw_policy,
            prompt_ids,
            tokenizer=tokenizer,
            group_size=group_size,
            max_new_tokens=max_new_toks,
            temperature=temperature,
            top_p=top_p,
        )

        # 2. Score rollouts with verifiers
        keys = ("total_reward", "r_num", "r_concept", "r_format", "r_clean")
        scored = [reward_engine.score(comp, item, terminated=t)
                  for comp, t in zip(completions, terminated)]
        group_rewards = [s["total_reward"] for s in scored]

        # 3. Compute advantages
        advantages = compute_grpo_advantages(group_rewards).to(device)

        # 4. Policy gradient pass
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)

        prompt_len = prompt_ids.shape[1]
        targets = generated_seqs[:, 1:].clone()
        inputs = generated_seqs[:, :-1]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy_logits, _ = policy_model(inputs, targets=targets)
            with torch.no_grad():
                ref_logits, _ = ref_model(inputs, targets=targets)

            log_probs = F.log_softmax(policy_logits.float(), dim=-1)
            ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)

            target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            ref_target_log_probs = torch.gather(ref_log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

            # Mask exactly the tokens the policy sampled (completion tokens up
            # to and including its own EOS) — never the prompt, never the
            # lockstep padding appended after a rollout finished.
            mask = torch.zeros_like(targets, dtype=torch.float32)
            for i, L in enumerate(comp_lens):
                mask[i, prompt_len - 1: prompt_len - 1 + L] = 1.0
            seq_tokens = mask.sum(dim=1).clamp(min=1.0)

            # k3 KL estimator per token
            kl_div = torch.exp(ref_target_log_probs - target_log_probs) - (ref_target_log_probs - target_log_probs) - 1.0

            # Per-sequence mean, then group mean: sequences contribute equally
            # regardless of length (avoids the token-mean length bias).
            adv = advantages.unsqueeze(-1)
            per_tok = -(target_log_probs * adv - kl_coeff * kl_div) * mask
            loss = (per_tok.sum(dim=1) / seq_tokens).mean()

            with torch.no_grad():
                entropy_tok = -(log_probs.exp() * log_probs).sum(dim=-1)
                mean_entropy = (entropy_tok * mask).sum() / mask.sum().clamp(min=1.0)
                mean_kl = (kl_div * mask).sum() / mask.sum().clamp(min=1.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        # 5. Adaptive KL controller + hard early stop (synchronized KL)
        kl_value = mean_kl.detach()
        if dist.is_initialized():
            dist.all_reduce(kl_value, op=dist.ReduceOp.SUM)
            kl_value = kl_value / world_size
        kl_scalar = kl_value.item()
        kl_window.append(kl_scalar)
        if kl_scalar > 2.0 * kl_target:
            kl_coeff = min(kl_coeff * 1.5, 1.0)
        elif kl_scalar < 0.5 * kl_target:
            kl_coeff = max(kl_coeff / 1.5, 0.005)
        rolling_kl = sum(kl_window) / len(kl_window)
        if len(kl_window) == kl_window.maxlen and rolling_kl > kl_stop:
            stop_flag.fill_(1.0)
        if dist.is_initialized():
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)

        # 6. Logging
        if is_main:
            elapsed = time.time() - start_time
            means = {k: sum(s[k] for s in scored) / len(scored) for k in keys}
            mean_len = sum(comp_lens) / len(comp_lens)
            stop_rate = sum(terminated) / len(terminated)

            print(
                f"GRPO Step {step:3d}/{max_steps} | Reward: {means['total_reward']:+.4f} | "
                f"Num: {means['r_num']:+.2f} | Concept: {means['r_concept']:+.2f} | "
                f"Fmt: {means['r_format']:+.2f} | KL: {kl_scalar:.4f} (β={kl_coeff:.3f}) | "
                f"H: {mean_entropy.item():.2f} | len: {mean_len:.0f} | stop: {stop_rate:.0%} | {elapsed:.1f}s",
                flush=True,
            )
            with open(log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, f"{means['total_reward']:.4f}", f"{means['r_num']:.4f}",
                    f"{means['r_concept']:.4f}", f"{means['r_format']:.4f}",
                    f"{means['r_clean']:.4f}", f"{loss.item():.4f}", f"{kl_scalar:.4f}",
                    f"{kl_coeff:.4f}", f"{mean_entropy.item():.4f}", f"{mean_len:.1f}",
                    f"{stop_rate:.4f}", f"{grpo_cfg['learning_rate']}", f"{elapsed:.1f}",
                ])

        # 7. Checkpoint (weights only — a 113M model's AdamW state triples file size)
        if is_main and (step % grpo_cfg.get("save_every", 100) == 0 or step == max_steps or stop_flag.item() > 0):
            ckpt_save_path = output_dir / f"grpo_ckpt_step_{step}.pt"
            torch.save({
                "step": step,
                "model": raw_policy.state_dict(),
                "model_cfg": raw_policy.cfg.__dict__,
            }, ckpt_save_path)
            print(f"  [SAVED] Checkpoint -> {ckpt_save_path}")

        if stop_flag.item() > 0:
            if is_main:
                print(f"[EARLY STOP] rolling KL {rolling_kl:.3f} > kl_stop {kl_stop} at step {step} — "
                      f"policy is drifting from the SFT reference; stopping before degeneration.")
            break

    cleanup_ddp()
    if is_main:
        print("\n" + "=" * 80)
        print("🎉 Stage 3 GRPO Training Complete!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training Engine for nanoGPT-Seis (V2)")
    parser.add_argument("--config", type=str, default="configs/rl_grpo.yaml", help="Path to GRPO YAML config")
    args = parser.parse_args()

    train_grpo(ROOT / args.config)


if __name__ == "__main__":
    main()
