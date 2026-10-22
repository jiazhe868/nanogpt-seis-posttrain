"""Phase III: group-relative REINFORCE with KL regularization.

For each step: sample G completions for each of P prompts per GPU, score them
with src.rl.verifiers, centre the rewards within each group, take one
policy-gradient step on the whole batch.

    advantages = (rewards - group_mean) / scale
    per_token  = -(logp * advantage) + beta * kl_term          (sampled tokens only)

The group mean replaces PPO's learned critic. With num_iterations: 1 (v4, v9 on) the
importance ratio is identically 1 and the clipped surrogate reduces to REINFORCE;
num_iterations: 2 (v5-v8) reuses each batch with a clipped ratio — see README section 5.4.
Masks cover only sampled tokens: including forced-EOS padding puts gradient on
tokens the policy never chose (PITFALLS D1). 2-GPU data parallel, bf16.

v4 — why the v3 reward and KL curves never settled (README 5.8):

  * ONE prompt per GPU per step. Each logged point was the mean of 8 rollouts
    of a single prompt (rank 0 only): step-to-step reward SD 0.35 against a
    trend of ~0.3, lag-1 autocorrelation 0.04 — the curve was prompt-difficulty
    noise. `prompts_per_step` batches P prompts per GPU, and every logged
    number is all-reduced over the whole global batch.
  * Single-type batches. A step was all-numerical or all-conceptual, two
    reward distributions with different means, so even the rolling average
    wobbled with the type mix. `stratify_types` fixes the mix in every batch.
  * A bang-bang beta controller. x1.5 per step outside a [0.5, 2] x target
    deadband sent beta to its floor by step 6 (KL unopposed for 60 steps),
    then 0.038 -> 0.192 in five steps at step ~100, where it stayed: the
    objective changed 38x mid-run and the policy was dragged back toward the
    reference (stop rate 0.94 -> 0.51). `kl_controller: fixed` keeps one
    objective; `proportional` is the Ziegler et al. (2019) controller on an EMA.
  * Per-group std normalization. A group whose rewards differ only by a 0.01
    length penalty was rescaled to unit advantages — full-size gradient on
    noise. `adv_norm: batch` divides by one std pooled over the batch.
  * k3 as the KL loss. Its gradient is (1 - p_ref/p) grad log p, unbounded
    where the policy has suppressed a token the reference likes, and in
    expectation it is the gradient of KL(ref || policy), not the KL it logs.
    `kl_loss: k2` (0.5 * log_ratio^2) gives exactly grad KL(policy || ref)
    on-policy. k3 is still what is logged as the KL value.
  * No warmup: Adam's first bias-corrected steps are its largest.
  * Rollouts re-encoded the whole context for every token (O(T^2), ~27 s per
    step for 8 rollouts per GPU). They now decode all P*G rows together with
    a KV cache (~13 s per step for 128 per GPU).

Set prompts_per_step: 1, stratify_types: false, adv_norm: group,
loss_agg: seq_mean, kl_loss: k3, kl_controller: legacy, warmup_steps: 0 to
recover the v3 algorithm (configs/rl_grpo_v3.yaml).

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sft.dataset import DEFAULT_SYSTEM_PROMPT, IM_END_ID, load_bpe_tokenizer
from src.sft.chat import format_chatml_token_stream
from src.sft.train_sft import GPT, load_pretrained_sft_model
from src.rl.verifiers import CompositeRewardEngine
COMPONENTS = ("r_num", "r_concept", "r_format", "r_clean", "r_echo", "r_length", "r_term")


def setup_ddp() -> Tuple[int, int, int]:
    """Initialize the process group (gradients are all-reduced by hand, not by DDP)."""
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


def build_prompt_ids(tokenizer: Any, prompt_text: str) -> List[int]:
    """ChatML prompt with the SAME system turn and encoding as SFT training."""
    return format_chatml_token_stream(
        tokenizer=tokenizer,
        history=[],
        new_user_message=prompt_text,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


# Rollouts: batched, left-padded, KV-cached decoding lives in src/rl/rollout.py (shared with
# src.rl.eval_rl). Re-exported here for older imports.
from src.rl.pot import run_program  # noqa: E402
from src.rl.rollout import (  # noqa: E402,F401
    AMP_DTYPE, _cached_attention, _cached_step, _sample, decode_completion, generate_rollouts)


# ===========================================================================
# Objective
# ===========================================================================

def compute_grpo_advantages(rewards: List[float]) -> torch.Tensor:
    """v3 advantage: A_i = (R_i - mean) / (std + eps) within one group."""
    r_tensor = torch.tensor(rewards, dtype=torch.float32)
    if len(rewards) <= 1:
        return torch.zeros_like(r_tensor)
    std_r = r_tensor.std(unbiased=False)
    if std_r < 1e-8:
        return torch.zeros_like(r_tensor)
    return (r_tensor - r_tensor.mean()) / (std_r + 1e-8)


def compute_batch_advantages(groups: List[List[float]], mode: str) -> List[List[float]]:
    """Group-mean baseline; `mode` picks the scale.

    group — divide by each group's own std (v3). A group whose rewards differ
            by 0.01 gets the same unit-size advantages as one that differs by 2.
    batch — divide by ONE std pooled over every informative group in the batch:
            the update scale stays fixed, small reward gaps stay small.
    none  — no scaling (Dr. GRPO; Liu et al. 2025).
    Zero-spread groups get zero advantage under every mode.
    """
    if mode == "group":
        return [compute_grpo_advantages(g).tolist() for g in groups]
    centred = []
    for g in groups:
        m = sum(g) / len(g)
        c = [r - m for r in g]
        centred.append(c if max(abs(x) for x in c) > 1e-8 else [0.0] * len(g))
    if mode == "none":
        return centred
    informative = [x for c in centred if any(c) for x in c]
    if not informative:
        return centred
    scale = math.sqrt(sum(x * x for x in informative) / len(informative)) + 1e-8
    return [[x / scale for x in c] for c in centred]


def full_logits(model: GPT, idx: torch.Tensor, start: int) -> torch.Tensor:
    """Logits for positions start..T-1 (the full forward pass, lm_head on the tail only)."""
    T = idx.shape[1]
    x = model.drop(model.tok_emb(idx))
    cos, sin = model.rope_cos[:, :, :T], model.rope_sin[:, :, :T]
    for blk in model.blocks:
        x, _ = blk(x, cos, sin)
    logits = model.lm_head(model.norm(x[:, start:]))
    if model.cfg.mup:
        logits = logits * model.output_mult.to(logits.dtype)
    return logits


def allreduce_grads(model: GPT, world_size: int) -> None:
    """SUM gradients across ranks; each rank's loss is already divided by the
    GLOBAL normalizer, so the sum is the full-batch gradient."""
    if world_size == 1:
        return
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


def all_gather(obj: Any, world_size: int) -> List[Any]:
    if world_size == 1:
        return [obj]
    out: List[Any] = [None] * world_size
    dist.all_gather_object(out, obj)
    return out


class KLController:
    """beta for the reference-KL term.

    fixed        — one objective for the whole run.
    proportional — Ziegler et al. (2019): beta *= 1 + rate * clip(KL_ema/target - 1, +-0.2),
                   at most +-(0.2*rate) per step, on an EMA so one noisy batch cannot swing it.
    legacy       — the v3 bang-bang controller (x1.5 outside [0.5, 2] x target, floor 0.005).
    """

    def __init__(self, mode: str, beta: float, target: float, rate: float = 0.1,
                 ema: float = 0.9, beta_min: float = 1e-3, beta_max: float = 1.0):
        assert mode in ("fixed", "proportional", "legacy"), mode
        self.mode, self.beta, self.target, self.rate = mode, beta, target, rate
        self.ema_coeff, self.beta_min, self.beta_max = ema, beta_min, beta_max
        self.kl_ema: Optional[float] = None

    def update(self, kl: float) -> float:
        self.kl_ema = kl if self.kl_ema is None else self.ema_coeff * self.kl_ema + (1 - self.ema_coeff) * kl
        if self.mode == "proportional":
            err = max(-0.2, min(0.2, self.kl_ema / self.target - 1.0))
            self.beta = min(self.beta_max, max(self.beta_min, self.beta * (1.0 + self.rate * err)))
        elif self.mode == "legacy":
            if kl > 2.0 * self.target:
                self.beta = min(self.beta * 1.5, 1.0)
            elif kl < 0.5 * self.target:
                self.beta = max(self.beta / 1.5, 0.005)
        return self.beta


class PromptSampler:
    """Without-replacement prompt schedule; optionally a fixed type mix per batch.

    With `stratify`, each type keeps its own shuffled queue and a running credit
    of `n * pool_fraction`, so every batch carries the pool's type proportions
    to within one prompt and the long-run mix is exact.
    """

    def __init__(self, items: List[Dict[str, Any]], seed: int, stratify: bool):
        self.rng = random.Random(seed)
        self.stratify = stratify
        by_type: Dict[str, List[int]] = defaultdict(list)
        for i, it in enumerate(items):
            by_type[it["type"] if stratify else "all"].append(i)
        self.pools = dict(sorted(by_type.items()))
        n = len(items)
        self.frac = {t: len(ix) / n for t, ix in self.pools.items()}
        self.credit = {t: 0.0 for t in self.pools}
        self.queues: Dict[str, List[int]] = {t: [] for t in self.pools}

    def _pop(self, t: str) -> int:
        if not self.queues[t]:
            self.queues[t] = list(self.pools[t])
            self.rng.shuffle(self.queues[t])
        return self.queues[t].pop()

    def draw(self, n: int) -> List[int]:
        for t in self.credit:
            self.credit[t] += n * self.frac[t]
        counts = {t: max(0, math.floor(c)) for t, c in self.credit.items()}
        for t in sorted(self.credit, key=lambda t: self.credit[t] - counts[t], reverse=True)[:n - sum(counts.values())]:
            counts[t] += 1
        out = []
        for t, k in counts.items():
            self.credit[t] -= k
            out.extend(self._pop(t) for _ in range(k))
        self.rng.shuffle(out)
        return out


# ===========================================================================
# Scoring and summaries
# ===========================================================================

TOKENS_CALL = "<<"   # a calculator call (src.rl.rollout, `calculator=True`)


def correctness_reward(text: str, score: Dict[str, Any], require_program: bool, notation: bool = False) -> float:
    """reward_mode="correctness" (v9): the only signal is whether the executed answer is right.

    1 within 1%, 0.5 within 5%, else 0 -- and 0 if the program does not run (with pot), so a
    model-written number cannot collect reward when the program fails. Format, termination and
    concept terms are dropped: after SFT they are saturated, yet their gradient points the same
    way on every prompt and so dominated v8's updates while the per-problem correctness signal
    cancelled out (audit §15.12)."""
    if require_program and not program_runs(text, notation):
        return 0.0
    return max(0.0, float(score["r_num"]))


def program_runs(text: str, notation: bool = False) -> bool:
    if "<think>" not in text or "</think>" not in text:
        return False
    body = text[text.find("<think>") + len("<think>"):text.find("</think>")]
    return run_program(body, notation=notation)[0] is not None


def execution_reward(text: str, score: Dict[str, Any], notation: bool = False) -> float:
    """reward_mode="execution" (v10): 0 if the program does not run, 0.2 if it runs but is wrong,
    1 if the executed answer is within 5%.

    The failure analysis (audit §15.13) found 45.5% of the student's programs crash -- 30% of all
    samples on an undefined variable name, 12% on syntax -- and v9 scored a crash and a wrong
    answer alike (0), so nothing rewarded running code. Running is a skill shared by every
    problem, so its gradient adds up across prompts instead of cancelling."""
    if not program_runs(text, notation):
        return 0.0
    return 1.0 if score["r_num"] >= 0.5 else 0.2


STEP_WEIGHT = 0.3            # below 0.5, so no wrong answer outscores one within 5%
_TEACHER_STEPS: Dict[Any, List[float]] = {}


def _rel_close(a: float, b: float, tol: float = 0.01) -> bool:
    d = abs(b) if abs(b) > 1e-12 else 1.0
    return abs(a - b) / d <= tol


def _computed_values(env: Dict[str, float], givens: List[float]) -> List[float]:
    return [v for v in env.values() if isinstance(v, (int, float)) and math.isfinite(v)
            and not any(_rel_close(v, g) for g in givens)]


def step_f1(text: str, item: Dict[str, Any], notation: bool = False) -> float:
    """F1 between the values the student's program computes and those the teacher's program computes
    (`item["program"]`), within 1%, numbers stated in the question excluded. Recall credits the
    teacher's steps the student reached; precision stops a student from spraying extra lines."""
    from src.rl.verifiers import NumericalVerifier
    if not item.get("program") or "<think>" not in text or "</think>" not in text:
        return 0.0
    givens = NumericalVerifier().extract_numbers(item["prompt"])
    key = item.get("id")
    if key not in _TEACHER_STEPS:
        _TEACHER_STEPS[key] = _computed_values(run_program(item["program"])[1], givens)
    steps = _TEACHER_STEPS[key]
    body = text[text.find("<think>") + len("<think>"):text.find("</think>")]
    mine = _computed_values(run_program(body, notation=notation)[1], givens)
    if not steps or not mine:
        return 0.0
    recall = sum(any(_rel_close(t, s) for s in mine) for t in steps) / len(steps)
    precision = sum(any(_rel_close(s, t) for t in steps) for s in mine) / len(mine)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def step_reward(text: str, score: Dict[str, Any], item: Dict[str, Any], notation: bool = False) -> float:
    """reward_mode="step" (v16): the correctness reward plus STEP_WEIGHT x step_f1, so a problem that
    no sample solves still ranks its samples by how much of the teacher's working they reproduce.
    0 if the program does not run, as in "correctness"."""
    if not program_runs(text, notation):
        return 0.0
    return max(0.0, float(score["r_num"])) + STEP_WEIGHT * step_f1(text, item, notation)


def score_rollouts(engine, tokenizer, items, completions, terminated, group_size,
                   reward_mode: str = "composite", require_program: bool = False, notation: bool = False):
    """Score prompt-major rollouts; returns one list of score dicts per group."""
    groups = []
    for gi, item in enumerate(items):
        rows = range(gi * group_size, (gi + 1) * group_size)
        gr = []
        for r in rows:
            text = decode_completion(tokenizer, completions[r])
            sc = engine.score(text, item, terminated=terminated[r])
            is_num = item["type"] == "numerical" and item["final_numerical_value"] is not None
            if reward_mode == "correctness" and is_num:
                sc["total_reward"] = correctness_reward(text, sc, require_program, notation)
            elif reward_mode == "execution" and is_num:
                sc["total_reward"] = execution_reward(text, sc, notation)
            elif reward_mode == "step" and is_num:
                sc["total_reward"] = step_reward(text, sc, item, notation)
            runs = float(program_runs(text, notation)) if require_program else float("nan")
            gr.append({**sc, "type": item["type"], "is_numerical": is_num, "len": len(completions[r]),
                       "calc": float(TOKENS_CALL in text), "terminated": bool(terminated[r]), "runs": runs})
        groups.append(gr)
    return groups


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def summarize(groups: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """Batch metrics over every rollout of every group (the caller gathers all ranks)."""
    flat = [s for g in groups for s in g]
    num = [s for s in flat if s["is_numerical"]]
    other = [s for s in flat if not s["is_numerical"]]
    out = {
        "mean_reward": _mean(s["total_reward"] for s in flat),
        "reward_numerical": _mean(s["total_reward"] for s in num),
        "reward_conceptual": _mean(s["total_reward"] for s in other),
        "num_acc1": _mean(float(s["r_num"] >= 1.0) for s in num),
        "think_rate": _mean(float(s["r_format"] > 0) for s in num),   # numerical items expect <think>
        "stop_rate": _mean(float(s["terminated"]) for s in flat),
        "mean_comp_tokens": _mean(s["len"] for s in flat),
        "calc_rate": _mean(s["calc"] for s in num),            # numerical rollouts calling the calculator
        "prog_run_rate": _mean(s["runs"] for s in num),        # pot: programs that execute
        # Reward opportunity: only within-group spread creates gradient (README 5.5).
        "zero_var_frac": _mean(float(_std(s["total_reward"] for s in g) < 1e-8) for g in groups),
        "group_std": _mean(_std(s["total_reward"] for s in g) for g in groups),
    }
    for c in COMPONENTS:
        out[c] = _mean(s[c] for s in flat)
    for c in ("r_num", "r_concept", "r_format", "r_term"):
        out[f"std_{c}"] = _mean(_std(s[c] for s in g) for g in groups)
    return out


@torch.no_grad()
def evaluate_heldout(model, tokenizer, engine, items, prompt_ids, cfg, device, seed) -> List[List[Dict]]:
    """Expected reward under the SAMPLING policy on fixed held-out prompts.

    This is the RL objective itself, measured on prompts the optimizer never
    sees, with the same generator seed at every evaluation (common random
    numbers), so successive points differ mainly by the policy.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    groups = []
    bs = cfg["eval_batch_prompts"]
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        comps, term = generate_rollouts(model, prompt_ids[i:i + bs], cfg["eval_samples"],
                                        cfg["max_new_tokens"], cfg["temperature"], cfg["top_p"],
                                        generator=gen, calculator=cfg["calculator"],
                                        tokenizer=tokenizer, pot=cfg["pot"], pot_notation=cfg["pot_notation"])[:2]
        groups.extend(score_rollouts(engine, tokenizer, chunk, comps, term, cfg["eval_samples"],
                                     cfg["reward_mode"], cfg["pot"], cfg["pot_notation"]))
    return groups


TRAIN_COLUMNS = [
    "step", "mean_reward", "num_reward", "concept_reward", "format_reward", "clean_reward",
    "policy_loss", "kl_div", "kl_coeff", "entropy", "mean_comp_tokens", "stop_rate", "lr",
    "elapsed_sec",
    # v4 additions
    "reward_numerical", "reward_conceptual", "num_acc1", "think_rate", "r_echo", "r_length",
    "r_term", "zero_var_frac", "group_std", "std_r_num", "std_r_concept", "std_r_format",
    "std_r_term", "kl_loss", "grad_norm", "n_rollouts", "gen_sec", "step_sec",
    "update_sec", "clip_frac", "tis_w_mean", "tis_trunc_frac", "sampler_mismatch",
    # v6 (calculator)
    "calc_rate", "forced_frac",
    # v10 (program execution)
    "prog_run_rate",
]
EVAL_COLUMNS = ["step", "mean_reward", "reward_numerical", "reward_conceptual", "num_acc1",
                "think_rate", "stop_rate", "mean_comp_tokens", "r_num", "r_concept",
                "r_format", "r_term", "zero_var_frac", "n_rollouts", "calc_rate", "prog_run_rate"]


def _fmt(v):
    return f"{v:.5g}" if isinstance(v, float) else str(v)


# ===========================================================================
# Training loop
# ===========================================================================

def train_grpo(config_path: Path) -> None:
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    g = config["grpo"]
    cfg = {
        "group_size": int(g.get("group_size", 8)),
        "prompts_per_step": int(g.get("prompts_per_step", 1)),
        "micro_batch_rows": int(g.get("micro_batch_rows", 16)),
        "max_new_tokens": int(g.get("max_new_tokens", 384)),
        "temperature": float(g.get("temperature", 1.0)),
        "top_p": float(g.get("top_p", 1.0)),
        "max_steps": int(g.get("max_steps", 200)),
        "lr": float(g["learning_rate"]),
        "warmup_steps": int(g.get("warmup_steps", 0)),
        "grad_clip": float(g.get("grad_clip", 1.0)),
        "adv_norm": g.get("adv_norm", "group"),
        "loss_agg": g.get("loss_agg", "seq_mean"),
        "kl_loss": g.get("kl_loss", "k3"),
        "kl_controller": g.get("kl_controller", "legacy"),
        "kl_coeff": float(g.get("kl_coeff", 0.04)),
        "kl_target": float(g.get("kl_target", 0.05)),
        "kl_rate": float(g.get("kl_rate", 0.1)),
        "kl_stop": float(g.get("kl_stop", 0.5)),
        "num_iterations": int(g.get("num_iterations", 1)),
        "epsilon": float(g.get("epsilon", 0.2)),
        "epsilon_high": float(g.get("epsilon_high", g.get("epsilon", 0.2))),
        "tis_cap": (float(g["tis_cap"]) if g.get("tis_cap") is not None else None),
        "calculator": bool(g.get("calculator", False)),
        "pot": bool(g.get("pot", False)),
        "pot_notation": bool(g.get("pot_notation", False)),
        "reward_mode": g.get("reward_mode", "composite"),
        "stratify_types": bool(g.get("stratify_types", False)),
        "seed": int(g.get("seed", 42)),
        "eval_every": int(g.get("eval_every", 0)),
        "eval_prompts": int(g.get("eval_prompts", 0)),
        "eval_samples": int(g.get("eval_samples", 2)),
        "eval_batch_prompts": int(g.get("eval_batch_prompts", 32)),
    }
    assert cfg["adv_norm"] in ("group", "batch", "none")
    assert cfg["loss_agg"] in ("seq_mean", "token_mean", "constant")
    assert cfg["kl_loss"] in ("k3", "k2")
    assert cfg["reward_mode"] in ("composite", "correctness", "execution", "step")
    assert cfg["num_iterations"] >= 1
    use_tis = cfg["tis_cap"] is not None

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"] + 1000 * rank)

    if is_main:
        print("=" * 80)
        print("Stage 3 GRPO training for nanoGPT-Seis (v4)")
        print(f"   World size: {world_size} | device: {device}")
        print("   " + " | ".join(f"{k}={v}" for k, v in cfg.items()))
        print("=" * 80, flush=True)

    # The reference is always the SFT model; the policy may start elsewhere
    # (`policy_init`, e.g. a mid-run RL checkpoint for an intervention test).
    base_ckpt = ROOT / config["model"]["base_checkpoint"]
    ref_model = load_sft_model(base_ckpt, device)
    policy = load_sft_model(ROOT / config["model"].get("policy_init", config["model"]["base_checkpoint"]), device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    if world_size > 1:  # identical start on every rank
        for p in policy.parameters():
            dist.broadcast(p.data, src=0)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], weight_decay=0.0, betas=(0.9, 0.95))

    def load_items(path):
        with open(ROOT / path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    rl_prompts = load_items(config["data"]["train_file"])
    missing = [k for k in ("prompt", "type", "final_numerical_value", "key_concepts")
               if any(k not in item for item in rl_prompts)]
    assert not missing, f"RL dataset schema mismatch, missing fields: {missing} — regenerate with src.rl.prepare_rl_dataset"
    n_num = sum(1 for it in rl_prompts if it["type"] == "numerical")
    n_num_target = sum(1 for it in rl_prompts if it["type"] == "numerical" and it["final_numerical_value"] is not None)
    assert n_num == 0 or n_num_target > 0.5 * n_num, \
        f"NumericalVerifier would be mostly dead: {n_num_target}/{n_num} numerical items have targets"

    tok_path = ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json"
    tokenizer = load_bpe_tokenizer(tok_path)
    reward_engine = CompositeRewardEngine()

    # Fixed held-out prompts: a type-stratified prefix of a seeded shuffle of the
    # validation split, sharded across ranks.
    eval_items: List[Dict[str, Any]] = []
    chosen: List[int] = []
    if cfg["eval_every"] > 0 and cfg["eval_prompts"] > 0:
        val = load_items(config["data"]["val_file"])
        picker = PromptSampler(val, seed=cfg["seed"] + 7, stratify=True)
        chosen = picker.draw(min(cfg["eval_prompts"], len(val)))
        eval_items = [val[i] for i in chosen][rank::world_size]
    eval_prompt_ids = [build_prompt_ids(tokenizer, it["prompt"]) for it in eval_items]
    if is_main:
        print(f"[INFO] RL prompts: {len(rl_prompts)} | numerical with target: {n_num_target}/{n_num} | "
              f"held-out eval prompts: {len(chosen)}", flush=True)

    output_dir = ROOT / config["save"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv = ROOT / config["save"]["log_file"]
    eval_csv = log_csv.with_name(log_csv.stem + "_eval.csv")
    if is_main:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(TRAIN_COLUMNS)
        if eval_items:
            with open(eval_csv, "w", newline="") as f:
                csv.writer(f).writerow(EVAL_COLUMNS)

    def run_eval(step: int) -> None:
        if not eval_items:
            return
        t0 = time.time()
        groups = evaluate_heldout(policy, tokenizer, reward_engine, eval_items, eval_prompt_ids,
                                  cfg, device, seed=cfg["seed"] + 99 + rank)
        groups = [gr for part in all_gather(groups, world_size) for gr in part]
        if is_main:
            m = summarize(groups)
            m["n_rollouts"] = sum(len(gr) for gr in groups)
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([step] + [_fmt(m[c]) for c in EVAL_COLUMNS[1:]])
            print(f"[EVAL] step {step:3d} | held-out reward {m['mean_reward']:+.4f} "
                  f"(num {m['reward_numerical']:+.3f} / concept {m['reward_conceptual']:+.3f}) | "
                  f"acc@1% {m['num_acc1']:.3f} | think {m['think_rate']:.2f} | stop {m['stop_rate']:.2f} | "
                  f"len {m['mean_comp_tokens']:.0f} | calc {m['calc_rate']:.2f} | runs {m['prog_run_rate']:.2f} | {time.time() - t0:.0f}s", flush=True)

    sampler = PromptSampler(rl_prompts, seed=cfg["seed"], stratify=cfg["stratify_types"])
    kl_ctl = KLController(cfg["kl_controller"], cfg["kl_coeff"], cfg["kl_target"], rate=cfg["kl_rate"])
    kl_window: List[float] = []
    G, P = cfg["group_size"], cfg["prompts_per_step"]
    start_time = time.time()

    eval_every = cfg["eval_every"]
    if eval_every > 0:
        run_eval(0)

    for step in range(1, cfg["max_steps"] + 1):
        t_step = time.time()
        lr = cfg["lr"] * min(1.0, step / cfg["warmup_steps"]) if cfg["warmup_steps"] > 0 else cfg["lr"]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Every rank draws the same global batch (same seed), then takes its shard.
        global_idx = sampler.draw(P * world_size)
        items = [rl_prompts[i] for i in global_idx[rank::world_size]]
        prompt_ids = [build_prompt_ids(tokenizer, it["prompt"]) for it in items]

        # 1. Rollouts (one CUDA-graph decode; the sampler's own log-probs kept for TIS)
        #    With the calculator, " = value" after each `<<expr>>` is written by the environment;
        #    those tokens are observations, not actions, and are masked out of the loss below.
        roll = generate_rollouts(policy, prompt_ids, G, cfg["max_new_tokens"], cfg["temperature"],
                                 cfg["top_p"], return_logprobs=use_tis, calculator=cfg["calculator"],
                                 tokenizer=tokenizer, pot=cfg["pot"], pot_notation=cfg["pot_notation"])
        completions, terminated = roll[0], roll[1]
        samp_lp = roll[2] if use_tis else None
        #    With pot, "Answer: <value>" after </think> is the interpreter's output: masked the same way.
        forced = roll[-1] if (cfg["calculator"] or cfg["pot"]) else None
        n_forced = sum(sum(f) for f in forced) if forced else 0
        gen_sec = time.time() - t_step

        # 2. Score, gather the global batch (rewards are tiny), compute advantages
        local_groups = score_rollouts(reward_engine, tokenizer, items, completions, terminated, G,
                                      cfg["reward_mode"], cfg["pot"], cfg["pot_notation"])
        all_groups = all_gather(local_groups, world_size)
        global_groups = [gr for part in all_groups for gr in part]
        offset = sum(len(part) for part in all_groups[:rank])
        advs = compute_batch_advantages([[s["total_reward"] for s in gr] for gr in global_groups],
                                        cfg["adv_norm"])[offset:offset + len(local_groups)]
        n_rows_global = len(global_groups) * G
        n_tok_global = sum(s["len"] for gr in global_groups for s in gr)
        normalizer = {"constant": n_rows_global * cfg["max_new_tokens"],
                      "token_mean": max(n_tok_global, 1),
                      "seq_mean": n_rows_global}[cfg["loss_agg"]]

        # 3. Policy-gradient passes: `num_iterations` optimizer steps on this rollout batch.
        #    Iteration 0 is on-policy (ratio = 1, so the gradient is exactly -A grad log p);
        #    later iterations reuse the rollouts with the clipped ratio to the iteration-0 policy.
        #    TIS weights each token by min(p_train / p_sampler, cap): the bf16 KV-cached sampler
        #    and the training forward pass differ numerically (mean |dlogp| ~ 0.02).
        flat_adv = [a for gr in advs for a in gr]
        prompt_len = [len(p) for p in prompt_ids for _ in range(G)]
        # pg_sum, kl_loss_sum, kl_k3_sum, entropy_sum, n_tok, clipped_tok, later_tok, tis_w_sum, tis_trunc, mismatch_sum
        stats = torch.zeros(10, device=device)
        mb = cfg["micro_batch_rows"]
        cache: List[Dict[str, torch.Tensor]] = []
        t_upd = time.time()
        for it in range(cfg["num_iterations"]):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            for b, lo in enumerate(range(0, len(completions), mb)):
                if it == 0:
                    rows = list(range(lo, min(lo + mb, len(completions))))
                    seqs = [prompt_ids[r // G] + completions[r] for r in rows]
                    T = max(len(q) for q in seqs)
                    idx = torch.full((len(rows), T), IM_END_ID, dtype=torch.long, device=device)
                    for jj, q in enumerate(seqs):
                        idx[jj, :len(q)] = torch.tensor(q, dtype=torch.long, device=device)
                    start = min(prompt_len[r] for r in rows) - 1    # first position predicting a completion token
                    targets = idx[:, start + 1:]
                    mask = torch.zeros_like(targets, dtype=torch.float32)
                    samp = torch.zeros_like(mask)
                    for jj, r in enumerate(rows):
                        a = prompt_len[r] - 1 - start
                        mask[jj, a:a + len(completions[r])] = 1.0
                        if forced is not None and any(forced[r]):
                            mask[jj, a:a + len(completions[r])] -= torch.tensor(forced[r], device=device, dtype=torch.float32)
                        if use_tis:
                            samp[jj, a:a + len(completions[r])] = torch.tensor(samp_lp[r], device=device)
                    adv = torch.tensor([flat_adv[r] for r in rows], device=device).unsqueeze(-1)
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=device.type == "cuda"):
                        ref_logits = full_logits(ref_model, idx[:, :-1], start)
                    with torch.no_grad():
                        ref_logp = F.log_softmax(ref_logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    del ref_logits
                    c = {"idx": idx, "start": start, "targets": targets, "mask": mask, "adv": adv,
                         "ref_logp": ref_logp, "samp": samp}
                    cache.append(c)
                else:
                    c = cache[b]
                idx, start, targets, mask, adv = c["idx"], c["start"], c["targets"], c["mask"], c["adv"]

                with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=device.type == "cuda"):
                    logits = full_logits(policy, idx[:, :-1], start)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                if it == 0:
                    c["old_logp"] = logp.detach()
                    if use_tis:
                        raw = torch.exp(c["old_logp"] - c["samp"])
                        c["tis_w"] = raw.clamp(max=cfg["tis_cap"])
                        with torch.no_grad():
                            stats[7] += (c["tis_w"] * mask).sum()
                            stats[8] += ((raw > cfg["tis_cap"]).float() * mask).sum()
                            stats[9] += ((c["old_logp"] - c["samp"]).abs() * mask).sum()
                    else:
                        c["tis_w"] = torch.ones_like(mask)
                ratio = torch.exp(logp - c["old_logp"])
                surrogate = torch.minimum(ratio * adv,
                                          ratio.clamp(1.0 - cfg["epsilon"], 1.0 + cfg["epsilon_high"]) * adv)
                log_ratio = logp - c["ref_logp"]                  # policy minus reference
                kl_k3 = torch.exp(-log_ratio) + log_ratio - 1.0   # k3, >= 0, estimates KL(policy || ref)
                kl_term = kl_k3 if cfg["kl_loss"] == "k3" else 0.5 * log_ratio.pow(2)

                pg_tok = -surrogate * c["tis_w"] * mask
                kl_tok = kl_term * mask
                if cfg["loss_agg"] == "seq_mean":
                    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    pg_tok, kl_tok = pg_tok / denom, kl_tok / denom
                loss = (pg_tok.sum() + kl_ctl.beta * kl_tok.sum()) / normalizer
                loss.backward()

                with torch.no_grad():
                    if it == 0:
                        ent = -(log_probs.exp() * log_probs).sum(dim=-1)
                        stats[:5] += torch.stack([pg_tok.sum(), kl_tok.sum(), (kl_k3 * mask).sum(),
                                                  (ent * mask).sum(), mask.sum()])
                    else:
                        outside = (ratio < 1.0 - cfg["epsilon"]) | (ratio > 1.0 + cfg["epsilon_high"])
                        stats[5] += (outside.float() * mask).sum()
                        stats[6] += mask.sum()
                del logits, log_probs

            allreduce_grads(policy, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg["grad_clip"])
            optimizer.step()
        update_sec = time.time() - t_upd

        # 4. Global KL -> controller and early-stop guard
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        pg_sum, kl_loss_sum, kl_sum, ent_sum, n_tok, clipped, later_tok, tis_sum, tis_trunc, mismatch = stats.tolist()
        n_forced_g = float(sum(all_gather(n_forced, world_size)))
        kl_scalar = kl_sum / max(n_tok, 1.0)
        beta_used = kl_ctl.beta
        kl_ctl.update(kl_scalar)
        kl_window = (kl_window + [kl_scalar])[-20:]
        rolling_kl = sum(kl_window) / len(kl_window)
        stop = len(kl_window) == 20 and rolling_kl > cfg["kl_stop"]

        # 5. Logging (every number covers the whole global batch)
        if is_main:
            m = summarize(global_groups)
            m.update({
                "step": step, "num_reward": m["r_num"], "concept_reward": m["r_concept"],
                "format_reward": m["r_format"], "clean_reward": m["r_clean"],
                "policy_loss": pg_sum / normalizer, "kl_loss": kl_loss_sum / normalizer,
                "kl_div": kl_scalar, "kl_coeff": beta_used, "entropy": ent_sum / max(n_tok, 1.0),
                "lr": lr, "elapsed_sec": time.time() - start_time, "grad_norm": float(grad_norm),
                "n_rollouts": n_rows_global, "gen_sec": gen_sec, "step_sec": time.time() - t_step,
                "update_sec": update_sec, "clip_frac": clipped / later_tok if later_tok else 0.0,
                "tis_w_mean": tis_sum / max(n_tok, 1.0) if use_tis else 1.0,
                "tis_trunc_frac": tis_trunc / max(n_tok, 1.0) if use_tis else 0.0,
                "sampler_mismatch": mismatch / max(n_tok, 1.0) if use_tis else float("nan"),
                "forced_frac": n_forced_g / max(n_tok + n_forced_g, 1.0),
            })
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([_fmt(m[c]) for c in TRAIN_COLUMNS])
            print(f"step {step:3d}/{cfg['max_steps']} | R {m['mean_reward']:+.3f} "
                  f"(num {m['reward_numerical']:+.2f} / con {m['reward_conceptual']:+.2f}) | "
                  f"acc {m['num_acc1']:.2f} think {m['think_rate']:.2f} stop {m['stop_rate']:.2f} "
                  f"len {m['mean_comp_tokens']:.0f} | zero-var {m['zero_var_frac']:.2f} | "
                  f"KL {kl_scalar:.4f} β {beta_used:.3f} | H {m['entropy']:.2f} | |g| {float(grad_norm):.2f} | "
                  f"clip {m['clip_frac']:.3f} tis {m['tis_w_mean']:.3f} calc {m['calc_rate']:.2f} runs {m['prog_run_rate']:.2f} | "
                  f"{m['step_sec']:.1f}s (gen {gen_sec:.1f}s, update {update_sec:.1f}s)", flush=True)

        # 6. Held-out objective, then checkpoint (weights only)
        if eval_every > 0 and (step % eval_every == 0 or step == cfg["max_steps"]):
            run_eval(step)
        if is_main and (step % config["save"].get("save_every", 100) == 0 or step == cfg["max_steps"] or stop):
            ckpt_save_path = output_dir / f"grpo_ckpt_step_{step}.pt"
            torch.save({"step": step, "model": policy.state_dict(), "model_cfg": policy.cfg.__dict__},
                       ckpt_save_path)
            print(f"  [SAVED] Checkpoint -> {ckpt_save_path}", flush=True)

        if stop:
            if is_main:
                print(f"[EARLY STOP] rolling KL {rolling_kl:.3f} > kl_stop {cfg['kl_stop']} at step {step} — "
                      f"policy is drifting from the SFT reference; stopping before degeneration.")
            break

    cleanup_ddp()
    if is_main:
        print("=" * 80)
        print("Stage 3 GRPO training complete")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training Engine for nanoGPT-Seis (v4)")
    parser.add_argument("--config", type=str, default="configs/rl_grpo.yaml", help="Path to GRPO YAML config")
    args = parser.parse_args()
    train_grpo(ROOT / args.config)


if __name__ == "__main__":
    main()
