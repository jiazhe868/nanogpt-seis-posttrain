"""Continued pretraining of the 113M base on a weighted mixture of token streams (scripts/midtrain/build_mix.py).

    torchrun --standalone --nproc_per_node=2 scripts/midtrain/train_mid.py --config configs/midtrain/mid8b.yaml [--resume]

The loop is ../nanogpt_seis/src/train.py (bf16, DDP, gradient accumulation, cosine LR after a linear re-warm) with
two changes: each sequence is drawn from a source chosen by the `mix` weights, and training starts from `init`
(weights only; a fresh optimizer). The last 0.5% of every stream is held back to report a loss per source next to the
original validation split. Checkpoints keep the base format ("model", "model_cfg"), so src.sft.train_sft loads them.
"""
import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from nanogpt_seis.src.model.gqa_gpt import GPT, GPTConfig  # noqa: E402

HOLD = 0.005


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-iters", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    tc = cfg["train"]
    if a.max_iters is not None:
        tc["max_iters"] = a.max_iters

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0; world = 1; device = "cuda"
    master = rank == 0
    torch.manual_seed(tc["seed"] + rank)
    rng = np.random.default_rng(tc["seed"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # ---- data: one memmap per source, the last HOLD of each held back for evaluation ----
    data_dir = ROOT / cfg["data_dir"]
    names = list(cfg["mix"])
    w = np.array([cfg["mix"][n] for n in names], dtype=np.float64); w /= w.sum()
    streams = {n: np.memmap(data_dir / f"{n}.bin", dtype=np.uint16, mode="r") for n in names}
    cut = {n: int(len(s) * (1 - HOLD)) for n, s in streams.items()}
    val = np.memmap(ROOT / cfg["val"], dtype=np.uint16, mode="r")
    B, T, accum = tc["batch_size"], tc["block_size"], tc["grad_accum"]

    def take(s, lo, hi, n):
        ix = rng.integers(lo, hi - T - 1, size=n)
        return np.stack([s[i:i + T + 1].astype(np.int64) for i in ix])

    def batch(split=None):
        if split is None:                                   # training: a source per sequence
            src = rng.choice(len(names), size=B, p=w)
            rows = np.concatenate([take(streams[names[k]], 0, cut[names[k]], int((src == k).sum()))
                                   for k in range(len(names)) if (src == k).any()])
        elif split == "val":
            rows = take(val, 0, len(val), B)
        else:
            rows = take(streams[split], cut[split], len(streams[split]), B)
        t = torch.from_numpy(rows).pin_memory().to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:]

    # ---- model ----
    init = torch.load(ROOT / cfg["init"], map_location="cpu")
    model_cfg = dict(init["model_cfg"]); model_cfg["block_size"] = T
    model = GPT(GPTConfig(**{k: model_cfg[k] for k in GPTConfig.__dataclass_fields__ if k in model_cfg})).to(device)
    out_dir = ROOT / tc["out_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    last = out_dir / "ckpt_last.pt"
    it = 0
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); it = ck["iter_num"]
    else:
        model.load_state_dict(init["model"])
    del init
    raw = model
    opt = raw.configure_optimizers(tc["weight_decay"], tc["lr"], (tc["beta1"], tc["beta2"]), "cuda")
    if a.resume and last.exists():
        opt.load_state_dict(ck["optimizer"]); del ck
    if tc.get("compile", True):
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    if master:
        print(f"[mid] tokens/iter {accum * world * B * T:,} | mix " +
              ", ".join(f"{n} {p:.3f} ({len(streams[n]) / 1e9:.2f}B)" for n, p in zip(names, w)), flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for split in ["val"] + names:
            ls = []
            for _ in range(tc["eval_iters"]):
                x, y = batch(split)
                with ctx:
                    ls.append(model(x, y)[1].item())
            out[split] = sum(ls) / len(ls)
        model.train()
        return out

    def save(path, with_opt):
        state = {"model": raw.state_dict(), "model_cfg": model_cfg, "iter_num": it, "mix": cfg["mix"]}
        if with_opt:
            state["optimizer"] = opt.state_dict()
        torch.save(state, path)

    csv = out_dir / "log.csv"
    if master and not (a.resume and csv.exists()):
        csv.write_text("iter,lr," + ",".join(["val"] + names) + "\n")
    x, y = batch()
    t0 = time.time()
    while it <= tc["max_iters"]:
        lr = get_lr(it, tc["warmup_iters"], tc["max_iters"], tc["lr"], tc["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        if it % tc["eval_interval"] == 0 or it == tc["max_iters"]:
            ev = evaluate()
            if master:
                print(f"[eval] iter {it} lr {lr:.2e} | " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
                with csv.open("a") as f:
                    f.write(f"{it},{lr:.3e}," + ",".join(f"{ev[k]:.4f}" for k in ["val"] + names) + "\n")
        if master and it > 0 and (it % tc["save_every"] == 0 or it == tc["max_iters"]):
            save(out_dir / f"ckpt_iter{it}.pt", False)
            save(last, True)
        if it == tc["max_iters"]:
            break
        for micro in range(accum):
            if ddp:
                model.require_backward_grad_sync = micro == accum - 1
            with ctx:
                loss = model(x, y)[1] / accum
            x, y = batch()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if master and it % tc["log_interval"] == 0 and it > 0:
            dt = (time.time() - t0) / tc["log_interval"]; t0 = time.time()
            print(f"[mid] iter {it}: loss {loss.item() * accum:.4f} {dt * 1000:.0f} ms/iter", flush=True)
        it += 1
    if master:
        save(out_dir / "ckpt.pt", False)
        print(f"[mid] done -> {out_dir / 'ckpt.pt'}", flush=True)
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
