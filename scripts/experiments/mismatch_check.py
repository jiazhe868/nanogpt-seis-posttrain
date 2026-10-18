"""Train-inference mismatch check (README §5.16): how far the rollout engine's log-probs are from the
training engine's on the same tokens.

Samples completions from the Phase IV checkpoint (notation2) with the rollout engine (src/rl/rollout.py: KV-cached,
one token at a time) and re-scores the same tokens with the trainer's full forward pass
(src/rl/train_grpo.py: full_logits + fp32 log-softmax). Runs on CPU in fp32, so the GPUs stay free; bf16
on the trainer side is emulated with CPU autocast. Writes docs/mismatch_check.json.

    python scripts/experiments/mismatch_check.py
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl import rollout  # noqa: E402
from src.rl.train_grpo import build_prompt_ids, full_logits, load_sft_model  # noqa: E402
from src.sft.dataset import load_bpe_tokenizer  # noqa: E402

CKPT = "checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt"
PROMPTS = "data/rl/val_numerical_dev.jsonl"
N_PROMPTS, G, NEW, SEED, CAP = 8, 4, 128, 0, 2.0


def main() -> None:
    torch.set_num_threads(32)
    dev = torch.device("cpu")
    model = load_sft_model(ROOT / CKPT, dev).eval()
    tok = load_bpe_tokenizer(ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json")
    items = [json.loads(l) for l in open(ROOT / PROMPTS, encoding="utf-8")][:N_PROMPTS]
    prompts = [build_prompt_ids(tok, it["prompt"]) for it in items]

    def trainer_logp(completions, temperature=1.0, bf16=False):
        """Per-token log-probs of the sampled tokens, computed as train_grpo does."""
        out = []
        for r, comp in enumerate(completions):
            p = prompts[r // G]
            idx = torch.tensor([p + comp], dtype=torch.long)
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                logits = full_logits(model, idx[:, :-1], len(p) - 1)
            lp = F.log_softmax(logits.float() / temperature, dim=-1)
            out.append(lp[0].gather(-1, torch.tensor(comp)[:, None]).squeeze(-1))
        return out

    def compare(train, samp, p_train):
        d = torch.cat([a - b for a, b in zip(train, samp)])               # log pi_train - log mu_sampler
        seq = torch.tensor([float((a - b).sum()) for a, b in zip(train, samp)])
        p = torch.cat(p_train).exp()
        bins = {"p < 0.1": p < 0.1, "0.1 <= p < 0.9": (p >= 0.1) & (p < 0.9), "p >= 0.9": p >= 0.9}
        return {"mean_abs_dlogp": float(d.abs().mean()), "max_abs_dlogp": float(d.abs().max()),
                "mean_weight": float(d.exp().mean()), "frac_weight_above_cap": float((d.exp() > CAP).float().mean()),
                "seq_logratio_std": float(seq.std()), "n_tokens": d.numel(),
                "mean_abs_dlogp_by_prob": {k: float(d[m].abs().mean()) if m.any() else None for k, m in bins.items()},
                "tokens_by_prob": {k: int(m.sum()) for k, m in bins.items()}}

    results = {}
    for T in (1.0, 0.8):
        torch.manual_seed(SEED)
        t0 = time.time()
        comps, _, samp = rollout.generate(model, prompts, G, NEW, temperature=T, cuda_graph=False, return_logprobs=True)
        samp = [torch.tensor(s[:len(c)]) for s, c in zip(samp, comps)]
        fp32 = trainer_logp(comps)
        rows = {"trainer fp32 (T=1) vs sampler fp32": compare(fp32, samp, fp32),
                "trainer bf16 (T=1) vs sampler fp32": compare(trainer_logp(comps, bf16=True), samp, fp32)}
        if T != 1.0:
            rows[f"trainer fp32 at T={T} vs sampler fp32"] = compare(trainer_logp(comps, temperature=T), samp, fp32)
        results[f"sampled at T={T}"] = rows
        print(f"[T={T}] {len(comps)} completions, {sum(map(len, comps))} tokens, {time.time() - t0:.0f}s")
        for name, r in rows.items():
            print(f"  {name:40s} mean|dlogp| {r['mean_abs_dlogp']:.2e}  max {r['max_abs_dlogp']:.2e}  "
                  f"w>{CAP:g} {100 * r['frac_weight_above_cap']:.2f}%  seq std {r['seq_logratio_std']:.3f}  "
                  f"by prob {r['mean_abs_dlogp_by_prob']}")
    out = {"_provenance": {"checkpoint": CKPT, "prompts": f"{PROMPTS} (first {N_PROMPTS})", "samples_per_prompt": G,
                           "max_new_tokens": NEW, "seed": SEED, "device": "cpu", "tis_cap": CAP,
                           "sampler": "src.rl.rollout.generate (KV-cached, fp32 on CPU)",
                           "trainer": "src.rl.train_grpo.full_logits + fp32 log-softmax; bf16 via CPU autocast"},
           "results": results}
    json.dump(out, open(ROOT / "docs" / "mismatch_check.json", "w"), indent=1)
    print("wrote docs/mismatch_check.json")


if __name__ == "__main__":
    main()
