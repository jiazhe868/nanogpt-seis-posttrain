"""Batched, left-padded, KV-cached decoding for RL rollouts and evaluation.

All rows (prompts × samples) decode together. Prompts of different lengths are LEFT-padded
so every row's next token sits in the same column, and pad keys are masked out of attention.
RoPE makes q_m · k_n depend only on m − n, so a row whose real tokens start at column `pad`
attends exactly as if they started at 0. In fp32 the cached path matches the model's full
forward pass to ~3e-5.

Decoding cost is memory traffic, not FLOPs (a 113M model at batch 128 is far from
compute-bound), so two things matter:

  * single-token decode steps attend over the n_kv grouped heads directly — the n_head query
    heads of each group become the query "rows" of one SDPA call — instead of materialising
    the repeat_interleave'd K/V (3× the cache) at every step;
  * the KV cache is stored in bf16 under autocast — what SDPA consumes there anyway.

Sampling follows GPT._sample: temperature, then the CTRL-style repetition penalty over every
token already in the row (prompt included), then top-k / top-p. `rp_exempt_ids` removes
tokens from the penalty — e.g. digits, so a number copied from the question or computed in
<think> is not pushed away.
"""

from __future__ import annotations

import gc
import re

import math
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.sft.dataset import IM_END_ID
import src.sft.train_sft  # noqa: F401,E402  (puts the pretraining repo, nanogpt_seis, on sys.path)
from nanogpt_seis.src.model.gqa_gpt import _apply_rope  # noqa: E402

AMP_DTYPE = torch.bfloat16


def _cached_attention(attn, x, cos, sin, k_cache, v_cache, pos, mask):
    """GroupedQueryAttention.forward with a preallocated cache and a key-padding mask."""
    q = _apply_rope(attn._project(x, attn.wq.weight, attn.n_head), cos, sin)
    k = _apply_rope(attn._project(x, attn.wk.weight, attn.n_kv), cos, sin)
    v = attn._project(x, attn.wv.weight, attn.n_kv)
    B, _, T, _ = q.shape
    k_cache[:, :, pos:pos + T] = k
    v_cache[:, :, pos:pos + T] = v
    k_all, v_all = k_cache[:, :, :pos + T], v_cache[:, :, :pos + T]
    rep = attn.n_head // attn.n_kv
    if T == 1 and rep > 1:
        # (B, n_head, 1, hd) -> (B, n_kv, rep, hd): each KV group's query heads as query rows
        qg = q.reshape(B, attn.n_kv, rep, attn.hd)
        y = F.scaled_dot_product_attention(qg.to(k_all.dtype), k_all, v_all, attn_mask=mask)
        y = y.reshape(B, attn.n_head, 1, attn.hd)
    else:
        if rep > 1:
            k_all = k_all.repeat_interleave(rep, dim=1)
            v_all = v_all.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q.to(k_all.dtype), k_all, v_all, attn_mask=mask)
    return attn.wo(y.transpose(1, 2).contiguous().view(B, T, attn.n_head * attn.hd))


def _cached_step(model, ids, pos, caches, mask) -> torch.Tensor:
    """Run `ids` (B, T) at buffer positions pos..pos+T-1; return last-position logits (fp32)."""
    T = ids.shape[1]
    x = model.tok_emb(ids)
    cos = model.rope_cos[:, :, pos:pos + T]
    sin = model.rope_sin[:, :, pos:pos + T]
    for blk, (k_cache, v_cache) in zip(model.blocks, caches):
        x = x + _cached_attention(blk.attn, blk.attn_norm(x), cos, sin, k_cache, v_cache, pos, mask)
        x = x + blk.mlp(blk.mlp_norm(x))
    logits = model.lm_head(model.norm(x[:, -1:, :]))[:, -1, :]
    if model.cfg.mup:
        logits = logits * model.output_mult.to(logits.dtype)
    return logits.float()


def _sample(logits, temperature, top_k, top_p, generator, return_logp=False):
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        sorted_logits[(probs.cumsum(-1) - probs) > top_p] = -float("inf")   # mass BEFORE the token
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits.float(), dim=-1)
    tok = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    if return_logp:   # log-prob of the sampled token under the distribution it was drawn from
        return tok, torch.log(probs.gather(-1, tok[:, None]).squeeze(-1).clamp_min(1e-30))
    return tok


class GraphDecoder:
    """The single-token decode step captured as one CUDA graph.

    Decoding a 113M model is launch-bound (~700 small kernels per token across 16 layers), so
    per-step time barely depends on batch size. The graph replays all of them with one launch.
    Everything it touches lives in static buffers: the KV cache over a fixed length L, the
    key-padding mask, the input ids and the position (a tensor, so RoPE and the cache write
    index with it). Attention runs over all L slots and the mask hides those not yet written,
    so shapes never change between steps. Parameters are read in place, so optimizer updates
    are seen without re-capturing.
    """

    def __init__(self, model, B: int, L: int, cache_dtype):
        cfg = model.cfg
        dev = model.tok_emb.weight.device
        self.model, self.B, self.L = model, B, L
        self.caches = [(torch.zeros(B, cfg.n_kv_head, L, cfg.head_dim, device=dev, dtype=cache_dtype),
                        torch.zeros(B, cfg.n_kv_head, L, cfg.head_dim, device=dev, dtype=cache_dtype))
                       for _ in model.blocks]
        self.ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.mask = torch.zeros(B, 1, 1, L, dtype=torch.bool, device=dev)
        self.graph = None
        self.logits = None

    def _step(self):
        m = self.model
        x = m.tok_emb(self.ids)
        cos = m.rope_cos.index_select(2, self.pos)
        sin = m.rope_sin.index_select(2, self.pos)
        self.mask.index_fill_(3, self.pos, True)             # this token's own key becomes visible
        for blk, (k_cache, v_cache) in zip(m.blocks, self.caches):
            attn = blk.attn
            h = blk.attn_norm(x)
            q = _apply_rope(attn._project(h, attn.wq.weight, attn.n_head), cos, sin)
            k = _apply_rope(attn._project(h, attn.wk.weight, attn.n_kv), cos, sin)
            v = attn._project(h, attn.wv.weight, attn.n_kv)
            k_cache.index_copy_(2, self.pos, k.to(k_cache.dtype))
            v_cache.index_copy_(2, self.pos, v.to(v_cache.dtype))
            rep = attn.n_head // attn.n_kv
            qg = q.reshape(self.B, attn.n_kv, rep, attn.hd).to(k_cache.dtype)
            y = F.scaled_dot_product_attention(qg, k_cache, v_cache, attn_mask=self.mask)
            y = y.reshape(self.B, 1, attn.n_head * attn.hd)
            x = x + attn.wo(y)
            x = x + blk.mlp(blk.mlp_norm(x))
        logits = m.lm_head(m.norm(x))[:, -1, :]
        if m.cfg.mup:
            logits = logits * m.output_mult.to(logits.dtype)
        return logits.float()

    def capture(self, use_amp: bool):
        def run():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp, cache_enabled=False):
                return self._step()
        saved = [(k.clone(), v.clone()) for k, v in self.caches]
        mask0, pos0 = self.mask.clone(), self.pos.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):                                  # warm up kernels/allocator off the graph
                run()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = run()
        for (k, v), (k0, v0) in zip(self.caches, saved):      # undo the warm-up/capture side effects
            k.copy_(k0); v.copy_(v0)
        self.mask.copy_(mask0); self.pos.copy_(pos0)

    def step(self, ids: torch.Tensor, pos: int) -> torch.Tensor:
        self.ids.copy_(ids)
        self.pos.fill_(pos)
        self.graph.replay()
        return self.logits


_GRAPHS: dict = {}
# Each graph pins a full static KV cache: ~1.6 GB at 128 rows x 768 tokens. Six of them, plus the copy
# made during capture, filled a 24 GB A30 once the v13/v14 pools mixed many prompt lengths.
MAX_GRAPHS = 2


def _graph_decoder(model, B: int, L: int, cache_dtype, use_amp: bool) -> GraphDecoder:
    """One captured decoder per (model, batch, bucketed length); least recently used is evicted."""
    key = (id(model), B, L, cache_dtype, use_amp)
    if key in _GRAPHS:
        _GRAPHS[key] = _GRAPHS.pop(key)                       # mark as most recently used
        return _GRAPHS[key]
    while len(_GRAPHS) >= MAX_GRAPHS:
        _GRAPHS.pop(next(iter(_GRAPHS)))
        gc.collect()
        torch.cuda.empty_cache()                              # return its memory before the next capture
    dec = GraphDecoder(model, B, L, cache_dtype)
    dec.capture(use_amp)
    _GRAPHS[key] = dec
    return dec


@torch.no_grad()
def generate(
    model,
    prompts: Sequence[List[int]],
    n_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    rp_exempt_ids: Optional[torch.Tensor] = None,
    stop_ids: Sequence[int] = (IM_END_ID,),
    generator: Optional[torch.Generator] = None,
    cuda_graph: bool = True,
    bf16_logits: bool = False,
    return_logprobs: bool = False,
    calculator: bool = False,
    tokenizer: Any = None,
    return_forced: bool = False,
    rp_scope: str = "all",
    rp_window: int = 64,
    pot: bool = False,
    pot_notation: bool = False,
):
    """`n_samples` completions per prompt, rows ordered prompt-major.

    With `calculator`, a row whose text ends in `<<expr>>` gets " = value" forced into it
    (expr evaluated by src.rl.verifiers.safe_eval_number, value formatted like the drills in
    tools/numeric_drills.py), so arithmetic is done exactly instead of by the model.

    Returns (completion token ids, terminated flags), plus — with `return_logprobs` — each
    sampled token's log-probability under the sampling distribution (for importance-sampling
    correction of the sampler/trainer numerical mismatch), plus — with `return_forced` — a 0/1
    flag per token marking calculator output (environment tokens, not policy actions: RL masks
    them out of the loss). A completion includes the stop token it emitted; lockstep padding
    after it is not returned.

    With `pot` (program of thought, src.rl.pot), when a row closes its reasoning with `</think>`
    the reasoning is executed as a restricted program and "Answer: <value>" is forced after it;
    those tokens are flagged like calculator output.

    `rp_scope` decides which tokens the repetition penalty counts as already seen: "all" (prompt
    + output, the GPT._sample rule), "output" (generated tokens only, so copying a value given
    in the question is not penalised), or "window" (only the last `rp_window` generated tokens:
    still breaks local loops, but restating a result derived earlier is not penalised).
    """
    assert rp_scope in ("all", "output", "window"), rp_scope
    model.eval()
    device = model.tok_emb.weight.device
    use_amp = device.type == "cuda"
    rows = [p for p in prompts for _ in range(n_samples)]
    B, Lp = len(rows), max(len(p) for p in rows)
    L = Lp + max_new_tokens
    graph = None
    if cuda_graph and device.type == "cuda":
        L = min(model.cfg.block_size, -(-L // 128) * 128)    # bucket the length so graphs are reused
        graph = _graph_decoder(model, B, L, AMP_DTYPE if use_amp else torch.float32, use_amp)
    assert L <= model.cfg.block_size, f"prompt {Lp} + {max_new_tokens} new tokens > block"
    V = model.cfg.vocab_size
    stop = torch.tensor(list(stop_ids), device=device)

    tokens = torch.full((B, L), int(stop_ids[0]), dtype=torch.long, device=device)
    valid = torch.ones((B, L), dtype=torch.bool, device=device)
    seen = torch.zeros((B, V), dtype=torch.bool, device=device) if repetition_penalty != 1.0 else None
    win = torch.zeros((B, V), dtype=torch.int16, device=device) if seen is not None and rp_scope == "window" else None
    for i, p in enumerate(rows):
        t = torch.tensor(p, dtype=torch.long, device=device)
        tokens[i, Lp - len(p):Lp] = t
        valid[i, :Lp - len(p)] = False
        if seen is not None and rp_scope == "all":
            seen[i, t] = True
    penalised = None
    if seen is not None:
        penalised = torch.ones(V, dtype=torch.bool, device=device)
        if rp_exempt_ids is not None:
            penalised[rp_exempt_ids.to(device)] = False

    cfg = model.cfg
    cache_dtype = AMP_DTYPE if use_amp else torch.float32
    if graph is not None:
        caches = graph.caches                                 # prefill writes into the graph's cache
        graph.mask.zero_()
        graph.mask[:, 0, 0, :Lp] = valid[:, :Lp]
    else:
        caches = [(torch.empty(B, cfg.n_kv_head, L, cfg.head_dim, device=device, dtype=cache_dtype),
                   torch.empty(B, cfg.n_kv_head, L, cfg.head_dim, device=device, dtype=cache_dtype))
                  for _ in model.blocks]

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    comp_lens = torch.zeros(B, dtype=torch.long, device=device)
    rows_idx = torch.arange(B, device=device)
    samp_logp = torch.zeros((B, max_new_tokens), device=device) if return_logprobs else None
    tool = calculator or pot
    if tool:
        assert tokenizer is not None, "calculator=True / pot=True need the tokenizer"
        from src.rl.verifiers import safe_eval_number
        from src.rl.pot import run_program
        gt_mask = torch.zeros(V, dtype=torch.bool, device=device)
        gt_ids = [i for i in range(tokenizer.get_vocab_size()) if ">" in (tokenizer.id_to_token(i) or "")]
        gt_mask[torch.tensor(gt_ids, device=device)] = True
        queue: List[List[int]] = [[] for _ in range(B)]
        # After a forced value the model must not keep writing digits onto it (" = 5" + ".833"
        # was the failure): for the next sampled token, ban tokens that would extend a number.
        # A bare "." (sentence end) is allowed, after which digits stay banned for one more step.
        cont_ban = torch.zeros(V, dtype=torch.bool, device=device)
        for i in range(tokenizer.get_vocab_size()):
            s_ = tokenizer.id_to_token(i) or ""
            if s_[:1].isdigit() or (s_[:1] in ".,eE" and s_[1:2].isdigit()):
                cont_ban[i] = True
        dot_id = tokenizer.token_to_id(".")
        post = torch.zeros(B, dtype=torch.bool, device=device)
    forced = [[] for _ in range(B)] if return_forced else None
    with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=use_amp):
        # Prefill. A pad query may attend only to itself, so no softmax row is empty.
        causal = torch.ones(Lp, Lp, dtype=torch.bool, device=device).tril()
        eye = torch.eye(Lp, dtype=torch.bool, device=device)
        mask = ((causal[None] & valid[:, None, :Lp]) | eye[None])[:, None]
        logits = _cached_step(model, tokens[:, :Lp], 0, caches, mask)

        for t in range(max_new_tokens):
            if bf16_logits:
                # GPT._sample under autocast: temperature, penalty and top-k act on bf16 logits
                logits = logits.to(AMP_DTYPE)
            if tool and bool(post.any()):
                logits = logits.masked_fill(post[:, None] & cont_ban[None], float("-inf"))
            if seen is not None:
                pen = seen & penalised[None]
                lg = logits / max(temperature, 1e-6)
                lg = torch.where(pen, torch.where(lg > 0, lg / repetition_penalty, lg * repetition_penalty), lg)
                drawn = _sample(lg, 1.0, top_k, top_p, generator, return_logprobs)
            else:
                drawn = _sample(logits, temperature, top_k, top_p, generator, return_logprobs)
            nxt, lp = drawn if return_logprobs else (drawn, None)
            if lp is not None:
                samp_logp[:, t] = lp
            nxt = torch.where(finished, torch.full_like(nxt, int(stop_ids[0])), nxt)
            if tool:
                forced_now = torch.zeros(B, dtype=torch.bool, device=device)
                ended = []
                if any(queue):
                    for i in range(B):
                        if queue[i] and not bool(finished[i]):
                            nxt[i] = queue[i].pop(0)       # calculator output, not a sampled token
                            forced_now[i] = True
                            if not queue[i]:
                                ended.append(i)
                            if forced is not None:
                                forced[i].append(t)
                post = post & ~forced_now & (nxt == dot_id)
                if ended:
                    post[torch.tensor(ended, device=device)] = True
            tokens[:, Lp + t] = nxt
            comp_lens += (~finished).long()
            finished |= torch.isin(nxt, stop)
            if win is not None:
                win[rows_idx, nxt] += 1
                if t >= rp_window:
                    win[rows_idx, tokens[:, Lp + t - rp_window]] -= 1
                seen = win > 0
            elif seen is not None:
                seen[rows_idx, nxt] = True
            if tool:
                for i in (gt_mask[nxt] & ~finished).nonzero().flatten().tolist():
                    if queue[i]:
                        continue
                    tail = tokenizer.decode(tokens[i, max(Lp, Lp + t - 63):Lp + t + 1].tolist()).rstrip()
                    if pot and tail.endswith("</think>"):
                        text = tokenizer.decode(tokens[i, Lp:Lp + t + 1].tolist())
                        body = text[text.rfind("<think>") + len("<think>"):text.rfind("</think>")] if "<think>" in text else ""
                        val, _ = run_program(body, notation=pot_notation)
                        if val is not None:
                            queue[i] = tokenizer.encode(f"Answer: {val:.4g}", add_special_tokens=False).ids
                        continue
                    if calculator and tail.endswith(">>") and "<<" in tail:
                        expr = tail[tail.rfind("<<") + 2:-2]
                        # "1,000,000 / 50,000": operands copied verbatim from the question keep
                        # their thousands separators; pow(2, 3)-style argument commas do not match
                        val = safe_eval_number(re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr))
                        if val is not None:
                            queue[i] = tokenizer.encode(f" = {val:.4g}", add_special_tokens=False).ids
            if finished.all() or t == max_new_tokens - 1:
                break
            if graph is not None:
                logits = graph.step(tokens[:, Lp + t:Lp + t + 1], Lp + t)
            else:
                logits = _cached_step(model, tokens[:, Lp + t:Lp + t + 1], Lp + t, caches,
                                      valid[:, None, None, :Lp + t + 1])

    out = tokens[:, Lp:].tolist()
    lens = comp_lens.tolist()
    comps = [out[i][:lens[i]] for i in range(B)]
    ret = [comps, finished.tolist()]
    if return_logprobs:
        lp_rows = samp_logp.tolist()
        ret.append([lp_rows[i][:lens[i]] for i in range(B)])
    if return_forced:
        flags = []
        for i in range(B):
            f = [0] * lens[i]
            for t in forced[i]:
                if t < lens[i]:
                    f[t] = 1
            flags.append(f)
        ret.append(flags)
    return tuple(ret)


def generate_rollouts(model, prompts, group_size, max_new_tokens, temperature=1.0, top_p=1.0,
                      eos_token_id: int = IM_END_ID, generator=None, return_logprobs: bool = False,
                      calculator: bool = False, tokenizer: Any = None, pot: bool = False,
                      pot_notation: bool = False):
    """RL rollouts: plain sampling from the policy (no penalty, no top-k) — the REINFORCE
    gradient assumes on-policy samples. With `calculator` or `pot`, the per-token forced flags
    (tool output, masked from the loss) come last."""
    return generate(model, prompts, group_size, max_new_tokens, temperature=temperature, top_p=top_p,
                    stop_ids=(eos_token_id,), generator=generator, return_logprobs=return_logprobs,
                    calculator=calculator, tokenizer=tokenizer, pot=pot, pot_notation=pot_notation,
                    return_forced=calculator or pot)


def decode_completion(tokenizer: Any, ids: List[int], stop_ids: Sequence[int] = (IM_END_ID,)) -> str:
    for i, t in enumerate(ids):
        if t in stop_ids:
            ids = ids[:i]
            break
    return tokenizer.decode(ids)


def digit_token_ids(tokenizer: Any) -> torch.Tensor:
    """Vocabulary ids whose surface form contains a digit."""
    n = tokenizer.get_vocab_size()
    return torch.tensor([i for i in range(n) if any(ch.isdigit() for ch in (tokenizer.id_to_token(i) or ""))])
