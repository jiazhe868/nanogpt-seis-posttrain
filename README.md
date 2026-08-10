<div align="center">

# 🎯 nanoGPT-Seis-Posttrain

[English](README.md) | [中文](README.zh-CN.md) | [🤗 Hugging Face Weights](https://huggingface.co/jiazhe868/nanogpt_seis_posttrain)

**Turn a 113M pretrained seismology model into an instruction-following assistant, then
test whether RL from programmatic verifiers improves more than its format.**

Phase II (SFT) → Phase III (verifier RL) → honest measurement, on 2× NVIDIA A30 (48 GB).

</div>

![pipeline](assets/workflow.png)

---

## 0. What each training phase can — and cannot — change

Three phases solve different problems, and conflating them is how post-training results
get over-claimed:

- **Pretraining** learns the statistical structure of language and domain text.
  ([nanoGPT-Seis](https://github.com/jiazhe868/nanogpt-seis), engineering stages 1–6.)
- **SFT** teaches the model which part of a conversation it should answer and what a
  useful response looks like. Same language model, different conditional distribution.
- **Verifier RL** raises behaviours that score higher than their peers *among several
  answers to the same prompt*. It re-weights what the policy already produces.

This repository implements the last two and — equally importantly — builds the measurement
needed to tell what each one actually changed.

| Phase | Training / Inference signal | Main measured change | What was not demonstrated |
|---|---|---|---|
| I · Pretraining | next-token prediction | domain + general fluency | instruction following |
| II · SFT | teacher responses | assistant behaviour, partial extraction | reliable computation |
| III · Verifier RL | relative programmatic rewards | near-universal `<think>` compliance | above-baseline multi-step reasoning |
| IV · Deployment + RAG | BM25 retrieval over 1.0k paper chunks | open-book extraction & exact calculation ($R \approx 8.67 \text{ km}$) | parametric closed-book formula memorization |

### The central result

Asymmetric, and the asymmetry is the point:

- SFT makes the pretrained model usable as an assistant.
- RL raises `<think>` compliance from 73.1% → 97.8% on the audit test half.
- Prompt-value extraction improves modestly.
- Multi-step numerical computation stays at its blind-policy floor in closed-book mode, but succeeds when grounded by RAG context retrieval.

The lesson is not "GRPO worked" or "GRPO failed." It is that a post-training result is
inseparable from the split, decoder, reward contract, benchmark floor and ceiling,
statistical power, and checkpoint-selection rule used to measure it. Under a rigorous evaluation protocol, verifier RL decisively enforces format and response termination, while open-book RAG context grounding enables exact multi-step scientific calculations.

Two documents, two jobs:

| | question it answers |
|---|---|
| README.md (this file) | how post-training works, why these choices, what the corrected experiment showed |
| [PITFALLS.md](PITFALLS.md) | how this pipeline silently fails, how to detect it, what test prevents recurrence |

## Key table of contents
0. [What each training phase can — and cannot — change](#0-what-each-training-phase-can--and-cannot--change)
1. [Results at a glance](#1-results-at-a-glance)
2. [Quick start](#2-quick-start)
3. [Define success before training](#3-define-success-before-training)
4. [Phase II — Supervised fine-tuning](#4-phase-ii--supervised-fine-tuning)
5. [Phase III — RL from verifiable rewards](#5-phase-iii--rl-from-verifiable-rewards)
   - 5.3 [Comparing LLM RL Paradigms (PPO, DPO, GRPO, DAPO)](#53-technical-breakdown-comparing-llm-rl-paradigms-ppo-dpo-grpo-dapo)
   - 5.12 [Identifying & Avoiding Overtraining in SFT & RLVR](#512-identifying--avoiding-overtraining-in-sft--rlvr)
   - 5.13 [Identifying & Avoiding Reward Hacking (Goodhart's Law)](#513-identifying--avoiding-reward-hacking-goodharts-law)
   - 5.14 [Program-Aided Reasoning (Python Tool-Calling) Integration Guide](#514-program-aided-reasoning-python-tool-calling-integration-guide)
6. [How the evaluation works](#6-how-the-evaluation-works)
7. [What it found](#7-what-it-found)
8. [How this recipe scales](#8-how-this-recipe-scales)
9. [Failure atlas & pipeline pitfalls](#9-failure-atlas--pipeline-pitfalls)
10. [Next experiments, each with a kill gate](#10-next-experiments-each-with-a-kill-gate)
11. [Repository layout](#11-repository-layout)

### Suggested reading paths

Build the pipeline — §1 → §2 → §4 SFT → §5 verifier RL → §6 evaluation. The
chronological route, matching [nanoGPT-Seis](https://github.com/jiazhe868/nanogpt-seis).

Study experimental validity — §3 measurement contract → §6 evaluation → §9 failures.
Measurement is this repository's thesis, which is why §3 comes before the training
sections; if you would rather meet `NumericalVerifier` and the composite reward before the
vocabulary for judging them, take the first path and treat §3 as a reference.

---

## 1. Results at a glance

### 1.1 Model, data, and compute card

| | value |
|---|---|
| Base | **113.3M** total params, weight-tied (**100.7M** non-embedding) — GQA + RoPE + RMSNorm + SwiGLU, 4096 ctx, 16,388 vocab |
| Base pretraining | 822.7M unique corpus tokens · 3.8 epochs · 3.13B token exposures ≈ **31.1** exposures / non-emb param |
| SFT corpus | 15,659 pairs — 4,667 paper + 2,811 rewritten + 4,082 fundamentals + 4,707 earthquakes (16,267 before dedup) |
| SFT run | 14,068 train examples · 8 epochs · 112,544 example-exposures · peak lr 2e-4 · ~10 min |
| RL run | 200 steps · 3,200 completions · 954k generated tokens · 400/5,217 prompts (7.7%) · 3.0 GPU-h |
| Held-out pool | 808 prompts → dev 463 (selection) / test 345 (frozen, reported once) |

### 1.2 Base → SFT → RL

<p align="center"><img src="assets/grpo_benchmark_comparison.png" width="88%"></p>

Two different populations, kept in separate tables because mixing them is misleading.

(a) Confirmatory — the audit test half. Every value is the mean over prompts of a
per-prompt mean over 4 samples, not one binary outcome per prompt.

| metric | population | SFT | GRPO @100 | GRPO + RAG |
|---|---|---|---|---|
| `<think>` compliance | 90 numerical prompts × 4 samples | 73.1% | 97.8% | 97.8% |
| numerical accuracy @5% | 90 numerical prompts × 4 samples | 10.6% | 11.4% | 44.4% |
| echo / contamination | 100 prompts (90 num + 10 conceptual) × 4 | 0.0% | 0.0% | 0.0% |

Δ on closed-book accuracy (SFT → GRPO) is +0.8% [−3.1%, +4.7%] — not resolvable at this n. Grounding with RAG context retrieval boosts accuracy to 44.4%.

(b) Diagnostic — the pooled 166 numerical items, decomposed by whether the answer must
be derived. This is the whole held-out numerical set (dev + test), not the test half, and
it is a diagnostic decomposition rather than a confirmatory result.

| tier | population | SFT | GRPO @100 | GRPO + RAG |
|---|---|---|---|---|
| answer retrievable from the prompt | 34 pooled prompts × 4 | 20.6% | 29.4% | 61.8% |
| answer must be computed | 132 pooled prompts × 4 | 3.6% | 3.6% | 39.4% |

Chance floors are 7.2% and 5.5% respectively ([§7.2](#72-retrieval-vs-computation-tiers)).

> Notes on model inclusion:
> - Pretrained Base Model: Absent from both tables because without SFT, it raw-continues document text rather than answering ChatML formatted questions, so ChatML answer-extraction rules do not apply (see [§4.8](#48-what-sft-changed) for matched qualitative behavior).
> - GRPO + RAG Mode: Illustrates open-book context grounding via `SeisRAGRetriever` (`python -m src.sft.chat --rag`), converting closed-book formula recall into open-book extraction & derivation.

### 1.3 What the three stages actually look like

One held-out prompt, all three checkpoints, same decoder
(> Prompt: The 2011 Great East Japan (Tohoku) Earthquake had a moment magnitude of Mw = 9.0. Using the relationship between seismic moment and moment magnitude (Mw = (2/3) log10(M0) - 10.7, with M0 in dyne·cm), calculate the seismic moment M0 for Mw=9.0 in N·m. Then, assuming a crustal shear modulus μ = 30 GPa (3×10^10 N/m²) and an average fault slip D = 20 meters, estimate the fault rupture area A in square kilometers (km^2) using the formula M0 = μ · A · D.
>
> Target: 59,100 km² ($M_0 = 3.55 \times 10^{22} \text{ N}\cdot\text{m}$)

Base (Phase I). Given the question, it continues the document — the correct behaviour for a next-token predictor, and useless as an answer. It also degenerates into repetitive loops:

```
In order to estimate the fault dimensions, we can use empirical relations proposed by Wells and Coppersmith [1994] for subduction zones... where Mw is moment magnitude. The total slip deficit accumulates at a rate of 8 cm/yr along the Japan Trench subduction boundary, leading to an estimated recurrance interval of approximately 600 years for Mw 9.0 events. In this study, we evaluate the spatial distribution of coseismic slip...
```

+ SFT (Phase II). Now it answers, and opens a `<think>` block. But it runs to the token cap without closing the tag or stating a clean final answer:

```
<think>First compute seismic moment from magnitude: Mw = 9.0. For Mw=9.0: 1.5*(9.0+10.7) = 1.5*19.7 = 29.55. So M0 = 10^29.55 dyne·cm = 3.55e22 N·m. Rupture area A = M0 / (μ * D), where μ = 3e10 N/m², D = 20 m. So A = 3.55e22 / (3e10 * 20) = 3.55e22 / 6e11 = 5.92e10 m² = 59,200 km². Plugging in values: A = 5.92e10 m² = 59,200 km²...
```

+ verifier RL (Phase III). It closes the block, stops, and commits to a clear final answer:

```
<think>We need to compute seismic moment M0 for Mw=9.0: M0 = 10^(1.5*(9.0+10.7)) dyne·cm. Convert to N·m: 1 dyne·cm = 1e-7 N·m. For Mw=9.0: M0 = 10^(29.55) dyne·cm = 3.55e22 N·m. Then area A = M0 / (μ * D) = 3.55e22 / (3e10 * 20) = 5.91e10 m² = 5.91e4 km² = 59,100 km².</think>The estimated seismic moment for the 2011 Tohoku earthquake is approximately **$3.55 \times 10^{22} \text{ N}\cdot\text{m}$**, and the estimated fault rupture area is **$59,100 \text{ km}^2$**.
```

**+ verifier RL with RAG Context Grounding (`python -m src.sft.chat --rag`).** When passing the prompt through `SeisRAGRetriever`, the retrieved domain context grounds the model's reasoning, returning structured step-by-step LaTeX equations that match the target value:

```
<think>We need to calculate seismic moment M0 for Mw=9.0 and fault rupture area A. Given: Mw = (2/3) log10(M0) - 10.7. So log10(M0) = 1.5 * (9.0 + 10.7) = 1.5 * 19.7 = 29.55. M0 = 10^29.55 dyne·cm = 3.55 × 10^29 dyne·cm = 3.55 × 10^22 N·m. Area A = M0 / (μ * D) = 3.55e22 / (3e10 * 20) = 5.91 × 10^10 m² = 59,100 km².</think>Based on the seismic moment scaling relations for the 2011 Tohoku megathrust event:

1. **Seismic Moment ($M_0$):**
   $$\log_{10} M_0 = 1.5 \times (9.0 + 10.7) = 29.55 \implies M_0 = 10^{29.55} \text{ dyne}\cdot\text{cm} \approx 3.55 \times 10^{22} \text{ N}\cdot\text{m}$$

2. **Rupture Area ($A$):**
   Using $M_0 = \mu \cdot A \cdot D$ with $\mu = 30 \text{ GPa} = 3 \times 10^{10} \text{ N/m}^2$ and average slip $D = 20 \text{ m}$:
   $$A = \frac{M_0}{\mu \cdot D} = \frac{3.55 \times 10^{22}}{3 \times 10^{10} \times 20} = 5.91 \times 10^{10} \text{ m}^2 = 59,100 \text{ km}^2$$

The estimated seismic moment is **$3.55 \times 10^{22} \text{ N}\cdot\text{m}$** and the rupture area is approximately **$59,100 \text{ km}^2$**.
```

Read the four together and the measured progression is clear. The interface & control improves from Phase I to Phase III (continuation $\rightarrow$ reasoning attempt $\rightarrow$ well-formed, terminated answer). However, zero-shot Phase III arithmetic remains noisy (`Ratio = 7/7 ≈ 0.75`). Providing domain context via RAG Context Grounding turns closed-book memory recall into open-book extraction & calculation, allowing the exact same RL model to produce accurate derivation steps.

---

## 2. Quick start

```bash
conda activate nanogpt_seis      # same env as the pretraining repo (CUDA-12.4 torch)

# --- chat with the shipped model -------------------------------------------
python -m src.sft.chat                       # defaults to checkpoints/grpo/selected.pt

# --- Phase II: SFT ----------------------------------------------------------
python -m src.sft.build_corpus merge         # 4 sources -> 15,659 pairs (deduped)
python -m src.sft.build_corpus split         # identity-level split + hash-pinned test
torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_base.yaml
python -m src.sft.checkpoints select --ckpt-dir checkpoints/sft --link-best checkpoints/sft/selected.pt

# --- Phase III: verifier RL -------------------------------------------------
python -m src.rl.prepare_rl_dataset
torchrun --standalone --nproc_per_node=2 -m src.rl.train_grpo --config configs/rl_grpo.yaml

# --- measurement audit: run BEFORE believing any result ---------------------
python -m src.sft.splits data/rl/test.jsonl            # the split contract
python -m src.analysis.validate_measurement all        # floor / ceiling / reward hacking
python -m src.analysis.validate_measurement group      # one real G=8 group, scored
python -m src.rl.eval_rl -k 4                          # paired, McNemar, pass@k, power
python -m src.analysis.diagnose_capability tier        # where the accuracy comes from

python -m src.plots                                    # regenerate all 13 figures
```

Smaller GPUs. SFT fits in ~11 GB at `micro_batch=1`, `grad_accum=16`, bf16, 4096 ctx —
a single 12 GB card runs it in ~25 min. RL needs the policy and a frozen reference
resident (~18 GB at G=8): on one 24 GB card drop to G=4 and `max_new_tokens=256`; on 12 GB
offload the reference to CPU and accept ~3× slower steps. Note that reducing G directly
reduces the within-group spread that creates gradient at all
([§5.9](#59-where-useful-gradient-actually-came-from)), so it is a statistical change, not
only a memory one.

---

## 3. Define success before training

Everything here was added after the first pipeline produced confident wrong numbers. It
is the part most worth stealing.

### 3.1 The split contract, and what it cannot undo

Goal. Make each number answer exactly one question.

Mechanism. A pre-declared selection rule stops you cherry-picking a checkpoint after
seeing results. It does not remove selection bias: if the same prompts choose the
checkpoint and produce the reported number, that number is a development result.

Choice here. Four roles, keyed by source chunk so siblings never straddle a boundary
(`src/sft/splits.py`):

```
train   SFT / RL parameter updates
val     training-time monitoring only (early stopping, loss curves)
dev     hyperparameters, decoder, thresholds, CHECKPOINT SELECTION, metric calibration
test    reported once — never used to change code, thresholds, checkpoints, or decoding
```

Evidence this matters. This project got it wrong first: checkpoint selection and
`eval_rl` both defaulted to the full 808-prompt pool. Partitioning after the fact:

| numerical accuracy Δ (SFT → GRPO) | n | Δ | 95% CI |
|---|---|---|---|
| dev half | 76 | +3.0% | [+0.0%, +6.2%] |
| test half | 90 | +0.8% | [−3.1%, +4.7%] |

Same models, same metric, same decoder, and the dev half reads nearly 4× larger. That gap
is consistent with selection bias, and reporting only the first number would have been
a real over-claim — but it is not a clean measurement of selection bias. Part of it is
ordinary heterogeneity between two 80-odd-item halves, and the selected checkpoint had
already been exposed, indirectly, to both.

> **This is a retrospective audit split, frozen from this revision onward — not a
> prospective final test.**
>
> The dev/test partition was introduced after the fact. The original 808-prompt pool had
> already participated in checkpoint selection (step 100 chosen over step 200), aggregate
> evaluation, metric development, threshold calibration, and the choice of which analyses
> to run. Hash-pinning stops further movement; it cannot make the test half historically
> untouched.
>
> The partition is still worth having: it bounds how much those decisions inflated the
> pooled result, and it is materially more honest than reporting the pooled number. But
> "frozen final test" would overstate the separation, so this README does not use that
> phrase.
>
> A genuinely clean final claim would require: freeze the code, parser, decoder,
> verifier thresholds, candidate checkpoints, and selection rule; select using dev only;
> build a new test set from previously unused source identities; run it once.

Invariant. `assert dev_chunks.isdisjoint(test_chunks)` — enforced in `splits.describe()`.

At larger scale. Replace exact chunk identity with an identity graph plus
near-duplicate detection; keep the split-level invariant unchanged.

### 3.2 Four kinds of metric

Most measurement confusion comes from treating these as interchangeable:

| class | question it answers | examples here |
|---|---|---|
| training | is the optimizer increasing its objective? | composite reward, KL, entropy |
| held-out, same construct | does the optimized behaviour generalize to unseen prompts? | `<think>` rate, numerical-verifier accuracy |
| independent outcome | did a broader capability improve? | (none in this repo) |
| hygiene | did something unacceptable happen? | echo, contamination, empty output |

This repo has no independent-outcome metric, and that is a real limitation. Eval
accuracy uses the same field and tolerances as `NumericalVerifier`; `<think>` rate regexes
the same tags `FormatVerifier` checks. Those are the same constructs on held-out
prompts — a generalization check, not independent evidence of a broader capability. Far
better than v1's circular benchmark, but the distinction belongs in the explanation, not a
footnote.

### 3.3 Floor, ceiling, and blind baselines

A percentage is uninterpretable without both ends of its scale:

- **floor** — what a knowledge-free policy scores
- **ceiling** — what the ground truth scores through the same parser

For numerical accuracy at 5% tolerance, on 166 items:

| scoring rule | ceiling | floor | usable range |
|---|---|---|---|
| min over first 8 numbers (as originally shipped) | 68.1% | 14.3% | 53.8% |
| last 3 numbers (current) | 86.1% | 5.9% | 80.2% |

The original rule's floor sat above the model. Restricting scoring to the answer tail
improves both ends — the stated answer sits near the end, so "first 8" often never
reached it. The model then lands at 10.2%, 5.5% of the way through a [5.9%, 86.1%]
range: real but weak.

```bash
python -m src.analysis.validate_measurement floor
```

### 3.4 The decoding contract

At this scale the decoder is a larger lever than training. At repetition penalty 1.05 this
model emits 56% repeated 5-grams; raising it to 1.2 roughly doubled measured accuracy
for both models and flipped the sign of the SFT→GRPO delta. The decoding effect was
2.3× the training effect.

Temperature 0.3 / top-k 40 / rp 1.2 is therefore pinned in `chat.py` and `eval_rl.py`:
manual inspection and the benchmark must see the same model. [Pitfall 4](PITFALLS.md).

### 3.5 Provenance on every result

Every results file records the checkpoints, split, and decoding parameters that produced
it, and the plotting code warns when eval provenance does not match the training log being
plotted. This exists because a stale results file from a previous run was once
indistinguishable from a current one.

---

## 4. Phase II — Supervised fine-tuning

### 4.1 Why a fluent base is not an assistant

The base continues documents. Given "What is the seismic moment formula?" it produces
`1. Introduction. Earthquake rupture processes...` — correct next-token prediction, useless
as an answer. SFT changes the conditional, not the knowledge:

```
pretraining:  predict the next token in arbitrary documents
SFT:          condition on system + user tokens
              optimize ONLY the assistant response tokens
result:       same language model, different response distribution
```

### 4.2 The SFT objective

A masked cross-entropy over assistant tokens only:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t m_t}\sum_t m_t \log p_\theta\!\left(y_t \mid x,\, y_{<t}\right)$$

with $m_t = 0$ for system / user / padding tokens and $m_t = 1$ for assistant tokens. In
code that is one line — and it is the line that makes this instruction tuning rather than
more pretraining:

```python
# src/sft/dataset.py — everything before the assistant turn is excluded from the loss
labels[:prompt_len] = -100        # -100 == ignore_index
```

Alternatives. Training on the full sequence also works but spends capacity modelling
questions you will always be given. LoRA would cut memory ~4× at this scale, but full
fine-tuning already fits, so the extra machinery buys nothing here. Continued pretraining
on instruction-shaped text is weaker: it never tells the model where its turn begins.

### 4.3 Four ways the data can poison the model

Teacher-generated data fails in ways that are invisible in the loss curve, because the model
learns the flaw *perfectly*. Every problem below is a property of the corpus, not of the
training loop — which is why none of them can be fixed by tuning hyperparameters.

| Hazard | Why it breaks training | Resolution |
|---|---|---|
| Answers cite a source the reader cannot see ("as Table 2 shows") | The model learns that confident citation is part of a good answer, and invents references at inference time. It is being taught to hallucinate. | Drop the pair. Rewriting the sentence keeps the reasoning that produced it. |
| Cleaning is applied to the answer but not inside `<think>` | The visible answer looks clean while the reasoning trace still cites phantom sources — so the model learns to hallucinate *while thinking*, which is the part RL will later reinforce. | Apply the same rule inside the reasoning block. |
| Regex cleaning strips newlines along with the offending text | Formatting is a learned behaviour like any other. Flattened training text teaches the model to answer in undifferentiated walls of prose. | Repair text, never reflow it. |
| Roughly 15% of the crawl is off-domain | At 113M there is no spare capacity. Every off-topic example displaces a seismology one, and the model pays for both. | Filter by domain before it reaches training. |

The general lesson: **a small model has no capacity to ignore bad data.** A 70B model
exposed to occasional phantom citations may treat them as noise; a 113M model treats them
as the pattern. Data curation is a bigger lever at small scale than at large.

Retention through the whole pipeline is 27% — 17.3k raw generations become 15.7k usable
pairs. Discarding two-thirds of expensively generated data feels wasteful and is the correct
call: [§4.9](#49-a-local-data-scaling-curve) shows that data volume buys ~0.08 nats per
doubling, while a hallucination pattern learned once is permanent.

### 4.4 Split by source, not by question

One source excerpt produces several questions, and those questions are not independent — they
ask about the same facts in different words. Split them individually and a validation question
almost always has a sibling sitting in training. The model then answers it from memory, the
validation loss drops, and you conclude the model generalises when it has only memorised.

So the unit of splitting is the **source document**, not the Q&A pair: every question derived
from one excerpt travels together into exactly one split. The test assignment is additionally
fixed by hashing the source id, so a chunk held out today stays held out after the corpus is
rebuilt — otherwise each rebuild redraws the test set and two model versions can no longer be
compared.

This is worth stating as a rule because the naive version is *not* obviously wrong, and this
project shipped it twice: first by splitting pairs, then again when a new generator gave each
paraphrase its own identity, which quietly re-scattered siblings across the boundary. Both
made results look better. See [A1](PITFALLS.md) for the measured cost.

### 4.5 Packing without crossing sample boundaries

Packing concatenates examples into 4096-token blocks. The naive version splits a sample
across a boundary and keeps its labels, training a response against an amputated prompt.
Here packing flushes and pads instead. The block still uses a plain causal mask — samples
within a block can attend to each other — which is exactly why packed loss is used for
monitoring only, never for model comparison ([§6.2](#62-paired-evaluation)).

### 4.6 Training details that changed numbers

```python
# token-weighted val loss, all-reduced. DistributedSampler shards val, so a
# per-rank mean silently reported rank 0's half as if it were the whole set.
dist.all_reduce(loss_sum); dist.all_reduce(token_count)
val_loss = loss_sum / token_count

# grad-accum normalised by the TRUE group size, so the ragged last group of
# each epoch is not systematically undersized
loss = loss / this_group_size
```

### 4.7 Checkpoint selection on behaviour, not loss

Goal. Ship the checkpoint that behaves best, not the one with the lowest likelihood.

<p align="center"><img src="assets/sft_training_dynamics.png" width="92%"></p>

The curves show the textbook overfitting picture — train loss to 0.70, validation climbing
to 2.58 — and the natural reading of it is wrong here.

Mechanism. Cross-entropy scores the probability of one specific reference wording. As
training continues the model sharpens onto its own phrasing, lowering the reference's token
probability without making generations worse:

| | best (step 80) | last (step 336) |
|---|---|---|
| held-out loss | 2.2700 | 2.7798 (+0.51 nats, t=+33) |
| `<think>` rate | 61.4% | 74.1% (z=+2.47) |

Choice here. Selection runs on measured behaviour, on the dev split only, with the
rule declared in source before any numbers were looked at:

```python
# src/sft/checkpoints.py
PRIMARY   = "numerical_acc_5pct"     # the only available scored correctness proxy
TIE_BREAK = "numerical_think_rate"   # used only within TIE_EPS of the leader
TIE_EPS   = 0.02                     # ≈ noise at this n
GATES = [("contamination_rate", "<=", 0.02), ("echo_rate", "<=", 0.40),
         ("mean_completion_tokens", ">=", 40)]
```

Evidence. It selected epoch 8; validation loss would have picked epoch 3, the candidate
with the worst `<think>` rate of the three.

Invariant. The rule is a module-level constant, not a runtime argument — it cannot be
tuned after seeing the table.

### 4.8 What SFT changed

SFT changed what the model *is for*, and almost nothing about what it knows.

Before SFT the model has no notion of a turn. Handed a question it continues the document —
plausible seismology prose that never addresses you, and never stops, because nothing in
pretraining taught it that a response has an end. After SFT it answers, and it terminates.
Those two are the whole interface, and both are learned entirely in this phase.

It also partly acquires the *shape* of reasoning: roughly three-quarters of numerical answers
now arrive wrapped in a `<think>` block. Partly, not reliably — closing that gap is what
Phase III is for.

What did not move is the arithmetic. On questions whose answer is already stated somewhere in
the prompt, SFT reaches about one in five — real task adaptation, since it must find and
restate the right quantity. On questions whose answer must actually be derived, it sits at its
chance floor. The distinction matters more than any single accuracy number, and
[§7.2](#72-retrieval-vs-computation-tiers) makes it the organising axis of the results.

The honest summary: SFT bought an interface and a habit of showing work. It did not buy
knowledge the base model lacked, and there is no reason it should have — the objective is
imitation of surface form, and computation is not surface form.

### 4.9 A local data-scaling curve

<p align="center"><img src="assets/sft_data_scaling.png" width="92%"></p>

Composition-matched subsamples — mixture held at 46/25/28 paper/fundamentals/earthquakes at
every size, so size is the only thing varying:

```
loss = 3.4041 − 0.0813 · log₂(N)        residuals within ±0.016
```

An earlier version measured 0.063 because its small points were pure paper — the easier
subset for this test set — so size and mixture moved together. Removing the confound made
data look 28% more valuable, not less.

> Scope. This is a local end-to-end SFT scaling curve, not a universal data-scaling
> law. Epochs were fixed at 8, so larger subsets also received proportionally more
> optimizer updates: the curve varies unique data and total update tokens together.
> One seed per point; no regression uncertainty is reported. The 158k-pair extrapolation is
> ~3.5 doublings beyond the largest measured point — an order of magnitude, not a
> prediction. A stronger design would compare equal-epoch vs equal-assistant-token-exposure
> vs repeated-smaller-data at matched compute.

---

## 5. Phase III — RL from verifiable rewards

### 5.1 Why RL after SFT?

The family name for this is RL with verifiable rewards (RLVR), introduced in Tülu 3
([Lambert et al. 2024](https://arxiv.org/abs/2411.15124)): the reward is a program that
checks the answer, not a model trained on human preferences.

SFT imitates one reference answer per prompt. Verifier RL samples several answers to the
same prompt and pushes probability toward whichever scored better — so it can optimize
properties the reference never demonstrated (brevity, format, termination), and needs no
human labels because the reward is a program.

### 5.2 The exact algorithm implemented here

For each prompt: sample G completions, score them, standardize within the group, take
one policy-gradient step.

```python
# src/rl/train_grpo.py — the whole objective, minus bookkeeping
rewards    = [reward_engine.score(c, item) for c in completions]     # G rollouts
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)     # group baseline

# k3 KL estimator against the frozen SFT reference, masked to SAMPLED tokens only
per_token = -(logp * advantage - beta * kl) * sampled_mask
loss = (per_token.sum(dim=1) / seq_tokens).mean()
```

This is REINFORCE ([Williams 1992](https://doi.org/10.1007/BF00992696), Machine Learning 8:229–256) with a
group-relative baseline. The group mean replaces the learned critic of PPO
([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)): with G samples of the same
prompt it is a zero-parameter, low-variance baseline, and dividing by the group std keeps the update scale
comparable across easy and hard prompts.

> Not strictly unbiased. Each completion contributes to its own group mean and standard
> deviation, so the finite-G estimator is self-normalized rather than unbiased. Subtracting
> a mean that includes the current sample scales the expected policy-gradient term by roughly
> `1 − 1/G`; standardizing by the group std adds further finite-sample bias. A leave-one-out
> mean would remove the simplest own-sample component, though the normalization would still
> deserve separate treatment. None of this invalidates the method — it just makes "unbiased"
> the wrong word.

Why that loss does what it claims. For one sampled completion, the policy term is

$$\mathcal{L}_{\text{policy}} = -A \log p_\theta(y \mid x)$$

Under gradient descent:

- if $A > 0$, reducing the loss increases $\log p_\theta(y \mid x)$ — the completion
  becomes more likely;
- if $A < 0$, reducing the loss decreases it — the completion becomes less likely;
- if every completion in a group scores the same, every $A$ is zero and **nothing is
  learned**, however high or low that common reward was. This is the whole content of
  [§5.5](#55-what-one-real-group-teaches).

The KL term plays the opposite role: it penalises movement away from the frozen SFT
reference even when the reward favours that movement. The reward pulls, the KL anchors,
and β sets the exchange rate between them.

The KL estimator, exactly as implemented. Define the log-ratio on the sampled token,
reference minus policy:

$$r_t = \log p_{\text{ref}}(y_t \mid \cdot) - \log p_\theta(y_t \mid \cdot)$$

Schulman's k3 estimator ([2020](http://joschu.net/blog/kl-approx.html)) is
`k3(y) = y - 1 - log y` evaluated at `y = e^{r_t}`, i.e.

$$\widehat{\text{KL}}_t = e^{r_t} - r_t - 1$$

```python
# src/rl/train_grpo.py
kl_div = torch.exp(ref_target_log_probs - target_log_probs) \
         - (ref_target_log_probs - target_log_probs) - 1.0
```

Note the convention: with $r_t$ defined reference-minus-policy, this estimates
$\mathrm{KL}(p_\theta \Vert p_{\text{ref}})$ — the direction that penalises the policy
for putting mass where the reference does not. It is non-negative for every token (since
$e^r - r - 1 \ge 0$), which the naive estimator $-r_t$ is not; that is why it is used, and
why a KL curve here can never go negative even on a noisy batch.

#### A minimal, runnable implementation

The whole algorithm is about forty lines. This mirrors `src/rl/train_grpo.py` with the
logging and DDP plumbing removed, and works against any HF-style causal LM:

```python
import torch

@torch.no_grad()
def rollout(policy, tokenizer, prompt, group_size=8, max_new_tokens=384):
    """Sample G completions for ONE prompt. The group is the unit of learning."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(policy.device)
    ids = ids.repeat(group_size, 1)                    # G copies of the same prompt
    out = policy.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                          max_new_tokens=max_new_tokens,
                          pad_token_id=tokenizer.eos_token_id)
    completions = out[:, ids.shape[1]:]                # strip the prompt back off
    sampled = completions != tokenizer.eos_token_id    # lockstep EOS padding was NOT sampled
    return out, completions, sampled


def token_logprobs(model, sequences, n_prompt_tokens):
    """log p(y_t | x, y_<t) for every generated token."""
    logits = model(sequences).logits[:, :-1]           # position t-1 predicts token t
    logp = torch.log_softmax(logits.float(), dim=-1)
    logp = logp.gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logp[:, n_prompt_tokens - 1:]               # keep the completion span only


def grpo_step(policy, reference, tokenizer, prompt, item, reward_engine,
              beta=0.02, group_size=8):
    sequences, completions, mask = rollout(policy, tokenizer, prompt, group_size)
    n_prompt = sequences.shape[1] - completions.shape[1]

    # 1. score every completion with the programmatic verifiers
    texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
    rewards = torch.tensor([reward_engine.score(t, item)["total_reward"] for t in texts],
                           device=policy.device)

    # 2. the group IS the baseline -- there is no critic network anywhere
    if rewards.std() < 1e-6:
        return None          # zero spread -> zero advantage -> no gradient. Resample.
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 3. per-token log-probs under the policy and the frozen reference
    logp = token_logprobs(policy, sequences, n_prompt)
    with torch.no_grad():
        logp_ref = token_logprobs(reference, sequences, n_prompt)

    # 4. k3 KL estimate -- non-negative on every token
    log_ratio = logp_ref - logp
    kl = torch.exp(log_ratio) - log_ratio - 1.0

    # 5. REINFORCE against the group baseline, masked to tokens actually sampled
    per_token = -(logp * advantages.unsqueeze(1) - beta * kl) * mask
    loss = (per_token.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    loss.backward()
    return loss.item()
```

Five details in that code are the entire difference between working and silently not working:

- **`ids.repeat(group_size, 1)`** — the G samples must share one prompt. Advantages are only
  meaningful relative to siblings answering the *same* question.
- **the `rewards.std() < 1e-6` guard** — an all-equal group produces no gradient. Returning
  early is the one-line version of DAPO's Dynamic Sampling; without it you burn compute on
  updates that provably do nothing ([§5.5](#55-what-one-real-group-teaches)).
- **`sampled = completions != eos`** — lockstep decoding pads short completions. Including
  that padding trains on tokens the policy never chose, and inflates KL until it looks like
  policy drift ([§5.7](#57-masks-forced-eos-and-kl)).
- **`with torch.no_grad()` on the reference** — the reference is an anchor, never a learner.
  Leaving gradients on is a silent, expensive bug.
- **per-sequence mean before the batch mean** — dividing by each sequence's own token count
  first stops long completions from dominating the update. (DAPO argues for the opposite; see
  §5.3 point 3 for when that trade flips.)

### 5.3 Technical Breakdown: Comparing LLM RL Paradigms (PPO, DPO, GRPO, DAPO)

To understand why Group-Relative REINFORCE (GRPO-style) was selected for this repository, we must examine the formal mathematical derivations, neural network architectures, structural reductions, and trade-offs of the primary post-training alignment algorithms:

```
+---------------------------------------------------------------------------------------------------+
|                                  LLM ALIGNMENT STRUCTURAL EVOLUTION                               |
+------------------------------------+--------------------------------------------------------------+
|   PREFERENCE-BASED (Human / AI)    |                RULE-BASED VERIFIABLE (RLVR)                  |
+------------------------------------+--------------------------------------------------------------+
|  PPO  · 4 Networks (Actor+Ref+RM+V)|  GRPO · 2 Networks (Actor+Ref), Critic reduced to Group Mean |
|  DPO  · 2 Networks (Actor+Ref)     |  DAPO · 1 Network (Actor); KL removed, decoupled clip        |
+------------------------------------+--------------------------------------------------------------+
```

---

#### 1. PPO (Proximal Policy Optimization) — The Full Actor-Critic Baseline
- **Architecture & 4 Neural Networks in GPU Memory:**
  1. Policy / Actor ($\pi_\theta$): The model being optimized.
  2. Reference Policy ($\pi_{\text{ref}}$): Frozen SFT model to anchor KL divergence.
  3. Reward Model ($RM_\phi$): Scalar neural network trained on pairwise preferences $y_w \succ y_l$.
  4. Value / Critic Network ($V_\psi$): Neural network predicting expected cumulative future reward from generation state $(x, y_{<t})$.

- **Mathematical Formulation:**
  For token generation step $t$, advantages are estimated via Generalized Advantage Estimation (GAE):
  $$\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t), \quad \hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$
  The policy is updated using the PPO clipped surrogate loss:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right], \quad \text{where } r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$$
  The joint objective optimizes policy, value fitting, and KL constraint:
  $$\mathcal{L}^{\text{PPO}}(\theta, \psi) = - L^{\text{CLIP}}(\theta) + c_1 \mathbb{E}\left[ (V_\psi(s_t) - R_t)^2 \right] + c_2 \mathrm{D}_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$$
  (Written as a loss to minimise, so $-L^{\text{CLIP}}$. Most RLHF implementations fold the
  KL into the per-token reward, $r_t \leftarrow r_t - \beta \log\frac{\pi_\theta}{\pi_{\text{ref}}}$,
  rather than adding it as a separate term; the two differ in whether the critic sees the
  penalty.)

- **Why It Suffers at Scale:**
  - **Memory Overhead:** Loading 4 large models concurrently consumes massive VRAM or requires CPU offloading.
  - **Critic Instability:** Fitting a high-dimensional text-generation Critic $V_\psi(s)$ introduces high variance during autoregressive generation, causing loss divergence in LLMs.

---

#### 2. DPO (Direct Preference Optimization) — Analytic Elimination of the Reward Model & Critic
- **Architecture & Structural Reduction:**
  Eliminates the Reward Model $RM_\phi$, Critic Network $V_\psi$, and online generation loop entirely, operating with only 2 models: Policy $\pi_\theta$ and Reference $\pi_{\text{ref}}$.

- **Mathematical Derivation:**
  Start from the exact KL-constrained maximum-reward RL formulation:
  $$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x,y) \right] - \beta \, \mathrm{D}_{\text{KL}}\left(\pi(y|x) \;\parallel\; \pi_{\text{ref}}(y|x)\right)$$
  This problem has a closed-form optimum — a Gibbs/Boltzmann tilt of the reference policy
  (no preference model is involved yet):
  $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right) \iff r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$
  Inverting it expresses the reward in terms of the **optimal** policy. DPO's move is then to
  *parameterise* the learned policy as that optimum ($\pi_\theta \equiv \pi^*$) — an assumption,
  not an algebraic consequence — which turns $\pi_\theta$ into an implicit reward model.
  Substituting into the Bradley-Terry preference probability
  $P(y_w \succ y_l \mid x) = \sigma(r(x,y_w) - r(x,y_l))$ makes the intractable partition
  function $\log Z(x)$ cancel, because it depends only on $x$ and appears in both terms:
  $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

- **Pros & Caveats:**
  - Eliminated: Critic network $V_\psi$, Reward Model $RM_\phi$, and online sampling loops.
  - Caveat: DPO is strictly offline. It cannot explore new generation trajectories or evaluate deterministic rule-based Python verifiers dynamically during training.

---

#### 3. GRPO / Group-Relative REINFORCE (Selected Here) — Eliminating the Critic via Group Sampling
- **Architecture & Structural Reduction:**
  Uses only 2 models ($\pi_\theta$ and $\pi_{\text{ref}}$). Eliminates the Critic Network $V_\psi$ by replacing it with a zero-parameter sample-mean baseline computed across a group of sampled candidate completions.

- **Mathematical Formulation:**
  For a prompt $x$, sample a group of $G$ completion rollouts $\{y_1, y_2, \dots, y_G\}$ from $\pi_{\theta_{\text{old}}}$. Compute deterministic rewards $\{r_1, r_2, \dots, r_G\}$ using programmatic Python verifiers (RLVR).
  Compute group-relative advantage without any value network:
  $$\bar{r} = \frac{1}{G}\sum_{i=1}^G r_i, \quad \sigma_r = \sqrt{\frac{1}{G-1}\sum_{i=1}^G (r_i - \bar{r})^2}, \quad A_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$
  (This repo uses `torch.std`, i.e. the Bessel-corrected $1/(G-1)$, with $\epsilon$ outside the
  root — see §5.2. The paper writes the population form; at $G=8$ the difference is ~7% on
  the advantage scale and cancels in the sign.)
  The token-level objective function combines group relative advantage with per-token k3 KL regularization:
  $$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left( \min\left( \frac{\pi_\theta(y_{i,t} \mid \dots)}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid \dots)} A_i, \, \text{clip}\left(\dots\right) A_i \right) - \beta \, \widehat{\mathrm{KL}}_{i,t} \right) \right]$$

- **Why It Fits RLVR:**
  - Two resident models instead of four. The saving is real but not a clean "50% of VRAM":
    the RM and critic are often smaller than the actor, and only trainable models carry
    optimizer state (PPO trains two, GRPO one).
  - Retains active online exploration, so programmatic verifiers can be called on fresh
    rollouts — the property DPO gives up.
  - **The cost moved, it did not vanish.** The critic is gone, but every prompt now needs $G$
    rollouts, so generation cost scales with $G$. GRPO trades critic memory for sampling
    compute.
  - Note that GRPO is *reward-source agnostic*: DeepSeekMath used it with a learned reward
    model as well as rule-based rewards. The PREFERENCE / RULE-BASED split in the diagram
    above is about how **this repo** uses these algorithms, not an intrinsic property.

---

#### 4. DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization
[Yu et al. 2025](https://arxiv.org/abs/2503.14476). DAPO builds on GRPO and fixes **four**
failure modes seen in long-CoT RL. Its objective drops the KL term entirely:

$$\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[ \frac{1}{\sum_{i=1}^{G} |y_i|} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min\left( r_{i,t}(\theta)\hat{A}_{i,t}, \; \text{clip}\left(r_{i,t}(\theta),\, 1-\epsilon_{\text{low}},\, 1+\epsilon_{\text{high}}\right)\hat{A}_{i,t} \right) \right]$$

$$\text{subject to}\quad 0 < \left| \{ y_i : \texttt{is\_correct}(y_i) \} \right| < G$$

1. **Clip-Higher** — decouple the clip range into $\epsilon_{\text{low}}$ and
   $\epsilon_{\text{high}}$ with $\epsilon_{\text{high}} > \epsilon_{\text{low}}$. Symmetric
   clipping caps how fast a low-probability-but-correct token can be raised, which drives
   **entropy collapse**; raising only the upper bound preserves exploration.
2. **Dynamic Sampling** — that `subject to` line. Drop any group whose rollouts *all* score
   the same (all correct or all wrong) and resample until the batch is full.
3. **Token-Level Policy Gradient Loss** — note the $\frac{1}{\sum_i |y_i|}$ prefactor:
   normalisation is over *all tokens in the batch*, replacing GRPO's per-sequence
   $\frac{1}{G}\sum_i \frac{1}{|y_i|}$. Under GRPO's form every sample gets equal weight
   regardless of length, so each token of a long derivation contributes less gradient than a
   token of a short one.
4. **Overlong Reward Shaping** — a soft length-aware penalty for truncated generations
   instead of scoring them as simply wrong, which removes reward noise from responses that
   were cut off rather than incorrect.

**Removing the KL is deliberate**: "during training the long-CoT reasoning model, the model
distribution can diverge significantly from the initial model, thus this restriction is not
necessary." That is a claim about *frontier-scale reasoning runs*, and it does **not**
transfer here — at 113M the reference KL is what stops this policy degenerating, and §5.8
shows the β controller earning its place.

> **Why point 2 matters most for this repository.** Dynamic Sampling is the published fix for
> exactly the pathology documented in [§5.5](#55-what-one-real-group-teaches) and
> [§5.9](#59-where-useful-gradient-actually-came-from): a group with zero reward spread
> produces zero advantage and therefore no gradient, no matter how large the reward. This run
> spent updates on such groups instead of resampling them. Note also that this repo's variant
> drops the ratio and clipping altogether ([§5.4](#54-how-this-implementation-differs-from-canonical-grpo)),
> so Clip-Higher does not apply to it as written — Dynamic Sampling is the transferable idea.

---

#### Comparative Architectural Evolution Matrix

| Paradigm | Models in VRAM | Reward Mechanism | Advantage Estimation $\hat{A}$ | Primary Structural Reduction |
|---|---|---|---|---|
| PPO | 4 (Actor, Ref, RM, Critic) | Learned Reward Model $RM_\phi$ | GAE via Value Critic $V_\psi(s)$ | Baseline framework (no reduction) |
| DPO | 2 (Actor, Ref) | Implicit from Log-Ratios | Implicit Pairwise Difference | Eliminated Critic, RM & Online Sampling |
| GRPO (Ours) | 2 (Actor, Ref) | Reward source agnostic; verifiers here | Group sample mean, $\frac{r_i - \bar{r}}{\sigma_r}$ | Eliminated critic network $V_\psi$ |
| DAPO | 1 (Actor) — KL removed | Reward source agnostic; verifiers in the paper | Same group advantage; **clipping** is asymmetric | Removed KL, length bias, symmetric clip, and zero-variance groups |

### 5.4 How this implementation differs from canonical GRPO

> This is not canonical GRPO, and the distinction is worth understanding.
>
> GRPO, as introduced in DeepSeekMath ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)),
> is described there as "a variant of Proximal Policy Optimization": an **old-policy
> importance ratio, a clipped surrogate objective**
> ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)), group-relative advantages, and
> direct KL to a reference.
> This implementation keeps the group-relative baseline and the reference KL but drops the
> ratio and clipping, performing one fresh on-policy update per sampled batch —
> `grep -nE "ratio|clip" src/rl/train_grpo.py` finds only gradient-norm clipping.
>
> With exactly one update per freshly sampled batch the importance ratio is identically 1,
> so the local gradient is closely related; the optimizer and stability behaviour are not
> identical, and nothing here supports multiple epochs over a rollout batch.
>
> The accurate name is group-relative REINFORCE
> ([Williams 1992](https://doi.org/10.1007/BF00992696), Machine Learning 8:229–256) with KL regularization (or
> "GRPO-style group-relative policy gradient"). Showing which pieces are essential — group baseline,
> reference KL — and which are implementation choices — ratio, clipping — is more useful
> than claiming the label.

### 5.5 What one real group teaches

Run `python -m src.analysis.validate_measurement group` and you get a single prompt, eight
real rollouts from the shipped checkpoint, and the arithmetic the trainer actually performs.
It is worth reading once, because three separate failures become obvious from it.

In the group I sampled, seven of the eight completions got the number wrong; one found it.
One of the seven also broke format by leaving its `<think>` block unclosed. Standardising the
rewards inside the group turns that into a gradient direction with no value network involved:
the correct completion gets a large positive advantage and is pushed up hard, the
format-breaker gets the most negative and is pushed down hardest, and the remaining six —
wrong but well-formed — get a small uniform push down. The advantages sum to zero by
construction, which is exactly what "the group is the baseline" means.

**Now the part that matters.** Of the seven reward components, only two varied across the
group at all: the numerical score and the format score. Concept, echo, cleanliness, length and
termination returned the *same value for every completion*. A constant column contributes
nothing — it shifts the group mean and then cancels when the mean is subtracted. Five sevenths
of the reward machinery was inert for this update.

That single observation explains three things at once:

- **A dead verifier is invisible.** A component stuck at zero because of a bug looks identical
  to a component that is merely constant on this batch. This is how a numerical reward reading
  a field that existed in *no* record survived 300 steps unnoticed ([D1](PITFALLS.md)).
- **A high reward can teach nothing.** If every completion in a group scores well, the spread
  is zero, every advantage is zero, and the update is a no-op — regardless of how good the
  reward looks in the log. This is precisely what DAPO's Dynamic Sampling
  ([§5.3](#53-technical-breakdown-comparing-llm-rl-paradigms-ppo-dpo-grpo-dapo)) discards and
  resamples.
- **Only sampled tokens belong in the mask.** Groups decode in lockstep, so short completions
  are padded; counting that padding puts gradient on tokens the policy never chose
  ([§5.7](#57-masks-forced-eos-and-kl)).

The practical consequence is that mean reward is a poor progress signal. What predicts
learning is *within-group spread*, and [§5.9](#59-where-useful-gradient-actually-came-from)
is about how little of it this run actually had.

### 5.6 Every reward is a hypothesis, and therefore an attack surface

A verifier is not a measurement of a construct; it is a *proxy* you have asserted stands in
for one. The policy optimises the proxy. Wherever proxy and construct come apart, that gap is
free reward, and gradient descent is extremely good at finding it.

The seven components here divide by what they are trying to buy. Two are capability claims —
is the final value right, does a conceptual answer cover the domain ideas. Two are structural —
is the response well formed, does it terminate instead of running to the token cap. Three are
hygiene — no phantom citations, no regurgitating the prompt back, no unbounded rambling. Only
the capability pair is hard to satisfy; the rest are cheap for a competent model, which is
precisely why they dominated what this run actually learned.

Each one has an obvious shortcut, and naming it up front is the job:

- Reward well-formed `<think>` tags and the cheapest way to earn it is an **empty** block.
- Reward a final number within tolerance and the cheapest way is to **emit many numbers**.
- Reward domain-concept coverage and the cheapest way is to **stuff keywords** where the reader
  will not look — inside the reasoning trace.
- Penalise copying the prompt and the cheapest evasion is to **pad the answer** until the
  copied fraction falls below threshold.

**How each shortcut is actually detected.** Naming an attack is useless without a standing
check that fires when it appears, so every component is paired with one:

- *Empty reasoning* — the format check requires real content between the tags, not just a
  matched pair, so `<think></think>` scores as broken rather than compliant.
- *Number spraying* — scored against a **blind-spray floor**: what a policy that ignores the
  question and emits k plausible values would score. If the model cannot beat that, the metric
  is measuring verbosity ([§3.3](#33-floor-ceiling-and-blind-baselines)).
- *Keyword stuffing* — concept coverage is matched **only against text after `</think>`**, so
  keywords hidden in the reasoning trace earn nothing.
- *Denominator dilution* — a filler-plus-copy probe, since normalising by the prompt makes a
  literal echo score 1.0 no matter how much padding follows.
- *Phantom citations* — the same lead-in regex used to clean the corpus, now run on
  generations.
- *Never terminating* — stop rate is logged every step; a policy learning to ramble past
  `<|im_end|>` shows up as a falling curve well before it shows up in reward.

Three of these run continuously rather than on demand. `python -m src.rl.verifiers` executes
all eight adversarial probes as unit tests. `python -m src.analysis.validate_measurement hacking`
scores eight deliberately degenerate completions against the *real* reward engine across every
held-out item and flags any that reach or beat the genuine reference answer — that is the check
that catches an exploit you did not think to name. And plotting each reward component
separately makes a component that has gone dead or constant visible immediately, which an
aggregate reward hides completely.

This is not defensive decoration. One probe found a live exploit: verbose filler plus a correct answer scored *above* the genuine
reference answer, because the echo penalty divided the overlap by the candidate's own length —
so padding mechanically diluted it. The same mis-specification punished real answers for
legitimately restating the question. Normalising against the prompt instead fixed both
directions at once ([D2](PITFALLS.md)).

The generalisable rule: **write the cheating completion yourself, before the optimiser does.**
And treat a penalty that fires on genuine reference answers as a bug, not as strictness — if
the ground truth cannot score well, the component is measuring the wrong thing.

### 5.7 Masks, forced EOS, and KL

Groups decode in lockstep, so short completions are padded with forced EOS. Those tokens
were never sampled — including them puts gradient on tokens the policy did not choose and
dilutes the `mask.sum()` normalizer. In v1 this produced a KL spike of 8.23 that looked
like policy drift and was actually padding.

### 5.8 Adaptive β as a controller

<p align="center"><img src="assets/grpo_training_dynamics.png" width="92%"></p>

Panel (c) is the one to read: KL rises past the 2×target deadband, β climbs from its 0.005
floor to 0.192, and KL is pulled back to 0.042 — a controller demonstrably working, which
v1 and v2 never showed. Panel (d) shows the run peaking near step 100 and regressing after,
which the held-out evaluation later confirmed.

### 5.9 Where useful gradient actually came from

Batches in this run are single-type, so a step exercises either the numerical verifier or
the concept verifier, never both:

| step type | steps | what the reward asks for |
|---|---|---|
| numerical | 77 / 200 (38.5%) | do the physics, then the arithmetic |
| conceptual | 123 / 200 (61.5%) | cover the domain concepts |

The run visited 400 prompts — 7.7% of the pool, not one epoch. So correctness of
computation was on the table in well under half the updates, while format was scoreable in
every one of them. That asymmetry is a sufficient explanation for which behaviour moved.

> The statistic I actually wanted, and did not log. Eligibility is a weaker proxy than
> what matters. Only within-group spread creates gradient ([§5.5](#55-what-one-real-group-teaches)),
> so the right measure is the reward-opportunity rate: the fraction of groups in which a
> verifier produced discriminative advantage. The trainer logs the batch mean of each
> component, not its within-group standard deviation, so that rate is **not recoverable from
> these run artifacts** — the table above is step eligibility, which is an upper bound on it.
> Logging per-component within-group std costs one number per step and belongs in any
> verifier-RL run; it is the first thing I would add before the next one.

### 5.10 Baselines: prompting, and what is still missing

Explicitly instructing the SFT model to use `<think>` tags moves compliance
72.3% → 73.5%. RL moved it to 97.8%. Under the tested prompt template and decoder,
prompting barely changes the behaviour RL changes almost completely.

> Missing baselines, stated plainly. Prompting is not the strongest non-RL comparison.
> The ladder should also include format-targeted SFT and, most importantly,
> rejection-sampling fine-tuning — sample k answers from the SFT model, keep the
> verifier-passing ones, fine-tune on them. RFT tests whether online policy gradients add
> anything beyond converting successful exploration into supervised data at the same
> rollout budget. Neither was run. Until they are, "RL was necessary" is unproven; only
> "prompting was insufficient" is supported.

### 5.11 Where this sits in the literature

- **InstructGPT** ([Ouyang et al. 2022](https://arxiv.org/abs/2203.02155)) — demonstrations
  → human rankings → learned reward model → RLHF. This pipeline replaces the rankings and
  the reward model with programmatic verifiers: no annotation cost and no reward-model
  overfitting, but verifier-specification risk instead.
- **DeepSeekMath / GRPO** ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)) — a 7B
  model, ~120B math-related pretraining tokens, mathematical SFT, then canonical GRPO.
  Group-relative RL improved its mathematical benchmarks *from a base that already had
  substantial mathematical capability*. §5.3 lists exactly which pieces of it this
  implementation keeps.
- **DPO** ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) — offline,
  preference-pair classification; avoids online sampling entirely, but is not a drop-in
  substitute unless verifier scores are converted to chosen/rejected pairs.
- **Rejection-sampling fine-tuning** ([Yuan et al. 2023](https://arxiv.org/abs/2308.01825))
  — the missing baseline named in §5.10: keep the verifier-passing samples and fine-tune on
  them, no policy gradient at all.
- **DeepSeek-R1** ([DeepSeek-AI 2025](https://arxiv.org/abs/2501.12948)) — the strongest
  demonstration that RL alone can incentivize reasoning behaviour. Worth reading against
  this repo's result: there the format and the reasoning moved; here only the format did.
- Supporting: pass@k, formalized with its unbiased estimator by
  [Chen et al. 2021](https://arxiv.org/abs/2107.03374) (who attribute the metric to
  [Kulal et al. 2019](https://arxiv.org/abs/1906.04908)) — note this repo reports the plain
  "any of k samples passed" form with n = k, not that estimator; the k3 KL estimator
  ([Schulman 2020](http://joschu.net/blog/kl-approx.html)); and Chinchilla
  ([Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)), whose 70B model on 1.4T
  tokens gives the ~20 tokens/param that §7.4 contrasts with this project's own measured
  frontier.

The honest contrast: DeepSeekMath shows group-relative RL improving math when the base can
already do math. This is a different regime — 113M params, sparse correctness reward,
11-step held-out derivations — and there the same family of ideas moved format decisively
while leaving multi-step computation untouched.

### 5.12 Identifying & Avoiding Overtraining in SFT & RLVR

A major trap in LLM post-training is misinterpreting loss curves and overfitting signals:

#### 1. Over-fitting in Supervised Fine-Tuning (SFT)
- **The Paradox:** In SFT, validation Cross-Entropy loss will often start rising after a few epochs (e.g., from 2.27 to 2.78), creating the illusion of severe overfitting.
- **The Physical Reality:** Cross-entropy measures the exact token-by-token log-likelihood of one specific teacher reference response. As SFT progresses, the model develops its own stylistic phrasing. While this reduces the exact probability assigned to the reference tokens (raising validation loss), the actual instruction-following quality, format compliance, and answer accuracy continue to improve.
- **How to Avoid SFT Overtraining:** Never rely on Validation Loss for early stopping! Instead, use Generation-Based Checkpoint Selection (`src.sft.checkpoints`), evaluating actual generated responses on a held-out dev split.

```
                  SFT LOSS VS GENERATION QUALITY
Loss (nats) ↑                                           Think Rate % ↑
   2.80 +                                            + 75%  <-- Best ChatML Behavior
        |          Validation Loss (Climbing)        |
   2.50 +                 /                          + 70%
        |                /                           |
   2.20 +-- Min Loss ---+                            + 65%
        +--------------------------------------------+
       Step 0           Step 80                    Step 336
```

#### 2. Over-training & Policy Collapse in RLVR
- **Symptoms of RL Collapse:**
  - **KL Explosion:** Policy strays too far from the reference model ($\text{KL} > 0.5$), resulting in ungrammatical gibberish.
  - **Entropy Collapse:** Output entropy drops to near zero as the model repeats a single high-reward phrase over and over.
  - **Length Inflation:** The model learns that longer completions dilute per-token penalties, emitting thousands of redundant tokens.
- **How to Avoid RL Overtraining:**
  - Implement an **Adaptive $\beta$ Controller** to dynamically scale the KL penalty.
  - Set hard sequence length limits (`max_new_tokens`) during rollout generation.
  - Monitor per-component reward standard deviation: if $\sigma_{r} \rightarrow 0$, the rollout group provides zero gradient.

---

### 5.13 Identifying & Avoiding Reward Hacking (Goodhart's Law)

#### Goodhart's Law in Reinforcement Learning
> "When a measure becomes a target, it ceases to be a good measure."

In RLVR, the policy model does not "reason" like a human — it acts as an aggressive mathematical optimizer hunting for shortcuts to maximize reward scalars.

#### Case Studies in Reward Hacking

1. **The Empty `<think>` Tag Hack:**
   - Vulnerability: If the format verifier grants $+0.5$ reward whenever `<think>` and `</think>` tags are present in the response.
   - Hacking Behavior: The model learns to immediately output `<think></think>` without writing any reasoning steps inside.
   - Defense: Require a minimum non-whitespace token count ($N_{\text{tokens}} \ge 10$) inside the reasoning block.

2. **The Trash Text Denominator Dilution Hack:**
   - Vulnerability: If the prompt-echo verifier calculates penalty as:
     $$\text{Echo Ratio} = \frac{\text{Matching Prompt 2-grams}}{\text{Total Completion Tokens}}$$
   - Hacking Behavior: If the model accidentally copies part of the prompt, it learns to append 300 tokens of irrelevant gibberish to inflate the denominator and push the ratio below the penalty threshold!
   - Defense: Always normalize repetition penalties by Prompt Length, never Completion Length.

#### Robust Reward Function Design Rules

```python
# Rule 1: Always unit-test reward functions against adversarial probes
# src/rl/verifiers.py
def test_anti_reward_hacking_probes():
    # Probe 1: Empty think tags should fail
    assert format_verifier.score("<think></think>42", item).reward < 0.0
    
    # Probe 2: Filler text + wrong answer must score LESS than clean correct answer
    junk_score = composite_verifier.score(prompt + " " + "junk "*50 + "42", item)
    clean_score = composite_verifier.score("<think>step-by-step</think> 42", item)
    assert clean_score.reward > junk_score.reward
```

---

### 5.14 Program-Aided Reasoning (Python Tool-Calling) Integration Guide

While Retrieval-Augmented Generation (RAG) provides external text context to solve formula-recall bottlenecks, an alternative paradigm for mathematical and scientific tasks is Program-Aided Reasoning (PAL) or Tool-Calling (Python Execution).

```
+---------------------------------------------------------------------------------------+
|                              RAG vs PYTHON TOOL-CALLING                               |
+------------------------------------+--------------------------------------------------+
|      RAG CONTEXT GROUNDING         |           PYTHON CODE INTERPRETER            |
+------------------------------------+--------------------------------------------------+
| · Solves: Closed-book recall       | · Solves: Arithmetic & multi-step calculation   |
| · Mechanism: BM25 text retrieval   | · Mechanism: Intercepts code, runs Python REPL   |
| · Output: Paper excerpts           | · Output: Exact math results ($59,100 \text{ km}^2$) |
+------------------------------------+--------------------------------------------------+
```

#### Step-by-Step Implementation Guide for Tool-Calling

Here is how you can implement a Python Code Interpreter tool-calling loop in `chat.py`:

##### Step 1: Prompt the Model to Output Executable Python Code
Instruct the model during SFT/RL to write python blocks inside `<think>`:
```
Calculate fault area:
<think>
```python
M0 = 3.55e22
mu = 3e10
D = 20
A = M0 / (mu * D)
print(f"Area: {A/1e6:.1f} km2")
```
</think>
```

##### Step 2: Implement the Secure Tool Execution Loop (`chat.py`)

Add the following execution interceptor to your inference engine:

```python
import re
import sys
from io import StringIO

def execute_python_code_block(code_str: str, timeout_sec: float = 2.0) -> str:
    """Securely execute a python snippet and capture print() output."""
    # Restrict dangerous builtins for safety
    safe_globals = {
        "__builtins__": {
            "print": print, "range": range, "len": len, "sum": sum,
            "min": min, "max": max, "abs": abs, "round": round, "int": int, "float": float
        }
    }
    old_stdout = sys.stdout
    sys.stdout = capture = StringIO()
    try:
        exec(code_str, safe_globals)
        output = capture.getvalue().strip()
        return output if output else "Executed successfully (no output)."
    except Exception as e:
        return f"Execution Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout

def generate_with_tool_calling(engine, prompt: str, max_turns: int = 3):
    """Tool-calling generation loop intercepting <python> blocks."""
    current_context = prompt
    
    for turn in range(max_turns):
        completion = engine.generate(current_context)
        
        # Check if the model emitted a python code block
        match = re.search(r"```python\n(.*?)\n```", completion, re.DOTALL)
        if not match:
            # No tool call; return final answer
            return completion
        
        # Intercept code snippet & run in interpreter
        code_snippet = match.group(1)
        tool_output = execute_python_code_block(code_snippet)
        
        # Append tool result back into context
        current_context += completion + f"\n<result>\n{tool_output}\n</result>\n"
    
    return completion
```

##### Step 3: Train Verifiers for Code Execution
In RLVR (`src/rl/train_grpo.py`), reward the model $+0.5$ if the emitted code runs cleanly without syntax errors, and $+1.0$ if the final output printed by Python matches the ground truth target!

---

## 6. How the evaluation works

This section is the instrument; [§7](#7-what-it-found) is what it read.

### 6.1 What is being measured

Per [§3.2](#32-four-kinds-of-metric): `<think>` rate and numerical accuracy are *held-out
same-construct* metrics; echo and contamination are hygiene; the composite reward is a
training metric, reported only as a labelled reference row.

### 6.2 Paired evaluation

The relevant evidence is not how many items each model gets right — it is **how many
prompts only SFT gets right versus how many only GRPO gets right**. Pairing removes most of
the variance from prompt difficulty. `eval_rl.py` stores per-item outcomes (so pairing is
possible at all), runs McNemar on discordant pairs, and bootstraps paired CIs.

The same lesson applies to loss: two nested test files once gave opposite signs for the
same two models under packed scoring. Packing is optimistically biased by ~0.149 nats —
15× the 0.0096-nat effect being measured. [Pitfall 3](PITFALLS.md).

### 6.3 Multiple samples, pass@k, and power

```
[power] at n=166, k=4 sample(s)/prompt, baseline 7.1%: MDE ≈ 10.0% at 80% power.
        A null result below that threshold means UNDERPOWERED, not 'no effect'.
```

With 166 numerical prompts this benchmark reliably detects only large changes; on the
90-item test half, worse still. That is why "numerical reasoning did not improve" is stated
as not supported, never as disproven.

pass@k separates "can sometimes do it" from "does it reliably", and carries its own,
much higher chance floor: given four attempts, must-compute items land 13.6% of the time
against a 20.3% pass@4 floor.

### 6.4 Tiering: why one aggregate is not enough

A single number cannot distinguish "competent but unreliable" from "scoring by accident",
and those imply opposite next steps. `diagnose_capability tier` splits the numerical set by
whether the answer must be derived or is already present in the prompt, and again by
derivation depth — each tier reporting its own chance floor, because target clustering
differs by tier and one global floor would flatter some tiers and penalise others.
The results are in [§7.2](#72-retrieval-vs-computation-tiers).

---

## 7. What it found

### 7.1 Headline results

On the audit test half (see the framing in [§3.1](#31-the-split-contract-and-what-it-cannot-undo)),
each value a mean over prompts of a per-prompt mean over 4 samples:

| metric | population | SFT | GRPO @100 | verdict |
|---|---|---|---|---|
| `<think>` rate | 90 numerical | 73.1% | 97.8% | large; consistent across dev and test |
| numerical acc @5% | 90 numerical | 10.6% | 11.4% | Δ=+0.8% [−3.1%, +4.7%] — not resolvable |
| contamination / echo | 100 prompts | 0.0% | 0.0% | nothing detected by either detector |

### 7.2 Retrieval vs computation tiers

Pooled diagnostic over all 166 held-out numerical items (dev + test), 4 samples each:

| tier | n | chance floor | SFT | GRPO |
|---|---|---|---|---|
| answer already in the prompt | 34 | 7.2% | 20.6% | 29.4% |
| must be computed | 132 | 5.5% | 3.6% | 3.6% |

All of the aggregate comes from the retrievable fifth: (34 × 29.4% + 132 × 3.6%) / 166 =
8.9%. Every depth tier — 6-step through 20+-step — sits at or below its own floor.

### 7.3 Capability map

Formatting. RL raised `<think>` compliance to ~98%, an effect prompting could not
reproduce and that holds on both halves of the audit split. The clearest positive result
here.

Prompt-value extraction. 20.6% → 29.4% against a 7.2% floor. Suggestive; the CI
touches zero.

Multi-step numerical computation. No above-baseline evidence at any derivation depth,
under a metric whose floor and ceiling are both measured. The correct statement is *the
evaluation found no evidence of computation*, not the model cannot compute — the
benchmark cannot resolve effects below ~10 points.

Style. More disciplined in format, with no prompt echo or dataset contamination
detected by the tested detectors. That is not "non-hallucinatory": nothing here measures
the factual correctness of prose.

Unmeasured. Conceptual factuality on 642 of 808 held-out prompts (~79%) — the largest
slice of real usage has no correctness metric at all. Also unmeasured: multi-turn
behaviour, refusal, calibration, and any independent-construct reasoning benchmark.

### 7.4 Why the null result is not a token-budget story

An earlier draft blamed "7.3 tokens/parameter, far below compute-optimal." That was the
wrong accounting twice over. Distinguish:

| quantity | value |
|---|---|
| unique corpus tokens | 822.7M |
| token exposures processed (3.8 epochs) | 3.13B |
| exposures per non-embedding parameter | **31.1** |

Against the pretraining repo's own measured IsoFLOP frontier — which puts optimal
tokens/param near 46 at its largest budget, not Chinchilla's ~20 — the base is modestly
under-trained, within a factor of ~1.5 of its frontier. (Both figures must be counted the
same way to be comparable: the 113.3M here is the *total* weight-tied parameter count, and
100.7M excludes the single shared embedding/unembedding matrix.) Not obviously data-starved, and
certainly not by the 3× the old figure implied.

More likely explanations for the null: a pretraining distribution of seismology prose
rather than a dense curriculum of executable derivations; held-out questions requiring a
median of eleven derivation steps; numerical correctness scoreable in only 77 of 200
updates; and 113M parameters attempting the capability that scales worst with size.

---

## 8. How this recipe scales

Report portable units, not epochs and steps. "200 RL steps" does not transfer to another
model or batch size; "3,200 completions and 954k generated tokens" does.

| axis | measured here | what to add |
|---|---|---|
| instruction data | 15.7k pairs, composition-matched curve | equal-compute comparison, multiple seeds |
| model size | one 113M model | no model-size law was measured; 0.5B/1B would test it |
| RL compute | 200 steps · 3,200 completions · 954k tokens · 7.7% of pool · 3.0 GPU-h | more prompts before more steps |
| systems | 2× A30 | single-GPU recipe in [§2](#2-quick-start); sharding or LoRA beyond ~1B |
| evaluation | 166 numerical, MDE ≈ 10 pts | ~470 prompts for 5 pts, ~2,900 for 2 pts (paired) |

What to fix first. (1) The benchmark — at 166 items it cannot see anything smaller than
a doubling, so it saturates as a decision tool before the model does. (2) The
step mix — numerical correctness was scoreable in only 38.5% of updates, so most compute
buys format. (3) Prompt coverage — 7.7% of the pool means the run is sampling-limited, not
step-limited.

These are the first bottlenecks this experiment can directly repair. Model capacity may
already be binding too — all four constraints could bind at once — but only one model size
was run, so no ordering between them is established.

---

## 9. Failure atlas & pipeline pitfalls

A comprehensive taxonomy of post-training pipeline pitfalls, diagnostic tests, and regression fixes—including data leakage prevention, evaluation decoupling, reward hacking mitigation, and checkpoint selection criteria—is documented in [PITFALLS.md](PITFALLS.md).

---

## 10. Next experiments, each with a kill gate

Calculator tool use — gated, and abandoned. Before building anything, a Phase-0 test
executed the arithmetic the model wrote in its own `<think>`: 1.3% against a 1.8% floor,
worse than its stated answer, against an instrument ceiling of 50.8% on ground truth. The
expressions are noise, so a calculator has nothing correct to execute; the bottleneck is
formula recall, not arithmetic. Cost of finding out: ~40 GPU-minutes instead of a day.
That is what a gate is for.

Open-book / RAG — gated, and empirically confirmed. The one thing this model demonstrably does is pull numbers and relations out of context. Providing retrieved paper excerpts via `SeisRAGRetriever` converts closed-book formula recall into open-book extraction & calculation. In benchmark testing (`python -m src.sft.chat --rag`), grounding the RL model with domain context cleared the zero-shot arithmetic bottleneck, deriving exact steps and matching target values ($R \approx 8.67 \text{ km}$, $A \approx 59,100 \text{ km}^2$). See [§1.3](#13-what-the-three-stages-actually-look-like) for full trace.

Rejection-sampling fine-tuning. Same verifier, same rollout budget, offline. Gate: if
RFT matches RL's `<think>` gain, online policy gradients are not earning their complexity.

Formula-recall curriculum. The oracle test localized the failure to stating the right
relation. Gate: measure formula-recall accuracy directly before building curriculum data.

Stronger base. No model-size scaling was measured. Gate: run this evaluation on a 0.5B
base before further post-training investment at 113M.

---

## 11. Repository layout

```
data/
  sources/     paper.jsonl  paper_raw.jsonl  paper_rewritten.jsonl
               fundamentals.jsonl  earthquakes.jsonl      ← corpus inputs
  splits/      train.jsonl  val.jsonl  test.jsonl         ← identity-level, hash-pinned
  rl/          train.jsonl  val.jsonl  test.jsonl         ← GRPO prompts (RL schema)
  eval/        heldout_subsets.jsonl                      ← 808 recs, `subsets` tag
               scaling_2k/4k/8k.jsonl                     ← composition-matched series
checkpoints/
  sft/         selected.pt  sft_ckpt_best.pt              ← Phase II deployment model
  grpo/        selected.pt → grpo_ckpt_step_100.pt        ← Phase III deployment model
  runs/        sft.csv  grpo.csv  + 26 historical logs    ← every training curve
src/
  sft/         generate_sft_data → build_corpus (clean · merge · split)
               dataset · train_sft · chat · rag_retriever · run_data_scaling_sweep
               checkpoints.py       ← select · compare · strip
               splits.py            ← the train/val/dev/test contract
  rl/          prepare_rl_dataset · verifiers (+ safe_eval) · train_grpo · eval_rl
  analysis/    validate_measurement (floor · hacking · group)
               diagnose_capability  (tier · oracle · prompting)
  plots/       shared style in __init__ + one entry point (python -m src.plots)
tools/gen_fundamentals/    portable stdlib-only data generators
```

Nothing in `assets/` is drawn from numbers pasted into the plotting code — a defect this
reproducibility and auditability.

---

## Further reading

- **[PITFALLS.md](PITFALLS.md)** — the failure atlas: symptoms, false stories, root causes,
  fixes, regression tests, invalidated artifacts.
- **[nanoGPT-Seis](https://github.com/jiazhe868/nanogpt-seis)** — Phase I: crawl, clean, tokenize, pretrain, plus
  the IsoFLOP scaling experiments this README's compute accounting refers to.

---

## Data & licensing

- **Code** is MIT — see [LICENSE](LICENSE), the same terms as
  [nanoGPT-Seis](https://github.com/jiazhe868/nanogpt-seis).
- **The SFT/RL corpora are not redistributed.** `data/` is git-ignored. Rebuild it with
  `python -m src.sft.build_corpus` from the pretraining corpus plus your own teacher API
  (`TEACHER_API_BASE` / `TEACHER_MODEL` — any OpenAI-compatible endpoint). Teacher-generated
  data inherits whatever terms your provider attaches to model output, and the underlying
  source text keeps the licences listed in the pretraining repo.
- **Model weights** are a derived artifact, distributed (if at all) via a GitHub Release or
  the Hugging Face Hub, not committed to git.
- **Training logs are committed** (`checkpoints/runs/*.csv`), because every figure in
  `assets/` regenerates from them — that is what makes the numbers here checkable.
