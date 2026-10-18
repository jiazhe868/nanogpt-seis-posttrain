<div align="center">

# nanoGPT-Seis-Posttrain

[English](README.md) | [中文](README.zh-CN.md) | [🤗 Hugging Face Weights](https://huggingface.co/jiazhe868/nanogpt_seis_posttrain)

Post-training a 113M seismology language model to answer domain questions, including numerical ones:
supervised fine-tuning, GRPO with verifiable rewards, and answers computed by executing the model's
own equations. Everything runs on 2× NVIDIA A30.

</div>

![pipeline](assets/workflow.png)

## Summary

A pretrained model continues text; it does not answer questions. Supervised fine-tuning (SFT) on
14.1k question–answer pairs written from seismology papers teaches the 113M model to answer
(Phase II). Its numerical answers are rarely right: 4.7% on 153 held-out questions from papers it
was never fine-tuned on.

SFT answers are often unfinished: the reasoning block stays open and no final answer is given.
GRPO with a verifier reward fixes the format (Phase III): well-formed answers rise from 71% to
98%. Accuracy moves little (6.1%), because correct numerical answers are too rare for the reward to
reinforce.

Small models often fail at multi-step arithmetic, since every digit is sampled and errors
compound. Tests showed a second problem: the model also picks the wrong relations. Therefore, I
let the model write its working as short equations that an interpreter executes, in the notation
of the papers it was pretrained on. I trained it on 105k such solutions written by a local 27B
teacher (Phase IV). Accuracy rises to 19.7%, and a majority vote over 8 executed answers gives
22.2%.

GRPO on executed answers adds up to 3 points on the held-out set, while the problems it trains on
go from 49% to 85%. A transfer study shows why: a gain reaches a new question in proportion to how
similar it is to the training problems, and the held-out questions come from papers the model never
practised on. Practising the first step of problems the model never solves makes it solve them 7× as
often. A single pass over 9,000 broader problems adds at most 2.4 points (§5.19). That run, with
problems from every paper in the corpus and vote@8, is the deployed model: 23.3%.

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Quick start](#2-quick-start)
3. [Define success before training](#3-define-success-before-training)
4. [Phase II — Supervised fine-tuning](#4-phase-ii--supervised-fine-tuning)
5. [Phase III — RL from verifiable rewards](#5-phase-iii--rl-from-verifiable-rewards)
   - 5.14 [A calculator fixes arithmetic, not setups](#514-a-calculator-fixes-arithmetic-not-setups)
   - 5.16 [Finding the bottleneck](#517-finding-the-bottleneck)
   - 5.17 [Phase IV — executed equations from a local teacher](#518-phase-iv--executed-equations-from-a-local-teacher)
   - 5.18 [Where GRPO helps: a transfer study](#519-where-grpo-helps-a-transfer-study)
6. [How the evaluation works](#6-how-the-evaluation-works)
7. [What it found](#7-what-it-found)
8. [How this recipe scales](#8-how-this-recipe-scales)
9. [Failure atlas & pipeline pitfalls](#9-failure-atlas--pipeline-pitfalls)
10. [Repository layout](#10-repository-layout)

[PITFALLS.md](PITFALLS.md) collects the failure modes met while building this pipeline and the checks
that catch them.

---

## 1. Results at a glance

### 1.1 Model, data and compute

| | |
|---|---|
| Base model | 113.3M parameters (GQA, RoPE, RMSNorm, SwiGLU), 4096-token context, pretrained on 823M tokens of seismology papers, Wikipedia and FineWeb-Edu |
| Phase II data | 15.7k question–answer pairs from paper chunks, fundamentals and historical earthquakes: 14.1k for training, the rest held out |
| Phase IV data | 105k solutions in paper notation, checked by execution, written by a local Qwen3.8-27B-FP8 teacher; plus 6k Phase II pairs replayed |
| RL | GRPO, 8 samples per question, rewards from programmatic verifiers |
| Hardware | 2× NVIDIA A30 (24 GB each) |
| Evaluation | 153 numerical questions from papers never used in fine-tuning (most appear in the pretraining text), 6 seeds × 4 samples each |

### 1.2 Results by phase

<p align="center"><img src="assets/grpo_benchmark_comparison.png" width="92%"></p>

| phase | model | well-formed answers | accuracy | needs computation (134) | within 1% |
|---|---|---|---|---|---|
| I | pretrained base | — | — | — | — |
| II | SFT | 71% | 4.7% | 2.8% | 1.9% |
| III | + GRPO | 98% | 6.1% | 3.1% | 2.2% |
| IV | executed equations | 100% | 19.7% | 20.1% | 16.7% |
| | + GRPO on the corpus | 100% | 22.0% | 20.7% | 19.3% |
| deployed | + vote@8 | 100% | 23.3% | 21.6% | 20.5% |

An answer counts as correct within 5% of the reference. A model that ignores the question scores
3.3% on the questions that need computation; the 27B teacher, solving them blind, scores 48.5%.

Phase III changed the format much more than the numbers (4.7% → 6.1%), because there were few
correct answers to reinforce (§5.9). Phase IV changed the numbers by moving the arithmetic to an interpreter
(§5.18). GRPO on top of Phase IV, on problems written from every paper in the corpus but never the
test questions, adds 2.4 points, and vote@8 makes it the deployed model (§5.19). Its gain comes from
questions whose answer appears in the prompt: on the questions that need computation it scores
21.6%, against 22.4% for executed equations with vote@8 alone. RL gains stay near the problems the
model practises on (§5.19). The remaining gap to the teacher is mostly topic knowledge, meaning the right relation and
constants for a paper the model has not practised on.

### 1.3 One question, every phase

<p align="center"><img src="assets/stage_outputs.png" width="96%"></p>

---

## 2. Quick start

```bash
conda activate nanogpt_seis      # same env as the pretraining repo (CUDA-12.4 torch)

# --- chat with the deployed model ------------------------------------------
python -m src.sft.chat                       # the deployed model (configs/deploy.yaml): GRPO on the corpus,
                                             # equations executed at </think>, vote@8 over the answers
python -m src.sft.chat --prompt "An Mw 6.9 earthquake ... what is M0 in N m?"
python -m src.sft.chat --ckpt-path checkpoints/grpo/selected.pt   # the earlier prose model (GRPO v4)

# --- Phase II: SFT ----------------------------------------------------------
python -m src.sft.build_corpus merge         # 4 sources -> 15,659 pairs (deduped)
python -m src.sft.build_corpus split         # identity-level split + hash-pinned test
torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_base.yaml
python -m src.sft.checkpoints select --ckpt-dir checkpoints/sft --link-best checkpoints/sft/selected.pt

# --- Phase III: verifier RL -------------------------------------------------
python -m src.rl.prepare_rl_dataset
torchrun --standalone --nproc_per_node=2 -m src.rl.train_grpo --config configs/rl_grpo.yaml

# --- Phase IV: executed equations (teacher data: §5.18) ---------------------
torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_pot_notation2.yaml --snapshot-epochs
torchrun --standalone --nproc_per_node=2 -m src.rl.train_grpo --config configs/rl_grpo_v14_corpus.yaml
                                             # the deployed run; its pool: scripts/experiments/v1314_chain.sh

# --- measurement audit: run BEFORE believing any result ---------------------
python -m src.sft.splits data/rl/test.jsonl            # the split contract
python -m src.analysis.validate_measurement all        # floor / ceiling / reward hacking
python -m src.analysis.validate_measurement group      # one real G=8 group, scored
python scripts/experiments/real6_eval.py real6_deployed.json \
    checkpoints/grpo_v14_corpus/grpo_ckpt_step_75.pt 31,32,33,34,35,36 1.0 1 deployed   # §1.2: 6 seeds × 4
python -m src.rl.eval_rl --grpo-ckpt checkpoints/grpo/selected.pt --val-file data/rl/test.jsonl \
    -k 4 --out-json eval_rl_prose.json                 # prose models: paired, McNemar, pass@k, power
python -m src.analysis.diagnose_capability tier        # where the accuracy comes from

python -m src.plots                                    # regenerate every figure in assets/
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
> pooled result. It is not a fully frozen final test, so this README does not call it one.
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

This repo has no independent-outcome metric. Eval accuracy uses the same field and
tolerances as `NumericalVerifier`; `<think>` rate regexes the same tags `FormatVerifier`
checks. Those are the same constructs on held-out prompts: a generalization check, not
evidence of a broader capability. §5.13 sets the training reward, the advantage and the
evaluation side by side, and shows how each is designed against reward hacking.

### 3.3 Floor, ceiling, and blind baselines

A percentage is uninterpretable without both ends of its scale:

- **floor** — what a knowledge-free policy scores
- **ceiling** — what the ground truth scores through the same parser

For numerical accuracy at 5% tolerance, on the original 166 items:

| scoring rule | ceiling | floor | usable range |
|---|---|---|---|
| min over first 8 numbers (as originally shipped) | 68.1% | 14.3% | 53.8% |
| last 3 numbers (current) | 86.1% | 5.9% | 80.2% |

The original rule's floor sat above the model. Restricting scoring to the answer tail
improves both ends — the stated answer sits near the end, so "first 8" often never
reached it. On the 134 relabelled questions that need computation, the scale now runs from
3.3% (a model that ignores the question) to 85.8% (the reference answers), and the deployed
model scores 21.6%.

```bash
python -m src.analysis.validate_measurement floor
```

### 3.4 The decoding contract

At this scale the decoder can move results as much as training. For the prose models,
repetition penalty 1.05 gives 56% repeated 5-grams; raising it to 1.2 roughly doubled
measured accuracy for both models and flipped the sign of the SFT→GRPO delta, an effect 2.3×
the training effect. The executed-equation model needs the opposite: its programs reuse
symbol names, so it decodes with no penalty (§5.18).

The decoder is therefore pinned with the model: `configs/deploy.yaml` holds the deployed
settings (temperature 0.3, top-k 40, rp 1.0, vote@8), `chat.py` uses temperature 0.3 /
top-k 40 / rp 1.2 for the prose models, and evaluation uses the same settings as serving.
[PITFALLS B3](PITFALLS.md).

### 3.5 Provenance on every result

The evaluation scripts write the checkpoint, split and decoding parameters next to the
scores (`src/rl/eval_rl.py`, `scripts/experiments/real6_eval.py`), and the plotting code warns
when eval provenance does not match the training log being plotted. This exists because a stale results file from a previous run was once
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

```math
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t m_t}\sum_t m_t \log p_\theta\!\left(y_t \mid x,\, y_{\lt t}\right)
```

with $`m_t = 0`$ for system / user / padding tokens and $`m_t = 1`$ for assistant tokens. In
code it is a few lines, and they are what make this instruction tuning rather than more
pretraining:

```python
# src/sft/dataset.py — system, user and assistant-header tokens get label -100 (ignored)
system_labels = [ignore_index] * len(system_tokens)
user_labels = [ignore_index] * len(user_tokens)
assistant_header_labels = [ignore_index] * len(assistant_header_tokens)
response_target_labels = list(response_target_tokens)      # the loss is computed only here
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
| About 13% of raw generations were off-domain | At 113M there is no spare capacity. Every off-topic example displaces a seismology one, and the model pays for both. | Filter by domain before it reaches training. |

The general lesson: **a small model has no capacity to ignore bad data.** A 70B model
exposed to occasional phantom citations may treat them as noise; a 113M model treats them
as the pattern. Data curation is a bigger lever at small scale than at large.

Cleaning kept 27% of the first paper batch: 17.3k raw generations became 4.7k usable pairs.
Rewriting recoverable pairs (2.5k) and two generated sources, fundamentals (4.0k) and historical
earthquakes (4.4k), grew the corpus to 15.7k pairs, 14.1k of them for training. Discarding
three-quarters of expensively generated data feels wasteful and is the correct call: [§4.9](#49-a-local-data-scaling-curve) shows that data volume buys ~0.08 nats per
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
loss = loss / group_size
```

### 4.7 Checkpoint selection on behaviour, not loss

Goal. Ship the checkpoint that behaves best, not the one with the lowest likelihood.

<p align="center"><img src="assets/sft_training_dynamics.png" width="92%"></p>

The curves (peak learning rate 2e-4) look like textbook overfitting: by step 200 train loss
is down to 1.18 while validation loss has risen from 2.14 at step 80 to 2.29 (2.58 by the last
step, 336). The natural reading of it is wrong here.

Mechanism. Cross-entropy scores the probability of one specific reference wording. As
training continues the model sharpens onto its own phrasing, lowering the reference's token
probability without making generations worse:

| | best (step 80) | last (step 336) |
|---|---|---|
| held-out loss | 2.2700 | 2.7798 (+0.51 nats, t=+33) |
| `<think>` rate (earlier 166-question test) | 61.4% | 74.1% (z=+2.47) |

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

Evidence. On the shipped Phase II run (peak learning rate 1e-4) the rule selected epoch 8;
validation loss would have picked epoch 3 (step 120), the candidate with the worst `<think>`
rate of the three.

Invariant. The rule is a module-level constant, not a runtime argument — it cannot be
tuned after seeing the table.

### 4.8 What SFT changed

SFT changed what the model *is for*, and almost nothing about what it knows.

Before SFT the model has no notion of a turn. Handed a question it continues the document —
plausible seismology prose that never addresses you, and never stops, because nothing in
pretraining taught it that a response has an end. After SFT it answers, and it terminates.
Those two are the whole interface, and both are learned entirely in this phase.

It also partly acquires the *shape* of reasoning: about 70% of numerical answers
now arrive wrapped in a `<think>` block. Partly, not reliably — closing that gap is what
Phase III is for.

What did not move is the arithmetic. On questions whose answer is already stated somewhere in
the prompt, SFT reaches about one in five — real task adaptation, since it must find and
restate the right quantity. On questions whose answer must actually be derived, it sits at its
chance floor. The distinction matters more than any single accuracy number, and
[§1.2](#12-results-by-phase) reports results split this way.

SFT taught an interface and a habit of showing work. It did not add knowledge the base model
lacked: the objective imitates the surface form of answers, and computation is not surface form.

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
> One seed per point; no regression uncertainty is reported. A stronger design would compare equal-epoch vs equal-assistant-token-exposure
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
one policy-gradient step. The listing is the v3 objective that §5.5–§5.9 analyse. From v4 on, the
trainer still centres each group on its mean, but scales by one std pooled over the batch, uses a
k2 KL loss and a constant normaliser (§5.8).

```python
# src/rl/train_grpo.py, v3 settings (adv_norm: group, kl_loss: k3, loss_agg: seq_mean)
rewards    = [reward_engine.score(c, item) for c in completions]     # G rollouts
advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)     # group baseline

# k3 KL estimator against the frozen SFT reference, masked to SAMPLED tokens only
per_token = -(logp * advantage - beta * kl) * sampled_mask
loss = (per_token.sum(dim=1) / seq_tokens).mean()
```

This is REINFORCE ([Williams 1992](https://doi.org/10.1007/BF00992696), Machine Learning 8:229–256) with a
group-relative baseline. The group mean replaces the learned critic of PPO
([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)): with G samples of the same
prompt it is a zero-parameter, low-variance baseline. v3 then divided by each group's std to keep the
update scale comparable across easy and hard prompts; v4 divides by one std pooled over the batch,
so a group whose rewards differ by 0.01 keeps small advantages.

> Not strictly unbiased. Each completion contributes to its own group mean and standard
> deviation, so the finite-G estimator is self-normalized rather than unbiased. Subtracting
> a mean that includes the current sample scales the expected policy-gradient term by roughly
> `1 − 1/G`; standardizing by the group std adds further finite-sample bias. A leave-one-out
> mean would remove the simplest own-sample component, though the normalization would still
> deserve separate treatment. None of this invalidates the method — it just makes "unbiased"
> the wrong word.

Why that loss does what it claims. For one sampled completion, the policy term is

```math
\mathcal{L}_{\text{policy}} = -A \log p_\theta(y \mid x)
```

Under gradient descent:

- if $`A \gt 0`$, reducing the loss increases $`\log p_\theta(y \mid x)`$ — the completion
  becomes more likely;
- if $`A \lt 0`$, reducing the loss decreases it — the completion becomes less likely;
- if every completion in a group scores the same, every $`A`$ is zero and **nothing is
  learned**, however high or low that common reward was. This is the whole content of
  [§5.5](#55-what-one-real-group-teaches).

The KL term plays the opposite role: it penalises movement away from the frozen SFT
reference even when the reward favours that movement. The reward pulls, the KL anchors,
and β sets the exchange rate between them.

The KL estimator, as logged (and as v3's KL loss). Define the log-ratio on the sampled token,
reference minus policy:

```math
r_t = \log p_{\text{ref}}(y_t \mid \cdot) - \log p_\theta(y_t \mid \cdot)
```

Schulman's k3 estimator ([2020](http://joschu.net/blog/kl-approx.html)) is
`k3(y) = y - 1 - log y` evaluated at `y = e^{r_t}`, i.e.

```math
\widehat{\text{KL}}_t = e^{r_t} - r_t - 1
```

```python
# src/rl/train_grpo.py
log_ratio = logp - c["ref_logp"]                    # policy minus reference
kl_k3 = torch.exp(-log_ratio) + log_ratio - 1.0     # e^{r_t} - r_t - 1 >= 0, always logged
kl_term = kl_k3 if cfg["kl_loss"] == "k3" else 0.5 * log_ratio.pow(2)   # v3: k3; v4 on: k2
```

Note the convention: with $`r_t`$ defined reference-minus-policy, this estimates
$`\mathrm{KL}(p_\theta \Vert p_{\text{ref}})`$ — the direction that penalises the policy
for putting mass where the reference does not. It is non-negative for every token (since
$`e^r - r - 1 \ge 0`$), which the naive estimator $`-r_t`$ is not; that is why it is used, and
why a KL curve here can never go negative even on a noisy batch.

#### A minimal, runnable implementation

The whole algorithm is about forty lines. This mirrors the v3 settings of `src/rl/train_grpo.py` with the
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
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

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

Five details in that code decide whether it works:

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
  §5.3 point 3 for when that trade flips.) v4 replaced it with a constant normaliser (rollouts ×
  `max_new_tokens`), so every token weighs the same, which is DAPO's argument.

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
  1. Policy / Actor ($`\pi_\theta`$): The model being optimized.
  2. Reference Policy ($`\pi_{\text{ref}}`$): Frozen SFT model to anchor KL divergence.
  3. Reward Model ($`RM_\phi`$): Scalar neural network trained on pairwise preferences $`y_w \succ y_l`$.
  4. Value / Critic Network ($`V_\psi`$): Neural network predicting expected cumulative future reward from generation state $`(x, y_{\lt t})`$.

- **Mathematical Formulation:**
  For token generation step $`t`$, advantages are estimated via Generalized Advantage Estimation (GAE):

```math
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t), \quad \hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
```

  The policy is updated using the PPO clipped surrogate loss:

```math
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right], \quad \text{where } r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{\lt t})}{\pi_{\text{old}}(y_t \mid x, y_{\lt t})}
```

  The joint objective optimizes policy, value fitting, and KL constraint:

```math
\mathcal{L}^{\text{PPO}}(\theta, \psi) = - L^{\text{CLIP}}(\theta) + c_1 \mathbb{E}\left[ (V_\psi(s_t) - R_t)^2 \right] + c_2 \mathrm{D}_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})
```

  (Written as a loss to minimise, so $`-L^{\text{CLIP}}`$. Most RLHF implementations fold the
  KL into the per-token reward, $`r_t \leftarrow r_t - \beta \log\frac{\pi_\theta}{\pi_{\text{ref}}}`$,
  rather than adding it as a separate term; the two differ in whether the critic sees the
  penalty.)

- **Why It Suffers at Scale:**
  - **Memory Overhead:** Loading 4 large models concurrently consumes massive VRAM or requires CPU offloading.
  - **Critic Instability:** Fitting a high-dimensional text-generation Critic $`V_\psi(s)`$ introduces high variance during autoregressive generation, causing loss divergence in LLMs.

---

#### 2. DPO (Direct Preference Optimization) — Analytic Elimination of the Reward Model & Critic
- **Architecture & Structural Reduction:**
  Eliminates the Reward Model $`RM_\phi`$, Critic Network $`V_\psi`$, and online generation loop entirely, operating with only 2 models: Policy $`\pi_\theta`$ and Reference $`\pi_{\text{ref}}`$.

- **Mathematical Derivation:**
  Start from the exact KL-constrained maximum-reward RL formulation:

```math
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x,y) \right] - \beta \, \mathrm{D}_{\text{KL}}\left(\pi(y|x) \;\parallel\; \pi_{\text{ref}}(y|x)\right)
```

  This problem has a closed-form optimum — a Gibbs/Boltzmann tilt of the reference policy
  (no preference model is involved yet):

```math
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right) \iff r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
```

  Inverting it expresses the reward in terms of the **optimal** policy. DPO's move is then to
  *parameterise* the learned policy as that optimum ($`\pi_\theta \equiv \pi^*`$) — an assumption,
  not an algebraic consequence — which turns $`\pi_\theta`$ into an implicit reward model.
  Substituting into the Bradley-Terry preference probability
  $`P(y_w \succ y_l \mid x) = \sigma(r(x,y_w) - r(x,y_l))`$ makes the intractable partition
  function $`\log Z(x)`$ cancel, because it depends only on $`x`$ and appears in both terms:

```math
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
```

- **Pros & Caveats:**
  - Eliminated: Critic network $`V_\psi`$, Reward Model $`RM_\phi`$, and online sampling loops.
  - Caveat: DPO is strictly offline. It cannot explore new generation trajectories or evaluate deterministic rule-based Python verifiers dynamically during training.

---

#### 3. GRPO / Group-Relative REINFORCE (Selected Here) — Eliminating the Critic via Group Sampling
- **Architecture & Structural Reduction:**
  Uses only 2 models ($`\pi_\theta`$ and $`\pi_{\text{ref}}`$). Eliminates the Critic Network $`V_\psi`$ by replacing it with a zero-parameter sample-mean baseline computed across a group of sampled candidate completions.

- **Mathematical Formulation:**
  For a prompt $`x`$, sample a group of $`G`$ completion rollouts $`\{y_1, y_2, \dots, y_G\}`$ from $`\pi_{\theta_{\text{old}}}`$. Compute deterministic rewards $`\{r_1, r_2, \dots, r_G\}`$ using programmatic Python verifiers (RLVR).
  Compute group-relative advantage without any value network:

```math
\bar{r} = \frac{1}{G}\sum_{i=1}^G r_i, \quad \sigma_r = \sqrt{\frac{1}{G-1}\sum_{i=1}^G (r_i - \bar{r})^2}, \quad A_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}
```

  (v3 used the population std, `std(unbiased=False)`, with $`\epsilon`$ outside the root. From v4
  on, each group is centred on its own mean and scaled by one std pooled over the batch's
  informative groups, `adv_norm: batch` — §5.8.)
  The token-level objective function combines group relative advantage with per-token k3 KL regularization:

```math
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left( \min\left( \frac{\pi_\theta(y_{i,t} \mid \dots)}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid \dots)} A_i, \, \text{clip}\left(\dots\right) A_i \right) - \beta \, \widehat{\mathrm{KL}}_{i,t} \right) \right]
```

- **Why It Fits RLVR:**
  - Two resident models instead of four. The saving is real but not a clean "50% of VRAM":
    the RM and critic are often smaller than the actor, and only trainable models carry
    optimizer state (PPO trains two, GRPO one).
  - Retains active online exploration, so programmatic verifiers can be called on fresh
    rollouts — the property DPO gives up.
  - **The cost moved, it did not vanish.** The critic is gone, but every prompt now needs $`G`$
    rollouts, so generation cost scales with $`G`$. GRPO trades critic memory for sampling
    compute.
  - Note that GRPO is *reward-source agnostic*: DeepSeekMath used it with a learned reward
    model as well as rule-based rewards. The PREFERENCE / RULE-BASED split in the diagram
    above is about how **this repo** uses these algorithms, not an intrinsic property.

---

#### 4. DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization
[Yu et al. 2025](https://arxiv.org/abs/2503.14476). DAPO builds on GRPO and fixes **four**
failure modes seen in long-CoT RL. Its objective drops the KL term entirely:

```math
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[ \frac{1}{\sum_{i=1}^{G} |y_i|} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min\left( r_{i,t}(\theta)\hat{A}_{i,t}, \; \mathrm{clip}\left(r_{i,t}(\theta),\, 1-\epsilon_{\text{low}},\, 1+\epsilon_{\text{high}}\right)\hat{A}_{i,t} \right) \right]
```

```math
\text{subject to}\quad 0 \lt \left| \{ y_i : \text{is-correct}(y_i) \} \right| \lt G
```

1. **Clip-Higher** — decouple the clip range into $`\epsilon_{\text{low}}`$ and
   $`\epsilon_{\text{high}}`$ with $`\epsilon_{\text{high}} \gt \epsilon_{\text{low}}`$. Symmetric
   clipping caps how fast a low-probability-but-correct token can be raised, which drives
   **entropy collapse**; raising only the upper bound preserves exploration.
2. **Dynamic Sampling** — that `subject to` line. Drop any group whose rollouts *all* score
   the same (all correct or all wrong) and resample until the batch is full.
3. **Token-Level Policy Gradient Loss** — note the $`\frac{1}{\sum_i |y_i|}`$ prefactor:
   normalisation is over *all tokens in the batch*, replacing GRPO's per-sequence
   $`\frac{1}{G}\sum_i \frac{1}{|y_i|}`$. Under GRPO's form every sample gets equal weight
   regardless of length, so each token of a long derivation contributes less gradient than a
   token of a short one.
4. **Overlong Reward Shaping** — a soft length-aware penalty for truncated generations
   instead of scoring them as simply wrong, which removes reward noise from responses that
   were cut off rather than incorrect.

**Removing the KL is deliberate**: "during training the long-CoT reasoning model, the model
distribution can diverge significantly from the initial model, thus this restriction is not
necessary." That is a claim about *frontier-scale reasoning runs*, and it does **not**
transfer here — at 113M the reference KL is what stops this policy degenerating. §5.8
shows the other half of that lesson: how β is set matters as much as whether it exists.

> **Why point 2 matters most for this repository.** Dynamic Sampling is the published fix for
> exactly the pathology documented in [§5.5](#55-what-one-real-group-teaches) and
> [§5.9](#59-where-useful-gradient-actually-came-from): a group with zero reward spread
> produces zero advantage and therefore no gradient, no matter how large the reward. The v3 run
> spent updates on such groups instead of resampling them. The executed-equation runs (v9, v11
> onward) build their pools from learnable problems only, solved 1–7 times in 8 by the starting
> model (`tools/filter_learnable_prompts.py`): Dynamic Sampling done offline. Clip-Higher applies
> only when a batch is reused; v5–v8 ran it with [0.8, 1.28]
> ([§5.4](#54-how-this-implementation-differs-from-canonical-grpo)).

---

#### Comparative Architectural Evolution Matrix

| Paradigm | Models in VRAM | Reward Mechanism | Advantage Estimation $`\hat{A}`$ | Primary Structural Reduction |
|---|---|---|---|---|
| PPO | 4 (Actor, Ref, RM, Critic) | Learned Reward Model $`RM_\phi`$ | GAE via Value Critic $`V_\psi(s)`$ | Baseline framework (no reduction) |
| DPO | 2 (Actor, Ref) | Implicit from Log-Ratios | Implicit Pairwise Difference | Eliminated Critic, RM & Online Sampling |
| GRPO (Ours) | 2 (Actor, Ref) | Reward source agnostic; verifiers here | Group sample mean, $`\frac{r_i - \bar{r}}{\sigma_r}`$ | Eliminated critic network $`V_\psi`$ |
| DAPO | 1 (Actor) — KL removed | Reward source agnostic; verifiers in the paper | Same group advantage; **clipping** is asymmetric | Removed KL, length bias, symmetric clip, and zero-variance groups |

### 5.4 How this implementation differs from canonical GRPO

> This is not canonical GRPO, and the distinction is worth understanding.
>
> GRPO, as introduced in DeepSeekMath ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)),
> is described there as "a variant of Proximal Policy Optimization": an **old-policy
> importance ratio, a clipped surrogate objective**
> ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)), group-relative advantages, and
> direct KL to a reference.
> This implementation keeps the group-relative baseline and the reference KL. Most runs (v4,
> and v9 onward) take one update per freshly sampled batch: the importance ratio is then
> identically 1 and the clipped surrogate reduces to REINFORCE. v5–v8 set `num_iterations: 2`,
> reusing each batch once with the ratio to the first-pass policy clipped to [0.8, 1.28]
> (DAPO's Clip-Higher) and truncated importance weights (`tis_cap`) for the gap between the
> KV-cached sampler and the training forward pass (§5.16).
>
> The accurate name for the one-update runs is group-relative REINFORCE
> ([Williams 1992](https://doi.org/10.1007/BF00992696), Machine Learning 8:229–256) with KL regularization (or
> "GRPO-style group-relative policy gradient"). Showing which pieces are essential — group baseline,
> reference KL — and which are implementation choices — ratio, clipping — is more useful
> than claiming the label.

### 5.5 What one real group teaches

Run `python -m src.analysis.validate_measurement group` and you get a single prompt, eight
real rollouts from the Phase III GRPO checkpoint (`checkpoints/grpo/selected.pt`), and the
arithmetic the trainer actually performs.
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
precisely why they dominated what this run actually learned. That is also why the
executed-equation runs score correctness alone (1 within 1%, 0.5 within 5%, 0 otherwise or when
the program does not run): after SFT the other terms are saturated, and their shared gradient
drowned the per-problem signal.

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
scores seven deliberately degenerate completions and the genuine reference against the *real*
reward engine across every
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

### 5.8 Why the v3 reward and KL curves never settled — and the v4 trainer

A verifier-RL run on a fixed prompt pool should trace a recognisable shape: reward rises
fast while the cheap behaviours (format, termination) are learned, rises slowly while the
hard components move, then plateaus as within-group spread runs out; KL to the reference
rises and levels off. The shipped v3 run did neither. Its reward peaked near step 100 and
regressed, and its KL climbed to 0.10 before being dragged back to 0.04
(`assets/grpo_training_dynamics_v3.png`, and the grey curves below). Reading `checkpoints/runs/grpo.csv`
closely, four separate things were wrong: two are measurement artefacts that made the curve
unreadable, and two are optimisation defects that changed what was being learned.

<p align="center"><img src="assets/grpo_stability_v3_vs_v4.png" width="98%"></p>

| cause | evidence in the v3 log | v4 fix (`configs/rl_grpo.yaml`) |
|---|---|---|
| **each logged point was one prompt** | rank 0 logged the mean of 8 rollouts of a single prompt: step-to-step SD 0.35, lag-1 autocorrelation 0.04. After removing type means and a smooth trend the residual SD is 0.26, so even a 20-step mean carries ±0.06 — the size of every feature on the curve | `prompts_per_step: 16` per GPU → 32 prompts × 8 = 256 rollouts per update; every logged number is over the whole global batch; the held-out objective is measured every 10 steps |
| **single-type batches** | numerical steps average −0.64, conceptual −0.19; the numerical share of a 20-step window ranges 20%–70%, so the type mix alone moves the rolling mean by up to 0.22 | `stratify_types: true` — every batch carries the pool's mix |
| **a bang-bang β controller** | ×1.5 per step outside a [0.5, 2]×target deadband. β hit its 0.005 floor at step 6 and stayed there to step 65 (KL unopposed); then five consecutive ×1.5 steps took it 0.038 → 0.192 (steps 96–100), where it stayed for the last 100 steps because KL never left the deadband. The objective changed 38× mid-run; stop rate fell 0.94 (steps 81–100) → 0.51 (131–150) | `kl_controller: fixed`, β = 0.04 — one objective for the whole run (a Ziegler-style `proportional` controller is implemented but was not exercised here) |
| **advantages that amplify noise** | per-group std normalisation turns a group whose rewards differ by a 0.01 length penalty into unit-size advantages | `adv_norm: batch` — centre per group, scale by one std pooled over the batch |

Three smaller defects were fixed at the same time: the KL **loss** used k3, whose gradient
is $`(1 - p_{\text{ref}}/p)\,\nabla\log p`$ — unbounded on tokens the policy has suppressed,
and in expectation the gradient of $`\mathrm{KL}(p_{\text{ref}}\Vert p)`$ rather than the
$`\mathrm{KL}(p\Vert p_{\text{ref}})`$ it logs — so v4 uses $`k_2 = \tfrac12(\log p - \log p_{\text{ref}})^2`$,
whose on-policy gradient is exactly $`\nabla\mathrm{KL}(p\Vert p_{\text{ref}})`$ (k3 is still
what gets logged); there was no LR warmup, so Adam's largest steps came first
(`warmup_steps: 10`); and the loss weighted each sequence by $`1/|y|`$ (`loss_agg: constant`
divides the token sum by rollouts × `max_new_tokens`, the plain REINFORCE gradient). None of
this was affordable at v3's speed — rollouts re-encoded the whole context for every token —
so v4 decodes all P·G rows in one left-padded, KV-cached batch. A v3 step (8 rollouts per
GPU) took ~27 s; a v4 step (128 per GPU) takes ~13 s, ~11 s of it generation. The cached
decoder matches the full forward pass to 3×10⁻⁵ in fp32.

**Was the β jump really what undid v3, or just coincident with it?** Both arms below
restart from the shipped v3 step-100 weights with the v3 algorithm, the same prompt order
and the same seed; the only difference is β.

<p align="center"><img src="assets/grpo_beta_intervention.png" width="98%"></p>

| held-out, restarted at v3 step 100 | +10 | +20 | +30 | +40 | +50 | +60 | mean +10…+60 |
|---|---|---|---|---|---|---|---|
| reward, β = 0.192 (what the controller set) | −0.460 | −0.469 | −0.425 | −0.347 | −0.387 | −0.308 | **−0.399** |
| reward, β = 0.035 (its mean over steps 81–100) | −0.271 | −0.292 | −0.296 | −0.233 | −0.236 | −0.265 | **−0.266** |
| stop rate, β = 0.192 | 0.56 | 0.58 | 0.66 | 0.72 | 0.73 | 0.77 | |
| stop rate, β = 0.035 | 0.78 | 0.79 | 0.86 | 0.93 | 0.89 | 0.81 | |

Both start from −0.233 / stop 0.82. Within ten steps the β = 0.192 arm loses 0.23 of
held-out reward, its stop rate falls to 0.56 and its numerical `<think>` rate 0.96 → 0.84;
the β = 0.035 arm stays near where it started and is better at every one of the six
evaluations. The high-β arm's KL also falls 0.10 → 0.045 in 60 steps, retracing the original
run's 0.10 → 0.042 almost exactly (panel c). So the post-step-100 regression is reproduced by
changing β alone: the controller was undoing the run. (One seed per arm; both arms share the fresh-optimizer restart,
which is why even the low-β arm dips slightly at first. The gap between the arms is the β
effect.)

**What v4 looks like.** Same SFT start, same reward, same learning rate, 300 steps. The
standard four-panel view of the run, now with the held-out objective drawn over the
training reward in panel (a):

<p align="center"><img src="assets/grpo_training_dynamics.png" width="92%"></p>


| steps | training reward | held-out objective | KL | stop rate | zero-spread groups |
|---|---|---|---|---|---|
| 0 | — | −0.646 | 0 | 0.41 (held-out) | — |
| 1–25 | −0.393 | −0.391 (step 10), −0.268 (20) | 0.013 | 0.75 | 0.10 |
| 26–100 | −0.229 | −0.229 (mean of steps 30–100) | 0.028 → 0.037 | 0.90 | 0.16 |
| 101–200 | −0.183 | −0.200 (steps 110–200) | 0.040 → 0.051 | 0.92 | 0.17 |
| 201–300 | −0.179 | −0.184 (steps 210–300) | 0.055 → 0.057 | 0.93 | 0.20 |

Rapid rise, slow rise, plateau — in the training reward, the held-out objective, and KL.
Step-to-step SD of the training reward around its trend is 0.05 (v3: 0.30). Smoothed entropy
stays within 2.34–2.49 nats (no collapse); 25-step mean completion length 250–285 tokens.

The plateau has a visible cause. Format and termination saturate by step ~25 (held-out
`<think>` rate 0.57 → 0.96 by step 20), and their within-group spread collapses (within-group std
of `r_format` ~0.2 → 0.02): there is nothing left to separate siblings on. Panel (f) shows the
share of zero-spread groups — groups that produce no gradient at all — rising from 0 to
~20%. The slow phase after step 100 is the concept-coverage component (held-out conceptual
reward −0.29 → −0.03), while held-out numerical accuracy stays where it started (7.3% at
step 0, 4.7–10.7% throughout). **A stable optimiser exposes the ceiling more clearly; it
does not raise it.** The plateau level is set by what this reward can still distinguish on
a 113M model, not by the trainer.

**Decoder check.** A cleaner curve is worth nothing if the model got worse.
(These numbers use the old numerical labels. With corrected labels the verdict is the same: v4 − v3 = +2.0 pp [−0.7, +4.9], audit §15.6, `docs/eval_rl_v2labels_k4.json`.)
Scored with `src.rl.eval_rl` under the prose models' pinned decoder (temp 0.3 / top-k 40 / rp 1.2) on the
same 186 prompts × 4 samples as `docs/eval_rl_paired_k4.json`, paired by prompt
(`docs/eval_rl_v4_k4.json`):

| metric (166 numerical / 186 total prompts × 4) | SFT | v3 @100 (shipped, hand-selected) | **v4 @300 (final step, no selection)** | v4 − v3@100, 95% CI |
|---|---|---|---|---|
| numerical accuracy @1% | 5.4% | 5.4% | 6.8% | +1.4 pp [−0.6, +3.3] n.s. |
| numerical accuracy @5% | 7.1% | 8.9% | 9.8% | +0.9 pp [−1.4, +3.3] n.s. |
| `<think>` rate (numerical) | 70.2% | 97.9% | 98.6% | +0.8 pp [−0.8, +2.3] n.s. |
| pass@4 @5% | 16.9% | 21.1% | 23.5% | — |
| echo / contamination | 0.0% | 0.0% | 0.0% | — |
| mean completion tokens | 315 | 252 | 216 | −36 [−43, −28] significant |

The v3@100 row was re-scored and matches the shipped per-item results on 186/186 prompts,
so the pairing is exact. v4's final checkpoint is statistically indistinguishable from v3's
hand-selected best on every capability and format metric, and 36 tokens shorter. That is the
practical payoff: v3 needed a held-out selection step to rescue its step-100 checkpoint from
the regression that followed; v4 has no regression to rescue it from, so the last checkpoint
is a safe one. Against SFT, v4's pooled acc@5% gain is +2.7 pp [+0.3, +5.0], but it comes from
the dev half (3.0% → 7.9%); on the frozen test half it is +0.8% [−2.5%, +4.2%], identical to v3
(§3.1). Re-running the §6.4 tiering on v4 gives the same split as v3: retrievable +11.8%,
must-compute +0.4% (n.s.). Closed-book computation did not move.

```bash
torchrun --standalone --nproc_per_node=2 -m src.rl.train_grpo --config configs/rl_grpo.yaml     # v4, ~80 min on 2x A30
torchrun --standalone --nproc_per_node=2 -m src.rl.train_grpo --config configs/rl_grpo_v3.yaml  # the v3 algorithm
python -m src.rl.train_grpo --config configs/rl_grpo_intervention_beta0192.yaml               # β-only restart, one GPU
python -m src.plots grpo
```

### 5.9 Where useful gradient actually came from

Batches in this run are single-type, so a step exercises either the numerical verifier or
the concept verifier, never both:

| step type | steps | what the reward asks for |
|---|---|---|
| numerical | 77 / 200 (38.5%) | do the physics, then the arithmetic |
| conceptual | 123 / 200 (61.5%) | cover the domain concepts |

The run visited 400 prompts — 7.7% of the pool, not one epoch. So correctness of
computation was on the table in well under half the updates, while format was scoreable in
every one of them. That difference explains which behaviour moved.

> The statistic I actually wanted, and did not log. Eligibility is a weaker proxy than
> what matters. Only within-group spread creates gradient ([§5.5](#55-what-one-real-group-teaches)),
> so the right measure is the reward-opportunity rate: the fraction of groups in which a
> verifier produced discriminative advantage. The trainer logs the batch mean of each
> component, not its within-group standard deviation, so that rate is **not recoverable from
> these run artifacts** — the table above is step eligibility, which is an upper bound on it.
> Logging per-component within-group std costs one number per step and belongs in any
> verifier-RL run; it is the first thing I would add before the next one.
>
> **Now logged (v4).** `checkpoints/runs/grpo_v4.csv` carries `zero_var_frac` (groups with no
> reward spread, hence no gradient) and the within-group std of `r_num`, `r_concept`,
> `r_format` and `r_term`, every step, over the whole batch. Over steps 26–300: 16–20% of
> groups are zero-spread; `r_format` spread collapses to ~0.02 once `<think>` compliance
> saturates; `r_concept` keeps ~0.15 and `r_num` ~0.09. The concept verifier is where the
> remaining gradient comes from, which matches where the held-out objective still moves
> (§5.8). v4 batches are also type-stratified, so both verifiers are exercised every step.

### 5.10 Baselines: prompting and rejection sampling

Explicitly instructing the SFT model to use `<think>` tags moves compliance from 72.3% to
73.5%; GRPO moved it to 98.2% (`docs/prompted_baseline.json`, 166-question set). Prompting
barely changes the behaviour RL changes almost completely.

Rejection-sampling fine-tuning (RFT) keeps the student's own verified answers and fine-tunes
on them, with no policy gradient. In the transfer study (§5.19) it lifts the 576 practised
problems from 48.8% to 68.3%; GRPO on the same problems reaches 85.4%, and SFT on the
teacher's solutions of the same problems stays at 47.3%. On the held-out questions GRPO and
RFT stay level, +0.1 and +0.2 points (`docs/rl_transfer_diagnosis.json`).

### 5.11 Where this sits in the literature

- **InstructGPT** ([Ouyang et al. 2022](https://arxiv.org/abs/2203.02155)) — demonstrations
  → human rankings → learned reward model → RLHF. This pipeline replaces the rankings and
  the reward model with programmatic verifiers: no annotation cost and no reward-model
  overfitting, but verifier-specification risk instead.
- **DeepSeekMath / GRPO** ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)) — a 7B
  model, ~120B math-related pretraining tokens, mathematical SFT, then canonical GRPO.
  Group-relative RL improved its mathematical benchmarks *from a base that already had
  substantial mathematical capability*. §5.4 lists which pieces of it this
  implementation keeps.
- **DPO** ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) — offline,
  preference-pair classification; avoids online sampling entirely, but is not a drop-in
  substitute unless verifier scores are converted to chosen/rejected pairs.
- **Rejection-sampling fine-tuning** ([Yuan et al. 2023](https://arxiv.org/abs/2308.01825))
  — keep the verifier-passing samples and fine-tune on them, no policy gradient. Run here as
  a baseline (§5.10): it reaches 68.3% on the practised problems, against 85.4% for GRPO.
- **DeepSeek-R1** ([DeepSeek-AI 2025](https://arxiv.org/abs/2501.12948)) — the strongest
  demonstration that RL alone can incentivize reasoning behaviour. Worth reading against
  this repo's result: there format and reasoning moved together on broad benchmarks; here the
  format moved everywhere, and accuracy moved mainly on problems close to the practised ones.
- Supporting: pass@k, formalized with its unbiased estimator by
  [Chen et al. 2021](https://arxiv.org/abs/2107.03374) (who attribute the metric to
  [Kulal et al. 2019](https://arxiv.org/abs/1906.04908)) — note this repo reports the plain
  "any of k samples passed" form with n = k, not that estimator; the k3 KL estimator
  ([Schulman 2020](http://joschu.net/blog/kl-approx.html)); and Chinchilla
  ([Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)), whose 70B model on 1.4T
  tokens set the ~20 tokens-per-parameter rule of thumb.

DeepSeekMath shows group-relative RL improving math when the base model can already do math.
This project is a smaller regime: 113M parameters and a sparse correctness reward. Here GRPO fixed
the format (71% → 98%) and raised accuracy on the problems it practised (49% → 85%). On held-out
questions from papers it never practised on it adds up to 3 points (+0.1 to +3.3), because a gain
reaches a new question in proportion to its similarity to the practised problems (§5.19).

### 5.12 Overtraining: what to watch

- **SFT.** Validation loss on the teacher's exact wording rises after a few epochs while
  behaviour keeps improving. Select checkpoints on generated behaviour, never on loss (§4.7).
- **RL.** The three warning signs are:
  - KL running away from the reference;
  - entropy collapsing onto one phrasing;
  - completions growing to dilute per-token penalties.

  §5.8 shows the curves for each. The repair is a fixed KL coefficient, a length cap, and logging
  the share of zero-variance groups (they give no gradient).
- **Transfer.** Accuracy on the RL pool rising while held-out accuracy stays flat means the pool is
  too far from the held-out questions, not that RL is broken (§5.19).

### 5.13 Reward, advantage, and evaluation: why they are kept apart

Three numbers in this README can look alike but do different jobs:

```
training:  rollout ── verifier ──▶ reward ── minus group mean ──▶ advantage ──▶ gradient
decisions: held-out question ── deployment decoder ──▶ accuracy, format, hygiene ──▶ checkpoint, reported result
```

| | reward | advantage | evaluation |
|---|---|---|---|
| what it is | the verifier's score for one completion (§5.6) | the reward minus its group's mean; from v4 on scaled by one batch-wide std (§5.2) | accuracy and hygiene on held-out questions, measured outside training |
| what it answers | how the verifier rates this answer | which answers beat their siblings | did the model get better at the task? |
| where it is used | turned into advantages | multiplies the gradient of every token in the completion | picks checkpoints (dev split) and reports results (held-out set) |
| reaches the gradient | through the advantage | directly | never |

**Why the evaluation must stay separate.** RL maximises the reward, so the training reward rises by
construction: it is the objective, not evidence of progress. The reward is also a proxy for the task
(§5.6), and wherever proxy and task come apart the optimiser collects reward without getting better,
which is reward hacking. The advantage says even less about quality: it sums to zero in every group,
so it only ranks siblings. Therefore, no decision here uses the reward or the advantage; decisions use
an evaluation the optimiser never sees. The first version of this pipeline scored SFT and RL with the
training reward engine and reported "+590%". 73% of that gain was `<think>`-tag compliance, and held-out
accuracy did not move significantly ([PITFALLS B1](PITFALLS.md)).

**How the reward is built against hacking.**
- Every component's cheapest shortcut is named and paired with a standing probe (§5.6): empty
  `<think>` blocks, number spraying, keyword stuffing, padding.
- The reward is unit-tested against degenerate completions before training.
  `python -m src.rl.verifiers` runs eight adversarial probes, and
  `python -m src.analysis.validate_measurement hacking` scores seven degenerate completions and the
  genuine reference on every held-out item, flagging any degenerate answer that ties or beats the
  reference. It found one live exploit: padding plus a correct answer out-scored the genuine answer
  ([PITFALLS D2](PITFALLS.md)).
- Once the cheap terms saturate, only correctness is rewarded. The executed-equation runs score 1
  within 1%, 0.5 within 5%, and 0 otherwise or when the program fails, so no format or length term is
  left to farm.
- The interpreter writes the final answer. The model writes equations, `src/rl/pot.py` executes them
  and the decoder forces `Answer: <value>`; the forced tokens are masked out of the loss (§5.7), so the
  policy earns reward only through working that computes the value.
- The numerical verifier reads the last three numbers of the answer, so spraying candidates no longer
  pays (§3.3).
- The KL anchor to the SFT model limits how far the policy can drift while chasing a loophole.

**How the evaluation is built to catch what the reward misses.**
- Different questions: held-out questions from papers never used in fine-tuning, split by source
  chunk and hash-pinned (§3.1). The reward only ever sees the training pool.
- Different decoding: the deployment settings (temperature 0.3, top-k 40, vote@8 for the deployed
  model), not the training sampler.
- Separate metrics, never a composite: accuracy within 5% and 1%, `<think>` rate, echo,
  contamination and length are reported side by side (§3.2, §6.1). A composite that rises while
  accuracy stays flat is the signature of hacking.
- A known scale: each metric has a measured floor (a blind number spray, a model that ignores the
  question) and ceiling (the reference answers through the same parser), §3.3.
- Selection apart from reporting: checkpoints are chosen on a dev split by a rule fixed in code
  before results are seen, and the held-out set only reports (§3.1, §4.7).
- Statistics: 6 seeds × 4 samples, paired tests and a power line (§6).

Evaluation and reward share one definition of correctness, the reference value within a tolerance.
The evaluation applies it to different questions, with a different decoder and with statistics
([PITFALLS C2](PITFALLS.md)).

**What the separation caught.**

| case | the reward said | the evaluation said |
|---|---|---|
| v1 composite reward | +590% | 73% of the gain was format; accuracy flat (B1) |
| padding probe | filler + correct answer 0.56, genuine answer 0.42 | after the fix both score 0.77 (D2) |
| old scoring rule | the model scored 13.3% | a blind number spray scored 14.3% (C1) |
| Phase III (v4) | reward up; well-formed answers 71% → 98% | accuracy 4.7% → 6.1%: format learned, computation barely (§5.9) |
| transfer study | practised problems 49% → 85% | held-out questions +0.1 to +3.3: gains stay near practised problems (§5.19) |

---

### 5.14 A calculator fixes arithmetic, not setups

Small models often slip on arithmetic in long derivations, because every digit is sampled and one
wrong digit spoils the result. A calculator removes that error. The model writes `<<expr>>`, and
the decoder inserts ` = value` (`src/rl/rollout.py`; the expression is evaluated by an AST
whitelist, never `exec`). In RL the inserted tokens are observations, so they are masked out of the
loss.

I taught the habit with 6,000 computation drills (`tools/numeric_drills.py`) and with the SFT
corpus rewritten to call the calculator where the call reproduces the reference value
(`src/sft/calculator_format.py`). Then I ran GRPO with the calculator in the loop.

| | held-out drills | real questions | answers that call it | needs computation |
|---|---|---|---|---|
| SFT | 3.9% | 4.6% | 0% | 2.8% |
| + drills | 86.4% | 5.3% | 10% | 3.4% |
| + calculator-format corpus | 81.0% | 5.1% | 93% | 3.1% |
| + GRPO with the calculator | — | 5.4% | 91% | 2.9% |

The habit transferred: 93% of answers call the calculator, and answers with a wrong mental step
fell from 82% to 19%. Accuracy did not move, because the calculator executes whatever setup the
model writes, and the setups were wrong (a made-up formula, or `<<2.8-3.0 * 2.8>>` for "2.8 / 3").
The bottleneck had moved from arithmetic to choosing the relation and the operands.

Self-consistency helped a little. Sampling 8 answers and returning the one most others agree with
raised the GRPO model from 6.4% to 9.2% (8 seeds pooled, `docs/eval_vote_pooled8.json`), mostly on
questions whose answer appears in the prompt (`src/rl/self_consistency.py`). The deployed model uses the same vote over executed answers.

### 5.15 Speed: where RL time goes

<p align="center"><img src="assets/grpo_speed_v4b_vs_v5.png" width="92%"></p>

At 113M parameters, RL time is mostly decoding. Generation took 82% of each step at 0.4% MFU,
because each token launches about 700 small kernels. Therefore, I made two changes:
- I captured the decode step as a CUDA graph (`rollout.GraphDecoder`, static KV cache). This cut
  generation from 11.0 s to 2.9 s per 128 rows per GPU. Each captured graph pins its own KV cache
  (about 1.6 GB at 128 rows × 768 tokens), so at most two are kept; with six, a pool of many prompt
  lengths ran out of memory.
- I reused each rollout batch for two optimizer steps, as TRL's GRPO trainer does: μ = 2, ratio
  clipped to [0.8, 1.28], truncated importance sampling (§5.16).

Evaluation uses the same batched decoder, so ~45 min per model became ~20 s.

| 300 updates on 2× A30 | wall time | per rollout batch | MFU per step |
|---|---|---|---|
| original trainer | 82 min | 13.4 s | 2.3% |
| + CUDA-graph decoding | 27.7 min | 4.7 s | 6.5% |
| + two updates per rollout batch | 17.4 min | 6.1 s | 8.3% |

The final trainer is 4.7× faster and reaches the same held-out accuracy (6.4%, 8 seeds). Its KL ends
higher (0.10 vs 0.06), so with two updates per batch a lower learning rate keeps the KL down.

### 5.16 Rollout engine, training engine, and the mismatch between them

RL computes the same policy twice. The **rollout engine** samples completions; the **training
engine** computes the log-probabilities that the gradient uses. Here both run the same weights in
the same process, with different arithmetic:

| | rollout engine (`src/rl/rollout.py`) | training engine (`src/rl/train_grpo.py`) |
|---|---|---|
| job | sample G completions per prompt | score every sampled token, take the gradient |
| computation | one token at a time, static KV cache, decode step captured as a CUDA graph | one forward pass over the whole sequence |
| precision | bf16 matmuls and KV cache, fp32 softmax | bf16 matmuls, fp32 log-softmax |
| sampling | temperature 1.0 (Phase III) or 0.8 (executed-equation runs), no top-k, no penalty | scores at temperature 1 |

The two engines therefore disagree slightly on each token's probability. The samples come from the
sampler's distribution $`\mu`$, while the gradient is computed for the trainer's policy $`\pi_\theta`$.
This gap is the **train–inference mismatch**.

**Why it matters.** Policy-gradient methods often assume on-policy samples. The gradient is an
expectation under the policy being trained:

```math
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\left[A(y)\,\nabla_\theta \log \pi_\theta(y)\right]
```

With samples from $`\mu`$ instead, the estimate is biased. Small token-level gaps add up, because a
sequence's log-probability is a sum over its tokens:

```math
\log \frac{\pi_\theta(y)}{\mu(y)} = \sum_t \left[\log \pi_\theta(y_t \mid y_{\lt t}) - \log \mu(y_t \mid y_{\lt t})\right]
```

The gap is also largest on unlikely tokens, which is where an importance weight explodes. Large-model
RL stacks run separate inference and training engines (for example vLLM and FSDP). There the
mismatch quietly turns on-policy training off-policy ([Yao et al. 2025](https://fengyao.notion.site/off-policy-rl)) and has collapsed
long runs ([Liu et al. 2025](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)).

**Measured here.** `scripts/experiments/mismatch_check.py` samples 32 completions (3.1k tokens) from
the Phase IV model (notation2) on CPU and re-scores the same tokens with the trainer's forward pass
(`docs/mismatch_check.json`). The training logs record the same gap on GPU:

| comparison | mean gap per token (nats) | largest | tokens with weight above 2 | spread of the sequence log-ratio |
|---|---|---|---|---|
| both engines fp32 | 1.0e-6 | 5e-5 | 0% | 0.00 |
| trainer bf16, sampler fp32 | 0.0055 | 0.22 | 0% | 0.19 |
| sampled at T = 0.8, scored at T = 1 | 0.033 | 1.57 | 0.6% | 1.06 |
| sampled and scored at T = 0.8 | 9e-7 | 4e-5 | 0% | 0.00 |
| training logs, both bf16, T = 1 (v5–v8) | 0.014–0.021 | — | at most 0.002% | — |
| training logs, both bf16, T = 0.8 (v9 on) | 0.015–0.055 | — | 0.15–0.8% | — |

The results show three things:
- in matched precision the two engines agree to a millionth of a nat, so the cached decoder itself
  is exact;
- bf16 rounding alone moves a token's log-probability by 0.0055 on average, and a 100-token
  completion's probability by about ±20%. The gap is 0.041 on tokens with probability under 0.1 and
  0.0004 on tokens above 0.9;
- the deliberate temperature gap is six times larger than the rounding gap.

**Mitigations used here.** Therefore, the trainer combines three defences.

*Same precision and the same distribution.* Both engines use bf16 matmuls and an fp32 softmax, and
Phase III samples the plain policy (temperature 1, no top-k, no penalty), so the sampler draws from
the distribution the trainer scores. Scoring at the sampler's temperature removes the temperature gap
entirely (fourth row). FP16 keeps 10 mantissa bits to BF16's 7, so it rounds 8× more finely;
switching RL training and inference to FP16 removes most of the mismatch, at the cost of loss
scaling ([Qi et al. 2025](https://arxiv.org/abs/2510.26788)).

*Truncated importance sampling (TIS).* The sampler returns the log-probability of every token it
draws, and the trainer weights each token's gradient by

```math
w_t = \min\left(\frac{\pi_\theta(y_t \mid y_{\lt t})}{\mu(y_t \mid y_{\lt t})},\ C\right), \qquad C = 2
```

An uncapped weight makes the estimate unbiased for $`\pi_\theta`$, but one rare token could then
dominate a batch; the cap trades a little bias for bounded variance. The weight is per token because
the sequence ratio varies too much (last column above). TIS is separate from the PPO ratio. When a
batch is reused (v5–v8), $`r_t = \pi_\theta / \pi_{\text{old}}`$ compares the policy with the
trainer's own first-pass log-probs and handles staleness; $`w_t`$ handles the engine gap. Every run
since v5 uses $`C = 2`$. At temperature 1 the weights averaged 1.000 and no step had more than about 2 tokens in
100,000 at the cap, so TIS is a guard there. At temperature 0.8 it reweights the samples
toward the temperature-1 policy, and 0.15–0.8% of tokens are capped.

*Score centering.* A constant reward is harmless only because
$`\mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(y)] = 0`$, and that identity holds only under
$`\pi_\theta`$. Under the sampler,

```math
\mathbb{E}_{y \sim \mu}\left[b\,\nabla_\theta \log \pi_\theta(y)\right] = -\,b\,\nabla_\theta\,\mathrm{KL}(\mu \,\Vert\, \pi_\theta)
```

so a reward offset $`b`$ pulls the policy toward the sampler's distribution when it is positive, like
SFT on the samples, and pushes it away when it is negative, as v3's mostly negative composite reward
would have. The group-mean baseline removes $`b`$ from every group exactly: the advantages sum to
zero, and only the differences between completions train the model.

Tokens the policy did not sample, such as padding, forced EOS and tool output, have no sampler
probability at all and are masked out of the loss (§5.7).

```bash
python scripts/experiments/mismatch_check.py     # CPU, about 1 min; writes docs/mismatch_check.json
```

### 5.17 Finding the bottleneck

<p align="center"><img src="assets/numerical_diagnosis.png" width="98%"></p>

More training on the same data often looks like the fix for low accuracy. Before adding data, I
measured each step of producing a numerical answer separately, on the 134 questions that need
computation. For reference, the reference answers themselves score 85.8% under this metric, and a
model that ignores the question scores 3.3%.

| step | test | result |
|---|---|---|
| derive the value | does the right value appear anywhere in the answer? | 14.7%, close to chance (13.1%) |
| restate a finished derivation | give the model the full reference reasoning | 39% (53% without the repetition penalty) |
| memorise | questions it was trained on | 5.1%, the same as held-out |
| arithmetic | calculator in the loop (§5.14) | wrong steps 82% → 19%, accuracy unchanged |

The model rarely derives the value at all, even for questions it trained on.

Binding numbers to a relation can be learned close to training:
- with new numbers in trained questions, accuracy goes from 3% to 18% (`src/sft/counterfactual.py`);
- on unseen phrasings of generated word problems, it goes from 2% to 68% (`tools/word_problems.py`).

It does not carry over to new question types. At this stage the bottleneck was the number of distinct question
types: about 1,250 in the corpus, where TinyGSM used about 12M synthetic problems for a 125M model.
Therefore, I generated many more problems, each with an executable solution (§5.18).

### 5.18 Phase IV — executed equations from a local teacher

Small models do well on math word problems when they are trained on very many problems with
executable solutions (PAL, TinyGSM). Both the problems and the solutions were missing here, and a
local teacher can write both. `Qwen/Qwen3.8-27B-FP8` (Apache-2.0) runs under vLLM on the two A30s
at about 900 tokens/s, and scores 49.7% on the held-out questions when it solves them blind.

Therefore, I had the teacher do three things (`tools/gen_pot_problems.py`):
- write exercises seeded by training questions;
- rewrite every real numerical training question in new scenarios;
- turn real questions into executable solutions.

Every solution is executed and kept only if it reproduces its label. Real-style questions are also
re-instantiated with new numbers, without the teacher (`tools/pot_counterfactuals.py`).

The student writes its working as a short program. `src/rl/pot.py` executes it at `</think>`, and
the decoder writes `Answer: <value>`.

| data | written | kept |
|---|---|---|
| exercises seeded by training questions | 34.8k | 24.4k |
| real questions as executable solutions | 2.3k | 1.8k |
| the same with new numbers | 11.8k | 9.1k |
| rewrites of real questions (2 rounds × 6) | 38.8k | 22.4k |
| rewrites with new numbers | 58.9k | 47.2k |

<p align="center"><img src="assets/numerical_track.png" width="92%"></p>

| milestone | needs computation | all | vote@8 |
|---|---|---|---|
| SFT, prose answers | 2.8% | 4.7% | 8.5% |
| executed Python solutions | 6.2% | 6.8% | 9.8% |
| + real-style rewrites, no repetition penalty | 11.1% | 11.3% | 14.4% |
| paper-notation solutions | 17.3% | 17.8% | 19.2% |
| + recovered data | 20.1% | 19.7% | 22.2% |
| + GRPO on the corpus (deployed, §5.19) | 20.7% | 22.0% | 23.3% |
| 27B teacher, blind | 48.5% | 49.7% | — |

The results show three effects:
- **Real style transfers; volume alone does not.** Teacher exercises raise accuracy on more
  exercises of the same kind (4% at 3k, 21% at 30k) but barely move the real questions. Rewrites in
  the multi-part style of the real questions doubled the must-compute score.
- **Programs need decoding without a repetition penalty.** The penalty punishes reusing variable
  names, which a program must do. Removing it raised the share of programs that run from 40% to 57%.
- **The notation matters most.** The base model was pretrained on papers, not code, and 45.5% of
  its Python programs crashed, mostly on misspelt names. The same solutions written as paper
  equations (`M0 = 10^(1.5*Mw + 9.1) [N m]`) give 17.8% instead of 11.3%, with 29% less data. The
  working is shorter, runs more often (75% vs 58%) and is right more often when it runs (23% vs 19%).

GRPO on the executed answers then adds up to 3 points: +3.3 on the first paper-notation model at
its best checkpoint, +1.3 on notation2 with problems from the training papers, and +2.4 with problems
from every paper in the corpus, which is the deployed model (checkpoints chosen on a dev split). §5.19 explains why the gain is small on held-out questions. The full data
build merges several notation files and 6k replayed Phase II pairs (audit §15.14–15.15):

```bash
python tools/gen_pot_problems.py generate --model Qwen/Qwen3.8-27B-FP8 --tp 2 --n-seeds 10000 --per-seed 4 \
    --seed-pool all --out data/sources/pot/gen.jsonl            # also: rewrite, convert, solve, audit
python tools/gen_pot_problems.py notation --inp data/sources/pot/notation_inputs.jsonl \
    --out data/sources/pot/notation_all.jsonl                    # the same solutions in paper notation
python tools/pot_counterfactuals.py data/sources/pot/notation2_convert.jsonl \
    data/sources/pot/notation2_convert_cf.jsonl --variants 6 --notation
python tools/format_pot_sft.py --notation --require-verified --out data/sources/pot/sft_notation.jsonl \
    data/sources/pot/notation_all.jsonl data/sources/pot/notation2_convert_cf.jsonl
torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_pot_notation2.yaml --snapshot-epochs
```

### 5.19 Where GRPO helps: a transfer study

Small models often gain a lot from GRPO on arithmetic or GSM8K-style problems. Here GRPO added up to
3 points on held-out questions. This section finds where its gains go (details:
[audit §15.17](audit_report_sft_grpo.md)).

**How GRPO learns.**
- For each question, GRPO samples a group of 8 answers and scores them with the verifier: 1 within
  1% of the label, 0.5 within 5%, otherwise 0.
- It subtracts the group's mean score. Answers above the mean become more likely; answers below it
  become less likely.
- A KL penalty keeps the policy near the SFT model.

Two consequences matter here:
- A question the model always solves, or never solves, gives every answer the same score and
  therefore no gradient. So the RL pools keep only questions solved 1–7 times out of 8.
- RL mostly turns "solved sometimes" into "solved reliably", that is pass@k into pass@1
  ([Yue et al. 2025](https://arxiv.org/abs/2504.13837)).

**The test.** I trained GRPO on 576 teacher problems the model had never seen (set A). Then I
measured the gain on test sets at known distances from them:
- never-solved problems from the same papers;
- 600 problems from 92 excerpts of other papers (B600);
- the real held-out questions.

The similarity of two questions is the share of content words they have in common (word-Jaccard).

<p align="center"><img src="assets/rl_transfer_diagnosis.png" width="96%"></p>

**RL learns its pool.** On the 576 trained problems, accuracy goes from 48.8% to 85.4% in 75 steps.
Every kind of error falls by about three quarters:
- crashes: 9.3% → 2.3%;
- unit errors: 9.0% → 2.3%;
- wrong setups: 19.3% → 5.4%.

**Gains travel by similarity.** On problems the run never trained on, in percentage points:

| similarity to the nearest trained question | same papers, never solved | other papers |
|---|---|---|
| below 0.2 | +1.0 | −1.3 |
| 0.2–0.35 | +5.0 | +0.3 |
| 0.35–0.5 | +9.9 | +9.7 |
| 0.5 and above | +17.1 | +27.2 (17 problems) |

The pattern is the same within a paper and across papers. A gradient step on one question changes
the model's answer to another question in proportion to how similar the two look to the network
(the neural tangent kernel picture; [Jacot et al. 2018](https://arxiv.org/abs/1806.07572)). Word
overlap is a rough proxy for that similarity, and it predicts the gains well.

A second run confirms it. GRPO on 230 other problems from B600's own papers lifts B600 by +10.1
points (23.9% → 33.9%), against +1.6 for GRPO on set A.

**What carries over is the program, not the setup.** Accuracy is the chance that the program runs
times the chance that it is right when it runs.
- On trained problems both rise: 70% → 86%, and 50% → 77%.
- On other papers only the first rises, a little (60% → 64%); the second stays at 27%.

Choosing the right relation and constants for a topic is knowledge about that topic, and RL
sharpens it only where it practises. Standard constants are the exception: on other papers,
problems that need an unstated constant gain +4.1 points.

**Why the held-out questions move little.**
- The small-model GRPO results test on problems from the same generator as the training set, so
  every test question has a close neighbour in training.
- The held-out questions here come from papers the model never practised on. Their nearest neighbours in the RL pools are
  far: median similarity 0.23–0.27, and only 20–29% above 0.35.
- That predicts a held-out gain of a few points, which is what was measured.

The same is known for small transformers and facts: fine-tuning on questions about some facts does
not carry over to other facts unless pretraining showed each fact in several phrasings
([Allen-Zhu & Li, *Physics of Language Models* 3.1](https://arxiv.org/abs/2309.14316)).

#### Simpler questions as stepping stones

A question the model never solves gives no gradient, and about 60% of realistic questions are like
that. Therefore, I had the teacher turn each problem into a short question about one early quantity
of its solution (1–2 steps; the label comes from executing the original solution). Then I ran GRPO
on the learnable ones.

<p align="center"><img src="assets/rl_curriculum.png" width="88%"></p>

| | original questions | one-step versions |
|---|---|---|
| never solved | 63% | 52% |
| accuracy on other papers, before RL | 23.9% | 34.5% |
| GRPO on the pool | 48.8% → 85.4% | 47.0% → 90.3% |
| gain on other papers | +1.6 | +3.0 |

The simple questions are easier, and RL masters them. On their own, though, they do not lift the
original multi-part questions from other papers. Their value shows on the hardest problems:
- never-solved problems whose own first step was practised: 1.5% → 10.8%;
- the other never-solved problems: 1.8% → 4.8%.

A first-step question turns a problem that gives no gradient into one that does.

#### Reading does not replace practice

If setup knowledge stays local, perhaps the model can learn it by reading. The teacher restated each
source excerpt of B and A in 8 styles, with no questions. I then continued training the Phase IV
model (notation2) on 3,000 replayed pairs plus either the notes about B or, as a control, the notes about A.

| | notes about B | notes about A |
|---|---|---|
| B questions (start 14.3%) | 14.5% | 14.6% |
| A questions (start 18.1%) | 16.8% | 15.5% |

Reading about a paper did not change accuracy on questions about it, while practice did (+10.1
points from RL on the same papers). At this scale, the model learns a paper's setups by solving
problems about it in the format it answers in.

#### Breadth without repetition

The similarity law says a held-out question gains when a close neighbour is practised. Therefore, I
built two broad pools of learnable problems and ran GRPO over each once:
- **strict pool (v13):** 9,031 problems, all derived from the fine-tuning papers: a third teacher
  rewrite round, exercises on every training topic, stepping stones for never-solved problems, and
  the A and simple-A sets;
- **corpus pool (v14):** the same pool plus 952 problems written from the held-out papers' excerpts
  (650 exercises and 302 stepping stones), filtered against the held-out questions (no shared
  13-gram, word-Jaccard below 0.5). The model practises on every paper in its corpus, never on the
  test questions.

These pools covered the held-out questions better than any earlier pool (median nearest-neighbour
similarity 0.28–0.30, 34–37% above 0.35), and the similarity law predicted about +5.5 points if every
neighbour were learned (`docs/eval_v13_v14.json`):

| model (checkpoint) | held-out accuracy | vs base, 95% CI | needs computation | answer in prompt | vote@8 |
|---|---|---|---|---|---|
| notation2, the Phase IV base | 19.7% | — | 20.1% | 16.7% | 22.2% |
| strict pool, dev-selected (step 25) | 20.1% | +0.4 [−2.5, +3.2] | 18.8% | 29.2% | 21.1% |
| strict pool, end of the pass (step 142) | 21.0% | +1.3 [−2.6, +5.4] | 19.7% | 30.0% | 21.6% |
| corpus pool, dev-selected (step 75), deployed | 22.0% | +2.4 [−1.5, +6.3] | 20.7% | 31.1% | 23.3% |
| corpus pool, end of the pass (step 156) | 22.0% | +2.4 [−1.6, +6.4] | 21.3% | 27.2% | 23.7% |

The results show:
- the corpus pool scores higher than the strict pool (+2.4 against +0.4 to +1.3 points); no gain is
  significant on 153 questions. Its dev-selected checkpoint, 23.3% with vote@8, is now the deployed
  model. The end-of-pass checkpoint scores 23.7%, but picking it on the held-out set would be
  selection on the test ([PITFALLS E2](PITFALLS.md));
- the gain sits on the 19 questions whose answer appears in the prompt (+10 to +15 points,
  significant); on the 134 that need computation, accuracy moves by −1.3 to +1.2 points;
- the measured gain is a fraction of the prediction;
- dev accuracy, mostly the B600 exercises, fell by 2–3 points over the pass while the held-out
  questions did not, so dev selection picked early checkpoints.

The difference from the transfer study is repetition. There, each of 576 problems was practised
about 8 times and rose from 49% to 85%. Here each problem was practised once, and even fresh problems
from the pool's own distribution, the closest neighbours of what was practised, rose only from 43% to
48% over the pass. With so little learned on the pool itself, little could reach the held-out
questions. At this scale GRPO turns repeated practice on a problem into skill on it and its near
neighbours; a single pass over many problems mostly does not.

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
15× the 0.0096-nat effect being measured. [PITFALLS B2](PITFALLS.md).

### 6.3 Multiple samples, pass@k, and power

```
[power] at n=153, k=4 sample(s)/prompt, baseline 4.7%: MDE ≈ 9.3% at 80% power.
        A null result below that threshold means UNDERPOWERED, not 'no effect'.
```

With 153 numerical questions, one comparison reliably detects a change of about 9 points from
the SFT baseline (4.7%) and about 14 points from the Phase IV baseline (19.7%). Phase IV's +15
points clears that bar; GRPO's gain of up to 3 points on top of it does not, which is why §5.19
measures RL on larger problem sets.

pass@k separates "can sometimes do it" from "does it reliably": with four attempts the Phase IV
model solves 26.3% of the questions at least once, against 19.7% for one attempt, and vote@8
turns part of that gap into a single answer (22.2%).

### 6.4 Tiering: why one aggregate is not enough

A single number cannot distinguish "competent but unreliable" from "scoring by accident",
and those imply opposite next steps. `diagnose_capability tier` splits the numerical set by
whether the answer must be derived or is already present in the prompt, and again by
derivation depth — each tier reporting its own chance floor, because target clustering
differs by tier and one global floor would flatter some tiers and penalise others.
The results are in [§1.2](#12-results-by-phase).

---

## 7. What it found

- **Format.** GRPO raises well-formed answers from 71% to 98% (§5.8).
- **Numerical accuracy.** It rises from 4.7% to 22.2% on held-out questions once the model writes
  equations that an interpreter executes (§5.18), and to 23.3% after GRPO on the corpus, the deployed
  model (§5.19). On questions that need computation it rises from 2.8% to 22.4% (21.6% for the
  deployed model).
- **Reach of RL.** Gains are large on practised problems and fall off with the distance from them
  (§5.19). Practising first steps unlocks problems that were never solved. A single pass over 9,000
  new problems adds at most 2.4 points: the gains come with repeated practice.
- **Not measured yet:**
  - the factual accuracy of conceptual answers (641 of the 808 held-out prompts);
  - multi-turn behaviour and calibration.

---

## 8. How this recipe scales

Report portable units, not epochs and steps. "300 RL steps" does not transfer to another
model or batch size; "76.8k completions and 20.7M generated tokens" does.

| axis | measured here | what to add |
|---|---|---|
| instruction data | 14.1k Phase II pairs (composition-matched curve, §4.9); 111k Phase IV rows | equal-compute comparison, multiple seeds |
| model size | one 113M model | no model-size law was measured; 0.5B/1B would test it |
| RL compute | Phase III (v4): 300 steps · 76.8k completions · 20.7M tokens · 2.7 GPU-h | broader problem pools before more steps (§5.19) |
| systems | 2× A30 | single-GPU recipe in [§2](#2-quick-start); sharding or LoRA beyond ~1B |
| evaluation | 153 numerical × 6 seeds × 4 samples; MDE ≈ 9 pts at 4.7%, 14 pts at 19.7% | ~1,100 questions for 5 pts, ~6,400 for 2 pts at 19.7% (unpaired) |

What limited each phase, in order: the answer format (fixed by GRPO, §5.8), the arithmetic
(fixed by executed equations, §5.18), and topic coverage, since RL gains stay near the practised
problems (§5.19). Only one model size was run, so model capacity was not tested.

---

## 9. Failure atlas & pipeline pitfalls

A comprehensive taxonomy of post-training pipeline pitfalls, diagnostic tests, and regression fixes—including data leakage prevention, evaluation decoupling, reward hacking mitigation, and checkpoint selection criteria—is documented in [PITFALLS.md](PITFALLS.md).

---

## 10. Repository layout

```
data/
  sources/     paper.jsonl  paper_raw.jsonl  paper_rewritten.jsonl
               fundamentals.jsonl  earthquakes.jsonl      ← corpus inputs
  splits/      train.jsonl  val.jsonl  test.jsonl         ← identity-level, hash-pinned
               train_pot_notation2.jsonl                  ← Phase IV: 105k executed solutions + 6k replayed pairs (git-ignored)
  rl/          train.jsonl  val.jsonl  test.jsonl         ← GRPO prompts (RL schema)
  eval/        heldout_subsets.jsonl                      ← 808 recs, `subsets` tag
               scaling_2k/4k/8k.jsonl                     ← composition-matched series
configs/deploy.yaml                                       ← the DEPLOYED model and its decoder: GRPO v14 step 75,
                                                            executed paper-notation equations, vote@8 (§1.2)
configs/     sft_*.yaml · rl_grpo_*.yaml                  ← one config per run
checkpoints/
  grpo_v14_corpus/ grpo_ckpt_step_75.pt                   ← the deployed checkpoint (named in configs/deploy.yaml)
  sft_pot_notation2/ sft_ckpt_epoch4.pt                    ← Phase IV: the executed-equation student
  sft/         selected.pt  sft_ckpt_best.pt              ← Phase II selection (prose SFT)
  grpo/        selected.pt → ../grpo_v4/grpo_ckpt_step_300.pt  ← Phase III selection (v4, prose;
               chosen by the src.sft.checkpoints rule on the dev half: docs/checkpoint_selection_rl_v4.json)
               grpo_ckpt_step_{100,200}.pt                 ← the v3 run (step 100 was the previous selection)
  grpo_v4/     grpo_ckpt_step_{50,100,…,300}.pt            ← v4 trainer run, README §5.8
  grpo_v4b/ grpo_v5/                                       ← corrected labels; + μ=2 / TIS (§5.15)
  sft_drills/ sft_calc/ grpo_v6_calc/                      ← calculator experiments (§5.14)
  sft_pot_notation/ grpo_v11_notation/ grpo_v12_notation/  ← paper notation, + GRPO (§5.18)
  grpo_transfer_{lo,hi,Bsame}/ grpo_simpleA/ transfer_{sft,rft}A/  ← transfer study (§5.19)
  grpo_v13_broad/ grpo_v14_corpus/                         ← one pass over the broad pools (§5.19)
  runs/        sft.csv  grpo.csv  + 60 more logs          ← every training curve
               grpo_v4.csv  grpo_v4_eval.csv               ← v4 batch + held-out curves
               grpo_v3_intervention_beta{0192,0035}{,_eval}.csv  ← β-only restart test
               grpo_v4b · grpo_v5 · grpo_v6_calc {,_eval}.csv
src/
  sft/         generate_sft_data → build_corpus (clean · merge · split)
               dataset · train_sft · chat · rag_retriever · run_data_scaling_sweep
               checkpoints.py       ← select · compare · strip
               splits.py            ← the train/val/dev/test contract
  rl/          prepare_rl_dataset · numerical_target (label rule) · verifiers (+ safe_eval)
               rollout (batched CUDA-graph decoder; calculator and executed equations) · train_grpo · eval_rl
               pot (executes the paper-notation equations) · self_consistency (vote@k)
  sft/calculator_format.py  ← rewrites verified reference arithmetic as <<expr>> calls
  sft/counterfactual.py     ← re-instantiates real questions with new numbers (program extraction)
  analysis/    validate_measurement (floor · hacking · group)
               diagnose_capability  (tier · oracle · prompting)
               tier_vote            (acc@1 / vote@k by tier, from stored answers)
  plots/       shared style in __init__ + one entry point (python -m src.plots)
tools/gen_fundamentals/    portable stdlib-only data generators
tools/numeric_drills.py    calculator-format computation drills (15 templates)
tools/word_problems.py     lexically diverse word problems, 14 families, held-out phrasings
tools/gen_pot_problems.py  local-teacher (vLLM) exercises with executed program solutions
tools/pot_counterfactuals.py  real questions re-instantiated from their programs
tools/format_pot_sft.py    teacher solutions → SFT rows (`--notation` for paper notation)
tools/filter_learnable_prompts.py  keeps problems the student solves sometimes, not always
tools/build_rl_pool.py · gen_stepping_stones.py · gen_chunk_problems.py  ← RL pools (§5.19)
scripts/experiments/       transfer-study and evaluation scripts (real6_eval.py: the 6-seed held-out eval;
                           mismatch_check.py: sampler vs trainer log-probs, §5.16)
```

Nothing in `assets/` is drawn from numbers pasted into the plotting code: every figure
regenerates from a committed log or results file (`python -m src.plots`), which is what keeps
the numbers reproducible and auditable.

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
- **Data.** The Phase II splits (`data/splits/{train,val,test}.jsonl`), the RL prompt files
  (`data/rl/{train,val,test}.jsonl`), the evaluation subsets (`data/eval/`) and the source files
  (`data/sources/*.jsonl`) are committed. Other generated sets are git-ignored and rebuild from
  scripts: the Phase II corpus with `python -m src.sft.build_corpus` and any OpenAI-compatible
  teacher (`TEACHER_API_BASE` / `TEACHER_MODEL`); the Phase IV solutions
  (`data/splits/train_pot_notation2.jsonl`) with `tools/gen_pot_problems.py` and a local vLLM
  teacher (§5.18). Teacher-generated data inherits whatever terms your provider attaches to model
  output; the source text keeps the licences listed in the pretraining repo.
- **Model weights** are a derived artifact, distributed (if at all) via a GitHub Release or
  the Hugging Face Hub, not committed to git.
- **Training logs are committed** (`checkpoints/runs/*.csv`), because every figure in
  `assets/` regenerates from them — that is what makes the numbers here checkable.
