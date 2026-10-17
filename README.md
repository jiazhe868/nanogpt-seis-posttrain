<div align="center">

# nanoGPT-Seis-Posttrain

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
| III · Verifier RL | relative programmatic rewards | near-universal `<think>` compliance; +37 pp on the problems it practises on, spreading by similarity (§5.18) | gains on questions far from its pool |
| II+ · Executed-equation SFT | 105k teacher solutions in paper notation, run by an interpreter | closed-book numerical accuracy 4.7% → 19.7%, 22.2% with vote@8 (deployed, §1.4) | parity with the 27B teacher (49.7%) |
| IV · Deployment + RAG | BM25 retrieval over 1.0k paper chunks | open-book extraction & exact calculation ($R \approx 8.67 \text{ km}$) | parametric closed-book formula memorization |

### The central result

Asymmetric, and the asymmetry is the point:

- SFT makes the pretrained model usable as an assistant.
- RL raises `<think>` compliance from 73.1% → 97.8% on the audit test half.
- Prompt-value extraction improves modestly.
- Multi-step numerical computation, which first sat at its blind-policy floor in closed-book mode, now reaches 19.7% on the
  real held-out questions (22.2% with voting; must-compute 2.8% → 20.1%) once the model writes paper-notation equations
  that an interpreter executes ([§1.4](#14-the-numerical-accuracy-track-at-a-glance-october-2026), [§5.17](#517-program-of-thought-with-a-local-teacher-the-first-must-compute-gain)).
  RAG grounding remains the route to exact open-book calculation.
- GRPO lifts the problems it practises on from 49% to 85–88%, and its gains reach new questions in proportion to their
  similarity to the practised ones ([§5.18](#518-where-rl-helps-a-transfer-study)).

The lesson is not "GRPO worked" or "GRPO failed." It is that a post-training result is
inseparable from the split, decoder, reward contract, benchmark floor and ceiling,
statistical power, and checkpoint-selection rule used to measure it. Under a rigorous evaluation protocol, verifier RL decisively enforces format and response termination, while open-book RAG context grounding enables exact multi-step scientific calculations. In closed-book mode, executed paper-notation solutions make multi-step calculation work, and RL sharpens it wherever its training problems reach.

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
   - 5.14 [Calculator tool use and test-time compute — built and measured](#514-calculator-tool-use-and-test-time-compute--built-and-measured)
   - 5.15 [Speed: where RL time goes, and what TRL's trainer contributed](#515-speed-where-rl-time-goes-and-what-trls-trainer-contributed)
   - 5.16 [Why numerical accuracy is stuck: a stage-by-stage diagnosis](#516-why-numerical-accuracy-is-stuck-a-stage-by-stage-diagnosis)
   - 5.17 [Program of thought with a local teacher: the first must-compute gain](#517-program-of-thought-with-a-local-teacher-the-first-must-compute-gain)
   - 5.18 [Where RL helps: a transfer study](#518-where-rl-helps-a-transfer-study)
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
| RL run, stabilised trainer ([§5.8](#58-why-the-v3-reward-and-kl-curves-never-settled--and-the-v4-trainer)) | 300 steps · 76,800 completions · 20.7M generated tokens · 9,600 prompt draws (1.84 passes over 5,217) · 2.7 GPU-h incl. 31 held-out evals |
| RL run, fast trainer ([§5.15](#515-speed-where-rl-time-goes-and-what-trls-trainer-contributed)) | 150 rollout batches × 2 updates · 38,400 completions · 10.4M generated tokens · 17.4 min on 2× A30 (0.58 GPU-h) incl. 16 held-out evals · MFU 8.3% (v4: 2.3%) |
| Held-out pool | 808 prompts → dev 463 (selection) / test 345 (frozen, reported once) |

### 1.2 Base → SFT → RL

<p align="center"><img src="assets/grpo_benchmark_comparison.png" width="88%"></p>

> **Corrected labels.** On 2026-10-15 the numerical labels were rebuilt: about half of the old
> ones were a year, an age or a value from the question, picked up by a regex. The chart and table
> use the corrected labels (`docs/eval_rl_v2labels_k4.json`: 153 numerical + 20 conceptual
> held-out prompts × 4 samples, 95% bootstrap CIs over prompts). The original old-label
> measurements, and why they changed, are in [PITFALLS C3](PITFALLS.md) and audit §15.6.

| corrected labels (k=4) | population | SFT | GRPO v3 @100 | GRPO v4 @300 |
|---|---|---|---|---|
| numerical accuracy @5% | 153 numerical | 5.7% | 6.0% | 8.0% |
| numerical accuracy @5%, frozen test half | 84 numerical | 6.8% | 7.4% | 9.2% |
| numerical accuracy @1% | 153 numerical | 1.6% | 2.1% | 3.4% |
| `<think>` compliance | 153 numerical | 69.1% | 97.4% | 98.0% |
| answer already in the prompt (floor 6.3%) | 19 numerical | 23.7% | 26.3% | 34.2% |
| answer must be computed (floor 5.1%) | 134 numerical | 3.2% | 3.2% | 4.3% |

Paired, v4 − SFT:
- acc@5% +2.3% [−0.3%, +4.9%] (test half +2.4% [−0.9%, +6.0%]);
- must-compute +1.1% [−0.9%, +3.2%];
- `<think>` +28.9% [+24.3%, +33.3%].

The first RL stage changed format decisively. Closed-book computation needed a different
answer format, which the numerical track below supplies: 4.7% → 22.2% on the same questions
(§1.4).

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

### 1.4 The numerical-accuracy track at a glance (October 2026)

**Where it started.** After SFT and GRPO v4 the model answered in the right format (`<think>`,
then a final answer) but its numbers were rarely right: 6.1% on the 153 real held-out numerical
questions, and 3.1% on the 134 that must be computed. The question-blind floor for those is 3.3%.
Over the next days the track found out why, one measurement at a time, and turned it around:
**closed-book accuracy is now 19.7% (22.2% with voting), and 20.1% on must-compute (6.5×)**.
The steps are summarised here; §5.14–5.18 and [audit §15.5–15.17](audit_report_sft_grpo.md)
have the details.

**1. Measure honestly first** (§5.16, audit §15.6).
- About half of the original numerical labels were wrong (a year or an age picked up by a
  regex), so they were rebuilt and re-checked on a blind sample.
- Every score now comes with the right reference points: answer-in-prompt vs must-compute tiers,
  a question-blind floor, an instrument ceiling (86%) and a teacher ceiling (49.7%).
- Headline numbers are averaged over 6 seeds × 4 samples, with paired bootstrap CIs.

**2. Find the real bottleneck** (§5.14, §5.16).
- Handing the model a correct derivation and asking it only to finish showed that it could not
  reliably carry out the arithmetic.
- A calculator tool fixed the arithmetic, which exposed the next layer: the *setups* (which
  relation, which operands) were wrong.
- More varied training questions were needed, not more training on the same ones.

**3. Let the computer compute; let the model set up** (§5.17).
- The student now writes its working as a short list of equations; an interpreter
  (`src/rl/pot.py`) executes them, and the decoder writes `Answer: <value>`.
- Python programs worked (11.3%). Equations in *paper notation* (`M0 = μ·A·D [N m]`) worked much
  better (17.8%) from less data: the base model learned from papers and Wikipedia, never from
  code, so this format matches what it already knows.

**4. Generate the data locally.** `Qwen3.8-27B-FP8` (Apache-2.0) runs under vLLM on this server.
Every label is checked by execution, and seeds come only from the SFT training split.

| data | count | check |
|---|---|---|
| new exercises seeded by training questions | 34.8k → 24.4k used | an independent re-solve must agree (92% do) |
| real training questions turned into executable solutions | 2.3k → 1.8k | must reproduce the reference answer |
| the same with new numbers (no teacher needed) | 11.8k → 9.1k | re-executed |
| teacher rewrites of real training questions (2 rounds × 6) | 38.8k → 22.4k | re-solved |
| rewrites with new numbers | 58.9k → 47.2k | re-executed |
| conversion of everything to paper notation | ~60k inputs | kept only if it executes to the same answer |
| **final SFT mix (the deployed student)** | **110.9k pairs** | 104.9k numerical + 6.0k original-SFT replay |

**5. Rewards and verifiers.**
- *Composite reward (the first RL runs).* Seven components: final value within tolerance; concept
  coverage after `</think>`; non-empty `<think>`; termination; and penalties for echoing the
  prompt, invented citations and rambling. Each is paired with an adversarial probe (§5.6).
- *Correctness reward (all later runs).* The equations are executed: reward 1 if the result is within
  1%, 0.5 within 5%, 0 otherwise or if they do not run.
- *Training setup.* Tokens written by the interpreter are masked from the loss. Only questions
  in the learnable band (solved 1–7 times out of 8) are trained on, and checkpoints are chosen
  on a dev split.

**6. Results** (153 real held-out numerical questions, 6 seeds × 4):

| model | all | must-compute | vote@8 |
|---|---|---|---|
| SFT, prose answers | 4.7% | 2.8% | 8.5% |
| + first GRPO (composite reward) | 6.1% | 3.1% | 8.1% |
| executed Python solutions | 11.3% | 11.1% | 14.4% |
| paper-notation solutions | 17.8% | 17.3% | 19.2% |
| + GRPO on executed answers | **21.2%** (+3.3 pp, significant) | 20.1% | — |
| paper notation, more data (**deployed**) | 19.7% | 20.1% | **22.2%** |
| + GRPO on executed answers | 21.0% (+1.3 pp) | 20.4% | 22.7% |
| teacher (27B), blind | 49.7% | 48.5% | — |

<p align="center"><img src="assets/numerical_track.png" width="92%"></p>

**7. Were these gains expected?**
- *SFT: yes, and in the expected way.* A 113M model doing arithmetic "in its head" sits at the
  floor. Offloading the computation and training on many verified, executable solutions is the
  known recipe for small models (PAL/PoT; [TinyGSM](https://arxiv.org/abs/2312.09241)), and it
  worked. The most instructive gain was the format: solutions written the way the pretraining
  data writes them.
- *RL: yes, once you look at where its gains go* (§5.18).
  - On the problems it practises on, GRPO takes accuracy from 49% to 85–88%. That is the
    small-model GRPO result, reproduced.
  - Its gains spread to other questions in proportion to their similarity to the practised ones:
    +10 to +24 pp for close variants, about 0 for unrelated questions.
  - The real held-out questions are, by design, from papers the model never practised on, so
    they sit far from every RL pool. That predicts small held-out gains, which is what was
    measured. It also says what to do next: build RL pools that land near the questions you care
    about.

---

## 2. Quick start

```bash
conda activate nanogpt_seis      # same env as the pretraining repo (CUDA-12.4 torch)

# --- chat with the shipped model -------------------------------------------
python -m src.sft.chat                       # the deployed model (configs/deploy.yaml): paper-notation
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

# --- measurement audit: run BEFORE believing any result ---------------------
python -m src.sft.splits data/rl/test.jsonl            # the split contract
python -m src.analysis.validate_measurement all        # floor / ceiling / reward hacking
python -m src.analysis.validate_measurement group      # one real G=8 group, scored
python -m src.rl.eval_rl -k 4                          # paired, McNemar, pass@k, power
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
[§1.2](#12-base--sft--rl) and §1.4 report every result split this way.

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

$$\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[ \frac{1}{\sum_{i=1}^{G} |y_i|} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min\left( r_{i,t}(\theta)\hat{A}_{i,t}, \; \mathrm{clip}\left(r_{i,t}(\theta),\, 1-\epsilon_{\text{low}},\, 1+\epsilon_{\text{high}}\right)\hat{A}_{i,t} \right) \right]$$

$$\text{subject to}\quad 0 < \left| \{ y_i : \text{is-correct}(y_i) \} \right| < G$$

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
transfer here — at 113M the reference KL is what stops this policy degenerating. §5.8
shows the other half of that lesson: how β is set matters as much as whether it exists.

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
is $(1 - p_{\text{ref}}/p)\,\nabla\log p$ — unbounded on tokens the policy has suppressed,
and in expectation the gradient of $\mathrm{KL}(p_{\text{ref}}\Vert p)$ rather than the
$\mathrm{KL}(p\Vert p_{\text{ref}})$ it logs — so v4 uses $k_2 = \tfrac12(\log p - \log p_{\text{ref}})^2$,
whose on-policy gradient is exactly $\nabla\mathrm{KL}(p\Vert p_{\text{ref}})$ (k3 is still
what gets logged); there was no LR warmup, so Adam's largest steps came first
(`warmup_steps: 10`); and the loss weighted each sequence by $1/|y|$ (`loss_agg: constant`
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
changing β alone, and the "controller working" in the old version of this section was the
controller undoing the run. (One seed per arm; both arms share the fresh-optimizer restart,
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

**Deployment-decoder check.** A cleaner curve is worth nothing if the model got worse.
(These numbers use the old numerical labels; the corrected comparison is in §1.2 and gives the same verdict, with v4 − v3 at +2.0% [−0.7%, +4.9%].)
Scored with `src.rl.eval_rl` under the pinned decoder (temp 0.3 / top-k 40 / rp 1.2) on the
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
(§1.2). Re-running the §7.2 tiering on v4 gives the same split as v3: retrievable +11.8%,
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
every one of them. That asymmetry is a sufficient explanation for which behaviour moved.

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
  tokens gives the ~20 tokens/param that §1.1 contrasts with this project's own measured
  frontier (31 exposures per non-embedding parameter).

The honest contrast: DeepSeekMath shows group-relative RL improving math when the base can
already do math. This is a different regime — 113M params, sparse correctness reward,
11-step held-out derivations — and there the same family of ideas moved format decisively
while leaving multi-step computation untouched.

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
  too far from the held-out questions, not that RL is broken (§5.18).

### 5.13 Reward hacking: what to watch

Every reward component has a cheapest shortcut: empty `<think>` tags, sprayed numbers, keyword
stuffing, padding to dilute an echo ratio. Each one here is paired with a standing probe, and
the full list with the one live exploit that was found and fixed is in §5.6. Unit-test the
reward against deliberately degenerate completions before training (`python -m
src.rl.verifiers`, `python -m src.analysis.validate_measurement hacking`).

---

### 5.14 Calculator tool use and test-time compute — built and measured

The §10 gate said a calculator would not help: the expressions this model writes are noise.
That was a prediction from executing its own `<think>` arithmetic, so the calculator was then
built, trained in, and put inside the RL loop to test it directly. Full log in
[audit §15.8](audit_report_sft_grpo.md).

**Mechanism.** The model writes `<<expr>>`, and the decoder (`src/rl/rollout.py`,
`calculator=True`) evaluates it with `safe_eval_number` and forces ` = value` into the
sequence. `safe_eval_number` is an AST whitelist of numbers, operators and math functions. No
`exec` is used: a restricted `__builtins__` dict is not a sandbox. In RL those forced tokens are
observations, so `train_grpo` masks them out of the policy and KL terms (`calculator: true`).

```python
# src/rl/rollout.py — inside the decode loop
if tail.endswith(">>") and "<<" in tail:
    val = safe_eval_number(tail[tail.rfind("<<") + 2:-2])
    if val is not None:
        queue[i] = tokenizer.encode(f" = {val:.4g}").ids    # forced next tokens, flagged
```

**Teaching it.**
- `tools/numeric_drills.py`: 15 seismology computation templates, 6,000 training drills and
  279 held-out drills.
- `src/sft/calculator_format.py`: rewrites the SFT corpus's own arithmetic into calls, only
  where the call reproduces the reference's stated value (2,832 of 3,705 numerical pairs).
- Continued SFT on each mixture takes ~3 min.

| 153 held-out numerical items, acc@5% | held-out drills | real items | real completions calling it | must-compute (floor 3.3%) |
|---|---|---|---|---|
| SFT | 3.9% | 4.6% | 0% | 2.8% |
| + computation drills | **86.4%** | 5.3% | 10% | 3.4% |
| + calculator-format corpus | 81.0% | 5.1% | **93%** | 3.1% |
| + calculator-in-the-loop RL | — | 5.4% | 91% | 2.9% |
| first GRPO (no calculator), for reference | — | 6.7% | 0% | 3.1% |

- On drills, where the formula is the template, the calculator takes the model from 4% to 86%.
- On real questions, the habit transferred: 93% of completions call it, and completions with a
  wrong mental-arithmetic step fell 82% → 40% (19% after calculator RL). Accuracy did not move.
- The calculator faithfully executes the wrong setup: a hallucinated formula, or
  `<<2.8-3.0 * 2.8>>` for "2.8 / 3".
- A decoder bug, fixed later (§5.16): a forced value was not terminated, so the model could
  extend it (` = 5` + `.833`). Re-scoring every calculator model after the fix moved results by
  ≤ 1 pp.
- RL with the calculator found nothing to reinforce. Correct setups are as rare as before, so
  `r_num` stayed at its floor whether or not a rollout used the calculator.
- Calculator use in the T=1.0 training rollouts dipped 0.80 → 0.35 early in RL and recovered
  to 0.69. The dip came from truncation: longer calculator rollouts missed `</think>` and paid
  the format penalty. The numbers did not cause it.

**Self-consistency** (`src/rl/self_consistency.py`; `eval_rl` reports `vote_at_k_5pct`).
- Rule: sample k answers and return the one whose final-answer numbers agree with the most
  others. It is scored as one sample, so it is comparable with acc@1, unlike pass@k.
- Results over 8 seeds × 4 samples:

| acc@5% | acc@1 | vote@8 | vote@8 − acc@1 | answer in prompt (19) | must-compute (134) |
|---|---|---|---|---|---|
| SFT | 4.9% | 5.7% | +0.9 pp [−0.5, +2.3] | 18.6% → 25.0% | 2.9% → 3.0% |
| first GRPO | 6.4% | **9.2%** | **+2.7 pp [+0.9, +4.7]** | 29.3% → 43.4% | 3.2% → 4.3% |

Voting is the one test-time method here that pays: with the prose model it reached 9.2% against
SFT's 4.9%, and it is part of the deployed configuration today (§1.4). It costs 8× the decode, which batches into a single
call. Like RL, though, it works only where the answer can be read off the question. Agreement
cannot select a right answer that the model almost never produces.

```bash
python -m src.sft.chat --prompt "..."                                            # deployed model, vote@8 built in
python -m src.sft.chat --ckpt-path checkpoints/sft_calc/sft_ckpt_epoch2.pt --prompt "..." --calculator
python -m src.rl.eval_rl -k 8 --calculator                                       # reports vote@k too
python -m src.analysis.tier_vote docs/eval_vote_pooled8.json --baseline SFT       # the tables above
python -m src.analysis.tier_vote docs/eval_calc_k4.json --baseline SFT+calc
```

### 5.15 Speed: where RL time goes, and what TRL's trainer contributed

<p align="center"><img src="assets/grpo_speed_v4b_vs_v5.png" width="92%"></p>

- **Before.** At 113M parameters, RL is a decoding problem. In the stabilised trainer, generation took 82% of each
  step at 0.4% MFU (A30 bf16 peak 165 TFLOPS). Decode is launch-bound, at ~700 kernels per
  token.
- **CUDA graphs.** Capturing the decode step as a CUDA graph (`rollout.GraphDecoder`, static KV
  cache, length-bucketed) cut generation from 11.0 s to 2.9 s per 128 rows per GPU. A
  grouped-head GQA kernel and a bf16 cache alone gave 10%.
- **Reusing each rollout batch, from TRL's [GRPO trainer](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md).**
  `num_iterations` (μ=2) takes two optimiser steps per rollout batch. The ratio is clipped to
  [1 − 0.2, 1 + 0.28] (DAPO clip-higher). Truncated importance sampling (TIS, cap 2) guards the
  bf16 sampler/trainer mismatch. That mismatch is 0.020 nats per token, so the TIS weights
  average 1.000: a guard here, not a correction.
- **Eval.** The same batched decoder replaced one-sequence-at-a-time streaming in `eval_rl`:
  ~45 min per model became ~20 s.

| run (300 updates, 2× A30) | wall | per rollout batch | generation share | MFU gen / update / step | held-out objective, last 100 updates | KL at end |
|---|---|---|---|---|---|---|
| stabilised trainer (old labels) | 82 min | 13.4 s | 82% | 0.4% / 11% / 2.3% | (old labels; not comparable) | 0.057 |
| + corrected labels, CUDA-graph decoding | 27.7 min | 4.7 s | 60% | 1.7% / 15% / 6.5% | −0.181 | 0.062 |
| **+ rollout reuse** (μ=2, clip-higher, TIS) | **17.4 min** | 6.1 s (2 updates) | 47% | 1.7% / 15% / **8.3%** | −0.187 | 0.103 |

MFU counts 2N FLOPs per generated token. For the update it counts the reference forward plus
the policy forward and backward over prompt plus completion. The final trainer is 4.7× faster in wall
time than where it started, and 1.6× faster than CUDA-graph decoding alone, at an equal held-out
plateau and equal deployment accuracy (6.4% both, 8 seeds). Its KL ends higher (0.10 vs 0.06): with μ=2, lower the learning
rate if KL matters.

Not adopted from TRL, each for a measured reason:
- **β=0.** While the first trainer's adaptive β sat at ≈0.005 (steps 6–65), its KL climbed from
  0.003 to 0.09 in 60 steps; at a fixed β=0.04, the stabilised trainer took 300 updates to reach
  0.057 (§5.8).
- **`mask_truncated_completions`.** The termination reward already prices truncation.
- **`top_entropy_quantile`.** Token entropy here is flat at 2.3–2.6 nats, with no collapse to
  protect against.

### 5.16 Why numerical accuracy is stuck: a stage-by-stage diagnosis

<p align="center"><img src="assets/numerical_diagnosis.png" width="98%"></p>

The task was to improve numerical accuracy without retrieval, so "the model is too small" was
treated as a hypothesis to test. Each stage of producing a numerical answer was measured
separately (full log: [audit §15.9](audit_report_sft_grpo.md)). Instrument ceiling on the 134
must-compute items: 85.8%. Question-blind floor: 3.3%.

| stage | test | result |
|---|---|---|
| derive the value | does the right value appear anywhere in the completion? | at chance: 14.7% vs 13.1% for another item's target |
| read the question | 41 items need ≤ 1 operation on the question's own numbers | every model 3–5% |
| restate a finished derivation | prefill the full reference `<think>`, let the model answer | **39%** (rp 1.2); **53%** with the penalty off |
| finish a nearly finished one | prefill 90% of it | 16–22% |
| memorise | numerical questions it was trained on | 5.1%, the same as held-out; 3.9% after training to loss 0.39 |
| arithmetic | calculator in the loop (§5.14) | wrong mental steps 82% → 19%, accuracy unchanged |

Two of these are fixable mechanics:
- **The repetition penalty.** It counts the prompt and the derivation as "seen", so copying a
  given value or restating a result is penalised. Applying it to generated tokens only
  (`rp_scope="output"`) wins on all three pre-registered criteria for v4 (dev 5.4 → 6.5%, test
  7.2 → 7.7%). The decoding contract is unchanged pending a decision.
- **The SFT final answers.** They re-derive in prose instead of restating, so the model attempts
  every derivation twice. The training format now opens the answer with `Answer: <result>.`.

Neither is the bottleneck. The model never derives the value in the first place, even on
questions it trained on, because sampling a ~300-token free-prose derivation compounds small
per-token errors (computed result tokens: top-1 0.74 on train, 0.61 held-out).

**What training fixes, and how far it transfers.** Two generators make binding values to a
relation trainable at scale:
- `src/sft/counterfactual.py` extracts each reference derivation as a program, puts new numbers
  in the real question, and re-evaluates every step exactly. 1,248 training questions yield
  4,926 variants.
- `tools/word_problems.py` produces 30k problems over 14 families, including applying a
  formula the question states, with 54 phrasings of which 20% are held out.

| acc@5% | trained questions, new numbers | unseen phrasings | held-out questions, new numbers | real held-out (6 seeds) |
|---|---|---|---|---|
| SFT | 3.1% | 2.0% | 4.2% | 4.7% |
| + counterfactuals + word problems | **18.0%** | **67.8%** | 4.0% | **6.5%** (+1.9 pp, no RL) |
| first GRPO, for reference | 2.4% | 1.4% | 4.4% | 6.1% |

- **Close to training, the skill is learnable.** Accuracy goes from 3% to 18% with new numbers
  in trained questions, and from 2% to 68% on phrasings never seen.
- **It does not abstract to new question types**, not even when the held-out computation is a
  plain ratio that 212 training items share. On real questions SFT alone now matches the first GRPO model.
  Must-compute stays near the floor (3.9–4.1%, n.s.).
- **The limit is the number of distinct question types**, ~1,250 in this corpus. TinyGSM needed
  ~12M synthetic problems for a 125M model to reach 63% on GSM8K.
- **The non-RAG path is data.** That means a large, diverse, teacher-generated set of seismology
  problems in the real question style with program solutions, trained exactly as here. Or a
  larger base.

```bash
python -m src.sft.counterfactual data/splits/train.jsonl data/splits/train_cf.jsonl --variants 8
python tools/word_problems.py --n 30000 --out data/sources/word_problems_train.jsonl \
    --heldout-out data/eval/word_problems_heldout.jsonl --holdout 0.2 --n-heldout 800
torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_wp2.yaml --snapshot-epochs
python -m src.analysis.tier_vote docs/eval_numerical_real6.json --baseline SFT
```

### 5.17 Program of thought with a local teacher: the first must-compute gain

§5.16 located the limit in too few distinct question types. Small models that do well on numerical
word problems are trained on very many problems with *executable* solutions. This section builds
that on this server, with no external API (full log: [audit §15.10–15.16](audit_report_sft_grpo.md)).

- **Teacher.** `Qwen/Qwen3.8-27B-FP8` (Apache-2.0) runs under vLLM on the 2× A30 at ~900 tok/s,
  with FP8 weight-only kernels. Solving blind, it scores 49.7% on the held-out questions, which is
  the practical ceiling for distillation.
- **Data.** `tools/gen_pot_problems.py` does three things:
  - writes exercises seeded by training questions;
  - rewrites every real numerical training question in new scenarios;
  - turns real questions into executable solutions.

  Every solution is executed and kept only if it reproduces its label; generated exercises must
  also survive an independent re-solve. `tools/pot_counterfactuals.py` re-instantiates the
  real-style questions with new numbers, no teacher needed.
- **Answering.** The student writes its working as a short program; `src/rl/pot.py` executes it at
  `</think>`, and the decoder writes `Answer: <value>`.

| milestone (153 real held-out numerical questions, 6 seeds × 4) | must-compute (134) | all | vote@8 |
|---|---|---|---|
| SFT, prose answers | 2.8% | 4.7% | 8.5% |
| executed Python solutions: teacher exercises + real questions with new numbers | 6.2% | 6.8% | 9.8% |
| + real-style rewrites, decoded without the repetition penalty | 11.1% | 11.3% | 14.4% |
| paper-notation solutions (the same solutions, written as paper equations) | 17.3% | 17.8% | 19.2% |
| + recovered notation data (**deployed**) | **20.1%** | **19.7%** | **22.2%** |
| 27B teacher, blind | 48.5% | 49.7% | — |

What each step taught:
1. **Real style is what transfers.** Teacher exercises improve accuracy on more exercises of the
   same kind steeply (4% at 3k, 21% at 30k) but barely move real questions. Real questions with new
   numbers, and rewrites in the real multi-part style, doubled must-compute.
2. **Decode programs without the repetition penalty.** It penalises reusing variable names, which a
   program must do, so it turns working into crashes (programs that run: 40% → 57%).
3. **Write solutions the way the base model reads.** The base model was pretrained on papers and
   Wikipedia, not code, and 45.5% of its Python programs crashed, mostly on misspelt names. The same
   solutions rewritten as paper equations (`M0 = 10^(1.5*Mw + 9.1) [N m]`, short symbols, units in
   brackets) give 17.8% against 11.3%, from 29% less data. The working is shorter, runs more often
   (75% vs 58%) and is more often right when it runs (23% vs 19%).
4. **GRPO adds on top, where its pool reaches.** Correctness-reward GRPO on the executed answers
   adds +3.3 pp (significant) to the first paper-notation student and +1.3 pp to the deployed one.
   §5.18 shows where those gains go.

```bash
python tools/gen_pot_problems.py generate --model Qwen/Qwen3.8-27B-FP8 --tp 2 --n-seeds 10000 --per-seed 4 \
    --seed-pool all --out data/sources/pot/gen.jsonl            # then: solve --inp ... ; convert ; audit
python tools/pot_counterfactuals.py data/sources/pot/convert_27b.jsonl data/sources/pot/convert_cf.jsonl --variants 6
python tools/format_pot_sft.py --require-verified --out data/sources/pot/sft_gen.jsonl data/sources/pot/gen_solved.jsonl
torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft --config configs/sft_pot_realcf.yaml --snapshot-epochs
```

### 5.18 Where RL helps: a transfer study

> **In one sentence.** GRPO is a precise tool: it reliably lifts the problems it practises on
> (+37 pp) and their close neighbours (+10 to +24 pp). How far a gain travels can be predicted
> from how similar a question is to the training questions. That turns "why doesn't RL move the
> held-out score?" into a concrete recipe for making it move.
> Full tables and provenance: [audit §15.17](audit_report_sft_grpo.md). Figure:
> `assets/rl_transfer_diagnosis.png`.

#### 5.18.1 The puzzle

By §5.17 the student had come a long way:
- **Format.** RL taught it the answer format (`<think>` compliance 69% → 98%, §5.8).
- **Accuracy.** Paper-notation SFT took real held-out accuracy from 4.7% to 19.7%.
- **One RL win.** GRPO added a significant +3.3 pp on top of the first notation student (§15.15).
- **A smaller second step.** On the stronger student, the next GRPO run added +1.3 pp (§15.16).

Meanwhile, reports of ~100M-parameter models gaining a lot from GRPO on arithmetic, Countdown
or GSM8K-style problems are common. So is our RL set up wrong, or is it measuring something
different? This section answers that with a controlled experiment. The answer is encouraging:
the RL is working exactly as it should, and the experiment shows where its gains go.

#### 5.18.2 RL in one page (for readers new to it)

- **The policy is the model.** Given a question, it writes a solution token by token. Each
  possible solution has a probability under the current weights.
- **The verifier is the teacher's red pen.** Here the solution is a short list of equations; the
  interpreter executes them and the reward is 1 if the result is within 1% of the label, 0.5 if
  within 5%, else 0. Nothing about style is rewarded, only the executed answer.
- **GRPO in four steps** ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)):
  1. sample a *group* of G = 8 solutions to the same question;
  2. score each one;
  3. compute each solution's *advantage*, its reward minus the group's mean reward (scaled by
     the batch's standard deviation);
  4. take a gradient step that raises the probability of every token in above-average solutions
     and lowers it in below-average ones.

  The group mean plays the role PPO's learned critic would play: it says what "normal" looks like
  for this question, so the model is pushed only toward what is *better than its own usual*.
- **Why only medium-difficulty questions teach anything.** If all 8 solutions are right, or all
  8 are wrong, every advantage is zero and the gradient is exactly zero. RL learns only from
  questions the model *sometimes* solves. That is why every RL pool here is filtered to questions
  solved 1–7 times out of 8 (the "learnable band", DAPO's dynamic sampling idea,
  [Yu et al. 2025](https://arxiv.org/abs/2503.14476)).
- **The KL anchor.** A penalty β·KL(policy ‖ SFT model) keeps the policy close to where it
  started, so it improves without forgetting how to write. KL is measured in nats per token: the
  average change in log-probability the update has made.
- **Sharpening versus discovering.** *pass@k* is "solved at least once in k tries"; *pass@1* is
  "solved on a single try". RLVR mostly turns pass@k into pass@1: it makes the model reliably do
  what it could already do sometimes
  ([Yue et al. 2025](https://arxiv.org/abs/2504.13837)). Keep this in mind for §5.18.6.

#### 5.18.3 The experiment: a ladder of distances

To see where RL's improvement goes, we need test problems at known distances from the training
problems. The 27B teacher wrote new problems from source chunks the student has never trained on
(none of them in SFT):

| set | what it is | role |
|---|---|---|
| **A** (2,027 problems, 115 papers) | 576 are in the learnable band | **RL training pool** |
| A, never solved (1,272) | same papers as the pool, but the student solved them 0/8 | *same paper* (not trained on) |
| **B600** (600 problems, 92 other papers) | no overlap with A | *other papers* |
| B-other (700 problems, B600's papers) | different problems from the same papers as B600 | a second RL pool |
| **real** (153 questions) | the real held-out benchmark | *the target* |

To measure "how close" two questions are, we use **word-Jaccard similarity**: the share of
content words they have in common (intersection over union of their word sets). It is crude, but
as we will see, it predicts transfer well. GRPO runs exactly the recipe of §15.15: lr 5e-6,
β 0.02, correctness reward, 150 steps × 32 questions × 8 samples. All scores use the deployment
decoder (T = 0.3, 2 seeds × 4 samples) with paired 95% bootstrap CIs.

<p align="center"><img src="assets/rl_transfer_diagnosis.png" width="92%"></p>

#### 5.18.4 Finding 1: RL learns quickly and thoroughly

On its own 576 problems, GRPO takes accuracy from **48.8% to 85.4%** in 75 steps (88.1% at 150).
The improvement is not a single trick. Every kind of error shrinks by about three quarters
(400 problems × 4 samples):

| error class (share of samples) | before RL | after RL |
|---|---|---|
| program crashes | 9.3% | 2.3% |
| off by a power of ten (units) | 9.0% | 2.3% |
| off by a common factor (2, π, 60…) | 2.7% | 0.6% |
| other wrong setup | 19.3% | 5.4% |
| **correct** | **51.7%** | **86.1%** |

This is the small-model GRPO result reproduced. The optimiser, the reward and the masking all
work: a 113M model can learn a domain's problem-solving from verifiable rewards.

#### 5.18.5 Finding 2: improvements travel by similarity

The same run, scored on problems it never trained on, grouped by similarity to the nearest
training question:

| similarity to the nearest trained question | never-solved problems from the *same* papers | problems from *other* papers (B600) |
|---|---|---|
| < 0.2 (unrelated) | +1.0 pp | −1.3 pp |
| 0.2–0.35 | +5.0 pp | +0.3 pp |
| 0.35–0.5 | +9.9 pp | +9.7 pp |
| ≥ 0.5 (a close variant) | **+17.1 pp** | **+27.2 pp** (n = 17) |

The pattern is the same within A and across papers: **the gain on a question is set by how
similar it is to something the model practised on.** Problems the student had *never* solved go
from 1.9% to 19.0% when a close variant was trained on. Problems with nothing similar in the pool
stay where they were, whichever paper they come from.

*Why this happens (the fundamental picture).*
- All questions share one set of weights. A gradient step on question x changes the model's output
  on another question x′ by roughly η · K(x′, x) · (the push applied at x). Here K is the
  similarity of the two questions *as the network represents them* (the neural tangent kernel,
  [Jacot et al. 2018](https://arxiv.org/abs/1806.07572); influence functions,
  [Koh & Liang 2017](https://arxiv.org/abs/1703.04730)).
- Updates therefore spread to questions that look alike *to the model*. Word overlap is a cheap
  proxy for that, and the table shows it is a good one.

A second RL pool confirms it. GRPO on 230 *other* problems from B600's own papers lifts B600 by
**+10.1 pp** [+7.4, +12.8] (23.9% → 33.9%). GRPO on set A moved it +1.6 pp. B600's gain again
follows similarity: +1.3, +7.3, +11.7 and +23.6 pp across the four bins.

#### 5.18.6 Finding 3: what carries over, and what stays local

Accuracy here factorises into two parts: **does the program run**, and **given that it runs, is
the setup right** (the right relation, constants and units). Measured at T = 1 (200 problems × 4):

| set | program runs (before → after) | correct given it runs (before → after) |
|---|---|---|
| trained problems (A) | 70% → 86% | **50% → 77%** |
| other papers (B600) | 60% → 64% | 27% → 28% |
| real | 53% → 56% | 25% → 24% |

- **The program-writing skill carries over.** Programs run more often on every set: fewer
  misspelt symbols, fewer undefined quantities.
- **The setup stays local.** Choosing the right relation and constants for a topic is knowledge
  about that topic. RL sharpens it where it practised (+27 pp) and leaves other topics as they
  were.
- **Shared knowledge is the bridge.** On other-paper problems that need a standard constant the
  question does not state (a rigidity, a magnitude-formula coefficient), RL on A gains
  +4.1 pp [+1.3, +7.0]. Constants that recur across papers travel; paper-specific setups do not.

Does RL change the model only on the trained questions? It does not. Its per-token KL from the SFT
model is the same on trained prompts (0.061 nats), other-paper prompts (0.058) and real prompts
(0.058). The model moved everywhere. On unseen prompts it even learned to down-weight long and
crashing solutions. But a model can only be steered toward a correct setup it can already
produce, and for topics far from the pool the steering has no target.

#### 5.18.7 Why the small-model GRPO results look different

Those results are real, and Finding 1 reproduces them: +37 pp on the pool. They are measured on
test problems drawn from the same generator as the training problems (arithmetic, Countdown,
GSM8K-style), so every test problem has a close neighbour in training. Our real held-out
questions come from papers the model never practised on, and they are far from every RL pool
built so far:

| real questions vs | median nearest-neighbour similarity | share with a neighbour ≥ 0.35 |
|---|---|---|
| RL set A (576) | 0.23 | 20% |
| the previous RL pool (3,031) | 0.27 | 29% |
| all SFT questions (110,903) | 0.32 | 44% |

With only 20–29% of real questions in the "+10 pp zone", the expected held-out gain is a few
points, which matches the +0.1 to +3.3 pp measured. The two observations, big gains on the pool
and small gains on held-out, follow from one rule.

This also matches what is known about small transformers and knowledge. Fine-tuning on questions
about some facts does not carry over to questions about other facts unless pretraining showed
each fact in several phrasings (Allen-Zhu & Li, *Physics of Language Models* 3.1,
[arXiv:2309.14316](https://arxiv.org/abs/2309.14316)). Our base model saw each paper once.

#### 5.18.8 What this means in practice

The rule gives three levers, in order of expected payoff:

1. **Put training problems near the test questions.**
   - An RL pool helps the questions it lands near. To lift held-out questions it must be
     *broad*: tens of thousands of distinct problems covering the domain's question types, each
     seen about once.
   - Most of the same-paper gain arrived within 25 steps; repeating a small pool only adds
     memorisation.
2. **Practise on the documents the questions are about.**
   - RL on problems written from a paper lifts new questions about that paper by ~10 pp.
   - For a model meant to answer questions about a known corpus, this is the natural recipe:
     generate RL problems from every document in it.
   - It changes what a held-out split measures (new questions about seen papers versus unseen
     papers), so it is a protocol choice to make explicitly.
3. **Give the hardest questions stepping stones** (§5.18.9). A question the model never solves
   gives RL no gradient. A first-step version of it does, and practising that step lifts the full
   problem about 7×.

Reading alone is not a substitute. Restating each paper in several styles did not move accuracy
on questions about it at this scale (§5.18.10). The model learns a paper's setups by practising
problems about it in the format it answers in.

#### 5.18.9 Are the questions too hard? Simpler questions as stepping stones

For RL, "too hard" has a precise meaning (§5.18.2). If the model never solves a question, all
8 samples score 0, every advantage is 0, and the question contributes **no gradient at all**. By
that measure a large share of realistic questions is currently out of RL's reach:
- 56% of the real held-out questions are never solved in 32 tries;
- 63% of set A and 61% of B600 are never solved in 8.

Two earlier results suggested the number of solution steps is not the obstacle:
- inside the pool, problems of every length gain 28–41 pp;
- never-solved problems rise when a close variant is practised.

**The test.**
- The 27B teacher turned each problem in A and B600 into a short question that asks for *one
  early quantity* of the original solution: 1–2 computed steps from the givens, in the same
  scenario. For example, "calculate the average slip" instead of "the average slip, then the
  slip rate in mm/day".
- The label is that intermediate value from executing the original solution, so it is exact.
- Result: 1,532 simple A questions and 427 simple B600 questions.
- GRPO then ran the same recipe on the 499 learnable simple A questions.

| | original questions | simple questions |
|---|---|---|
| SFT start on B600 (other papers) | 23.9% | 34.5% |
| share of A never solved (0/8) | 63% | 52% |
| share of A in the learnable band | 28% | 33% |
| GRPO on the pool, step 75 | 48.8% → 85.4% (+36.6) | 47.0% → **90.3%** (+43.3) |
| transfer to the same kind of question from other papers | +1.6 [−0.4, +3.7] | +3.0 [−0.3, +6.2] |

Simple questions are easier, more of them fall in the learnable band, and RL masters its pool
of them almost completely. They also carry over slightly further to other papers, plausibly
because a single step ("slip from moment, rigidity and area") recurs across papers more often
than a whole multi-part problem does. On their own, though, they do not raise the *original*
multi-part questions from other papers (B600 −1.5 pp, real +1.3 pp, both n.s.).

<p align="center"><img src="assets/rl_curriculum.png" width="88%"></p>

**The curriculum effect.** The interesting result is on the hardest problems: the original A
problems the model could never solve.

| never-solved original A problems (same papers as the pool) | n | before | after GRPO on simple questions |
|---|---|---|---|
| their own first step was in the simple RL pool | 205 | 1.5% | **10.8%** (+9.3 [+5.7, +13.2]) |
| their first step was not | 836 | 1.8% | 4.8% (+2.9 [+1.9, +4.1]) |

Practising the first step of a problem the model could never solve makes it solve the *whole*
problem seven times as often. That is how simplification helps RL: it turns a question that gives
no gradient into a stepping stone that does. For problems already in the learnable band, step
practice adds nothing over practising the full problem (−1.1 pp versus +37 pp).

**So: are the questions too hard?** For RL, about 60% of them are, in the precise sense above.
The productive response is not to make the benchmark easier but to make the *training
curriculum* gentler:
- keep the full problems the model sometimes solves (the learnable band);
- add first-step and sub-step versions of the problems it never solves;
- refresh the pool as the model improves, so newly solvable problems enter the band.

Combined with §5.18.8, the recipe is a broad pool near the target questions, with stepping-stone
versions of the hardest ones.


#### 5.18.10 Reading versus practice: a knowledge-augmentation pilot

If the setup knowledge for a paper is what stays local, can the model learn it by *reading*?
Small transformers extract knowledge far better when pretraining shows each fact in several
phrasings (Allen-Zhu & Li 3.1). Synthetic rephrasing of a small corpus followed by continued
pretraining is the standard way to provide that
([Yang et al. 2024, "Synthetic continued pretraining"](https://arxiv.org/abs/2409.07431)).

**Setup.**
- The 27B teacher restated each source chunk of B (87 papers) and A (105 papers) in 8 styles: a
  textbook section, lecture notes, an encyclopedia summary, a review-article paraphrase, and so
  on. The rules were: keep every equation, value and unit, use paper notation, and add no
  questions or worked examples.
- Arm **K** continues the deployed student on the 658 notes about B's papers plus 3,000 replay
  pairs (3 epochs, lr 5e-5).
- Arm **C** is the identical control on A's papers.
- If reading about a paper helps, K should beat C on B and C should beat K on A.

| | notes about B (K) | notes about A (C) |
|---|---|---|
| B questions (360 with recovered source text), start 14.3% | 14.5% (+0.2 [−1.9, +2.1]) | 14.6% (+0.3) |
| A questions (400 sampled), start 18.1% | 16.8% (−1.4) | 15.5% (−2.7 [−4.5, −1.0]) |
| real held-out, start 19.7% | 20.1% | 17.6% |

**Result.** Reading about a paper did not change accuracy on questions about it. Practice did:
RL on 230 problems from the same papers lifted B600 by +10.1 pp (§5.18.5).

At this scale, the model learns a paper's setups by *solving problems about it in the format it
answers in*, not by reading prose. This matches the "mixed training" finding of Allen-Zhu & Li:
knowledge becomes extractable when it is practised in the form in which it will be queried.

Caveats for anyone scaling this up:
- 8 restatements × 3 epochs is about 24 exposures per paper. Knowledge-storage studies use far
  more (~100 or more).
- Roughly 3% of the notes were empty because the chunk was a bibliography.

The next version to try interleaves notes with practice problems on the same papers.

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
The results are in [§1.2](#12-base--sft--rl) and §1.4.

---

## 7. What it found

### 7.1 Headline results

| capability | result | where |
|---|---|---|
| answer format (`<think>`, termination) | SFT 69% → GRPO v4 98% | §1.2, §5.8 |
| closed-book numerical accuracy, 153 real held-out questions | 4.7% (SFT) → 19.7% (executed-equation SFT) → **22.2%** with vote@8 (deployed) | §1.4, §5.17 |
| must-compute questions (floor 3.3%) | 2.8% → 20.1% (22.4% with vote@8) | §1.4 |
| GRPO on executed answers | +37 pp on its pool; +10 pp on new questions from the same papers; +0 to +3.3 pp on the held-out set | §5.18 |
| stepping stones | practising a never-solved problem's first step lifts the full problem 1.5% → 10.8% | §5.18.9 |
| practical ceiling | the 27B teacher, solving blind: 49.7% | §5.17 |

The stage-by-stage chart with CIs is in §1.4.

### 7.2 Capability map

- **Formatting.** Solved by RL: `<think>` compliance ~98%, an effect prompting could not
  reproduce.
- **Prompt-value extraction.** On the answer-in-prompt tier, 23.7% → 34.2% from SFT to v4.
- **Multi-step computation.**
  - Closed-book, it is now well above its floor: must-compute 20.1% against 3.3%, once the
    student writes equations that are executed.
  - The remaining errors are setups (relation, constants, units) for topics the model has not
    practised.
- **Generalisation of RL.** Gains travel by similarity to the practised problems (§5.18).
- **Unmeasured.**
  - Conceptual factuality on 642 of the 808 held-out prompts.
  - Multi-turn behaviour, refusal and calibration.

The first evaluation, which used the old labels and found no computation, is kept in the audit
report (§10–§14) and [PITFALLS.md](PITFALLS.md). §5.16–§5.18 supersede its explanations.

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

**Running: broad-pool RL with stepping stones, under two evaluation protocols** (audit §15.18).
- **Shared design.**
  - Problems the student has never seen, built by `tools/build_rl_pool.py`: deduplicated against
    the SFT data, at most 2 per template.
  - First-step versions of the never-solved ones (`tools/gen_stepping_stones.py`).
  - One pass, so each problem is seen about once; the checkpoint is chosen on a dev set before
    the test set is touched.
- **Strict protocol** (`configs/rl_grpo_v13_broad.yaml`). Problems come only from
  SFT-training seeds, so the held-out questions stay about papers the model never practised on.
- **Corpus protocol** (`configs/rl_grpo_v14_corpus.yaml`).
  - Adds exercises the teacher wrote from the papers the held-out questions were drawn from
    (`tools/gen_chunk_problems.py`).
  - None shares a 13-gram with a held-out question, or has word-Jaccard ≥ 0.5 with one.
  - It measures new questions about papers the model has practised on: the setting of a model
    that answers questions about a known corpus. §5.18.5 predicts roughly +10 pp there.
- **Gate.** Keep a run if the dev set improves over the deployed student. The similarity law's
  prediction for each pool is logged before training.

**Queued.**
- **Notes interleaved with practice.**
  - Reading alone did not help (§5.18.10); practice did.
  - Gate: the K-versus-C crossover of §5.18.10 with practice problems mixed in.
- **Stronger base.**
  - No model-size scaling was measured.
  - Gate: this evaluation on a 0.5B base before more post-training investment at 113M.

**Closed.**
- *Calculator* (§5.14): it removed arithmetic errors and exposed setup errors.
- *Self-consistency*: deployed as vote@8.
- *Diverse question types*: supplied by teacher data (§5.17).
- *RFT*: it learns its pool, like GRPO, but transfers less (§5.18).
- *RAG* (§1.3): open-book extraction works; its archived 44.4% figure has no reproduction script.

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
configs/deploy.yaml                                       ← the DEPLOYED model and its decoder: sft_pot_notation2,
                                                            executed paper-notation equations, vote@8 (§1.4)
checkpoints/
  sft_pot_notation2/ sft_ckpt_epoch4.pt                    ← the deployed checkpoint (named in configs/deploy.yaml)
  sft/         selected.pt  sft_ckpt_best.pt              ← Phase II selection (prose SFT)
  grpo/        selected.pt → ../grpo_v4/grpo_ckpt_step_300.pt  ← Phase III selection (v4, prose;
               chosen by the src.sft.checkpoints rule on the dev half: docs/checkpoint_selection_rl_v4.json)
               grpo_ckpt_step_{100,200}.pt                 ← the v3 run (step 100 was the previous selection)
  grpo_v4/     grpo_ckpt_step_{50,100,…,300}.pt            ← v4 trainer run, README §5.8
  grpo_v4b/ grpo_v5/                                       ← corrected labels; + μ=2 / TIS (§5.15)
  sft_drills/ sft_calc/ grpo_v6_calc/                      ← calculator experiments (§5.14)
  runs/        sft.csv  grpo.csv  + 26 historical logs    ← every training curve
               grpo_v4.csv  grpo_v4_eval.csv               ← v4 batch + held-out curves
               grpo_v3_intervention_beta{0192,0035}{,_eval}.csv  ← β-only restart test
               grpo_v4b · grpo_v5 · grpo_v6_calc {,_eval}.csv
src/
  sft/         generate_sft_data → build_corpus (clean · merge · split)
               dataset · train_sft · chat · rag_retriever · run_data_scaling_sweep
               checkpoints.py       ← select · compare · strip
               splits.py            ← the train/val/dev/test contract
  rl/          prepare_rl_dataset · numerical_target (label rule) · verifiers (+ safe_eval)
               rollout (batched CUDA-graph decoder, calculator) · train_grpo · eval_rl
               self_consistency (vote@k)
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
- **The SFT/RL corpora are not redistributed.** `data/` is git-ignored. Rebuild it with
  `python -m src.sft.build_corpus` from the pretraining corpus plus your own teacher API
  (`TEACHER_API_BASE` / `TEACHER_MODEL` — any OpenAI-compatible endpoint). Teacher-generated
  data inherits whatever terms your provider attaches to model output, and the underlying
  source text keeps the licences listed in the pretraining repo.
- **Model weights** are a derived artifact, distributed (if at all) via a GitHub Release or
  the Hugging Face Hub, not committed to git.
- **Training logs are committed** (`checkpoints/runs/*.csv`), because every figure in
  `assets/` regenerates from them — that is what makes the numbers here checkable.
