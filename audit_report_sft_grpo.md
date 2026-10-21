# 🔍 Experimental ledger: Phase II SFT + Phase III verifier-RL pipeline

**What this document is for.** It records *what actually ran*: every audit finding, every
remediation, every measurement, in the order it happened — including superseded numbers and
why they were superseded. It is the evidence trail, not the explanation.

- **[README.md](README.md)** — how the pipeline works, why these choices, what the corrected
  experiment showed.
- **[PITFALLS.md](PITFALLS.md)** — the failure atlas: symptoms, false stories, root causes,
  regression tests, and a status table for every superseded number.

> **Update (2026-07-29, same day):** every fix in §5 has been implemented and the pipeline re-run.
> See [§6 Remediation log](#6-remediation-log-what-was-actually-changed) at the end of this document
> and README [§7 What it found](README.md#7-what-it-found) for the current results.

> **Naming note (2026-07-31).** Two renamings apply throughout, and historical text below
> predates both.
> 1. **Stages → phases.** "Stage 2 / Stage 3" below mean **Phase II (SFT)** and
>    **Phase III (verifier RL)**; Phase I is pretraining in [nanoGPT-Seis](https://github.com/jiazhe868/nanogpt-seis).
> 2. **"GRPO" is a directory name, not an algorithm claim.** What is implemented is
>    **group-relative REINFORCE with KL regularization** — group-relative advantages plus
>    reference KL, but no importance ratio and no clipped surrogate, one on-policy update per
>    sampled batch. See README [§5.3](README.md#53-how-this-differs-from-canonical-grpo).
>
> Run directories were also renamed for clarity — `sft_v6` → `checkpoints/sft`,
> `grpo_v3` → `checkpoints/grpo`, and every other run's log moved to
> `checkpoints/runs/<descriptive>.csv`. Historical run names below (`grpo_v1`, `grpo_v2`,
> `ab_16k`, `strat_*`, …) refer to those logs.

> **Module map (2026-08-08).** Scripts named below were later consolidated. Where they live
> now:
>
> | named below | now |
> |---|---|
> | `clean_sft_data_v2.py`, `merge_datasets.py`, `format_sft_dataset.py` | `src.sft.build_corpus` (`clean` / `merge` / `split`) |
> | `select_checkpoint.py`, `compare_checkpoints.py`, `strip_optimizer_state.py` | `src.sft.checkpoints` (`select` / `compare` / `strip`) |
> | `metric_validity.py`, `reward_hacking_probe.py`, `worked_group.py` | `src.analysis.validate_measurement` (`floor` / `hacking` / `group`) |
> | `tier_benchmark.py`, `oracle_execution.py`, `prompted_baseline.py` | `src.analysis.diagnose_capability` (`tier` / `oracle` / `prompting`) |
> | `src/rl/safe_eval.py` | `src.rl.verifiers` (`safe_eval_number`) |
> | `src/plots/style.py` | `src/plots/__init__.py` |

## Current status

Read this first. Everything below §6 is chronological, and early sections contain claims
that later sections retract.

| question | current answer | authoritative section |
|---|---|---|
| Current final results | retrospective audit split; the format gain survives, computation does not | §14 |
| Current algorithm name | group-relative REINFORCE with KL regularization | §14.8 / README §5.3 |
| Current token accounting | 27.6 exposures per non-embedding parameter | §14.7 |
| Current capability conclusion | extraction signal; no above-floor computation | §14.3 |
| Current split status | retrospective audit split, frozen from 2026-08-08 onward | §14.1 |
| Historical audit snapshot | **superseded** — retained for provenance | §§1–5 |

Section status labels: **§§1–5** HISTORICAL (retained for provenance) · **§6** remediation
log · **§7** SUPERSEDED BY §10.2 · **§§8–12** historical measurements, individually dated ·
**§13** SUPERSEDED IN PART BY §14 · **§14** CURRENT AS OF 2026-08-08.

---

**Date:** 2026-07-29
**Scope:** Code correctness, data quality, evaluation validity, and improvement plan under current constraints (2× A30 24 GB, 823M-token pretrain corpus, a teacher API for data generation).
**Method:** Three parallel deep code/data audits + first-hand generation probes on both checkpoints (temperature 0.2).

---

## Original audit snapshot — 2026-07-29, superseded

> **HISTORICAL.** Written before any remediation. Present-tense claims below describe
> the pipeline as it stood on 2026-07-29. For the current state see **Current status**
> above; every fix is logged in §6 and the current results are in §14.

The pipeline is pedagogically well-built — label masking, weight-tied embedding resize, on-policy GRPO with k3 KL estimator, and advantage normalization are all implemented correctly. However, **the three headline results of this repo are not currently trustworthy**:

1. **SFT val loss 2.1237 and the data-scaling law are inflated by 100% document-level train/val leakage.** The split shuffles Q&A *pairs*, not source chunks; every val question has ~1.9 sibling Q&A pairs from the same paper excerpt in train.
2. **GRPO never trained on numerical accuracy at all.** The `NumericalVerifier` reads a field (`final_numerical_value`) that exists in 0 of 16,425 RL records — `num_reward = 0.0000` at every one of the 300 logged steps. Root cause: `prepare_rl_dataset.py` calls `math.isnan` without importing `math`, so the fixed "V2 hack-proof" dataset was never generated and training silently used the stale schema.
3. **The "+590% improvement" is circular.** `eval_rl.py` scores with the *same* `CompositeRewardEngine` used as the training reward, on only the first 20 val prompts, 1 sample each; the comparison numbers are additionally hardcoded in `plot_grpo_results.py`. ~73% of the measured reward gain is `<think>`-tag format compliance.

Qualitative reality check (my own probes at temperature 0.2): both SFT and GRPO models produce fluent seismology-register prose that is factually wrong. GRPO genuinely improved surface form (correct M₀ = μAD and M_w = ⅔log₁₀M₀ − 10.7 statements, no repetition loops) but not factual content (e.g., both models place deep-focus earthquakes at 10–30 km; the real answer is 300–700 km via transformational faulting / dehydration embrittlement).

---

## 1. Confirmed bugs — SFT pipeline

> **HISTORICAL — retained for provenance.** Every item here was fixed; see §6.

| # | Severity | Bug | Location |
|---|---|---|---|
| S1 | 🔴 | Train/val split is pair-level; `chunk_id` discarded before the shuffle → 864/864 val pairs share a source chunk with train (mean 1.88 train siblings each). All val-based conclusions (LR sweep ranking, scaling law, early stopping step) are contaminated. | `format_sft_dataset.py:64-80` |
| S2 | 🔴 | Val loss is never all-reduced across DDP ranks; `DistributedSampler` shards val, so best-checkpoint selection and every logged val number use only rank 0's half of the val set. | `train_sft.py:202-241, 304, 427-441` |
| S3 | 🟡 | Val loss = mean of per-batch means over highly variable unmasked-token counts (padding tail blocks get equal weight). A batch with zero unmasked targets yields NaN and silently disables best-checkpoint saving forever. | `train_sft.py:237-241` |
| S4 | 🟡 | Sequence packing: (a) no block-diagonal mask → cross-sample attention leakage in train *and* val; (b) samples split mid-response across block boundaries keep their labels, so responses are trained with amputated prompts. | `dataset.py:197-226` |
| S5 | 🟡 | Gradient-accumulation: final partial group of each epoch still divides by full `grad_accum` → systematically undersized last step per epoch. | `train_sft.py:391-397` |
| S6 | 🟡 | Multi-turn inference encodes history assistant turns with an extra `\n` before `<|im_end|>` never seen in training (single-turn is consistent). | `chat.py:95` vs `dataset.py:108-109` |
| S7 | 🟢 | REPL ignores `--temperature/--top-k` (hardcoded); default `no_repeat_ngram=3` is hostile to formulaic scientific text and caused the observed word-salad corruption at eval; base-checkpoint fallback would crash on ChatML token IDs (no resize to 16,388). | `chat.py:347, 190, 132-167` |
| S8 | 🟢 | Data-scaling sweep: hardcoded subsets, final point mislabeled "12.6k (Full)" but is 15k (actual full = 16,425); fixed 3 epochs conflates data volume with optimizer steps. Configs `lr_sweep.yaml`/`data_scaling.yaml` are read by nothing. | `run_data_scaling_sweep.py:26, 81, 153` |

**Done correctly (verified):** label shift and prompt masking (no off-by-one; `<|im_end|>` in loss span), embedding resize preserving weight tying with optimizer created after resize, fp32 cross-entropy under bf16 autocast, gradient clipping on deduplicated params, RoPE positions with KV cache, stop condition on `{16385, 0}`.

## 2. Confirmed bugs — data quality

> **HISTORICAL — retained for provenance.** Every item here was fixed; see §6.

| # | Severity | Finding |
|---|---|---|
| D1 | 🔴 | **~60% of train records still contain source references after "cleaning"** ("the study" 21%, "the text" 14%, "et al." 12%, Figure/Table N ~11%, "the authors" 7%…). The regex list only covers a narrow set of lead-ins. |
| D2 | 🔴 | `<think>` blocks were **exempted** from cleaning (`clean_sft_data.py:77-82`): 46% of numerical reasoning chains cite invisible tables ("From Table 3: slip rate = 90 mm/yr…") — this directly trains citation hallucination inside CoT. |
| D3 | 🟡 | Regex cleaning **damaged grammar**: ~6.4% of responses opened with verb-deleted fragments, ~26% of instructions have dangling clauses ("Summarize the key findings and methods used of the 2014 Iquique earthquake"), and `\s+→' '` flattening removed **every newline in every response** — the model was fine-tuned to emit single-paragraph walls of text. |
| D4 | 🟡 | RL dataset is a 100% positional copy of the SFT train set (same prompts the policy was already fit on); `boxed_target` holds full prose (not a number); `target_numbers` is a noisy dump (avg ~20 numbers/item incl. years and intermediate steps). |
| D5 | 🟢 | Good news: 0 duplicate pairs, 0 exact/near-duplicate instruction leakage, 98–99.5% of numerical answers end with a verifiable number, and spot-recomputation of 8/10 sampled teacher derivations was arithmetically correct. |

## 3. Confirmed bugs — GRPO pipeline

> **HISTORICAL — retained for provenance.** Every item here was fixed; see §6.

| # | Severity | Bug | Location |
|---|---|---|---|
| G1 | 🔴 | `NumericalVerifier` reads `rl_item["final_numerical_value"]` — present in **0/16,425** records → numeric reward dead for the entire run (`num_reward=0.0000`, all 300 steps in `grpo_log.csv`). The 33% "numerical" prompts were scored by keyword overlap instead. | `verifiers.py:190` |
| G2 | 🔴 | Root cause of G1: `prepare_rl_dataset.py` uses `math.isnan` with no `import math` → `NameError` on first numeric answer; the upgraded dataset build crashed and stale old-schema files were used. | `prepare_rl_dataset.py:47` |
| G3 | 🔴 | Circular evaluation: `eval_rl.py` scores SFT vs GRPO with the training reward itself, first 20 val prompts, 1 completion each, temp 0.3, no error bars; the "+0.05 → +0.295 (+590%)" numbers are hardcoded in `plot_grpo_results.py:211-212`. | `eval_rl.py:66,124` |
| G4 | 🟡 | Reward is hackable and partially hacked: `KeyConceptVerifier` matches against the full text **including `<think>`** — stuffing all 8 keywords in a think block scores the 1.5 ceiling; `key_concepts` are just the reference's top-8 frequent words (ROUGE-1 in disguise; "earthquake"/"earthquakes" counted as two concepts). ~73% of the run's reward gain was `<think>`-tag formatting. | `verifiers.py:98-102`, `prepare_rl_dataset.py:54-64` |
| G5 | 🟡 | Forced-EOS padding tokens (appended so groups decode in lockstep) are **inside** the policy loss and KL: short completions get gradient on hundreds of never-sampled EOS tokens, and they dilute `mask.sum()` normalization. | `train_grpo.py:115, 272-284` |
| G6 | 🟡 | Rollouts sampled at temp 0.7 + top-p 0.95 but the loss uses untempered log-probs with no importance correction (biased off-policy REINFORCE). No KL controller: rolling KL climbed from 0.05 to 0.34 with a spike of **8.23 at step 283**; the "peak reward +0.494" at step ~300 coincides with the KL blow-up — late gains are policy drift, not improvement. | `train_grpo.py:98-112, 264-276` |
| G7 | 🟡 | GRPO rollouts and eval omit the system turn used in every SFT training example → both RL training and evaluation run the model off-distribution. | `train_grpo.py:213`, `eval_rl.py:78` |
| G8 | 🟢 | Config rot: `clip_ratio`, `ppo_epochs`, cosine scheduler, warmup, grad-accum, micro-batch settings in `rl_grpo.yaml` are all unused; actual loss is plain REINFORCE (the "Clipped Policy Gradient Loss" plot title is wrong); actual batch is 2 prompts × G=4 = 8 rollouts/step, not the documented 32. Only ~600 unique prompts were ever sampled from the 16,425-prompt set. | `configs/rl_grpo.yaml`, `train_grpo.py:209, 282` |

**Done correctly (verified):** group-relative advantage with degenerate-case handling, correct token/logit alignment and prompt masking, k3 KL estimator against a properly frozen SFT reference, strictly on-policy (1 update per rollout batch — no stale-logprob bug class), rewards computed on completions truncated at first `<|im_end|>`, honest Goodhart discussion in the README of the day (that discussion now lives in README §3.2 and §5.5).

---

## 4. Performance evaluation (what the model can actually do)

> **SUPERSEDED BY §13 AND §14.** Measured before the metric-validity fix, so the
> accuracy figures here are inflated by the min-over-8 scoring rule.

- **Reported:** val loss 2.1237 (ppl 8.4). **Reality:** measured on a leaked, half-sharded val set → the true held-out loss is unknown and certainly worse. The monotonic data-scaling curve is partly a leakage artifact (more train data → more sibling coverage of val chunks).
- **Style/format (real gains):** ChatML compliance, clean scientific register, elimination of SFT-era repetition loops (`M_ij − M_ij`), correct canonical formula statements. GRPO measurably helped here — this is the honest achievement of Stage 3.
- **Factual content (no gain):** both checkpoints hallucinate specifics (event magnitudes, depths, mechanisms) and cannot do arithmetic. Probes at temp 0.2 confirm this is not a sampling artifact.
- **Arithmetic:** never rewarded (G1), never learned. No evidence 300 GRPO steps could have taught it even with a live verifier — but the experiment was not actually run.

---

## 5. Improvement plan under current constraints

> **HISTORICAL.** Implemented in §6; outcomes in §§7–14. The token-per-parameter
> reasoning in this section is **superseded by §14.7**.

Constraints: 113M model, 823M-token pretrain corpus (≈7 tok/param — itself below the ~20:1 compute-optimal ratio), 2× A30 (48 GB total), teacher-API budget, ~5,860 source chunks / 17.3k Q&A pairs.

### Phase 0 — Fix measurement first (hours, CPU-only mostly)
1. **Re-split by `chunk_id`** (better: by paper/DOI) before train/val cut. Re-evaluate the existing `sft_ckpt_best.pt` on the new held-out val to get the first trustworthy number.
2. Fix DDP val: `all_reduce` a token-weighted loss sum / token count (fixes S2+S3 together).
3. Fix eval decoding defaults: `no_repeat_ngram=0`, honor CLI args in the REPL, gentle `repetition_penalty≤1.05`.

### Phase 1 — Reframe the task to match a 113M model's capacity (highest impact)
A 113M closed-book model cannot store paper-specific facts; the current data *demands* it. Two complementary reformulations, both cheap with the existing corpus:
1. **Grounded (open-book) SFT:** regenerate the dataset with the source excerpt *included in the user prompt* ("Given the following excerpt: … Question: …"). "The text says…" then becomes *correct* rather than hallucinatory, and inference matches the RAG deployment path already built (`rag_retriever.py`). Reading comprehension + extraction is squarely within 113M capacity; closed-book recall is not.
2. **Closed-book subset restricted to general principles:** keep only Q&A answerable from textbook knowledge (definitions, canonical formulas, qualitative mechanisms); drop paper-specific parameter questions from the closed-book split.
3. Replace the regex cleaner with a **teacher-model rewrite pass** (~17k short calls): rewrite responses as standalone, preserve markdown structure/newlines, clean *inside* `<think>`, and emit a `\boxed{}` final answer for numerical items. This simultaneously fixes D1–D3 and produces machine-verifiable RL targets.

### Phase 2 — Retrain SFT correctly (half a day on 2× A30)
- Document-level split; no mid-response packing splits (drop or pad the tail; or add block-diagonal masks via `flex_attention`/manual mask — optional given item lengths ~550 tokens);
- fix grad-accum normalization on partial groups;
- keep LR 2e-4 / cosine / early stopping — that part of the methodology was fine, but re-run the sweep on the fixed val before trusting a ranking.

### Phase 3 — Re-run GRPO for a goal it can actually achieve (a day)
1. Regenerate the RL dataset (fix `import math`; extract a single numeric `final_numerical_value` from the rewritten `\boxed{}` answers; verify schema matches `verifiers.py` with an assert at load time).
2. Harden the reward: concept matching on the post-`<think>` answer only (strip think), lemma-deduplicated concept lists, numeric reward on the boxed answer only (not `min` over all numbers), mask forced-EOS padding out of loss/KL, add completion-length + entropy logging.
3. Stability: KL controller or hard early-stop on rolling KL (the current run should arguably have stopped near step ~250); G=8; more than ~600 unique prompts; greedy-consistent objective (sample at the temperature you optimize, or add importance weights).
4. **Evaluation must be reward-independent:** exact-match accuracy on boxed numerical answers over the *held-out* split, plus SFT-val cross-entropy drift, plus a small LLM-as-judge factuality score on ~100 prompts. Never reuse the training reward as the benchmark.
5. Realistic target: GRPO on a 113M model can enforce format, brevity, echo-avoidance, and *grounded-context faithfulness* (with open-book prompts, a verifier can check that cited numbers appear in the provided context — a well-posed, hack-resistant reward). It will not create closed-book knowledge or mental arithmetic.

### Phase 4 — Structural options (pick per project goals)
- **Tool use:** teach `<python>…</python>` calculator calls inside `<think>` via SFT (the teacher model can rewrite the 5.4k numerical items into tool-call format), execute offline during RL rollouts, reward exact boxed answers. This is the only credible route to numerical correctness at 113M and is feasible on this hardware.
- **RAG-first inference:** make retrieval the default path; consider upgrading BM25 to a small dense retriever if retrieval quality limits it.
- **Scale, if the teaching narrative allows:** 2× A30 can full-finetune ~0.5–1.5B (bf16 + grad checkpointing; LoRA trivially). A Qwen2.5-0.5B/1.5B base would remove the fluency ceiling while reusing the entire SFT/GRPO pipeline. Staying from-scratch: the 823M-token corpus is the binding constraint — a bigger from-scratch model would be even more data-starved; multi-epoch pretraining (2–3 epochs) is the cheaper lever.

### Housekeeping
- Checkpoints embed AdamW state (1.36 GB each; 6 GRPO snapshots ≈ 8 GB) — save weights-only for snapshots.
- Delete or implement dead config keys (`clip_ratio`, `ppo_epochs`, scheduler…) so the YAML documents what the code does; fix the "Clipped Policy Gradient Loss" plot title and the hardcoded benchmark numbers in `plot_grpo_results.py`.
- README claims to correct: "16,388-token vocab" is right, but §9.2's +590% table, the scaling-law table, and the 2.1237 headline all need caveats or re-measurement after Phase 0.

---

## 6. Remediation log (what was actually changed)

All work below was executed on 2026-07-29 following this audit. The from-scratch 113M model was kept
(no backbone swap); every change is a correctness or measurement fix plus a data-quality rebuild.

### 6.1 Code changes

| File | Change | Fixes |
|---|---|---|
| `src/sft/format_sft_dataset.py` | Group pairs by `chunk_id`; assign whole chunks to val; carry `chunk_id` into processed records | S1 (100% document leakage) |
| `src/sft/train_sft.py` | Token-weighted val loss with `dist.all_reduce`; `inf` guard instead of silent NaN; grad-accum normalized by true group size; weights-only best checkpoints; new `--eval-only` mode | S2, S3, S5 |
| `src/sft/dataset.py` | Packing never splits a sample across block boundaries (flush + pad instead) | S4b |
| `src/sft/chat.py` | `no_repeat_ngram` 3→0, temperature 0.7→0.2, top-k 50→20, repetition_penalty 1.1→1.05; REPL honors CLI args; multi-turn history matches training template; base-ckpt fallback resizes to 16,388 with a warning | S6, S7 |
| `src/sft/clean_sft_data_v2.py` | **New.** Quality filter replacing the v1 rewriter: 2 grammar-preserving rewrites, then *drop* anything still referencing a phantom source (incl. inside `<think>`); domain filter; newlines preserved | D1, D2, D3 |
| `src/sft/run_data_scaling_sweep.py` | Full-size point resolved at runtime (no more mislabeled "12.6k (Full)"); points to `processed_v2`; documents the fixed-epoch confound | S8 |
| `src/rl/prepare_rl_dataset.py` | `import math` (the root cause); scientific-notation parsing; concepts exclude prompt words and prefix-duplicates; **hard schema self-check** after writing | G1, G2, G4 |
| `src/rl/verifiers.py` | Concept matching on post-`<think>` text only; number-spam cap; sci-notation parsing; `CleanLanguage` returns 0.0 (not +0.2) when clean; format verifier is task-type aware; 5 adversarial unit tests | G4 |
| `src/rl/train_grpo.py` | Load-time schema assertion; system turn in rollouts; loss/KL masked to actually-sampled tokens; per-sequence-then-group mean; on-policy sampling (T=1.0); without-replacement prompt schedule; adaptive KL + hard early stop; entropy and length logging | G5, G6, G7 |
| `src/rl/eval_rl.py` | Rewritten around **reward-independent** metrics (numerical accuracy @1%/@5%, `<think>` rate, contamination, echo, length) on the held-out split; training reward reported last, labeled | G3 |
| `src/rl/plot_grpo_results.py` | Benchmark chart reads real eval JSON instead of hardcoded numbers; panel (d) shows entropy + completion length instead of a mislabeled "clipped" loss; numerical sub-reward plotted | G3, G8 |
| `src/rl/diagnose_sft_vs_grpo.py` | System turn added to prompts; checkpoint paths updated | G7 |
| `configs/rl_grpo.yaml` | Removed all keys the code never read; documented the actual objective | G8 |

### 6.2 Data rebuild

```
17,289 raw pairs
  −  5,936  contaminated instruction (phantom source reference)
  −  4,316  contaminated response
  −  2,313  off-domain (OS history, medical negligence, brewing, urban studies)
  −     51  malformed <think> tags
  −      6  phantom data reference
  =  4,667  clean pairs   (27.0% retention; 1,113 grammar-preserving rewrites applied)
```

Chunk-level split → **4,434 train / 233 val**, from 3,111 / 156 disjoint source chunks.
**Verified: 0 chunks shared between train and val** (v1: 822/822 shared).

The 73% drop is intentional. A closed-book 113M model trained on answers that cite invisible tables
learns to fabricate citations; those pairs are worse than no data. Note the retained set is still
38% `<think>`-bearing and type-balanced (2,217 conceptual / 1,625 numerical / 592 summarization).

### 6.3 Measured results

**Old v1 checkpoint re-scored on the clean document-disjoint val set:** loss `1.6547` (ppl 5.23) —
lower than its own reported `2.1237`, which only confirms the two numbers measure different things
and that the v1 figure cannot be compared to anything.

**SFT v3** (4,434 clean pairs, document-disjoint val, all-reduced token-weighted loss):
best val loss **`2.1844`** at step 40 / epoch 2, from a 116-step 8-epoch run. Overfitting begins
early (train 1.22 vs val 2.32 by epoch 5) — expected with 4.4k samples, and now honestly visible
because the val set no longer contains siblings of the training questions.

**GRPO v2** — 200 steps from `checkpoints/sft_v3/sft_ckpt_best.pt`, G=8, lr 5e-6, live numerical
verifier (99.5% of numerical prompts carry a verified target, vs 0% in v1). Completed without
triggering the KL early-stop (max KL 0.0792 vs threshold 0.5). Reward −0.296 → −0.029 across
50-step segments, with R_num −0.284 → −0.194 and R_format +0.054 → +0.170; entropy (2.57 → 2.66)
and completion length (352 → 374) both stable, so neither entropy nor length collapsed.

**Held-out comparison** (177 document-disjoint prompts, reward-independent metrics, decoded at
temperature 0.3 / top-k 40 / repetition penalty 1.2): `<think>` rate on numerical prompts 46.2% →
**93.2%** (z = +7.82, significant); numerical accuracy @1% 15.4% → 12.0% and @5% 18.8% → 15.4%
(**down**, 18→14 and 22→18 of 117, z ≈ −0.7, not significant); echo 7.3% → 2.8% (z = −1.94,
borderline); contamination 0.0% for both.

The honest reading: **GRPO moved form, not knowledge — and did not improve numerical accuracy at
all.** That is the same qualitative conclusion v1 published, but v1 asserted it on the back of a
fabricated "+590%" produced by a dead verifier and a circular benchmark; here it is measured, with
significance tests printed by `eval_rl.py` on every run so the null results cannot quietly become
wins.

**A second measurement trap, found after the first pass.** An earlier version of this section
reported `<think>` 8.5% → 55.6% and a small *positive* numerical delta. Those numbers were taken at
repetition penalty 1.05, where this model emits **56% verbatim-repeated 5-grams** and often never
closes its `<think>` tag — the eval was scoring a decoding artifact. Sweeping the decoder over 4
prompts × 250 tokens:

| temperature | rp 1.05 | rp 1.15 | rp 1.25 |
|---|---|---|---|
| 0.2 | **0.562** | 0.162 | 0.000 |
| 0.5 | 0.105 | 0.005 | 0.000 |
| 0.7 | 0.039 | 0.002 | 0.000 |
| 0.9 | 0.007 | 0.003 | 0.000 |

(values = fraction of repeated 5-grams; lower is better). Raising the penalty to 1.2 removed the
loops entirely, roughly doubled measured accuracy for *both* models (SFT 7.7% → 15.4% @1%), and
flipped the sign of the SFT→GRPO numerical delta. v1 had papered over the same failure with
`no_repeat_ngram=3`, which banned repeated trigrams and produced the `M0_code` / `N·mi` word-salad
instead. `chat.py` and `eval_rl.py` now default to temperature 0.3 / top-k 40 / rp 1.2.

The generalizable lesson: **at 113M, a decoding-parameter mistake can be larger than the training
effect being measured.** Fix and pin the decoder before comparing checkpoints.

### 6.4 Known remaining limitations

- **Retention is low (27%).** An LLM rewrite pass would recover much of the dropped 73%
  by restating source-dependent answers as standalone knowledge instead of discarding them. No API
  key is configured in this environment, so the deterministic filter was used instead.
- **Packing still uses a plain causal mask**, so samples within a 4096-token block can attend to each
  other. Mid-response splitting is fixed; block-diagonal masking is not implemented.
- **4.4k SFT pairs is small.** The honest fix is more clean data (regeneration with a
  standalone-answer prompt), not more epochs.
- **The base model is still 823M pretrain tokens at 113M params** (≈7 tokens/param, below
  compute-optimal). This bounds everything downstream.

### 6.5 Additional findings surfaced during remediation

Three issues that the original audit did not catch, found while re-running the pipeline:

1. **~15% of the corpus is off-domain.** The pretraining crawl pulled in unrelated open-access
   papers, so the teacher generated Q&A about Amiga operating-system history, medical-negligence
   litigation, Cusco urban studies, and *Brettanomyces* brewing microbiology. On a 113M domain model
   this is pure capacity waste. `clean_sft_data_v2.py` now applies a domain filter (2,313 pairs
   dropped).

2. **Greedy decoding degenerates.** At `top_k=1` the model enters repetition loops
   (`(Δt / Δt) * (Δt / Δt) * …`, `G = A * Ω * Ω * Ω * …`) that never close the `<think>` block. A
   "deterministic" greedy evaluation therefore measures a decoding artifact: it reported a 2.6%
   `<think>` rate and 51.4% echo rate versus 8.5% / 47.5% at the deployment configuration. Evaluation
   now uses temperature 0.2 / top-k 20 with a fixed seed.

3. **`lr = 1e-6` leaves the policy frozen at this batch size.** A 76-step trial at the v1 learning
   rate reached a per-token KL of only ~0.001 against a 0.05 target, with the adaptive β pinned at its
   floor — i.e. essentially no policy movement. This reframes the v1 run: its KL climb to 0.3+ (and
   the 8.2 spike) was driven by gradients on never-sampled EOS padding tokens, not by learning. With
   padding correctly masked out, the learning rate was raised to 5e-6.

4. **Checkpoint storage.** 39.7 GB of checkpoints carry redundant AdamW state; stripping it
   (`python -m src.sft.checkpoints strip checkpoints/ --apply`) recovers **29.1 GB**. Left as a
   user decision — the script defaults to a dry run and keeps `.bak` copies.

### 6.6 Qualitative confirmation

Same prompt, both v2 checkpoints, deployment sampling:

> *"A fault of area 400 km², slip 1.5 m, shear modulus 30 GPa. Calculate the seismic moment M0."*

**SFT v2** — opens `<think>`, states the correct formula, then loops forever and never closes the tag
or produces an answer:
> `<think>We need to compute M0 = μ * A * D … M0 = 3e10 * 5e8 * 5e8 = 3e10 * 5e8 * 5e8 = 3e10 * 5e8 * 5e8 = …`

**GRPO v2** — closes `<think>` and emits a distinct final-answer section:
> `<think>… μ = 30 GPa = 30 × 10^9 Pa, A = 200 km^2 …</think>The seismic moment M0 is calculated as M0 = μ * A * D …`

GRPO fixed the *structure* (the block terminates, an answer section exists) and did not fix the
*content* (it misreads 400 km² as 200 km², substitutes area for slip, and the arithmetic is wrong).
That is precisely what the held-out numbers say: `<think>` rate +47 pp and significant, numerical
accuracy +1.7 pp and indistinguishable from noise.

---

## 7. SFT data-scaling law (2026-07-30)

> **SUPERSEDED BY §10.2** — this curve varied corpus size and mixture together.

Three models trained identically (8 epochs, lr 2e-4, early stopping) on nested
subsets of the same 9.4k corpus, all scored on the **same** 281-pair
hash-pinned held-out test set:

| train pairs | test loss | perplexity |
|---|---|---|
| 2,100 | 2.2590 | 9.57 |
| 4,216 | 2.1869 | 8.91 |
| 8,507 | **2.1312** | **8.42** |

Log-linear fit, residuals within ±0.006 nats:

    loss = 2.9549 - 0.0633 * log2(N)        # 0.0633 nats per doubling

Extrapolation (upper bound — the trend must flatten eventually):

| pairs | loss | ppl |
|---|---|---|
| 13,000 (planned +4.5k fundamentals) | 2.090 | 8.08 |
| 20,000 | 2.050 | 7.77 |
| 40,000 | 1.987 | 7.29 |
| ~104,000 | 1.900 | 6.69 |

**Implication.** The slope is shallow. The planned fundamentals expansion buys
about **0.039 nats (~4% perplexity)** — worth the spend, but not a change in
kind. Reaching loss 1.90 would take roughly **12x** the current corpus.

**Why the base model is the real ceiling.** The pretrained checkpoint stopped
at iter 8,000 with val loss 2.6645, still descending (2.7967 -> 2.6645 over the
final doubling of compute, ~0.13 nats — *twice* the SFT-data slope). But it had
already consumed ~3.8 epochs of the 823M-token corpus, at the edge of the
~4-epoch useful-repeat budget, and sits at ~7.3 tokens/parameter versus the
~20:1 compute-optimal ratio. So the base is **data-limited, not
compute-limited**: more pretraining epochs on the same crawl will not help much;
more pretraining *tokens* would.

**Practical reading.** Paper-derived SFT data is now exhausted (17,289 raw ->
9,421 usable after cleaning, rewriting and deduplication, from only 5,860 source
chunks). The synthesized fundamentals path is the one that is not bounded by the
crawl, but at 0.0633 nats/doubling it would need tens of thousands of pairs to
matter, and it cannot repair coherence failures that originate in an
undertrained 113M base.

---

## 8. Historical-earthquake data, and a leak in my own split design (2026-07-30)

A third generator (`tools/gen_fundamentals/generate_earthquakes.py`, 109 events)
added 4,409 pairs; the expanded 63-topic curriculum added 4,034. Merged corpus:
**15,659 pairs** (4,667 paper + 2,549 rewritten + 4,034 fundamentals + 4,409
earthquakes), 14,068 train.

### 8.1 The leak

The first evaluation showed earthquake perplexity dropping 10.41 -> 4.75, which
was too good. Cause: `chunk_id` was computed as `md5(topic::variant)`, so every
*variant* became its own chunk and the splitter scattered variants of the same
event across train and test. Measured: **100% of test events and topics also
appeared in training under a different variant** — precisely the sibling leak
the chunk-level split exists to prevent, reintroduced by the generator.

Fixed by making `chunk_id = md5(identity)` and keying resume on
`(chunk_id, variant)` separately, in both generators. Existing data was remapped
rather than regenerated (710 fundamentals records -> 63 chunks; 860 earthquake
records -> 109 chunks). After the fix, 0 of the held-out identities appear in
training.

### 8.2 Leak-free results

Both models scored on identical held-out data. The paper subset is clean for
both; the fundamentals subset excludes the 100 pairs the 8.5k model had
memorised from the original 33-topic curriculum.

| held-out subset | 8.5k model | 14.1k model | delta |
|---|---|---|---|
| paper (356 pairs, 48 blocks) | 2.1671 (ppl 8.73) | **2.1566** (8.64) | −0.011 |
| earthquakes (203 pairs, 6 unseen events) | 2.1782 (8.83) | **1.7022** (5.49) | **−0.476** |
| fundamentals (149 pairs, 4 unseen topics) | **2.3725** (10.72) | 2.3872 (10.88) | +0.015 |

The earthquake gain is large and survives the leak fix — the model generalises
to events it never saw, not merely memorising the catalogue. (The pre-fix figure
was −0.784, so roughly 40% of the apparent gain had been leakage.) Paper-domain
modelling improves slightly; the fundamentals subset is flat within noise.

> [!CAUTION]
> The earthquake and fundamentals subsets are small — 6 and 4 held-out
> identities, 9 and 11 packed blocks. The earthquake effect is large enough to
> be credible; the fundamentals difference is not distinguishable from noise at
> this sample size. Reserving more identities for test would tighten this, at
> the cost of training data.

### 8.3 Why the aggregate number misleads

Scored on the original `common_test.jsonl`, the 14.1k model looked *worse*
(2.1390 vs 2.1312). That file is 100% paper-derived, built before the new
sources existed, so it cannot see what the new data bought and it penalises the
distribution shift. Always score per source when the training mixture changes.

---

## 9. Sequence packing corrupts small-effect evaluation (2026-07-30)

Two nested paper-only test sets gave **opposite signs** for the same two models:

| test file | 8.5k | 14.1k | verdict |
|---|---|---|---|
| `common_test.jsonl` (281 pairs) | 2.1312 | 2.1390 | 14.1k worse |
| `test_paper.jsonl` (356 pairs, superset) | 2.1671 | 2.1566 | 14.1k better |

Both are cleanly held out from both models, same type mix, same mean length,
and the smaller file is a strict subset of the larger. The contradiction is an
artefact of **packed evaluation**:

1. Pairs are concatenated into 4096-token blocks under a plain causal mask, so
   each pair is conditioned on whatever unrelated pairs precede it in its
   block. Change the file, change the packing, change the number.
2. The loss is token-weighted over the block, so long pairs dominate.

Scoring each pair in isolation removes both problems and permits a **paired**
test (same pair, both models, so per-pair difficulty cancels).
`src/sft/compare_checkpoints.py --unpacked` does this.

### Unpacked, paired results (14.1k − 8.5k)

| held-out subset | n | mean diff | 95% bootstrap CI | t | verdict |
|---|---|---|---|---|---|
| paper (`common_test`) | 281 | **−0.0100** | [−0.0154, −0.0045] | −3.66 | significant |
| paper (`test_paper`) | 356 | **−0.0092** | [−0.0137, −0.0045] | −3.83 | significant |
| earthquakes (6 unseen events) | 203 | **−0.5334** | [−0.5626, −0.5009] | −34.1 | significant |
| fundamentals (4 unseen topics) | 149 | +0.0039 | [−0.0136, +0.0221] | +0.43 | not significant |

The two paper files now **agree**: the 14.1k corpus is consistently, if slightly,
better on paper-domain text. The earthquake gain is enormous and unambiguous
(ppl 9.73 → 5.70 on events never trained on). Fundamentals is flat.

### Practical rules this establishes

- **Packed loss is fine for training-time monitoring, not for model comparison.**
  It is optimistically biased (extra context lowers loss: 2.1312 packed vs
  2.2799 unpacked on the same file and model) and its noise exceeds a 0.01-nat
  effect.
- **Always pair.** Comparing two means of independently noisy per-pair losses
  wastes most of the statistical power; the paired difference has far smaller
  variance because item difficulty cancels.
- Earlier packed numbers in this report (§7, §8) are directionally right where
  the effect is large, and unreliable where it is ~0.01 nats. The §7 scaling
  slope is essentially unchanged when recomputed unpacked (−0.064 vs −0.063
  nats/doubling), because the packing bias is roughly constant across models on
  a fixed file — it is the *cross-file* and *small-effect* comparisons that break.

---

## 10. Two follow-up questions, answered empirically (2026-07-30)

### 10.1 Is the train/val gap a sign that 400 steps is too many, or the LR too large?

Neither, exactly: it is **too little data**, and the run budget is simply wasteful.

An 8-epoch run at three learning rates on the same 14.1k corpus:

| lr | best val | at step (epoch) | final train | final val |
|---|---|---|---|---|
| 5e-5 | 2.1391 | 160 (4) | 1.511 | 2.199 |
| **1e-4** | **2.1296** | 120 (3) | 1.158 | 2.328 |
| 2e-4 | 2.1441 | 80 (2) | 0.701 | 2.576 |

Read three ways:

- **The LR is not "too large" in the divergence sense.** All three converge
  smoothly to a clean minimum; none is unstable. The spread between the best
  values is 0.015 nats — smaller than the paper-domain effect of doubling the
  corpus.
- **1e-4 is mildly better than 2e-4** (−0.015 nats) and reaches its minimum
  later, at epoch 3 instead of 2, which also leaves a less brutal train/val
  gap. Worth adopting, but it is a refinement, not a fix.
- **Lowering the LR does not remove the gap.** Even at 5e-5 the final train
  loss is 1.51 against val 2.20. The model memorises whatever it is given;
  more optimizer care cannot manufacture generalisation from 14k examples.

On the step budget: 8 epochs is ~3-4x more than needed. Early stopping already
protects the checkpoint (best is saved), so the cost is wasted compute, not a
worse model. Training 3-4 epochs at lr 1e-4 would get the same checkpoint in
half the time.

### 10.2 Why weren't the small scaling points composition-matched?

They should have been — the earlier curve was confounded. Measured composition
of the original points:

| point | paper | fundamentals | earthquakes |
|---|---|---|---|
| 4.2k | 100% | — | — |
| 8.5k | 76% | 24% | — |
| 14.1k | 46% | 25% | 28% |

Size and mixture varied together, so the "scaling law" was the superposition of
two effects. Redone with **stratified chunk-level subsamples of the 14.1k
corpus**, holding the mixture at 46/25/28 (±1%) at every size, all scored
unpacked on the same pinned test set:

| pairs | test loss | ppl |
|---|---|---|
| 2,060 | 2.5005 | 12.19 |
| 4,069 | 2.4347 | 11.41 |
| 8,034 | 2.3654 | 10.65 |
| 14,068 | 2.2700 | 9.68 |

    loss = 3.4041 - 0.0813 * log2(N)      residuals within ±0.016

**The clean slope is 0.081 nats/doubling, 28% steeper than the 0.063 measured
from the confounded series.** The old curve understated the value of data
because its low-size points were pure paper — the easier, in-distribution
subset for this test set — which flattened the apparent trend.

Revised extrapolation: 20k -> 2.242, 40k -> 2.161, 100k -> 2.053. Reaching loss
2.0 needs roughly 158k pairs. Still far out of reach, but the return on the
next doubling is meaningfully better than previously reported.

### 10.3 The per-domain panel had to be rebuilt too

The original panel (b) of `sft_data_scaling` compared the 8.5k model against
14.1k. Once panel (a) became a fixed-mixture series, that comparison was both
inconsistent with it (two different "8k" models in one figure) and close to
tautological: the 8.5k corpus contained **no earthquake data at all**, so its
poor earthquake score measured the absence of a corpus, not a scaling effect.

Rebuilt against `strat_8k` — same 46/25/28 mixture, half the size — the panel
now decomposes the panel-(a) scaling step by domain:

| held-out subset | 8.0k | 14.1k | paired diff | 95% CI | t |
|---|---|---|---|---|---|
| paper | 2.4105 | 2.3136 | **−0.0969** | [−0.1033, −0.0908] | −30.1 |
| earthquakes | 1.8951 | 1.7413 | **−0.1538** | [−0.1728, −0.1353] | −16.1 |
| fundamentals | 2.5941 | 2.4689 | **−0.1252** | [−0.1371, −0.1133] | −19.8 |

All three improve significantly, by a fairly uniform 0.10-0.15 nats. That is a
different and more useful claim than the old panel: **scaling the corpus at
fixed mixture lifts every domain**, rather than one corpus rescuing its own
domain while the others stay flat. The earlier "+0.004, n.s." on fundamentals
was a mixture artefact, not evidence that fundamentals data had stopped paying.

---

## 11. Is the late-epoch gap really overfitting? (2026-07-30)

The training-dynamics figure shows train loss falling to 0.70 while validation
climbs to 2.58 — a 1.9-nat gap. Two questions follow: is the held-out
degradation real, and does it matter?

### 11.1 Real in likelihood

`sft_ckpt_best.pt` (step 80) vs `sft_ckpt_last.pt` (step 336), scored unpacked
and paired on the 281-pair held-out test set:

| | test loss | ppl |
|---|---|---|
| best (step 80) | **2.2700** | 9.68 |
| last (step 336) | 2.7798 | 16.12 |

paired diff **+0.5098 nats**, 95% CI [+0.4794, +0.5386], t = +33.5. The
held-out likelihood degradation is unambiguous.

### 11.2 Not real in behaviour

The same two checkpoints, generating on 226 held-out prompts at the deployment
decoding configuration:

| metric | best (step 80) | last (step 336) | significance |
|---|---|---|---|
| Numerical accuracy @1% | 7.2% | **10.2%** | z = +0.97, n.s. |
| Numerical accuracy @5% | 10.2% | **13.3%** | z = +0.85, n.s. |
| `<think>` rate (numerical) | 61.4% | **74.1%** | **z = +2.47, significant** |
| Contamination rate | 0.0% | 0.0% | — |
| Prompt-echo rate | 18.6% | 19.9% | z = +0.36, n.s. |
| Mean completion length | 308 | 309 | — |

The late checkpoint is **not worse on anything measured, and significantly
better on format compliance**. Its numerical accuracy is nominally higher too,
though not significantly.

### 11.3 Why both are true

Cross-entropy scores the probability the model assigns to **one specific
reference wording**. As training continues the model sharpens onto its own
preferred phrasing, which lowers the likelihood of the reference's exact tokens
without making its own generations worse. Held-out NLL and generation quality
are measuring different things, and here they diverge by half a nat.

This confirms an earlier observation that was dismissed as noise: the v3 "last"
checkpoint formatted better than the val-optimal "best" checkpoint.

### 11.4 Consequences

- **"Epochs after ~200 are useless" is not quite right.** They cost 0.51 nats
  of held-out likelihood and buy a significant improvement in `<think>`
  compliance. Useless for likelihood, mildly useful for behaviour.
- **Selecting checkpoints on validation loss is the wrong criterion here.**
  It picks step 80; behaviour prefers step 336. Checkpoint selection should use
  the reward-independent behavioural metrics from `src/rl/eval_rl.py`.
- **One risk the behavioural metrics do not capture:** at train loss 0.70 the
  late checkpoint has substantially memorised the training set and may
  reproduce training answers verbatim for near-duplicate questions. That is not
  visible in these aggregate numbers and would need a targeted check before
  shipping the late checkpoint.
- The lr sweep in §10.1 remains valid, but note its ranking was by validation
  loss; a behaviour-based ranking of those three runs has not been done.

---

## 12. Behaviour-based checkpoint selection and the V3 GRPO run (2026-07-31)

Acting on §11, checkpoint selection was moved off validation loss, the verifier
fixes were exercised in a fresh GRPO run, and both figures in `assets/` were
regenerated from that run's real logs.

### 12.1 SFT selection (`src/sft/select_checkpoint.py`)

The rule is declared in the source before any numbers were looked at: rank on
`numerical_acc_5pct`, break ties within 2 pp on `numerical_think_rate`, and
hard-gate on contamination / echo / non-empty output. On `checkpoints/sft`:

| checkpoint | num@5% | num@1% | `<think>` | contam | echo | len | gate |
|---|---|---|---|---|---|---|---|
| `sft_ckpt_best.pt` (ep3, val-optimal) | 12.0% | 9.6% | 57.2% | 0.0% | 21.0% | 321 | PASS |
| `sft_ckpt_epoch4.pt` | **13.9%** | 9.6% | 63.3% | 0.0% | 21.5% | 334 | PASS |
| `sft_ckpt_epoch8.pt` | 12.7% | 9.0% | **72.9%** | 0.0% | 25.8% | 315 | PASS |

Selected: **epoch 8** → `checkpoints/sft/selected.pt`. Validation loss would
have picked epoch 3, which has the *worst* `<think>` rate of the three — §11
reproduced on an independent run.

Stated plainly: epoch 4 has the highest raw num@5% (13.9%), and the tie-break
is what moved the pick to epoch 8. All three sit inside the 2 pp band, which at
n=166 is noise; the choice rests on the tie-break, not on an accuracy win.

### 12.2 GRPO V3 run

200/200 steps from `sft_v6/selected.pt`, ~27 s/step, no KL early-stop.

| metric | steps 1–50 | 51–100 | 101–150 | 151–200 |
|---|---|---|---|---|
| mean reward | −0.463 | **−0.253** | −0.365 | −0.356 |
| KL | 0.037 | 0.090 | 0.064 | 0.042 |
| β (adaptive) | 0.006 | 0.018 | 0.192 | 0.192 |
| entropy | 2.37 | 2.58 | 2.47 | 2.42 |
| stop rate | 0.640 | **0.883** | 0.673 | 0.670 |

Two things worth recording:

- **The adaptive KL controller demonstrably worked**, unlike V1/V2. KL peaked
  at 0.142, crossed the 2×target deadband, drove β from its 0.005 floor to
  0.192, and KL was pulled back to 0.042. Between 0.025 and 0.10 β holds
  constant by design — that is the deadband, not a stuck controller.
- **The TerminationVerifier worked.** Stop rate averaged 0.64–0.88 against
  27.5% in V2.

### 12.3 The run peaks near step 100, and both metrics agree

> [!WARNING]
> **The numbers in §12.3 and §12.4 were measured with a broken scoring rule and
> are superseded by §13.** The accuracy metric's floor exceeded the model, and
> the echo threshold fired on genuine answers. Corrected values are in §13.4;
> the *conclusions* about step 100 vs 200 survive and in fact strengthen, but
> the prompt-echo result does not. Left in place unedited so the correction is
> auditable.

Reward peaks at step ~105 and then regresses. Evaluating both saved checkpoints
reward-independently on the hash-pinned held-out set (166 numerical + 60
conceptual, deployment decoding):

| metric | SFT | GRPO @100 | GRPO @200 |
|---|---|---|---|
| numerical acc @1% | 9.0% | 9.0% | 10.8% |
| numerical acc @5% | 12.7% | 13.3% | 13.3% |
| `<think>` rate | 72.9% | **98.2%** (z=+6.55) | 91.6% (z=+4.45) |
| contamination | 0.0% | 0.4% | 0.0% |
| prompt-echo | 21.2% | **11.1%** (z=−2.94) | 16.4% (n.s.) |

Applying the §12.1 rule unchanged: num@5% ties exactly at 13.3%, so the
tie-break on `<think>` rate selects **step 100** →
`checkpoints/grpo/selected.pt`. Step 100 also cuts prompt-echo
significantly where step 200 does not. Panel (d) of the training figure shows
the same peak independently, from the training log rather than the eval.

### 12.4 What GRPO did and did not buy

**It did not improve numerical accuracy.** 12.7% → 13.3% at 5% tolerance is
Δ=+0.006, z=+0.16 — indistinguishable from noise, as is the @1% change. Any
claim that this RL stage made the model better at arithmetic is unsupported by
these numbers.

**It did improve output discipline, substantially and significantly.**
`<think>` compliance 72.9% → 98.2% (z=+6.55) and prompt-echo 21.2% → 11.1%
(z=−2.94). Completion length fell 308 → 251 tokens with no entropy collapse.

The composite training reward improved −0.509 → −0.245, but that is the
objective being optimised and is reported only as a reference row; it is not
evidence of capability gain.

### 12.5 Figure-pipeline defects fixed while regenerating

- `plot_grpo_results.py` read the *previous* run directory (`grpo_v2`) — it would have plotted the
  **old** run. Now defaults to v3 and takes `--log` / `--eval-json`.
- Axis limits were hardcoded to V2's ranges and would have silently clipped V3
  data. Now data-driven.
- `compute_rolling_mean` back-filled the leading 19 points with a constant,
  drawing a flat line across the opening of every panel that read as "nothing
  happened for 20 steps". Replaced with an expanding-window mean.
- `eval_rl.py` wrote results with no provenance, so a stale file was
  indistinguishable from a current one — and the pre-existing
  `docs/eval_rl_results.json` was in fact from the V2 run. It now records
  checkpoints, val file, and decoding params, and the plotter warns loudly on a
  run-directory mismatch.
- Panel (c) now pairs KL with entropy (drift vs collapse) and panel (d) pairs
  completion length with the new `stop_rate`.

### 12.6 Open

- Step 200 is kept, not deleted; the last 100 steps are a measured regression on
  every reward-independent metric except num@1% (itself not significant).
- The memorisation risk flagged in §11.4 is still unchecked for the selected SFT
  checkpoint.
- Numerical accuracy remains ~13%. If that is the target, the lever is data and
  the base model, not more RL steps against these verifiers.

---

## 13. Are the metrics representative, and is the reward being hacked? (2026-07-31)

> **CURRENT for the metric and reward fixes; SUPERSEDED IN PART BY §14**, which shows
> the headline was measured on prompts that had already chosen the checkpoint.

Two questions asked directly. Answering them empirically rather than by
argument found that the primary capability metric was invalid and that the
reward had one live hacking channel. Both are now fixed; several numbers in
§12 are superseded.

### 13.1 The numerical-accuracy metric could not demonstrate capability

`eval_rl` scored `min(rel_err)` over the **first 8** numbers in the answer, so
every additional number was another lottery ticket. Measured on the 166
held-out numerical items (`src/analysis/metric_validity.py`):

| k numbers sprayed | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| acc@5% of a question-blind policy | 2.0% | 4.0% | 5.8% | 9.3% | **14.3%** |

A policy that knows only the empirical answer distribution and nothing about
the question scores **14.3%** — above the GRPO model's reported 13.3% and the
SFT model's 12.7%. The metric had no demonstrated dynamic range at the
operating point.

The ceiling was also wrong: reference answers scored only **68.1%** against
their own `final_numerical_value`, so ~32% of items were unwinnable by
construction.

### 13.2 Fix: score the answer tail

Restricting scoring to the last 3 numbers improves *both* ends, because the
stated answer sits near the end and "first 8" often never reached it:

| rule | ceiling | floor | usable range |
|---|---|---|---|
| min over first 8 (old) | 68.1% | 14.3% | 53.8% |
| **last 3 (`ANSWER_TAIL_NUMBERS`)** | **86.1%** | **5.9%** | **80.2%** |
| last 1 | 58.4% | 1.9% | 56.5% |

The old anti-spam cap (`reject if >8 numbers`) then became redundant and
harmful: tail scoring already bounds an attacker to 3 chances regardless of how
many numbers they emit (a 40-number spray succeeds 5.7% of the time either
way), while the cap fired on **40.4%** of genuine reference answers, which
carry a median of 6 numbers and a p90 of 34. Removing it lifted mean `r_num` on
the ground truth from +0.274 to +0.723.

### 13.3 One live reward-hacking channel, and a reward that punished the truth

`src/analysis/reward_hacking_probe.py` scores eight hand-built degenerate
completions against the real `CompositeRewardEngine` on all 166 items. Six were
decisively rejected — number spray −0.563, keyword stuffing in `<think>` −0.918,
empty think block −0.918 — confirming the earlier hardening works. But:

| probe | vs genuine reference |
|---|---|
| verbose filler + correct answer | **+0.139 (exploit)** |

`EchoPenaltyVerifier` divided overlap by the **candidate's** n-gram count, so
appending filler inflated the denominator and walked the ratio back under
threshold without removing any copied text. The same mis-specification
penalised genuine reference answers by **−0.428** on average for legitimately
restating the question.

Fixed by normalising against the **prompt** — the construct actually of interest
and immune to padding (a literal echo scores 1.000 with or without filler).
Thresholds recalibrated on 808 genuine answers (max 0.760): 0.85 penalises
0.00%, 0.70 penalises 0.62%. Echo penalty on the ground truth: −0.428 → −0.006.
Padding now ties the genuine answer rather than beating it — no gradient toward
it. Two regression tests were added to `verifiers.py` (adaptive 40-number spray,
padding dilution), bringing the adversarial suite to 8.

> [!NOTE]
> `checkpoints/grpo` was **trained with the pre-fix echo verifier**. The fix
> changes future runs; it does not invalidate v3, whose completion length fell
> (308 → 251) rather than rising, indicating the padding channel was not being
> exploited in practice.

### 13.4 Corrected results (supersedes §12.3 / §12.4)

Re-measured end to end with the corrected scoring:

| metric | SFT | GRPO @100 | GRPO @200 |
|---|---|---|---|
| numerical acc @5% | 9.6% | **10.2%** | 9.6% |
| numerical acc @1% | 7.8% | 6.6% | 8.4% |
| `<think>` rate | 72.9% | **98.2%** (z=+6.55) | 91.6% (z=+4.45) |
| contamination | 0.0% | 0.4% | 0.0% |
| prompt-echo | 0.0% | 0.0% | 0.0% |
| mean completion tokens | 308 | 251 | 274 |

Three claims change:

- **Accuracy is lower than reported** (13.3% → 10.2%): the old figure was
  inflated by the lottery-ticket effect.
- **The model clears the floor, but only just** — 10.2% against a 5.9% floor
  and an 86.1% ceiling, i.e. **5.5% of the way through the usable range**. It is
  a real but weak signal; small deltas here remain untrustworthy.
- **The prompt-echo win is gone.** The previously significant 21.2% → 11.1%
  (z=−2.94) was an artifact of the miscalibrated threshold, which also fired on
  11% of genuine answers. Under a correct detector no model echoes.

Checkpoint selection is unaffected and in fact better justified: step 100 now
wins on the **primary** metric outright (10.2% vs 9.6%) rather than only on the
tie-break, and still leads on `<think>` rate.

**Net conclusion.** After correcting the measurement, GRPO's only surviving
significant effect is `<think>` format compliance — the metric that overlaps
most with its own reward.

### 13.5 "Reward-independent" was the wrong term

The evaluation shares constructs with the reward:

| eval metric | reward counterpart | independent? |
|---|---|---|
| numerical acc @1/5% | `NumericalVerifier`, same field, same tolerances | **no** |
| `<think>` rate | `FormatVerifier`, same tags | **no** |
| contamination | `CleanLanguageVerifier` | no |
| prompt-echo | `EchoPenaltyVerifier` | no |

Every headline metric is the same construct measured on held-out prompts — a
generalisation check, not construct independence. That is still far better than
v1's circular benchmark (which scored the *composite training reward* on
training-distribution prompts), but the distinction must be stated.

### 13.6 What remains open

- **No metric covers conceptual factuality**, which is 642 of 808 held-out
  prompts (~79%). The largest slice of real usage has nothing measuring whether
  the model is right. An LLM-as-judge pass is the obvious gap-filler.
- **No construct-independent metric exists at all** now that echo reads zero.
- The probe scores `r_term = 0` because hand-built completions lack
  `<|im_end|>`, so it understates the TerminationVerifier's counterbalancing
  effect. Re-running with EOS included would tighten §13.3.
- Model answers' actual number counts are unmeasured (only 3 completions are
  stored per run), so the floor comparison assumes an 8-number answer.

---

## 14. Capability tiering, statistical power, and the split contract (2026-08-08)

> **CURRENT AS OF 2026-08-08.**

§13 fixed the metric. This section answers the next question — *what is the
corrected 10.2% actually made of, and is it measured on prompts that were
allowed to produce it* — and adds the accounting the earlier sections got wrong.

### 14.1 The headline was measured on prompts that had already voted

`select_checkpoint` and `eval_rl` both defaulted to the **full 808-prompt
held-out pool**. The same prompts chose GRPO step 100 over step 200 and then
produced the reported delta. The pre-declared selection rule (§12.2) prevents
cherry-picking; it does nothing about this.

`src/sft/splits.py` partitions the pool by `md5("seis-final-test::" + chunk_id)`
into a **dev** half (selection, thresholds, decoder, metric calibration) and a
frozen **test** half (reported once). Chunk-keyed, so siblings never straddle
the boundary; hash-pinned, so it survives corpus rebuilds.

```
split contract for data/rl/test.jsonl
  dev     463 prompts    76 numerical   115 chunks
  test    345 prompts    90 numerical   100 chunks
  invariant: dev_chunks.isdisjoint(test_chunks) -> OK
```

Re-deriving the SFT→GRPO numerical delta on each half from the stored per-item
outcomes prices the contamination directly:

| half | n | Δ acc@5% | 95% CI (paired bootstrap) |
|---|---|---|---|
| dev (selection touched it) | 76 | **+3.0%** | [+0.0%, +6.2%] |
| **test (frozen)** | 90 | **+0.8%** | [−3.1%, +4.7%] |

Same models, same metric, same decoder; the dev half reads nearly 4× larger.
The test row is the reportable one. `<think>` compliance is unaffected by the
split (73.1% → 97.8% on test), which is consistent with an effect far larger
than the selection artifact.

**Scope of this partition.** It is *retrospective*. The 808-prompt pool had
already been used for checkpoint selection, aggregate evaluation, metric
development, threshold calibration, and the choice of which analyses to run, so
hash-pinning freezes the split from 2026-08-08 onward but cannot make the test
half historically untouched. The 3.0-vs-0.8 gap is therefore **consistent with**
selection bias rather than a clean measurement of it: some of it is ordinary
heterogeneity between two 80-odd-item halves. Call this a retrospective audit
split, not a prospective final test. A clean final claim needs a new test set
built from previously unused source identities, run once, after freezing code,
parser, decoder, thresholds, candidates, and selection rule.

### 14.2 Paired outcomes, pass@k, and what this benchmark can resolve

`eval_rl.py` now stores **per-item outcomes**, which is what makes pairing
possible at all, and adds `--num-samples/-k`, pass@k, McNemar on discordant
pairs, paired bootstrap CIs, and an explicit power line:

```
[power] at n=166, k=4 sample(s)/prompt, baseline 7.1%: MDE ~ 10.0% at 80% power.
        A null result below that threshold means UNDERPOWERED, not 'no effect'.
```

At 166 numerical prompts (90 on the frozen half) this benchmark reliably detects
only large changes. Every null result in this document is therefore *not
supported*, never *disproven*. Reaching a 5-point MDE needs ~470 paired prompts;
2 points needs ~2,900.

pass@k separates "can sometimes do it" from "does it reliably", and carries its
own higher chance floor: must-compute items land **13.6%** of the time given
four attempts, against a **20.3% pass@4 floor** — still below chance.

### 14.3 Where the 10.2% comes from

Splitting the numerical set by whether the answer must be *derived* rather than
*retrieved* from the prompt:

| tier | n | chance floor | SFT | GRPO |
|---|---|---|---|---|
| answer already present in the prompt | 34 | 7.2% | 20.6% | **29.4%** |
| **must be computed** | 132 | 5.5% | 3.6% | **3.6%** |

(34 × 29.4% + 132 × 3.6%) / 166 = 8.9% — essentially the whole aggregate comes
from the retrievable fifth. Every derivation-depth tier from 6 steps to 20+ sits
at or below its own floor. The single aggregate number was hiding the only thing
worth knowing.

Held-out numerical questions require a **median of 11 derivation steps**.

### 14.4 Phase-0 gate: calculator tool use, tested and abandoned

Before building tool-use plumbing, a direct test: extract the arithmetic the
model writes inside its own `<think>` and execute it with a restricted AST
evaluator (`src/rl/safe_eval.py` — no `eval()`, 11 hostile inputs blocked in its
self-test).

| condition | acc@5% |
|---|---|
| model's stated answer | 3.6% |
| **model's own arithmetic, executed** | **1.3%** |
| blind floor | 1.8% |
| instrument ceiling (ground-truth expressions executed) | 50.8% |

Executing the model's expressions is *worse* than its stated answer and below
the floor, against an instrument that recovers half of the ground truth. The
expressions are noise, so a calculator has nothing correct to execute — the
bottleneck is formula recall, not arithmetic. Tool use was stopped here, at a
cost of ~40 GPU-minutes.

### 14.5 Prompting baseline

Explicitly instructing the SFT model to use `<think>` tags moves compliance
**72.3% → 73.5%**; RL reached 98.2%. Under the tested prompt template and pinned
decoder, prompting does not reproduce the format gain.

This supports "prompting was insufficient", **not** "RL was necessary". The
missing rungs of the ladder are format-targeted SFT and — most importantly —
**rejection-sampling fine-tuning** (sample k, keep verifier-passing completions,
fine-tune on them at the same rollout budget). Neither was run.

### 14.6 One worked group, and where the gradient actually came from

`src/analysis/worked_group.py` scores a real G=8 group from the shipped
checkpoint with the real verifiers. On the sampled group: 1 completion correct
(r_num +0.5, advantage +2.12), 1 format-broken (r_format −0.5, advantage −1.57),
mean R = −0.4375, std = 0.6781 — and **5 of 7 reward components had within-group
std of exactly 0.000**, contributing nothing to that update. Only within-group
spread creates gradient; a constant column cancels in the baseline exactly like a
dead verifier does.

What the run log supports (batches are single-type, so the two rows partition
the run):

| step type | steps | what the reward asks for |
|---|---|---|
| numerical | **77 / 200 (38.5%)** | the physics, then the arithmetic |
| conceptual | 123 / 200 (61.5%) | cover the domain concepts |

**Correction.** An earlier revision of this section reported "reward-opportunity
rates" of 74/200 and 168/200. Those are not recoverable from these artifacts:
`checkpoints/runs/grpo.csv` logs the batch *mean* of each reward component, not
its within-group standard deviation, and only the latter determines whether a
component produced gradient. The table above is step *eligibility*, an upper
bound on the opportunity rate. Logging per-component within-group std is one
number per step and should be added before the next run.

The run also visited **400 of 5,217 prompts (7.7%)** — sampling-limited, not
step-limited.

### 14.7 Corrected compute accounting

Earlier drafts described the base model as "7.3 tokens/parameter, far below
compute-optimal". That conflated three different quantities:

| quantity | value |
|---|---|
| total parameters | 125.9M |
| **non-embedding parameters** | **113.3M** |
| unique corpus tokens | 822.7M |
| epochs | 3.8 |
| **token exposures processed** | **3.13B** |
| **exposures per non-embedding parameter** | **27.6** |

Against the pretraining repo's *own measured* IsoFLOP frontier — which puts the
optimal ratio near **46** at its largest budget, not Chinchilla's ~20 — the base
is modestly under-trained, within a factor of ~1.7 of its frontier. The null
numerical result is therefore not well explained as data starvation.

Post-training compute, in units that transfer to another model or batch size:

| | value |
|---|---|
| SFT | 14,068 examples × 8 epochs = 112,544 example-exposures, ~10 min on 2× A30 |
| RL | 200 steps · **3,200 completions** · **954k generated tokens** · 3.0 GPU-h |

"200 steps" does not transfer; "3,200 completions and 954k generated tokens"
does.

### 14.8 What §14 supersedes

- The pooled **+3.0%** accuracy delta → **+0.8% [−3.1%, +4.7%]** on the frozen
  test half (§14.1).
- "7.3 tokens/parameter, data-starved" → **27.6 exposures per non-embedding
  parameter** vs a measured frontier of ~46 (§14.7).
- The aggregate numerical accuracy as a capability statement → the tiered
  breakdown, in which must-compute accuracy never clears its floor (§14.3).
- Any use of "GRPO" as an algorithm claim → group-relative REINFORCE with KL
  regularization (see the naming note at the top).

## 15. Why the GRPO curves never settled, and the v4 trainer (2026-10-15)

Prompted by: "the reward and KL curves are not rapid rise → slow rise → plateau." Full
write-up in README §5.8–5.9 and PITFALLS D3 / F3; this section is the log.

### 15.1 Diagnosis of the shipped v3 run (`checkpoints/runs/grpo.csv`)

- **Measurement:** 1 prompt per GPU per step, logged from rank 0 only → step SD 0.35, lag-1
  autocorr 0.04; residual SD 0.26 after type means + cubic trend (20-step-mean SE ±0.06).
  Single-type batches: numerical steps −0.64 vs conceptual −0.19; numerical share of a
  20-window 20–70% → type mix alone moves the rolling mean by up to 0.22.
- **Type-adjusted blocks** (38.5/61.5 mix): −0.419 / −0.269 / −0.367 / −0.367 → the hump is real.
- **Controller:** β at the 0.005 floor for steps 6–65; ×1.5 at steps 66, 79, 91, 95–100 →
  0.192 from step 100 to the end (the deadband held it). §12.2's "controller demonstrably
  worked" is withdrawn: it returned KL to target by undoing the learned behaviour.

### 15.2 Causal test (`configs/rl_grpo_intervention_beta{0192,0035}.yaml`)

From v3@100 weights, v3 algorithm, same prompts/seed, only β differs; held-out objective
(275 val prompts × 2, temp 1.0, fixed seed) every 10 steps for 60 steps:
β=0.192 → −0.460 at +10, mean −0.399; β=0.035 → −0.271 at +10, mean −0.266; worse at 6/6
evals. The β=0.192 arm's KL 0.10 → 0.045 retraces the original run's 0.10 → 0.042.

### 15.3 v4 (`configs/rl_grpo.yaml`, `checkpoints/grpo_v4/`, `checkpoints/runs/grpo_v4{,_eval}.csv`)

32 prompts × 8 per step, type-stratified, fixed β=0.04, k2 KL loss, batch-pooled advantage
std, constant loss normaliser, 10-step warmup, batched KV-cache rollouts (13 s/step vs 27 s;
cache verified to 3e-5 vs full forward in fp32). 300 steps, 82 min, 2.7 GPU-h, 76,800
completions. Held-out objective −0.646 → −0.268 (step 20) → −0.229 (30–100) → −0.200
(110–200) → −0.184 (210–300); KL 0.013 → 0.037 → 0.051 → 0.057; per-step SD 0.05.
Zero-spread groups 0 → ~20%; held-out numerical accuracy flat (4.7–10.7%).

### 15.4 Deployment-decoder check (`docs/eval_rl_v4_k4.json`)

Same protocol as `docs/eval_rl_paired_k4.json` (166 num + 20 conceptual × 4, temp 0.3 /
top-k 40 / rp 1.2, seed 1234). v3@100 re-scored identically on 186/186 prompts.
v4@300 (final step, pre-specified) vs v3@100: acc@1 +1.4 pp, acc@5 +0.9 pp, think +0.8 pp —
all n.s.; −36 tokens (significant). vs SFT: acc@5 +2.7 pp [+0.3, +5.0] (borderline, unadjusted).
`checkpoints/grpo/selected.pt` was NOT changed — still v3@100.

### 15.5 Follow-ups the same day: benchmark rebuild, deployment switch, numerical probes

- **GitHub sync.** github.com/jiazhe868/nanogpt-seis-posttrain had seven README edits that were
  missing locally (title emoji removed, Hugging Face weights link, DAPO KaTeX fixes `\mathrm{clip}`
  and `is-correct`). These are now ported into both READMEs. The code matched the pre-change
  backups byte for byte. **The GitHub `.gitignore` ignores all of `checkpoints/`**, so none of the
  `checkpoints/runs/*.csv` logs the figures are built from are published. The local `.gitignore`
  (unpushed) whitelists them.
- **The archived k=1 benchmark does not reproduce.** Re-scoring with the current `eval_rl`
  (seed 1234 + 1000·idx): SFT acc@5 9.6% → 5.4%, v3@100 10.2% → 11.4%, v3@200 9.6% → 7.2%.
  That is k=1 noise (SE ≈ 2.1 pp per bar), and it is why the benchmark figure moved to the k=4 file,
  where SFT and v3@100 reproduce per-item on 186/186 prompts. Saved as `docs/eval_rl_k1_rescore.json`.
- **Frozen test half (90 numerical × 4):** v4@300 acc@5 11.4% = v3@100 11.4% (Δ +0.0 [−3.1, +3.3]);
  vs SFT +0.8 [−2.5, +4.2]. The pooled +2.7 pp comes from the dev half (3.0% → 7.9%).
  Tiers (pooled, k=4): v4 retrievable 32.4% (+11.8 vs SFT, sig.), must-compute 4.0% (+0.4, n.s.).
- **Deployment.** The `src.sft.checkpoints` rule on the dev half (acc@5, tie-break `<think>` within
  2 pp, gates) picks v4@300 over v3@100 (7.9% vs 5.9%, 98.7% vs 98.0%). `checkpoints/grpo/selected.pt`
  now points to `../grpo_v4/grpo_ckpt_step_300.pt`. Record: `docs/checkpoint_selection_rl_v4.json`.
  The Hugging Face weights are still the old ones.
- **RAG column:** no committed script (local or GitHub) reproduces it. `chat.py --test --rag` uses
  hand-written prompts, and the JSON says n=90 while tier table (b) implies 166. Marked unreproduced.
- **Label noise in numerical targets.** `prepare_rl_dataset.extract_final_numerical_value` takes the
  last regex match of `[=≈~]` / "is …" in the reference *final answer*. Example: target 2.5 from
  "~2.5 ka", when the derivation computes Mw ≈ 7.5. Across train/held-out, 11.7% / 9.6% of targets
  appear in neither the derivation nor the prompt, and only 54% / 62% equal the derivation's last
  "=" result. Many prompts ask for two quantities but carry one label. Not yet fixed.
- **Numerical probes (scratch, train + dev only).** v4 at temp 1.0 on 192 train numerical prompts:
  retrievable pass@8 81% / per-sample 28%; must-compute pass@8 24% / per-sample 4.5% (≈ blind floor).
  On the dev half (k=8): exempting digit tokens from rp 1.2 gives retrievable acc@5 +16.7 pp
  [+2.5, +30.8], must-compute +0.0, `<think>` −5.4 pp. Majority vote over 8 samples is *worse*
  than per-sample on must-compute (−4.5 pp [−7.4, −2.3]).
- **Oracle-knowledge gate for test-time correction** (dev must-compute, n=61, k=8, v4, pinned
  decoder): baseline 5.3%; L1 with the reference's symbolic equations in the prompt 4.1%
  (−1.2 [−3.1, +0.6]); L2 with the reference derivation minus its final step prefilled into
  `<think>` 4.7% (−0.6 [−2.9, +1.6]). 48/48 L2 completions close `</think>`, so this is not a
  formatting artefact. Inspected failures: one correct answer scored wrong because of a bad label
  (Mw 7.5 vs label 2.5), and wrong final arithmetic even for a single division. Injecting knowledge
  does not help. Arithmetic and value binding have to leave the model (fixed formula templates + a
  deterministic solver), or a stronger solver has to do the work.

### 15.6 Numerical labels fixed (2026-10-15)

- **Rule:** `src/rl/numerical_target.py`. The label is the last value the reference COMPUTES
  (the `=`/`≈` result whose lead contains arithmetic) that the final answer restates; an
  explicit "the answer is X" wins; no label for questions that ask for no number. The old rule
  is kept as `prepare_rl_dataset.legacy_extract_final_numerical_value`.
- **Validation by blind adjudication** (old/new shown as A/B in random order, judged from the
  question and reference only):
  - tuning sample: 26/29 (tuned on, so not an estimate);
  - blind 1: 36/40 after tuning (tuned on);
  - blind 2, untouched at v5: old 60% → new 78% (24 → 31/40); after later tweaks, 33/40;
  - **blind 3, frozen rule, untouched: old 50% → new 80%** (20 → 32/40); disagreements
    16 new / 4 old / 2 tie, sign test p = 0.012.
  - Two failed attempts are kept as lessons: v3 was only 28 vs 27/40 on a fresh sample (it
    excluded given values and broke on coincidences such as "under 40"); v6, "last
    restatement in the final answer", dropped to 26–28/40 because it rewarded percent
    restatements and recapped inputs.
- **Applied** with `python -m src.rl.prepare_rl_dataset relabel` to `data/rl/{train,val,test}.jsonl`
  (originals in `data/rl/archive_labels_v1/`; each record keeps `final_numerical_value_v1` and
  `label_source`):
  - train 2,108 numerical: 54% unchanged, 33% changed, 278 unlabelled;
  - val 118: 52% / 33% / 18;
  - held-out 167: 63% / 29% / 13, leaving 153 scorable (dev 69, test 84).
  - Only label fields changed (verified record-by-record), and a fresh `process_sft_file` build
    gives identical labels.
- **Answer-in-prompt tier was half label error:** 34 → 19 items. 17 of the old 34 were labelled
  with a given value while the real answer is computed.
- **Ceiling/floor barely moved** (86.1% → 86.3%, 5.8% → 5.3%): the v1 ceiling was partly
  circular, because v1 labels were taken from the final answer's own last numbers.
- **Regression test:** `python -m src.rl.numerical_target` (10 cases; the legacy rule fails 4).
  `eval_rl` now stores per-sample final-answer numbers plus the target, so relabels can be
  re-scored offline.
- **Re-scored on corrected labels** (`docs/eval_rl_v2labels_k4.json`, 153 numerical × 4):
  - acc@5: SFT 5.7%, v3@100 6.0%, v4@300 8.0%; test half 6.8 / 7.4 / 9.2%.
  - The same generations scored on v1 labels give 9.5 / 9.5 / 9.6%.
  - v4 − SFT +2.3 [−0.3, +4.9]; v4 − v3 +2.0 [−0.7, +4.9]; must-compute 3.2 / 3.2 / 4.3% against a
    5.1% floor.
  - The dev-half selection still picks v4@300 (6.5% vs 4.3%).
- **MFU of the v4 RL run** (2× A30, 165 TFLOPS dense BF16 each; fwd ≈ 227 MFLOP/token incl. the tied head):
  - whole step 2.6%;
  - generation 0.6% (1.0 TFLOP/s/GPU, 82% of step time);
  - update 11.5% (19 TFLOP/s/GPU).
  - The 31 held-out evals add 16 of the 82 wall minutes.
  - The single-sequence `eval_rl` is far lower still: ~45 min for 692 completions.

### 15.7 Decoding, acceleration, and TRL ideas (2026-10-15)

- **Why a corrected-label retrain was skipped.** Starting from SFT at temperature 1.0 on 256 training
  prompts, the per-sample hit rate on v2 labels is 3.0%, BELOW its blind floor of 4.0%. pass@8 is 19%
  against a 28% floor, and only 19% of groups have any numerical spread. Under v1 labels, 34% of the
  rewarded hits were numbers copied from the question; under v2, 22%.
- **Digit-exempt repetition penalty: rejected.** Pre-registered rule on the dev half (v4@300,
  4 seeds × k=4, v2 labels): pick the best acc@5 subject to <think> ≥ baseline − 2 pp and repeated
  5-grams ≤ baseline + 2 pp.
  | config | acc@5 | <think> | rep5 |
  |---|---|---|---|
  | rp 1.2 (baseline) | 5.2 ± 0.7 | 99.5 | 2.5 |
  | rp 1.2 digits exempt | 5.8 ± 1.5 | 91.6 | 9.4 |
  | rp 1.1 | 5.2 | 94.7 | 13.3 |
  | rp 1.1 digits exempt | 5.9 | 88.0 | 18.7 |
  The baseline is kept. The earlier "+16.7 pp on retrievable items" was a v1-label artefact: exempting
  digits let the model copy given numbers, which matched mislabelled answers.
- **Batched decoder vs streaming decoder.** Forcing the same token path through both gives a mean TV
  distance of 0.005–0.013 per step and the same argmax 97–100% of the time; the CUDA-graph path is
  bit-identical to eager. Over 4 seeds, though, batched acc@5 runs ~1–1.5 pp below the streaming
  path's single run, for SFT and v4 alike, while format metrics agree. The cause is bf16
  forward-pass numerics (bf16 KV cache and attention kernel vs the streaming path's mixed-precision
  cache), not the logit arithmetic (matching it in bf16 barely moved the TV). Compare configs within
  one decoder only.
- **Speed.**
  - Batched `eval_rl` (src/rl/rollout.py): 173 prompts × 4 samples × 2 models in 42 s, against
    ~45 min per model streamed.
  - CUDA-graph decode step: generation 1.9× faster; decode is launch-bound (~700 kernels per token).
    The grouped-head attention and bf16 cache alone gave only 10%.
  - In training, generation fell from 11.0 s to 2.85 s per 128-rows-per-GPU batch.
- **From TRL's GRPO trainer:** `num_iterations` (μ) with ε 0.2 / ε_high 0.28 clip-higher, and
  truncated importance sampling (TIS, cap 2). The sampler/trainer mismatch is 0.020 nats per token,
  so the TIS weights average 1.000 with nothing truncated: a guard here, not a correction.
  - v5 = v4 + μ=2 + clip-higher + TIS: 300 updates in **17.4 min wall vs 82 min for v4 (4.7×)**,
    including 16 held-out evals. Clip fraction on the second pass is ~0.06%, so the reuse is
    essentially on-policy.
  - KL reached 0.098 at 300 updates (v4: 0.057). Held-out objective −0.644 → −0.284 (20 updates)
    → ~−0.20 plateau → −0.149, on v2 labels.

### 15.8 Calculator tool use, calculator-in-the-loop RL, and self-consistency (2026-10-15)

The question: the 113M model cannot do closed-book arithmetic, so can a calculator, trained in
by SFT and then by RL, fix numerical accuracy? Everything below is scored on the 153 held-out
numerical items of `data/rl/test.jsonl` (v2 labels), batched decoder at the deployment settings,
4 seeds × k=4 unless stated, paired bootstrap 95% CIs. The items are split into the 19 whose
answer is already stated in the prompt ("retrievable") and the 134 where it must be computed.
The question-blind floor (emit three other items' targets) is 3.3% on must-compute.

- **Mechanism** (`src/rl/rollout.py`, `calculator=True`). When a row's text ends in `<<expr>>`,
  the decoder evaluates `expr` with `safe_eval_number` (AST whitelist, no `exec`) and forces
  ` = {value:.4g}` into the row. The forced tokens are flagged (`return_forced`). The RL trainer
  masks them out of the policy and KL terms (`calculator: true` in the config), because they
  are observations, not actions.
- **Training data.**
  - `tools/numeric_drills.py`: 15 templates (Mw↔M0, energy and amplitude ratios,
    Gutenberg–Richter, S–P distance, EEW lead time, stress drop, Poisson probability, …).
    6,000 training drills; 279 held-out drills with no shared template+numbers.
  - `src/sft/calculator_format.py`: rewrites a reference step `EXPR = VALUE` into a call only
    when `safe_eval_number(EXPR)` reproduces VALUE within 2%, so wrong reference arithmetic is
    never blessed by a call. 2,832 of 3,705 numerical SFT pairs were converted (9,831 calls).
- **Continued SFT** (from `checkpoints/sft/selected.pt`, lr 5e-5, 2 epochs, ~3 min on 2× A30):
  - `sft_drills`: drills plus 6,000 replayed SFT pairs.
  - `sft_calc`: the converted corpus plus the drills (20,068 pairs).

  | model | held-out drills (calc on) | real acc@5 | real calls calc | compl. with a wrong mental step | must-compute |
  |---|---|---|---|---|---|
  | SFT | 3.9% | 4.6% | 0% | 82% | 2.8% |
  | SFT+drills | 86.4% | 5.3% | 10% | 77% | 3.4% |
  | SFT+calc | 81.0% | 5.1% (4.8% calc off) | **93%**, 2.7 calls each | **40%** | 3.1% |

  The "wrong mental step" checker also flags 32% of the references' own steps (it misparses
  fragments), so read that column as relative. The habit transferred, and arithmetic errors
  halved. Accuracy did not move. The failures are setup failures: hallucinated formulas
  ("log R = 1 − e^{−μ}/μ"), wrong operands (`<<2.8-3.0 * 2.8>>` for "2.8 / 3"), and quantities
  the question never asked for. The calculator executes them exactly. This is what the Phase-0
  gate in README §10 predicted ("the expressions are noise").
- **Calculator-in-the-loop RL** (`configs/rl_grpo_v6_calc.yaml`): v5's recipe (μ=2, ε 0.2/0.28,
  TIS), initialised from and KL-anchored to `sft_calc`, 150 rollout batches (17 min).
  - The held-out objective follows v5's: −0.668 → −0.28 by step 10 → −0.20 at step 150.
  - Calculator use on held-out rollouts went 0.80 → 0.35 by step 30, then back to 0.69.
    The dip is a truncation artefact, not a learned preference. At T=1.0, calculator rollouts
    from `sft_calc` run longer (356 vs 339 tokens), hit the 384-token cap more often
    (terminated 25% vs 33%), and so miss `</think>` (`r_format` +0.003 vs +0.110). Within mixed
    groups they score −0.28 lower. Their `r_num` is at the floor either way (−0.95 vs −0.92).
    After RL shortened the outputs, the gap was −0.01 and use recovered.
  - Deployment result: v6@150 5.4% (must-compute 2.9%), vs SFT+calc 5.1% (+0.3 [−0.9, +1.7])
    and v4@300 6.7% on the same seeds. At the deployment decoder v6 calls the calculator in 91%
    of completions (1.9 calls each), and only 19% contain a wrong mental-arithmetic step. Taking the setup out of the arithmetic gave the reward
    nothing to reinforce, because correct setups almost never occur.
- **Self-consistency** (`src/rl/self_consistency.py`; `eval_rl` reports `vote_at_k_5pct`;
  `python -m src.sft.chat --vote 8`).
  - Rule: return the one sample whose tail-3 answer numbers agree (5%) with the most other
    samples, scored as a single sample. Comparable with acc@1; pass@k is an oracle.
  - Pooled over 8 seeds (21–24 and 31–34; 32 samples per prompt, four disjoint 8-subsets;
    `python -m src.analysis.tier_vote docs/eval_vote_pooled8.json`, and `docs/eval_calc_k4.json`
    for the calculator rows above):

    | model | acc@1 | vote@8 | vote@8 − acc@1 | retrievable acc@1 → vote@8 | must-compute acc@1 → vote@8 |
    |---|---|---|---|---|---|
    | SFT | 4.9% | 5.7% | +0.9 [−0.5, +2.3] | 18.6 → 25.0% | 2.9 → 3.0% |
    | v4@300 | 6.4% | **9.2%** | **+2.7 [+0.9, +4.7]** | 29.3 → 43.4% | 3.2 → 4.3% |
    | v5@150 | 6.4% | 8.5% | +2.1 [+0.4, +4.0] | 28.1 → 47.4% | 3.3 → 3.0% |

  - v4 − SFT at acc@1 is +1.6 [+0.3, +2.8] over the same 8 seeds, all of it on the retrievable
    tier (+10.7 pp), with must-compute +0.3 (n.s.).
  - Voting compounds RL's retrievable gain and does nothing for must-compute. The memory note
    from the v1-label era ("voting hurts must-compute, −4.5 pp") is superseded: on v2 labels
    the must-compute effect is ~0.
  - Dropping values copied from the prompt before voting made it worse: correct answers often
    restate a given quantity.
- **Where this leaves numerical accuracy.**
  - Every intervention tried sits within 0.5 pp of the 3.3% chance floor on must-compute items:
    SFT, drills, calculator SFT, RL v4/v5, calculator RL, voting, and digit-exempt repetition
    penalty (§15.7).
  - The measurable gains are all on items whose answer is in the prompt: RL +10 pp; RL + vote@8
    +25 pp (18.6% → 43.4%).
  - Deployable best: v4 + vote@8, 9.2% vs SFT's 4.9% at acc@5.
  - What remains is formula recall and binding values to the right formula, which is a
    pretraining-scale capability, or retrieval (README §10, RAG).

### 15.9 Why numerical accuracy is stuck: a stage-by-stage diagnosis (2026-10-15)

Asked to improve numerical accuracy without RAG, so model size is treated as a hypothesis, not an
excuse. Everything is on the 153 held-out numerical items (v2 labels). The instrument ceiling
(reference answers, tail-3 at 5%) is 86.3%, and 85.8% on the 134 must-compute items; the
question-blind floor there is 3.3%. Figure: `assets/numerical_diagnosis.png`. Numbers:
`docs/numerical_diagnosis.json` and `docs/eval_numerical_real6.json`.

**What fails.**
1. **The right value is never derived.** It appears anywhere in a completion at the rate of
   another item's target (SFT 14.7% vs 13.1%; inside `<think>` 7.6% vs 7.7%). The final answer
   cannot restate what the reasoning never produced.
2. **Questions are not read.** 41 items need at most one operation on the question's own
   numbers (0.5% tolerance; another item's target is reachable that way 12% of the time). Every
   model scores 3–5% on them. On "felt over 1,000,000 km² vs 50,000 km², how many times
   larger?" SFT writes about a "magnitude difference".
3. **A finished derivation is not restated** (oracle ladder: prefill the reference `<think>` to
   a fraction f and let the model finish).

   | must-compute acc@5 | f=0 | 0.5 | 0.75 | 0.9 | 1.0 (only the answer left) |
   |---|---|---|---|---|---|
   | SFT, rp 1.2 | 2.6% | 4.1% | 11.8% | 15.9% | 38.8% |
   | SFT, rp 1.0 | — | — | 14.2% | 21.6% | **52.8%** |
   | SFT+calc | 2.2% | 4.1% | 11.9% | 19.0% | 45.1% |

   Two causes:
   - The repetition penalty also counts the prompt and the derivation, so restating a computed
     value is penalised (−14 pp at f=1).
   - The SFT final answers re-derive in prose instead of restating ("M₀ = μ A D_avg = …"), so
     the model makes a second, independent attempt and fails it. Even at f=0.9 the final answer
     often states a different number from one already written in the prefix.
4. **It is not memorisation or undertraining in the usual sense.**
   - The shipped SFT made 336 optimizer updates (train loss 1.12) and scores 5.1% on numerical
     questions it was trained on, the same as on held-out ones (labels verified identical,
     153/153).
   - Continued SFT on all 3,705 numerical pairs drove train loss to 0.39 and seen-question
     accuracy stayed at 3.9%; greedy decoding without a penalty reached 9.2%.
   - Teacher-forced, the model predicts computed result tokens at top-1 0.74 (train) / 0.61
     (held-out). Over ~300-token free-prose derivations with ~4 tokens per number, sampled
     reproduction compounds to near zero.
5. **Arithmetic is real but secondary.** With the calculator, completions with a wrong
   mental-arithmetic step fell 82% → 19%, and nothing improved (§15.8).
6. **A calculator bug, found and fixed.** A forced value was not terminated, so the model
   extended it (` = 5` + `.833`, ` = 0.05` + `625`). The decoder now bans number-continuation
   tokens after a forced value. Re-scoring every calculator model changed results by ≤ 1 pp.
7. **Tokenizer.** Numbers split inconsistently ('13.636' → `13 . 6 36`; '1,000,000' and
   '1000000' share no tokens), which makes copying a number into an expression a transformation.
   Calculator expressions now keep the question's digits verbatim, and the calculator drops
   thousands separators (argument commas, as in `pow(2, 3)`, are kept).

**What training fixes, and how far it transfers.** All models below are trained from SFT. Sets:
fresh-number counterfactuals of *training* questions; generated word problems in phrasings
never trained; fresh-number counterfactuals of *held-out* questions; the real held-out items.

| acc@5 (2 seeds × 4) | trained questions, new numbers | unseen phrasings | held-out questions, new numbers | real held-out |
|---|---|---|---|---|
| SFT | 3.1% | 2.0% | 4.2% | 4.7% |
| counterfactual SFT (`configs/sft_cf.yaml`) | 15.6% | — | 3.5% | 5.1% |
| + 30k diverse word problems (`configs/sft_wp2.yaml`) | **18.0%** | **67.8%** | 4.0% | 6.3% |
| + GRPO with calculator (`configs/rl_grpo_v7_wp.yaml`) | 11.4% | 66.0% | 4.7% | 6.2% |

- **Counterfactuals** (`src/sft/counterfactual.py`). Each reference derivation is extracted as
  a program (verified steps, operands traced to question values), the given values are
  perturbed, every step is re-evaluated, and the question, reasoning and answer are rewritten.
  1,248 of 3,242 labelled training items yield a program (4,926 variants). Binding becomes
  learnable: 3% → 16–18% on new numbers in trained questions.
- **Word problems** (`tools/word_problems.py`). 14 families, ~60 quantities, 54 phrasings, 20%
  held out, including formula application ("Using the relation y = a x^b …"). Accuracy on
  phrasings never seen goes 2% → 68%.
- **Transfer to new real question types is zero.** This holds even where the held-out
  computation is a plain ratio, a shape 212 training items share (3.2%). After word-problem SFT,
  the model now recognises "how many times larger" on the real felt-area question and extracts
  the values, but miscopies one (1,000,000 → 500,000) or divides the wrong way.
- **Real held-out over 6 seeds** (`docs/eval_numerical_real6.json`):
  - word-problem SFT: 6.5%, +1.9 pp [+0.0, +4.0] over SFT, without RL;
  - GRPO v4: 6.1%, +1.4 [+0.4, +2.6];
  - v7: 6.3%;
  - must-compute: 2.8% (SFT), 3.1% (v4), 3.9% (word-problem SFT), 4.1% (v7). None of these
    is significant, and all sit near the 3.3% floor.
- **Decoder** (pre-registered, dev half). Applying the repetition penalty to generated tokens
  only (`rp_scope="output"` in `src.rl.rollout.generate`) is better than the shipped rule on
  all three criteria for v4: acc 5.4 → 6.5%, `<think>` 98.4 → 99.4%, repeated 5-grams 8.7 →
  6.9%. On the frozen test half it gives 7.2 → 7.7%. The decoding contract was not changed;
  that is the user's call.

**Conclusion.**
- Numerical accuracy is not limited by arithmetic, labels, the RL algorithm, the decoder, or
  training length. Each of those was measured, and the fixable parts were fixed, for small
  real gains.
- It is limited by **the number of distinct question types**. The 113M model learns to read
  values and apply a relation for wordings close to its training data, at 16–68% where it was
  3%, and does not abstract the skill to new question types.
- 1,248 program-bearing question types (and 54 generated phrasings) are far too few. TinyGSM
  needed ~12M synthetic problems for a 125M model to reach 63% on GSM8K.
- The path without RAG is data: a large, diverse, teacher-generated set of seismology problems
  in the real question style, with program solutions, trained the same way. A larger base is
  the other option. That set needs a teacher endpoint (`TEACHER_API_BASE`) and the user's go-ahead.

### 15.10 Program-of-thought with a local teacher: the first must-compute gain (2026-10-16)

The §15.9 conclusion was that the limit is the number of distinct question types; the published
~100M-parameter successes also answer with executable programs. Both were tested here with no
external API, using a local Apache-2.0 teacher.

**Teacher and pipeline.**
- `Qwen/Qwen3.8-27B-FP8` on the 2× A30: vLLM 0.24, tensor parallel 2, FP8 weight-only Marlin
  kernels (Ampere has no FP8 tensor cores), ~900 tok/s.
- Fixes needed: `VLLM_USE_FLASHINFER_SAMPLER=0` (the system nvcc 11.5 is too old for
  FlashInfer's JIT), and `max_num_seqs=64` (the GDN recurrent state is allocated per sequence).
- `tools/gen_pot_problems.py` modes:
  - `generate`: exercises seeded by training questions only;
  - `solve`: independent re-solve, kept when the two executed answers agree within 1%;
  - `convert`: real training questions as programs, kept only when the program reproduces the label;
  - `audit`: answerability of the held-out questions.

  Outputs sharing a 13-word sequence with any held-out prompt are dropped.
- `src/rl/pot.py`: a restricted interpreter (assignments, arithmetic, whitelisted math, no
  loops, attributes or imports beyond `math`). `rollout.generate(pot=True)` executes the
  program at `</think>` and forces `Answer: <value>`.

**What the benchmark allows.** The teacher judged 125 of 153 held-out questions answerable
closed-book. The 27B solving blind, one program each, scores 49.7% (must-compute 48.5%;
`docs/teacher_27b_heldout_blind.jsonl`). That is the practical ceiling for distillation here.

**Data.**
- Pilot: 4,800 requested → 3,602 run → 2,405 verified.
- Overnight bulk run (10k seed prompts from all 14k training questions, 22:23–04:10):
  34,832 run, 32,190 verified (92%), giving 29,940 training pairs after the held-out seed split.
- 2,334 of 3,242 labelled real training questions converted to programs.
- `tools/pot_counterfactuals.py` re-instantiates those programs with new numbers, keeping
  formula constants fixed: 11,770 real-style variants with exact labels.

**Scaling** (students trained from SFT with converted real questions and replay; 2 seeds;
`docs/eval_pot_scaling_*.json`):

| generated exercises | generated held-out | real held-out | real must-compute |
|---|---|---|---|
| 2.1k (pilot) | 3.5% | 2.8% | 2.1% |
| 3k | 4.2% | 2.5% | 1.9% |
| 10k | 11.3% | 3.8% | 3.6% |
| 30k | 21.3% | 4.6% | 4.2% |
| 30k + 11.8k real-style variants | 20.8% | 6.9% | — |

- Accuracy on teacher-style exercises rises steeply with data. Real accuracy rises slowly:
  the generated exercises are short and single-quantity, while real questions are long and
  multi-part.
- The real-style variants are what move held-out real questions. On held-out real questions
  re-instantiated with new numbers (257 variants of the 66 held-out questions the teacher solves;
  `data/eval/real_heldout_newnumbers.jsonl`), the student scores 9.8% against SFT's 3.4%:
  +6.4 pp [+3.7, +9.6].

**6-seed result** (`docs/eval_numerical_real6.json`, `python -m src.analysis.tier_vote … --baseline SFT`):

| | must-compute acc@1 | must-compute vote@8 | in-prompt acc@1 | all acc@1 | all vote@8 |
|---|---|---|---|---|---|
| SFT | 2.8% | 5.5% | 18.0% | 4.7% | 8.5% |
| GRPO v4 | 3.1% | 3.5% | 27.4% | 6.1% | 8.1% |
| program format + real-style variants (`configs/sft_pot_realcf.yaml`) | **6.2%** | **9.2%** | 11.2% | 6.8% | 9.8% |
| hybrid vote: 4 v4 prose + 4 program samples | — | 8.1% | 34.2% | — | **11.3%** [7.8, 15.0] |

- Must-compute: +3.4 pp [+0.8, +6.2] over SFT and +3.1 pp [+0.6, +5.9] over v4. This is the
  first significant gain on questions whose answer is not in the prompt, at twice the chance
  floor.
- Voting over executed answers adds +3.0 pp [+1.1, +5.2] on must-compute; voting over prose
  never did.
- The program format loses the in-prompt items, where prose restates a given value. A vote over
  prose and program samples recovers them.

**Next levers, by expected value.**
1. More real-style data: teacher rewrites of the real training questions (new scenario and
   numbers, same multi-part style, program solution). The counterfactuals show real style is
   what transfers.
2. RL with execution rewards on prompts with exact labels, starting from the program-format
   student. Its success rate is ~10–20% there, enough reward spread to learn from.
3. In-prompt questions in program form (`answer = <given value>`), so a single model covers
   both tiers.

### 15.11 Real-style rewrites at scale, and RL on the program-of-thought student (2026-10-16)

**Data** (`tools/gen_pot_problems.py rewrite`).
- For each of the 3,701 real numerical training questions, the 27B wrote 6 new questions in the
  same style: multi-part, a new scenario and new numbers, with a program for the *last*
  quantity asked, matching how labels are defined.
- 19,431 parsed and ran (88% of 22,206 requested); 14,553 were verified by independent re-solve.
- After the held-out seed split: 12,863 training rewrites plus 33,000 re-instantiations with new
  numbers (`tools/pot_counterfactuals.py`, formula constants fixed).
- Student mixture (`configs/sft_pot_v2.yaml`, 95,907 pairs):
  - 29,940 teacher exercises;
  - 2,334 real questions as programs and 11,770 of their re-instantiations;
  - 12,863 rewrites and 33,000 of their re-instantiations;
  - 6,000 replayed non-numerical pairs.

**RL** (`configs/rl_grpo_v8_pot.yaml`): the v5 recipe with `pot: true`. The interpreter's
`Answer: <value>` tokens (~0.9% of each completion) are masked from the loss.
- Prompt pool: the 5,217 real RL prompts plus 11,000 exact-label prompts. The run made 4,800
  draws, 0.3 passes over the pool.
- It ran out of memory at step 134 on long prompts; checkpoints 25–125 exist. The config now uses
  `micro_batch_rows: 8`.
- Training-batch accuracy stayed flat at 0.30–0.33 and entropy stayed flat (1.14 → 1.09). The
  reward gain came from `<think>` rate (0.83 → 0.97) and termination (0.67 → 0.90), the same
  pattern as v4.
- The correctness term is not too weak: it spans +1 to −1, more than any other component. RL
  simply did not change numerical ability in 300 updates at lr 5e-6.

| 2 seeds × 4 (`docs/eval_pot_v2_sets_*.json`) | real held-out | real held-out, new numbers | rewrite held-out | generated held-out |
|---|---|---|---|---|
| `sft_pot_realcf` | 6.9% | 9.8% | 8.2% | 20.8% |
| **`sft_pot_v2`** | **9.9%** | 12.1% | **14.3%** | 25.1% |
| v8 @50 | 9.2% | 12.0% | 11.7% | 26.1% |
| v8 @125 | 8.7% | 13.1% | 11.5% | 25.8% |

**6 seeds × 4 on the real held-out set** (`docs/eval_numerical_real6.json`):

| | must-compute acc@1 | must-compute vote@8 | in-prompt acc@1 | all acc@1 | all vote@8 |
|---|---|---|---|---|---|
| SFT | 2.8% | 5.5% | 18.0% | 4.7% | 8.5% |
| GRPO v4 (deployed) | 3.1% | 3.5% | 27.4% | 6.1% | 8.1% |
| `sft_pot_realcf` | 6.2% | 9.2% | 11.2% | 6.8% | 9.8% |
| **`sft_pot_v2`** | **9.4%** | **11.9%** | 10.1% | **9.4%** | 11.8% |
| hybrid vote: 4 v4 prose + 4 `sft_pot_v2` program samples | — | 10.7% | 36.0% | — | **13.8%** [9.7, 18.3] |
| Qwen3.8-27B teacher, blind, one program | 48.5% | — | 57.9% | 49.7% | — |

- **Must-compute.** `sft_pot_v2` vs SFT: +6.6 pp [+3.0, +10.5]; vs v4: +6.2 pp [+2.6, +10.0].
  That is 3.4× SFT and 2.8× the chance floor.
- **What drove it:** real-style data. Teacher exercises alone did not transfer (§15.10);
  re-instantiated real questions doubled must-compute; rewrites added another 50%.
- **Deployable best:** the hybrid vote, 13.8% overall, about 3× SFT and 2.3× v4.
- **Next steps:**
  - more rewrites per question (the yield was 88%, and rewrite held-out still rises);
  - program-form training for in-prompt items, so a single model covers both tiers;
  - a much longer or higher-learning-rate RL run (or rejection-sampling fine-tuning) before
    calling RL useless here: this run moved format only.

### 15.12 Why GRPO gives no gain on the program-of-thought student (2026-10-16)

Three GRPO runs on the program-format student (v8 at lr 5e-6, v8b at lr 3e-5, v9) gave no
numerical gain. The diagnosis, step by step:

1. **The reward was not the problem.** In v8 the within-group std of `r_num` was ~0.40, against
   0.03 for format and termination, so the advantage was dominated by correctness. The
   correctness term spans +1 to −1.
2. **The step size was not the problem.** At lr 3e-5 the run got *worse* at its own objective:
   batch accuracy 0.30 → 0.12, held-out acc@1% 0.167 → 0.03, KL 0.33. On 96 training prompts,
   total reward went +0.146 → −0.293, correctness −0.22 → −0.78, and programs that run 53% → 33%.
3. **v9 removed every confound we could name.** It used:
   - reward = executed-answer correctness only, 0 if the program does not run;
   - prompts the student had never seen (fresh 27B rewrites plus fresh-number variants,
     `data/rl/train_v9_learnable.jsonl`), filtered to 1–7 passes out of 8
     (`tools/filter_learnable_prompts.py`): 2,823 of 5,471;
   - ~2.5 passes, T=0.8, μ=1, lr 1e-5, β=0.01 (`configs/rl_grpo_v9_pot.yaml`).

   Even so, reward on its own training pool fell 0.52 → 0.38 while entropy rose 0.18 → 1.07
   and KL reached 0.22. Deployment-decoder evaluation (`docs/eval_grpo_v9_*.json`):

   | | real held-out | rewrite held-out |
   |---|---|---|
   | start | 9.9% | 14.3% |
   | v9 @50 | 6.5% | 10.3% |
   | v9 @150 | 9.9% | 10.2% |

4. **The optimiser and the masking are correct.** GRPO on 32 fixed prompts for 40 steps, each
   step on all 32, raised reward on them:
   - T=0.8 with truncated importance sampling: 0.63 → 0.94, entropy 0.15 → 0.05;
   - T=1.0 pure on-policy: 0.36 → 0.86.

**Conclusion.** At this model size, a policy-gradient update earned on one problem does not
transfer to others. It is the RL counterpart of the SFT finding in §15.9, that skills learned
for one question wording do not carry to new question types. On a handful of repeated problems
GRPO memorises them. Across thousands of distinct problems the per-problem updates add up to
noise that raises entropy and slowly degrades program-writing. The lever that works is SFT on
more real-style teacher data. A second rewrite round was launched (`configs/sft_pot_v3.yaml`).

### 15.13 What the program student gets wrong, the decoder, and a second rewrite round (2026-10-16)

**Failure analysis** (`sft_pot_v2`, 160 held-out rewrites × 4 samples at T=0.8, exact labels; each
failure classified against the teacher's program):

| outcome | share |
|---|---|
| program crashes | **45.5%** (undefined variable name 30%, syntax 12%, other 3.5%) |
| correct | 15.9% |
| wrong, other | 15.5% |
| off by a power of ten (units) | 11.7% |
| off by a common factor (2, π, 60, 1000, …) | 5.2% |
| returned a teacher intermediate value (wrong final quantity) | 2.8% |
| no closed program | 2.7% |

- **GRPO v10** (`configs/rl_grpo_v10_pot.yaml`, reward: crash 0, runs but wrong 0.2, correct 1)
  targeted the crashes. Over 100 rollout batches the run rate stayed flat (0.72 → 0.71) and so
  did accuracy. The run then ran out of memory during a held-out eval.
- **A lenient interpreter** (`run_program_lenient`) repairs misspelled names, skips unparsable
  lines and reads paper notation. On the same completions it lifted the run rate from 57% to
  89%, but accuracy stayed at 13% (`docs/eval_lenient_interpreter.json`). The crashes are
  symptoms of incoherent working, not near-misses, so rewarding "it runs" cannot add
  correctness. As the user pointed out, the base model was pretrained on papers and Wikipedia,
  not code.
- **The repetition penalty breaks programs.** It penalises re-emitting tokens already written,
  but a program must reuse its variable names exactly. Six seeds × 4
  (`docs/eval_numerical_real6_decoder.json`):

  | | all | in prompt | must-compute |
  |---|---|---|---|
  | `sft_pot_v2`, rp 1.2 | 9.4% | 10.1% | 9.4% |
  | `sft_pot_v2`, rp 1.0 | 11.7% | 21.3% | 10.3% |

  Program-format students should be decoded with rp 1.0. The prose decoding contract (B3) is
  unchanged.

**Round 2 of real-style rewrites** (`configs/sft_pot_v3.yaml`):
- Data: 19,331 rewrites → 14,294 verified → 12,646 usable after the held-out split, plus the
  2,471 fresh v9-pool rewrites and 25,912 re-instantiations; 136,934 pairs in total.
- On the exact-label sets (2 seeds, `docs/eval_pot_v3_sets.json`), the gains are significant:

  | set | `sft_pot_v2` | `sft_pot_v3` | difference |
  |---|---|---|---|
  | rewrite held-out | 14.3% | 17.7% | +3.4 pp [+2, +5] |
  | generated held-out | 25.1% | 27.6% | +2.5 pp [+1, +4] |

- On the real held-out set the difference is within noise: 6 seeds at rp 1.0, must-compute
  11.1% vs 10.3%.

**Best configurations, 6 seeds × 4 on the real held-out set:**
- must-compute acc@1: `sft_pot_v3` at rp 1.0, **11.1%** (SFT 2.8%; +8.3 pp [+4.6, +12.5]);
- must-compute vote@8: **13.9%**;
- overall: hybrid vote of 4 v4 prose samples + 4 `sft_pot_v2` samples at rp 1.0, **15.9%**
  [11.2, 20.5] (in-prompt 43.9%, must-compute 11.9%), against 4.7% for SFT and 6.1% for v4.

The real held-out set now has too little power for further gains of 1–2 pp (CIs ±3–4 pp). The
600-item exact-label rewrite held-out set is the sensitive measure from here on.

### 15.14 Paper-notation solutions: matching the format to the base model (2026-10-17)

The user pointed out that the base model was pretrained on domain papers, Wikipedia and other
text, not code, so it should not be expected to write Python. The test: rewrite every solution as
paper-style equations and train the same way.

**Format.** One equation per line, `SYMBOL = expression [unit]`:
- short standard symbols (`Mw`, `M0`, `μ`, `v_p`; at most 8 characters);
- `^` for powers, `log10` / `ln` (a bare `log` is rejected);
- no comments, no code keywords, no `answer =` line; the last equation is the answer.

`src/rl/pot.py` executes it deterministically with `notation=True` (normalising `^ × · − π` and
stripping the unit brackets), and the decoder uses it at `</think>` (`pot_notation=True`).

**Data** (`tools/gen_pot_problems.py notation`). The 27B rewrote all 60,251 unique solutions behind
`sft_pot_v3`. A rewrite was kept only if it executed to the original program's answer within
0.1% and used short symbols:
- kept: 42,158 (70%);
- rejected: 9,051 bad format, 5,508 symbols too long, 1,828 that did not run, 1,706 with a
  different answer.

New-number variants were made from the notation solutions in the same proportions as before.
The mixture (`configs/sft_pot_notation.yaml`) has 96,836 pairs, 71% of `sft_pot_v3`'s 136,934.

**Results** (real held-out set, 6 seeds × 4, rp 1.0; `docs/eval_numerical_real6_notation.json`):

| | must-compute acc@1 | must-compute vote@8 | in-prompt acc@1 | all acc@1 | all vote@8 |
|---|---|---|---|---|---|
| SFT | 2.8% | 5.5% | 18.0% | 4.7% | 8.5% |
| GRPO v4 (deployed) | 3.1% | 3.5% | 27.4% | 6.1% | 8.1% |
| `sft_pot_v3` (Python programs) | 11.1% | 13.9% | 12.5% | 11.3% | 14.4% |
| **`sft_pot_notation`** | **17.3%** | **18.7%** | 21.5% | **17.8%** | **19.2%** |
| notation, rp 1.2 | 14.3% | 15.4% | 14.9% | 14.4% | 15.7% |
| **hybrid vote: 4 v4 prose + 4 notation samples** | — | 18.0% | 38.6% | — | **20.6%** [15.1, 25.9] |

- Must-compute: +14.5 pp [+9.1, +20.4] over SFT, and +6.2 pp [+2.0, +10.8] over `sft_pot_v3`.
  The overall gain over `sft_pot_v3` is +6.5 pp [+2.6, +10.9].
- The exact-label sets (2 seeds × 4, `docs/eval_notation_sets.json`) agree:

  | set | `sft_pot_v3` | notation | difference |
  |---|---|---|---|
  | real held-out, new numbers | 15.6% | 23.6% | +8.1 pp [+4.8, +11.7] |
  | generated held-out | 36.2% | 41.5% | +5.3 pp [+2.3, +8.1] |
  | rewrite held-out | 24.5% | 20.2% | −4.3 pp [−7.0, −1.5] |

  On the rewrite set the notation student saw ~30% fewer rewrite examples, plausibly skewed away
  from complex ones by the verification filter.

**Mechanism.** On real questions the notation student writes shorter working, crashes less, and
is more often right when its working runs:

| real held-out | programs that run | accuracy when it runs | lines of working |
|---|---|---|---|
| `sft_pot_v3` | 58% | 18.5% | 13.1 |
| notation | 75% | 23.4% | 8.5 |

On the generated rewrites both students crash equally often (~24.5%). The format helps most on
questions phrased like the papers the base model read.

One new, specific failure appears: 10.6% of notation completions compute the target (or a
teacher intermediate) but end on a different line, since in notation the last line is the answer.

**Conclusion.** Matching the solution format to the pretraining distribution was worth more than
40k extra training pairs: real accuracy rose from 11.3% to 17.8%.

**Next:**
- recover the ~14.5k format and long-symbol rejections with a feedback re-prompt, to close the
  data gap;
- revisit GRPO on this student, which now has a better prior to sharpen;
- address the trailing-line failure.

### 15.15 GRPO gains on the paper-notation student; recovered notation data (2026-10-17)

**GRPO v11** (`configs/rl_grpo_v11_notation.yaml`) uses the v9 setup on `sft_pot_notation`:
correctness-only reward on executed answers; prompts the student never trained on (fresh-number
variants of round-1 rewrites), filtered to 1–7 passes of 8 at T=0.8 (2,868 of 6,000); 200 rollout
batches, lr 5e-6, β 0.02.

Checkpoint trajectory with the deployment decoder (seeds 33–34 × 4, `docs/eval_grpo_v11_and_notation2.json`):

| checkpoint | real held-out | real held-out, new numbers |
|---|---|---|
| start | 17.6% | 24.3% |
| step 50 | 18.9% (+1.2) | 27.5% (+3.3) |
| **step 100** | **20.9% (+3.3 [+0.4, +6.2])** | **28.3% (+4.0 [+1.3, +6.8])** |
| step 150 | 20.2% (+2.5) | 28.1% (+3.8) |
| step 200 | 18.7% (+1.1) | 27.1% (+2.9) |

Six seeds × 4 at step 100 (`docs/eval_numerical_real6_final.json`):

| | all acc@1 | must-compute acc@1 |
|---|---|---|
| step 100 | **21.2%** | **20.1%** |
| vs start | +3.3 pp [+0.8, +6.3] | +2.9 pp [+0.1, +5.7] |
| vs SFT | — | +17.4 pp [+11.4, +23.6] |

- It costs 0.9–2.5 pp on the generated and rewrite held-out sets (seed 31).
- This is the first GRPO run in this project to beat its own starting point on real questions.
- Same trainer, reward and prompt construction as v9, which lost 3–4 pp from the
  Python-program student. What changed is the student's format: working written in notation
  native to its pretraining gives GRPO a behaviour to sharpen that transfers across problems.
- The gain peaks at ~1.1 passes over the pool and decays after; checkpoint 200 is +0.9 pp (n.s.)
  over 6 seeds.
- **Caveat:** step 100 was identified using evaluation families that overlap the reported ones.
  All four saved checkpoints were positive on both real sets, but the next run should select its
  checkpoint on a separate dev split.

**Recovered notation data.** The 18,093 rewrites the verification filter rejected were
re-prompted with an explicit reminder of the violated rules, and 6,483 (36%) were recovered.
`configs/sft_pot_notation2.yaml` trains on 110,903 pairs (vs 96,836).
- 6 seeds × 4, `sft_pot_notation2`:
  - all 19.7%, must-compute **20.1%** (+2.8 pp vs `sft_pot_notation`, n.s.);
  - vote@8: all **22.2%**, must-compute **22.4%**.
- 2 seeds on the exact-label sets: rewrite held-out +2.0 pp, real new numbers +2.6 pp,
  generated held-out +1.1 pp, all borderline.

**Standing, 153 real held-out numerical questions (6 seeds × 4):**

| | must-compute acc@1 | all acc@1 | all vote@8 |
|---|---|---|---|
| SFT | 2.8% | 4.7% | 8.5% |
| GRPO v4 (deployed) | 3.1% | 6.1% | 8.1% |
| `sft_pot_notation2` | 20.1% | 19.7% | **22.2%** |
| notation + GRPO v11 @100 | 20.1% | **21.2%** | 22.0% |
| Qwen3.8-27B teacher, blind | 48.5% | 49.7% | — |

**Next:** GRPO from `sft_pot_notation2`, early-stopped near one pass with the checkpoint chosen
on a dev split. RL and the data recovery are independent gains, so they may add.

### 15.16 GRPO from the stronger notation student, with dev-split selection (2026-10-17)

`configs/rl_grpo_v12_notation.yaml` uses v11's recipe on `sft_pot_notation2`:
- 3,031 learnable prompts out of 6,000 that this student had never trained on (it already
  solves 64% of candidates);
- checkpoints every 25 steps;
- **checkpoint chosen on a dev split** (the 96 labelled numerical questions of `data/rl/val.jsonl`,
  excluding 3 chunks shared with test; 4 samples) before any test evaluation.

| step | 0 (start) | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| dev reward | **0.378** | 0.359 | 0.345 | 0.365 | 0.331 | 0.332 | 0.319 |

The rule picked step 75 (best among steps > 0), but the dev split preferred the pre-RL
checkpoint. RL barely moved the student: KL 0.013 at step 150, training accuracy flat at ~0.63,
entropy 0.10–0.12.

Test results for step 75:

| set | `notation2` | v12 @75 | difference |
|---|---|---|---|
| real held-out, all (6 seeds × 4) | 19.7% | 21.0% | +1.3 pp [−1.4, +4.0] |
| real held-out, must-compute | 20.1% | 20.4% | +0.3 pp |
| real held-out, vote@8 | 22.2% | 22.7% | +0.5 pp |
| real held-out with new numbers (2 seeds) | — | — | +3.0 pp [+0.0, +6.0] |
| generated held-out | — | — | +1.9 pp |
| rewrite held-out | — | — | −0.5 pp |

**Conclusion.**
- On the paper-notation student, GRPO gave +3.3 pp from the weaker SFT checkpoint (v11, with
  the selection caveat in §15.15) and nothing significant from the stronger one (v12, selected
  on dev).
- RL and the extra SFT data fix overlapping failures. Once the SFT student is confident on the
  kind of problem RL can verify, there is little left for GRPO to sharpen at this scale.
- The dependable levers remain:
  - the solution format (paper notation over Python: +6.5 pp);
  - real-style teacher data;
  - decoding without the repetition penalty;
  - voting over executed answers.

**Final standing** (153 real held-out numerical questions, 6 seeds × 4, `docs/eval_numerical_real6_final.json`):

| | must-compute acc@1 | all acc@1 | all vote@8 |
|---|---|---|---|
| SFT | 2.8% | 4.7% | 8.5% |
| GRPO v4 (deployed) | 3.1% | 6.1% | 8.1% |
| `sft_pot_notation2` | 20.1% | 19.7% | 22.2% |
| `sft_pot_notation2` + GRPO v12 @75 | 20.4% | 21.0% | 22.7% |
| Qwen3.8-27B teacher, blind | 48.5% | 49.7% | — |

### 15.17 Where GRPO's gains go: a controlled transfer study (2026-10-17)

**Motivation.** GRPO had a mixed record on the real held-out set:
- it decisively fixed format and termination (v4);
- it gained +3.3 pp from the first paper-notation student (§15.15);
- it gained +1.3 pp (n.s.) from the stronger one (§15.16).

There are two candidate explanations:
- (i) RL is not learning;
- (ii) RL learns, but what it learns does not reach the held-out questions.

This section separates them with test sets built at known distances from the training problems.
It then measures the rule that decides how far a gain travels. Results are in
`docs/rl_transfer_diagnosis.json`; the figure is `assets/rl_transfer_diagnosis.png`
(`python -m src.plots grpo`).

#### Design

**Sets.** All are 27B-teacher problems in the real-question style. Labels come from executing
the teacher's program. None of these problems, and none of A's or B's source chunks' questions,
were in SFT.

| set | file | size | what it is for |
|---|---|---|---|
| A | `data/rl/transfer_A.jsonl` | 2,027 problems, 115 chunks | source of the RL pool |
| A-learnable | `transfer_A_learnable.jsonl` | 576 (75 chunks) | **RL pool**: solved 1–7/8 by `sft_pot_notation2` at T = 0.8 |
| A0-shared / A0-unshared | `transfer_A_zero_{shared,unshared}.jsonl` | 1,041 / 231 | never solved (0/8); chunk does / does not contain a pool problem |
| A8-solved | `transfer_A_solved.jsonl` | 179 | always solved (8/8) |
| B600 | `transfer_B600.jsonl` | 600, 92 chunks | other papers: chunk-disjoint from A |
| B-other(-learnable) | `transfer_Bother(_learnable).jsonl` | 700 (230), 93 chunks | same papers as B600, different problems: a second RL pool |
| real | `data/rl/test.jsonl` | 153 scorable | real held-out; chunks disjoint from SFT, A and B |

**Similarity.** For each test question we take its most similar training question and compute
their word-Jaccard similarity: |W₁ ∩ W₂| / |W₁ ∪ W₂| over content words of 3 or more letters,
found with an inverted index over the pool (`scripts/experiments/nn_sim.py`).

**Arms** (all from `checkpoints/sft_pot_notation2/sft_ckpt_epoch4.pt`):

| arm | config | recipe |
|---|---|---|
| GRPO-lo on A | `configs/rl_grpo_transfer_lo.yaml` | v11 recipe: correctness reward, lr 5e-6, β 0.02 (k2), ε 0.2/0.28, T 0.8; 150 steps × 32 prompts × 8 samples (each pool prompt seen ~8×) |
| GRPO-hi on A | `rl_grpo_transfer_hi.yaml` | the same at lr 1.5e-5 |
| GRPO on B-other | `rl_grpo_transfer_Bsame.yaml` | GRPO-lo on B-other-learnable (each prompt seen ~21×) |
| RFT-A | `configs/transfer_rftA.yaml` | SFT on the student's own correct samples: 16 per A problem, up to 2 kept (1,464 pairs); 2 epochs, lr 2e-5 |
| teacher SFT-A | `transfer_sftA.yaml` | SFT on 1,279 teacher notation solutions of A; 2 epochs, lr 2e-5 |

RFT-A and teacher SFT-A were trained on all of A, including 157 and 617 of the A0-shared
problems respectively. They are therefore not transfer tests on that set.

**Scoring.** Deployment decoding: notation PoT, T = 0.3, top-k 40, rp 1.0, 2 seeds × 4.
Differences are paired, with 2,000-sample bootstrap 95% CIs over problems.

#### Results

**1. RL learns its pool quickly and thoroughly.** GRPO-lo goes 48.8% → 85.4% (step 75) →
88.1% (step 150). The failure taxonomy (`scripts/experiments/taxonomy_file.py`; 400 problems × 4) shows
every error class shrinking by about three quarters:

| | correct | crash | power of ten | common factor | teacher's intermediate | right value computed, other returned | other setup |
|---|---|---|---|---|---|---|---|
| A-learnable, SFT start | 51.7% | 9.3% | 9.0% | 2.7% | 4.2% | 3.9% | 19.3% |
| A-learnable, GRPO @75 | 86.1% | 2.3% | 2.3% | 0.6% | 2.0% | 1.3% | 5.4% |
| B600, SFT start | 24.8% | 15.9% | 16.2% | 7.1% | 5.7% | 2.6% | 27.4% |
| B600, GRPO @75 | 26.6% | 14.3% | 15.9% | 7.4% | 7.0% | 1.8% | 26.8% |

**2. The transfer ladder.**

| test set | n | SFT start | GRPO-lo @75 | GRPO-lo @150 | RFT-A | teacher SFT-A | GRPO-hi @75 |
|---|---|---|---|---|---|---|---|
| A-learnable (trained) | 576 | 48.8% | **85.4%** | **88.1%** | 68.3% | 47.3% | 63.7% |
| A0-shared (never solved, chunk trained on) | 1,041 | 1.8% | **10.9%** (+9.1 [+7.5, +10.8]) | 12.5% | 10.6%* | 8.7%* | — |
| A0-unshared (never solved, chunk not trained on) | 231 | 1.1% | 1.7% (+0.6) | 1.7% | 4.0%* | 1.9% | — |
| A8-solved | 179 | 99.4% | 97.6% | 97.6% | 96.6% | 94.5% | — |
| B600, sometimes solved (1–7/8)† | 201 | 48.1% | 50.8% (+2.7 [−2.1, +7.9]) | 44.3% | 46.3% | 38.9% | 30.6% |
| B600, never solved (0/8)† | 364 | 3.2% | 4.8% (+1.6 [+0.1, +3.4]) | 3.6% | 3.4% | 3.1% | 4.7% |
| B600, all | 600 | 23.9% | 25.5% (+1.6 [−0.4, +3.7]) | 22.2% | 23.4% | 19.9% | 16.9% |
| real | 153 | 19.7% | 19.8% (+0.1) | 18.1% | 19.9% | 16.4% | 13.0% |

\* includes problems the arm trained on directly. † strata from an independent base sample
(8 at T = 0.8; `tools/filter_learnable_prompts.py on B600`).

How to read the table:
- Problems in the pool's papers that resemble pool problems improve.
- Problems from other papers start at the same accuracy as the pool (the 48% strata) and still
  barely move, so difficulty is not the limiter.
- The high learning rate costs every set outside the pool.
- Teacher SFT at lr 2e-5 for 2 epochs is too short to fit its own problems (47.3%) and costs B600
  4 pp: a small continued-SFT set without replay mostly perturbs. Replay, or more epochs at a
  lower rate, would be needed for it to teach.

**3. Gains follow similarity to the nearest trained question.** Gain over the SFT start, by
nearest-neighbour similarity bin:

| similarity bin | A0-shared, GRPO on A @75 | B600, GRPO on A @75 (other papers) | B600, GRPO on B-other @150 (same papers) |
|---|---|---|---|
| < 0.2 | +1.0 [−1.0, +3.2] (n 85) | −1.3 [−3.8, +1.1] (n 235) | +1.3 [−1.5, +4.2] (n 154) |
| 0.2–0.35 | +5.0 [+3.1, +7.3] (n 382) | +0.3 [−2.9, +3.3] (n 272) | +7.3 [+3.8, +11.3] (n 218) |
| 0.35–0.5 | +9.9 [+6.9, +13.0] (n 328) | +9.7 [+2.5, +17.6] (n 76) | +11.7 [+4.7, +19.5] (n 96) |
| ≥ 0.5 | +17.1 [+12.6, +21.8] (n 246) | +27.2 [+10.3, +45.6] (n 17) | +23.6 [+15.7, +31.2] (n 132) |

The three series agree: a gain is set by how close a question is to something the model
practised on, whichever paper it comes from. In the middle bin (0.2–0.35), same-paper problems
gain more (+5.0, +7.3) than other-paper problems (+0.3). At moderate word overlap, a shared topic
adds something on top.

*Mechanism.* Under a gradient step on example x, the change in the model's output on x′ is, to
first order, η · K(x′, x) · ∂ℓ/∂f(x), where K is the network's tangent kernel ([Jacot et al.
2018](https://arxiv.org/abs/1806.07572); the same quantity drives influence functions, [Koh &
Liang 2017](https://arxiv.org/abs/1703.04730)). Updates spread in proportion to representational
similarity. Word overlap is a crude observable proxy for K, and it already separates the
outcomes cleanly.

**4. RL on the same papers lifts new questions about them by ~10 pp.** GRPO on the 230
B-other-learnable problems:

| | B600 at T = 0.8 (trainer eval, acc@1%) | B600, deployment decoding | real |
|---|---|---|---|
| start | 19.5% | 23.9% | 19.7% |
| step 25 | 27.7% | — | — |
| step 50 / 75 / 100 | 26.4% / 28.3% / 27.8% | 31.3% / 31.6% / 31.9% | 17.6% / 18.3% / 17.0% |
| step 125 / 150 | 29.2% / — | — / **33.9% (+10.1 [+7.4, +12.8])** | — / 16.3% (−3.4 [−8.1, +1.2]) |

For comparison, GRPO on A reaches 20.8% at best on the T = 0.8 eval. Most of the gain arrives
within 25 steps.

**5. What carries over is the program-writing skill; what stays local is the setup.** Accuracy =
P(runs) × P(correct | runs). Plain sampling at T = 1, 200 problems × 4
(`scripts/experiments/locality.py`):

| set | runs: SFT → GRPO | correct given runs: SFT → GRPO | accuracy |
|---|---|---|---|
| trained (A-learnable) | 70.0% → 86.0% | 49.6% → 76.9% | 34.7% → 66.1% |
| other papers (B600) | 59.5% → 64.1% | 27.3% → 27.7% | 16.3% → 17.7% |
| real | 53.1% → 56.4% | 24.6% → 23.8% | 13.1% → 13.4% |

- **Crashes fall everywhere.** On B600: misspelt symbols 7.6% → 4.6% of samples; never-defined
  symbols 12.2% → 10.0%. On A the same classes halve.
- **Recovered programs are not more often right off-pool.** On B600 the programs that no longer
  crash are right at the usual 27% rate, so the net accuracy gain there is about 1 pp.
- **Shared constants are the exception.** On B600 problems whose teacher program needs a number
  the question does not state (318 of 600), GRPO @75 gains +4.1 pp [+1.3, +7.0]. On problems
  that state every number (282) the change is −1.2 pp [−4.0, +1.4]. This gain fades by step 150
  (−0.8 pp).

**6. The update is global; only pool-like setups win.**
- **The policy moved everywhere.** The exact per-token KL(GRPO ‖ SFT), summed over the
  vocabulary along GRPO samples, is 0.061 nats on pool prompts, 0.058 on B600 and 0.058 on real.
- **It also discriminates off-pool.** On SFT samples for unseen prompts, GRPO lowers wrong
  solutions more than right ones. Per-token Δlog p on B600: correct −0.024, runs-but-wrong
  −0.054, crashes −0.072 (pool: −0.047 / −0.111 / −0.132).
- **What it learned there.** It learned to down-weight surface features of failure (long
  working, undefined symbols). On topics far from the pool, the probability freed from those
  goes to other wrong setups, because no correct setup there is being reinforced.

**7. Difficulty is not what limits transfer.** Gain by the number of computed steps in the
teacher's solution (`scripts/experiments/difficulty.py`):

| solution steps | A-learnable (in pool) | B600, GRPO on A | B600, GRPO on B-other |
|---|---|---|---|
| 1 | +28.4 (n 51) | +0.2 (n 77) | +14.9 |
| 2 | +40.5 (n 172) | −2.7 (n 121) | +7.5 |
| 3–4 | +36.5 (n 266) | +2.2 (n 265) | +8.6 |
| 5+ | +34.3 (n 87) | +5.1 (n 137) | +12.4 |

- **Inside the pool, every difficulty level gains 28–41 pp.** The SFT start's accuracy does fall
  with steps (on B600: 34–38% for 1–2 steps, 17–18% for 3+).
- **Off the pool, no step count gains much.** Similarity, not step count, decides transfer.
- **The real questions have the same mix.** Real held-out problems have the same step
  distribution as A and B (median 3 steps). On them, v11/v12 gains are spread across step counts
  and too noisy to rank.

**8. The real questions sit far from every pool.** Nearest-neighbour similarity:

| real questions vs | median | ≥ 0.35 | ≥ 0.5 |
|---|---|---|---|
| A-learnable (576) | 0.23 | 20% | 5% |
| v12 pool (3,031) | 0.27 | 29% | 8% |
| all SFT questions (110,903) | 0.32 | 44% | 20% |

B600 is closer to the SFT data (median 0.38) than the real questions are.

#### Interpretation

- **GRPO works on this model.** Given verifiable problems in its learnable band, a 113M model
  goes from ~49% to 85–88% on them, with every error class falling. This is the small-model
  GRPO result.
- **Its gains travel by similarity.** The rule is the same within a paper and across papers,
  and matches the first-order picture of how a gradient step generalises.
- **Small-model GRPO benchmarks draw train and test from one generator**, so every test problem
  has a close neighbour in training. The real held-out questions here come from papers the model
  never practised on, and they are far from every RL pool (median similarity 0.23–0.27). That
  predicts the small held-out gains measured (+0.1 to +3.3 pp).
- **What stays local is knowledge.** After SFT and execution, the remaining errors are about
  which relation, constants and units apply to a topic. RL sharpens these where it practises,
  and shared constants travel. It does not supply knowledge of a topic it has not practised on.
  This is consistent with RLVR mainly turning pass@k into pass@1
  ([Yue et al. 2025](https://arxiv.org/abs/2504.13837)). It is also consistent with small models
  not transferring QA across facts seen in a single phrasing (Allen-Zhu & Li, *Physics of
  Language Models* 3.1).

#### What follows

1. **Coverage is the lever.**
   - An RL pool lifts the questions near it: roughly +10 pp at similarity 0.35–0.5 and +17 to
     +24 pp at ≥ 0.5.
   - To lift held-out questions, a pool must be broad: tens of thousands of distinct problems
     across the domain's question types, each seen about once. Most of the gain arrives in the
     first 25 steps; repetition adds memorisation.
2. **Train on the target documents when the use case allows it.** RL on problems written from a
   paper lifts new questions about it by ~10 pp. For a model that answers questions about a
   known corpus, generating the RL pool from the whole corpus is the natural recipe. It changes
   what a held-out split measures, so it is a protocol decision.
3. **Stepping stones for the hardest questions.** About 60% of realistic questions are never
   solved and give no gradient. Sub-step versions bring them into the learnable band (see
   "Simpler questions" below: never-solved problems whose first step was practised rise 1.5% →
   10.8%).
4. **The bound.** On the real questions, 56% are never solved in 32 samples, and
   majority-of-32 is 23.5% against 16.7% pass@1 at T = 0.8. Without new knowledge, RL can at most
   close the part of that gap that lies near its pool.

#### Simpler questions: difficulty, transfer and a curriculum effect

**Question.** Are the questions too hard for RL, and would simpler ones let it improve more?

**Construction** (`tools/gen_stepping_stones.py`; `data/rl/simple_{A,B600,A_learnable}.jsonl`):
- **Choosing the step.** From each teacher program, pick the computed variable of dependency
  depth 1–2 with the most operations, provided every given value it depends on is stated
  verbatim in the original question.
- **Writing the question.** The 27B writes a short self-contained question asking only for that
  quantity, in the same scenario, stating its givens and the answer unit, and keeping a formula
  only if the original stated it. Questions that drop a needed given value are discarded.
- **Label.** The variable's value from executing the original program, so it is exact.
- **Yield.** 2,627 problems → 1,987 plannable → 1,959 questions: 1,532 simple-A and 427
  simple-B600.
- **Learnability filter on simple-A** (8 at T = 0.8): pass histogram 790 / 140 / 86 / 61 / 45 /
  52 / 56 / 59 / 243 for 0–8 passes. That gives 499 learnable, mean pass 0.296 (original A:
  0.207; 63% never solved, versus 52%).
- **Training.** GRPO with the transfer-lo recipe (`configs/rl_grpo_simpleA.yaml`), 150 steps.

| deployment decoding, 2 seeds × 4 | SFT start | GRPO on simple-A @75 | @150 | GRPO on original A @75 |
|---|---|---|---|---|
| simple-A-learnable (pool) | 47.0% | **90.3%** (+43.3 [+40.0, +46.9]) | 93.2% | 54.6% (+7.6) |
| simple-B600 (other papers) | 34.5% | 37.5% (+3.0 [−0.3, +6.2]) | 35.4% (+0.9) | 32.7% (−1.8) |
| B600, original | 23.9% | 22.4% (−1.5 [−3.5, +0.4]) | 20.1% (−3.7) | 25.5% (+1.6) |
| real | 19.7% | 21.0% (+1.3 [−2.4, +4.7]) | 19.4% | 19.8% |
| A-learnable, original | 48.8% | 47.5% (−1.3) | 45.2% | **85.4%** |
| A0-shared, original (never solved) | 1.8% | 6.0% (+4.2 [+3.0, +5.4]) | 6.3% | 10.9% (+9.1) |

The trainer's eval on simple-B600 (T = 0.8) went 28.6% → 32.4% (step 25) → 32.9% (step 125).

*The curriculum effect* (figure `assets/rl_curriculum.png`). A0-shared problems whose own first step was in the simple pool rise from
1.5% to 10.8% (+9.3 [+5.7, +13.2], n = 205); the rest gain +2.9 [+1.9, +4.1] (n = 836).
Practising the first step makes a never-solved full problem about 7× more likely to be solved.
For A-learnable problems (already in the band), practising their first step gives −1.1 pp
(n.s.).

On simple-B600, the gain does not order cleanly by nearest-neighbour similarity: +5.5, +3.8,
−6.2 and +6.2 pp across the four bins, with small n in the upper two. Short questions make word
overlap a noisier proxy, and single steps are more generic than whole problems.

**Reading.**
- *Simple questions are easier and more learnable.* The start is +10.6 pp on other-paper sets,
  and the pool reaches 90–93%.
- *They carry over slightly further to other papers*: +3.0 pp versus +1.6 pp for original
  questions, both borderline.
- *They do not by themselves lift the original multi-part questions* from other papers.
- *Their real value is as stepping stones.* They give gradient to problems that have none,
  lifting never-solved problems 7×.

Recommended curriculum: full problems in the learnable band, plus sub-step versions of the
never-solved ones, refreshed as the model improves.


#### Knowledge-augmentation pilot: reading versus practice

**Question.** Can the model learn a paper's setups by reading restatements of it, without
question-answer training on it?

**Setup** (`scripts/experiments/gen_notes.py`, `notes_sft.py`, `pilot_chain2.sh`):
- **Recovering the source text.** The source chunks of A and B were recovered from the
  pretraining corpus with the chunking and id rule of `src/sft/generate_sft_data.py`
  (`scripts/experiments/recover_chunks.py`): 400 of 425 ids; 87 of B600's 92 chunks; 105 of A's 115).
- **Notes.** The 27B wrote 8 restatements per chunk (1,536 prompts → 1,447 notes of ~2,000
  characters, 354 tok/s). Rules: keep every equation, value and unit; paper notation; no
  questions or worked examples. About 3% are statements that the chunk (a bibliography) has no
  quantitative content.
- **Arms.**
  - K: `configs/pilot_K_notesB.yaml`, 658 B-notes + 3,000 replay pairs from
    `train_pot_notation2`.
  - C: `pilot_C_notesA.yaml`, 658 A-notes + the same replay.
  - Both: 3 epochs, lr 5e-5, from `sft_pot_notation2`.
- **Evaluation.** Deployment decoding, 2 seeds × 4:
  - B-rec: 360 B600 rows whose chunk text was recovered; 21 have the target value in the chunk.
  - A-rec400: 400 sampled A rows.
  - real.

| | K (notes on B) | C (notes on A) |
|---|---|---|
| B-rec, start 14.3% | 14.5% (+0.2 [−1.9, +2.1]); +0.1 without answer-in-chunk rows | 14.6% (+0.3 [−1.7, +2.3]) |
| A-rec400, start 18.1% | 16.8% (−1.4 [−3.2, +0.3]) | 15.5% (−2.7 [−4.5, −1.0]) |
| real, start 19.7% | 20.1% (+0.4) | 17.6% (−2.1) |

**Reading.**
- The crossover the hypothesis predicts (K > C on B, C > K on A) does not appear. Notes about a
  paper do not raise accuracy on questions about it, and continued training on prose slightly
  perturbs QA (both arms lose on A).
- Against the B-other GRPO arm (+10.1 pp on B600 from 230 practice problems on the same papers),
  practice in the answer format is far more effective than reading at this scale.
- This is consistent with Allen-Zhu & Li's observation that knowledge becomes extractable when
  it is trained in the form in which it is queried (mixed QA + text training).

**Caveats.** About 24 exposures per chunk is low by knowledge-storage standards, and the notes
were not mixed with QA on the same papers. The informative next variant interleaves notes with
practice problems on the same chunks.


### 15.18 Deployment switch, GRPO v13 (broad pool + stepping stones), and the README reorganisation (2026-10-17)

**Deployment.** The served model is now `sft_pot_notation2` with vote@8.
- The deployed checkpoint and its decoding contract live in `configs/deploy.yaml`:
  - executed paper-notation equations at `</think>`;
  - T 0.3, top-k 40, rp 1.0;
  - vote@8 over the executed answers.
- `python -m src.sft.chat` serves it when no `--ckpt-path` is given (`load_deploy_config`,
  `SFTChatEngine.decoding`). The REPL and `--test` vote over executed answers.
- An explicitly named checkpoint keeps the prose decoder (`--decoder pot_notation` overrides).
- Measured: 19.7% single-sample and 22.2% vote@8 on the 153 real held-out numerical questions
  (6 seeds × 4, `docs/eval_numerical_real6_final.json`). The previous default (GRPO v4,
  `checkpoints/grpo/selected.pt`, kept as the Phase III selection) scores 6.1% / 8.1%.
- Smoke tests:
  - The README Tohoku example returns 5.914e4 km² (target 59,100), with 8 of 8 samples agreeing.
  - A fault 40 × 15 km with 1.2 m slip at 30 GPa returns Mw 6.856 (correct).

**GRPO v13 design** (`configs/rl_grpo_v13_broad.yaml`, chain `scripts/experiments/v1314_chain.sh`). This
applies §15.17's rule: the pool must be broad, seen about once, and give the never-solved problems
stepping stones.
- **Existing unseen candidates** (`tools/build_rl_pool.py candidates`): v9/v11/v12 candidate pools,
  25,471 rows.
  - 20,670 are near-duplicates of a notation2 SFT question (word-Jaccard ≥ 0.8). Most are
    counterfactual number-variants of templates the student has already trained on; Jaccard over
    words ignores numbers, so number-variants count as duplicates by design.
  - 1,075 duplicates and 52 rows from held-out chunks are also removed.
  - 3,674 candidates remain.
- **New teacher problems:**
  - round-3 real-style rewrites (3 per real numerical training question);
  - 3,000 × 3 exercises seeded from all training topics;
  - converted and deduplicated the same way (`build_rl_pool.py teacher`).
- **Stepping stones.** Up to 8,000 never-solved teacher problems with programs get first-step
  questions (`tools/gen_stepping_stones.py`).
- **Pool.** Learnable rows (1–7 of 8 for notation2) from all of the above, plus A-learnable and
  simple-A-learnable, at most 2 per number-masked template. The chain logs the pool's
  nearest-neighbour coverage of the real questions, and the similarity law's predicted gain,
  before training.
- **Training and selection.**
  - One pass: 32 prompts per GPU × 2 GPUs per step.
  - Checkpoint chosen on a dev set: `val_numerical_dev` + B600; B chunks are kept out of the
    pool.
  - Real test: 6 seeds × 4 (seeds 31–36, as for the final standing).

**Corpus protocol arm, GRPO v14** (added the same evening at the user's request; chain
`scripts/experiments/v1314_chain.sh` replaced the strict-only chain).
- **Source text.** The held-out source chunks were recovered from the pretraining corpus: test 205
  of 215, val 196 of 196.
- **Exercises.** `tools/gen_chunk_problems.py` has the 27B write 8 exercises per excerpt, with
  executable programs. It drops any exercise that shares a 13-gram with a held-out question or
  has word-Jaccard ≥ 0.5 with one.
- **Pool.** v14 is the v13 pool plus these exercises (learnable ones) and their stepping stones.
  B's chunks stay out of both pools, so B600 remains an untouched check.
- **Training and comparison.** Both runs use the same recipe and the same dev-selection rule. The
  real test scores the two protocols side by side.

**README reorganisation.** To keep the main story readable:
- **§1.2.** Keeps the corrected-label comparison. The old-label tables (a)/(b) and the
  unreproduced RAG column are now pointers to PITFALLS C3 and §15.6.
- **§5.12–5.13.** The generic overtraining and reward-hacking tutorials became short watch lists
  pointing to §4.7, §5.6 and §5.8.
- **§7.** Rewritten around the current results; the first evaluation stays in §10–§14 and
  PITFALLS.
- **Next-experiments section removed** (a later pass the same evening). The README now opens with
  highlights, and its §1.2 headline figure is regenerated from the 6-seed results of the current
  models (`fig_grpo_benchmark_comparison`). The SFT dynamics figure stops at step 200 (best
  checkpoint: step 80).
- **README restructure** (a further pass). The README opens with a plain summary that follows the
  pipeline figure, then §1 by phase, with the example answers drawn as an image
  (`assets/stage_outputs.png` from `docs/stage_outputs.json`). §5.14–5.18 are rewritten as short
  observation → bottleneck → method → result steps. Claims about honesty, the contrast framing and
  dates are removed, and the figures carry fewer words.
- **New figures, read from results files at plot time:**
  - `assets/workflow.png`: the pipeline with each stage's measured numbers. It replaces a
    hard-coded diagram that showed old-label numbers and the unreproduced RAG figure.
  - `assets/numerical_track.png`: accuracy at every stage, with CIs and vote@8.

### 15.19 GRPO v13 and v14 results, a CUDA-graph memory fix, and README additions (2026-10-17)

**Runs.** The two arms planned in §15.18. Both use the v12 recipe (correctness reward, T = 0.8,
lr 5e-6, β 0.02, TIS cap 2), start from `sft_pot_notation2`, and make one pass over their pool
(64 prompts per step):

| arm | pool | steps |
|---|---|---|
| v13, strict | 9,031 learnable problems. Existing candidates 1,171; teacher rewrite round 3 and topic exercises 5,481; stepping stones 1,385; A 576; simple-A 499 | 142 |
| v14, corpus | the v13 pool plus 650 exercises written from held-out excerpts and 302 of their stepping stones (9,983) | 156 |

`tools/build_rl_pool.py` coverage of the real held-out questions:
- median nearest-neighbour Jaccard: 0.28 (v13) and 0.30 (v14);
- share at 0.35 or above: 34% and 37%;
- similarity-law prediction: +5.5 and +5.8 pp, an upper end that assumes every neighbour is learned.

**Incident.** The first launch died with CUDA out-of-memory errors: v13 after step 50, v14 after
step 75. The crash came inside `GraphDecoder.capture`. Each captured graph pins a static KV cache
(about 1.6 GB at 128 rows × 768 tokens per GPU). The cache held up to 6 graphs, and with the
pools' many prompt lengths plus the eval batch shape, it filled the 24 GB A30. The fix in
`src/rl/rollout.py`: keep at most 2 graphs, evict the least recently used one, and call
`gc.collect()` and `torch.cuda.empty_cache()` before the next capture. A smoke test over 9 batch
shapes holds 2 graphs and at most 8.5 GB reserved. The partial runs are kept in
`checkpoints/grpo_v1{3_broad,4_corpus}_oom_partial` (held-out: v13-25 20.5%, v14-50 21.9%).
Both arms were then rerun to completion.

**Dev selection** (`val_numerical_dev` + B600, 2 seeds × 4, deployment decoding):
- notation2: 26.26%.
- v13: 26.42 (step 25), 26.40 (50), 25.40 (75), 25.20 (100), 24.43 (125), 23.26 (142).
- v14: 26.28 (25), 26.24 (50), 26.62 (75), 24.14 (100), 25.38 (125), 25.09 (150), 24.12 (156).
- Selected: v13-25 and v14-75.

**Real held-out** (153 questions, 6 seeds × 4, `docs/eval_v13_v14.json`; differences are paired
against notation2):

| model | acc@5% | vs notation2 | must-compute (134) | answer in prompt (19) | vote@8 |
|---|---|---|---|---|---|
| notation2 | 19.66 | — | 20.1 | 16.7 | 22.2 |
| v13-25 (selected) | 20.1 | +0.4 [−2.5, +3.2] | 18.8 | 29.2 | 21.1 |
| v13-142 (end) | 21.0 | +1.3 [−2.6, +5.4] | 19.7 | 30.0 | 21.6 |
| v14-75 (selected) | 22.0 | +2.4 [−1.5, +6.3] | 20.7 | 31.1 | 23.3 |
| v14-156 (end) | 22.0 | +2.4 [−1.6, +6.4] | 21.3 | 27.2 | 23.7 |

Accuracy on each step's fresh pool problems: v13 43.5% → 48.1% and v14 41.7% → 46.6% (first and
last fifth of the pass). KL ends at 0.039, the zero-variance share averages 0.36–0.37, and TIS
truncation is 0.22–0.23%.

**Interpretation.**
- No held-out gain is significant. The corpus arm is ahead of the strict arm (+2.4 against +0.4 to
  +1.3 pp), and v14-156 with vote@8 is the best measurement so far (23.7%).
- The gains are on the 19 answer-in-prompt questions (+10.5 to +14.9 pp, significant). On the 134
  must-compute questions they range from −1.3 to +1.2 pp. This is the same pattern as v12 (+8.6
  and +0.3 pp).
- The measured gain is a fraction of the similarity-law prediction. The transfer study practised
  576 problems about 8 times each (75 steps × 64 prompts), lifting them from 48.8% to 85.4%. These
  arms practise each problem once, and even fresh in-distribution problems gain only about 5 pp
  over the pass. The law's per-bin gains were measured under repeated practice, so they do not
  carry over to one-pass breadth.
- Dev (mostly B600) falls 2–3 pp over the pass while the real held-out questions do not. Dev
  selection therefore picks early checkpoints, which score the same as the final ones on the real
  test.
- **Deployment switched to v14-75 + vote@8** (`configs/deploy.yaml`), on request. It is the
  dev-selected checkpoint; v14-156 scores higher on the real held-out set (23.7%), but choosing it
  there would be test-set selection (PITFALLS E2). Compared with notation2 + vote@8, the switch
  gives 23.3% against 22.2% overall and 20.5% against 18.1% within 1%, but 21.6% against 22.4% on
  the must-compute questions. The headline figure, pipeline figure and track figure were
  regenerated (`docs/eval_numerical_real6_final.json` gains the v14-75 runs).

**README additions the same evening.**
- §5.16, the train–inference mismatch, renumbered the old §5.16–5.18 to §5.17–5.19. It is measured
  by `scripts/experiments/mismatch_check.py` (`docs/mismatch_check.json`).
- §5.13 was rewritten to set reward, advantage and evaluation side by side.
- The v13/v14 results above are the last subsection of §5.19, "Breadth without repetition".

### 15.20 Ceilings, decoding routing, repeated practice (v15), and an in-distribution set (2026-10-21)

**Ceilings** (`scripts/experiments/ceiling_check.py`, `docs/eval_ceilings.json`).
- **Numerical:** 64 samples per real held-out question at T = 0.8. For the deployed model:
  pass@1 19.7, pass@8 37.1 and pass@64 62.7; 37.3% of questions are never solved and 2.6% always
  solved. notation2 before RL: pass@1 18.1, pass@8 35.5 and pass@64 62.7, with the same 37.3%
  never solved. So RL sharpened answers within what the model could already reach and did not
  widen it.
- **Concept coverage:** the share of key concepts in the visible answer, over 641 non-numerical
  held-out prompts with 2 samples each. The reference answers score 100% and the question text
  3.7%. SFT 38.7, GRPO v4 40.0 (its reward had a concept term), the deployed model as served 29.6,
  and the deployed weights with prose decoding 37.0.

**Routing** (`src/sft/question_router.py`, `src/sft/chat.py`). Questions that ask for a number keep
the executed equations with vote@8. The others get prose decoding (rp 1.2, one sample). Tuned on
`data/rl/val.jsonl`. On the held-out set it sends all 153 scored numerical questions to the
equations and 640 of the 641 others to prose. Deployed concept coverage: 29.6% → 37.0%.

**GRPO v15** (`configs/rl_grpo_v15_repeat.yaml`, `scripts/experiments/v15_chain.sh`,
`docs/eval_v15.json`).
- **Setup:** starts from the deployed v14-75. The 4,045 corpus problems were re-filtered against it;
  1,073 are learnable (2,765 never solved, 204 always solved). 135 steps, about 8 visits each.
- **Training:** accuracy on the practised problems rose from 37.4% to 71.3% (first and last fifth).
  KL ends at 0.058.
- **Dev** (`val_numerical_dev`, 4 × 4): deployed 36.8. v15: 28.1 (step 25), 24.5 (50), 20.8 (75),
  19.4 (100), 21.0 (125), 20.3 (135). Selected: the deployed model.
- **Real held-out, step 75:** 21.4%, −0.7 [−4.2, +2.8]. On the 98 questions with practised source
  papers −1.9 [−6.1, +2.0]; on the 55 without +1.6 [−5.2, +7.8].
- **Reading:** eight visits to teacher-style problems that state every number sharpen that style and
  lose ground on real questions, and on the training distribution (below).

**In-distribution set** (`scripts/experiments/indist_chain.sh`, `data/eval/indist_300.jsonl`,
`docs/eval_indist.json`).
- **Construction:** 853 new 27B problems seeded by SFT-train questions (`generate`, seed 9001).
  766 have a label confirmed by an independent re-solve within 1%, and 762 remain after removing
  near-duplicates (word-Jaccard ≥ 0.8) of every SFT and RL training set. 300 were sampled with no
  solve-rate filter.
- **Results** (6 × 4, acc@5%):
  - SFT 2.0 and v4 1.8 (prose; trained on another style);
  - notation2 35.5 (vote@8 38.2);
  - v14-75 37.6 (vote@8 38.9);
  - v15-135 32.7.
- **README:** a column in §1.2 and a panel in the headline figure.

### 15.21 Concept coverage by phase (2026-10-21)

**Concept coverage by phase** (`docs/eval_ceilings.json`, 641 conceptual held-out prompts × 2,
prose decoding). SFT 38.7, v4 40.0, notation2 36.9, v14-75 37.0, deployed (routed) 37.0. The
question text alone names 3.7% of the key concepts, the reference answers 100%. Added as a column in
README §1.2 and a panel in the headline figure; the figure's "must compute" group was dropped to keep
it narrow (the column stays in the table).

**Why GRPO and routing barely move it.** Routing only restores prose decoding (29.6 → 37.0 is the
recovery from the equation decoder, not a gain over SFT). In GRPO v4, 81% of the conceptual prompts
were SFT training questions, and the concept reward is coarse (tiers 0.8 / 0.4 / 0.1 / −0.5) next to
the format reward. On its own training questions SFT names 42.3% (400 prompts × 2), so the model
cannot reproduce most concepts even for questions it was trained on: capacity, not the reward, is the
main limit.
