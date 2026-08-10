# 🔍 Experimental ledger: Phase II SFT + Phase III verifier-RL pipeline

What this document is for. It records what actually ran: every audit finding, every
remediation, every measurement, in the order it happened — including superseded numbers and
why they were superseded. It is the evidence trail, not the explanation.

- [README.md](README.md) — how the pipeline works, why these choices, what the corrected
  experiment showed.
- [PITFALLS.md](PITFALLS.md) — the failure atlas: symptoms, false stories, root causes,
  regression tests, and a status table for every superseded number.

> Update (2026-07-29, same day): every fix in §5 has been implemented and the pipeline re-run.
> See [§6 Remediation log](#6-remediation-log-what-was-actually-changed) at the end of this document
> and README [§7 What it found](README.md#7-what-it-found) for the current results.

> Naming note (2026-07-31). Two renamings apply throughout, and historical text below
> predates both.
> 1. Stages → phases. "Stage 2 / Stage 3" below mean Phase II (SFT) and
>    Phase III (verifier RL); Phase I is pretraining in [nanoGPT-Seis](https://github.com/jiazhe868/nanogpt-seis).
> 2. "GRPO" is a directory name, not an algorithm claim. What is implemented is
>    group-relative REINFORCE with KL regularization — group-relative advantages plus
>    reference KL, but no importance ratio and no clipped surrogate, one on-policy update per
>    sampled batch. See README [§5.3](README.md#53-how-this-differs-from-canonical-grpo).
>
> Run directories were also renamed for clarity — `sft_v6` → `checkpoints/sft`,
> `grpo_v3` → `checkpoints/grpo`, and every other run's log moved to
> `checkpoints/runs/<descriptive>.csv`. Historical run names below (`grpo_v1`, `grpo_v2`,
> `ab_16k`, `strat_*`, …) refer to those logs.

> Module map (2026-08-08). Scripts named below were later consolidated. Where they live
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
| Historical audit snapshot | superseded — retained for provenance | §§1–5 |

Section status labels: §§1–5 HISTORICAL (retained for provenance) · §6 remediation
log · §7 SUPERSEDED BY §10.2 · §§8–12 historical measurements, individually dated ·
§13 SUPERSEDED IN PART BY §14 · §14 CURRENT AS OF 2026-08-08.

---

Date: 2026-07-29
Scope: Code correctness, data quality, evaluation validity, and improvement plan under current constraints (2× A30 24 GB, 823M-token pretrain corpus, a teacher API for data generation).
Method: Three parallel deep code/data audits + first-hand generation probes on both checkpoints (temperature 0.2).

---

## Original audit snapshot — 2026-07-29, superseded

> HISTORICAL. Written before any remediation. Present-tense claims below describe
> the pipeline as it stood on 2026-07-29. For the current state see Current status
> above; every fix is logged in §6 and the current results are in §14.

The pipeline is pedagogically well-built — label masking, weight-tied embedding resize, on-policy GRPO with k3 KL estimator, and advantage normalization are all implemented correctly. However, the three headline results of this repo are not currently trustworthy:

1. SFT val loss 2.1237 and the data-scaling law are inflated by 100% document-level train/val leakage. The split shuffles Q&A pairs, not source chunks; every val question has ~1.9 sibling Q&A pairs from the same paper excerpt in train.
2. GRPO never trained on numerical accuracy at all. The `NumericalVerifier` reads a field (`final_numerical_value`) that exists in 0 of 16,425 RL records — `num_reward = 0.0000` at every one of the 300 logged steps. Root cause: `prepare_rl_dataset.py` calls `math.isnan` without importing `math`, so the fixed "V2 hack-proof" dataset was never generated and training silently used the stale schema.
3. The "+590% improvement" is circular. `eval_rl.py` scores with the same `CompositeRewardEngine` used as the training reward, on only the first 20 val prompts, 1 sample each; the comparison numbers are additionally hardcoded in `plot_grpo_results.py`. ~73% of the measured reward gain is `<think>`-tag format compliance.

Qualitative reality check (my own probes at temperature 0.2): both SFT and GRPO models produce fluent seismology-register prose that is factually wrong. GRPO genuinely improved surface form (correct M₀ = μAD and M_w = ⅔log₁₀M₀ − 10.7 statements, no repetition loops) but not factual content (e.g., both models place deep-focus earthquakes at 10–30 km; the real answer is 300–700 km via transformational faulting / dehydration embrittlement).

---

## 1. Confirmed bugs — SFT pipeline

> HISTORICAL — retained for provenance. Every item here was fixed; see §6.

| # | Severity | Bug | Location |
|---|---|---|---|
| S1 | 🔴 | Train/val split is pair-level; `chunk_id` discarded before the shuffle → 864/864 val pairs share a source chunk with train (mean 1.88 train siblings each). All val-based conclusions (LR sweep ranking, scaling law, early stopping step) are contaminated. | `format_sft_dataset.py:64-80` |
| S2 | 🔴 | Val loss is never all-reduced across DDP ranks; `DistributedSampler` shards val, so best-checkpoint selection and every logged val number use only rank 0's half of the val set. | `train_sft.py:202-241, 304, 427-441` |
| S3 | 🟡 | Val loss = mean of per-batch means over highly variable unmasked-token counts (padding tail blocks get equal weight). A batch with zero unmasked targets yields NaN and silently disables best-checkpoint saving forever. | `train_sft.py:237-241` |
| S4 | 🟡 | Sequence packing: (a) no block-diagonal mask → cross-sample attention leakage in train and val; (b) samples split mid-response across block boundaries keep their labels, so responses are trained with amputated prompts. | `dataset.py:197-226` |
| S5 | 🟡 | Gradient-accumulation: final partial group of each epoch still divides by full `grad_accum` → systematically undersized last step per epoch. | `train_sft.py:391-397` |
| S6 | 🟡 | Multi-turn inference encodes history assistant turns with an extra `\n` before `<|im_end|>` never seen in training (single-turn is consistent). | `chat.py:95` vs `dataset.py:108-109` |
| S7 | 🟢 | REPL ignores `--temperature/--top-k` (hardcoded); default `no_repeat_ngram=3` is hostile to formulaic scientific text and caused the observed word-salad corruption at eval; base-checkpoint fallback would crash on ChatML token IDs (no resize to 16,388). | `chat.py:347, 190, 132-167` |
| S8 | 🟢 | Data-scaling sweep: hardcoded subsets, final point mislabeled "12.6k (Full)" but is 15k (actual full = 16,425); fixed 3 epochs conflates data volume with optimizer steps. Configs `lr_sweep.yaml`/`data_scaling.yaml` are read by nothing. | `run_data_scaling_sweep.py:26, 81, 153` |

Done correctly (verified): label shift and prompt masking (no off-by-one; `<|im_end|>` in loss span), embedding resize preserving weight tying with optimizer created after resize, fp32 cross-entropy under bf16 autocast, gradient clipping on deduplicated params, RoPE positions with KV cache, stop condition on `{16385, 0}`.

## 2. Confirmed bugs — data quality

> HISTORICAL — retained for provenance. Every item here was fixed; see §6.

| # | Severity | Finding |
|---|---|---|
| D1 | 🔴 | ~60% of train records still contain source references after "cleaning" ("the study" 21%, "the text" 14%, "et al." 12%, Figure/Table N ~11%, "the authors" 7%…). The regex list only covers a narrow set of lead-ins. |
| D2 | 🔴 | `<think>` blocks were exempted from cleaning (`clean_sft_data.py:77-82`): 46% of numerical reasoning chains cite invisible tables ("From Table 3: slip rate = 90 mm/yr…") — this directly trains citation hallucination inside CoT. |
| D3 | 🟡 | Regex cleaning damaged grammar: ~6.4% of responses opened with verb-deleted fragments, ~26% of instructions have dangling clauses ("Summarize the key findings and methods used of the 2014 Iquique earthquake"), and `\s+→' '` flattening removed every newline in every response — the model was fine-tuned to emit single-paragraph walls of text. |
| D4 | 🟡 | RL dataset is a 100% positional copy of the SFT train set (same prompts the policy was already fit on); `boxed_target` holds full prose (not a number); `target_numbers` is a noisy dump (avg ~20 numbers/item incl. years and intermediate steps). |
| D5 | 🟢 | Good news: 0 duplicate pairs, 0 exact/near-duplicate instruction leakage, 98–99.5% of numerical answers end with a verifiable number, and spot-recomputation of 8/10 sampled teacher derivations was arithmetically correct. |

## 3. Confirmed bugs — GRPO pipeline

> HISTORICAL — retained for provenance. Every item here was fixed; see §6.

| # | Severity | Bug | Location |
|---|---|---|---|
| G1 | 🔴 | `NumericalVerifier` reads `rl_item["final_numerical_value"]` — present in 0/16,425 records → numeric reward dead for the entire run (`num_reward=0.0000`, all 300 steps in `grpo_log.csv`). The 33% "numerical" prompts were scored by keyword overlap instead. | `verifiers.py:190` |
| G2 | 🔴 | Root cause of G1: `prepare_rl_dataset.py` uses `math.isnan` with no `import math` → `NameError` on first numeric answer; the upgraded dataset build crashed and stale old-schema files were used. | `prepare_rl_dataset.py:47` |
| G3 | 🔴 | Circular evaluation: `eval_rl.py` scores SFT vs GRPO with the training reward itself, first 20 val prompts, 1 completion each, temp 0.3, no error bars; the "+0.05 → +0.295 (+590%)" numbers are hardcoded in `plot_grpo_results.py:211-212`. | `eval_rl.py:66,124` |
| G4 | 🟡 | Reward is hackable and partially hacked: `KeyConceptVerifier` matches against the full text including `<think>` — stuffing all 8 keywords in a think block scores the 1.5 ceiling; `key_concepts` are just the reference's top-8 frequent words (ROUGE-1 in disguise; "earthquake"/"earthquakes" counted as two concepts). ~73% of the run's reward gain was `<think>`-tag formatting. | `verifiers.py:98-102`, `prepare_rl_dataset.py:54-64` |
| G5 | 🟡 | Forced-EOS padding tokens (appended so groups decode in lockstep) are inside the policy loss and KL: short completions get gradient on hundreds of never-sampled EOS tokens, and they dilute `mask.sum()` normalization. | `train_grpo.py:115, 272-284` |
| G6 | 🟡 | Rollouts sampled at temp 0.7 + top-p 0.95 but the loss uses untempered log-probs with no importance correction (biased off-policy REINFORCE). No KL controller: rolling KL climbed from 0.05 to 0.34 with a spike of 8.23 at step 283; the "peak reward +0.494" at step ~300 coincides with the KL blow-up — late gains are policy drift, not improvement. | `train_grpo.py:98-112, 264-276` |
| G7 | 🟡 | GRPO rollouts and eval omit the system turn used in every SFT training example → both RL training and evaluation run the model off-distribution. | `train_grpo.py:213`, `eval_rl.py:78` |
| G8 | 🟢 | Config rot: `clip_ratio`, `ppo_epochs`, cosine scheduler, warmup, grad-accum, micro-batch settings in `rl_grpo.yaml` are all unused; actual loss is plain REINFORCE (the "Clipped Policy Gradient Loss" plot title is wrong); actual batch is 2 prompts × G=4 = 8 rollouts/step, not the documented 32. Only ~600 unique prompts were ever sampled from the 16,425-prompt set. | `configs/rl_grpo.yaml`, `train_grpo.py:209, 282` |

Done correctly (verified): group-relative advantage with degenerate-case handling, correct token/logit alignment and prompt masking, k3 KL estimator against a properly frozen SFT reference, strictly on-policy (1 update per rollout batch — no stale-logprob bug class), rewards computed on completions truncated at first `<|im_end|>`, honest Goodhart discussion in the README of the day (that discussion now lives in README §3.2 and §5.5).

---

## 4. Performance evaluation (what the model can actually do)

> SUPERSEDED BY §13 AND §14. Measured before the metric-validity fix, so the
> accuracy figures here are inflated by the min-over-8 scoring rule.

- Reported: val loss 2.1237 (ppl 8.4). Reality: measured on a leaked, half-sharded val set → the true held-out loss is unknown and certainly worse. The monotonic data-scaling curve is partly a leakage artifact (more train data → more sibling coverage of val chunks).
- Style/format (real gains): ChatML compliance, clean scientific register, elimination of SFT-era repetition loops (`M_ij − M_ij`), correct canonical formula statements. GRPO measurably helped here — this is the honest achievement of Stage 3.
- Factual content (no gain): both checkpoints hallucinate specifics (event magnitudes, depths, mechanisms) and cannot do arithmetic. Probes at temp 0.2 confirm this is not a sampling artifact.
- Arithmetic: never rewarded (G1), never learned. No evidence 300 GRPO steps could have taught it even with a live verifier — but the experiment was not actually run.

---

## 5. Improvement plan under current constraints

> HISTORICAL. Implemented in §6; outcomes in §§7–14. The token-per-parameter
> reasoning in this section is superseded by §14.7.

Constraints: 113M model, 823M-token pretrain corpus (≈7 tok/param — itself below the ~20:1 compute-optimal ratio), 2× A30 (48 GB total), teacher-API budget, ~5,860 source chunks / 17.3k Q&A pairs.

### Phase 0 — Fix measurement first (hours, CPU-only mostly)
1. Re-split by `chunk_id` (better: by paper/DOI) before train/val cut. Re-evaluate the existing `sft_ckpt_best.pt` on the new held-out val to get the first trustworthy number.
2. Fix DDP val: `all_reduce` a token-weighted loss sum / token count (fixes S2+S3 together).
3. Fix eval decoding defaults: `no_repeat_ngram=0`, honor CLI args in the REPL, gentle `repetition_penalty≤1.05`.

### Phase 1 — Reframe the task to match a 113M model's capacity (highest impact)
A 113M closed-book model cannot store paper-specific facts; the current data demands it. Two complementary reformulations, both cheap with the existing corpus:
1. Grounded (open-book) SFT: regenerate the dataset with the source excerpt included in the user prompt ("Given the following excerpt: … Question: …"). "The text says…" then becomes correct rather than hallucinatory, and inference matches the RAG deployment path already built (`rag_retriever.py`). Reading comprehension + extraction is squarely within 113M capacity; closed-book recall is not.
2. Closed-book subset restricted to general principles: keep only Q&A answerable from textbook knowledge (definitions, canonical formulas, qualitative mechanisms); drop paper-specific parameter questions from the closed-book split.
3. Replace the regex cleaner with a teacher-model rewrite pass (~17k short calls): rewrite responses as standalone, preserve markdown structure/newlines, clean inside `<think>`, and emit a `\boxed{}` final answer for numerical items. This simultaneously fixes D1–D3 and produces machine-verifiable RL targets.

### Phase 2 — Retrain SFT correctly (half a day on 2× A30)
- Document-level split; no mid-response packing splits (drop or pad the tail; or add block-diagonal masks via `flex_attention`/manual mask — optional given item lengths ~550 tokens);
- fix grad-accum normalization on partial groups;
- keep LR 2e-4 / cosine / early stopping — that part of the methodology was fine, but re-run the sweep on the fixed val before trusting a ranking.

### Phase 3 — Re-run GRPO for a goal it can actually achieve (a day)
1. Regenerate the RL dataset (fix `import math`; extract a single numeric `final_numerical_value` from the rewritten `\boxed{}` answers; verify schema matches `verifiers.py` with an assert at load time).
2. Harden the reward: concept matching on the post-`<think>` answer only (strip think), lemma-deduplicated concept lists, numeric reward on the boxed answer only (not `min` over all numbers), mask forced-EOS padding out of loss/KL, add completion-length + entropy logging.
3. Stability: KL controller or hard early-stop on rolling KL (the current run should arguably have stopped near step ~250); G=8; more than ~600 unique prompts; greedy-consistent objective (sample at the temperature you optimize, or add importance weights).
4. Evaluation must be reward-independent: exact-match accuracy on boxed numerical answers over the held-out split, plus SFT-val cross-entropy drift, plus a small LLM-as-judge factuality score on ~100 prompts. Never reuse the training reward as the benchmark.
5. Realistic target: GRPO on a 113M model can enforce format, brevity, echo-avoidance, and grounded-context faithfulness (with open-book prompts, a verifier can check that cited numbers appear in the provided context — a well-posed, hack-resistant reward). It will not create closed-book knowledge or mental arithmetic.

### Phase 4 — Structural options (pick per project goals)
- Tool use: teach `<python>…</python>` calculator calls inside `<think>` via SFT (the teacher model can rewrite the 5.4k numerical items into tool-call format), execute offline during RL rollouts, reward exact boxed answers. This is the only credible route to numerical correctness at 113M and is feasible on this hardware.
- RAG-first inference: make retrieval the default path; consider upgrading BM25 to a small dense retriever if retrieval quality limits it.
- Scale, if the teaching narrative allows: 2× A30 can full-finetune ~0.5–1.5B (bf16 + grad checkpointing; LoRA trivially). A Qwen2.5-0.5B/1.5B base would remove the fluency ceiling while reusing the entire SFT/GRPO pipeline. Staying from-scratch: the 823M-token corpus is the binding constraint — a bigger from-scratch model would be even more data-starved; multi-epoch pretraining (2–3 epochs) is the cheaper lever.

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
| `src/sft/clean_sft_data_v2.py` | New. Quality filter replacing the v1 rewriter: 2 grammar-preserving rewrites, then drop anything still referencing a phantom source (incl. inside `<think>`); domain filter; newlines preserved | D1, D2, D3 |
| `src/sft/run_data_scaling_sweep.py` | Full-size point resolved at runtime (no more mislabeled "12.6k (Full)"); points to `processed_v2`; documents the fixed-epoch confound | S8 |
| `src/rl/prepare_rl_dataset.py` | `import math` (the root cause); scientific-notation parsing; concepts exclude prompt words and prefix-duplicates; hard schema self-check after writing | G1, G2, G4 |
| `src/rl/verifiers.py` | Concept matching on post-`<think>` text only; number-spam cap; sci-notation parsing; `CleanLanguage` returns 0.0 (not +0.2) when clean; format verifier is task-type aware; 5 adversarial unit tests | G4 |
| `src/rl/train_grpo.py` | Load-time schema assertion; system turn in rollouts; loss/KL masked to actually-sampled tokens; per-sequence-then-group mean; on-policy sampling (T=1.0); without-replacement prompt schedule; adaptive KL + hard early stop; entropy and length logging | G5, G6, G7 |
| `src/rl/eval_rl.py` | Rewritten around reward-independent metrics (numerical accuracy @1%/@5%, `<think>` rate, contamination, echo, length) on the held-out split; training reward reported last, labeled | G3 |
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

Chunk-level split → 4,434 train / 233 val, from 3,111 / 156 disjoint source chunks.
Verified: 0 chunks shared between train and val (v1: 822/822 shared).

The 73% drop is intentional. A closed-book 113M model trained on answers that cite invisible tables
learns to fabricate citations; those pairs are worse than no data. Note the retained set is still
38% `<think>`-bearing and type-balanced (2,217 conceptual / 1,625 numerical / 592 summarization).

### 6.3 Measured results

Old v1 checkpoint re-scored on the clean document-disjoint val set: loss `1.6547` (ppl 5.23) —
lower than its own reported `2.1237`, which only confirms the two numbers measure different things
and that the v1 figure cannot be compared to anything.

SFT v3 (4,434 clean pairs, document-disjoint val, all-reduced token-weighted loss):
best val loss `2.1844` at step 40 / epoch 2, from a 116-step 8-epoch run. Overfitting begins
early (train 1.22 vs val 2.32 by epoch 5) — expected with 4.4k samples, and now honestly visible
because the val set no longer contains siblings of the training questions.

GRPO v2 — 200 steps from `checkpoints/sft_v3/sft_ckpt_best.pt`, G=8, lr 5e-6, live numerical
verifier (99.5% of numerical prompts carry a verified target, vs 0% in v1). Completed without
triggering the KL early-stop (max KL 0.0792 vs threshold 0.5). Reward −0.296 → −0.029 across
50-step segments, with R_num −0.284 → −0.194 and R_format +0.054 → +0.170; entropy (2.57 → 2.66)
and completion length (352 → 374) both stable, so neither entropy nor length collapsed.

Held-out comparison (177 document-disjoint prompts, reward-independent metrics, decoded at
temperature 0.3 / top-k 40 / repetition penalty 1.2): `<think>` rate on numerical prompts 46.2% →
93.2% (z = +7.82, significant); numerical accuracy @1% 15.4% → 12.0% and @5% 18.8% → 15.4%
(down, 18→14 and 22→18 of 117, z ≈ −0.7, not significant); echo 7.3% → 2.8% (z = −1.94,
borderline); contamination 0.0% for both.

The honest reading: **GRPO moved form, not knowledge — and did not improve numerical accuracy at
all.** That is the same qualitative conclusion v1 published, but v1 asserted it on the back of a
fabricated "+590%" produced by a dead verifier and a circular benchmark; here it is measured, with
significance tests printed by `eval_rl.py` on every run so the null results cannot quietly become
wins.

A second measurement trap, found after the first pass. An earlier version of this section
reported `<think>` 8.5% → 55.6% and a small positive numerical delta. Those numbers were taken at
repetition penalty 1.05, where this model emits 56% verbatim-repeated 5-grams and often never
closes its `<think>` tag — the eval was scoring a decoding artifact. Sweeping the decoder over 4
prompts × 250 tokens:

| temperature | rp 1.05 | rp 1.15 | rp 1.25 |
|---|---|---|---|
| 0.2 | 0.562 | 0.162 | 0.000 |
| 0.5 | 0.105 | 0.005 | 0.000 |
| 0.7 | 0.039 | 0.002 | 0.000 |
| 0.9 | 0.007 | 0.003 | 0.000 |

(values = fraction of repeated 5-grams; lower is better). Raising the penalty to 1.2 removed the
loops entirely, roughly doubled measured accuracy for both models (SFT 7.7% → 15.4% @1%), and
flipped the sign of the SFT→GRPO numerical delta. v1 had papered over the same failure with
`no_repeat_ngram=3`, which banned repeated trigrams and produced the `M0_code` / `N·mi` word-salad
instead. `chat.py` and `eval_rl.py` now default to temperature 0.3 / top-k 40 / rp 1.2.

The generalizable lesson: **at 113M, a decoding-parameter mistake can be larger than the training
effect being measured.** Fix and pin the decoder before comparing checkpoints.

### 6.4 Known remaining limitations

- Retention is low (27%). An LLM rewrite pass would recover much of the dropped 73%
  by restating source-dependent answers as standalone knowledge instead of discarding them. No API
  key is configured in this environment, so the deterministic filter was used instead.
- Packing still uses a plain causal mask, so samples within a 4096-token block can attend to each
  other. Mid-response splitting is fixed; block-diagonal masking is not implemented.
- 4.4k SFT pairs is small. The honest fix is more clean data (regeneration with a
  standalone-answer prompt), not more epochs.
- The base model is still 823M pretrain tokens at 113M params (≈7 tokens/param, below
  compute-optimal). This bounds everything downstream.

### 6.5 Additional findings surfaced during remediation

Three issues that the original audit did not catch, found while re-running the pipeline:

1. ~15% of the corpus is off-domain. The pretraining crawl pulled in unrelated open-access
   papers, so the teacher generated Q&A about Amiga operating-system history, medical-negligence
   litigation, Cusco urban studies, and Brettanomyces brewing microbiology. On a 113M domain model
   this is pure capacity waste. `clean_sft_data_v2.py` now applies a domain filter (2,313 pairs
   dropped).

2. Greedy decoding degenerates. At `top_k=1` the model enters repetition loops
   (`(Δt / Δt) * (Δt / Δt) * …`, `G = A * Ω * Ω * Ω * …`) that never close the `<think>` block. A
   "deterministic" greedy evaluation therefore measures a decoding artifact: it reported a 2.6%
   `<think>` rate and 51.4% echo rate versus 8.5% / 47.5% at the deployment configuration. Evaluation
   now uses temperature 0.2 / top-k 20 with a fixed seed.

3. `lr = 1e-6` leaves the policy frozen at this batch size. A 76-step trial at the v1 learning
   rate reached a per-token KL of only ~0.001 against a 0.05 target, with the adaptive β pinned at its
   floor — i.e. essentially no policy movement. This reframes the v1 run: its KL climb to 0.3+ (and
   the 8.2 spike) was driven by gradients on never-sampled EOS padding tokens, not by learning. With
   padding correctly masked out, the learning rate was raised to 5e-6.

4. Checkpoint storage. 39.7 GB of checkpoints carry redundant AdamW state; stripping it
   (`python -m src.sft.checkpoints strip checkpoints/ --apply`) recovers 29.1 GB. Left as a
   user decision — the script defaults to a dry run and keeps `.bak` copies.

### 6.6 Qualitative confirmation

Same prompt, both v2 checkpoints, deployment sampling:

> "A fault of area 400 km², slip 1.5 m, shear modulus 30 GPa. Calculate the seismic moment M0."

SFT v2 — opens `<think>`, states the correct formula, then loops forever and never closes the tag
or produces an answer:
> `<think>We need to compute M0 = μ  A  D … M0 = 3e10 * 5e8 * 5e8 = 3e10 * 5e8 * 5e8 = 3e10 * 5e8 * 5e8 = …`

GRPO v2 — closes `<think>` and emits a distinct final-answer section:
> `<think>… μ = 30 GPa = 30 × 10^9 Pa, A = 200 km^2 …</think>The seismic moment M0 is calculated as M0 = μ  A  D …`

GRPO fixed the structure (the block terminates, an answer section exists) and did not fix the
content (it misreads 400 km² as 200 km², substitutes area for slip, and the arithmetic is wrong).
That is precisely what the held-out numbers say: `<think>` rate +47 pp and significant, numerical
accuracy +1.7 pp and indistinguishable from noise.

---

## 7. SFT data-scaling law (2026-07-30)

> SUPERSEDED BY §10.2 — this curve varied corpus size and mixture together.

Three models trained identically (8 epochs, lr 2e-4, early stopping) on nested
subsets of the same 9.4k corpus, all scored on the same 281-pair
hash-pinned held-out test set:

| train pairs | test loss | perplexity |
|---|---|---|
| 2,100 | 2.2590 | 9.57 |
| 4,216 | 2.1869 | 8.91 |
| 8,507 | 2.1312 | 8.42 |

Log-linear fit, residuals within ±0.006 nats:

    loss = 2.9549 - 0.0633 * log2(N)        # 0.0633 nats per doubling

Extrapolation (upper bound — the trend must flatten eventually):

| pairs | loss | ppl |
|---|---|---|
| 13,000 (planned +4.5k fundamentals) | 2.090 | 8.08 |
| 20,000 | 2.050 | 7.77 |
| 40,000 | 1.987 | 7.29 |
| ~104,000 | 1.900 | 6.69 |

Implication. The slope is shallow. The planned fundamentals expansion buys
about 0.039 nats (~4% perplexity) — worth the spend, but not a change in
kind. Reaching loss 1.90 would take roughly 12x the current corpus.

Why the base model is the real ceiling. The pretrained checkpoint stopped
at iter 8,000 with val loss 2.6645, still descending (2.7967 -> 2.6645 over the
final doubling of compute, ~0.13 nats — twice the SFT-data slope). But it had
already consumed ~3.8 epochs of the 823M-token corpus, at the edge of the
~4-epoch useful-repeat budget, and sits at ~7.3 tokens/parameter versus the
~20:1 compute-optimal ratio. So the base is **data-limited, not
compute-limited**: more pretraining epochs on the same crawl will not help much;
more pretraining tokens would.

Practical reading. Paper-derived SFT data is now exhausted (17,289 raw ->
9,421 usable after cleaning, rewriting and deduplication, from only 5,860 source
chunks). The synthesized fundamentals path is the one that is not bounded by the
crawl, but at 0.0633 nats/doubling it would need tens of thousands of pairs to
matter, and it cannot repair coherence failures that originate in an
undertrained 113M base.

---

## 8. Historical-earthquake data, and a leak in my own split design (2026-07-30)

A third generator (`tools/gen_fundamentals/generate_earthquakes.py`, 109 events)
added 4,409 pairs; the expanded 63-topic curriculum added 4,034. Merged corpus:
15,659 pairs (4,667 paper + 2,549 rewritten + 4,034 fundamentals + 4,409
earthquakes), 14,068 train.

### 8.1 The leak

The first evaluation showed earthquake perplexity dropping 10.41 -> 4.75, which
was too good. Cause: `chunk_id` was computed as `md5(topic::variant)`, so every
variant became its own chunk and the splitter scattered variants of the same
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
| paper (356 pairs, 48 blocks) | 2.1671 (ppl 8.73) | 2.1566 (8.64) | −0.011 |
| earthquakes (203 pairs, 6 unseen events) | 2.1782 (8.83) | 1.7022 (5.49) | −0.476 |
| fundamentals (149 pairs, 4 unseen topics) | 2.3725 (10.72) | 2.3872 (10.88) | +0.015 |

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

Scored on the original `common_test.jsonl`, the 14.1k model looked worse
(2.1390 vs 2.1312). That file is 100% paper-derived, built before the new
sources existed, so it cannot see what the new data bought and it penalises the
distribution shift. Always score per source when the training mixture changes.

---

## 9. Sequence packing corrupts small-effect evaluation (2026-07-30)

Two nested paper-only test sets gave opposite signs for the same two models:

| test file | 8.5k | 14.1k | verdict |
|---|---|---|---|
| `common_test.jsonl` (281 pairs) | 2.1312 | 2.1390 | 14.1k worse |
| `test_paper.jsonl` (356 pairs, superset) | 2.1671 | 2.1566 | 14.1k better |

Both are cleanly held out from both models, same type mix, same mean length,
and the smaller file is a strict subset of the larger. The contradiction is an
artefact of packed evaluation:

1. Pairs are concatenated into 4096-token blocks under a plain causal mask, so
   each pair is conditioned on whatever unrelated pairs precede it in its
   block. Change the file, change the packing, change the number.
2. The loss is token-weighted over the block, so long pairs dominate.

Scoring each pair in isolation removes both problems and permits a paired
test (same pair, both models, so per-pair difficulty cancels).
`src/sft/compare_checkpoints.py --unpacked` does this.

### Unpacked, paired results (14.1k − 8.5k)

| held-out subset | n | mean diff | 95% bootstrap CI | t | verdict |
|---|---|---|---|---|---|
| paper (`common_test`) | 281 | −0.0100 | [−0.0154, −0.0045] | −3.66 | significant |
| paper (`test_paper`) | 356 | −0.0092 | [−0.0137, −0.0045] | −3.83 | significant |
| earthquakes (6 unseen events) | 203 | −0.5334 | [−0.5626, −0.5009] | −34.1 | significant |
| fundamentals (4 unseen topics) | 149 | +0.0039 | [−0.0136, +0.0221] | +0.43 | not significant |

The two paper files now agree: the 14.1k corpus is consistently, if slightly,
better on paper-domain text. The earthquake gain is enormous and unambiguous
(ppl 9.73 → 5.70 on events never trained on). Fundamentals is flat.

### Practical rules this establishes

- Packed loss is fine for training-time monitoring, not for model comparison.
  It is optimistically biased (extra context lowers loss: 2.1312 packed vs
  2.2799 unpacked on the same file and model) and its noise exceeds a 0.01-nat
  effect.
- Always pair. Comparing two means of independently noisy per-pair losses
  wastes most of the statistical power; the paired difference has far smaller
  variance because item difficulty cancels.
- Earlier packed numbers in this report (§7, §8) are directionally right where
  the effect is large, and unreliable where it is ~0.01 nats. The §7 scaling
  slope is essentially unchanged when recomputed unpacked (−0.064 vs −0.063
  nats/doubling), because the packing bias is roughly constant across models on
  a fixed file — it is the cross-file and small-effect comparisons that break.

---

## 10. Two follow-up questions, answered empirically (2026-07-30)

### 10.1 Is the train/val gap a sign that 400 steps is too many, or the LR too large?

Neither, exactly: it is too little data, and the run budget is simply wasteful.

An 8-epoch run at three learning rates on the same 14.1k corpus:

| lr | best val | at step (epoch) | final train | final val |
|---|---|---|---|---|
| 5e-5 | 2.1391 | 160 (4) | 1.511 | 2.199 |
| 1e-4 | 2.1296 | 120 (3) | 1.158 | 2.328 |
| 2e-4 | 2.1441 | 80 (2) | 0.701 | 2.576 |

Read three ways:

- The LR is not "too large" in the divergence sense. All three converge
  smoothly to a clean minimum; none is unstable. The spread between the best
  values is 0.015 nats — smaller than the paper-domain effect of doubling the
  corpus.
- 1e-4 is mildly better than 2e-4 (−0.015 nats) and reaches its minimum
  later, at epoch 3 instead of 2, which also leaves a less brutal train/val
  gap. Worth adopting, but it is a refinement, not a fix.
- Lowering the LR does not remove the gap. Even at 5e-5 the final train
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
tautological: the 8.5k corpus contained no earthquake data at all, so its
poor earthquake score measured the absence of a corpus, not a scaling effect.

Rebuilt against `strat_8k` — same 46/25/28 mixture, half the size — the panel
now decomposes the panel-(a) scaling step by domain:

| held-out subset | 8.0k | 14.1k | paired diff | 95% CI | t |
|---|---|---|---|---|---|
| paper | 2.4105 | 2.3136 | −0.0969 | [−0.1033, −0.0908] | −30.1 |
| earthquakes | 1.8951 | 1.7413 | −0.1538 | [−0.1728, −0.1353] | −16.1 |
| fundamentals | 2.5941 | 2.4689 | −0.1252 | [−0.1371, −0.1133] | −19.8 |

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
| best (step 80) | 2.2700 | 9.68 |
| last (step 336) | 2.7798 | 16.12 |

paired diff +0.5098 nats, 95% CI [+0.4794, +0.5386], t = +33.5. The
held-out likelihood degradation is unambiguous.

### 11.2 Not real in behaviour

The same two checkpoints, generating on 226 held-out prompts at the deployment
decoding configuration:

| metric | best (step 80) | last (step 336) | significance |
|---|---|---|---|
| Numerical accuracy @1% | 7.2% | 10.2% | z = +0.97, n.s. |
| Numerical accuracy @5% | 10.2% | 13.3% | z = +0.85, n.s. |
| `<think>` rate (numerical) | 61.4% | 74.1% | z = +2.47, significant |
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

- "Epochs after ~200 are useless" is not quite right. They cost 0.51 nats
  of held-out likelihood and buy a significant improvement in `<think>`
  compliance. Useless for likelihood, mildly useful for behaviour.
- Selecting checkpoints on validation loss is the wrong criterion here.
  It picks step 80; behaviour prefers step 336. Checkpoint selection should use
  the reward-independent behavioural metrics from `src/rl/eval_rl.py`.
- One risk the behavioural metrics do not capture: at train loss 0.70 the
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
| `sft_ckpt_epoch4.pt` | 13.9% | 9.6% | 63.3% | 0.0% | 21.5% | 334 | PASS |
| `sft_ckpt_epoch8.pt` | 12.7% | 9.0% | 72.9% | 0.0% | 25.8% | 315 | PASS |

Selected: epoch 8 → `checkpoints/sft/selected.pt`. Validation loss would
have picked epoch 3, which has the worst `<think>` rate of the three — §11
reproduced on an independent run.

Stated plainly: epoch 4 has the highest raw num@5% (13.9%), and the tie-break
is what moved the pick to epoch 8. All three sit inside the 2 pp band, which at
n=166 is noise; the choice rests on the tie-break, not on an accuracy win.

### 12.2 GRPO V3 run

200/200 steps from `sft_v6/selected.pt`, ~27 s/step, no KL early-stop.

| metric | steps 1–50 | 51–100 | 101–150 | 151–200 |
|---|---|---|---|---|
| mean reward | −0.463 | −0.253 | −0.365 | −0.356 |
| KL | 0.037 | 0.090 | 0.064 | 0.042 |
| β (adaptive) | 0.006 | 0.018 | 0.192 | 0.192 |
| entropy | 2.37 | 2.58 | 2.47 | 2.42 |
| stop rate | 0.640 | 0.883 | 0.673 | 0.670 |

Two things worth recording:

- The adaptive KL controller demonstrably worked, unlike V1/V2. KL peaked
  at 0.142, crossed the 2×target deadband, drove β from its 0.005 floor to
  0.192, and KL was pulled back to 0.042. Between 0.025 and 0.10 β holds
  constant by design — that is the deadband, not a stuck controller.
- The TerminationVerifier worked. Stop rate averaged 0.64–0.88 against
  27.5% in V2.

### 12.3 The run peaks near step 100, and both metrics agree

> [!WARNING]
> **The numbers in §12.3 and §12.4 were measured with a broken scoring rule and
> are superseded by §13.** The accuracy metric's floor exceeded the model, and
> the echo threshold fired on genuine answers. Corrected values are in §13.4;
> the conclusions about step 100 vs 200 survive and in fact strengthen, but
> the prompt-echo result does not. Left in place unedited so the correction is
> auditable.

Reward peaks at step ~105 and then regresses. Evaluating both saved checkpoints
reward-independently on the hash-pinned held-out set (166 numerical + 60
conceptual, deployment decoding):

| metric | SFT | GRPO @100 | GRPO @200 |
|---|---|---|---|
| numerical acc @1% | 9.0% | 9.0% | 10.8% |
| numerical acc @5% | 12.7% | 13.3% | 13.3% |
| `<think>` rate | 72.9% | 98.2% (z=+6.55) | 91.6% (z=+4.45) |
| contamination | 0.0% | 0.4% | 0.0% |
| prompt-echo | 21.2% | 11.1% (z=−2.94) | 16.4% (n.s.) |

Applying the §12.1 rule unchanged: num@5% ties exactly at 13.3%, so the
tie-break on `<think>` rate selects step 100 →
`checkpoints/grpo/selected.pt`. Step 100 also cuts prompt-echo
significantly where step 200 does not. Panel (d) of the training figure shows
the same peak independently, from the training log rather than the eval.

### 12.4 What GRPO did and did not buy

It did not improve numerical accuracy. 12.7% → 13.3% at 5% tolerance is
Δ=+0.006, z=+0.16 — indistinguishable from noise, as is the @1% change. Any
claim that this RL stage made the model better at arithmetic is unsupported by
these numbers.

It did improve output discipline, substantially and significantly.
`<think>` compliance 72.9% → 98.2% (z=+6.55) and prompt-echo 21.2% → 11.1%
(z=−2.94). Completion length fell 308 → 251 tokens with no entropy collapse.

The composite training reward improved −0.509 → −0.245, but that is the
objective being optimised and is reported only as a reference row; it is not
evidence of capability gain.

### 12.5 Figure-pipeline defects fixed while regenerating

- `plot_grpo_results.py` read the previous run directory (`grpo_v2`) — it would have plotted the
  old run. Now defaults to v3 and takes `--log` / `--eval-json`.
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

> CURRENT for the metric and reward fixes; SUPERSEDED IN PART BY §14, which shows
> the headline was measured on prompts that had already chosen the checkpoint.

Two questions asked directly. Answering them empirically rather than by
argument found that the primary capability metric was invalid and that the
reward had one live hacking channel. Both are now fixed; several numbers in
§12 are superseded.

### 13.1 The numerical-accuracy metric could not demonstrate capability

`eval_rl` scored `min(rel_err)` over the first 8 numbers in the answer, so
every additional number was another lottery ticket. Measured on the 166
held-out numerical items (`src/analysis/metric_validity.py`):

| k numbers sprayed | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| acc@5% of a question-blind policy | 2.0% | 4.0% | 5.8% | 9.3% | 14.3% |

A policy that knows only the empirical answer distribution and nothing about
the question scores 14.3% — above the GRPO model's reported 13.3% and the
SFT model's 12.7%. The metric had no demonstrated dynamic range at the
operating point.

The ceiling was also wrong: reference answers scored only 68.1% against
their own `final_numerical_value`, so ~32% of items were unwinnable by
construction.

### 13.2 Fix: score the answer tail

Restricting scoring to the last 3 numbers improves both ends, because the
stated answer sits near the end and "first 8" often never reached it:

| rule | ceiling | floor | usable range |
|---|---|---|---|
| min over first 8 (old) | 68.1% | 14.3% | 53.8% |
| last 3 (`ANSWER_TAIL_NUMBERS`) | 86.1% | 5.9% | 80.2% |
| last 1 | 58.4% | 1.9% | 56.5% |

The old anti-spam cap (`reject if >8 numbers`) then became redundant and
harmful: tail scoring already bounds an attacker to 3 chances regardless of how
many numbers they emit (a 40-number spray succeeds 5.7% of the time either
way), while the cap fired on 40.4% of genuine reference answers, which
carry a median of 6 numbers and a p90 of 34. Removing it lifted mean `r_num` on
the ground truth from +0.274 to +0.723.

### 13.3 One live reward-hacking channel, and a reward that punished the truth

`src/analysis/reward_hacking_probe.py` scores eight hand-built degenerate
completions against the real `CompositeRewardEngine` on all 166 items. Six were
decisively rejected — number spray −0.563, keyword stuffing in `<think>` −0.918,
empty think block −0.918 — confirming the earlier hardening works. But:

| probe | vs genuine reference |
|---|---|
| verbose filler + correct answer | +0.139 (exploit) |

`EchoPenaltyVerifier` divided overlap by the candidate's n-gram count, so
appending filler inflated the denominator and walked the ratio back under
threshold without removing any copied text. The same mis-specification
penalised genuine reference answers by −0.428 on average for legitimately
restating the question.

Fixed by normalising against the prompt — the construct actually of interest
and immune to padding (a literal echo scores 1.000 with or without filler).
Thresholds recalibrated on 808 genuine answers (max 0.760): 0.85 penalises
0.00%, 0.70 penalises 0.62%. Echo penalty on the ground truth: −0.428 → −0.006.
Padding now ties the genuine answer rather than beating it — no gradient toward
it. Two regression tests were added to `verifiers.py` (adaptive 40-number spray,
padding dilution), bringing the adversarial suite to 8.

> [!NOTE]
> `checkpoints/grpo` was trained with the pre-fix echo verifier. The fix
> changes future runs; it does not invalidate v3, whose completion length fell
> (308 → 251) rather than rising, indicating the padding channel was not being
> exploited in practice.

### 13.4 Corrected results (supersedes §12.3 / §12.4)

Re-measured end to end with the corrected scoring:

| metric | SFT | GRPO @100 | GRPO @200 |
|---|---|---|---|
| numerical acc @5% | 9.6% | 10.2% | 9.6% |
| numerical acc @1% | 7.8% | 6.6% | 8.4% |
| `<think>` rate | 72.9% | 98.2% (z=+6.55) | 91.6% (z=+4.45) |
| contamination | 0.0% | 0.4% | 0.0% |
| prompt-echo | 0.0% | 0.0% | 0.0% |
| mean completion tokens | 308 | 251 | 274 |

Three claims change:

- Accuracy is lower than reported (13.3% → 10.2%): the old figure was
  inflated by the lottery-ticket effect.
- The model clears the floor, but only just — 10.2% against a 5.9% floor
  and an 86.1% ceiling, i.e. 5.5% of the way through the usable range. It is
  a real but weak signal; small deltas here remain untrustworthy.
- The prompt-echo win is gone. The previously significant 21.2% → 11.1%
  (z=−2.94) was an artifact of the miscalibrated threshold, which also fired on
  11% of genuine answers. Under a correct detector no model echoes.

Checkpoint selection is unaffected and in fact better justified: step 100 now
wins on the primary metric outright (10.2% vs 9.6%) rather than only on the
tie-break, and still leads on `<think>` rate.

Net conclusion. After correcting the measurement, GRPO's only surviving
significant effect is `<think>` format compliance — the metric that overlaps
most with its own reward.

### 13.5 "Reward-independent" was the wrong term

The evaluation shares constructs with the reward:

| eval metric | reward counterpart | independent? |
|---|---|---|
| numerical acc @1/5% | `NumericalVerifier`, same field, same tolerances | no |
| `<think>` rate | `FormatVerifier`, same tags | no |
| contamination | `CleanLanguageVerifier` | no |
| prompt-echo | `EchoPenaltyVerifier` | no |

Every headline metric is the same construct measured on held-out prompts — a
generalisation check, not construct independence. That is still far better than
v1's circular benchmark (which scored the composite training reward on
training-distribution prompts), but the distinction must be stated.

### 13.6 What remains open

- No metric covers conceptual factuality, which is 642 of 808 held-out
  prompts (~79%). The largest slice of real usage has nothing measuring whether
  the model is right. An LLM-as-judge pass is the obvious gap-filler.
- No construct-independent metric exists at all now that echo reads zero.
- The probe scores `r_term = 0` because hand-built completions lack
  `<|im_end|>`, so it understates the TerminationVerifier's counterbalancing
  effect. Re-running with EOS included would tighten §13.3.
- Model answers' actual number counts are unmeasured (only 3 completions are
  stored per run), so the floor comparison assumes an 8-number answer.

---

## 14. Capability tiering, statistical power, and the split contract (2026-08-08)

> CURRENT AS OF 2026-08-08.

§13 fixed the metric. This section answers the next question — *what is the
corrected 10.2% actually made of, and is it measured on prompts that were
allowed to produce it* — and adds the accounting the earlier sections got wrong.

### 14.1 The headline was measured on prompts that had already voted

`select_checkpoint` and `eval_rl` both defaulted to the **full 808-prompt
held-out pool**. The same prompts chose GRPO step 100 over step 200 and then
produced the reported delta. The pre-declared selection rule (§12.2) prevents
cherry-picking; it does nothing about this.

`src/sft/splits.py` partitions the pool by `md5("seis-final-test::" + chunk_id)`
into a dev half (selection, thresholds, decoder, metric calibration) and a
frozen test half (reported once). Chunk-keyed, so siblings never straddle
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
| dev (selection touched it) | 76 | +3.0% | [+0.0%, +6.2%] |
| test (frozen) | 90 | +0.8% | [−3.1%, +4.7%] |

Same models, same metric, same decoder; the dev half reads nearly 4× larger.
The test row is the reportable one. `<think>` compliance is unaffected by the
split (73.1% → 97.8% on test), which is consistent with an effect far larger
than the selection artifact.

Scope of this partition. It is retrospective. The 808-prompt pool had
already been used for checkpoint selection, aggregate evaluation, metric
development, threshold calibration, and the choice of which analyses to run, so
hash-pinning freezes the split from 2026-08-08 onward but cannot make the test
half historically untouched. The 3.0-vs-0.8 gap is therefore consistent with
selection bias rather than a clean measurement of it: some of it is ordinary
heterogeneity between two 80-odd-item halves. Call this a retrospective audit
split, not a prospective final test. A clean final claim needs a new test set
built from previously unused source identities, run once, after freezing code,
parser, decoder, thresholds, candidates, and selection rule.

### 14.2 Paired outcomes, pass@k, and what this benchmark can resolve

`eval_rl.py` now stores per-item outcomes, which is what makes pairing
possible at all, and adds `--num-samples/-k`, pass@k, McNemar on discordant
pairs, paired bootstrap CIs, and an explicit power line:

```
[power] at n=166, k=4 sample(s)/prompt, baseline 7.1%: MDE ~ 10.0% at 80% power.
        A null result below that threshold means UNDERPOWERED, not 'no effect'.
```

At 166 numerical prompts (90 on the frozen half) this benchmark reliably detects
only large changes. Every null result in this document is therefore *not
supported, never disproven*. Reaching a 5-point MDE needs ~470 paired prompts;
2 points needs ~2,900.

pass@k separates "can sometimes do it" from "does it reliably", and carries its
own higher chance floor: must-compute items land 13.6% of the time given
four attempts, against a 20.3% pass@4 floor — still below chance.

### 14.3 Where the 10.2% comes from

Splitting the numerical set by whether the answer must be derived rather than
retrieved from the prompt:

| tier | n | chance floor | SFT | GRPO |
|---|---|---|---|---|
| answer already present in the prompt | 34 | 7.2% | 20.6% | 29.4% |
| must be computed | 132 | 5.5% | 3.6% | 3.6% |

(34 × 29.4% + 132 × 3.6%) / 166 = 8.9% — essentially the whole aggregate comes
from the retrievable fifth. Every derivation-depth tier from 6 steps to 20+ sits
at or below its own floor. The single aggregate number was hiding the only thing
worth knowing.

Held-out numerical questions require a median of 11 derivation steps.

### 14.4 Phase-0 gate: calculator tool use, tested and abandoned

Before building tool-use plumbing, a direct test: extract the arithmetic the
model writes inside its own `<think>` and execute it with a restricted AST
evaluator (`src/rl/safe_eval.py` — no `eval()`, 11 hostile inputs blocked in its
self-test).

| condition | acc@5% |
|---|---|
| model's stated answer | 3.6% |
| model's own arithmetic, executed | 1.3% |
| blind floor | 1.8% |
| instrument ceiling (ground-truth expressions executed) | 50.8% |

Executing the model's expressions is worse than its stated answer and below
the floor, against an instrument that recovers half of the ground truth. The
expressions are noise, so a calculator has nothing correct to execute — the
bottleneck is formula recall, not arithmetic. Tool use was stopped here, at a
cost of ~40 GPU-minutes.

### 14.5 Prompting baseline

Explicitly instructing the SFT model to use `<think>` tags moves compliance
72.3% → 73.5%; RL reached 98.2%. Under the tested prompt template and pinned
decoder, prompting does not reproduce the format gain.

This supports "prompting was insufficient", not "RL was necessary". The
missing rungs of the ladder are format-targeted SFT and — most importantly —
rejection-sampling fine-tuning (sample k, keep verifier-passing completions,
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
| numerical | 77 / 200 (38.5%) | the physics, then the arithmetic |
| conceptual | 123 / 200 (61.5%) | cover the domain concepts |

Correction. An earlier revision of this section reported "reward-opportunity
rates" of 74/200 and 168/200. Those are not recoverable from these artifacts:
`checkpoints/runs/grpo.csv` logs the batch mean of each reward component, not
its within-group standard deviation, and only the latter determines whether a
component produced gradient. The table above is step eligibility, an upper
bound on the opportunity rate. Logging per-component within-group std is one
number per step and should be added before the next run.

The run also visited 400 of 5,217 prompts (7.7%) — sampling-limited, not
step-limited.

### 14.7 Corrected compute accounting

Earlier drafts described the base model as "7.3 tokens/parameter, far below
compute-optimal". That conflated three different quantities:

| quantity | value |
|---|---|
| total parameters | 125.9M |
| non-embedding parameters | 113.3M |
| unique corpus tokens | 822.7M |
| epochs | 3.8 |
| token exposures processed | 3.13B |
| exposures per non-embedding parameter | 27.6 |

Against the pretraining repo's own measured IsoFLOP frontier — which puts the
optimal ratio near 46 at its largest budget, not Chinchilla's ~20 — the base
is modestly under-trained, within a factor of ~1.7 of its frontier. The null
numerical result is therefore not well explained as data starvation.

Post-training compute, in units that transfer to another model or batch size:

| | value |
|---|---|
| SFT | 14,068 examples × 8 epochs = 112,544 example-exposures, ~10 min on 2× A30 |
| RL | 200 steps · 3,200 completions · 954k generated tokens · 3.0 GPU-h |

"200 steps" does not transfer; "3,200 completions and 954k generated tokens"
does.

### 14.8 What §14 supersedes

- The pooled +3.0% accuracy delta → +0.8% [−3.1%, +4.7%] on the frozen
  test half (§14.1).
- "7.3 tokens/parameter, data-starved" → **27.6 exposures per non-embedding
  parameter** vs a measured frontier of ~46 (§14.7).
- The aggregate numerical accuracy as a capability statement → the tiered
  breakdown, in which must-compute accuracy never clears its floor (§14.3).
- Any use of "GRPO" as an algorithm claim → group-relative REINFORCE with KL
  regularization (see the naming note at the top).
