# Seismology Fundamentals SFT Generator

Standalone, portable data generator. **Python 3.8+ standard library only** — no
`pip install`, no imports from the nanoGPT-Seis repo. Copy this folder to any
machine with API access, run it, copy the resulting `.jsonl` back.

```
tools/gen_fundamentals/
├── curriculum.py               # syllabus of fundamentals (data, not code)
├── generate_fundamentals.py    # generator: settled physics
├── earthquakes.py              # catalogue of historical events (data, not code)
├── generate_earthquakes.py     # generator: the empirical record
├── rewrite_standalone.py       # recovers contaminated paper pairs
└── README.md
```

Three generators, three kinds of knowledge:

| script | teaches | grounded in |
|---|---|---|
| `generate_fundamentals.py` | principles — how earthquakes work | `curriculum.py`, 63 topics |
| `generate_earthquakes.py` | cases — what specific earthquakes did | `earthquakes.py`, 109 events |
| `rewrite_standalone.py` | recovers the existing paper corpus | `sft_dataset_raw.jsonl` |

---

## Why this dataset exists

The existing SFT corpus was auto-generated from random paper excerpts, so it
teaches *specific studies* rather than the subject. Coverage audit of the 4,434
cleaned pairs:

| Topic | Examples | % of corpus |
|---|---|---|
| Liquefaction | 218 | 4.9% |
| P-wave **and** S-wave mentioned together | 71 | 1.6% |
| …with "compressional"/"longitudinal" | 15 | 0.3% |
| …with "transverse"/particle motion | 7 | 0.2% |
| Deep-focus earthquakes | 10 | 0.2% |
| …with the actual mechanism | **3** | **0.1%** |

Measured model behaviour tracked that distribution exactly — a usable answer on
liquefaction, incoherent answers on P/S waves and deep-focus earthquakes (it
placed them at 10–30 km instead of 300–700 km). The model was never taught the
fundamentals. This generator supplies the missing textbook layer.

---

## Usage

### 1. Transfer

Copy the whole `gen_fundamentals/` folder. Nothing else is needed.

### 2. Preview — no API calls, no key

```bash
python3 generate_fundamentals.py --dry-run
```

Prints the curriculum breakdown, the request plan, and one fully rendered
prompt so you can inspect exactly what will be sent.

### 3. Configure the endpoint

Works with any OpenAI-compatible `/chat/completions` API.

```bash
export TEACHER_API_KEY="..."
export TEACHER_API_BASE="https://<your-provider>/v1"
export TEACHER_MODEL="<model-name>"

python3 generate_fundamentals.py --variants 2 --workers 4

# or pass them per invocation
python3 generate_fundamentals.py --api-base https://<your-provider>/v1 --model <model-name>

# Local vLLM / Ollama / llama.cpp — no key needed for localhost
python3 generate_fundamentals.py --api-base http://localhost:8000/v1 --model my-model
```

There is no built-in provider default: `--api-base` and `--model` must come from
the flags or from `TEACHER_API_BASE` / `TEACHER_MODEL`.

### 4. Scale — add topics, not variants

One pass over the curriculum is **63 topics → ~378 pairs**. `--variants N` runs
N batches per topic, each with a different framing angle (20 defined, cycled).

| `--variants` | API calls | ≈ pairs |
|---|---|---|
| 1 | 63 | 380 |
| 5 | 315 | 1,900 |
| 12 | 756 | 4,500 |
| 20 | 1,260 | 7,500 |

**Start with `--variants 1` and read the output** before spending on a large
run. Inspect a few responses by hand; the curriculum is easy to tune.

> [!IMPORTANT]
> **Variants have sharply diminishing returns — prefer more topics.** Measured
> on a real 12-variant run (fraction of word-types each variant contributed
> that no earlier variant of the same topic had):
>
> | variant | 1 | 3 | 5 | 7 | 9 | 11 |
> |---|---|---|---|---|---|---|
> | new content | 55% | 33% | 30% | 26% | 21% | **12%** |
>
> By variant 11 you are paying full price for ~12% new material — the rest is
> paraphrase. Past `--variants 20` the framing angles also start to cycle, so
> variant 20 reuses variant 0's angle on the same topic with the same facts,
> which is close to a duplicate.
>
> Adding a topic restarts that curve at 55%. If you want more data, edit
> `curriculum.py` first and re-run at a **moderate** variant count.

Because resume is keyed on `(topic, variant)`, adding topics and re-running the
same command generates **only the new work** and retries earlier failures:

```bash
# after expanding curriculum.py from 33 to 63 topics, with 386 chunks already done:
python3 generate_fundamentals.py --variants 12 --workers 8 --out fundamentals.jsonl
#   -> 756 wanted, 386 skipped, 370 new calls (360 fresh topics + 10 retries)
```

### 5. Interruptions are safe

Output is appended and every completed `(topic, variant)` is keyed by
`chunk_id`. Re-run the identical command and it skips finished work:

```
[resume] 164 chunks already in fundamentals.jsonl; skipping those
```

Use `--overwrite` to start from scratch.

### 6. Validate anywhere (no key)

```bash
python3 generate_fundamentals.py --validate-only --out fundamentals.jsonl
```

---

## Output format

JSONL, one line per topic-variant, matching the schema the repo's cleaner and
splitter already consume:

```json
{"chunk_id": "a1b2c3...", "source": "fundamentals",
 "topic": "Seismic body waves: P versus S", "section": "wave_physics",
 "variant": 0, "pair_count": 6,
 "pairs": [{"type": "conceptual", "instruction": "...", "response": "..."}]}
```

One topic = one `chunk_id`, so the repo's **chunk-level** train/val splitter
keeps a topic's questions together and cannot leak facts across the split.

---

## Built-in quality gates

Every pair is validated before it is written; failures are dropped and counted.
This is the same contamination logic the repo's `src.sft.build_corpus clean` applies,
enforced at generation time so you do not pay for samples that will be filtered
out later.

| Check | Rejects |
|---|---|
| Phantom source | "the text", "the study", "the authors", "et al.", "Table 2", "as described in" |
| Degenerate math | `W / W`, `M = M - M` — the model had learned these from bad training data |
| Think-tag hygiene | numerical answers need **exactly one** `<think>…</think>`; others need none |
| Final answer | numerical answers must have ≥20 chars containing a digit after `</think>` |
| Type/length | valid type, instruction ≥15 chars, response ≥80 chars |

Verified against a mock server: contaminated and degenerate samples are
correctly rejected, and clean output passes the repo's real cleaner at **100%
retention** (the paper-derived corpus retains 27%).

The generation prompt also pins the two things the model most often got wrong:
the M0 vs Mw unit convention (`-6.07` for N·m vs `-10.7` for dyne·cm), and the
requirement that every symbol in a formula be defined and denote one thing.

---

## Merging back into the training pipeline

On the training machine:

```bash
cp fundamentals.jsonl <repo>/data/

cd <repo>
# 1. clean the paper-derived corpus as usual
python -m src.sft.build_corpus clean \
    --input data/sources/paper_raw.jsonl --output data/sources/paper.jsonl

# 2. append the fundamentals (already clean — do NOT re-run the cleaner on it,
#    its domain filter is tuned for paper text and may drop valid theory Q&A)
cat data/sources/fundamentals.jsonl >> data/sources/paper.jsonl

# 3. rebuild the document-disjoint splits
python -m src.sft.build_corpus split \
    --input data/sources/paper.jsonl --output-dir data/processed_v2

# 4. retrain
torchrun --standalone --nproc_per_node=2 -m src.sft.train_sft \
    --train-path data/processed_v2/sft_train.jsonl \
    --val-path data/processed_v2/sft_val.jsonl \
    --out-dir checkpoints/sft_v4 --epochs 8 --learning-rate 2e-4
```

Then re-measure with `python -m src.rl.eval_rl`, and check the specific gaps by
hand with `python -m src.sft.chat --ckpt-path checkpoints/sft_v4/sft_ckpt_best.pt --test`
— tests 2 and 5 (P/S waves, deep-focus earthquakes) are the ones this dataset
targets.

---

## Editing the curriculum

`curriculum.py` is a plain list of dicts. To add a topic:

```python
_t("Your topic name", "section_name",
   ["fact the answers must cover",
    "another fact — be specific, these are injected into the prompt"],
   DEFAULT_MIX,                        # or CONCEPT_ONLY / QUANT_HEAVY
   notes="Optional emphasis, e.g. a known model failure to correct."),
```

The `must_cover` bullets are pasted verbatim into the generation prompt, so
write them as assertions of what is true. `notes` is the lever for correcting a
specific observed failure — see the deep-focus entry for an example.

`python3 curriculum.py` prints a breakdown by section and question type.

---

## Cost estimate

Roughly 900 prompt tokens and 2,500–3,500 completion tokens per call. At
`--variants 12` (396 calls) that is on the order of **1.5M tokens** — a few
dollars at commodity API pricing, more on frontier models. Run `--variants 1`
first and multiply by your observed cost.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--out` | `fundamentals.jsonl` | output path |
| `--variants` | `1` | batches per topic |
| `--workers` | `4` | parallel calls; lower it on 429s |
| `--api-base` | `$TEACHER_API_BASE` | required; most providers need the `/v1` suffix |
| `--model` | `$TEACHER_MODEL` | required |
| `--temperature` | `0.8` | higher = more varied questions |
| `--max-tokens` | `4000` | raise if responses are truncated |
| `--timeout` / `--max-retries` | `180` / `5` | retries use exponential backoff on 429/5xx |
| `--limit-topics` | `0` (all) | smoke test on the first N topics |
| `--dry-run` | | print plan + example prompt, call nothing |
| `--validate-only` | | re-check an existing file |
| `--overwrite` | | truncate instead of resuming |


---

## Historical earthquakes (`generate_earthquakes.py`)

Fundamentals teach principles; this teaches the empirical record — what
specific earthquakes did, why, what they destroyed, and what the field learned.

```bash
python3 generate_earthquakes.py --dry-run                    # no key needed
python3 generate_earthquakes.py --variants 1 --workers 4     # pilot, 43 calls
python3 generate_earthquakes.py --variants 12 --workers 8    # ~3,000 pairs
```

**109 events → 626 pairs per pass**, Mw 3.4–9.5, depths 3–680 km, 1755–2024 (98 of them
since 1960, the well-instrumented era), spanning great
subduction earthquakes, deep-focus events, continental strike-slip, blind
thrusts, vulnerability-dominated disasters, and induced seismicity. Question
types: `factual`, `mechanism`, `impact`, `significance`, `numerical`.

### The hallucination guard

This generator is riskier than the fundamentals one. Settled physics is safe to
paraphrase; specific events invite confabulated magnitudes, dates and death
tolls, and a wrong number reads exactly as fluent as a right one.

Three defences:

1. **Verified seed facts.** Every entry in `earthquakes.py` carries checked
   specifics, injected as the only permitted source. The prompt forbids
   inventing casualty figures and requires disputed numbers to stay as ranges.
2. **Fact-contradiction validation.** Generated text is parsed for magnitudes
   and years; a value that contradicts the seed *and* sits next to the event's
   name is rejected (Mw tolerance 0.25, year proximity window 35 chars).
   Legitimate comparisons to other earthquakes are preserved.
3. **Implausibility rejection.** Any magnitude outside 2.5–9.9 attached to a
   named event is fabricated by definition.

Verified against a mock server that deliberately poisons every third batch with
a wrong magnitude: **all poisoned batches rejected, zero fabrications reaching
the output.** Temperature defaults to 0.6 rather than 0.8 — facts, not prose
variety.

> [!NOTE]
> An earlier version of the validator captured only one digit before the
> decimal, so a fabricated "Mw 10.5" parsed as "0.5", was dismissed as
> implausible, and passed. Two fabrications reached the output before this was
> caught. If you extend the validator, test it against poisoned input rather
> than trusting it.

### Merging all three sources

```bash
python -m src.sft.build_corpus merge \
    --cleaned data/sft_paper_clean.jsonl \
    --rewritten data/sources/paper_rewritten.jsonl \
    --fundamentals data/sources/fundamentals.jsonl \
    --extra data/sources/earthquakes.jsonl \
    --output data/sft_dataset_v5.jsonl

python -m src.sft.build_corpus split --input data/sft_dataset_v5.jsonl \
    --output-dir data/split_v5
```

`--extra` is repeatable. One event = one `chunk_id`, so all questions about a
given earthquake stay on the same side of the train/test split — otherwise the
model could be asked about Tohoku at test time having trained on twelve other
Tohoku questions.

### Expected yield

| `--variants` | API calls | ≈ pairs |
|---|---|---|
| 1 | 109 | 630 |
| 4 | 436 | 2,500 |
| 8 | 872 | 5,000 |
| 12 | 1,308 | 7,500 |

16 framing angles are defined, so novelty holds to roughly `--variants 16`
before they cycle. Adding events to `earthquakes.py` beats raising variants —
same reasoning as the curriculum, and the catalogue is easy to extend.

Validator regression: **109 events x 3 probes (truthful / fabricated magnitude /
fabricated year) = 0 false positives, 0 false negatives.** Re-run it after
editing the catalogue or the checks:

```bash
python3 - <<'PY'
import sys; sys.path.insert(0,'.')
from generate_earthquakes import check_pair, name_anchors, seed_mw
from earthquakes import EARTHQUAKES
# ... see the regression snippet in the repo history
PY
```
