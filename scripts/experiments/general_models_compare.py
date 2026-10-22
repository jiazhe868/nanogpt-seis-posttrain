"""The deployed 113M model against general Qwen models of every size, scored the same way (README §1.4).

Inputs (written by the evaluation scripts into work/experiments):
  blind_<name>.json      one Python program per question, within 1% (general_models_strict.py)
  baseline_<name>.json   free-prose answers to the conceptual prompts (llm_baseline_eval.py), for key concepts
Our model: docs/eval_numerical_real6_final.json (6 seeds x 4) and docs/eval_ceilings.json (conceptual answers).

"Practised question types": held-out questions whose nearest problem in the deployed model's training data
(Phase II + Phase IV SFT, the v14 RL pool; rows written from the test papers' excerpts removed) has a
word-Jaccard similarity of at least 0.4 (the measure of scripts/experiments/nn_sim.py). Key concepts at equal
length: each general-model answer is cut to the mean word count of ours on the same prompt.

    python scripts/experiments/general_models_compare.py      # writes docs/eval_general_models.json
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plots.grpo import _concept_items, _headline_items, _in_prompt_fn  # noqa: E402

W = ROOT / "work" / "experiments"
DEPLOYED = "notation2 + GRPO v14 @75 (corpus, dev-selected)"
GENERAL = {"Qwen3.5-0.8B": (0.8e9, "qwen0.8B", "baseline_qwen0.8B.json"), "Qwen3.5-2B": (2e9, "qwen2B", "baseline_qwen2B.json"),
           "Qwen3.5-4B": (4e9, "qwen4B", "baseline_qwen4B.json"), "Qwen3.5-9B": (9e9, "qwen9B", "baseline_qwen9b_long.json"),
           "Qwen3.8-27B": (27e9, "qwen27b", None)}
POOL = (("data/splits/train.jsonl", "instruction"), ("data/splits/train_pot_notation2.jsonl", "instruction"),
        ("data/rl/train_v14_corpus.jsonl", "prompt"))
THRESHOLD = 0.4


def words(q):
    return frozenset(re.findall(r"[a-zA-Z]{3,}", q.lower()))


def nearest(questions, pool):
    """Max word-Jaccard of each question to the pool (inverted index over words in < 5% of the pool)."""
    P = [words(p) for p in pool]
    df = collections.Counter(w for p in P for w in p)
    inv = collections.defaultdict(list)
    for j, p in enumerate(P):
        for w in p:
            if df[w] < 0.05 * len(P):
                inv[w].append(j)
    out = []
    for q in questions:
        w = words(q)
        cand = collections.Counter(j for x in w for j in inv.get(x, ()))
        out.append(max([len(w & P[j]) / len(w | P[j]) for j, _ in cand.most_common(300)] or [0.0]))
    return out


def boot(values, rng, n=4000):
    v = np.asarray(values, float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [100 * v.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)]


def main():
    rng = np.random.default_rng(0)
    in_prompt, test = _in_prompt_fn()
    runs = json.load(open(ROOT / "docs/eval_numerical_real6_final.json", encoding="utf-8"))["results"][DEPLOYED]
    ours1, ours8 = _headline_items(runs), _headline_items(runs, 0.05)
    ids = sorted(ours1)
    tc = {it["chunk_id"] for it in test.values()}
    pool = [json.loads(l)[k] for p, k in POOL for l in open(ROOT / p, encoding="utf-8")
            if json.loads(l).get("chunk_id") not in tc]
    sim = dict(zip(ids, nearest([test[i]["prompt"] for i in ids], pool)))
    strict = {}
    for name, (_, tag, _) in GENERAL.items():
        per = collections.defaultdict(list)
        for out in json.load(open(W / f"blind_{tag}.json"))["seeds"].values():
            for o in out:
                per[o["id"]].append((o["verified"], o["within5"], o["answer2"] is None))
        strict[name] = {i: [float(np.mean([x[k] for x in v])) for k in range(3)] for i, v in per.items()}
    subsets = {"all": lambda i: True,
               "practised types": lambda i: sim[i] >= THRESHOLD,
               "other question types": lambda i: sim[i] < THRESHOLD,
               "must compute": lambda i: not in_prompt(i)}
    table = {}
    for sub, keep in subsets.items():
        q = [i for i in ids if keep(i)]
        row = {"n": len(q), "ours": {"within1_one_sample": boot([ours1[i]["acc1"] for i in q], rng),
                                     "within5_one_sample": boot([ours1[i]["acc5"] for i in q], rng),
                                     "within1_vote8": boot([ours8[i]["acc1"] for i in q], rng),
                                     "within5_vote8": boot([ours8[i]["acc5"] for i in q], rng)}}
        for name in GENERAL:
            row[name] = {"within1": boot([strict[name][i][0] for i in q], rng),
                         "within5": boot([strict[name][i][1] for i in q], rng),
                         "no_answer": 100 * float(np.mean([strict[name][i][2] for i in q])),
                         "ours_minus_within1": boot([ours1[i]["acc1"] - strict[name][i][0] for i in q], rng)}
        table[sub] = row
    # key concepts at equal length
    con_rows = json.load(open(ROOT / "docs/eval_ceilings.json", encoding="utf-8"))["concept"]["deployed_routed"]
    ours_con = _concept_items(con_rows)
    ours_words = collections.defaultdict(list)
    for r in con_rows:
        ours_words[r["id"]].append(r["words"])

    def cov(t, k):
        t = t.lower()
        return sum(c.lower() in t for c in k) / len(k)
    concepts = {"ours": {"same_length": boot([v["coverage"] for v in ours_con.values()], rng),
                         "words": float(np.mean([r["words"] for r in con_rows]))}}
    for name, (_, _, f) in GENERAL.items():
        if not f or not (W / f).exists():
            continue
        R = json.load(open(W / f))["results"]["concept"]["rows"]
        m = {r["id"]: float(np.mean([cov(" ".join(t.split()[:round(np.mean(ours_words[r["id"]]))]), test[r["id"]]["key_concepts"])
                                     for t in r["texts"]])) for r in R}
        concepts[name] = {"same_length": boot(list(m.values()), rng), "whole": boot([r["coverage"] for r in R], rng),
                          "words": float(np.mean([r["words"] for r in R])),
                          "ours_minus_same_length": boot([ours_con[i]["coverage"] - m[i] for i in m], rng)}
    prose = {}
    for name, (_, _, f) in GENERAL.items():
        if f and (W / f).exists():
            R = json.load(open(W / f))["results"]
            prose[name] = {"within5_heldout": 100 * float(np.mean([r["acc5"] for r in R["numerical"]["rows"]])),
                           "tokens": float(np.mean([r["tokens"] for r in R["numerical"]["rows"]]))}
    json.dump({"_provenance": {
        "strict": "one Python program per question (tools/gen_pot_problems.py SOLVE_PROMPT + RULES), T 0.3, top-p 0.95, "
                  "1,500 new tokens, no thinking, 4 seeds; FP8 weights; correct = executed answer within 1% of the label",
        "ours": "the deployed model, executed paper-notation equations, T 0.3, top-k 40, 6 seeds x 4 samples",
        "practised": f"nearest training problem with word-Jaccard >= {THRESHOLD}; training pool = {[p for p, _ in POOL]}, "
                     "rows from the 215 test excerpts removed",
        "concepts": "641 conceptual held-out prompts; general-model answers cut to the mean word count of ours on the same prompt",
        "prose": "for reference only: free prose up to 4,096 tokens, any of the last 3 numbers within 5% (llm_baseline_eval.py)"},
        "params": {name: p for name, (p, _, _) in GENERAL.items()} | {"ours": 113.3e6},
        "table": table, "concepts": concepts, "prose_reference": prose,
        "per_question": {"similarity": sim, "ours_within1": {i: ours1[i]["acc1"] for i in ids},
                         "strict_within1": {n: {i: v[0] for i, v in s.items()} for n, s in strict.items()}}},
        open(ROOT / "docs/eval_general_models.json", "w"), indent=1)
    for sub, row in table.items():
        print(f"{sub:22s} n={row['n']:3d} ours {row['ours']['within1_one_sample'][0]:5.1f} | " +
              " | ".join(f"{n.split('-')[1]} {row[n]['within1'][0]:5.1f}" for n in GENERAL))
    for n, c in concepts.items():
        print(f"concepts {n:13s} same length {c['same_length'][0]:5.1f}")


if __name__ == "__main__":
    main()
