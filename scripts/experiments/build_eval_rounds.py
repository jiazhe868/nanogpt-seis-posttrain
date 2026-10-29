"""The 153 held-out questions after every step of README §5.20–5.22, gathered into docs/eval_rounds.json.

Each model was scored with real6_eval.py (6 seeds x 4 samples, executed paper notation, T 0.3) and most with set_eval.py
on the 300 in-distribution problems (4 seeds x 4). This keeps per-question means only, so the figures and the README
tables compute their intervals from one small file:
    real6: acc5, acc1, vote8 (majority of 8 executed answers within 5%, src.analysis.tier_vote.pooled), think
    indist: acc5
Inputs are the work/experiments/*real6*.json / *indist*.json files written by the round chains
(scripts/experiments/{notation3_chain,round4_chain,pretrain_tokens_control,size_coverage_chain,round5_chain,
round5_rl_chain,round6_chain,round6_rl_more}.sh) and docs/eval_numerical_real6_final.json for the earlier models.

    python scripts/experiments/build_eval_rounds.py [--sync]
(--sync also copies the round-6 fine-tune and the deployed model into the headline results files.)
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.tier_vote import pooled  # noqa: E402
from src.rl.verifiers import NumericalVerifier  # noqa: E402

W = ROOT / "work/experiments"
FINAL = ROOT / "docs/eval_numerical_real6_final.json"
INDIST = ROOT / "docs/eval_indist.json"
SWEEP = ROOT.parent / "nanogpt_seis/checkpoints/scaling/scaling_fit.json"
OURS_N, OURS_D = 100.7e6, 3.15e9          # the 113M base: non-embedding parameters, pretraining tokens processed

def _indist(f):
    """The in-distribution run of the same checkpoint: epoch-4 re-runs carry the _e4 suffix like their 153 runs."""
    p = W / f"ctrl_indist_{f}.json"
    return p if p.exists() else W / f"ctrl_indist_{f.replace('_e4', '')}.json"


# name: (153 results, in-distribution results, what it is)
M = {
    "notation2": ((FINAL, "notation2 (recovered data)"), (INDIST, "notation2"),
                  dict(step="Phase IV fine-tune (start)", base="original", data="notation2", recipe="continued")),
    "v14": ((FINAL, "notation2 + GRPO v14 @75 (corpus, dev-selected)"), (INDIST, "v14-75"),
            dict(step="previous deployed model (GRPO on the corpus)", base="original", data="notation2", recipe="continued")),
    "round3": ((W / "n3_real6.json", None), (W / "n3_indist.json", None),
               dict(step="+ window A", base="original", data="notation3", recipe="continued")),
    "round4": ((W / "n4_real6.json", None), (W / "n4_indist.json", None),
               dict(step="+ window B", base="original", data="notation4", recipe="continued")),
    "round4_rl": ((W / "r4_real6.json", None), (W / "r4_indist.json", None),
                  dict(step="+ GRPO v18, one pass over window B's held-back problems", base="original", data="notation4",
                       recipe="continued")),
}
for d, order, f in (("notation3", 0, "113M_3150M"), ("notation3", 1, "113M_3150M_d1_e4"),
                    ("notation4", 0, "113M_3150M_n4_e4"), ("notation4", 1, "113M_3150M_n4_d1_e4"),
                    ("notation5", 0, "113M_3150M_n5"), ("notation5", 1, "113M_3150M_n5_d1")):
    M[f"one_{d}_o{order}"] = ((W / f"ctrl_real6_{f}.json", None), (_indist(f), None),
                              dict(step="one-stage fine-tune", base="original", data=d, recipe="one stage", order=order))
for d, order, f in (("notation3", 0, "113M_mid1p6B"), ("notation3", 1, "113M_mid1p6B_d1"),
                    ("notation4", 0, "113M_mid1p6B_n4"), ("notation4", 1, "113M_mid1p6B_n4_d1_e4"),
                    ("notation5", 0, "113M_mid1p6B_n5"), ("notation5", 1, "113M_mid1p6B_n5_d1"),
                    ("notation6", 0, "113M_mid1p6B_n6"), ("notation6", 1, "113M_mid1p6B_n6_d1")):
    M[f"mid_{d}_o{order}"] = ((W / f"ctrl_real6_{f}.json", None), (_indist(f), None),
                              dict(step="one-stage fine-tune of the mid-trained base", base="mid-trained", data=d,
                                   recipe="one stage", order=order))
for name, f, start, seed in (("v19", "v19_plain", "mid_notation5_o0", 42), ("v19b", "v19b_plain", "mid_notation5_o0", 43),
                             ("v20", "v20_plain", "mid_notation6_o0", 42), ("v20b", "v20b_plain", "mid_notation6_o0", 43)):
    M[name] = ((W / f"{f}_real6.json", None), (W / f"{f}_indist.json", None),
               dict(step="+ GRPO on fresh problems, three passes", start=start, rl_seed=seed))
M["v20"][2]["deployed"] = True
SIZES = {"xs_3180M": "C6e16_xs", "s_3180M": "C1.8e17_s", "m_1240M": "C1.8e17_m", "l_678M": "C1.8e17_l",
         "xl_1190M": "C5.4e17_xl", "2xl_82M": "C6e16_2xl", "2xl_246M": "C1.8e17_2xl", "2xl_738M": "C5.4e17_2xl"}


def results(src):
    path, key = src
    if not path.exists():
        return None, None
    d = json.load(open(path, encoding="utf-8"))
    return (d["results"][key] if key else next(iter(d["results"].values()))), d.get("ckpt")


def per_question(runs):
    tail = NumericalVerifier.ANSWER_TAIL_NUMBERS
    acc5, acc1 = {}, {}
    for r in runs:
        for x in r["per_item"]:
            if x["type"] == "numerical":
                acc5.setdefault(x["id"], []).append(x["acc5"]); acc1.setdefault(x["id"], []).append(x["acc1"])
    _, vote = pooled(runs, 8, tail)
    return {"acc5": {i: st.mean(v) for i, v in acc5.items()}, "acc1": {i: st.mean(v) for i, v in acc1.items()},
            "vote8": {i: v for i, v in vote.items() if v == v},
            "think": st.mean(r["numerical_think_rate"] for r in runs)}


ROUND6 = "round 6: mid-trained base, one stage, notation6"          # keys of the round-6 models in the docs files
V20 = "round 6 + GRPO v20 @99 (fresh problems, deployed)"
CEILINGS = ROOT / "docs/eval_ceilings.json"


def routed(eq, prose, test):
    """Concept rows as served: equations for the questions the router sends there, prose for the rest
    (src/sft/question_router.py; the same assembly as deployed_routed in docs/eval_ceilings.json)."""
    from src.sft.question_router import is_numerical_question
    rows = lambda n: json.load(open(W / f"ceiling_concept_{n}.json"))["rows"]
    keep = lambda r: {k: r[k] for k in ("id", "coverage", "verifier", "words")}
    num = lambda r: is_numerical_question(test[r["id"]]["prompt"]) if r["id"] in test else False
    return [keep(r) for r in rows(eq) if num(r)] + [keep(r) for r in rows(prose) if not num(r)]


def sync_headline():
    """The round-6 fine-tune and the deployed GRPO v20 model into the files the README figures read: their 6 x 4
    runs (eval_numerical_real6_final.json), in-distribution runs (eval_indist.json), routed concept rows and
    64-sample coverage at T 0.8 (eval_ceilings.json)."""
    test = {it["id"]: it for it in map(json.loads, open(ROOT / "data/rl/test.jsonl", encoding="utf-8"))}
    for path, entries in ((FINAL, {ROUND6: "ctrl_real6_113M_mid1p6B_n6.json", V20: "v20_plain_real6.json"}),
                          (INDIST, {"round6": "ctrl_indist_113M_mid1p6B_n6.json", "v20-99": "v20_plain_indist.json"})):
        d = json.load(open(path, encoding="utf-8"))
        for key, f in entries.items():
            d["results"][key] = next(iter(json.load(open(W / f))["results"].values()))
        json.dump(d, open(path, "w", encoding="utf-8"))
    c = json.load(open(CEILINGS, encoding="utf-8"))
    c["concept"]["round6_routed"] = routed("n6", "n6_prose", test)
    c["concept"]["v20_routed"] = routed("v20", "v20_prose", test)
    for key, name in (("round6", "n6"), ("v20", "v20")):
        f = W / f"ceiling_numerical_{name}.json"
        if f.exists():
            r = json.load(open(f))
            c["numerical"][key] = {"checkpoint": r["ckpt"], "n": r["n"], "correct": r["correct"]}
    c["_provenance"]["round6"] = ("round6_routed / v20_routed and numerical round6 / v20: the round-6 fine-tune "
                                  "(113M_mid1p6B_n6 e4) and the deployed GRPO v20 step 99, as above "
                                  "(scripts/experiments/build_eval_rounds.py)")
    json.dump(c, open(CEILINGS, "w", encoding="utf-8"))
    print(f"synced {ROUND6!r} and {V20!r} into docs/eval_numerical_real6_final.json, eval_indist.json, eval_ceilings.json")


def main():
    if "--sync" in sys.argv:
        sync_headline()
    sweep ={r["run_name"]: r for r in json.load(open(SWEEP))["runs"]} if SWEEP.exists() else {}
    for name, run in SIZES.items():
        r = sweep.get(run, {})
        M[f"sweep_{name}"] = ((W / f"ctrl_real6_{name}.json", None), (W / f"ctrl_indist_{name}.json", None),
                              dict(step="scaling-sweep base, one-stage fine-tune on notation3", base=run,
                                   params=r.get("N"), tokens=r.get("D"), data="notation3", recipe="one stage"))
    M["sweep_ours"] = ((W / "ctrl_real6_113M_3150M.json", None), (W / "ctrl_indist_113M_3150M.json", None),
                       dict(step="our base, one-stage fine-tune on notation3", base="original", params=OURS_N, tokens=OURS_D,
                            data="notation3", recipe="one stage"))
    out, missing = {}, []
    for name, (src, isrc, meta) in M.items():
        runs, ckpt = results(src)
        if runs is None:
            missing.append(name)
            continue
        row = dict(meta, ckpt=ckpt, real6=per_question(runs))
        iruns, ickpt = results(isrc)
        if iruns is not None and (ickpt is None or ckpt is None or ickpt == ckpt):   # the same checkpoint only
            row["indist"] = {"acc5": per_question(iruns)["acc5"]}
        out[name] = row
        r = row["real6"]
        print(f"{name:20s} acc5 {100 * st.mean(r['acc5'].values()):5.1f}  acc1 {100 * st.mean(r['acc1'].values()):5.1f}  "
              f"vote8 {100 * st.mean(r['vote8'].values()):5.1f}  in-dist "
              f"{100 * st.mean(row['indist']['acc5'].values()) if 'indist' in row else float('nan'):5.1f}  {ckpt or ''}")
    json.dump({"_provenance": __doc__.strip().split("\n\n")[0] + " Per-question means; see the script for each input.",
               "models": out}, open(ROOT / "docs/eval_rounds.json", "w"), indent=0)
    print(f"wrote docs/eval_rounds.json ({len(out)} models; missing: {missing or 'none'})")


if __name__ == "__main__":
    main()
