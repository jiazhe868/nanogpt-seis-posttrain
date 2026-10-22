"""A general instruction model, without any domain post-training, on this project's evaluation sets.

Same questions, same system prompt as the post-trained models (no extra instructions), same scoring:
  numerical: the last 3 numbers of the visible answer, within 5% / 1% (NumericalVerifier tail rule);
             LaTeX number formats (\\times 10^{19}, \\cdot, $...$, braces) are normalised first, since this
             model writes them and the post-trained models do not
  conceptual: share of the prompt's key concepts in the visible answer (as in ceiling_check.py)
Sets: the 153 real held-out numerical questions (4 samples), data/eval/indist_300.jsonl (4 samples),
the 641 non-numerical held-out prompts (2 samples).
README §1.4 uses its conceptual answers (key concepts at equal length); its free-prose numerical scores are a
reference only, since the deployed model is scored on one executed answer (general_models_strict.py).

    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/experiments/llm_baseline_eval.py --model Qwen/Qwen3.5-9B \\
        --quantization fp8 --out work/experiments/baseline_qwen9b.json [--think --sets numerical,indist]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rl.verifiers import NumericalVerifier  # noqa: E402
from src.sft.dataset import DEFAULT_SYSTEM_PROMPT  # noqa: E402

NV = NumericalVerifier()


def visible(t):
    return re.sub(r"^.*?</think>", "", t, flags=re.DOTALL)


def normalise(t):
    t = t.replace("\\times", "×").replace("\\cdot", "·").replace("\\,", "").replace("\\!", "")
    t = re.sub(r"\\[()\[\]]|\$", " ", t)
    return t.replace("{", "").replace("}", "")


def hits(text, target):
    nums = NV.extract_numbers(normalise(visible(text)))
    if not nums:
        return False, False
    den = abs(target) if abs(target) > 1e-7 else 1.0
    err = min(abs(v - target) / den for v in nums[-NV.ANSWER_TAIL_NUMBERS:])
    return err <= 0.05, err <= 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--sets", default="numerical,indist,concept")
    ap.add_argument("--numerical-file", default=None, help="items to score as the numerical set (e.g. hinted prompts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=None, help="64 for the 27B (per-sequence recurrent state)")
    ap.add_argument("--max-batched-tokens", type=int, default=None)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    extra = {k: v for k, v in (("max_num_seqs", a.max_num_seqs), ("max_num_batched_tokens", a.max_batched_tokens)) if v}
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, quantization=a.quantization, max_model_len=8192,
              gpu_memory_utilization=0.92 if extra else 0.9, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, **extra)
    test = [json.loads(l) for l in open(ROOT / "data/rl/test.jsonl", encoding="utf-8")]
    num_items = [json.loads(l) for l in open(a.numerical_file, encoding="utf-8")] if a.numerical_file else test
    sets = {"numerical": ([it for it in num_items if it["type"] == "numerical" and it.get("final_numerical_value") is not None], 4),
            "indist": ([json.loads(l) for l in open(ROOT / "data/eval/indist_300.jsonl", encoding="utf-8")], 4),
            "concept": ([it for it in test if it["type"] != "numerical" and it.get("key_concepts")], 2)}
    out = {"model": a.model, "quantization": a.quantization, "think": a.think, "max_tokens": a.max_tokens,
           "temperature": a.temperature, "system_prompt": DEFAULT_SYSTEM_PROMPT, "results": {}}
    for name in a.sets.split(","):
        items, n = sets[name]
        sp = SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, n=n, seed=0)
        msgs = [[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": it["prompt"]}] for it in items]
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": a.think}, use_tqdm=True)
        rows = []
        for it, o in zip(items, outs):
            texts = [c.text for c in o.outputs]
            if name == "concept":
                cov = [sum(k.lower() in visible(t).lower() for k in it["key_concepts"]) / len(it["key_concepts"]) for t in texts]
                rows.append({"id": it["id"], "coverage": sum(cov) / len(cov),
                             "words": sum(len(visible(t).split()) for t in texts) / len(texts),
                             "texts": [visible(t)[:6000] for t in texts]})
            else:
                h = [hits(t, float(it["final_numerical_value"])) for t in texts]
                rows.append({"id": it["id"], "acc5": sum(x[0] for x in h) / len(h), "acc1": sum(x[1] for x in h) / len(h),
                             "tokens": sum(len(c.token_ids) for c in o.outputs) / len(texts),
                             "truncated": sum(c.finish_reason == "length" for c in o.outputs) / len(texts),
                             "texts": [visible(t)[-2500:] for t in texts]})
        out["results"][name] = {"rows": rows, "example": outs[0].outputs[0].text[:1500]}
        key = "coverage" if name == "concept" else "acc5"
        print(f"{name}: {key} {100 * sum(r[key] for r in rows) / len(rows):.1f}% over {len(rows)} items", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
