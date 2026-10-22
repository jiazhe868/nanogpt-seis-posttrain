"""General models scored like the deployed model: one executed answer per question (README §1.4).

The Phase IV blind-solve protocol (`tools/gen_pot_problems.py solve`) for any model, several seeds, one load.

One Python program per question (SOLVE_PROMPT + RULES), T 0.3, top-p 0.95, 1,500 new tokens, no thinking;
correct ("verified") when the executed `answer` is within 1% of the label. Seed 0 with the 27B is the run
in docs/teacher_27b_heldout_blind.jsonl.

    python scripts/experiments/general_models_strict.py <model> <name> [quantization|none] [seeds, default 0,1,2,3] [items.jsonl]
(items: any numerical set with `prompt` / `final_numerical_value`; default data/rl/test.jsonl, the 153
held-out questions)
Writes work/experiments/blind_<name>.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import gen_pot_problems as G  # noqa: E402  (also sets the vLLM sampler / CUDA environment)


def main():
    model, name = sys.argv[1:3]
    quant = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "none" else None
    seeds = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2,3").split(",")]
    from vllm import LLM, SamplingParams
    items = sys.argv[5] if len(sys.argv) > 5 else ROOT / "data/rl/test.jsonl"
    rows = [{"id": x["id"], "question": x["prompt"], "answer": float(x["final_numerical_value"])}
            for x in map(json.loads, open(items, encoding="utf-8"))
            if x.get("type", "numerical") == "numerical" and x.get("final_numerical_value") is not None]
    llm = LLM(model=model, tensor_parallel_size=2, max_model_len=8192, gpu_memory_utilization=0.92, quantization=quant,
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, max_num_seqs=64, max_num_batched_tokens=4096)
    prompts = [G.SOLVE_PROMPT.format(q=r["question"], rules=G.RULES) for r in rows]
    res = {}
    for s in seeds:
        texts = G.chat(llm, SamplingParams(temperature=0.3, top_p=0.95, max_tokens=1500, seed=s), prompts, think=False)
        out = []
        for r, t in zip(rows, texts):
            a2, _ = G.run_program(t)
            out.append({"id": r["id"], "answer": r["answer"], "answer2": a2, "program2": G.extract_program(t),
                        "verified": a2 is not None and G.close(a2, r["answer"], 0.01),
                        "within5": a2 is not None and G.close(a2, r["answer"], 0.05)})
        res[s] = out
        print(f"{name} seed {s}: within 1% {100 * sum(o['verified'] for o in out) / len(out):.1f}%  "
              f"within 5% {100 * sum(o['within5'] for o in out) / len(out):.1f}%  "
              f"no answer {100 * sum(o['answer2'] is None for o in out) / len(out):.1f}%", flush=True)
    json.dump({"model": model, "quantization": quant, "protocol": "tools/gen_pot_problems.py solve, T 0.3, 1500 tokens",
               "seeds": res}, open(ROOT / f"work/experiments/blind_{name}.json", "w"))


if __name__ == "__main__":
    main()
