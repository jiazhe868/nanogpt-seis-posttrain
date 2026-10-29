"""Multi-seed evaluation of one checkpoint on any numerical problem file (README §1.2 in-distribution panel).

usage: [SAMPLES=4] [TEMP=0.3] [BATCH=prompts per batch] set_eval.py out.json items.jsonl ckpt seeds decoder name
    decoder: prose      (T 0.3, top-k 40, rp 1.2: the Phase II/III models)
             notation   (executed paper-notation equations, T 0.3, top-k 40, rp 1.0: Phase IV and later)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.rl.eval_rl import evaluate_checkpoint  # noqa: E402
from src.sft.chat import SFTChatEngine, format_chatml_token_stream  # noqa: E402
from src.sft.dataset import DEFAULT_SYSTEM_PROMPT  # noqa: E402

out, items_path, ck, seeds, decoder, name = sys.argv[1:7]
items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
items = [it for it in items if it.get("type") == "numerical" and it.get("final_numerical_value") is not None]
# A model with a short context (the scaling sweep's 1,024 tokens) is scored on the prompts that leave room for
# 384 new tokens; the result lists the questions it answered, so comparisons can use the common ones.
engine = SFTChatEngine(checkpoint_path=Path(ck))
room = engine.model.cfg.block_size - 384
fits = [len(format_chatml_token_stream(engine.tokenizer, [], it["prompt"], DEFAULT_SYSTEM_PROMPT)) <= room for it in items]
del engine
if not all(fits):
    print(f"{name}: {len(fits) - sum(fits)} of {len(fits)} prompts leave no room for 384 new tokens; skipped")
    items = [it for it, f in zip(items, fits) if f]
kw = dict(pot=True, pot_notation=True, repetition_penalty=1.0) if decoder == "notation" else dict(repetition_penalty=1.2)
n = int(os.environ.get("SAMPLES", 4))
runs = [evaluate_checkpoint(Path(ck), items, num_samples=n, temperature=float(os.environ.get("TEMP", 0.3)), seed=int(s),
                            batched=True, batch_prompts=int(os.environ.get("BATCH", max(1, 64 // n))), **kw)
        for s in seeds.split(",")]
json.dump({"ckpt": ck, "items": items_path, "decoder": decoder, "seeds": seeds, "results": {name: runs}}, open(out, "w"))
print(name, [round(r["numerical_acc_5pct"], 4) for r in runs])
