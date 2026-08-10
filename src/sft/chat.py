#!/usr/bin/env python3
"""Talk to the model: streaming REPL, single prompts, and a quick test suite.

Also the shared generation engine used by checkpoint selection, evaluation, and
the analysis scripts — which is the point. Decoding defaults are pinned here
(temperature 0.3 / top-k 40 / repetition penalty 1.2) so manual inspection and
the benchmark see the same model; at rp 1.05 this model emits 56% repeated
5-grams and the measured effect changes sign (PITFALLS B3).

    python -m src.sft.chat
    python -m src.sft.chat --prompt "How does stress drop relate to slip?"
    python -m src.sft.chat --test
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

# Resolve repository paths and append parent codes directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODES_ROOT = REPO_ROOT.parent
NANOGPT_SEIS_ROOT = CODES_ROOT / "nanogpt_seis"

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from nanogpt_seis.src.model.gqa_gpt import GPT, GPTConfig
from src.sft.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    IM_END_ID,
    IM_START_ID,
    load_bpe_tokenizer,
)
from src.sft.rag_retriever import SeisRAGRetriever


def format_chatml_token_stream(
    tokenizer: Tokenizer,
    history: List[Tuple[str, str]],
    new_user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    im_start_id: int = IM_START_ID,
    im_end_id: int = IM_END_ID,
) -> List[int]:
    """
    Constructs an exact ChatML token sequence from conversation history and user query.

    Args:
        tokenizer: BPE tokenizer instance.
        history: List of past (user_message, assistant_response) tuples.
        new_user_message: Current user query string.
        system_prompt: System persona description.
        im_start_id: Token ID for <|im_start|>.
        im_end_id: Token ID for <|im_end|>.

    Returns:
        List of integer token IDs representing the complete ChatML prompt.
    """
    system_header = tokenizer.encode("system\n", add_special_tokens=False).ids
    user_header = tokenizer.encode("user\n", add_special_tokens=False).ids
    assistant_header = tokenizer.encode("assistant\n", add_special_tokens=False).ids
    newline_ids = tokenizer.encode("\n", add_special_tokens=False).ids

    # 1. System Turn
    system_body = tokenizer.encode(system_prompt.strip() + "\n", add_special_tokens=False).ids
    token_stream = [im_start_id] + system_header + system_body + [im_end_id] + newline_ids

    # 2. History Turns
    for user_msg, assistant_msg in history:
        user_body = tokenizer.encode(user_msg.strip() + "\n", add_special_tokens=False).ids
        token_stream += [im_start_id] + user_header + user_body + [im_end_id] + newline_ids

        # Match training exactly: dataset.py encodes the assistant response
        # with NO newline between the response text and <|im_end|>.
        assistant_body = tokenizer.encode(assistant_msg.strip(), add_special_tokens=False).ids
        token_stream += [im_start_id] + assistant_header + assistant_body + [im_end_id] + newline_ids

    # 3. Current User Turn & Assistant Trigger
    new_user_body = tokenizer.encode(new_user_message.strip() + "\n", add_special_tokens=False).ids
    token_stream += [im_start_id] + user_header + new_user_body + [im_end_id] + newline_ids
    token_stream += [im_start_id] + assistant_header

    return token_stream


class SFTChatEngine:
    """
    Inference and chat engine for fine-tuned nanoGPT-Seis model.
    Handles weight loading, vocabulary extension, and streaming generation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initializes the chat engine.

        Args:
            checkpoint_path: Path to SFT or base model .pt checkpoint file.
            tokenizer_path: Path to tokenizer.json file.
            device: Target torch device ('cuda' or 'cpu').
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # Silent CPU fallback is a trap: generation drops from ~75 tok/s
                # to a few tok/s, and the usual cause is simply the wrong
                # interpreter (a torch build without CUDA) rather than absent
                # hardware. Say so loudly instead of just running slowly.
                device = "cpu"
                print(
                    "[WARN] CUDA not available to this interpreter — running on CPU "
                    f"(expect a few tok/s instead of ~75).\n"
                    f"       torch {torch.__version__} @ {sys.executable}\n"
                    "       If this machine has GPUs, you are probably using the wrong "
                    "python; try the project env, e.g.\n"
                    "       conda run -n nanogpt_seis python -m src.sft.chat ...",
                    flush=True,
                )
        self.device = device

        # Resolve checkpoint path fallback hierarchy, newest stage first. The
        # `selected.pt` entries are the checkpoints chosen on measured behaviour
        # by `src.sft.checkpoints select`, NOT on validation
        # loss. The top-level sft_ckpt_best.pt is the RETRACTED v1 model
        # (contaminated data, leaked split) and must never be the default.
        #
        # Keep this list current when a new run supersedes an old one: a stale
        # entry here means `python -m src.sft.chat` silently exercises the
        # previous generation of the model while appearing to test the new one.
        if checkpoint_path is not None and not checkpoint_path.exists():
            # An explicit --ckpt-path that does not exist is a typo, not a
            # request to fall back to some other model.
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if checkpoint_path is None:
            candidates = [
                REPO_ROOT / "checkpoints" / "grpo" / "selected.pt",
                REPO_ROOT / "checkpoints" / "grpo" / "grpo_ckpt_step_200.pt",
                REPO_ROOT / "checkpoints" / "sft" / "selected.pt",
                REPO_ROOT / "checkpoints" / "sft" / "sft_ckpt_best.pt",
                NANOGPT_SEIS_ROOT / "checkpoints" / "ckpt.pt",
            ]
            checkpoint_path = next((p for p in candidates if p.exists()), None)
            if checkpoint_path is None:
                raise FileNotFoundError("No valid model checkpoint found.")
            print(f"[INFO] No --ckpt-path given; using {checkpoint_path.relative_to(REPO_ROOT)}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config_dict = checkpoint.get("model_cfg", checkpoint.get("config", {}))

        # Instantiate GPT model
        config = GPTConfig(
            vocab_size=model_config_dict.get("vocab_size", 16388),
            block_size=model_config_dict.get("block_size", 4096),
            n_layer=model_config_dict.get("n_layer", 16),
            n_head=model_config_dict.get("n_head", 12),
            n_kv_head=model_config_dict.get("n_kv_head", 4),
            d_model=model_config_dict.get("d_model", 768),
            ffn_multiple_of=model_config_dict.get("ffn_multiple_of", 256),
            rope_theta=model_config_dict.get("rope_theta", 10000.0),
            dropout=0.0,
        )

        model = GPT(config)

        # Resize embeddings if checkpoint has old 16,384 vocab
        state_dict = checkpoint["model"]
        ckpt_vocab_size = state_dict["tok_emb.weight"].shape[0]
        if ckpt_vocab_size != config.vocab_size:
            from src.sft.train_sft import resize_model_embeddings
            resize_model_embeddings(model, new_vocab_size=ckpt_vocab_size)

        model.load_state_dict(state_dict)

        # ChatML prompts use token IDs 16384/16385; a base (pretrain-only)
        # checkpoint has a 16,384-row embedding and would crash on lookup.
        # Extend with randomly initialized rows so the fallback path works,
        # and warn: an un-finetuned model has never seen these tokens.
        if model.tok_emb.weight.shape[0] < 16388:
            from src.sft.train_sft import resize_model_embeddings
            resize_model_embeddings(model, new_vocab_size=16388)
            print("[WARN] Checkpoint has base 16,384 vocab: ChatML special-token "
                  "embeddings are randomly initialized (base model was never "
                  "instruction-tuned; expect raw-LM behavior).")
        self.model = model.eval().to(self.device)
        self.config = config

        # Load Tokenizer
        tok_path = tokenizer_path or (NANOGPT_SEIS_ROOT / "data" / "tokenized" / "tokenizer.json")
        self.tokenizer = load_bpe_tokenizer(tok_path)

        self.amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if "cuda" in self.device
            else torch.autocast("cpu", enabled=False)
        )

    @torch.no_grad()
    def stream_response(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
        no_repeat_ngram: int = 0,
    ) -> Generator[Dict[str, float | str | int], None, None]:
        """
        Generates response tokens using KV-cache streaming, yielding token text updates.

        Args:
            prompt_tokens: List of ChatML prompt token IDs.
            max_new_tokens: Maximum response tokens to generate.
            temperature: Sampling temperature (higher = more creative).
            top_k: Top-k filtering count.
            repetition_penalty: Penalty multiplier for already generated tokens.
            no_repeat_ngram: N-gram size to hard-ban repetition.

        Yields:
            Dict containing {'text_chunk': str, 'ttft_ms': float, 'total_tokens': int, 'tok_per_sec': float}.
        """
        input_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)[None]

        generated_token_ids: List[int] = []
        previous_text = ""
        start_time = time.time()
        first_token_time: Optional[float] = None

        with self.amp_context:
            for token_tensor in self.model.generate_stream(
                idx=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            ):
                if first_token_time is None:
                    first_token_time = time.time()

                token_id = int(token_tensor[0, 0])

                # Natural early stopping on <|im_end|> or <|endoftext|>
                if token_id in (IM_END_ID, 0):
                    break

                generated_token_ids.append(token_id)
                current_text = self.tokenizer.decode(generated_token_ids)

                if len(current_text) > len(previous_text):
                    text_chunk = current_text[len(previous_text):]
                    previous_text = current_text

                    ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else 0.0
                    elapsed = time.time() - start_time
                    tok_per_sec = len(generated_token_ids) / max(0.001, elapsed)

                    yield {
                        "text_chunk": text_chunk,
                        "full_text": current_text,
                        "ttft_ms": ttft_ms,
                        "total_tokens": len(generated_token_ids),
                        "tok_per_sec": tok_per_sec,
                    }

    def generate_single_prompt(
        self,
        user_query: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
    ) -> Tuple[str, Dict[str, float]]:
        """Convenience method to generate a full response for a single user query."""
        prompt_tokens = format_chatml_token_stream(
            tokenizer=self.tokenizer,
            history=[],
            new_user_message=user_query,
            system_prompt=system_prompt,
        )

        full_text = ""
        stats = {"ttft_ms": 0.0, "total_tokens": 0, "tok_per_sec": 0.0}

        for update in self.stream_response(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ):
            full_text = str(update["full_text"])
            stats["ttft_ms"] = float(update["ttft_ms"])
            stats["total_tokens"] = int(update["total_tokens"])
            stats["tok_per_sec"] = float(update["tok_per_sec"])

        return full_text, stats


def run_interactive_repl(
    engine: SFTChatEngine,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    use_rag: bool = False,
    retriever: Optional[SeisRAGRetriever] = None,
    temperature: float = 0.3,
    top_k: int = 40,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.2,
):
    """Launches an interactive terminal conversation session (REPL)."""
    print("================================================================================")
    print("🌍 SeisGPT Assistant — Interactive Terminal (nanoGPT-Seis-Posttrain)")
    print(f"   Model: {engine.config.d_model}d / {engine.config.n_layer}L ({engine.model.num_params():,} params)")
    print(f"   Context Window: {engine.config.block_size} tokens | Device: {engine.device}")
    print(f"   RAG Grounding: {'[ON]' if use_rag else '[OFF] (type /rag to toggle)'}")
    print("   Type '/clear' to reset history, '/rag' to toggle context, '/exit' to quit.")
    print("================================================================================\n")

    history: List[Tuple[str, str]] = []
    rag_active = use_rag
    if rag_active and retriever is None:
        retriever = SeisRAGRetriever()

    while True:
        try:
            user_input = input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting SeisGPT terminal.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "exit", "quit"):
            print("Exiting SeisGPT terminal.")
            break
        if user_input.lower() in ("/clear", "clear"):
            history.clear()
            print("[INFO] Conversation history cleared.")
            continue
        if user_input.lower() in ("/rag", "rag"):
            rag_active = not rag_active
            if rag_active and retriever is None:
                retriever = SeisRAGRetriever()
            status_str = "ENABLED (Context Grounding ON)" if rag_active else "DISABLED (Zero-Shot ON)"
            print(f"[INFO] RAG Retriever mode is now {status_str}.")
            continue

        formatted_user_query = user_input
        if rag_active and retriever is not None:
            results = retriever.search(user_input, top_k=1)
            if results:
                doc, score = results[0]
                print(f"\033[90m[RAG Retriever: Fetched chunk '{doc['chunk_id']}' | Score: {score:.2f}]\033[0m")
                formatted_user_query = f"CONTEXT EXCERPT:\n---\n{doc['text'][:1500]}\n---\n\nQUESTION:\n{user_input}"

        prompt_tokens = format_chatml_token_stream(
            tokenizer=engine.tokenizer,
            history=history,
            new_user_message=formatted_user_query,
            system_prompt=system_prompt,
        )

        print("\nAssistant > ", end="", flush=True)

        full_response = ""
        last_stats = {"ttft_ms": 0.0, "total_tokens": 0, "tok_per_sec": 0.0}

        for update in engine.stream_response(
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        ):
            chunk = str(update["text_chunk"])
            print(chunk, end="", flush=True)
            full_response = str(update["full_text"])
            last_stats["ttft_ms"] = float(update["ttft_ms"])
            last_stats["total_tokens"] = int(update["total_tokens"])
            last_stats["tok_per_sec"] = float(update["tok_per_sec"])

        print(f"\n\n\033[90m[Stats: TTFT = {last_stats['ttft_ms']:.1f} ms | Generated {last_stats['total_tokens']} tokens | Speed = {last_stats['tok_per_sec']:.1f} tok/s]\033[0m")
        history.append((user_input, full_response))


def run_benchmark_eval_test(
    engine: SFTChatEngine,
    use_rag: bool = False,
    temperature: float = 0.3,
    top_k: int = 40,
    max_new_tokens: int = 400,
    repetition_penalty: float = 1.2,
):
    """Runs a series of domain benchmark evaluation prompts to verify assistant quality and throughput.

    These are qualitative smoke prompts with known-correct answers, useful for
    eyeballing response quality. For quantitative, held-out, paired
    scoring use `python -m src.rl.eval_rl` instead.
    """
    benchmark_prompts = [
        # Formula recall — canonical, should be exactly right
        "What is the mathematical relationship between seismic moment M_0, fault rupture area A, and average slip d?",
        # Conceptual — textbook physics, no paper-specific facts needed
        "Explain the physical difference between P-waves and S-waves in seismic wave propagation.",
        # Arithmetic — the known weak spot; checks whether <think> closes and an answer appears
        "Calculate the approximate stress drop for an earthquake given a fault length L = 20 km, width W = 10 km, and average slip d = 0.8 m.",
        # Conceptual mechanism
        "What are the key mechanisms causing soil liquefaction during major earthquakes?",
        # Depth/mechanism question the v1 model got badly wrong (answered 10-30 km)
        "Why do deep-focus earthquakes occur in subduction zones but not in the shallow crust?",
    ]

    retriever = SeisRAGRetriever() if use_rag else None

    print("================================================================================")
    print(f"🧪 Running Automated Domain Evaluation Benchmark (RAG Mode: {use_rag})")
    print("================================================================================\n")

    for idx, prompt in enumerate(benchmark_prompts, start=1):
        print(f"--- [Test {idx}/{len(benchmark_prompts)}] User Prompt ---")
        print(f"{prompt}\n")

        query_to_send = prompt
        if use_rag and retriever is not None:
            results = retriever.search(prompt, top_k=1)
            if results:
                doc, score = results[0]
                print(f"[RAG Context Attached: Chunk '{doc['chunk_id']}' | Score: {score:.2f}]")
                query_to_send = f"CONTEXT EXCERPT:\n---\n{doc['text'][:1500]}\n---\n\nQUESTION:\n{prompt}"

        print("Response:")
        prompt_tokens = format_chatml_token_stream(
            tokenizer=engine.tokenizer,
            history=[],
            new_user_message=query_to_send,
        )

        last_stats = {"ttft_ms": 0.0, "total_tokens": 0, "tok_per_sec": 0.0}
        for update in engine.stream_response(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ):
            print(str(update["text_chunk"]), end="", flush=True)
            last_stats["ttft_ms"] = float(update["ttft_ms"])
            last_stats["total_tokens"] = int(update["total_tokens"])
            last_stats["tok_per_sec"] = float(update["tok_per_sec"])

        print(f"\n\n\033[90m[Metrics: TTFT = {last_stats['ttft_ms']:.1f} ms | Tokens = {last_stats['total_tokens']} | Throughput = {last_stats['tok_per_sec']:.1f} tok/s]\033[0m\n")

    print("[SUCCESS] Evaluation benchmark suite completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="SeisGPT Assistant Chat & Evaluation Terminal")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt execution mode")
    parser.add_argument("--rag", action="store_true", help="Enable RAG Context Grounding mode")
    parser.add_argument("--test", action="store_true", help="Run automated domain evaluation benchmark")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling count")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Penalty on already-generated tokens. Below ~1.15 this model loops badly "
                             "(at temp 0.2 / rp 1.05, 56%% of 5-grams are verbatim repeats).")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new response tokens")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System persona prompt")
    parser.add_argument("--device", type=str, default=None,
                        help="Force a device, e.g. 'cuda', 'cuda:1', 'cpu' (default: cuda if available)")
    args = parser.parse_args()

    engine = SFTChatEngine(
        checkpoint_path=Path(args.ckpt_path) if args.ckpt_path else None,
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
        device=args.device,
    )
    # Report the device in every mode; previously only the REPL banner showed it,
    # so --prompt / --test runs could sit on CPU without any indication.
    print(f"[INFO] Device: {engine.device}", flush=True)

    retriever = SeisRAGRetriever() if args.rag else None

    if args.test:
        run_benchmark_eval_test(
            engine,
            use_rag=args.rag,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=min(args.max_new_tokens, 400),
            repetition_penalty=args.repetition_penalty,
        )
    elif args.prompt:
        user_query = args.prompt
        if args.rag and retriever is not None:
            results = retriever.search(user_query, top_k=1)
            if results:
                doc, score = results[0]
                print(f"[RAG Context Attached: Chunk '{doc['chunk_id']}' | Score: {score:.2f}]")
                user_query = f"CONTEXT EXCERPT:\n---\n{doc['text'][:1500]}\n---\n\nQUESTION:\n{args.prompt}"

        print(f"User Query: {args.prompt}\n")
        response, stats = engine.generate_single_prompt(
            user_query=user_query,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print("Assistant Response:")
        print(response)
        print(f"\n[Stats: TTFT = {stats['ttft_ms']:.1f} ms | Tokens = {stats['total_tokens']} | Speed = {stats['tok_per_sec']:.1f} tok/s]")
    else:
        run_interactive_repl(
            engine,
            system_prompt=args.system_prompt,
            use_rag=args.rag,
            retriever=retriever,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )


if __name__ == "__main__":
    main()
