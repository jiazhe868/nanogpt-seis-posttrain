#!/usr/bin/env python3
"""Generate instruction-response pairs from the pretraining corpus, via a teacher API.

Samples text chunks from ../nanogpt_seis/data, asks a large model for grounded
Q&A with <think> traces, and appends to JSONL incrementally so the run is
resumable. Concurrent, with rate-limit backoff, and stdlib-only HTTP.

Output is raw: run `src.sft.build_corpus clean` on it before use.

    export TEACHER_API_KEY=...
    python -m src.sft.generate_sft_data --output data/sources/paper_raw.jsonl \
        --target-pairs 10000 --concurrency 10
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

# System Prompt instructing the model on taxonomy and format
SYSTEM_PROMPT = """You are an expert seismologist and AI data curator.
Your task is to analyze the provided scientific paper/text excerpt and generate high-quality, grounded instruction-response pairs for fine-tuning a domain-specific seismology language model (context capacity: 4096 tokens).

REQUIREMENTS:
1. Generate EXACTLY 2 or 3 distinct, high-rigor instruction-response pairs based on the text.
2. Direct Scientific Grounding: All facts, equations, parameters, derivations, and conclusions MUST be grounded in or logically synthesized from the text.
3. Diverse & Deep Question Taxonomy:
   - Type A (Conceptual & Physical Synthesis): Explaining physical mechanisms, wave propagation dynamics, fault rupture, or seismic hazard models in depth.
   - Type B (Formulaic & Numerical Reasoning): Detailed step-by-step calculations or quantitative reasoning (e.g. seismic moment, magnitude scaling, stress drop, velocity ratios, distance decay).
   - Type C (Summarization & Parameter Extraction): Extracting multi-station/multi-event parameters (depth, focal mechanism, strike/dip/rake, PGA, location) or synthesizing overarching paper methodology and conclusions.
4. Step-by-Step Reasoning Format: For Type B questions, wrap the complete mathematical derivation or step-by-step logic inside <think>...</think> tags before giving the final answer.
5. Response Depth & Thoroughness: Do NOT produce superficial 1-sentence answers. Provide complete, academic, multi-paragraph explanations (up to 1,500-2,000 tokens) that thoroughly cover the scientific reasoning.
6. Tone & Phrasing: Do NOT say "according to the text" or "in this paper". Frame questions as realistic, expert user queries to a domain AI assistant (SeisGPT).
7. Language: Output in clean, academic, professional English.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema (no markdown blocks around the JSON):
{
  "pairs": [
    {
      "type": "conceptual",
      "instruction": "User question here...",
      "response": "Detailed multi-paragraph expert response..."
    },
    {
      "type": "numerical",
      "instruction": "Quantitative question here...",
      "response": "<think>Step-by-step derivation...</think>Final answer..."
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """TEXT EXCERPT:
---
{text_chunk}
---

Generate 2 or 3 grounded, high-rigor instruction-response pairs based on the above text. Ensure responses are thorough, detailed, and mathematically/conceptually complete. Return ONLY the JSON object."""


def load_completed_chunk_ids(output_file: str) -> Set[str]:
    """Load chunk hashes that have already been processed to support resuming."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "chunk_id" in data:
                    completed.add(data["chunk_id"])
            except Exception:
                continue
    return completed


def chunk_text(text: str, chunk_size: int = 8000, min_size: int = 1000) -> List[str]:
    """Split text into manageable chunks of approximately chunk_size characters."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_length + len(para) > chunk_size and current_length >= min_size:
            chunk_str = "\n\n".join(current_chunk)
            chunks.append(chunk_str)
            current_chunk = [para]
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += len(para)

    if current_chunk and current_length >= min_size:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def load_chunks_from_data(
    data_dir: str, max_chunks: int = 5000, chunk_size: int = 8000, domain_only: bool = True
) -> List[Tuple[str, str, str]]:
    """
    Scans data_dir for processed or raw JSONL files and returns a list of tuples:
    (chunk_id, source_name, text_chunk)
    """
    domain_sources = {"fulltext", "arxiv", "eartharxiv", "openalex", "substack", "journals", "preprints"}
    domain_keywords = {
        "earthquake", "seismic", "seismology", "fault", "tectonic", "tsunami",
        "seismometer", "epicenter", "hypocenter", "subduction", "lithosphere",
        "attenuation", "ground motion", "shear wave", "p-wave", "s-wave", "magnitude"
    }

    items = []
    processed_file = os.path.join(data_dir, "processed", "train.jsonl")
    raw_dir = os.path.join(data_dir, "raw")

    file_paths = []
    if os.path.exists(processed_file):
        file_paths.append((processed_file, "processed"))
    elif os.path.exists(raw_dir):
        for fname in ["fulltext.jsonl", "preprints.jsonl", "wikipedia.jsonl", "substack.jsonl", "arxiv.jsonl"]:
            fpath = os.path.join(raw_dir, fname)
            if os.path.exists(fpath):
                file_paths.append((fpath, fname.replace(".jsonl", "")))

    if not file_paths:
        print(f"[WARNING] No corpus JSONL files found in {data_dir}. Checking raw folder...")
        if os.path.exists(raw_dir):
            for fn in os.listdir(raw_dir):
                if fn.endswith(".jsonl") and not fn.startswith("fineweb") and not fn.startswith("wiki_general"):
                    file_paths.append((os.path.join(raw_dir, fn), fn.replace(".jsonl", "")))

    print(f"[INFO] Scanning {len(file_paths)} corpus files for text chunks...")
    for fpath, label in file_paths:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        src = doc.get("source", label)
                        text = doc.get("text", doc.get("full_text", ""))
                        if not text or len(text) < 500:
                            continue

                        if domain_only:
                            if src in ("general", "fineweb_edu", "wiki_general"):
                                continue
                            if src not in domain_sources:
                                text_lower = text[:2000].lower()
                                if not any(kw in text_lower for kw in domain_keywords):
                                    continue

                        doc_chunks = chunk_text(text, chunk_size=chunk_size)
                        for c_idx, c_text in enumerate(doc_chunks):
                            chunk_id = hashlib.md5(f"{doc.get('id', line_idx)}_{c_idx}_{c_text[:50]}".encode()).hexdigest()
                            items.append((chunk_id, src, c_text))
                            if len(items) >= max_chunks * 2:
                                break
                    except Exception:
                        continue
                    if len(items) >= max_chunks * 2:
                        break
        except Exception as e:
            print(f"[WARN] Error reading {fpath}: {e}")

    random.seed(42)
    random.shuffle(items)
    print(f"[INFO] Prepared total {len(items)} text chunks for SFT generation.")
    return items[:max_chunks]


def call_llm_api(
    prompt: str,
    api_key: str,
    api_base: str = "",
    model: str = "",
    max_tokens: int = 2500,
    max_retries: int = 5,
) -> Optional[str]:
    """Call OpenAI-compatible API endpoint via standard urllib with backoff retries."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    req_data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                return res_json["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8") if e.fp else str(e)
            if e.code in (429, 500, 502, 503, 504):
                sleep_time = min(2 ** attempt + random.uniform(0.5, 1.5), 30)
                time.sleep(sleep_time)
            else:
                print(f"[API Error] HTTP {e.code}: {err_msg}")
                break
        except Exception as e:
            sleep_time = min(2 ** attempt, 15)
            time.sleep(sleep_time)

    return None


def parse_json_response(response_str: str) -> List[Dict[str, str]]:
    """Clean and parse JSON output from the LLM response."""
    if not response_str:
        return []
    
    # Strip potential markdown fence wrappers if present
    cleaned = response_str.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "pairs" in data:
            return data["pairs"]
        elif isinstance(data, list):
            return data
    except Exception:
        # Fallback regex extraction
        match = re.search(r"\{\s*\"pairs\":\s*\[.*\]\s*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))["pairs"]
            except Exception:
                pass
    return []


def process_single_chunk(
    item: Tuple[str, str, str],
    api_key: str,
    api_base: str,
    model: str,
    max_tokens: int = 2500,
) -> Tuple[str, str, str, List[Dict[str, str]]]:
    """Worker function for processing a single chunk."""
    chunk_id, source, text_chunk = item
    prompt = USER_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
    raw_resp = call_llm_api(prompt=prompt, api_key=api_key, api_base=api_base, model=model, max_tokens=max_tokens)
    pairs = parse_json_response(raw_resp) if raw_resp else []
    return chunk_id, source, text_chunk, pairs


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset for nanoGPT-Seis using LLM API.")
    parser.add_argument("--data-dir", type=str, default="../nanogpt_seis/data", help="Path to nanogpt_seis data directory")
    parser.add_argument("--output", type=str, default="data/sources/paper_raw.jsonl", help="Output JSONL filepath")
    parser.add_argument("--target-pairs", type=int, default=10000, help="Target total number of Q&A pairs to generate")
    parser.add_argument("--max-chunks", type=int, default=5000, help="Max text chunks to process")
    parser.add_argument("--chunk-size", type=int, default=8000, help="Approx character length of text chunks (default: 8000 ~ 2000 tokens)")
    parser.add_argument("--max-tokens", type=int, default=2500, help="Max API response tokens (default: 2500 tokens)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent API worker threads")
    parser.add_argument("--model", type=str, default=os.getenv("TEACHER_MODEL", ""),
                        help="teacher model name exposed by --api-base")
    parser.add_argument("--api-base", type=str, default=os.getenv("TEACHER_API_BASE", ""),
                        help="OpenAI-compatible base URL, e.g. https://<host>/v1")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key (or set TEACHER_API_KEY / OPENAI_API_KEY)")
    args = parser.parse_args()

    # Determine API key
    api_key = args.api_key or os.getenv("TEACHER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] API key must be provided via --api-key or "
              "TEACHER_API_KEY / OPENAI_API_KEY.")
        sys.exit(1)

    if not args.api_base or not args.model:
        print("[ERROR] No teacher endpoint. Pass --api-base and --model, or set "
              "TEACHER_API_BASE / TEACHER_MODEL. Any OpenAI-compatible endpoint works.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    completed_chunk_ids = load_completed_chunk_ids(args.output)
    print(f"[INFO] Resuming generation. Found {len(completed_chunk_ids)} already completed chunk IDs in {args.output}.")

    chunks = load_chunks_from_data(args.data_dir, max_chunks=args.max_chunks, chunk_size=args.chunk_size)
    pending_chunks = [c for c in chunks if c[0] not in completed_chunk_ids]
    print(f"[INFO] {len(pending_chunks)} chunks pending API generation.")

    if not pending_chunks:
        print("[INFO] All requested chunks have already been processed.")
        return

    total_generated_pairs = 0
    successful_chunks = len(completed_chunk_ids)

    print(f"[INFO] Starting concurrent SFT generation using model='{args.model}' (threads={args.concurrency}, chunk_size={args.chunk_size}, max_tokens={args.max_tokens})...")
    start_time = time.time()

    with open(args.output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_single_chunk, item, api_key, args.api_base, args.model, args.max_tokens): item
                for item in pending_chunks
            }

            for idx, future in enumerate(as_completed(futures), 1):
                chunk_id, source, text_chunk, pairs = future.result()
                if pairs:
                    record = {
                        "chunk_id": chunk_id,
                        "source": source,
                        "pair_count": len(pairs),
                        "pairs": pairs,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    total_generated_pairs += len(pairs)
                    successful_chunks += 1

                if idx % 10 == 0 or idx == len(pending_chunks):
                    elapsed = time.time() - start_time
                    cps = idx / elapsed if elapsed > 0 else 0
                    print(
                        f" Progress: [{idx}/{len(pending_chunks)}] chunks processed | "
                        f"New Pairs: {total_generated_pairs} | Speed: {cps:.2f} chunks/s"
                    )

                if total_generated_pairs >= args.target_pairs:
                    print(f"[INFO] Target pairs count of {args.target_pairs} reached. Stopping.")
                    break

    print(f"[SUCCESS] SFT data generation complete! Output saved to {args.output}")


if __name__ == "__main__":
    main()
