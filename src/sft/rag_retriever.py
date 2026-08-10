#!/usr/bin/env python3
"""BM25 retrieval over the pretraining corpus, for open-book chat.

Indexes the paper chunks in ../nanogpt_seis/data/raw/*.jsonl and returns
context in under a millisecond. Used by chat.py; the open-book experiment it
would support is scoped but not yet run (README section 10).

    python -m src.sft.rag_retriever --query "seismic moment and stress drop"
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRETRAIN_RAW_DIR = REPO_ROOT.parent / "nanogpt_seis" / "data" / "raw"
FALLBACK_DATA_PATH = REPO_ROOT / "data" / "sources" / "paper_raw.jsonl"


def tokenize_text(text: str) -> List[str]:
    """Cleans and tokenizes text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_\-\^\.]+\b", text.lower())


def clean_paper_text(text: str, max_chars: int = 1500) -> str:
    """Cleans raw paper text by removing excessive headers, whitespace, and bibliographies."""
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Strip references / bibliography
    ref_match = re.search(r"\n\s*(References|BIBLIOGRAPHY|Acknowledgements)\b", text, flags=re.IGNORECASE)
    if ref_match:
        text = text[:ref_match.start()]

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(Figure|Fig\.|Table)\s+\d+[\:\.]\s*", stripped, flags=re.IGNORECASE) and len(stripped) < 120:
            continue
        cleaned_lines.append(stripped)

    text = " ".join(cleaned_lines)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_chars:
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > max_chars // 2:
            text = truncated[: last_period + 1]
        else:
            text = truncated

    return text


class SeisRAGRetriever:
    """
    Lightweight Okapi BM25 retriever indexed on authentic domain paper text chunks.
    """

    def __init__(self, raw_dir: Optional[Union[str, Path]] = None, max_chunks: int = 5000):
        self.raw_dir = Path(raw_dir) if raw_dir else PRETRAIN_RAW_DIR
        self.max_chunks = max_chunks

        self.chunks: List[Dict[str, str]] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf_weights: Dict[str, float] = {}

        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0

        self._build_index()

    def _build_index(self) -> None:
        """Reads JSONL files from pretraining raw dir and indexes text chunks."""
        candidate_files = []
        if self.raw_dir.exists():
            for fname in ["arxiv.jsonl", "journals.jsonl", "preprints.jsonl", "fulltext.jsonl"]:
                fpath = self.raw_dir / fname
                if fpath.exists():
                    candidate_files.append(fpath)

        if not candidate_files and FALLBACK_DATA_PATH.exists():
            candidate_files.append(FALLBACK_DATA_PATH)

        for fpath in candidate_files:
            if len(self.chunks) >= self.max_chunks:
                break
            try:
                line_count = 0
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(self.chunks) >= self.max_chunks or line_count >= 500:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        line_count += 1
                        rec = json.loads(line)
                        raw_text = rec.get("text", "") or rec.get("chunk_text", "")
                        if not raw_text and "pairs" in rec:
                            raw_text = "\n\n".join(
                                f"{p.get('instruction', '')}\n{p.get('response', '')}"
                                for p in rec["pairs"]
                            )

                        cleaned = clean_paper_text(raw_text)
                        if len(cleaned) >= 80:
                            tokens = tokenize_text(cleaned)
                            if len(tokens) >= 20:
                                tf = Counter(tokens)
                                doc_id = rec.get("id", f"doc_{len(self.chunks)}")
                                self.chunks.append({
                                    "chunk_id": str(doc_id),
                                    "title": rec.get("title", "Seismology Research Paper"),
                                    "source": fpath.stem,
                                    "text": cleaned,
                                })
                                self.doc_term_freqs.append(tf)
                                self.doc_lengths.append(len(tokens))

                                for term in tf:
                                    self.doc_freqs[term] += 1
            except Exception as e:
                print(f"[WARNING] Error reading {fpath}: {e}")

        self.num_docs = len(self.chunks)
        if self.num_docs == 0:
            print("[WARNING] Zero valid documents indexed for RAG retriever.")
            return

        self.avg_doc_len = sum(self.doc_lengths) / self.num_docs

        for term, df in self.doc_freqs.items():
            self.idf_weights[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        print(f"[INFO] SeisRAGRetriever indexed {self.num_docs} authentic paper chunks ({len(self.idf_weights)} terms).")

    def search(
        self,
        query: str,
        top_k: int = 1,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Tuple[Dict[str, str], float]]:
        """Searches index using Okapi BM25 scoring."""
        if self.num_docs == 0:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.num_docs

        for term in query_tokens:
            if term not in self.idf_weights:
                continue
            idf = self.idf_weights[term]

            for doc_idx, tf in enumerate(self.doc_term_freqs):
                term_count = tf.get(term, 0)
                if term_count > 0:
                    doc_len = self.doc_lengths[doc_idx]
                    tf_score = (term_count * (k1 + 1.0)) / (
                        term_count + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    )
                    scores[doc_idx] += idf * tf_score

        ranked_indices = sorted(range(self.num_docs), key=lambda i: scores[i], reverse=True)
        top_results = []

        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.0:
                top_results.append((self.chunks[idx], scores[idx]))

        if not top_results and self.num_docs > 0:
            top_results.append((self.chunks[0], 0.0))

        return top_results


def main():
    parser = argparse.ArgumentParser(description="SeisRAGRetriever Document Retrieval Test")
    parser.add_argument("--query", type=str, default="What is seismic moment M_0 and stress drop?", help="Search query")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k documents to retrieve")
    args = parser.parse_args()

    retriever = SeisRAGRetriever()
    results = retriever.search(query=args.query, top_k=args.top_k)

    print(f"\nQuery: '{args.query}'")
    print(f"Top {len(results)} Retrieved Authentic Paper Chunks:\n")

    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- [Result {i}] Title: {doc['title']} | ID: {doc['chunk_id']} | Score: {score:.4f} ---")
        snippet = doc["text"][:300]
        print(f"{snippet}...\n")


if __name__ == "__main__":
    main()
