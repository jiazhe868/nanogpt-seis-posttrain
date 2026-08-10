#!/usr/bin/env python3
"""ChatML dataset, prompt-loss masking, and sequence packing for SFT.

The one line that makes this instruction tuning rather than more pretraining is
the mask: system and user tokens get label -100, so cross-entropy is computed
only on assistant tokens. Packing concatenates short samples up to max_seq_len
but never splits one across a block boundary.

    python -m src.sft.dataset      # self-check on the current splits
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# Default constants
DEFAULT_SYSTEM_PROMPT = "You are SeisGPT, an expert AI assistant specializing in seismology and earthquake science."
IM_START_ID = 16384  # <|im_start|> token ID
IM_END_ID = 16385    # <|im_end|> token ID
IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignore index for masked prompt tokens


def load_bpe_tokenizer(tokenizer_path: Union[str, Path]) -> Tokenizer:
    """
    Loads the byte-level BPE tokenizer from disk.

    Args:
        tokenizer_path: Path to tokenizer.json file.

    Returns:
        Loaded tokenizers.Tokenizer instance.
    """
    path_obj = Path(tokenizer_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")
    return Tokenizer.from_file(str(path_obj))


def encode_chatml_sample(
    tokenizer: Tokenizer,
    instruction: str,
    response: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    im_start_id: int = IM_START_ID,
    im_end_id: int = IM_END_ID,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[List[int], List[int]]:
    """
    Encodes a single instruction-response pair into ChatML format with labels.

    Format:
      <|im_start|>system\n{system_prompt}<|im_end|>\n
      <|im_start|>user\n{instruction}<|im_end|>\n
      <|im_start|>assistant\n{response}<|im_end|>\n

    Labels:
      All tokens up to '<|im_start|>assistant\n' are set to `ignore_index` (-100).
      Tokens inside the response and trailing '<|im_end|>\n' keep their original token IDs.

    Args:
        tokenizer: BPE tokenizer instance.
        instruction: User query / prompt string.
        response: Assistant response string (may contain <think> tags).
        system_prompt: System persona description.
        im_start_id: Token ID for <|im_start|>.
        im_end_id: Token ID for <|im_end|>.
        ignore_index: Label value for tokens that should not contribute to loss (-100).

    Returns:
        Tuple of (input_ids, label_ids).
    """
    # Pre-tokenize role header strings
    system_header_ids = tokenizer.encode("system\n", add_special_tokens=False).ids
    user_header_ids = tokenizer.encode("user\n", add_special_tokens=False).ids
    assistant_header_ids = tokenizer.encode("assistant\n", add_special_tokens=False).ids
    newline_ids = tokenizer.encode("\n", add_special_tokens=False).ids

    # 1. System Turn (Masked out from loss)
    system_body_ids = tokenizer.encode(system_prompt.strip() + "\n", add_special_tokens=False).ids
    system_tokens = [im_start_id] + system_header_ids + system_body_ids + [im_end_id] + newline_ids
    system_labels = [ignore_index] * len(system_tokens)

    # 2. User Turn (Masked out from loss)
    user_body_ids = tokenizer.encode(instruction.strip() + "\n", add_special_tokens=False).ids
    user_tokens = [im_start_id] + user_header_ids + user_body_ids + [im_end_id] + newline_ids
    user_labels = [ignore_index] * len(user_tokens)

    # 3. Assistant Header Turn (Masked out from loss)
    assistant_header_tokens = [im_start_id] + assistant_header_ids
    assistant_header_labels = [ignore_index] * len(assistant_header_tokens)

    # 4. Assistant Response Target (Loss computed here)
    response_body_ids = tokenizer.encode(response.strip(), add_special_tokens=False).ids
    response_target_tokens = response_body_ids + [im_end_id] + newline_ids
    response_target_labels = list(response_target_tokens)

    # Concatenate all sections
    input_ids = system_tokens + user_tokens + assistant_header_tokens + response_target_tokens
    label_ids = system_labels + user_labels + assistant_header_labels + response_target_labels

    return input_ids, label_ids


class SFTDataset(Dataset):
    """
    PyTorch Dataset for Supervised Fine-Tuning of nanoGPT-Seis.

    Supports sequence packing to concatenate multiple short ChatML samples
    into fixed block size (e.g. 4096 tokens) for maximum GPU compute efficiency.
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: Tokenizer,
        max_seq_len: int = 4096,
        pack_sequences: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        im_start_id: int = IM_START_ID,
        im_end_id: int = IM_END_ID,
        ignore_index: int = IGNORE_INDEX,
    ):
        """
        Initializes the SFT dataset.

        Args:
            jsonl_path: Path to formatted JSONL file (containing instruction & response).
            tokenizer: BPE tokenizer instance.
            max_seq_len: Maximum sequence length / block size (e.g. 4096).
            pack_sequences: If True, packs multiple Q&A pairs into blocks of max_seq_len.
            system_prompt: System persona string.
            im_start_id: Special token ID for <|im_start|>.
            im_end_id: Special token ID for <|im_end|>.
            ignore_index: Label value for prompt masking (-100).
        """
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pack_sequences = pack_sequences
        self.system_prompt = system_prompt
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.ignore_index = ignore_index

        self.samples: List[Dict[str, torch.Tensor]] = []
        self._load_and_process_data()

    def _load_and_process_data(self) -> None:
        """Reads JSONL file, encodes pairs into ChatML, and packs or truncates sequences."""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found at: {self.jsonl_path}")

        raw_samples: List[Tuple[List[int], List[int]]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    instruction = record.get("instruction", "").strip()
                    response = record.get("response", "").strip()
                    if instruction and response:
                        input_ids, label_ids = encode_chatml_sample(
                            tokenizer=self.tokenizer,
                            instruction=instruction,
                            response=response,
                            system_prompt=self.system_prompt,
                            im_start_id=self.im_start_id,
                            im_end_id=self.im_end_id,
                            ignore_index=self.ignore_index,
                        )
                        raw_samples.append((input_ids, label_ids))
                except Exception:
                    continue

        if self.pack_sequences:
            self._pack_sequence_buffers(raw_samples)
        else:
            self._process_unpacked_samples(raw_samples)

    def _pack_sequence_buffers(self, raw_samples: List[Tuple[List[int], List[int]]]) -> None:
        """Packs whole samples into fixed max_seq_len blocks.

        A sample is never split across two blocks: a response whose prompt sits
        in the previous block would be trained against unrelated context. When
        the next sample does not fit, the current block is padded and flushed.
        Samples longer than max_seq_len on their own are truncated.
        """
        buffer_inputs: List[int] = []
        buffer_labels: List[int] = []

        def flush() -> None:
            nonlocal buffer_inputs, buffer_labels
            if not buffer_inputs:
                return
            pad_len = self.max_seq_len - len(buffer_inputs)
            self.samples.append({
                "input_ids": torch.tensor(buffer_inputs + [self.im_end_id] * pad_len, dtype=torch.long),
                "labels": torch.tensor(buffer_labels + [self.ignore_index] * pad_len, dtype=torch.long),
            })
            buffer_inputs = []
            buffer_labels = []

        for input_ids, label_ids in raw_samples:
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                label_ids = label_ids[: self.max_seq_len]
            if len(buffer_inputs) + len(input_ids) > self.max_seq_len:
                flush()
            buffer_inputs.extend(input_ids)
            buffer_labels.extend(label_ids)

        flush()

    def _process_unpacked_samples(self, raw_samples: List[Tuple[List[int], List[int]]]) -> None:
        """Processes samples individually without packing, padding or truncating to max_seq_len."""
        for input_ids, label_ids in raw_samples:
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                label_ids = label_ids[: self.max_seq_len]
            else:
                pad_len = self.max_seq_len - len(input_ids)
                input_ids = input_ids + [self.im_end_id] * pad_len
                label_ids = label_ids + [self.ignore_index] * pad_len

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


class SFTDataCollator:
    """
    Data collator for batching SFT samples.
    Pads dynamically if sequences in batch vary in length.
    """

    def __init__(self, pad_token_id: int = IM_END_ID, ignore_index: int = IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_inputs = [sample["input_ids"] for sample in batch]
        batch_labels = [sample["labels"] for sample in batch]

        max_len = max(len(seq) for seq in batch_inputs)

        padded_inputs = []
        padded_labels = []

        for seq_in, seq_lbl in zip(batch_inputs, batch_labels):
            pad_len = max_len - len(seq_in)
            if pad_len > 0:
                pad_in = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                pad_lbl = torch.full((pad_len,), self.ignore_index, dtype=torch.long)
                padded_inputs.append(torch.cat([seq_in, pad_in]))
                padded_labels.append(torch.cat([seq_lbl, pad_lbl]))
            else:
                padded_inputs.append(seq_in)
                padded_labels.append(seq_lbl)

        return {
            "input_ids": torch.stack(padded_inputs, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
        }


def test_dataset_loading():
    """Self-test function to verify tokenizer loading, ChatML encoding, and dataset packing."""
    repo_root = Path(__file__).resolve().parents[2]
    tokenizer_path = repo_root.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json"
    train_path = repo_root / "data" / "splits" / "train.jsonl"

    print(f"[TEST] Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_bpe_tokenizer(tokenizer_path)

    print(f"[TEST] Testing single ChatML sample encoding...")
    test_inst = "Calculate the seismic moment for a fault with rupture area A = 100 km^2 and average slip d = 1.5 m."
    test_resp = "<think>\nSeismic moment formula: M_0 = mu * A * d.\nAssuming shear modulus mu = 32 GPa:\nM_0 = 32e9 * (100 * 1e6) * 1.5 = 4.8e18 N*m.\n</think>\nThe calculated seismic moment M_0 is 4.80 x 10^18 N*m."

    input_ids, label_ids = encode_chatml_sample(tokenizer, test_inst, test_resp)
    print(f"  Total tokens: {len(input_ids)}")
    masked_count = sum(1 for label in label_ids if label == IGNORE_INDEX)
    target_count = len(label_ids) - masked_count
    print(f"  Masked tokens (-100): {masked_count} | Loss target tokens: {target_count}")

    if train_path.exists():
        print(f"[TEST] Loading packed SFTDataset from: {train_path}")
        dataset = SFTDataset(
            jsonl_path=train_path,
            tokenizer=tokenizer,
            max_seq_len=4096,
            pack_sequences=True,
        )
        print(f"  Total packed sequences (4096 tokens each): {len(dataset)}")
        sample = dataset[0]
        print(f"  Sample 0 input_ids shape: {sample['input_ids'].shape}")
        print(f"  Sample 0 labels shape: {sample['labels'].shape}")
        print(f"[SUCCESS] Dataset module self-test passed cleanly!")
    else:
        print(f"[WARNING] Formatted train file not found at {train_path}. Skipping dataset load test.")


if __name__ == "__main__":
    test_dataset_loading()
