#!/usr/bin/env python3
"""Train one SFT run per corpus size, to measure the local data-scaling curve.

Subsets are composition-matched: the paper/fundamentals/earthquakes mixture is
held fixed at every size, so size is the only thing varying. An earlier version
let size and mixture move together and understated the value of data by 28%
(PITFALLS F1). Epochs are fixed, so the curve varies unique data AND total
update tokens together — see the scope note in README section 4.9.

    python -m src.sft.run_data_scaling_sweep
"""

import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "splits"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "scaling_sweep_v2"
DOCS_DIR = REPO_ROOT / "docs"

# None = the full training file (labeled with its true size at runtime). The
# V1 sweep hardcoded 15000 and labeled it "12.6k (Full)" while the actual
# train set held 16,425 pairs.
SUBSETS = [1000, 2000, 3000, 4000, None]

# NOTE: each subset trains for a fixed 3 epochs, so smaller subsets also get
# proportionally fewer optimizer steps — the curve conflates data volume with
# training compute. Treat it as "loss vs data at fixed epochs", not a pure
# data-scaling law.


def prepare_subsets():
    """Create subset JSONL train files for each dataset size."""
    train_full_path = DATA_DIR / "train.jsonl"
    if not train_full_path.exists():
        raise FileNotFoundError(f"Full train dataset not found at {train_full_path}")

    with open(train_full_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    os.makedirs(DATA_DIR / "subsets", exist_ok=True)

    # Resolve None -> true full size once, so labels and filenames are honest.
    resolved = [c if c is not None else len(all_lines) for c in SUBSETS]

    for count in resolved:
        subset_path = DATA_DIR / "subsets" / f"sft_train_{count}.jsonl"
        lines_to_write = all_lines[:count] if count < len(all_lines) else all_lines
        with open(subset_path, "w", encoding="utf-8") as out_f:
            out_f.writelines(lines_to_write)
        print(f"[INFO] Created subset {count} samples -> {subset_path}")
    return resolved


def run_sweep(subset_sizes):
    """Train models on each dataset subset and log metrics."""
    results = []
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    python_bin = sys.executable
    torchrun_bin = str(Path(python_bin).parent / "torchrun")

    val_path = str(DATA_DIR / "val.jsonl")
    ckpt_pretrained = str(REPO_ROOT.parent / "nanogpt_seis" / "checkpoints" / "ckpt.pt")
    tokenizer_path = str(REPO_ROOT.parent / "nanogpt_seis" / "data" / "tokenized" / "tokenizer.json")

    print("================================================================================")
    print(f"🚀 Starting SFT Data Scaling Sweep (fixed 3 epochs): {subset_sizes}")
    print("================================================================================")

    for count in subset_sizes:
        train_path = str(DATA_DIR / "subsets" / f"sft_train_{count}.jsonl")
        out_dir = str(CHECKPOINT_DIR / f"subset_{count}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[SWEEP] Training on subset size: {count} samples...")
        start_time = time.time()

        cmd = [
            torchrun_bin, "--standalone", "--nproc_per_node=2",
            "-m", "src.sft.train_sft",
            "--ckpt-path", ckpt_pretrained,
            "--tokenizer-path", tokenizer_path,
            "--train-path", train_path,
            "--val-path", val_path,
            "--out-dir", out_dir,
            "--epochs", "3",
            "--batch-size", "4",
            "--grad-accum", "4",
            "--learning-rate", "2e-4",
            "--max-seq-len", "4096",
        ]

        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time

        # Read best validation loss from sft_log.csv
        log_csv = os.path.join(out_dir, "sft_log.csv")
        best_val = float("inf")
        final_train_loss = float("inf")
        if os.path.exists(log_csv):
            with open(log_csv, "r", encoding="utf-8") as f:
                lines = [line.strip().split(",") for line in f if line.strip()]
                header = lines[0]
                for row in lines[1:]:
                    if len(row) >= 4:
                        tr_loss = float(row[2])
                        val_loss = float(row[3])
                        if val_loss < best_val:
                            best_val = val_loss
                        final_train_loss = tr_loss

        results.append({
            "samples": count,
            "best_val_loss": best_val,
            "final_train_loss": final_train_loss,
            "elapsed_sec": elapsed,
        })
        print(f"[SWEEP RESULT] {count} samples -> Best Val Loss: {best_val:.4f} | Train Loss: {final_train_loss:.4f} | Time: {elapsed:.1f}s")

    # Save JSON results
    json_path = DOCS_DIR / "scaling_sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot scaling curve figure
    plot_scaling_curve(results)
    return results


def plot_scaling_curve(results):
    """Plot and save scientific figure of data scaling curve."""
    samples = [r["samples"] for r in results]
    val_losses = [r["best_val_loss"] for r in results]
    train_losses = [r["final_train_loss"] for r in results]

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(samples, val_losses, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Validation Loss")
    ax.plot(samples, train_losses, "s--", color="#ff7f0e", linewidth=2.0, markersize=7, label="Train Loss (3 Epochs)")

    for x, y in zip(samples, val_losses):
        ax.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1f77b4",
        )

    ax.set_title("nanoGPT-Seis (113M) SFT Data Scaling Law: Dataset Size vs. Loss", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Cleaned SFT Samples", fontsize=11, labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, labelpad=8)
    ax.set_xticks(samples)
    labels = [f"{s/1000:.0f}k" for s in samples[:-1]] + [f"{samples[-1]/1000:.1f}k (Full)"]
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()

    fig_path = DOCS_DIR / "sft_data_scaling.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"\n[SUCCESS] Generated data scaling law plot -> {fig_path}")


if __name__ == "__main__":
    sizes = prepare_subsets()
    run_sweep(sizes)
