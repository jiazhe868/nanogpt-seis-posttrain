"""Download the public corpora for continued pretraining into work/midtrain/raw/<name>/ (one directory per source).

    python -I scripts/midtrain/download.py finemath fineweb_edu cosmopedia:0-3

finemath     HuggingFaceTB/finemath, finemath-4plus (64 files, 18.4 GB)
fineweb_edu  HuggingFaceFW/fineweb-edu, sample/10BT (14 files, 28.5 GB); filtered to the top scores later
cosmopedia   HuggingFaceTB/smollm-corpus, cosmopedia-v2 (104 files, 122 GB); ':a-b' picks shards a..b
"""
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus/train-{:05d}-of-00064.parquet", 64),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/{:03d}_00000.parquet", 14),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/train-{:05d}-of-00104.parquet", 104),
}


def main():
    for arg in sys.argv[1:]:
        name, _, span = arg.partition(":")
        repo, pattern, n = SOURCES[name]
        a, b = map(int, span.split("-")) if span else (0, n - 1)
        files = [pattern.format(i) for i in range(a, b + 1)]
        out = ROOT / "work/midtrain/raw" / name
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        snapshot_download(repo, repo_type="dataset", allow_patterns=files, local_dir=out, max_workers=8)
        gb = sum(p.stat().st_size for p in out.rglob("*.parquet")) / 1e9
        print(f"{name}: {len(files)} files requested, {gb:.1f} GB on disk, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
