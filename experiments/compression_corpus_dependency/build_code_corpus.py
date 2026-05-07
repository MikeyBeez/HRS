"""Build a Python-code corpus from this repo's .py files.

Walks the repo, collects all .py files (excluding venv, __pycache__, third-party),
concatenates with file-boundary markers, and tokenizes with GPT-2 BPE.

Saves train/val splits to a .pt file with the same shape as the existing
WT103 cache: {"splits": {"train": <SplitObj>, "validation": <SplitObj>}}
where each SplitObj has a `.tokens` attribute (1D LongTensor).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/compression_corpus_dependency"

EXCLUDES = {".venv", "__pycache__", ".git"}


class SplitObj:
    """Mimics the schema of the existing WT103 cache split entries."""
    def __init__(self, tokens):
        self.tokens = tokens


def collect_py_files(root):
    paths = []
    for p in root.rglob("*.py"):
        # Check that no path segment WITHIN root is an excluded dir
        rel_parts = set(p.relative_to(root).parts)
        if rel_parts & EXCLUDES:
            continue
        if p.stat().st_size < 50:
            continue
        paths.append(p)
    return sorted(paths)


def main():
    paths = collect_py_files(REPO)
    print(f"Found {len(paths)} .py files")
    total_chars = sum(p.stat().st_size for p in paths)
    print(f"Total chars: {total_chars:,}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Concatenate with file boundary markers
    parts = []
    for p in paths:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        # Marker: a comment line that's stable across files
        rel = p.relative_to(REPO)
        marker = f"\n# === FILE: {rel} ===\n"
        parts.append(marker + txt)

    full_text = "\n".join(parts)
    print(f"Total text: {len(full_text):,} chars")

    print("Tokenizing ...")
    t0 = time.time()
    # Tokenize in chunks to avoid HF warning about long sequences
    chunk_size = 200_000
    all_ids = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        all_ids.extend(ids)
        if (i // chunk_size) % 10 == 0:
            print(f"  tokenized {i:,}/{len(full_text):,}  "
                  f"(tokens so far: {len(all_ids):,})")
    print(f"Total tokens: {len(all_ids):,}  wall: {time.time()-t0:.0f}s")

    tokens = torch.tensor(all_ids, dtype=torch.int64)

    # 90/10 train/val split
    n = tokens.shape[0]
    val_size = max(1024, n // 10)
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]
    print(f"Train: {train_tokens.shape}  val: {val_tokens.shape}")

    # Save with plain-dict schema (no custom class) for portability
    out_path = EXP / "data/code_corpus.pt"
    torch.save({
        "train": train_tokens,
        "validation": val_tokens,
        "n_files": len(paths),
        "total_chars": total_chars,
    }, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
