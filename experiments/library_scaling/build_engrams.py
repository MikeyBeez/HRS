"""Build 100k stored L5 engrams + 1100 query L0 engrams from WikiText-103.

Pipeline:
  1. Load WT-103 train, tokenize the concatenation, slice into 100k
     non-overlapping 200-token chunks.
  2. For each chunk: forward V22-Dickens base on first 100 tokens →
     L5 mean (the "stored" engram).
  3. For 1100 chunks (W training + queries): also forward on second
     100 tokens → L0 mean (the "paraphrase-like query").
  4. Save engrams to disk.

Batching: forward in batches of 16 to amortize launch overhead. V22 is
small enough that this should fit easily on a 16GB GPU.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, hidden_at_layer,
)

PPD = REPO / "experiments/per_passage_dickens"
LS = REPO / "experiments/library_scaling"

CHUNK_TOK = 200          # tokens per chunk
HALF = CHUNK_TOK // 2    # 100 stored, 100 query
N_CHUNKS = 100_000
N_QUERY_AND_W = 1100     # 1000 for W training + 100 for held-out queries
BATCH = 16


@torch.no_grad()
def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ---- Step 1: load WT-103 + tokenize + chunk ----
    print("Loading WikiText-103 train ...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    print(f"  rows: {len(ds)}")

    # Concatenate text rows. Many are empty or pure whitespace; skip those.
    print("  tokenizing rows until we have enough chunks ...")
    all_ids = []
    needed_tokens = N_CHUNKS * CHUNK_TOK + 1000  # buffer
    for row in ds:
        t = row["text"]
        if not t.strip():
            continue
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= needed_tokens:
            break
    print(f"  total tokens collected: {len(all_ids):,}")

    # Slice into N_CHUNKS non-overlapping windows.
    chunks = []
    for i in range(N_CHUNKS):
        start = i * CHUNK_TOK
        end = start + CHUNK_TOK
        if end > len(all_ids):
            print(f"  WARN: ran out of tokens at chunk {i}")
            break
        chunks.append(all_ids[start:end])
    chunks = np.array(chunks, dtype=np.int64)  # (N_CHUNKS, 200)
    print(f"  chunks shape: {chunks.shape}")

    # Save chunks to disk for reproducibility (small ~160MB)
    np.save(LS / "data/chunks.npy", chunks)
    print(f"  saved chunks to {LS / 'data/chunks.npy'}")

    # ---- Step 2: load V22-Dickens base ----
    print("\nLoading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    model.eval()
    # Note: no LoRA applied — base-model engrams (canonical Phase 47).

    # ---- Step 3: compute L5 stored engrams (first 100 tokens of each
    #             chunk), in batches.
    print(f"\nComputing L5 stored engrams for {len(chunks)} chunks "
          f"(first {HALF} tokens each) ...")
    stored_engrams = np.zeros((len(chunks), 1024), dtype=np.float32)
    t0 = time.time()
    for batch_start in range(0, len(chunks), BATCH):
        batch_end = min(batch_start + BATCH, len(chunks))
        ids_np = chunks[batch_start:batch_end, :HALF]
        ids_t = torch.tensor(ids_np, dtype=torch.long, device=device)
        # V22 forward signature: model(idx, step=0).
        # For L5 mean we use hidden_at_layer (which loops blocks 0..5).
        h5 = hidden_at_layer(model, ids_t, 5)  # (B, T, D)
        eng = h5.mean(dim=1)  # (B, D)
        stored_engrams[batch_start:batch_end] = eng.detach().cpu().numpy()
        if (batch_start // BATCH) % 200 == 0:
            elapsed = time.time() - t0
            done = batch_end
            eta = elapsed / max(1, done) * (len(chunks) - done)
            print(f"  [{done:6d}/{len(chunks)}]  elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s")
    np.save(LS / "data/stored_L5.npy", stored_engrams)
    print(f"  stored engrams shape: {stored_engrams.shape}  "
          f"saved to {LS / 'data/stored_L5.npy'}")
    print(f"  L5 wall: {time.time()-t0:.0f}s")

    # ---- Step 4: compute L0 query engrams for the first 1100 chunks
    #             (second 100 tokens). L0 = embedding mean, no transformer.
    print(f"\nComputing L0 query engrams for first {N_QUERY_AND_W} chunks "
          f"(second {HALF} tokens) ...")
    query_engrams = np.zeros((N_QUERY_AND_W, 1024), dtype=np.float32)
    t0 = time.time()
    for batch_start in range(0, N_QUERY_AND_W, BATCH):
        batch_end = min(batch_start + BATCH, N_QUERY_AND_W)
        ids_np = chunks[batch_start:batch_end, HALF:]
        ids_t = torch.tensor(ids_np, dtype=torch.long, device=device)
        # L0 mean is just the embedding mean (after dropout, training mode
        # would matter — we're in eval).
        emb = model.drop(model.tok_emb(ids_t))  # (B, T, D)
        eng = emb.mean(dim=1)  # (B, D)
        query_engrams[batch_start:batch_end] = eng.detach().cpu().numpy()
    np.save(LS / "data/query_L0.npy", query_engrams)
    print(f"  query engrams shape: {query_engrams.shape}  "
          f"saved to {LS / 'data/query_L0.npy'}")
    print(f"  L0 wall: {time.time()-t0:.0f}s")

    print(f"\nDone. Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
