"""Sanity check: does W ever generalize?

Test 1: Same-chunk L0→L5 (degenerate). Query = L0(first 100 tokens of
chunk i), stored = L5(first 100 tokens of chunk i). Identical inputs to
both forward passes. W's job is to learn the L0→L5 mapping for V22-
Dickens. Should achieve 100% if W is in any way functional.

Test 2: Overlapping-window paraphrase. Stored = L5(tokens[0:100]); query
= L0(tokens[50:150]). The query window OVERLAPS the stored window, so
they share content.

Test 3: Same as the original experiment for comparison.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import hidden_at_layer

PPD = REPO / "experiments/per_passage_dickens"
LS = REPO / "experiments/library_scaling"
N_TRAIN = 1000
N_QUERY = 100


def main():
    device = torch.device("cuda")
    print("Loading V22-Dickens base ...")
    model, _ = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    model.eval()

    print("Loading chunks ...")
    chunks = np.load(LS / "data/chunks.npy")
    print(f"  shape: {chunks.shape}")

    BATCH = 16

    @torch.no_grad()
    def forward_l5(token_slice):
        """Compute L5 means for a 2D token array (n, T)."""
        n = token_slice.shape[0]
        out = np.zeros((n, 1024), dtype=np.float32)
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            ids_t = torch.tensor(token_slice[s:e], dtype=torch.long, device=device)
            h5 = hidden_at_layer(model, ids_t, 5)
            out[s:e] = h5.mean(dim=1).detach().cpu().numpy()
        return out

    @torch.no_grad()
    def forward_l0(token_slice):
        out = np.zeros((token_slice.shape[0], 1024), dtype=np.float32)
        for s in range(0, token_slice.shape[0], BATCH):
            e = min(s + BATCH, token_slice.shape[0])
            ids_t = torch.tensor(token_slice[s:e], dtype=torch.long, device=device)
            emb = model.drop(model.tok_emb(ids_t))
            out[s:e] = emb.mean(dim=1).detach().cpu().numpy()
        return out

    def train_W_and_eval(L0_train, L5_train, L0_test, L5_test_lib,
                          test_truth, label):
        """Train W on (L0_train, L5_train) pairs (1 pair per target),
        evaluate top-1 routing of L0_test against L5_test_lib."""
        L0t = torch.tensor(L0_train, dtype=torch.float32, device=device)
        L5l = torch.tensor(L5_train, dtype=torch.float32, device=device)
        targets = torch.arange(L0t.shape[0], device=device)

        W = nn.Linear(1024, 1024, bias=False).to(device)
        nn.init.eye_(W.weight)
        opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.0,
                                  betas=(0.9, 0.95))
        keys_n = F.normalize(L5l, dim=-1)
        for step in range(500):
            proj_n = F.normalize(W(L0t), dim=-1)
            sim = (proj_n @ keys_n.T) / 0.05
            loss = F.cross_entropy(sim, targets)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            train_acc = (sim.argmax(dim=-1) == targets).float().mean().item()
            # Eval
            L0e = torch.tensor(L0_test, dtype=torch.float32, device=device)
            L5e_lib = torch.tensor(L5_test_lib, dtype=torch.float32, device=device)
            keys_e = F.normalize(L5e_lib, dim=-1)
            proj_e = F.normalize(W(L0e), dim=-1)
            sim_e = proj_e @ keys_e.T
            argmax = sim_e.argmax(dim=-1).cpu().numpy()
            top1 = (argmax == test_truth).mean()
        print(f"[{label}]  train_acc={train_acc:.3f}  test top-1={top1:.3f} "
              f"({(argmax == test_truth).sum()}/{len(test_truth)})")
        return top1

    HALF = 100
    # We need stored L5 and query L0 for chunks 0..N_TRAIN+N_QUERY-1.
    # For different paraphrase schemes:

    n_total = N_TRAIN + N_QUERY  # 1100

    # Forward pass once to get L5(first 100), L5(second 100),
    # L0(first 100), L0(second 100), L0(50..150) for n_total chunks.
    print(f"\nComputing L5 / L0 for {n_total} chunks across multiple windows ...")
    t0 = time.time()
    L5_first  = forward_l5(chunks[:n_total, :HALF])
    L5_second = forward_l5(chunks[:n_total, HALF:])
    L0_first  = forward_l0(chunks[:n_total, :HALF])
    L0_second = forward_l0(chunks[:n_total, HALF:])
    L0_overlap = forward_l0(chunks[:n_total, 50:150])
    print(f"  done in {time.time()-t0:.1f}s")

    # Test 1: same-chunk degenerate. query = L0(first), stored = L5(first).
    print("\n--- Test 1: same-chunk L0(first)→L5(first) [degenerate] ---")
    train_W_and_eval(
        L0_first[:N_TRAIN], L5_first[:N_TRAIN],
        L0_first[N_TRAIN:N_TRAIN+N_QUERY], L5_first[:N_TRAIN+N_QUERY],
        np.arange(N_TRAIN, N_TRAIN+N_QUERY),
        "same-chunk L0→L5",
    )

    # Test 2: overlapping-window paraphrase. query = L0(50..150), stored = L5(first 100).
    print("\n--- Test 2: overlapping-window query L0(50..150) → L5(first 100) ---")
    train_W_and_eval(
        L0_overlap[:N_TRAIN], L5_first[:N_TRAIN],
        L0_overlap[N_TRAIN:N_TRAIN+N_QUERY], L5_first[:N_TRAIN+N_QUERY],
        np.arange(N_TRAIN, N_TRAIN+N_QUERY),
        "overlapping window",
    )

    # Test 3: split-halves (the original setup)
    print("\n--- Test 3: split-halves (original) query L0(second) → L5(first) ---")
    train_W_and_eval(
        L0_second[:N_TRAIN], L5_first[:N_TRAIN],
        L0_second[N_TRAIN:N_TRAIN+N_QUERY], L5_first[:N_TRAIN+N_QUERY],
        np.arange(N_TRAIN, N_TRAIN+N_QUERY),
        "split-halves",
    )


if __name__ == "__main__":
    main()
