"""Run the full size sweep with three query schemes:

  Scheme A (degenerate, upper bound): query=L0(first 100), stored=L5(first 100).
    Same input to both forward modes; W learns the L0→L5 mapping.

  Scheme B (overlapping window): query=L0(tokens[50:150]), stored=L5(first 100).
    Query overlaps stored by 50 tokens. Realistic paraphrase-like.

  Scheme C (split halves, original): query=L0(tokens[100:200]), stored=L5(first 100).
    Disjoint halves. Hardest paraphrase test.

For each scheme:
  - Train W on chunks 0..999 of the corresponding (L0_query, L5_stored) pairs.
  - Evaluate at N in {1000, 5000, 20000, 50000, 100000}: route 100 held-out
    queries (chunks 1000..1099) and report top-1, top-5, top-10, ranks,
    routing cost, separation stats.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import hidden_at_layer

PPD = REPO / "experiments/per_passage_dickens"
LS = REPO / "experiments/library_scaling"

D = 1024
SIZES = [1000, 5000, 20000, 50000, 100000]
N_TRAIN = 1000
N_QUERY = 100
BATCH = 16
HALF = 100


@torch.no_grad()
def forward_layer(model, token_slice, layer):
    """If layer < 0, return embedding mean. Else hidden_at_layer mean."""
    n = token_slice.shape[0]
    out = np.zeros((n, D), dtype=np.float32)
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        ids_t = torch.tensor(token_slice[s:e], dtype=torch.long, device="cuda")
        if layer < 0:
            h = model.drop(model.tok_emb(ids_t))
        else:
            h = hidden_at_layer(model, ids_t, layer)
        out[s:e] = h.mean(dim=1).detach().cpu().numpy()
    return out


def train_W(L0_train, L5_train_lib, n_steps=500, lr=1e-3, temp=0.05,
             device="cuda"):
    L0t = torch.tensor(L0_train, dtype=torch.float32, device=device)
    L5l = torch.tensor(L5_train_lib, dtype=torch.float32, device=device)
    targets = torch.arange(L0t.shape[0], device=device)
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(L5l, dim=-1)
    for step in range(n_steps):
        proj_n = F.normalize(W(L0t), dim=-1)
        sim = (proj_n @ keys_n.T) / temp
        loss = F.cross_entropy(sim, targets)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        train_acc = (sim.argmax(dim=-1) == targets).float().mean().item()
    return W, train_acc


def measure_scheme(label, L0_query_full, L5_stored_full):
    """L0_query_full: (100k, D)  L5_stored_full: (100k, D)
    Train W on first 1000; test on 1000..1099 across SIZES."""
    device = torch.device("cuda")
    print(f"\n=== Scheme {label} ===")

    # Train W
    W, train_acc = train_W(L0_query_full[:N_TRAIN],
                             L5_stored_full[:N_TRAIN],
                             n_steps=500, device=device)
    print(f"  W train acc on 1000 pairs: {train_acc:.3f}")

    # Eval prep
    Q = torch.tensor(L0_query_full[N_TRAIN:N_TRAIN + N_QUERY],
                     dtype=torch.float32, device=device)
    truth = torch.arange(N_TRAIN, N_TRAIN + N_QUERY, device=device)

    with torch.no_grad():
        Q_proj = W(Q)
        Q_n = F.normalize(Q_proj, dim=-1)
    stored_t = torch.tensor(L5_stored_full, dtype=torch.float32, device=device)
    stored_n = F.normalize(stored_t, dim=-1)

    rng = np.random.default_rng(0)
    out_records = []
    for N in SIZES:
        keys_n = stored_n[:N]
        with torch.no_grad():
            sim = Q_n @ keys_n.T  # (100, N)
            sorted_idx = sim.argsort(dim=-1, descending=True)
        ranks = []
        n_top1 = n_top5 = n_top10 = 0
        for i in range(N_QUERY):
            true_id = truth[i].item()
            if true_id >= N:
                ranks.append(-1); continue
            r = (sorted_idx[i] == true_id).nonzero()[0, 0].item()
            ranks.append(r)
            if r == 0: n_top1 += 1
            if r < 5: n_top5 += 1
            if r < 10: n_top10 += 1
        valid = sum(1 for r in ranks if r >= 0)

        # Routing cost
        t_cost = time.time()
        with torch.no_grad():
            for _ in range(10):
                sim_b = Q_n @ keys_n.T
                _ = sim_b.argmax(dim=-1)
            torch.cuda.synchronize()
        cost_us = (time.time() - t_cost) / (10 * N_QUERY) * 1e6

        # Separation stats: 10000 random pairs
        n_pairs = 10000
        i_idx = rng.integers(0, N, size=n_pairs)
        j_idx = rng.integers(0, N, size=n_pairs)
        while True:
            mask = i_idx == j_idx
            if not mask.any(): break
            j_idx[mask] = rng.integers(0, N, size=int(mask.sum()))
        with torch.no_grad():
            v_i = stored_n[i_idx]; v_j = stored_n[j_idx]
            cos = (v_i * v_j).sum(dim=-1)
        sep = {
            "mean": float(cos.mean().item()),
            "max":  float(cos.max().item()),
            "p90":  float(cos.quantile(0.9).item()),
            "min":  float(cos.min().item()),
        }

        out_records.append({
            "N": N,
            "valid_queries": valid,
            "top1": n_top1 / max(1, valid),
            "top5": n_top5 / max(1, valid),
            "top10": n_top10 / max(1, valid),
            "ranks": ranks,
            "cost_us": cost_us,
            "sep": sep,
        })
        print(f"  N={N:6d} top1={n_top1/max(1,valid):.3f} "
              f"top5={n_top5/max(1,valid):.3f} top10={n_top10/max(1,valid):.3f} "
              f"cost={cost_us:.0f}us  sep mean={sep['mean']:.3f} "
              f"max={sep['max']:.3f}")
    return {"label": label, "train_acc_W": train_acc, "rows": out_records}


def main():
    device = torch.device("cuda")
    print("Loading V22-Dickens base + chunks ...")
    model, _ = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    model.eval()
    chunks = np.load(LS / "data/chunks.npy")
    print(f"  chunks: {chunks.shape}")

    # We already have stored_L5 from build_engrams.py — that's L5(first 100).
    # We need L0 for: full query schemes A/B/C.
    # Scheme A: query = L0(first 100)
    # Scheme B: query = L0(50..150) — overlapping
    # Scheme C: query = L0(100..200) — split halves (= the original query_L0.npy)

    print("\nComputing L0 (no-LoRA) for first 1100 chunks across 3 windows ...")
    t0 = time.time()
    L0_first    = forward_layer(model, chunks[:N_TRAIN+N_QUERY, :HALF], -1)
    L0_overlap  = forward_layer(model, chunks[:N_TRAIN+N_QUERY, 50:150], -1)
    L0_second   = np.load(LS / "data/query_L0.npy")[:N_TRAIN+N_QUERY]
    print(f"  done in {time.time()-t0:.1f}s")

    # Stored library (L5 of first 100 tokens) — already computed.
    L5_stored = np.load(LS / "data/stored_L5.npy")
    print(f"  L5_stored: {L5_stored.shape}")

    # Pad L0 query arrays to 100k (queries beyond 1100 aren't used for
    # eval, but the function expects same-shape arrays). Just zero-pad.
    def pad(L0, target_n=100_000):
        out = np.zeros((target_n, D), dtype=np.float32)
        out[:L0.shape[0]] = L0
        return out

    schemes = []
    schemes.append(measure_scheme(
        "A_same_chunk_L0first→L5first",
        pad(L0_first), L5_stored,
    ))
    schemes.append(measure_scheme(
        "B_overlapping_L0[50-150]→L5first",
        pad(L0_overlap), L5_stored,
    ))
    schemes.append(measure_scheme(
        "C_split_halves_L0second→L5first",
        pad(L0_second), L5_stored,
    ))

    out_path = LS / "results/measure_v2.json"
    out_path.write_text(json.dumps({"schemes": schemes,
                                     "sizes": SIZES}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
