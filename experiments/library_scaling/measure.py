"""Measure routing accuracy, cost, separation, and top-K rank-of-correct
across library sizes {1k, 5k, 20k, 50k, 100k}.

Queries: chunks 1000..1099 (held-out — NOT used for W training).
For each query: project L0 through W → unit-normalize → cosine vs all
stored engrams (first N) → argmax. Correct if argmax == query's index.

Routing cost: wall time for the (W @ q) + cosine vs N + argmax operation,
averaged across 100 queries.

Separation: sample 10000 random pairs of stored engrams (without
replacement of within-pair), compute cosine, report distribution.

Top-K: for each query, where in the sorted-by-cosine list is the
correct answer?
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
LS = REPO / "experiments/library_scaling"

D = 1024
SIZES = [1000, 5000, 20000, 50000, 100000]
N_QUERY = 100  # query indices 1000..1099 (after the 1000 used for W training)


def main():
    device = torch.device("cuda")
    print("Loading data ...")
    stored = np.load(LS / "data/stored_L5.npy")
    queries = np.load(LS / "data/query_L0.npy")
    print(f"  stored {stored.shape}  queries {queries.shape}")

    # Held-out queries: indices 1000..1099 in the chunk array.
    query_indices = np.arange(1000, 1000 + N_QUERY)

    # Move stored to GPU as float32 — 100k * 1024 * 4 = 400MB
    stored_t = torch.tensor(stored, dtype=torch.float32, device=device)
    stored_n = F.normalize(stored_t, dim=-1)
    print(f"  stored on GPU: {stored_n.shape}")

    queries_t = torch.tensor(queries, dtype=torch.float32, device=device)
    Q = queries_t[query_indices]  # (100, 1024)

    # Load W
    W_ck = torch.load(LS / "data/W.pt", map_location=device, weights_only=False)
    W = nn.Linear(D, D, bias=False).to(device)
    W.load_state_dict(W_ck["W_state"])
    W.eval()

    # Project queries
    with torch.no_grad():
        Q_proj = W(Q)
        Q_n = F.normalize(Q_proj, dim=-1)

    rng = np.random.default_rng(0)

    results = []
    for N in SIZES:
        t_size = time.time()
        keys_n = stored_n[:N]  # (N, 1024)
        truth = torch.tensor(query_indices, dtype=torch.long, device=device)

        # Routing accuracy + top-K analysis
        with torch.no_grad():
            sim = Q_n @ keys_n.T  # (100, N)
            # argsort descending
            sorted_idx = sim.argsort(dim=-1, descending=True)
            # rank of correct
            ranks = []
            n_correct_top1 = 0
            n_correct_top5 = 0
            n_correct_top10 = 0
            for i in range(N_QUERY):
                if query_indices[i] >= N:
                    # Correct answer is OUTSIDE the library — skip / mark fail
                    ranks.append(-1)
                    continue
                rank = (sorted_idx[i] == truth[i]).nonzero()[0, 0].item()
                ranks.append(rank)
                if rank == 0: n_correct_top1 += 1
                if rank < 5: n_correct_top5 += 1
                if rank < 10: n_correct_top10 += 1

        valid_queries = sum(1 for r in ranks if r >= 0)
        routing_top1 = n_correct_top1 / max(1, valid_queries)
        routing_top5 = n_correct_top5 / max(1, valid_queries)
        routing_top10 = n_correct_top10 / max(1, valid_queries)

        # Routing cost: time per query
        t_cost = time.time()
        with torch.no_grad():
            for _ in range(10):  # repeat to amortize launch
                sim = Q_n @ keys_n.T
                _ = sim.argmax(dim=-1)
            torch.cuda.synchronize()
        cost_per_query_us = (time.time() - t_cost) / (10 * N_QUERY) * 1e6

        # Separation stats: 10000 random pairs of distinct engrams
        n_pairs = 10000
        n_unique = N
        i_idx = rng.integers(0, n_unique, size=n_pairs)
        j_idx = rng.integers(0, n_unique, size=n_pairs)
        # Avoid same-index pairs
        while True:
            mask = i_idx == j_idx
            if not mask.any():
                break
            j_idx[mask] = rng.integers(0, n_unique, size=mask.sum())
        with torch.no_grad():
            v_i = stored_n[i_idx]
            v_j = stored_n[j_idx]
            cos = (v_i * v_j).sum(dim=-1)
        sep_mean = float(cos.mean().item())
        sep_max  = float(cos.max().item())
        sep_p90  = float(cos.quantile(0.9).item())
        sep_min  = float(cos.min().item())

        rec = {
            "N": N,
            "n_valid_queries": valid_queries,
            "routing_top1": routing_top1,
            "routing_top5": routing_top5,
            "routing_top10": routing_top10,
            "cost_per_query_us": cost_per_query_us,
            "sep_mean": sep_mean,
            "sep_max":  sep_max,
            "sep_p90":  sep_p90,
            "sep_min":  sep_min,
            "ranks":    ranks,
            "wall_s":   time.time() - t_size,
        }
        results.append(rec)
        print(f"  N={N:6d}  top1={routing_top1:.3f}  top5={routing_top5:.3f}  "
              f"top10={routing_top10:.3f}  cost={cost_per_query_us:.0f}us  "
              f"sep mean={sep_mean:.3f} max={sep_max:.3f} p90={sep_p90:.3f}  "
              f"wall={time.time()-t_size:.1f}s")

    out_path = LS / "results/measure.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
