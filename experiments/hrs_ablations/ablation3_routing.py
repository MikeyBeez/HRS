"""Ablation 3: routing mechanism variants.

Baseline: argmax cosine similarity over engrams.

Variants:
  cosine_argmax (baseline)
  euclidean_argmin
  dot_product_argmax (no normalization)
  topK_K2_oracle  : pick top 2; if correct is in top 2 -> count as correct
  topK_K3_oracle  : top 3
  topK_K5_oracle  : top 5
  learned_mlp     : 2-layer MLP, query engram -> softmax over adapter ids,
                    trained with cross-entropy on training paraphrases

Routing accuracy is the metric for all. Top-K reports the "in top-K" rate
which is the upper bound for any reranking scheme.
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

from experiments.hrs_ablations.util import (
    PPD, D, load_baseline_model, get_tokenizer, get_hidden, pool,
    load_library_and_adapters, held_out_queries,
)


def encode(tokenizer, text, device, ctx=512):
    ids = tokenizer.encode(text, add_special_tokens=False)[:ctx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def compute_engrams(model, tokenizer, items, layer, pool_op, device):
    out = []
    for it in items:
        ids_t = encode(tokenizer, it, device)
        v = pool(get_hidden(model, ids_t, layer), pool_op)
        out.append(v.cpu())
    return torch.stack(out)


def main():
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    model, cfg = load_baseline_model(device)
    library, _, _ = load_library_and_adapters(device)
    queries = held_out_queries(library)

    # Use canonical Phase 47 engrams: query=L0_mean, stored=L5_aggregate, with W
    print("Computing query engrams (L0_mean) for held-out paraphrases ...")
    q_L0 = compute_engrams(model, tokenizer, [q["probe"] for q in queries],
                            "L0", "mean", device).to(device)
    truth = torch.tensor([q["adapter_id"] for q in queries], device=device)

    print("Computing library L5 aggregates ...")
    lib_L5 = []
    for entry in library:
        per_para = []
        for p in entry["paraphrases_train"]:
            ids_t = encode(tokenizer, p, device)
            per_para.append(pool(get_hidden(model, ids_t, "L5"), "mean").cpu())
        lib_L5.append(torch.stack(per_para).mean(dim=0))
    lib_L5 = torch.stack(lib_L5).to(device)

    # Train projection W (500 steps), same as Phase 47
    print("Training projection W ...")
    L0_mat = []; aids = []
    for entry in library:
        for p in entry["paraphrases_train"]:
            ids_t = encode(tokenizer, p, device)
            L0_mat.append(pool(get_hidden(model, ids_t, "L0"), "mean"))
            aids.append(entry["id"])
    L0_mat = torch.stack(L0_mat).to(device)
    aids_t = torch.tensor(aids, device=device)

    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, betas=(0.9, 0.95),
                              weight_decay=0.0)
    keys_n = F.normalize(lib_L5, dim=-1)
    for step in range(500):
        proj_n = F.normalize(W(L0_mat), dim=-1)
        sim = (proj_n @ keys_n.T) / 0.05
        loss = F.cross_entropy(sim, aids_t)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print("  done")

    # Project query side
    with torch.no_grad():
        q_proj = W(q_L0)
        q_proj_n = F.normalize(q_proj, dim=-1)

    results = []
    t0 = time.time()

    # Variant: cosine argmax (Phase 47 baseline)
    sims_cos = q_proj_n @ keys_n.T
    pred = sims_cos.argmax(dim=-1)
    acc = (pred == truth).float().mean().item()
    results.append({"variant": "cosine_argmax", "routing_acc": acc})
    print(f"[cosine_argmax]      routing={acc:.3f}")

    # Variant: dot product (no normalization on either side)
    sims_dot = q_proj @ lib_L5.T
    pred = sims_dot.argmax(dim=-1)
    acc = (pred == truth).float().mean().item()
    results.append({"variant": "dot_product_argmax", "routing_acc": acc})
    print(f"[dot_product]        routing={acc:.3f}")

    # Variant: euclidean (argmin distance)
    dists = torch.cdist(q_proj, lib_L5)  # (Q, N)
    pred = dists.argmin(dim=-1)
    acc = (pred == truth).float().mean().item()
    results.append({"variant": "euclidean_argmin", "routing_acc": acc})
    print(f"[euclidean]          routing={acc:.3f}")

    # Variant: top-K (in-top-K rates)
    for K in (2, 3, 5):
        topk = sims_cos.topk(K, dim=-1).indices  # (Q, K)
        in_topk = (topk == truth.unsqueeze(-1)).any(dim=-1)
        acc = in_topk.float().mean().item()
        results.append({"variant": f"topK_K{K}", "routing_acc": acc})
        print(f"[topK_K{K}]            in-top-K={acc:.3f}")

    # Variant: learned MLP router
    print("Training learned MLP router (1024->512->50) ...")
    N = lib_L5.shape[0]
    mlp = nn.Sequential(
        nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, N),
    ).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(2000):
        logits = mlp(L0_mat)
        loss = F.cross_entropy(logits, aids_t)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        pred = mlp(q_L0).argmax(dim=-1)
        acc = (pred == truth).float().mean().item()
    results.append({"variant": "learned_mlp", "routing_acc": acc,
                    "mlp_train_loss_final": float(loss.item())})
    print(f"[learned_mlp]        routing={acc:.3f}  train_loss_final={loss.item():.3f}")

    # Variant: cosine without W (sanity baseline)
    q_n_noproj = F.normalize(q_L0, dim=-1)
    sims_noproj = q_n_noproj @ keys_n.T
    pred = sims_noproj.argmax(dim=-1)
    acc = (pred == truth).float().mean().item()
    results.append({"variant": "cosine_noW (L0 vs L5)", "routing_acc": acc})
    print(f"[cosine_noW (L0vsL5)] routing={acc:.3f}")

    out_path = REPO / "experiments/hrs_ablations/results/ablation3_routing.json"
    out_path.write_text(json.dumps({"results": results,
                                     "wall_s": time.time()-t0},
                                    indent=2))
    print(f"\nAblation 3 wall: {time.time()-t0:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
