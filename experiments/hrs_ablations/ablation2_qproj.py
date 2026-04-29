"""Ablation 2: query-side learned projection vs no projection.

Spec baseline (per the ablation spec): both query and stored use the same
first-layer mean operation -- no learned projection.

We test:
  no_proj         : query=L0_mean, stored=L5_aggregate (Phase 47 actually USES W)
  no_proj_L0_only : query=L0_mean, stored=L0_aggregate (no cross-layer)
  with_W_500      : Phase 47 baseline (W trained 500 steps InfoNCE)
  with_W_5000     : longer-trained W

Routing accuracy is the metric.
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
def compute_query_engrams(model, tokenizer, queries, layer, pool_op, device):
    out = []
    for q in queries:
        ids_t = encode(tokenizer, q["probe"], device)
        v = pool(get_hidden(model, ids_t, layer), pool_op)
        out.append(v.cpu())
    return torch.stack(out)


@torch.no_grad()
def compute_lib_engrams(model, tokenizer, library, layer, pool_op, device):
    out = []
    for entry in library:
        per_para = []
        for p in entry["paraphrases_train"]:
            ids_t = encode(tokenizer, p, device)
            per_para.append(pool(get_hidden(model, ids_t, layer), pool_op).cpu())
        out.append(torch.stack(per_para).mean(dim=0))
    return torch.stack(out)


def routing_acc(query_vecs, lib_keys, queries, W=None):
    if W is not None:
        q = W(query_vecs)
    else:
        q = query_vecs
    q_n = F.normalize(q, dim=-1)
    keys_n = F.normalize(lib_keys, dim=-1)
    sims = q_n @ keys_n.T  # (Q, N)
    pred = sims.argmax(dim=-1).cpu().numpy()
    truth = np.array([q["adapter_id"] for q in queries])
    return float((pred == truth).mean()), float(sims.max(dim=-1).values.mean().item())


def train_W(L0_mat, lib_keys, adapter_ids, n_steps=500, lr=1e-3, temp=0.05,
             device=None):
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(lib_keys, dim=-1)
    for step in range(n_steps):
        proj_n = F.normalize(W(L0_mat), dim=-1)
        sim = (proj_n @ keys_n.T) / temp
        loss = F.cross_entropy(sim, adapter_ids)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return W


def main():
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    model, cfg = load_baseline_model(device)
    library, _, _ = load_library_and_adapters(device)
    queries = held_out_queries(library)

    # Compute baseline engrams
    print("Computing query engrams (L0_mean) ...")
    q_L0 = compute_query_engrams(model, tokenizer, queries, "L0", "mean", device).to(device)
    print("Computing library engrams at L0 and L5 (mean) ...")
    lib_L5 = compute_lib_engrams(model, tokenizer, library, "L5", "mean", device).to(device)
    lib_L0 = compute_lib_engrams(model, tokenizer, library, "L0", "mean", device).to(device)

    # Per-paraphrase L0 mat for projection training
    print("Computing per-paraphrase L0_mean for InfoNCE ...")
    L0_mat = []
    adapter_ids = []
    for entry in library:
        for p in entry["paraphrases_train"]:
            ids_t = encode(tokenizer, p, device)
            v = pool(get_hidden(model, ids_t, "L0"), "mean")
            L0_mat.append(v); adapter_ids.append(entry["id"])
    L0_mat = torch.stack(L0_mat).to(device)
    adapter_ids_t = torch.tensor(adapter_ids, device=device)

    results = []
    t0 = time.time()

    # Variant: no projection, query L0 vs stored L5 (mismatch -> should fail)
    rout, msim = routing_acc(q_L0, lib_L5, queries, W=None)
    print(f"[no_proj_L0vsL5]   routing={rout:.3f}  max_sim={msim:.3f}")
    results.append({"variant": "no_proj_L0vsL5", "routing_acc": rout, "max_sim_mean": msim})

    # Variant: no projection, query L0 vs stored L0 (same layer, same pool)
    rout, msim = routing_acc(q_L0, lib_L0, queries, W=None)
    print(f"[no_proj_L0vsL0]   routing={rout:.3f}  max_sim={msim:.3f}")
    results.append({"variant": "no_proj_L0vsL0", "routing_acc": rout, "max_sim_mean": msim})

    # Variant: with W trained 500 steps (Phase 47 baseline)
    print("Training W (500 steps) ...")
    W500 = train_W(L0_mat, lib_L5, adapter_ids_t, n_steps=500, device=device)
    rout, msim = routing_acc(q_L0, lib_L5, queries, W=W500)
    print(f"[with_W_500]       routing={rout:.3f}  max_sim={msim:.3f}")
    results.append({"variant": "with_W_500", "routing_acc": rout, "max_sim_mean": msim})

    # Variant: with W trained 5000 steps
    print("Training W (5000 steps) ...")
    W5k = train_W(L0_mat, lib_L5, adapter_ids_t, n_steps=5000, device=device)
    rout, msim = routing_acc(q_L0, lib_L5, queries, W=W5k)
    print(f"[with_W_5000]      routing={rout:.3f}  max_sim={msim:.3f}")
    results.append({"variant": "with_W_5000", "routing_acc": rout, "max_sim_mean": msim})

    out_path = REPO / "experiments/hrs_ablations/results/ablation2_qproj.json"
    out_path.write_text(json.dumps({"results": results,
                                     "wall_s": time.time()-t0},
                                    indent=2))
    print(f"\nAblation 2 wall: {time.time()-t0:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
