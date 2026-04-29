"""Ablation 1: engram pooling operation variants.

Baseline: query=L0_mean, stored=L5_aggregate (mean over training paras of
L5_mean). Adapters are reused unchanged from per_passage_dickens.

For each variant we (a) recompute stored library keys per (layer, pool),
(b) recompute query engrams per held-out paraphrase, (c) train a fresh
projection W (1024×1024) via InfoNCE for query→stored mapping, (d) measure
routing accuracy on held-out (3 paraphrases × 50 adapters = 150 queries).

Variants:
  baseline      query=L0  stored=L5  pool=mean
  last_only     query=L5  stored=L5  pool=mean   (no cross-layer mapping)
  first_only    query=L0  stored=L0  pool=mean   (no cross-layer mapping)
  midstack      query=L2  stored=L2  pool=mean   (mid-layer same-side)
  max_pool      query=L0  stored=L5  pool=max    (max instead of mean)
  attn_pool     query=L0  stored=L5  pool=attn   (learned-query attention)

Also runs full retrieval eval on baseline + best variant for sanity.
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
from experiments.identity_ae.lora_wrapper import load_lora_state_dict
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.hrs_ablations.util import generate, check_match, GEN_TOKENS


VARIANTS = [
    {"name": "baseline_L0mean_L5mean", "q_layer": "L0", "s_layer": "L5", "pool_op": "mean"},
    {"name": "last_only_L5mean_L5mean", "q_layer": "L5", "s_layer": "L5", "pool_op": "mean"},
    {"name": "first_only_L0mean_L0mean", "q_layer": "L0", "s_layer": "L0", "pool_op": "mean"},
    {"name": "midstack_L2mean_L2mean", "q_layer": "L2", "s_layer": "L2", "pool_op": "mean"},
    {"name": "max_pool_L0_L5", "q_layer": "L0", "s_layer": "L5", "pool_op": "max"},
    {"name": "attn_pool_L0_L5", "q_layer": "L0", "s_layer": "L5", "pool_op": "attn"},
]


def encode(tokenizer, text, device, ctx=512):
    ids = tokenizer.encode(text, add_special_tokens=False)[:ctx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def compute_engrams_for_library(model, tokenizer, library, layer, pool_op,
                                 attn_query, device):
    """For each library entry, compute (a) per-paraphrase engrams over training
    paraphrases (used for InfoNCE training signal), (b) the aggregate stored
    key (mean of training-paraphrase engrams)."""
    out = []
    for entry in library:
        per_para = []
        for p in entry["paraphrases_train"]:
            ids_t = encode(tokenizer, p, device)
            h = get_hidden(model, ids_t, layer)
            v = pool(h, pool_op, attn_query=attn_query).detach().cpu()
            per_para.append(v)
        agg = torch.stack(per_para).mean(dim=0)
        out.append({"id": entry["id"], "per_para": per_para, "aggregate": agg})
    return out


def train_projection(L0_mat, L5_mat, adapter_ids, lib_l5, n_steps=500,
                      lr=1e-3, temp=0.05, device=None):
    """L0_mat: (4N, D) per-paraphrase query engrams.
    L5_mat: (4N, D) per-paraphrase stored engrams (unused if same layer).
    lib_l5: (N, D) stored aggregates.
    adapter_ids: (4N,) target indices.
    """
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(lib_l5, dim=-1)
    for step in range(n_steps):
        proj = W(L0_mat)
        proj_n = F.normalize(proj, dim=-1)
        sim = (proj_n @ keys_n.T) / temp
        loss = F.cross_entropy(sim, adapter_ids)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        proj_n = F.normalize(W(L0_mat), dim=-1)
        train_acc = ((proj_n @ keys_n.T).argmax(dim=-1) == adapter_ids).float().mean().item()
    return W, train_acc


def routing_eval(W, model, tokenizer, library, lib_keys_n, queries, layer,
                  pool_op, attn_query, device):
    """Returns (n_correct, max_sim_stats)."""
    n = 0; n_correct = 0; sims = []
    for q in queries:
        ids_t = encode(tokenizer, q["probe"], device)
        with torch.no_grad():
            v = pool(get_hidden(model, ids_t, layer), pool_op,
                     attn_query=attn_query)
            proj = W(v.unsqueeze(0))
            proj_n = F.normalize(proj, dim=-1)
            sim = (proj_n @ lib_keys_n.T).squeeze(0)
            top = sim.argmax().item()
            sims.append(sim.max().item())
        n += 1
        if top == q["adapter_id"]: n_correct += 1
    return {
        "routing_acc": n_correct / n,
        "n_correct": n_correct, "n": n,
        "max_sim_mean": float(np.mean(sims)),
        "max_sim_min": float(np.min(sims)),
    }


@torch.no_grad()
def retrieval_eval(W, model, tokenizer, library, lib_keys_n, adapter_sds,
                    queries, layer, pool_op, attn_query, device, seeds=(0,1,2)):
    """Full eval: route, load adapter, generate, substring-check answer."""
    n = 0; n_routing = 0; n_retrieval = 0
    for seed in seeds:
        for qi, q in enumerate(queries):
            ids_t = encode(tokenizer, q["probe"], device)
            v = pool(get_hidden(model, ids_t, layer), pool_op,
                     attn_query=attn_query)
            proj = W(v.unsqueeze(0))
            proj_n = F.normalize(proj, dim=-1)
            sim = (proj_n @ lib_keys_n.T).squeeze(0)
            routed = sim.argmax().item()
            load_lora_state_dict(model, adapter_sds[routed])
            torch.manual_seed(seed * 10000 + qi)
            ids2 = encode(tokenizer, q["probe"], device)
            gen = generate(model, ids2, GEN_TOKENS, gen_seed=seed*10000+qi)
            full = tokenizer.decode(gen[0], skip_special_tokens=True)
            cont = full[len(q["probe"]):]
            n += 1
            if routed == q["adapter_id"]: n_routing += 1
            if check_match(q["answer"], cont): n_retrieval += 1
            reset_lora_to_zero(model)
    return {"routing_acc": n_routing/n, "retrieval_acc": n_retrieval/n,
            "n": n}


def main():
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    model, cfg = load_baseline_model(device)
    library, _, adapter_sds = load_library_and_adapters(device)
    queries = held_out_queries(library)

    # Optional: full retrieval on baseline + best routing variant
    do_retrieval = True

    results = []
    t_total = time.time()
    for v in VARIANTS:
        t0 = time.time()
        # Optional learned attention query
        attn_query = None
        if v["pool_op"] == "attn":
            attn_query = nn.Parameter(torch.randn(D, device=device) * 0.02)

        # Compute training-paraphrase engrams (query side and stored side)
        q_lib = compute_engrams_for_library(model, tokenizer, library,
                                             v["q_layer"], v["pool_op"],
                                             attn_query, device)
        if v["s_layer"] == v["q_layer"]:
            s_lib = q_lib  # same operation, reuse
        else:
            s_lib = compute_engrams_for_library(model, tokenizer, library,
                                                 v["s_layer"], v["pool_op"],
                                                 attn_query, device)

        # Per-paraphrase mats and library aggregates
        L0_mat = []
        adapter_ids = []
        for entry in q_lib:
            for vec in entry["per_para"]:
                L0_mat.append(vec); adapter_ids.append(entry["id"])
        L0_mat = torch.stack(L0_mat).to(device)
        adapter_ids_t = torch.tensor(adapter_ids, device=device)
        lib_keys = torch.stack([e["aggregate"] for e in s_lib]).to(device)

        # Train projection (fresh) via InfoNCE
        W, train_acc = train_projection(L0_mat, None, adapter_ids_t, lib_keys,
                                          n_steps=500, device=device)

        # Routing eval
        lib_keys_n = F.normalize(lib_keys, dim=-1)
        rout = routing_eval(W, model, tokenizer, library, lib_keys_n, queries,
                             v["q_layer"], v["pool_op"], attn_query, device)

        rec = {
            "variant": v["name"],
            "q_layer": v["q_layer"], "s_layer": v["s_layer"],
            "pool_op": v["pool_op"],
            "proj_train_acc": train_acc,
            "routing_acc": rout["routing_acc"],
            "max_sim_mean": rout["max_sim_mean"],
            "max_sim_min": rout["max_sim_min"],
            "wall_s": time.time() - t0,
        }
        if do_retrieval and v["name"] in ("baseline_L0mean_L5mean", ):
            ret = retrieval_eval(W, model, tokenizer, library, lib_keys_n,
                                  adapter_sds, queries, v["q_layer"],
                                  v["pool_op"], attn_query, device)
            rec["retrieval_acc"] = ret["retrieval_acc"]
            rec["wall_s"] = time.time() - t0
        results.append(rec)
        print(f"[{v['name']}] proj_train_acc={train_acc:.3f}  "
              f"routing_acc={rout['routing_acc']:.3f}  "
              f"max_sim={rout['max_sim_mean']:.3f}  "
              + (f"retrieval={rec.get('retrieval_acc'):.3f}  "
                 if 'retrieval_acc' in rec else "") +
              f"wall={rec['wall_s']:.1f}s")

    # Now also run retrieval on the variant with the best routing
    best = max([r for r in results if "retrieval_acc" not in r],
                 key=lambda r: r["routing_acc"], default=None)
    if best is not None and do_retrieval:
        print(f"\n[Re-running retrieval on best non-baseline: {best['variant']}]")
        v = next(x for x in VARIANTS if x["name"] == best["variant"])
        attn_query = None
        if v["pool_op"] == "attn":
            attn_query = nn.Parameter(torch.randn(D, device=device) * 0.02)
        q_lib = compute_engrams_for_library(model, tokenizer, library,
                                             v["q_layer"], v["pool_op"],
                                             attn_query, device)
        if v["s_layer"] == v["q_layer"]:
            s_lib = q_lib
        else:
            s_lib = compute_engrams_for_library(model, tokenizer, library,
                                                 v["s_layer"], v["pool_op"],
                                                 attn_query, device)
        L0_mat = torch.stack([vec for entry in q_lib for vec in entry["per_para"]]).to(device)
        adapter_ids_t = torch.tensor(
            [entry["id"] for entry in q_lib for _ in entry["per_para"]],
            device=device)
        lib_keys = torch.stack([e["aggregate"] for e in s_lib]).to(device)
        W, _ = train_projection(L0_mat, None, adapter_ids_t, lib_keys,
                                  n_steps=500, device=device)
        lib_keys_n = F.normalize(lib_keys, dim=-1)
        ret = retrieval_eval(W, model, tokenizer, library, lib_keys_n,
                              adapter_sds, queries, v["q_layer"],
                              v["pool_op"], attn_query, device)
        for r in results:
            if r["variant"] == v["name"]:
                r["retrieval_acc"] = ret["retrieval_acc"]
                print(f"  retrieval={ret['retrieval_acc']:.3f}")

    out_path = REPO / "experiments/hrs_ablations/results/ablation1_pooling.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results,
                                     "wall_total_s": time.time()-t_total},
                                    indent=2))
    print(f"\nAblation 1 wall: {time.time()-t_total:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
