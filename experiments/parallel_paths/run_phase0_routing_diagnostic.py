"""Phase 0 — routing top-k diagnostic for Dickens-50.

Purpose: gate the parallel-paths k=3 experiment. If top-1 routing is already
near 100%, parallel-paths cannot help (no headroom). If top-1 is below 100%
but top-3 reliably contains the correct adapter, parallel-paths has a target
gap to recover.

For each of the 150 held-out probes, compute the W-projection routing scores
against all 50 library L5 keys, and record:
  - rank of the correct adapter in the score ordering
  - whether top-1, top-3, top-5 contain the correct adapter
  - per-fact-type breakdown

Output:
  results/phase0_routing_diagnostic.json    aggregates
  results/phase0_routing_topk.csv           per-probe ranks
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/parallel_paths"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero,
)
from experiments.identity_ae.lora_wrapper import apply_lora


D = 1024


@torch.no_grad()
def l0_mean(model, ids_t):
    """L0 = drop(tok_emb) mean. Same as evaluate.py."""
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0)


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())

    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    library_l5_n = F.normalize(library_l5, dim=-1)
    print(f"Library L5 keys: {library_l5.shape}")

    proj_ck = torch.load(
        DICKENS / "results/projection_W.pt",
        map_location=device, weights_only=False,
    )
    W = nn.Linear(D, D, bias=False).to(device)
    W.load_state_dict(proj_ck["W_state"])
    W.eval()
    print(f"Loaded W projection.")

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()
    print("Model loaded; LoRA at zero (base routing forward).")

    # Build held-out queries (same set as Phase 1 of k2_crossterms / Dickens-50 evaluate.py)
    held_out = []
    for entry in library:
        for q in entry["paraphrases_held_out"]:
            held_out.append({
                "adapter_id": entry["id"],
                "fact_type": entry["fact_type"],
                "probe": q,
                "answer": entry["answer"],
            })
    print(f"Held-out queries: {len(held_out)}")

    rows = []
    rank_buckets = defaultdict(int)
    type_topk = defaultdict(lambda: {"top1": 0, "top3": 0, "top5": 0, "n": 0,
                                       "ranks": []})
    for qi, q in enumerate(held_out):
        ids = tokenizer.encode(q["probe"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            l0 = l0_mean(model, ids_t)                  # (D,)
            proj = W(l0.unsqueeze(0))                    # (1, D)
            proj_n = F.normalize(proj, dim=-1)
            sim = (proj_n @ library_l5_n.T).squeeze(0)   # (50,)
            sorted_indices = torch.argsort(sim, descending=True).tolist()
        true_aid = q["adapter_id"]
        rank = sorted_indices.index(true_aid)            # 0 = top-1
        top1 = rank == 0
        top3 = rank < 3
        top5 = rank < 5
        rank_buckets[rank] += 1
        type_topk[q["fact_type"]]["n"] += 1
        type_topk[q["fact_type"]]["top1"] += int(top1)
        type_topk[q["fact_type"]]["top3"] += int(top3)
        type_topk[q["fact_type"]]["top5"] += int(top5)
        type_topk[q["fact_type"]]["ranks"].append(rank)
        rows.append({
            "qi": qi, "adapter_id": true_aid, "fact_type": q["fact_type"],
            "probe": q["probe"], "answer": q["answer"],
            "rank_of_correct": rank,
            "top1": top1, "top3": top3, "top5": top5,
            "top_score": float(sim[sorted_indices[0]].item()),
            "true_score": float(sim[true_aid].item()),
            "score_gap_top1_minus_true": float(
                sim[sorted_indices[0]].item() - sim[true_aid].item()),
        })

    n = len(rows)
    n_top1 = sum(r["top1"] for r in rows)
    n_top3 = sum(r["top3"] for r in rows)
    n_top5 = sum(r["top5"] for r in rows)

    print(f"\n{'='*72}\nPHASE 0 — routing top-k diagnostic\n{'='*72}")
    print(f"  Total probes:  {n}")
    print(f"  top-1 correct: {n_top1}/{n}  =  {n_top1/n:.3f}")
    print(f"  top-3 correct: {n_top3}/{n}  =  {n_top3/n:.3f}")
    print(f"  top-5 correct: {n_top5}/{n}  =  {n_top5/n:.3f}")
    print(f"\n  Rank distribution of correct adapter:")
    for r in sorted(rank_buckets):
        print(f"    rank {r:2d}: {rank_buckets[r]:3d} ({rank_buckets[r]/n:.3f})")
    print(f"\n  Per fact type (top-1 / top-3 / top-5):")
    for ft, d in sorted(type_topk.items()):
        m = d["n"]
        print(f"    {ft:>10s} (n={m:3d}): "
              f"top1={d['top1']/m:.3f}  top3={d['top3']/m:.3f}  top5={d['top5']/m:.3f}  "
              f"mean_rank={np.mean(d['ranks']):.2f}")

    headroom_top3 = (n_top3 - n_top1) / n
    print(f"\n  Headroom (top-3 contains correct but top-1 doesn't): "
          f"{n_top3 - n_top1}/{n} = {headroom_top3:.3f}")

    if n_top1 == n:
        decision = "KILL_NO_HEADROOM"
        rec = ("Top-1 routing is already 100% on these probes. "
               "Parallel-paths k=3 cannot improve retrieval — no headroom. "
               "Recommend not running Phase 1-3.")
    elif headroom_top3 < 0.02:
        decision = "MARGINAL_HEADROOM"
        rec = (f"Top-1 misses on {n - n_top1}/{n} probes; top-3 catches "
               f"{n_top3 - n_top1} of those. Total recoverable headroom is "
               f"{headroom_top3:.3f} — small. Parallel paths could only buy "
               f"a few percentage points at best.")
    else:
        decision = "PROCEED"
        rec = (f"Top-1 misses on {n - n_top1}/{n} probes; top-3 catches "
               f"{n_top3 - n_top1} of those. Recoverable headroom is "
               f"{headroom_top3:.3f}. Parallel-paths k=3 has a real target.")

    print(f"\n  DECISION: {decision}")
    print(f"  {rec}")

    with (EXP / "results/phase0_routing_topk.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qi", "adapter_id", "fact_type", "probe", "answer",
                     "rank_of_correct", "top1", "top3", "top5",
                     "top_score", "true_score", "score_gap_top1_minus_true"])
        for r in rows:
            w.writerow([r["qi"], r["adapter_id"], r["fact_type"], r["probe"][:120],
                         r["answer"], r["rank_of_correct"], int(r["top1"]),
                         int(r["top3"]), int(r["top5"]),
                         f"{r['top_score']:.4f}", f"{r['true_score']:.4f}",
                         f"{r['score_gap_top1_minus_true']:.4f}"])

    summary = {
        "n_probes": n,
        "top1_acc": n_top1 / n,
        "top3_acc": n_top3 / n,
        "top5_acc": n_top5 / n,
        "rank_distribution": dict(rank_buckets),
        "per_fact_type": {
            ft: {"n": d["n"], "top1": d["top1"]/d["n"], "top3": d["top3"]/d["n"],
                  "top5": d["top5"]/d["n"],
                  "mean_rank_of_correct": float(np.mean(d["ranks"])),
                  "max_rank_of_correct": int(np.max(d["ranks"]))}
            for ft, d in type_topk.items()
        },
        "headroom_top3_minus_top1": headroom_top3,
        "decision": decision,
        "recommendation": rec,
    }
    (EXP / "results/phase0_routing_diagnostic.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSaved {EXP/'results/phase0_routing_diagnostic.json'}")


if __name__ == "__main__":
    main()
