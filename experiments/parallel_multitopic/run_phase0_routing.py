"""Phase 0 routing diagnostic for multi-topic queries on Dickens-50.

For each of the 15 multi-topic queries (3 topics each), compute the
W-projection routing scores against all 50 library L5 keys and check
whether the 3 expected adapters appear in the top-k.

Decision (per spec):
  >=80% of queries with all expected adapters in top-k → proceed to Phase 1
  50-80% → proceed with caveats
  <50%  → stop, recommend query decomposition

Output:
  results/phase0_routing.csv         per-query top-k details
  results/phase0_summary.json        aggregates + decision
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
EXP = REPO / "experiments/parallel_multitopic"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import apply_lora


D = 1024


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0)


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    queries = json.loads((EXP / "queries.json").read_text())["queries"]
    print(f"Queries: {len(queries)}")

    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())

    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    library_l5_n = F.normalize(library_l5, dim=-1)

    proj_ck = torch.load(
        DICKENS / "results/projection_W.pt",
        map_location=device, weights_only=False,
    )
    W = nn.Linear(D, D, bias=False).to(device)
    W.load_state_dict(proj_ck["W_state"])
    W.eval()

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()

    rows = []
    type_stats = defaultdict(lambda: {"n": 0, "all_in_top3": 0,
                                        "all_in_top5": 0, "all_in_top10": 0})

    print(f"\n{'='*78}")
    print(f"  qid  type        all_in_top3  all_in_top5  ranks_of_expected")
    print(f"{'='*78}")

    for q in queries:
        ids = tokenizer.encode(q["text"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            l0 = l0_mean(model, ids_t)
            proj = W(l0.unsqueeze(0))
            proj_n = F.normalize(proj, dim=-1)
            sim = (proj_n @ library_l5_n.T).squeeze(0)   # (50,)
            sorted_idx = torch.argsort(sim, descending=True).tolist()

        expected = q["expected_adapters"]
        ranks = sorted([sorted_idx.index(aid) for aid in expected])
        max_rank = max(ranks)
        all_in_top3 = max_rank < 3
        all_in_top5 = max_rank < 5
        all_in_top10 = max_rank < 10

        top5 = sorted_idx[:5]
        top5_scores = [float(sim[i].item()) for i in top5]

        type_stats[q["type"]]["n"] += 1
        type_stats[q["type"]]["all_in_top3"] += int(all_in_top3)
        type_stats[q["type"]]["all_in_top5"] += int(all_in_top5)
        type_stats[q["type"]]["all_in_top10"] += int(all_in_top10)

        print(f"  {q['id']:3d}  {q['type']:<10s}  "
              f"{'yes' if all_in_top3 else 'no':>11s}  "
              f"{'yes' if all_in_top5 else 'no':>11s}  "
              f"{ranks}")
        rows.append({
            "qid": q["id"], "type": q["type"], "text": q["text"],
            "expected_adapters": expected, "ranks_of_expected": ranks,
            "max_rank_of_expected": max_rank,
            "all_in_top3": all_in_top3, "all_in_top5": all_in_top5,
            "all_in_top10": all_in_top10,
            "top5_adapter_ids": top5,
            "top5_scores": top5_scores,
        })

    n = len(rows)
    n_top3 = sum(r["all_in_top3"] for r in rows)
    n_top5 = sum(r["all_in_top5"] for r in rows)
    n_top10 = sum(r["all_in_top10"] for r in rows)

    print(f"\n{'='*78}")
    print(f"  All expected adapters in top-3: {n_top3}/{n} = {n_top3/n:.3f}")
    print(f"  All expected adapters in top-5: {n_top5}/{n} = {n_top5/n:.3f}")
    print(f"  All expected adapters in top-10: {n_top10}/{n} = {n_top10/n:.3f}")
    print(f"\n  Per query type (all-in-top3 / all-in-top5):")
    for ft, d in sorted(type_stats.items()):
        m = d["n"]
        print(f"    {ft:>10s} (n={m:2d}): top3={d['all_in_top3']/m:.3f}  "
              f"top5={d['all_in_top5']/m:.3f}  top10={d['all_in_top10']/m:.3f}")

    # Decision (use top-3 first since spec uses "topk where k = number of topics")
    frac_top3 = n_top3 / n
    if frac_top3 >= 0.80:
        decision = "PROCEED"
        rec = (f"Top-3 captures all expected adapters in {frac_top3:.1%} of "
               f"queries — exceeds the 80% threshold. Proceed to Phase 1.")
    elif frac_top3 >= 0.50:
        decision = "PROCEED_WITH_CAVEATS"
        rec = (f"Top-3 captures all expected adapters in {frac_top3:.1%} of "
               f"queries — between 50% and 80%. Proceed to Phase 1, but flag "
               f"that some queries will run with incomplete adapter sets. "
               f"Consider also testing k=5 (top-5 captured {n_top5/n:.1%}).")
    else:
        decision = "STOP_QUERY_DECOMPOSITION_NEEDED"
        rec = (f"Top-3 captures all expected adapters in only {frac_top3:.1%} "
               f"of queries — below the 50% threshold. Routing doesn't "
               f"surface multi-topic adapters reliably. The experiment as "
               f"specified should not run; query decomposition (parsing the "
               f"query into sub-queries and routing each separately) is the "
               f"alternative architecture and is out of scope per spec.")
    print(f"\n  DECISION: {decision}")
    print(f"  {rec}")

    with (EXP / "results/phase0_routing.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "text", "expected_adapters",
                     "ranks_of_expected", "max_rank_of_expected",
                     "all_in_top3", "all_in_top5", "all_in_top10",
                     "top5_adapter_ids", "top5_scores"])
        for r in rows:
            w.writerow([r["qid"], r["type"], r["text"][:120],
                         json.dumps(r["expected_adapters"]),
                         json.dumps(r["ranks_of_expected"]),
                         r["max_rank_of_expected"],
                         int(r["all_in_top3"]), int(r["all_in_top5"]),
                         int(r["all_in_top10"]),
                         json.dumps(r["top5_adapter_ids"]),
                         json.dumps([round(s, 4) for s in r["top5_scores"]])])

    summary = {
        "n_queries": n,
        "all_in_top3": n_top3 / n,
        "all_in_top5": n_top5 / n,
        "all_in_top10": n_top10 / n,
        "per_type": {ft: {"n": d["n"],
                            "all_in_top3": d["all_in_top3"] / d["n"],
                            "all_in_top5": d["all_in_top5"] / d["n"],
                            "all_in_top10": d["all_in_top10"] / d["n"]}
                       for ft, d in type_stats.items()},
        "decision": decision,
        "recommendation": rec,
        "details": rows,
    }
    (EXP / "results/phase0_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {EXP/'results/phase0_summary.json'}")


if __name__ == "__main__":
    main()
