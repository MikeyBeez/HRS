"""Phase 42b: Clause-split routing for compositional queries, with L0 engram.

Phase 42 used L5 nonstop_mean as the engram for clause-split routing on
compositional queries and got 4/5 BOTH retrieval with 10/10 routing
correctness, matching the oracle pair-selection upper bound. The retrieval
ceiling at 4/5 is set by the K=2 capacity ceiling (Phase 43 confirmed K=4
collapses), not by routing — routing was already saturated at 10/10.

Phase 44 found that L0 mean is a strictly better routing key for
Application 1: 90 percent held-out vs 75 percent for L5 nonstop_mean.
This script asks whether the same improvement carries to the clause-
split routing path used in Application 3. The headline expectation is
that routing correctness will stay at 10/10 (it was already perfect)
and BOTH retrieval may stay at 4/5 (capacity-limited, not routing-
limited), but we want to confirm directly and we want to see whether
the routing margin is wider at L0 (which would matter for harder
benchmarks than this 5-pair set).

Same setup as Phase 42 — same 5 compositional pairs, same block-stacking,
same `Also` clause split — only the engram extraction changes.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase42b_l0_clause_routing.py
"""

import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.phase41_activation_composition import (
    stack_two_state_dicts,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK_BASE  = 128
ALPHA_BASE = 256
N_STEPS    = 150
GEN_TOKENS = 80


SPLIT_PATTERN = re.compile(r"\s*\b(?:Also|And)\b,?\s*", re.IGNORECASE)

def split_query(query):
    parts = SPLIT_PATTERN.split(query.strip())
    return [p.strip() for p in parts if p.strip()]


# ----------------------------------------------------------------
# L0 mean engram: token embeddings, mean-pooled, no transformer pass.
# ----------------------------------------------------------------
@torch.no_grad()
def l0_mean_engram(model, tokenizer, text, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase42b")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    # ============================================================
    # PHASE A: Build the rank-128 library
    # ============================================================
    model, _ = load_model(device)
    apply_lora(model, rank=RANK_BASE, alpha=ALPHA_BASE, target_modules=L45_TARGETS)

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}\n")

    print("=" * 60)
    print("PHASE A: BUILD LIBRARY")
    print("=" * 60)
    library = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library.append({"sd": sd, "test": dict(test), "train_prompts": train_prompts})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    # Build library keys with L0 mean
    library_keys = []
    for entry in library:
        keys_for_entry = []
        for p in entry["train_prompts"]:
            keys_for_entry.append(l0_mean_engram(model, tokenizer, p, device))
        library_keys.append(keys_for_entry)

    library_sds  = [entry["sd"]  for entry in library]
    library_meta = [entry["test"] for entry in library]
    print(f"Library built in {time.time()-t0:.0f}s\n")

    del model
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE B: Build compositional pairs
    # ============================================================
    pairs = []
    numeric_idx = [i for i, m in enumerate(library_meta) if m["type"] == "numeric"]
    entity_idx  = [i for i, m in enumerate(library_meta) if m["type"] == "entity"]
    for k in range(5):
        ai = numeric_idx[k]
        bi = entity_idx[k]
        a = library_meta[ai]
        b = library_meta[bi]
        compo_q = f"{a['prompt']} Also, {b['prompt']}"
        pairs.append({
            "pair_idx":  k,
            "ai":        ai,
            "bi":        bi,
            "a_meta":    a,
            "b_meta":    b,
            "query":     compo_q,
            "true_pair": (ai, bi),
        })

    # ============================================================
    # PHASE C: Re-load model at rank 256
    # ============================================================
    print(f"{'='*60}")
    print(f"PHASE C: RANK-256 LoRA SETUP")
    print(f"{'='*60}")
    model, _ = load_model(device)
    apply_lora(model, rank=2*RANK_BASE, alpha=2*ALPHA_BASE, target_modules=L45_TARGETS)

    def top_k_route(text, k):
        q = l0_mean_engram(model, tokenizer, text, device)
        sims = []
        for ai, keys in enumerate(library_keys):
            best = max(cosine(q, kv) for kv in keys)
            sims.append((ai, best))
        sims.sort(key=lambda x: -x[1])
        return sims[:k], q

    # ============================================================
    # PHASE D: Three routing strategies
    # ============================================================
    def evaluate_method(name, route_fn):
        a_hits, b_hits, both_hits = 0, 0, 0
        rc_total = 0
        margins = []  # cosine margin between top-1 and top-2 of each clause
        rows = []
        for p in pairs:
            idx1, idx2, rc, margin = route_fn(p)
            rc_total += rc
            if margin is not None:
                margins.append(margin)
            sd_combined = stack_two_state_dicts(library_sds[idx1], library_sds[idx2],
                                                  scale1=1.0, scale2=1.0)
            sd_gpu = {k: v.to(device) for k, v in sd_combined.items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, p["query"], tokenizer, device, GEN_TOKENS)
            ah = check_passkey(gen, p["a_meta"]["passkey"])
            bh = check_passkey(gen, p["b_meta"]["passkey"])
            if ah: a_hits += 1
            if bh: b_hits += 1
            if ah and bh: both_hits += 1
            rows.append({
                "pair":     p["pair_idx"],
                "loaded":   [idx1, idx2],
                "true":     list(p["true_pair"]),
                "routing_correct": rc,
                "margin":   margin,
                "a_hit":    ah,
                "b_hit":    bh,
                "both":     ah and bh,
            })
        mean_margin = (sum(margins) / len(margins)) if margins else None
        margin_str = f"  margin {mean_margin:.3f}" if mean_margin is not None else ""
        print(f"  {name:38s}  A {a_hits}/5  B {b_hits}/5  BOTH {both_hits}/5  "
              f"routing {rc_total}/10{margin_str}")
        return {
            "a_hits": a_hits, "b_hits": b_hits, "both": both_hits,
            "routing_correct": rc_total, "mean_margin": mean_margin,
            "rows": rows,
        }

    print(f"\n{'='*72}")
    print("PHASE D: ROUTING METHOD COMPARISON (L0 mean engram)")
    print(f"{'='*72}")
    results = {}

    # Joint-engram top-2
    def joint_top2_route(p):
        topk, _ = top_k_route(p["query"], k=2)
        idx1, sim1 = topk[0]
        idx2, sim2 = topk[1]
        rc = sum(1 for x in [idx1, idx2] if x in p["true_pair"])
        return idx1, idx2, rc, sim1 - sim2

    results["joint_top2"] = evaluate_method(
        "joint-engram top-2", joint_top2_route)

    # Clause-split top-1+top-1 (the deployable path, the headline measurement)
    def clause_split_route(p):
        clauses = split_query(p["query"])
        if len(clauses) < 2:
            return joint_top2_route(p)
        margins = []
        idx_for_clause = []
        for c in clauses[:2]:
            topk, _ = top_k_route(c, k=2)
            idx_for_clause.append(topk[0][0])
            margins.append(topk[0][1] - topk[1][1])
        idx1, idx2 = idx_for_clause[0], idx_for_clause[1]
        rc = sum(1 for x in [idx1, idx2] if x in p["true_pair"])
        return idx1, idx2, rc, sum(margins) / 2

    results["clause_split_top1"] = evaluate_method(
        "clause-split top-1 each (DEPLOYABLE)", clause_split_route)

    # Oracle
    def oracle_route(p):
        return p["ai"], p["bi"], 2, None

    results["oracle"] = evaluate_method(
        "oracle pair (upper bound)", oracle_route)

    # ============================================================
    # SUMMARY vs Phase 42
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 42b SUMMARY vs Phase 42 (L5_nonstop_mean baseline)")
    print(f"{'='*72}")
    print(f"  {'Method':38s}  {'A':>3} {'B':>3} {'BOTH':>4}  {'routing':>8}  {'margin':>7}")
    print(f"  {'-'*38}  {'-'*3} {'-'*3} {'-'*4}  {'-'*8}  {'-'*7}")
    print(f"  Phase 42 (L5_nonstop_mean):")
    print(f"  {'  joint-engram top-2':38s}  {'4':>3} {'0':>3} {'0':>4}  {'5/10':>8}  {'—':>7}")
    print(f"  {'  clause-split top-1 each':38s}  {'4':>3} {'5':>3} {'4':>4}  {'10/10':>8}  {'—':>7}")
    print(f"  {'  oracle pair':38s}  {'4':>3} {'5':>3} {'4':>4}  {'10/10':>8}  {'—':>7}")
    print(f"  Phase 42b (L0_mean):")
    for label in ["joint_top2", "clause_split_top1", "oracle"]:
        r = results[label]
        margin_str = f"{r['mean_margin']:.3f}" if r["mean_margin"] is not None else "—"
        print(f"    {label:36s}  {r['a_hits']:>3} {r['b_hits']:>3} {r['both']:>4}  "
              f"{f'{r['routing_correct']}/10':>8}  {margin_str:>7}")

    out = {
        "rank_base":  RANK_BASE,
        "rank_stacked": 2 * RANK_BASE,
        "n_pairs":    len(pairs),
        "engram":     "L0_mean",
        "results":    results,
    }
    with open(results_dir / "l0_clause_routing.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
