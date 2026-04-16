"""Phase 42: Multi-address routing for compositional queries.

Phase 41 found that activation-level block-stacking is the correct operator
for combining two LoRA adapters: oracle composition (where the right pair is
loaded by hand) retrieved both passkeys 4/5 of the time. But the deployable
version — letting the cosine router pick the top-2 adapters from the joint
engram of the compositional query — collapsed to 0/5 BOTH because routing
returned the right pair only 5/10 of the time. The compositional query
"what is X AND when did Y" produces a joint engram dominated by the longer
clause, and the cosine ranker picks two numeric adapters instead of one
numeric and one entity.

The fix this script tests is the obvious one: **a compositional query is a
set of constraints, not a single point in routing space**. Decompose the
query into clauses (split on "Also," or "and"), route each clause
independently to its top-1 adapter, then combine the two routed adapters
via the block-stacking trick from Phase 41. If clause-level routing finds
the right adapters, the deployable composition path should match the
oracle path's 4/5 BOTH.

We test:

  (1) Joint-engram top-2 (the Phase 41 baseline, replayed for direct comparison)
  (2) Clause-split top-1+top-1 (the deployable version this phase tests)
  (3) Oracle composition (the Phase 41 upper bound, replayed)

Plus routing-correctness tallies for each method: how many of the right pair
actually got loaded.

The split strategy is the simplest possible: split the query on "Also," at
the sentence boundary. This works for the Phase 29 benchmark because all
compositional queries are constructed as `f"{a_prompt} Also, {b_prompt}"`,
but a deployed system would want a more robust query decomposer (regex on
"and"/"also"/"plus", or a learned splitter, or a small LM call to enumerate
sub-questions). The split mechanism is not the experimental variable here;
the variable is whether routing on the *clauses* (instead of the joint
query) finds the right pair of adapters.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase42_clause_routing.py
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
from experiments.identity_ae.phase31_weighted_pool import (
    make_key_weighted, cosine,
)
from experiments.identity_ae.phase41_activation_composition import (
    stack_two_state_dicts,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK_BASE  = 128
ALPHA_BASE = 256
N_STEPS    = 150
STRATEGY   = "nonstop_mean"
GEN_TOKENS = 80


# ----------------------------------------------------------------
# Query splitting: simplest possible decomposition.
# Splits on "Also," (case-insensitive) or "And" at sentence boundary.
# Returns a list of clause strings.
# ----------------------------------------------------------------
SPLIT_PATTERN = re.compile(r"\s*\b(?:Also|And)\b,?\s*", re.IGNORECASE)

def split_query(query):
    """Split a compositional query on 'Also' / 'And' boundaries."""
    parts = SPLIT_PATTERN.split(query.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts


# ----------------------------------------------------------------
# Routing helpers.
# ----------------------------------------------------------------
def engram_for_text(model, tokenizer, text, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    return make_key_weighted(model, tokenizer, ids_t, STRATEGY)


def top_k_route_for_engram(query_engram, library_keys, k):
    """Top-k library indices by max cosine over any of an entry's keys."""
    sims = []
    for ai, keys in enumerate(library_keys):
        best = max(cosine(query_engram, kv) for kv in keys)
        sims.append((ai, best))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase42")
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
    print("PHASE A: BUILD LIBRARY (Phase 38b protocol)")
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

    # Build library keys
    reset_lora_to_zero(model)
    library_keys = []
    for entry in library:
        keys_for_entry = []
        for p in entry["train_prompts"]:
            keys_for_entry.append(engram_for_text(model, tokenizer, p, device))
        library_keys.append(keys_for_entry)

    library_sds  = [entry["sd"]  for entry in library]
    library_meta = [entry["test"] for entry in library]

    del model
    torch.cuda.empty_cache()
    print(f"\nLibrary built in {time.time()-t0:.0f}s")

    # ============================================================
    # PHASE B: Build compositional pairs (Phase 29 / 41 set)
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

    # Verify the splitter works on these queries
    print(f"\nQuery splitting check:")
    for p in pairs:
        clauses = split_query(p["query"])
        print(f"  pair {p['pair_idx']}: split into {len(clauses)} clause(s)")
        for ci, c in enumerate(clauses):
            print(f"    [{ci}] {c[:80]}")

    # ============================================================
    # PHASE C: Re-load model at rank 256 (alpha 512)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"PHASE C: RANK-256 LoRA SETUP (alpha=512)")
    print(f"{'='*60}")
    model, _ = load_model(device)
    apply_lora(model, rank=2*RANK_BASE, alpha=2*ALPHA_BASE, target_modules=L45_TARGETS)

    # ============================================================
    # PHASE D: Three composition methods
    # ============================================================
    def evaluate_method(name, route_fn):
        """route_fn(pair) returns (idx1, idx2, routing_correct_count)."""
        a_hits, b_hits, both_hits = 0, 0, 0
        rc_total = 0
        rows = []
        for p in pairs:
            idx1, idx2, rc = route_fn(p)
            rc_total += rc
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
                "a_hit":    ah,
                "b_hit":    bh,
                "both":     ah and bh,
                "gen":      gen[:140],
            })
        print(f"  {name:38s}  A {a_hits}/5  B {b_hits}/5  BOTH {both_hits}/5  "
              f"routing_correct {rc_total}/10")
        return {
            "name":           name,
            "a_hits":         a_hits,
            "b_hits":         b_hits,
            "both":           both_hits,
            "routing_correct": rc_total,
            "rows":           rows,
        }

    print(f"\n{'='*72}")
    print("PHASE D: ROUTING METHOD COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Method':38s} {'A hit':>5} {'B hit':>5} {'BOTH':>5}  {'routing_correct':>16}")
    print(f"  {'-'*38} {'-'*5} {'-'*5} {'-'*5}  {'-'*16}")

    results = {}

    # ----- Method 1: Joint-engram top-2 (Phase 41 deployable replay) -----
    def joint_top2_route(p):
        q_engram = engram_for_text(model, tokenizer, p["query"], device)
        # We need to compute the engram under the BASE model with LoRA at zero,
        # which is what reset_lora_to_zero ensures. But we're now at rank 256
        # with whatever was last loaded. Reset before computing.
        reset_lora_to_zero(model)
        q_engram = engram_for_text(model, tokenizer, p["query"], device)
        topk = top_k_route_for_engram(q_engram, library_keys, k=2)
        idx1, _ = topk[0]
        idx2, _ = topk[1]
        rc = sum(1 for x in [idx1, idx2] if x in p["true_pair"])
        return idx1, idx2, rc

    results["joint_top2"] = evaluate_method(
        "joint-engram top-2 (Phase 41 baseline)", joint_top2_route)

    # ----- Method 2: Clause-split top-1+top-1 (THIS PHASE) -----
    def clause_split_route(p):
        clauses = split_query(p["query"])
        # If splitting fails (only 1 clause), fall back to joint top-2
        if len(clauses) < 2:
            return joint_top2_route(p)
        # Route each clause to its top-1
        reset_lora_to_zero(model)
        idx_for_clause = []
        for c in clauses[:2]:  # only the first two clauses
            c_engram = engram_for_text(model, tokenizer, c, device)
            topk = top_k_route_for_engram(c_engram, library_keys, k=1)
            idx_for_clause.append(topk[0][0])
        idx1, idx2 = idx_for_clause[0], idx_for_clause[1]
        rc = sum(1 for x in [idx1, idx2] if x in p["true_pair"])
        return idx1, idx2, rc

    results["clause_split_top1"] = evaluate_method(
        "clause-split top-1 each (DEPLOYABLE)", clause_split_route)

    # ----- Method 3: Oracle pair (Phase 41 upper bound replay) -----
    def oracle_route(p):
        return p["ai"], p["bi"], 2  # always routing-correct by construction

    results["oracle"] = evaluate_method(
        "oracle pair (Phase 41 upper bound)", oracle_route)

    # ============================================================
    # Print routing detail per pair, for the clause-split method
    # ============================================================
    print(f"\n{'='*72}")
    print("PER-PAIR ROUTING DETAIL (clause-split top-1)")
    print(f"{'='*72}")
    for row in results["clause_split_top1"]["rows"]:
        loaded = row["loaded"]
        true = row["true"]
        match_marker = lambda x: "✓" if x in true else "✗"
        print(f"  pair {row['pair']}: loaded {loaded[0]}{match_marker(loaded[0])} "
              f"+ {loaded[1]}{match_marker(loaded[1])}  "
              f"(true {true[0]}, {true[1]})  "
              f"A={'✓' if row['a_hit'] else '✗'} B={'✓' if row['b_hit'] else '✗'} "
              f"BOTH={'✓' if row['both'] else '✗'}")

    # ============================================================
    # Sample generations from clause-split method
    # ============================================================
    print(f"\n{'='*72}")
    print("SAMPLE GENERATIONS (clause-split top-1, all pairs)")
    print(f"{'='*72}")
    for row in results["clause_split_top1"]["rows"]:
        p = pairs[row["pair"]]
        print(f"  pair {row['pair']} expect A={p['a_meta']['passkey']}, "
              f"B={p['b_meta']['passkey']}")
        print(f"    {row['gen']!r}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 42 SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Method':38s} {'A hit':>5} {'B hit':>5} {'BOTH':>5}  {'routing':>10}")
    print(f"  {'-'*38} {'-'*5} {'-'*5} {'-'*5}  {'-'*10}")
    print(f"  Phase 29 weight-merge baselines: all 0/5 BOTH")
    print(f"  {'-'*38} {'-'*5} {'-'*5} {'-'*5}  {'-'*10}")
    for label in ["joint_top2", "clause_split_top1", "oracle"]:
        r = results[label]
        print(f"  {r['name']:38s} {r['a_hits']:>3}/5 {r['b_hits']:>3}/5 "
              f"{r['both']:>3}/5  {r['routing_correct']:>5}/10")

    out = {
        "rank_base":      RANK_BASE,
        "rank_stacked":   2 * RANK_BASE,
        "n_pairs":        len(pairs),
        "results":        results,
    }
    with open(results_dir / "clause_routing.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
