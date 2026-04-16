"""Phase 29: Top-k adapter composition for compositional queries.

The architecture in phases 21-28 loads exactly one adapter at a time. A
query that touches two absorbed passages cannot be answered with one
adapter — the loaded adapter knows about its passage but not the other.

LoRA's linearity makes adapter composition well-defined: the LoRA
contribution at each layer is `(x @ A @ B) * (alpha/rank)`, which is a
linear function of (A, B). Two adapters can be merged by averaging or
summing their A and B matrices and dividing alpha appropriately. We test
whether merging two adapters lets the model retrieve both passkeys from
a compositional query.

Test design:
  - Pick K=5 pairs of absorbed passages (one per type, paired with a
    different type)
  - For each pair, build a compositional query of the form
    "What is the X access code AND when did Y make their breakthrough?"
  - Route via the prompt's engram to find the top-2 adapters
  - Merge their LoRA states (mean of A's, mean of B's)
  - Generate, check both passkeys appear

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase29_composition.py
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, RANK, N_STEPS, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase21_per_passage_adapters import train_passage_adapter
from experiments.identity_ae.phase22_engram_key import (
    make_key, cosine_match, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


SOURCE = "L5_mean"


def merge_adapters(sd_list):
    """Average a list of LoRA state dicts. Returns a new state dict.

    LoRA contribution is (x A B) * scale. Mean-merging A and B gives the
    contribution (x · mean(A_i) · mean(B_i)) * scale, which is *not* the
    same as the average of individual contributions, but is a reasonable
    first-pass merge that preserves the linear structure of LoRA.
    """
    merged = {}
    keys = list(sd_list[0].keys())
    for k in keys:
        stacked = torch.stack([sd[k] for sd in sd_list], dim=0)
        merged[k] = stacked.mean(dim=0)
    return merged


def top_k_route(query_key, library, k=2):
    """Return top-k library indices by cosine similarity, plus their scores."""
    q_n = query_key / (query_key.norm() + 1e-8)
    sims = []
    for ai, entry in enumerate(library):
        # Each entry has a single key here (built from the original prompt)
        kv = entry["key"]
        kv_n = kv / (kv.norm() + 1e-8)
        sims.append((ai, float(torch.dot(q_n, kv_n))))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase29")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION (single-prompt, 150 steps each — phase 24 winner)
    # ============================================================
    print(f"{'='*60}")
    print(f"ABSORPTION PHASE")
    print(f"{'='*60}")
    library = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_passage_adapter(model, test["passage"], tokenizer, device,
                               n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}

        # Key from original prompt under base
        reset_lora_to_zero(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        key = make_key(model, ids_t, SOURCE)
        library.append({"key": key, "sd": sd, "test": dict(test)})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")
    print(f"  Library: {len(library)} adapters")

    # ============================================================
    # SINGLE-ADAPTER BASELINE: same-prompt retrieval
    # ============================================================
    print(f"\n{'='*60}")
    print(f"SINGLE-ADAPTER BASELINE (same-prompt)")
    print(f"{'='*60}")
    n_baseline = 0
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q = make_key(model, ids_t, SOURCE)
        topk = top_k_route(q, library, k=1)
        best_idx = topk[0][0]
        sd_gpu = {k: v.to(device) for k, v in library[best_idx]["sd"].items()}
        load_lora_state_dict(model, sd_gpu)
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        if check_passkey(gen, test["passkey"]):
            n_baseline += 1
    print(f"  Single-adapter retrieval: {n_baseline}/20 ({n_baseline/20:.0%})")

    # ============================================================
    # COMPOSITIONAL QUERIES
    # ============================================================
    # Build 5 compositional pairs: each pair takes one numeric and one entity
    # (different types so the questions don't conflict structurally)
    pairs = []
    numeric_tests = [t for t in tests if t["type"] == "numeric"]
    entity_tests = [t for t in tests if t["type"] == "entity"]
    for i in range(5):
        pairs.append((numeric_tests[i], entity_tests[i]))

    print(f"\n{'='*60}")
    print(f"COMPOSITIONAL QUERIES (5 pairs, top-2 adapter merge)")
    print(f"{'='*60}")

    results = []
    for pi, (a, b) in enumerate(pairs):
        # Build a compositional query that touches both passages
        compo_q = f"{a['prompt']} Also, {b['prompt']}"

        # Route by engram of the compositional query
        reset_lora_to_zero(model)
        ids = tokenizer.encode(compo_q, add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q = make_key(model, ids_t, SOURCE)
        topk = top_k_route(q, library, k=2)
        idx_1, sim_1 = topk[0]
        idx_2, sim_2 = topk[1]

        # Find the "true" indices for a and b in the library
        true_a = next(i for i, e in enumerate(library) if e["test"]["id"] == a["id"])
        true_b = next(i for i, e in enumerate(library) if e["test"]["id"] == b["id"])
        routed_correct_count = sum(1 for x in [idx_1, idx_2] if x in (true_a, true_b))

        # ----- Single-adapter (best match) baseline for compositional query -----
        sd_gpu = {k: v.to(device) for k, v in library[idx_1]["sd"].items()}
        load_lora_state_dict(model, sd_gpu)
        gen_single = generate_greedy(model, compo_q, tokenizer, device, 80)
        single_a = check_passkey(gen_single, a["passkey"])
        single_b = check_passkey(gen_single, b["passkey"])

        # ----- Top-2 merged adapter -----
        # Force the merge to use the TRUE pair so we measure adapter merging
        # capability separately from routing capability
        merged = merge_adapters([library[true_a]["sd"], library[true_b]["sd"]])
        merged_gpu = {k: v.to(device) for k, v in merged.items()}
        load_lora_state_dict(model, merged_gpu)
        gen_merged = generate_greedy(model, compo_q, tokenizer, device, 80)
        merged_a = check_passkey(gen_merged, a["passkey"])
        merged_b = check_passkey(gen_merged, b["passkey"])

        # ----- Top-2 routed merge (uses whatever the router picked) -----
        merged_routed = merge_adapters([library[idx_1]["sd"], library[idx_2]["sd"]])
        mr_gpu = {k: v.to(device) for k, v in merged_routed.items()}
        load_lora_state_dict(model, mr_gpu)
        gen_routed_merged = generate_greedy(model, compo_q, tokenizer, device, 80)
        rm_a = check_passkey(gen_routed_merged, a["passkey"])
        rm_b = check_passkey(gen_routed_merged, b["passkey"])

        results.append({
            "pair": pi,
            "a": {"id": a["id"], "type": a["type"], "passkey": a["passkey"], "prompt": a["prompt"]},
            "b": {"id": b["id"], "type": b["type"], "passkey": b["passkey"], "prompt": b["prompt"]},
            "compositional_query": compo_q,
            "topk_routed": [(idx_1, sim_1), (idx_2, sim_2)],
            "true_indices": (true_a, true_b),
            "routing_correct_count": routed_correct_count,
            "single_top1": {"a": single_a, "b": single_b, "gen": gen_single[:120]},
            "merge_oracle": {"a": merged_a, "b": merged_b, "gen": gen_merged[:120]},
            "merge_routed": {"a": rm_a, "b": rm_b, "gen": gen_routed_merged[:120]},
        })

    # ============================================================
    # DRIFT CHECK
    # ============================================================
    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    # ============================================================
    # SUMMARY
    # ============================================================
    def tally(key):
        a_hits = sum(1 for r in results if r[key]["a"])
        b_hits = sum(1 for r in results if r[key]["b"])
        both = sum(1 for r in results if r[key]["a"] and r[key]["b"])
        return a_hits, b_hits, both

    s_a, s_b, s_both = tally("single_top1")
    o_a, o_b, o_both = tally("merge_oracle")
    r_a, r_b, r_both = tally("merge_routed")

    print(f"\n{'='*60}")
    print(f"PHASE 29 SUMMARY (compositional queries, 5 pairs)")
    print(f"{'='*60}")
    print(f"  Method                 |  A hit | B hit | BOTH ")
    print(f"  -----------------------|--------|-------|------")
    print(f"  single top-1 adapter   |  {s_a}/5  |  {s_b}/5  | {s_both}/5")
    print(f"  oracle merge (true a+b)|  {o_a}/5  |  {o_b}/5  | {o_both}/5")
    print(f"  routed merge (top-2)   |  {r_a}/5  |  {r_b}/5  | {r_both}/5")

    routing_correct = sum(r["routing_correct_count"] for r in results)
    print(f"\n  Top-2 routing correctness: {routing_correct}/10 (max 2 per pair)")
    print(f"  Single-adapter baseline (same-prompt): {n_baseline}/20")
    print(f"  Val PPL drift: {drift:+.3f}%")

    print(f"\n  Sample compositional generations (first pair):")
    r0 = results[0]
    print(f"    Query: {r0['compositional_query']}")
    print(f"    Expected A: {r0['a']['passkey']}  Expected B: {r0['b']['passkey']}")
    print(f"    Single top-1 gen:    {r0['single_top1']['gen']!r}")
    print(f"    Oracle merge gen:    {r0['merge_oracle']['gen']!r}")
    print(f"    Routed merge gen:    {r0['merge_routed']['gen']!r}")

    summary = {
        "n_pairs": 5,
        "single_top1": {"a": s_a, "b": s_b, "both": s_both},
        "merge_oracle": {"a": o_a, "b": o_b, "both": o_both},
        "merge_routed": {"a": r_a, "b": r_b, "both": r_both},
        "routing_top2_correct": routing_correct,
        "single_adapter_baseline": n_baseline,
        "drift_pct": drift,
        "results": results,
    }
    with open(results_dir / "composition.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
