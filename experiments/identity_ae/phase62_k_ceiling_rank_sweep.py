"""Phase 62: K-ceiling × rank sweep — how does rank budget interact with K?

Phase 43 showed activation-level block-stacking hits a capacity ceiling as K
grows (all 4 passkeys recovered at K=4 but degrading toward K=8). The open
question: does the ceiling depend on per-adapter rank, or only on total
rank budget K×R?

Hypothesis A (rank-dominated): total budget K×R determines capacity. Lower
per-adapter rank at the same K should match higher-rank at lower K if K×R
is held constant. E.g. K=2 rank=64 (budget 128) ≈ K=2 rank=128 (budget 256)
at half the memory.

Hypothesis B (K-dominated): each additional adapter causes cross-talk
regardless of rank. Lower rank at the same K degrades retrieval because
individual adapters are weaker, not because the combined stack is wider.

Design: 8-passage library, first 8 from stratified_tests().
Ranks: [128, 64, 32, 16] — spans ×8 range.
K values: 1 (single baseline), 2, 4.
Training: 150 steps uniform multipara loss, same as Phase 43.
Generation: 80 tokens for single, 80 tokens for compositional.

K=2 pairs  (5 total): (0,1), (2,3), (4,5), (6,7), (0,2)   — intra & cross
K=4 groups (2 total): (0,1,2,3), (4,5,6,7)

Budget table reported:
  K=1 rank=128 → budget 128
  K=1 rank=64  → budget 64
  K=2 rank=128 → budget 256
  K=2 rank=64  → budget 128 (matches K=1 rank=128 budget)
  K=2 rank=32  → budget 64  (matches K=1 rank=64 budget)
  K=4 rank=128 → budget 512
  K=4 rank=64  → budget 256 (matches K=2 rank=128 budget)
  K=4 rank=32  → budget 128 (matches K=1 rank=128 budget)
  K=4 rank=16  → budget 64

Primary summary table: rows = K, cols = rank, cell = mean_fraction%.
Secondary table: rows = budget (K×R), showing that equal-budget configs
cluster together (or don't).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase62_k_ceiling_rank_sweep.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
RANK_VALUES = [128, 64, 32, 16]
N_STEPS     = 150
GEN_TOKENS  = 80   # enough for compositional queries
D           = 1024
N_PASSAGES  = 8    # first 8 from stratified_tests()


# ----------------------------------------------------------------
# Block-stacking: merge two LoRA state dicts via activation-level
# block-stacking.  lora_A shape: (in_f, R), lora_B shape: (R, out_f).
# Stacking along rank dim:
#   A_stack = cat([A1, A2], dim=1) → (in_f, 2R)
#   B_stack = cat([B1, B2], dim=0) → (2R, out_f)
# Then x @ A_stack @ B_stack = x@A1@B1 + x@A2@B2  (no cross terms).
# ----------------------------------------------------------------
def block_stack(sd1, sd2):
    """Merge two LoRA state dicts via activation-level block-stacking."""
    merged = {}
    for key in sd1:
        if "lora_A" in key:
            merged[key] = torch.cat([sd1[key], sd2[key]], dim=1)  # rank dim
        elif "lora_B" in key:
            merged[key] = torch.cat([sd1[key], sd2[key]], dim=0)  # rank dim
        else:
            merged[key] = sd1[key]
    return merged


def stack_k_sds(sds):
    """Block-stack a list of K state dicts into one rank-K*R state dict."""
    result = sds[0]
    for sd in sds[1:]:
        result = block_stack(result, sd)
    return result


# ----------------------------------------------------------------
# Clause splitting for compositional prompts.
# Splits on " Also, " first, then " and " tokens.
# ----------------------------------------------------------------
def split_clauses(query):
    """Split a compositional query into individual clauses."""
    clauses = []
    # Primary split: " Also, "
    parts = query.split(" Also, ")
    for part in parts:
        # Secondary: split further on ", " only if part looks compound
        sub = part.split(", and ")
        clauses.extend([s.strip() for s in sub if s.strip()])
    return [c for c in clauses if c]


# ----------------------------------------------------------------
# Build trial sets from 8-passage library
# ----------------------------------------------------------------
def build_trials(library_meta):
    """Construct K=1, K=2, K=4 trial sets from the first 8 passages."""
    def query_from(indices):
        prompts = [library_meta[i]["prompt"] for i in indices]
        return " Also, ".join(prompts)

    trials_by_k = {}

    # K=1: single-passage retrieval for all 8 passages
    trials_by_k[1] = [
        {
            "indices":  [i],
            "passkeys": [library_meta[i]["passkey"]],
            "query":    library_meta[i]["prompt"],
        }
        for i in range(8)
    ]

    # K=2: 5 pairs — 4 adjacent intra-type + 1 cross-type
    k2_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2)]
    trials_by_k[2] = [
        {
            "indices":  list(idx),
            "passkeys": [library_meta[i]["passkey"] for i in idx],
            "query":    query_from(idx),
        }
        for idx in k2_pairs
    ]

    # K=4: 2 groups of 4
    k4_groups = [(0, 1, 2, 3), (4, 5, 6, 7)]
    trials_by_k[4] = [
        {
            "indices":  list(idx),
            "passkeys": [library_meta[i]["passkey"] for i in idx],
            "query":    query_from(idx),
        }
        for idx in k4_groups
    ]

    return trials_by_k


# ----------------------------------------------------------------
# Core evaluation: run trials with a given model (already loaded
# with LoRA at K×rank) and a stacked state dict per trial.
# ----------------------------------------------------------------
def run_trials(model, tokenizer, device, trials, library_sds):
    """Evaluate compositional retrieval for a set of trials.

    Returns list of per-trial dicts with hit info.
    """
    per_trial = []
    for ti, trial in enumerate(trials):
        # Build stacked state dict for this trial
        sds_for_trial = [library_sds[idx] for idx in trial["indices"]]
        sd_stacked = stack_k_sds(sds_for_trial)
        sd_gpu = {k: v.to(device) for k, v in sd_stacked.items()}
        load_lora_state_dict(model, sd_gpu)

        gen = generate_greedy(model, trial["query"], tokenizer, device, GEN_TOKENS)

        hits = [check_passkey(gen, pk) for pk in trial["passkeys"]]
        n_hits = sum(hits)
        K = len(trial["passkeys"])

        per_trial.append({
            "trial":    ti,
            "K":        K,
            "indices":  trial["indices"],
            "passkeys": trial["passkeys"],
            "hits":     hits,
            "n_hits":   n_hits,
            "n_total":  K,
            "fraction": n_hits / K,
            "all":      n_hits == K,
            "any":      n_hits >= 1,
            "gen":      gen[:200],
        })

    return per_trial


# ----------------------------------------------------------------
# Aggregate trial results into summary stats
# ----------------------------------------------------------------
def aggregate(per_trial):
    n = len(per_trial)
    if n == 0:
        return {"n_trials": 0, "all_count": 0, "any_count": 0,
                "mean_fraction": 0.0, "found": 0, "total": 0}
    all_count = sum(r["all"] for r in per_trial)
    any_count = sum(r["any"] for r in per_trial)
    found = sum(r["n_hits"] for r in per_trial)
    total = sum(r["n_total"] for r in per_trial)
    return {
        "n_trials":      n,
        "all_count":     all_count,
        "any_count":     any_count,
        "mean_fraction": found / total if total > 0 else 0.0,
        "found":         found,
        "total":         total,
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase62")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()  # 20 passages: 5 numeric, 5 entity, 5 technical, 5 fact
    tests = tests[:N_PASSAGES]  # use first 8
    print(f"Using first {N_PASSAGES} passages:")
    for i, t in enumerate(tests):
        print(f"  [{i}] {t['type']:10s}  passkey={t['passkey']!r}")

    # Show the K=4 query so we know its length
    trial_q = " Also, ".join(t["prompt"] for t in tests[:4])
    n_toks_q4 = len(tokenizer.encode(trial_q))
    print(f"\nExample K=4 query ({n_toks_q4} tokens):\n  {trial_q[:180]}...")

    # Build trial sets (same for all rank conditions)
    library_meta = [dict(t) for t in tests]
    trials_by_k  = build_trials(library_meta)
    K_VALUES     = [1, 2, 4]

    print(f"\nTrial counts: " +
          ", ".join(f"K={k}: {len(trials_by_k[k])}" for k in K_VALUES))

    # ============================================================
    # Main loop: for each rank, build library then run all K trials
    # ============================================================
    # Results indexed by (rank, K)
    results = {}   # (rank, K) → aggregate dict
    all_per_trial = {}  # (rank, K) → list of per-trial dicts
    library_cache = {}  # rank → list of state dicts (one per passage)

    for rank in RANK_VALUES:
        alpha = rank * 2   # keep scaling = 2 per adapter
        print(f"\n{'='*72}")
        print(f"RANK={rank}  alpha={alpha}  (per-adapter subspace ≤{rank*100/D:.0f}% of {D})")
        print("="*72)

        # --------------------------------------------------------
        # PHASE A: build adapter library at this rank
        # --------------------------------------------------------
        print(f"\n  Building library ({N_PASSAGES} adapters × {N_STEPS} steps)...")
        model, _ = load_model(device)
        apply_lora(model, rank=rank, alpha=alpha, target_modules=L45_TARGETS)

        library_sds = []
        t0 = time.time()
        for i, test in enumerate(tests):
            reset_lora_to_zero(model)
            train_prompts = [test["prompt"]] + train_paraphrase(test)
            prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
            train_adapter_multipara(
                model, test["passage"], prompts_with_answers,
                tokenizer, device,
                n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR,
            )
            sd = {k: v.detach().cpu().clone()
                  for k, v in get_lora_state_dict(model).items()}
            library_sds.append(sd)

        library_cache[rank] = library_sds
        elapsed = time.time() - t0
        print(f"  Library built in {elapsed:.0f}s")

        # --------------------------------------------------------
        # PHASE B: for each K, run trials
        # --------------------------------------------------------
        for K in K_VALUES:
            stacked_rank = K * rank
            stacked_alpha = K * alpha
            print(f"\n  K={K}  stacked rank={stacked_rank}  "
                  f"subspace ≤{stacked_rank*100/D:.0f}%  budget={K*rank}")

            # Fresh model with LoRA at the stacked rank
            model_k, _ = load_model(device)
            apply_lora(model_k, rank=stacked_rank, alpha=stacked_alpha,
                       target_modules=L45_TARGETS)

            per_trial = run_trials(model_k, tokenizer, device,
                                   trials_by_k[K], library_sds)
            stats = aggregate(per_trial)

            results[(rank, K)]       = stats
            all_per_trial[(rank, K)] = per_trial

            all_str  = f"{stats['all_count']}/{stats['n_trials']}"
            any_str  = f"{stats['any_count']}/{stats['n_trials']}"
            mean_str = f"{stats['mean_fraction']:.0%}"
            print(f"    ALL: {all_str:6s}  ANY: {any_str:6s}  mean: {mean_str}")

            if K > 1:
                for r in per_trial:
                    hit_str = "".join("+" if h else "-" for h in r["hits"])
                    print(f"      trial {r['trial']}: idx {r['indices']}  "
                          f"[{hit_str}]  {r['n_hits']}/{r['n_total']}")

            del model_k
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # SUMMARY TABLES
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 62 SUMMARY")
    print("="*72)

    # Table 1: single-adapter baselines (K=1) by rank
    print("\nTable 1: Single-adapter baseline (K=1)")
    print(f"  {'rank':>6}  {'correct':>10}  {'mean%':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}")
    for rank in RANK_VALUES:
        s = results[(rank, 1)]
        print(f"  {rank:>6}  {s['all_count']:>3}/{s['n_trials']:<6}  "
              f"{s['mean_fraction']:>7.0%}")

    # Table 2: composition by (K, rank) — mean fraction recovered
    print("\nTable 2: Compositional retrieval — mean fraction recovered")
    header = f"  {'rank':>6}" + "".join(f"  K={k:>2}" for k in [2, 4])
    print(header)
    print("  " + "-"*6 + ("  " + "-"*5) * 2)
    for rank in RANK_VALUES:
        row = f"  {rank:>6}"
        for K in [2, 4]:
            s = results[(rank, K)]
            row += f"  {s['mean_fraction']:>5.0%}"
        print(row)

    # Table 3: ALL_K by (K, rank)
    print("\nTable 3: ALL_K count (all passkeys retrieved) — n_correct/n_trials")
    header = f"  {'rank':>6}" + "".join(f"  {'K='+str(k):>7}" for k in [2, 4])
    print(header)
    print("  " + "-"*6 + ("  " + "-"*7) * 2)
    for rank in RANK_VALUES:
        row = f"  {rank:>6}"
        for K in [2, 4]:
            s = results[(rank, K)]
            row += f"  {s['all_count']:>3}/{s['n_trials']:<3}"
        print(row)

    # Table 4: equal-budget grouping
    print("\nTable 4: Results by total budget (K×rank)")
    budgets = sorted(set(K * r for K in K_VALUES for r in RANK_VALUES))
    print(f"  {'budget':>8}  {'config':>12}  {'mean%':>8}  {'ALL':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*8}")
    for budget in budgets:
        for K in K_VALUES:
            for rank in RANK_VALUES:
                if K * rank == budget:
                    s = results[(rank, K)]
                    config_str = f"K={K} r={rank}"
                    all_str = f"{s['all_count']}/{s['n_trials']}"
                    print(f"  {budget:>8}  {config_str:>12}  "
                          f"{s['mean_fraction']:>8.0%}  {all_str:>8}")

    # ============================================================
    # Hypothesis test summary
    # ============================================================
    print(f"\n{'='*72}")
    print("Hypothesis A (rank-dominated): equal budget → equal performance")
    print("Hypothesis B (K-dominated):    higher K degrades regardless of rank")
    print()

    # Compare equal-budget pairs that differ in K
    print("Equal-budget comparison (K=1 r=128 vs K=2 r=64 vs K=4 r=32, budget=128):")
    for (K, r) in [(1, 128), (2, 64), (4, 32)]:
        s = results[(r, K)]
        print(f"  K={K} rank={r}: mean={s['mean_fraction']:.0%}  "
              f"ALL={s['all_count']}/{s['n_trials']}")

    print("\nEqual-budget comparison (K=1 r=64 vs K=2 r=32 vs K=4 r=16, budget=64):")
    for (K, r) in [(1, 64), (2, 32), (4, 16)]:
        s = results[(r, K)]
        print(f"  K={K} rank={r}: mean={s['mean_fraction']:.0%}  "
              f"ALL={s['all_count']}/{s['n_trials']}")

    print("\nEqual-budget comparison (K=2 r=128 vs K=4 r=64, budget=256):")
    for (K, r) in [(2, 128), (4, 64)]:
        s = results[(r, K)]
        print(f"  K={K} rank={r}: mean={s['mean_fraction']:.0%}  "
              f"ALL={s['all_count']}/{s['n_trials']}")

    # ============================================================
    # Save
    # ============================================================
    # Convert tuple keys to strings for JSON
    out = {
        "config": {
            "rank_values": RANK_VALUES,
            "k_values":    K_VALUES,
            "n_steps":     N_STEPS,
            "gen_tokens":  GEN_TOKENS,
            "n_passages":  N_PASSAGES,
        },
        "results": {
            f"rank{rank}_K{K}": {
                **results[(rank, K)],
                "rank": rank,
                "K":    K,
                "budget": K * rank,
                "per_trial": all_per_trial[(rank, K)],
            }
            for rank in RANK_VALUES for K in K_VALUES
        },
    }
    out_path = results_dir / "k_ceiling_rank_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
