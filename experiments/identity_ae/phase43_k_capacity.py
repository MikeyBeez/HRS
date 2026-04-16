"""Phase 43: K-capacity sweep for activation-level block-stacking.

Phase 41 showed that activation-level block-stacking of two LoRA adapters
recovers compositional retrieval (4/5 BOTH oracle, vs 0/5 weight merging).
The unanswered question is the *capacity ceiling*: at what K does the
residual stream saturate from too many simultaneous adapter contributions?

The capacity argument from §1.3: each rank-128 LoRA modifies a ≤128-dim
subspace of the 1024-dim residual stream. K stacked adapters at full scale
occupy at most K·128 dims. K=2 → ≤256 dims (25%), K=4 → 512 (50%),
K=8 → 1024 (100%). We expect the break somewhere in this range.

We test K ∈ {1, 2, 4, 8} on oracle pair selection (separating capacity from
routing). For each K we apply LoRA at rank K·128, alpha 2·K·128 (preserving
scaling = 2 per adapter), construct compositional queries that chain K
prompts with " Also, ", block-stack the K true adapters, and generate.
The metric per trial is the fraction of expected passkeys appearing in
the generation. We report ALL_K (all K passkeys present), ANY_K (≥1
present), and the mean fraction recovered.

Trial construction uses the 20 stratified passkey passages:
  K=1: 20 trials (every passage individually, sanity check)
  K=2: 5 trials (numeric_i + entity_i, Phase 29 pairs)
  K=4: 5 trials (numeric_i + entity_i + technical_i + fact_i)
  K=8: 5 trials (two of each type per query)

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase43_k_capacity.py
"""

import json
import random
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
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK_BASE  = 128
ALPHA_BASE = 256
N_STEPS    = 150
K_VALUES   = [1, 2, 4, 8]
GEN_TOKENS = 200    # longer queries need longer generation


# ----------------------------------------------------------------
# Block-stack K rank-R state dicts into one rank-K*R state dict.
# Convention from lora_wrapper: A is (in_f, R), B is (R, out_f).
# Stacking K of them: A_stacked is (in_f, K*R) by concat dim=1,
# B_stacked is (K*R, out_f) by concat dim=0. Then by block matrix
# multiplication x @ A_stacked @ B_stacked = Σₖ x @ Aₖ @ Bₖ.
# ----------------------------------------------------------------
def stack_k_state_dicts(sds):
    """Block-stack K rank-R state dicts into one rank-K*R state dict."""
    out = {}
    keys = list(sds[0].keys())
    for k in keys:
        if "lora_A" in k:
            out[k] = torch.cat([sd[k] for sd in sds], dim=1)
        elif "lora_B" in k:
            out[k] = torch.cat([sd[k] for sd in sds], dim=0)
        else:
            out[k] = sds[0][k]
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase43")
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
    library_sds = []
    library_meta = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library_sds.append(sd)
        library_meta.append(dict(test))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")
    print(f"Library built in {time.time()-t0:.0f}s")

    # The stratified order is num_0..4, ent_0..4, tech_0..4, fact_0..4 (indices 0..19)
    NUM, ENT, TECH, FACT = range(0, 5), range(5, 10), range(10, 15), range(15, 20)
    print(f"\nIndex layout: numeric={list(NUM)}, entity={list(ENT)}, "
          f"technical={list(TECH)}, fact={list(FACT)}")

    del model
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE B: Trial construction for each K
    # ============================================================
    def build_query(indices):
        """Chain the prompts at given indices with ' Also, '."""
        prompts = [library_meta[i]["prompt"] for i in indices]
        return " Also, ".join(prompts)

    trials_by_k = {}

    # K=1: every passage individually
    trials_by_k[1] = [
        {"indices": [i],
         "passkeys": [library_meta[i]["passkey"]],
         "query":     library_meta[i]["prompt"]}
        for i in range(20)
    ]

    # K=2: Phase 29's pairs (numeric_i + entity_i)
    trials_by_k[2] = []
    for i in range(5):
        idx = [list(NUM)[i], list(ENT)[i]]
        trials_by_k[2].append({
            "indices":  idx,
            "passkeys": [library_meta[j]["passkey"] for j in idx],
            "query":    build_query(idx),
        })

    # K=4: numeric_i + entity_i + technical_i + fact_i (one of each type)
    trials_by_k[4] = []
    for i in range(5):
        idx = [list(NUM)[i], list(ENT)[i], list(TECH)[i], list(FACT)[i]]
        trials_by_k[4].append({
            "indices":  idx,
            "passkeys": [library_meta[j]["passkey"] for j in idx],
            "query":    build_query(idx),
        })

    # K=8: two of each type per query, sliding window over the 5 indices per type
    trials_by_k[8] = []
    for i in range(5):
        j = (i + 1) % 5
        idx = [list(NUM)[i], list(NUM)[j],
               list(ENT)[i], list(ENT)[j],
               list(TECH)[i], list(TECH)[j],
               list(FACT)[i], list(FACT)[j]]
        trials_by_k[8].append({
            "indices":  idx,
            "passkeys": [library_meta[k]["passkey"] for k in idx],
            "query":    build_query(idx),
        })

    print(f"\nTrial counts: " +
          ", ".join(f"K={k}: {len(trials_by_k[k])}" for k in K_VALUES))

    # Show the K=8 query length so we know how long generation needs to be
    long_q = trials_by_k[8][0]["query"]
    print(f"\nLongest query (K=8, trial 0):\n  {long_q[:200]}{'...' if len(long_q)>200 else ''}")
    print(f"  query length: {len(tokenizer.encode(long_q))} tokens")

    # ============================================================
    # PHASE C: For each K, apply LoRA at rank K*128 and run trials
    # ============================================================
    all_results = {}

    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K={K}: rank={K*RANK_BASE}, alpha={K*ALPHA_BASE}, "
              f"residual subspace ≤{(K*RANK_BASE)*100/1024:.0f}% of 1024-dim")
        print(f"{'='*60}")

        # Reload model fresh and apply LoRA at the K-rank
        model, _ = load_model(device)
        apply_lora(model, rank=K*RANK_BASE, alpha=K*ALPHA_BASE,
                   target_modules=L45_TARGETS)

        trials = trials_by_k[K]
        per_trial = []
        all_correct = 0
        any_correct = 0
        total_pks = 0
        found_pks = 0

        for ti, trial in enumerate(trials):
            # Build the K-stacked state dict from the trial's adapter indices
            sds = [library_sds[idx] for idx in trial["indices"]]
            sd_combined = stack_k_state_dicts(sds)
            sd_gpu = {k: v.to(device) for k, v in sd_combined.items()}
            load_lora_state_dict(model, sd_gpu)

            # Generate
            gen = generate_greedy(model, trial["query"], tokenizer, device, GEN_TOKENS)

            # Check each expected passkey
            hits = [check_passkey(gen, pk) for pk in trial["passkeys"]]
            n_hits = sum(hits)
            total_pks += len(trial["passkeys"])
            found_pks += n_hits

            if n_hits == len(trial["passkeys"]):
                all_correct += 1
            if n_hits >= 1:
                any_correct += 1

            per_trial.append({
                "trial":    ti,
                "indices":  trial["indices"],
                "passkeys": trial["passkeys"],
                "hits":     hits,
                "n_hits":   n_hits,
                "n_total":  len(trial["passkeys"]),
                "fraction": n_hits / len(trial["passkeys"]),
                "gen":      gen[:200],
            })

        n_trials = len(trials)
        mean_fraction = found_pks / total_pks if total_pks > 0 else 0.0
        print(f"  ALL_K (all {K} passkeys retrieved):  {all_correct}/{n_trials} "
              f"({all_correct/n_trials:.0%})")
        print(f"  ANY_K (at least 1 retrieved):       {any_correct}/{n_trials} "
              f"({any_correct/n_trials:.0%})")
        print(f"  Mean fraction recovered:            {mean_fraction:.0%}  "
              f"({found_pks}/{total_pks} passkeys)")

        # Per-trial detail (K > 1 only; K=1 is just the per-passage sanity check)
        if K > 1:
            print(f"\n  Per-trial:")
            for r in per_trial:
                hit_str = "".join("✓" if h else "✗" for h in r["hits"])
                print(f"    trial {r['trial']}: indices {r['indices']}  "
                      f"hits {hit_str}  ({r['n_hits']}/{r['n_total']})")

        all_results[K] = {
            "K":             K,
            "rank":          K * RANK_BASE,
            "alpha":         K * ALPHA_BASE,
            "n_trials":      n_trials,
            "all_correct":   all_correct,
            "any_correct":   any_correct,
            "mean_fraction": mean_fraction,
            "found_pks":     found_pks,
            "total_pks":     total_pks,
            "per_trial":     per_trial,
        }

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 43 SUMMARY: K-capacity sweep")
    print(f"{'='*72}")
    print(f"  {'K':>3}  {'rank':>5}  {'subspace':>10}  {'ALL_K':>10}  {'ANY_K':>10}  {'mean':>10}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for K in K_VALUES:
        r = all_results[K]
        subspace = f"{r['rank']*100/1024:.0f}%"
        all_str = f"{r['all_correct']}/{r['n_trials']}"
        any_str = f"{r['any_correct']}/{r['n_trials']}"
        mean_str = f"{r['mean_fraction']:.0%}"
        print(f"  {K:>3}  {r['rank']:>5}  {subspace:>10}  {all_str:>10}  "
              f"{any_str:>10}  {mean_str:>10}")

    out = {
        "rank_base":  RANK_BASE,
        "alpha_base": ALPHA_BASE,
        "K_values":   K_VALUES,
        "results":    all_results,
    }
    with open(results_dir / "k_capacity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
