"""Phase 41: Activation-level composition via block-stacking (the right linearization).

Phase 29 / 29b found that mean-merging two LoRA state dicts produces 0/5 BOTH on
compositional queries. The interpretation in the paper was "linear merging fails
because LoRA encodes a direction in function space, not a separable concept."
That's true but incomplete: weight averaging is the *wrong* linearization. The
*correct* linearization is to sum the two adapters' **contributions** at every
instrumented layer, not to merge their matrices. Mathematically:

  contribution_A(x) + contribution_B(x)
    = (x @ A1 @ B1) * s + (x @ A2 @ B2) * s

is what we want. Mean-merging the matrices gives instead

  (x @ ((A1+A2)/2) @ ((B1+B2)/2)) * s
    = (1/4) * s * (x @ A1 @ B1 + x @ A1 @ B2 + x @ A2 @ B1 + x @ A2 @ B2)

The cross terms x @ A1 @ B2 and x @ A2 @ B1 are pure interference — matrix
products of two adapters trained independently with no shared semantics. They
are what destroy the merged adapter, and they vanish in the activation-level
sum because the block structure isolates them.

The block-stacking trick: take

  A_stacked = concat([A1, A2], dim=1)   # (in_f, 2R)
  B_stacked = concat([B1, B2], dim=0)   # (2R, out_f)

and verify that

  x @ A_stacked @ B_stacked
    = x @ [A1 | A2] @ [B1; B2]
    = x @ A1 @ B1 + x @ A2 @ B2

This is exactly the activation-level sum, implemented as a single rank-256
LoRA load. No `lora_wrapper.py` modification required: just apply LoRA at
rank 256 / alpha 512 to a fresh model and load the stacked state dict.

We test five things:

  (1) Oracle composition at full scale (alpha=512, scale=2 per contribution).
      Use the *true* pair of adapters for each compositional query. If
      activation-level composition is the right operator, this should retrieve
      both passkeys most of the time.

  (2) Oracle composition at half scale (multiply stacked A by 0.5). The
      summed contribution might be twice what each adapter was trained to
      produce, so halving each one reproduces the per-adapter magnitude.
      Tests whether scale matters.

  (3) Oracle composition at 1/sqrt(2) scale (~0.707). The geometric-mean
      compromise between full and half.

  (4) Single top-1 baseline at rank 256 (load adapter A in the first block,
      zero in the second block). Should recover Phase 29's single-adapter
      result and confirm the rank-256 plumbing doesn't itself cause issues.

  (5) Routed top-2 composition (the deployable version). Route the
      compositional query through the engram lookup, take the top-2
      adapters, stack them, generate. This is what a deployed system
      would do.

Phase 29's reference numbers (rank 512 prototype, all methods 0/5 BOTH):
  single top-1 adapter      A 3/5  B 0/5  BOTH 0/5
  oracle weight merge       A 0/5  B 1/5  BOTH 0/5
  routed weight merge       A 1/5  B 0/5  BOTH 0/5

Phase 29b (rank 128, same numbers, all 0/5 BOTH).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase41_activation_composition.py
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
from experiments.identity_ae.phase31_weighted_pool import (
    make_key_weighted, cosine,
)
from experiments.identity_ae.phase40_sparsity import round_trip_sparse_int8
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK_BASE  = 128         # Per-adapter rank (Phase 38b default)
ALPHA_BASE = 256         # Per-adapter alpha (preserves scaling = 2)
N_STEPS    = 150
STRATEGY   = "nonstop_mean"
GEN_TOKENS = 80          # Compositional queries are longer; allow more output

# Phase 40 best compression: prune at threshold 5e-3, then int8 quantize the
# surviving values, then dequantize on load. The result is a dense fp32 state
# dict with the lossy compression baked in (zeros at pruned positions, int8
# quantization noise on the rest). This is what a deployed library would store
# in sparse-bitmap-int8 format and decompress per query.
PRUNE_THRESHOLD = 5e-3


# ----------------------------------------------------------------
# Block-stacking: combine two rank-R state dicts into one rank-2R state dict.
# Convention from lora_wrapper: A is (in_f, rank), B is (rank, out_f).
# ----------------------------------------------------------------
def stack_two_state_dicts(sd1, sd2, scale1=1.0, scale2=1.0):
    """Block-stack two LoRA state dicts. The two adapters' contributions are
    summed at the activation level via the matrix-product identity.

    scale1, scale2 multiply each adapter's A matrix, which scales that adapter's
    contribution by the same factor (since contribution = (x @ A @ B) * scaling).
    Default 1.0 / 1.0 reproduces the activation-level sum at full per-adapter
    scale; 0.5 / 0.5 halves it; etc.
    """
    out = {}
    keys = list(sd1.keys())
    for k in keys:
        v1 = sd1[k]
        v2 = sd2[k]
        if "lora_A" in k:
            # (in_f, rank) — concat along dim=1 (rank axis)
            out[k] = torch.cat([v1 * scale1, v2 * scale2], dim=1)
        elif "lora_B" in k:
            # (rank, out_f) — concat along dim=0 (rank axis)
            out[k] = torch.cat([v1, v2], dim=0)
        else:
            # Should not happen for LoRA tensors, but be safe
            out[k] = v1
    return out


def single_in_double_slot(sd, slot, rank_base=RANK_BASE):
    """Build a rank-2R state dict that contains one rank-R adapter in the
    `slot` half and zeros in the other half. Used for the single-top-1 baseline
    test inside the rank-256 plumbing."""
    out = {}
    for k, v in sd.items():
        if "lora_A" in k:
            zeros = torch.zeros_like(v)
            if slot == 0:
                out[k] = torch.cat([v, zeros], dim=1)
            else:
                out[k] = torch.cat([zeros, v], dim=1)
        elif "lora_B" in k:
            zeros = torch.zeros_like(v)
            if slot == 0:
                out[k] = torch.cat([v, zeros], dim=0)
            else:
                out[k] = torch.cat([zeros, v], dim=0)
        else:
            out[k] = v
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase41")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    # ============================================================
    # PHASE A: Build the rank-128 library (Phase 38b protocol)
    # ============================================================
    model, _ = load_model(device)
    n_lora_base = apply_lora(model, rank=RANK_BASE, alpha=ALPHA_BASE,
                              target_modules=L45_TARGETS)
    print(f"Per-adapter LoRA: rank {RANK_BASE}, alpha {ALPHA_BASE}, "
          f"{n_lora_base:,} params, scaling = {ALPHA_BASE/RANK_BASE}\n")

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

    # Build library keys for routing (used in test 5)
    reset_lora_to_zero(model)
    library_keys = []
    for entry in library:
        keys_for_entry = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys_for_entry.append(make_key_weighted(model, tokenizer, ids_t, STRATEGY))
        library_keys.append(keys_for_entry)
    print(f"\nLibrary built: 20 adapters, {sum(len(k) for k in library_keys)} keys total")

    # Snapshot the library state dicts to CPU and free the rank-128 model
    library_sds_uncompressed = [entry["sd"] for entry in library]
    library_meta = [entry["test"] for entry in library]
    library_train_prompts = [entry["train_prompts"] for entry in library]

    # Apply Phase 40's best compression to every adapter: prune + int8 round-trip.
    # This is the lossy storage format the deployed system would actually use.
    print(f"\nApplying Phase 40 best compression (prune ≥ {PRUNE_THRESHOLD:.0e} + int8 round-trip)")
    library_sds = [round_trip_sparse_int8(sd, PRUNE_THRESHOLD)
                   for sd in library_sds_uncompressed]
    # Sanity: report sparsity
    def _sparsity(sd):
        z, t = 0, 0
        for v in sd.values():
            t += v.numel()
            z += int((v == 0).sum().item())
        return z / t if t > 0 else 0.0
    print(f"  Average sparsity after compression: "
          f"{sum(_sparsity(sd) for sd in library_sds) / len(library_sds) * 100:.1f}%")

    del model
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE B: Build Phase 29's compositional pairs
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
        })

    print(f"\nCompositional pairs (5 numeric + 5 entity, paired by index):")
    for p in pairs:
        print(f"  pair {p['pair_idx']}: passkey A={p['a_meta']['passkey']}  "
              f"passkey B={p['b_meta']['passkey']}")

    # ============================================================
    # PHASE C: Re-load model and apply LoRA at rank 256, alpha 512
    # ============================================================
    print(f"\n{'='*60}")
    print(f"PHASE C: RANK-{2*RANK_BASE} LoRA SETUP (alpha={2*ALPHA_BASE}, scaling=2)")
    print(f"{'='*60}")
    model, _ = load_model(device)
    n_lora_double = apply_lora(model, rank=2*RANK_BASE, alpha=2*ALPHA_BASE,
                                target_modules=L45_TARGETS)
    print(f"Stacked LoRA: rank {2*RANK_BASE}, alpha {2*ALPHA_BASE}, "
          f"{n_lora_double:,} params, scaling = {(2*ALPHA_BASE)/(2*RANK_BASE)}")
    print(f"This is the rank-doubled wrapper into which we load block-stacked\n"
          f"adapter pairs. The forward pass computes the activation-level sum\n"
          f"x @ A_stacked @ B_stacked = x @ A1 @ B1 + x @ A2 @ B2.\n")

    # ============================================================
    # PHASE D: Test 1 — Oracle composition at full scale (1.0)
    # ============================================================
    def evaluate_method(name, build_sd_fn):
        """Run a composition method on all 5 pairs and tally results."""
        a_hits, b_hits, both_hits = 0, 0, 0
        rows = []
        for p in pairs:
            sd_combined = build_sd_fn(p)
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
                "a_passkey": p["a_meta"]["passkey"],
                "b_passkey": p["b_meta"]["passkey"],
                "a_hit":    ah,
                "b_hit":    bh,
                "both":     ah and bh,
                "gen":      gen[:140],
            })
        print(f"  {name:36s}  A {a_hits}/5  B {b_hits}/5  BOTH {both_hits}/5")
        return {"a_hits": a_hits, "b_hits": b_hits, "both": both_hits, "rows": rows}

    print(f"{'='*72}")
    print("PHASE D: COMPOSITION RESULTS (5 pairs, oracle pair selection)")
    print(f"{'='*72}")
    print(f"  Method                                  A hit  B hit  BOTH")
    print(f"  {'-'*40} {'-'*5} {'-'*5} {'-'*5}")

    results = {}

    # Control: Oracle composition with UNCOMPRESSED adapters (so we can see
    # whether the compression itself costs anything on the composition path)
    results["oracle_full_scale_uncompressed"] = evaluate_method(
        "oracle stack, scale 1.0, UNCOMPRESSED",
        lambda p: stack_two_state_dicts(library_sds_uncompressed[p["ai"]],
                                          library_sds_uncompressed[p["bi"]],
                                          scale1=1.0, scale2=1.0))

    # Test 1: Oracle composition, full scale 1.0, on COMPRESSED adapters
    results["oracle_full_scale"] = evaluate_method(
        "oracle stack, scale 1.0 (compressed)",
        lambda p: stack_two_state_dicts(library_sds[p["ai"]], library_sds[p["bi"]],
                                          scale1=1.0, scale2=1.0))

    # Test 2: Oracle composition, half scale 0.5
    results["oracle_half_scale"] = evaluate_method(
        "oracle stack, scale 0.5 (half each)",
        lambda p: stack_two_state_dicts(library_sds[p["ai"]], library_sds[p["bi"]],
                                          scale1=0.5, scale2=0.5))

    # Test 3: Oracle composition, 1/sqrt(2) scale ≈ 0.707
    results["oracle_invsqrt2_scale"] = evaluate_method(
        "oracle stack, scale 1/√2 (≈0.707)",
        lambda p: stack_two_state_dicts(library_sds[p["ai"]], library_sds[p["bi"]],
                                          scale1=2**-0.5, scale2=2**-0.5))

    # Test 4: Single top-1 baseline (load only adapter A in slot 0, zeros in slot 1)
    results["single_top1_in_slot0"] = evaluate_method(
        "single A only (in slot 0, slot 1 = 0)",
        lambda p: single_in_double_slot(library_sds[p["ai"]], slot=0))

    # Test 4b: Single top-1 baseline with adapter B
    results["single_top1_B_in_slot1"] = evaluate_method(
        "single B only (in slot 1, slot 0 = 0)",
        lambda p: single_in_double_slot(library_sds[p["bi"]], slot=1))

    # ============================================================
    # PHASE E: Test 5 — Routed top-2 composition (the deployable version)
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE E: ROUTED TOP-2 COMPOSITION (deployable version)")
    print(f"{'='*72}")
    print("  Route the compositional query, take top-2 by cosine similarity,")
    print("  block-stack them at full scale, generate. This is what a deployed")
    print("  system would do.\n")

    def top_k_route(query_text, k=2):
        """Compute the engram of the query under the *base* model and find the
        top-k library indices by cosine similarity. We need to reset the LoRA
        to zero before computing the engram."""
        reset_lora_to_zero(model)
        ids = tokenizer.encode(query_text, add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
        sims = []
        for ai, keys in enumerate(library_keys):
            best_for_entry = max(cosine(q, k_) for k_ in keys)
            sims.append((ai, best_for_entry))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]

    routed_results = []
    routed_a_hits, routed_b_hits, routed_both = 0, 0, 0
    routing_correct = 0
    for p in pairs:
        topk = top_k_route(p["query"], k=2)
        idx1, sim1 = topk[0]
        idx2, sim2 = topk[1]
        # Routing correctness: how many of (true_a, true_b) are in (idx1, idx2)?
        rc = sum(1 for x in [idx1, idx2] if x in (p["ai"], p["bi"]))
        routing_correct += rc

        # Stack the routed top-2 (regardless of whether routing was correct)
        sd_combined = stack_two_state_dicts(library_sds[idx1], library_sds[idx2],
                                              scale1=1.0, scale2=1.0)
        sd_gpu = {k: v.to(device) for k, v in sd_combined.items()}
        load_lora_state_dict(model, sd_gpu)
        gen = generate_greedy(model, p["query"], tokenizer, device, GEN_TOKENS)
        ah = check_passkey(gen, p["a_meta"]["passkey"])
        bh = check_passkey(gen, p["b_meta"]["passkey"])
        if ah: routed_a_hits += 1
        if bh: routed_b_hits += 1
        if ah and bh: routed_both += 1
        routed_results.append({
            "pair":     p["pair_idx"],
            "topk":     [(idx1, sim1), (idx2, sim2)],
            "true_pair": (p["ai"], p["bi"]),
            "routing_correct_count": rc,
            "a_hit":    ah,
            "b_hit":    bh,
            "both":     ah and bh,
            "gen":      gen[:140],
        })

    print(f"  routed top-2 (full scale)             "
          f"A {routed_a_hits}/5  B {routed_b_hits}/5  BOTH {routed_both}/5")
    print(f"  Routing correctness: {routing_correct}/10 (max 2 per pair)")
    results["routed_top2"] = {
        "a_hits": routed_a_hits,
        "b_hits": routed_b_hits,
        "both":   routed_both,
        "routing_correct_total": routing_correct,
        "rows":   routed_results,
    }

    # ============================================================
    # Sample generations from each method, first pair
    # ============================================================
    print(f"\n{'='*72}")
    print("SAMPLE GENERATIONS (pair 0)")
    print(f"{'='*72}")
    p0 = pairs[0]
    print(f"  Query: {p0['query']}")
    print(f"  Expected A: {p0['a_meta']['passkey']}  Expected B: {p0['b_meta']['passkey']}")
    print()
    for label in ["oracle_full_scale", "oracle_half_scale", "oracle_invsqrt2_scale",
                  "single_top1_in_slot0", "routed_top2"]:
        r = results[label]
        if "rows" in r and r["rows"]:
            print(f"  [{label:25s}] {r['rows'][0]['gen']!r}")

    # ============================================================
    # SUMMARY vs Phase 29 baselines
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 41 SUMMARY vs Phase 29 / 29b")
    print(f"{'='*72}")
    print(f"  Phase 29 baselines (rank 512, weight merging):")
    print(f"    single top-1 adapter            A 3/5  B 0/5  BOTH 0/5")
    print(f"    oracle weight merge             A 0/5  B 1/5  BOTH 0/5")
    print(f"    routed weight merge             A 1/5  B 0/5  BOTH 0/5")
    print(f"  Phase 29b baselines (rank 128, weight merging): all 0/5 BOTH")
    print()
    print(f"  Phase 41 (rank 256, activation-level block stacking):")
    print(f"    [adapters compressed via Phase 40 best stack: prune ≥ 5e-3 + int8]")
    for label in ["oracle_full_scale_uncompressed", "oracle_full_scale",
                  "oracle_half_scale", "oracle_invsqrt2_scale",
                  "single_top1_in_slot0", "single_top1_B_in_slot1", "routed_top2"]:
        r = results[label]
        marker = "↓ uncompressed reference" if "uncompressed" in label else ""
        print(f"    {label:34s}  A {r['a_hits']}/5  B {r['b_hits']}/5  BOTH {r['both']}/5  {marker}")

    out = {
        "rank_base":      RANK_BASE,
        "alpha_base":     ALPHA_BASE,
        "rank_stacked":   2 * RANK_BASE,
        "alpha_stacked":  2 * ALPHA_BASE,
        "n_pairs":        len(pairs),
        "results":        results,
    }
    with open(results_dir / "activation_composition.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
