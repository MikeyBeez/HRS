"""Phase 40: LoRA weight sparsity analysis and pruning.

Rank reduction (Phase 38a-b) gave 4× compression. Int8 quantization (Phase 30/30b)
gave another 4×. Total 16× vs the rank-512 fp32 prototype. The next question:
how sparse are the trained LoRA matrices, and can magnitude pruning + sparse
storage stack on top to give us 2-5× more?

This script implements the four-part plan:

  Experiment 1: Weight distribution measurement
    For each of 20 rank-128 multi-prompt-trained adapters, compute the absolute-
    value distribution of A and B matrices separately and aggregated. Report
    fraction of values below thresholds [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1].

  Experiment 2: Pruning sweep
    For each threshold, zero out values |w| < t in every adapter, run same-prompt
    retrieval per-adapter (load adapter, generate from its absorbed prompt, check
    passkey). Find the largest threshold that preserves 20/20.

  Experiment 3: Sparse storage size
    At the best threshold, compute storage in four formats:
      - dense fp32 (baseline, the rank-128 default)
      - dense int8 (Phase 30b baseline)
      - sparse COO int8 (32-bit linear index + 8-bit value + per-tensor scale)
      - sparse bitmap int8 (1-bit-per-element bitmap + 8-bit values + scale)
    Report which format wins and at what compression ratio.

  Experiment 4: Combined compression stack
    Run the full Phase 38b retrieval pipeline (same-prompt + training-distribution
    paraphrase + held-out paraphrase, with full library routing) using pruned +
    int8 + sparse-stored adapters that have been round-tripped (sparse → dense
    → loaded into LoRA slots). Confirm retrieval matches Phase 38b numbers.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase40_sparsity.py
"""

import json
import math
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
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import (
    make_key_weighted, cosine,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
ALPHA = RANK * 2
N_STEPS = 150
STRATEGY = "nonstop_mean"
THRESHOLDS = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


# ----------------------------------------------------------------
# Distribution measurement (Experiment 1)
# ----------------------------------------------------------------
def absvals(sd, key_filter):
    """Concatenate the absolute values of all tensors whose key matches filter."""
    parts = []
    for k, v in sd.items():
        if key_filter(k):
            parts.append(v.detach().abs().flatten().cpu())
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def fraction_below(vals, threshold):
    if vals.numel() == 0:
        return 0.0
    return float((vals < threshold).float().mean().item())


def distribution_stats(sd):
    """Per-adapter statistics, separated for A and B."""
    a_vals = absvals(sd, lambda k: "lora_A" in k)
    b_vals = absvals(sd, lambda k: "lora_B" in k)
    all_vals = torch.cat([a_vals, b_vals])

    def stats(vals):
        if vals.numel() == 0:
            return {"n": 0}
        return {
            "n":     int(vals.numel()),
            "mean":  float(vals.mean().item()),
            "std":   float(vals.std().item()),
            "min":   float(vals.min().item()),
            "max":   float(vals.max().item()),
            "median": float(vals.median().item()),
        }

    return {
        "A":   stats(a_vals),
        "B":   stats(b_vals),
        "all": stats(all_vals),
        "frac_below": {
            f"{t:.0e}": {
                "A":   fraction_below(a_vals, t),
                "B":   fraction_below(b_vals, t),
                "all": fraction_below(all_vals, t),
            }
            for t in THRESHOLDS
        },
    }


# ----------------------------------------------------------------
# Pruning (Experiment 2)
# ----------------------------------------------------------------
def prune_state_dict(sd, threshold):
    """Return a new state dict with values |w| < threshold zeroed."""
    out = {}
    for k, v in sd.items():
        v_pruned = v.clone()
        mask = v_pruned.abs() >= threshold
        v_pruned = v_pruned * mask.float()
        out[k] = v_pruned
    return out


def state_dict_sparsity(sd):
    """Fraction of zero values in the state dict."""
    total = 0
    zeros = 0
    for v in sd.values():
        total += v.numel()
        zeros += int((v == 0).sum().item())
    return zeros / total if total > 0 else 0.0


# ----------------------------------------------------------------
# Storage measurement (Experiment 3)
# ----------------------------------------------------------------
def dense_fp32_bytes(sd):
    return sum(v.numel() * 4 for v in sd.values())


def dense_int8_bytes(sd):
    """Per-tensor int8 + 8 bytes per scale."""
    return sum(v.numel() * 1 + 8 for v in sd.values())


def sparse_coo_int8_bytes(sd, threshold):
    """Each nonzero stored as (int32 linear index, int8 value).
    Plus 8 bytes per tensor for the scale, plus 4 bytes for nnz count."""
    total = 0
    for v in sd.values():
        nnz = int((v.abs() >= threshold).sum().item())
        total += nnz * (4 + 1) + 8 + 4   # index + value, scale, count
    return total


def sparse_bitmap_int8_bytes(sd, threshold):
    """1-bit-per-element bitmap + int8 values for nonzero positions.
    Plus 8 bytes per tensor for the scale."""
    total = 0
    for v in sd.values():
        n = v.numel()
        nnz = int((v.abs() >= threshold).sum().item())
        bitmap_bytes = (n + 7) // 8
        total += bitmap_bytes + nnz * 1 + 8
    return total


# ----------------------------------------------------------------
# Round-trip: prune → quantize → store sparse → load → dense
# (used by Experiment 4 to verify retrieval survives the full stack)
# ----------------------------------------------------------------
def quant_dequant_int8(t):
    """Symmetric per-tensor int8 quantization round-trip."""
    if t.numel() == 0:
        return t
    max_abs = t.abs().max().item()
    if max_abs == 0:
        return torch.zeros_like(t)
    scale = max_abs / 127.0
    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
    return q.to(torch.float32) * scale


def round_trip_sparse_int8(sd, threshold):
    """Prune at threshold, int8-quantize the surviving values, then return
    a dense fp32 state dict for loading into the model."""
    out = {}
    for k, v in sd.items():
        mask = v.abs() >= threshold
        pruned = v * mask.float()
        # Quantize the pruned tensor (zeros stay zero through symmetric quant)
        out[k] = quant_dequant_int8(pruned)
    return out


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase40")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    model, _ = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}  (rank {RANK}, alpha {ALPHA})")
    print(f"Multi-prompt training (Phase 38b protocol), {N_STEPS} steps")
    print(f"Pruning thresholds to sweep: {THRESHOLDS}\n")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}\n")

    # ============================================================
    # Build library: 20 multi-prompt rank-128 adapters (Phase 38b protocol)
    # ============================================================
    print("=" * 60)
    print("BUILDING LIBRARY (Phase 38b protocol)")
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

    # Build library keys for Experiment 4 (full routing pipeline)
    reset_lora_to_zero(model)
    library_keys = []
    for entry in library:
        keys_for_entry = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys_for_entry.append(make_key_weighted(model, tokenizer, ids_t, STRATEGY))
        library_keys.append(keys_for_entry)
    print(f"Library built in {time.time()-t0:.0f}s\n")

    # ============================================================
    # EXPERIMENT 1: Weight distribution measurement
    # ============================================================
    print("=" * 60)
    print("EXPERIMENT 1: WEIGHT DISTRIBUTIONS")
    print("=" * 60)

    per_adapter_stats = []
    for entry in library:
        per_adapter_stats.append(distribution_stats(entry["sd"]))

    # Aggregate: average frac_below across all 20 adapters, separately for A/B/all
    agg_frac_below = {f"{t:.0e}": {"A": 0.0, "B": 0.0, "all": 0.0} for t in THRESHOLDS}
    for s in per_adapter_stats:
        for t_str in agg_frac_below:
            for kind in ["A", "B", "all"]:
                agg_frac_below[t_str][kind] += s["frac_below"][t_str][kind]
    for t_str in agg_frac_below:
        for kind in ["A", "B", "all"]:
            agg_frac_below[t_str][kind] /= len(per_adapter_stats)

    # Aggregate basic stats
    a_means = [s["A"]["mean"] for s in per_adapter_stats]
    b_means = [s["B"]["mean"] for s in per_adapter_stats]
    a_max   = [s["A"]["max"]  for s in per_adapter_stats]
    b_max   = [s["B"]["max"]  for s in per_adapter_stats]
    print(f"  Per-adapter |w| means: A: {sum(a_means)/len(a_means):.4f}  "
          f"B: {sum(b_means)/len(b_means):.4f}")
    print(f"  Per-adapter |w| maxes: A: {sum(a_max)/len(a_max):.4f}  "
          f"B: {sum(b_max)/len(b_max):.4f}\n")

    print(f"  {'Threshold':>10}  {'% below A':>10}  {'% below B':>10}  {'% below all':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")
    for t in THRESHOLDS:
        t_str = f"{t:.0e}"
        d = agg_frac_below[t_str]
        print(f"  {t:>10.0e}  {d['A']*100:>9.1f}%  {d['B']*100:>9.1f}%  "
              f"{d['all']*100:>11.1f}%")

    # Per-type breakdown of overall sparsity at 1e-2
    print(f"\n  Per-type fraction below 1e-2 (where pruning is moderately aggressive):")
    by_type = {"numeric": [], "entity": [], "technical": [], "fact": []}
    for s, entry in zip(per_adapter_stats, library):
        by_type[entry["test"]["type"]].append(s["frac_below"]["1e-02"]["all"])
    for ptype, vals in by_type.items():
        print(f"    {ptype:9s}: {sum(vals)/len(vals)*100:5.1f}%  "
              f"(min {min(vals)*100:.1f}%, max {max(vals)*100:.1f}%)")

    with open(results_dir / "weight_distributions.json", "w") as f:
        json.dump({
            "per_adapter": per_adapter_stats,
            "aggregate_frac_below": agg_frac_below,
        }, f, indent=2)

    # ============================================================
    # EXPERIMENT 2: Pruning sweep
    # ============================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: PRUNING SWEEP")
    print(f"{'='*60}")
    print(f"  Same-prompt retrieval, per-adapter (no routing).\n")

    pruning_results = []
    print(f"  {'Threshold':>10}  {'Sparsity':>10}  {'Retrieval':>12}  "
          f"{'num':>4} {'ent':>4} {'tech':>4} {'fact':>4}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
    for t in THRESHOLDS:
        n_correct = 0
        per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        sparsities = []
        for i, entry in enumerate(library):
            pruned_sd = prune_state_dict(entry["sd"], t)
            sparsities.append(state_dict_sparsity(pruned_sd))
            sd_gpu = {k: v.to(device) for k, v in pruned_sd.items()}
            load_lora_state_dict(model, sd_gpu)
            test = entry["test"]
            gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
            if check_passkey(gen, test["passkey"]):
                n_correct += 1
                per_type[test["type"]] += 1
        avg_sparsity = sum(sparsities) / len(sparsities)
        print(f"  {t:>10.0e}  {avg_sparsity*100:>9.1f}%  {n_correct:>3}/20 ({n_correct/20:>3.0%})  "
              f"{per_type['numeric']:>2}/5 {per_type['entity']:>2}/5 "
              f"{per_type['technical']:>2}/5 {per_type['fact']:>2}/5")
        pruning_results.append({
            "threshold": t,
            "avg_sparsity": avg_sparsity,
            "n_correct": n_correct,
            "per_type": per_type,
        })

    # Find the best threshold (highest sparsity that holds 20/20, then degraded)
    best_perfect = max((r for r in pruning_results if r["n_correct"] == 20),
                       key=lambda r: r["avg_sparsity"], default=None)
    if best_perfect is not None:
        print(f"\n  Highest threshold with 20/20: t={best_perfect['threshold']:.0e}, "
              f"sparsity {best_perfect['avg_sparsity']*100:.1f}%")
        best_threshold = best_perfect["threshold"]
    else:
        # Pick the highest threshold with the most retrievals
        best = max(pruning_results, key=lambda r: r["n_correct"])
        print(f"\n  No threshold reached 20/20. Best: t={best['threshold']:.0e}, "
              f"{best['n_correct']}/20")
        best_threshold = best["threshold"]

    with open(results_dir / "pruning_sweep.json", "w") as f:
        json.dump({
            "thresholds": THRESHOLDS,
            "results": pruning_results,
            "best_threshold": best_threshold,
        }, f, indent=2)

    # ============================================================
    # EXPERIMENT 3: Sparse storage size at best threshold
    # ============================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: SPARSE STORAGE SIZE")
    print(f"{'='*60}")
    print(f"  At best threshold t={best_threshold:.0e}\n")

    storage_per_format = {
        "dense_fp32":         0,
        "dense_int8":         0,
        "sparse_coo_int8":    0,
        "sparse_bitmap_int8": 0,
    }
    for entry in library:
        sd = entry["sd"]
        storage_per_format["dense_fp32"]         += dense_fp32_bytes(sd)
        storage_per_format["dense_int8"]         += dense_int8_bytes(sd)
        storage_per_format["sparse_coo_int8"]    += sparse_coo_int8_bytes(sd, best_threshold)
        storage_per_format["sparse_bitmap_int8"] += sparse_bitmap_int8_bytes(sd, best_threshold)

    n_adapters = len(library)
    print(f"  {'Format':>22}  {'Library (MB)':>14}  {'Per adapter (KB)':>18}  {'vs fp32':>10}")
    print(f"  {'-'*22}  {'-'*14}  {'-'*18}  {'-'*10}")
    fp32_baseline = storage_per_format["dense_fp32"]
    for fmt, total in storage_per_format.items():
        ratio = fp32_baseline / total if total > 0 else float("inf")
        print(f"  {fmt:>22}  {total/1e6:>13.2f}  {total/n_adapters/1024:>17.1f}   "
              f"{ratio:>8.1f}×")

    best_format = min(storage_per_format, key=storage_per_format.get)
    print(f"\n  Smallest format: {best_format}  "
          f"({storage_per_format[best_format]/1e6:.2f} MB total, "
          f"{fp32_baseline / storage_per_format[best_format]:.1f}× vs fp32)")

    with open(results_dir / "storage.json", "w") as f:
        json.dump({
            "best_threshold":  best_threshold,
            "storage_bytes":   storage_per_format,
            "n_adapters":      n_adapters,
            "compression_ratios": {
                fmt: fp32_baseline / total
                for fmt, total in storage_per_format.items()
            },
        }, f, indent=2)

    # ============================================================
    # EXPERIMENT 4: Combined compression stack — full pipeline
    # ============================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: COMBINED COMPRESSION STACK")
    print(f"{'='*60}")
    print(f"  Round-trip: rank 128 → prune at {best_threshold:.0e} → int8 → load")
    print(f"  Then run the full Phase 38b retrieval pipeline.\n")

    # Apply the full stack to every adapter in the library
    rt_library_sds = []
    for entry in library:
        rt_sd = round_trip_sparse_int8(entry["sd"], best_threshold)
        rt_library_sds.append(rt_sd)

    # ----- Same-prompt retrieval through routing -----
    def route(query_key):
        best_a, best_score = -1, -2.0
        for ai, keys in enumerate(library_keys):
            for k in keys:
                s = cosine(query_key, k)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    n_routed_sp, n_retr_sp = 0, 0
    for i, entry in enumerate(library):
        reset_lora_to_zero(model)
        ids = tokenizer.encode(entry["test"]["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
        best_a = route(q)
        if best_a == i:
            n_routed_sp += 1
        sd_gpu = {k: v.to(device) for k, v in rt_library_sds[best_a].items()}
        load_lora_state_dict(model, sd_gpu)
        gen = generate_greedy(model, entry["test"]["prompt"], tokenizer, device, 50)
        if check_passkey(gen, entry["test"]["passkey"]):
            n_retr_sp += 1

    print(f"  Same-prompt:           routing {n_routed_sp}/20 ({n_routed_sp/20:.0%})  "
          f"retrieval {n_retr_sp}/20 ({n_retr_sp/20:.0%})")

    # ----- Training-distribution paraphrase retrieval -----
    n_routed_td, n_retr_td = 0, 0
    for i, entry in enumerate(library):
        for para in train_paraphrase(entry["test"]):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
            best_a = route(q)
            if best_a == i:
                n_routed_td += 1
            sd_gpu = {k: v.to(device) for k, v in rt_library_sds[best_a].items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, para, tokenizer, device, 50)
            if check_passkey(gen, entry["test"]["passkey"]):
                n_retr_td += 1

    print(f"  Training-distribution: routing {n_routed_td}/60 ({n_routed_td/60:.0%})  "
          f"retrieval {n_retr_td}/60 ({n_retr_td/60:.0%})")

    # ----- Held-out paraphrase retrieval -----
    n_routed_ho, n_retr_ho = 0, 0
    for i, entry in enumerate(library):
        for para in held_out_paraphrase(entry["test"]):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
            best_a = route(q)
            if best_a == i:
                n_routed_ho += 1
            sd_gpu = {k: v.to(device) for k, v in rt_library_sds[best_a].items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, para, tokenizer, device, 50)
            if check_passkey(gen, entry["test"]["passkey"]):
                n_retr_ho += 1

    print(f"  Held-out:              routing {n_routed_ho}/60 ({n_routed_ho/60:.0%})  "
          f"retrieval {n_retr_ho}/60 ({n_retr_ho/60:.0%})")

    # ============================================================
    # Final summary
    # ============================================================
    rank512_fp32 = 20 * 10_485_760 * 4  # original prototype baseline, bytes
    final_bytes  = storage_per_format[best_format]
    total_compression = rank512_fp32 / final_bytes

    print(f"\n{'='*68}")
    print(f"PHASE 40 SUMMARY")
    print(f"{'='*68}")
    print(f"  Best pruning threshold:    {best_threshold:.0e}")
    print(f"  Average sparsity:          "
          f"{pruning_results[THRESHOLDS.index(best_threshold)]['avg_sparsity']*100:.1f}%")
    print(f"  Best storage format:       {best_format}")
    print(f"  20-passage library size:   {final_bytes/1e6:.2f} MB  "
          f"({final_bytes/n_adapters/1024:.1f} KB / adapter)")
    print()
    print(f"  Compression ratios:")
    print(f"    vs rank-128 fp32 (paper default):  "
          f"{fp32_baseline/final_bytes:.1f}×")
    print(f"    vs rank-512 fp32 (original):       {total_compression:.0f}×")
    print()
    print(f"  Phase 38b reference numbers (for comparison):")
    print(f"    same-prompt:           20/20")
    print(f"    training-distribution: 60/60")
    print(f"    held-out:              46/60")
    print(f"  Phase 40 round-trip numbers:")
    print(f"    same-prompt:           {n_retr_sp}/20")
    print(f"    training-distribution: {n_retr_td}/60")
    print(f"    held-out:              {n_retr_ho}/60")
    print()
    print(f"  1000-passage library at this format: ~{1000 * final_bytes / n_adapters / 1e9:.2f} GB")

    out = {
        "rank":              RANK,
        "best_threshold":    best_threshold,
        "best_format":       best_format,
        "library_bytes":     final_bytes,
        "compression_vs_rank128_fp32": fp32_baseline / final_bytes,
        "compression_vs_rank512_fp32": total_compression,
        "round_trip_retrieval": {
            "same_prompt":           {"routing": n_routed_sp, "retrieval": n_retr_sp, "n": 20},
            "training_distribution": {"routing": n_routed_td, "retrieval": n_retr_td, "n": 60},
            "held_out":              {"routing": n_routed_ho, "retrieval": n_retr_ho, "n": 60},
        },
        "phase38b_reference": {
            "same_prompt":           {"retrieval": 20, "n": 20},
            "training_distribution": {"retrieval": 60, "n": 60},
            "held_out":              {"retrieval": 46, "n": 60},
        },
    }
    with open(results_dir / "compression_stack.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
