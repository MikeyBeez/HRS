"""MPAR Cross-Prompt Retrieval at 7B Scale — v2 prompts.

Same Mistral-7B 4-bit setup, redesigned prompts with richer context
on both sides to test whether the remaining gap is prompt design or model scale.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpar_prompts_v2 import PROMPT_PAIRS_V2
from mpar_experiment import (
    cosine_distance_matrix,
    run_retrieval,
    run_threshold_sweep,
    plot_precision_recall,
    plot_layer_comparison,
    plot_distance_distributions,
)
from mpar_experiment_7b import load_model_4bit, extract_mpar_7b


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MPAR 7B v2 prompts")
    parser.add_argument("--output-dir", type=str, default="results/mpar_experiment_7b_v2")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    device = torch.device("cuda")
    model, tokenizer, n_layers, hidden_dim = load_model_4bit(args.model, device)

    if args.layers:
        layers_to_test = args.layers
    else:
        layers_to_test = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    print(f"  Testing layers: {layers_to_test} (of {n_layers})")

    n_pairs = len(PROMPT_PAIRS_V2)
    storage_prompts = [p[0] for p in PROMPT_PAIRS_V2]
    retrieval_prompts = [p[1] for p in PROMPT_PAIRS_V2]
    fact_tokens = [p[2] for p in PROMPT_PAIRS_V2]
    print(f"  Prompt pairs: {n_pairs} (v2 — enriched context)")

    # Check prompt length stats
    s_lens = [len(p.split()) for p in storage_prompts]
    r_lens = [len(p.split()) for p in retrieval_prompts]
    print(f"  Storage prompt words: min={min(s_lens)}, max={max(s_lens)}, mean={np.mean(s_lens):.0f}")
    print(f"  Retrieval prompt words: min={min(r_lens)}, max={max(r_lens)}, mean={np.mean(r_lens):.0f}")

    layer_results = {}
    layer_threshold_results = {}

    print(f"\nRunning layer sweep: {layers_to_test}")
    print("=" * 80)

    for layer in layers_to_test:
        print(f"\n--- Layer {layer} ---")
        t_layer = time.time()

        storage_vecs = extract_mpar_7b(
            storage_prompts, model, tokenizer, layer, batch_size=args.batch_size)
        retrieval_vecs = extract_mpar_7b(
            retrieval_prompts, model, tokenizer, layer, batch_size=args.batch_size)
        print(f"  MPAR shape: {storage_vecs.shape}")

        results = run_retrieval(storage_vecs, retrieval_vecs, n_pairs)
        layer_results[layer] = results

        print(f"  Top-1 accuracy: {results['top1_accuracy']:.1%}")
        print(f"  Top-3 accuracy: {results['top3_accuracy']:.1%}")
        print(f"  Top-5 accuracy: {results['top5_accuracy']:.1%}")
        print(f"  Mean correct distance: {results['mean_correct_distance']:.4f}")
        print(f"  Mean nearest incorrect: {results['mean_nearest_incorrect_distance']:.4f}")
        print(f"  Mean separation ratio: {results['mean_separation_ratio']:.4f}")
        print(f"  Fraction separated (<1.0): {results['fraction_separated']:.1%}")

        threshold_results = run_threshold_sweep(storage_vecs, retrieval_vecs, n_pairs)
        layer_threshold_results[layer] = threshold_results

        best_f1 = 0
        best_thresh_info = None
        for tr in threshold_results:
            p, r = tr["precision"], tr["recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh_info = {**tr, "f1": f1}
        if best_thresh_info:
            print(f"  Best F1 threshold: {best_thresh_info['threshold']:.2f} "
                  f"(P={best_thresh_info['precision']:.2f}, R={best_thresh_info['recall']:.2f}, "
                  f"F1={best_thresh_info['f1']:.2f})")

        plot_precision_recall(threshold_results, layer, output_dir)
        plot_distance_distributions(storage_vecs, retrieval_vecs, layer, output_dir)

        # Show worst failures
        failures = [p for p in results["per_pair"] if p["rank"] > 1]
        if failures:
            failures.sort(key=lambda x: x["rank"], reverse=True)
            print(f"  Failures ({len(failures)}):")
            for f in failures[:5]:
                idx = f["pair_idx"]
                ret_idx = f["retrieved_idx"]
                print(f"    Pair {idx}: rank={f['rank']}, sep={f['separation_ratio']:.3f}")
                print(f"      Storage:   \"{storage_prompts[idx][:80]}\"")
                print(f"      Retrieval: \"{retrieval_prompts[idx][:80]}\"")
                print(f"      Got:       \"{storage_prompts[ret_idx][:80]}\"")

        print(f"  Layer {layer} done in {time.time() - t_layer:.1f}s")

    plot_layer_comparison(layer_results, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY — {args.model} — v2 prompts")
    print("=" * 80)
    print(f"{'Layer':>6} | {'Top-1':>7} | {'Top-3':>7} | {'Top-5':>7} | "
          f"{'Correct':>8} | {'Incorrect':>10} | {'Sep Ratio':>10} | {'Frac Sep':>9}")
    print("-" * 80)
    for layer in sorted(layer_results.keys()):
        r = layer_results[layer]
        print(f"{layer:>6} | {r['top1_accuracy']:>6.1%} | {r['top3_accuracy']:>6.1%} | "
              f"{r['top5_accuracy']:>6.1%} | {r['mean_correct_distance']:>8.4f} | "
              f"{r['mean_nearest_incorrect_distance']:>10.4f} | "
              f"{r['mean_separation_ratio']:>10.4f} | {r['fraction_separated']:>8.1%}")
    print("-" * 80)
    print(f"Chance: {1/n_pairs:.1%} (top-1), {3/n_pairs:.1%} (top-3), {5/n_pairs:.1%} (top-5)")

    # Compare against v1 results
    v1_path = Path("results/mpar_experiment_7b/summary.json")
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        print("\n" + "=" * 80)
        print("COMPARISON: v1 prompts vs v2 prompts (7B, best layer each)")
        print("=" * 80)
        print(f"{'Prompts':>12} | {'Layer':>6} | {'Top-1':>7} | {'Top-3':>7} | "
              f"{'Top-5':>7} | {'Sep Ratio':>10} | {'Frac Sep':>9}")
        print("-" * 80)
        v1_best = max(v1["layer_summary"].items(), key=lambda x: x[1]["top1_accuracy"])
        g = v1_best[1]
        print(f"{'v1 (orig)':>12} | {v1_best[0]:>6} | {g['top1_accuracy']:>6.1%} | "
              f"{g['top3_accuracy']:>6.1%} | {g['top5_accuracy']:>6.1%} | "
              f"{g['mean_separation_ratio']:>10.4f} | {g['fraction_separated']:>8.1%}")
        best_v2_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["top1_accuracy"])
        r = layer_results[best_v2_layer]
        print(f"{'v2 (fixed)':>12} | {best_v2_layer:>6} | {r['top1_accuracy']:>6.1%} | "
              f"{r['top3_accuracy']:>6.1%} | {r['top5_accuracy']:>6.1%} | "
              f"{r['mean_separation_ratio']:>10.4f} | {r['fraction_separated']:>8.1%}")
        print("-" * 80)

    # Save
    all_results = {
        "n_pairs": n_pairs,
        "layers_tested": layers_to_test,
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "quantization": "4-bit NF4",
        "prompt_version": "v2",
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "threshold_sweeps": {str(k): v for k, v in layer_threshold_results.items()},
        "prompt_pairs": [
            {"storage": s, "retrieval": r, "fact": f}
            for s, r, f in PROMPT_PAIRS_V2
        ],
    }

    summary = {
        "n_pairs": n_pairs,
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "prompt_version": "v2",
        "layer_summary": {},
    }
    for layer in sorted(layer_results.keys()):
        r = layer_results[layer]
        summary["layer_summary"][str(layer)] = {
            "top1_accuracy": r["top1_accuracy"],
            "top3_accuracy": r["top3_accuracy"],
            "top5_accuracy": r["top5_accuracy"],
            "mean_correct_distance": r["mean_correct_distance"],
            "mean_nearest_incorrect_distance": r["mean_nearest_incorrect_distance"],
            "mean_separation_ratio": r["mean_separation_ratio"],
            "fraction_separated": r["fraction_separated"],
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nResults saved to {output_dir}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
