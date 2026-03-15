"""MPAR Cross-Prompt Retrieval v3 — 500 pairs, 10 domains, full evaluation.

Extends the v2 benchmark from 50 to 500 prompt pairs across diverse domains.
Uses identical MPAR extraction logic from the v2 experiment.
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpar_prompts_v3 import PROMPT_PAIRS_V3, DOMAIN_LABELS
from mpar_experiment import (
    cosine_distance_matrix,
    run_retrieval,
    run_threshold_sweep,
    plot_precision_recall,
    plot_distance_distributions,
)
from mpar_experiment_7b import load_model_4bit, extract_mpar_7b


def run_domain_breakdown(storage_vecs, retrieval_vecs, domain_labels, n_pairs):
    """Per-domain retrieval accuracy and separation ratio."""
    dist_matrix = cosine_distance_matrix(retrieval_vecs, storage_vecs)
    top1_predictions = np.argmin(dist_matrix, axis=1)

    domains = sorted(set(domain_labels))
    results = {}

    for domain in domains:
        indices = [i for i, d in enumerate(domain_labels) if d == domain]
        n = len(indices)

        correct = sum(1 for i in indices if top1_predictions[i] == i)
        top1_acc = correct / n

        top3_preds = np.argsort(dist_matrix, axis=1)[:, :3]
        top3_correct = sum(1 for i in indices if i in top3_preds[i])
        top3_acc = top3_correct / n

        top5_preds = np.argsort(dist_matrix, axis=1)[:, :5]
        top5_correct = sum(1 for i in indices if i in top5_preds[i])
        top5_acc = top5_correct / n

        correct_dists = [dist_matrix[i, i] for i in indices]
        incorrect_dists = []
        for i in indices:
            row = dist_matrix[i].copy()
            row[i] = np.inf
            incorrect_dists.append(row.min())

        sep_ratios = [c / (ic + 1e-10) for c, ic in zip(correct_dists, incorrect_dists)]

        results[domain] = {
            "n_pairs": n,
            "top1_accuracy": float(top1_acc),
            "top3_accuracy": float(top3_acc),
            "top5_accuracy": float(top5_acc),
            "mean_correct_distance": float(np.mean(correct_dists)),
            "mean_nearest_incorrect_distance": float(np.mean(incorrect_dists)),
            "mean_separation_ratio": float(np.mean(sep_ratios)),
            "fraction_separated": float(np.mean([1.0 if s < 1.0 else 0.0 for s in sep_ratios])),
        }

    return results


def find_hardest_pairs(storage_vecs, retrieval_vecs, storage_prompts, retrieval_prompts,
                       domain_labels, n_worst=5):
    """Find the pairs with the worst (highest) separation ratios."""
    dist_matrix = cosine_distance_matrix(retrieval_vecs, storage_vecs)
    top1_predictions = np.argmin(dist_matrix, axis=1)
    n = len(storage_prompts)

    pairs = []
    for i in range(n):
        correct_dist = dist_matrix[i, i]
        row = dist_matrix[i].copy()
        row[i] = np.inf
        nearest_incorrect_dist = row.min()
        sep = correct_dist / (nearest_incorrect_dist + 1e-10)
        rank = int(np.where(np.argsort(dist_matrix[i]) == i)[0][0]) + 1

        pairs.append({
            "idx": i,
            "domain": domain_labels[i],
            "rank": rank,
            "separation_ratio": float(sep),
            "correct_dist": float(correct_dist),
            "nearest_incorrect_dist": float(nearest_incorrect_dist),
            "storage": storage_prompts[i],
            "retrieval": retrieval_prompts[i],
            "retrieved_storage": storage_prompts[top1_predictions[i]],
            "retrieved_domain": domain_labels[top1_predictions[i]],
        })

    pairs.sort(key=lambda x: x["separation_ratio"], reverse=True)
    return pairs[:n_worst]


def plot_domain_comparison(domain_results, output_dir, layer):
    """Bar chart of per-domain accuracy and separation ratio."""
    domains = sorted(domain_results.keys())
    top1 = [domain_results[d]["top1_accuracy"] for d in domains]
    sep = [domain_results[d]["mean_separation_ratio"] for d in domains]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(domains))
    ax1.bar(x, top1, color='#2196F3', alpha=0.8)
    ax1.set_xlabel('Domain', fontsize=11)
    ax1.set_ylabel('Top-1 Accuracy', fontsize=11)
    ax1.set_title(f'Per-Domain Top-1 Accuracy (Layer {layer})', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(top1):
        ax1.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=8)

    ax2.bar(x, sep, color='#9C27B0', alpha=0.8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No separation')
    ax2.set_xlabel('Domain', fontsize=11)
    ax2.set_ylabel('Mean Separation Ratio', fontsize=11)
    ax2.set_title(f'Per-Domain Separation Ratio (Layer {layer}, lower=better)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sep):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f"domain_breakdown_layer{layer}.png", dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MPAR 7B v3 — 500 pairs, 10 domains")
    parser.add_argument("--output-dir", type=str, default="results/mpar_experiment_7b_v3")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 31])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    device = torch.device("cuda")
    model, tokenizer, n_layers, hidden_dim = load_model_4bit(args.model, device)

    layers_to_test = args.layers
    print(f"  Testing layers: {layers_to_test} (of {n_layers})")

    n_pairs = len(PROMPT_PAIRS_V3)
    storage_prompts = [p[0] for p in PROMPT_PAIRS_V3]
    retrieval_prompts = [p[1] for p in PROMPT_PAIRS_V3]
    fact_tokens = [p[2] for p in PROMPT_PAIRS_V3]
    print(f"  Prompt pairs: {n_pairs} (v3 — 10 domains)")

    # Validate prompt lengths
    s_lens = [len(p.split()) for p in storage_prompts]
    r_lens = [len(p.split()) for p in retrieval_prompts]
    print(f"  Storage words: min={min(s_lens)}, max={max(s_lens)}, mean={np.mean(s_lens):.0f}")
    print(f"  Retrieval words: min={min(r_lens)}, max={max(r_lens)}, mean={np.mean(r_lens):.0f}")

    s_violations = sum(1 for l in s_lens if l < 15 or l > 30)
    r_violations = sum(1 for l in r_lens if l < 10 or l > 25)
    if s_violations:
        print(f"  WARNING: {s_violations} storage prompts outside 15-30 word range")
    if r_violations:
        print(f"  WARNING: {r_violations} retrieval prompts outside 10-25 word range")

    layer_results = {}
    layer_domain_results = {}
    layer_threshold_results = {}
    layer_hardest = {}

    print(f"\nRunning layer sweep: {layers_to_test}")
    print("=" * 90)

    for layer in layers_to_test:
        print(f"\n--- Layer {layer} ---")
        t_layer = time.time()

        storage_vecs = extract_mpar_7b(
            storage_prompts, model, tokenizer, layer, batch_size=args.batch_size)
        retrieval_vecs = extract_mpar_7b(
            retrieval_prompts, model, tokenizer, layer, batch_size=args.batch_size)
        print(f"  MPAR shape: {storage_vecs.shape}")

        # Overall metrics
        results = run_retrieval(storage_vecs, retrieval_vecs, n_pairs)
        layer_results[layer] = results

        print(f"  Top-1: {results['top1_accuracy']:.1%} | "
              f"Top-3: {results['top3_accuracy']:.1%} | "
              f"Top-5: {results['top5_accuracy']:.1%}")
        print(f"  Correct dist: {results['mean_correct_distance']:.4f} | "
              f"Incorrect dist: {results['mean_nearest_incorrect_distance']:.4f} | "
              f"Sep ratio: {results['mean_separation_ratio']:.4f} | "
              f"Frac sep: {results['fraction_separated']:.1%}")

        # Domain breakdown
        domain_results = run_domain_breakdown(
            storage_vecs, retrieval_vecs, DOMAIN_LABELS, n_pairs)
        layer_domain_results[layer] = domain_results

        print(f"\n  {'Domain':<12} | {'Top-1':>6} | {'Top-3':>6} | {'Top-5':>6} | {'Sep':>7} | {'Frac':>6}")
        print(f"  {'-'*55}")
        for domain in sorted(domain_results.keys()):
            d = domain_results[domain]
            print(f"  {domain:<12} | {d['top1_accuracy']:>5.0%} | {d['top3_accuracy']:>5.0%} | "
                  f"{d['top5_accuracy']:>5.0%} | {d['mean_separation_ratio']:>7.3f} | "
                  f"{d['fraction_separated']:>5.0%}")

        # Threshold sweep
        threshold_results = run_threshold_sweep(storage_vecs, retrieval_vecs, n_pairs)
        layer_threshold_results[layer] = threshold_results

        best_f1 = 0
        best_info = None
        for tr in threshold_results:
            p, r = tr["precision"], tr["recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_info = {**tr, "f1": f1}
        if best_info:
            print(f"\n  Best F1: thresh={best_info['threshold']:.2f} "
                  f"P={best_info['precision']:.2f} R={best_info['recall']:.2f} "
                  f"F1={best_info['f1']:.2f}")

        # Hardest pairs
        hardest = find_hardest_pairs(
            storage_vecs, retrieval_vecs, storage_prompts, retrieval_prompts,
            DOMAIN_LABELS, n_worst=5)
        layer_hardest[layer] = hardest

        print(f"\n  Hardest pairs:")
        for h in hardest:
            print(f"    [{h['domain']}] rank={h['rank']}, sep={h['separation_ratio']:.3f}")
            print(f"      S: \"{h['storage'][:80]}\"")
            print(f"      R: \"{h['retrieval'][:80]}\"")
            print(f"      Got [{h['retrieved_domain']}]: \"{h['retrieved_storage'][:80]}\"")

        # Plots
        plot_precision_recall(threshold_results, layer, output_dir)
        plot_distance_distributions(storage_vecs, retrieval_vecs, layer, output_dir)
        plot_domain_comparison(domain_results, output_dir, layer)

        elapsed = time.time() - t_layer
        print(f"\n  Layer {layer} done in {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 90)
    print(f"SUMMARY — {args.model} — v3 (500 pairs, 10 domains)")
    print("=" * 90)
    print(f"{'Layer':>6} | {'Top-1':>7} | {'Top-3':>7} | {'Top-5':>7} | "
          f"{'Correct':>8} | {'Incorrect':>10} | {'Sep Ratio':>10} | {'Frac Sep':>9}")
    print("-" * 90)
    for layer in sorted(layer_results.keys()):
        r = layer_results[layer]
        print(f"{layer:>6} | {r['top1_accuracy']:>6.1%} | {r['top3_accuracy']:>6.1%} | "
              f"{r['top5_accuracy']:>6.1%} | {r['mean_correct_distance']:>8.4f} | "
              f"{r['mean_nearest_incorrect_distance']:>10.4f} | "
              f"{r['mean_separation_ratio']:>10.4f} | {r['fraction_separated']:>8.1%}")
    print("-" * 90)
    print(f"Chance: {1/n_pairs:.1%} (top-1), {3/n_pairs:.1%} (top-3), {5/n_pairs:.1%} (top-5)")

    # Compare to v2
    v2_path = Path("results/mpar_experiment_7b_v2/summary.json")
    if v2_path.exists():
        with open(v2_path) as f:
            v2 = json.load(f)
        print("\n" + "=" * 90)
        print("COMPARISON: v2 (50 pairs) vs v3 (500 pairs)")
        print("=" * 90)
        print(f"{'Version':>10} | {'N':>4} | {'Layer':>6} | {'Top-1':>7} | {'Top-3':>7} | "
              f"{'Top-5':>7} | {'Sep Ratio':>10}")
        print("-" * 70)
        v2_best = max(v2["layer_summary"].items(), key=lambda x: x[1]["top1_accuracy"])
        g = v2_best[1]
        print(f"{'v2':>10} | {'50':>4} | {v2_best[0]:>6} | {g['top1_accuracy']:>6.1%} | "
              f"{g['top3_accuracy']:>6.1%} | {g['top5_accuracy']:>6.1%} | "
              f"{g['mean_separation_ratio']:>10.4f}")
        best_v3_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["top1_accuracy"])
        r = layer_results[best_v3_layer]
        print(f"{'v3':>10} | {'500':>4} | {best_v3_layer:>6} | {r['top1_accuracy']:>6.1%} | "
              f"{r['top3_accuracy']:>6.1%} | {r['top5_accuracy']:>6.1%} | "
              f"{r['mean_separation_ratio']:>10.4f}")
        print("-" * 70)

    # Save results
    # Strip per_pair from layer_results to keep JSON manageable
    layer_results_summary = {}
    for layer, r in layer_results.items():
        layer_results_summary[str(layer)] = {k: v for k, v in r.items() if k != "per_pair"}

    all_results = {
        "n_pairs": n_pairs,
        "n_domains": 10,
        "layers_tested": layers_to_test,
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "quantization": "4-bit NF4",
        "prompt_version": "v3",
        "layer_results": layer_results_summary,
        "domain_results": {str(k): v for k, v in layer_domain_results.items()},
        "threshold_sweeps": {str(k): v for k, v in layer_threshold_results.items()},
        "hardest_pairs": {str(k): v for k, v in layer_hardest.items()},
        "per_pair_scores": {
            str(layer): results["per_pair"]
            for layer, results in layer_results.items()
        },
    }

    with open(output_dir / "mpar_results_v3.json", "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nResults saved to {output_dir}/mpar_results_v3.json")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
