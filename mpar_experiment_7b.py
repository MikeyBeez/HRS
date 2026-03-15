"""MPAR Cross-Prompt Retrieval at 7B Scale.

Same 50 prompt pairs as the GPT-2 experiment, but using Mistral-7B-Instruct-v0.3
loaded in 4-bit quantization. Tests whether larger models produce representations
abstract enough for reliable cross-prompt retrieval.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse exact prompt pairs from the GPT-2 experiment
from mpar_experiment import (
    PROMPT_PAIRS,
    cosine_distance_matrix,
    run_retrieval,
    run_threshold_sweep,
    plot_precision_recall,
    plot_layer_comparison,
    plot_distance_distributions,
)


def load_model_4bit(model_name, device):
    """Load a model in 4-bit quantization with bitsandbytes."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_name} in 4-bit...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    elapsed = time.time() - t0
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return model, tokenizer, n_layers, hidden_dim


def extract_mpar_7b(texts, model, tokenizer, layer, batch_size=4):
    """Extract mean-pooled hidden states from a specific layer of a 7B model.

    Processes in small batches to avoid OOM. Uses the model's device_map
    so tensors go where the model expects them.
    """
    model.eval()
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
        # Move to the device the model expects (device_map handles distribution)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers + 1) tensors
        # index 0 = embeddings, index 1 = layer 1, etc.
        hidden = outputs.hidden_states[layer].float()  # (B, T, D)

        # Mask out padding for mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (B, D)

        all_vectors.append(pooled.cpu().numpy())

        # Free memory
        del outputs, hidden, pooled
        torch.cuda.empty_cache()

    return np.concatenate(all_vectors, axis=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MPAR 7B Cross-Prompt Retrieval")
    parser.add_argument("--output-dir", type=str, default="results/mpar_experiment_7b")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to test (default: quarter/half/3quarter/final)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Load model
    device = torch.device("cuda")
    model, tokenizer, n_layers, hidden_dim = load_model_4bit(args.model, device)

    # Determine layers to test
    if args.layers:
        layers_to_test = args.layers
    else:
        # Quarter, half, three-quarter, final
        layers_to_test = [
            n_layers // 4,
            n_layers // 2,
            3 * n_layers // 4,
            n_layers,
        ]
    print(f"  Testing layers: {layers_to_test} (of {n_layers})")

    n_pairs = len(PROMPT_PAIRS)
    storage_prompts = [p[0] for p in PROMPT_PAIRS]
    retrieval_prompts = [p[1] for p in PROMPT_PAIRS]
    fact_tokens = [p[2] for p in PROMPT_PAIRS]
    print(f"  Prompt pairs: {n_pairs}")

    # Layer sweep
    layer_results = {}
    layer_threshold_results = {}

    print(f"\nRunning layer sweep: {layers_to_test}")
    print("=" * 80)

    for layer in layers_to_test:
        print(f"\n--- Layer {layer} ---")
        t_layer = time.time()

        # Extract MPARs
        storage_vecs = extract_mpar_7b(
            storage_prompts, model, tokenizer, layer, batch_size=args.batch_size)
        retrieval_vecs = extract_mpar_7b(
            retrieval_prompts, model, tokenizer, layer, batch_size=args.batch_size)
        print(f"  MPAR shape: {storage_vecs.shape}")

        # Run retrieval
        results = run_retrieval(storage_vecs, retrieval_vecs, n_pairs)
        layer_results[layer] = results

        print(f"  Top-1 accuracy: {results['top1_accuracy']:.1%}")
        print(f"  Top-3 accuracy: {results['top3_accuracy']:.1%}")
        print(f"  Top-5 accuracy: {results['top5_accuracy']:.1%}")
        print(f"  Mean correct distance: {results['mean_correct_distance']:.4f}")
        print(f"  Mean nearest incorrect: {results['mean_nearest_incorrect_distance']:.4f}")
        print(f"  Mean separation ratio: {results['mean_separation_ratio']:.4f}")
        print(f"  Fraction separated (<1.0): {results['fraction_separated']:.1%}")

        # Threshold sweep
        threshold_results = run_threshold_sweep(storage_vecs, retrieval_vecs, n_pairs)
        layer_threshold_results[layer] = threshold_results

        # Best F1
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

        # Plots
        plot_precision_recall(threshold_results, layer, output_dir)
        plot_distance_distributions(storage_vecs, retrieval_vecs, layer, output_dir)

        # Worst failures
        failures = [p for p in results["per_pair"] if p["rank"] > 1]
        if failures:
            failures.sort(key=lambda x: x["rank"], reverse=True)
            print(f"  Retrieval failures ({len(failures)}):")
            for f in failures[:5]:
                idx = f["pair_idx"]
                ret_idx = f["retrieved_idx"]
                print(f"    Pair {idx}: rank={f['rank']}, "
                      f"dist={f['correct_dist']:.4f}, "
                      f"sep={f['separation_ratio']:.3f}")
                print(f"      Storage:   \"{storage_prompts[idx][:70]}\"")
                print(f"      Retrieval: \"{retrieval_prompts[idx][:70]}\"")
                print(f"      Got:       \"{storage_prompts[ret_idx][:70]}\"")

        print(f"  Layer {layer} done in {time.time() - t_layer:.1f}s")

    # Layer comparison plot
    plot_layer_comparison(layer_results, output_dir)

    # Summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY TABLE — {args.model}")
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
    print(f"Chance level: {1/n_pairs:.1%} (top-1), {3/n_pairs:.1%} (top-3), {5/n_pairs:.1%} (top-5)")

    # GPT-2 comparison
    gpt2_results_path = Path("results/mpar_experiment/summary.json")
    if gpt2_results_path.exists():
        with open(gpt2_results_path) as f:
            gpt2_summary = json.load(f)
        print("\n" + "=" * 80)
        print("COMPARISON: GPT-2 vs 7B")
        print("=" * 80)
        print(f"{'Model':>20} | {'Layer':>6} | {'Top-1':>7} | {'Top-3':>7} | "
              f"{'Top-5':>7} | {'Sep Ratio':>10} | {'Frac Sep':>9}")
        print("-" * 80)
        # Best GPT-2 layer
        gpt2_best = max(gpt2_summary["layer_summary"].items(),
                       key=lambda x: x[1]["top1_accuracy"])
        g = gpt2_best[1]
        print(f"{'GPT-2 (best)':>20} | {gpt2_best[0]:>6} | {g['top1_accuracy']:>6.1%} | "
              f"{g['top3_accuracy']:>6.1%} | {g['top5_accuracy']:>6.1%} | "
              f"{g['mean_separation_ratio']:>10.4f} | {g['fraction_separated']:>8.1%}")
        # Best 7B layer
        best_7b_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["top1_accuracy"])
        r = layer_results[best_7b_layer]
        print(f"{'7B (best)':>20} | {best_7b_layer:>6} | {r['top1_accuracy']:>6.1%} | "
              f"{r['top3_accuracy']:>6.1%} | {r['top5_accuracy']:>6.1%} | "
              f"{r['mean_separation_ratio']:>10.4f} | {r['fraction_separated']:>8.1%}")
        print("-" * 80)

    # Save results
    all_results = {
        "n_pairs": n_pairs,
        "layers_tested": layers_to_test,
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "quantization": "4-bit NF4",
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "threshold_sweeps": {str(k): v for k, v in layer_threshold_results.items()},
        "prompt_pairs": [
            {"storage": s, "retrieval": r, "fact": f}
            for s, r, f in PROMPT_PAIRS
        ],
    }

    summary = {
        "n_pairs": n_pairs,
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
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
