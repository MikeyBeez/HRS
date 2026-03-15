"""MPAR Cross-Prompt Retrieval Experiment.

Verifies that mean-pooled mid-layer hidden states from GPT-2 are abstract
enough to bridge storage prompts and retrieval prompts that differ in surface form.

Extracts hidden states from GPT-2 small, mean pools across tokens to produce
MPAR vectors, then tests cross-prompt retrieval accuracy.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 50 paired prompts: (storage, retrieval, fact_token)
# ============================================================

PROMPT_PAIRS = [
    # Financial / shopping
    ("I bought $4.96 worth of groceries on August 2 2025.", "How much did I spend on groceries in August?", "$4.96"),
    ("The total for dinner at the restaurant was $78.50 including tip.", "What did dinner cost?", "$78.50"),
    ("My monthly rent payment is $2,350 due on the first.", "How much is the rent?", "$2,350"),
    ("The car repair estimate came back at $1,247.89.", "What will the car repair cost?", "$1,247.89"),
    ("I transferred $500 to the savings account on Monday.", "How much went to savings?", "$500"),
    # Medical
    ("The patient's blood pressure was 142 over 91 at the morning checkup.", "What was the blood pressure reading?", "142/91"),
    ("Her resting heart rate measured 72 beats per minute.", "What was the heart rate?", "72"),
    ("The lab results showed a cholesterol level of 218 mg/dL.", "What was the cholesterol level?", "218"),
    ("His blood glucose reading was 103 after fasting.", "What was the fasting glucose?", "103"),
    ("The prescription is for 20mg of lisinopril taken once daily.", "What is the medication dosage?", "20mg"),
    # Travel / codes
    ("The confirmation code for the flight is XK447B.", "What is the flight confirmation number?", "XK447B"),
    ("The hotel reservation number is HR-90215.", "What's the hotel booking reference?", "HR-90215"),
    ("The rental car pickup is at terminal 3 gate C.", "Where do I pick up the rental car?", "terminal 3 gate C"),
    ("The train departs from platform 7 at Central Station.", "Which platform does the train leave from?", "platform 7"),
    ("The Airbnb lockbox code is 4829.", "What's the code to get into the Airbnb?", "4829"),
    # Locations / offices
    ("Sarah's office is on the third floor room 312.", "Where does Sarah work?", "room 312"),
    ("The printer is located in the copy room on the second floor.", "Where is the printer?", "copy room second floor"),
    ("The parking garage entrance is on Oak Street.", "Where do you enter the parking garage?", "Oak Street"),
    ("The IT help desk is in building B room 105.", "Where is IT support located?", "building B room 105"),
    ("The emergency exit is through the stairwell at the east end of the hallway.", "Where's the emergency exit?", "east end stairwell"),
    # Dates / times
    ("The meeting starts at 2:15pm on Thursday March 19.", "When is the meeting scheduled?", "2:15pm Thursday March 19"),
    ("The project deadline is April 30 2025.", "When is the project due?", "April 30 2025"),
    ("The dentist appointment is Tuesday at 10:30am.", "When do I see the dentist?", "Tuesday 10:30am"),
    ("The concert is on Saturday June 7 at 8pm.", "When is the concert?", "Saturday June 7 8pm"),
    ("The lease expires on September 1 2026.", "When does the lease end?", "September 1 2026"),
    # People / contacts
    ("The project manager is David Chen and his extension is 4471.", "Who manages the project and how do I reach them?", "David Chen ext 4471"),
    ("The landlord's name is Margaret Thompson.", "Who is the landlord?", "Margaret Thompson"),
    ("The emergency contact is Dr. Patel at 555-0193.", "Who do I call in an emergency?", "Dr. Patel 555-0193"),
    ("The babysitter's number is 555-8824 and her name is Jessica.", "What's the babysitter's contact info?", "Jessica 555-8824"),
    ("The accountant handling the case is Robert Liu.", "Who is the accountant?", "Robert Liu"),
    # Technical / passwords / settings
    ("The WiFi password for the guest network is BlueSky2024.", "What's the WiFi password?", "BlueSky2024"),
    ("The database server IP address is 192.168.1.42.", "What is the database server address?", "192.168.1.42"),
    ("The API rate limit is 1000 requests per minute.", "What's the API rate limit?", "1000 per minute"),
    ("The SSH port for the staging server is 2222.", "What port do I use to SSH into staging?", "2222"),
    ("The admin panel login is username admin password TempPass99.", "What are the admin credentials?", "admin TempPass99"),
    # Measurements / quantities
    ("The living room dimensions are 14 by 18 feet.", "How big is the living room?", "14 by 18 feet"),
    ("The package weighs 3.7 kilograms.", "How much does the package weigh?", "3.7 kilograms"),
    ("The garden plot is 6 meters by 4 meters.", "What are the garden dimensions?", "6 by 4 meters"),
    ("The ceiling height in the loft is 9 feet 6 inches.", "How tall are the ceilings in the loft?", "9 feet 6 inches"),
    ("The tank capacity is 55 gallons.", "How much does the tank hold?", "55 gallons"),
    # Recipes / cooking
    ("The cake recipe calls for 2 and a quarter cups of flour.", "How much flour for the cake?", "2.25 cups"),
    ("Bake the casserole at 375 degrees for 45 minutes.", "What temperature and time for the casserole?", "375 degrees 45 minutes"),
    ("The marinade needs 3 tablespoons of soy sauce.", "How much soy sauce goes in the marinade?", "3 tablespoons"),
    ("Let the dough rise for 90 minutes in a warm place.", "How long does the dough need to rise?", "90 minutes"),
    ("The soup serves 8 people.", "How many servings does the soup make?", "8"),
    # Scores / statistics
    ("The final score was 24 to 17 in favor of the home team.", "What was the final score?", "24 to 17"),
    ("The student's GPA this semester is 3.74.", "What GPA did the student get?", "3.74"),
    ("The survey response rate was 43 percent.", "What percentage responded to the survey?", "43 percent"),
    ("The project completion rate stands at 87 percent.", "How far along is the project?", "87 percent"),
    ("The batting average for the season is .312.", "What's the batting average?", ".312"),
]


def extract_mpar(texts, model, tokenizer, layer, device, batch_size=16):
    """Extract mean-pooled hidden states from a specific layer.

    Args:
        texts: list of strings
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        layer: which layer to extract from (0-indexed)
        device: torch device
        batch_size: batch size for processing

    Returns:
        numpy array of shape (len(texts), hidden_dim)
    """
    model.eval()
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states is a tuple of (n_layers + 1) tensors, each (B, T, D)
        # index 0 = embeddings, index 1 = layer 1, etc.
        hidden = outputs.hidden_states[layer]  # (B, T, D)

        # Mask out padding tokens for mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (B, D)

        all_vectors.append(pooled.cpu().numpy())

    return np.concatenate(all_vectors, axis=0)


def cosine_distance_matrix(a, b):
    """Compute pairwise cosine distances between rows of a and b.

    Returns matrix D where D[i,j] = 1 - cosine_similarity(a[i], b[j]).
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    sim = a_norm @ b_norm.T
    return 1.0 - sim


def run_retrieval(storage_vecs, retrieval_vecs, n_pairs):
    """Run retrieval experiment and compute metrics.

    Args:
        storage_vecs: (N, D) numpy array of storage MPARs
        retrieval_vecs: (N, D) numpy array of retrieval MPARs
        n_pairs: number of pairs

    Returns:
        dict with metrics
    """
    dist_matrix = cosine_distance_matrix(retrieval_vecs, storage_vecs)  # (N, N)

    # Top-1 accuracy
    top1_predictions = np.argmin(dist_matrix, axis=1)
    top1_correct = sum(top1_predictions[i] == i for i in range(n_pairs))
    top1_acc = top1_correct / n_pairs

    # Top-3 accuracy
    top3_predictions = np.argsort(dist_matrix, axis=1)[:, :3]
    top3_correct = sum(i in top3_predictions[i] for i in range(n_pairs))
    top3_acc = top3_correct / n_pairs

    # Top-5 accuracy
    top5_predictions = np.argsort(dist_matrix, axis=1)[:, :5]
    top5_correct = sum(i in top5_predictions[i] for i in range(n_pairs))
    top5_acc = top5_correct / n_pairs

    # Distance analysis
    correct_distances = np.array([dist_matrix[i, i] for i in range(n_pairs)])

    nearest_incorrect_distances = []
    for i in range(n_pairs):
        row = dist_matrix[i].copy()
        row[i] = np.inf  # mask out correct
        nearest_incorrect_distances.append(row.min())
    nearest_incorrect_distances = np.array(nearest_incorrect_distances)

    # Separation ratio: correct / nearest_incorrect (< 1.0 means correct is closer)
    separation_ratios = correct_distances / (nearest_incorrect_distances + 1e-10)

    # Per-pair details
    per_pair = []
    for i in range(n_pairs):
        rank = int(np.where(np.argsort(dist_matrix[i]) == i)[0][0]) + 1
        per_pair.append({
            "pair_idx": i,
            "correct_dist": float(correct_distances[i]),
            "nearest_incorrect_dist": float(nearest_incorrect_distances[i]),
            "separation_ratio": float(separation_ratios[i]),
            "rank": rank,
            "retrieved_idx": int(top1_predictions[i]),
        })

    return {
        "top1_accuracy": float(top1_acc),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "mean_correct_distance": float(correct_distances.mean()),
        "std_correct_distance": float(correct_distances.std()),
        "mean_nearest_incorrect_distance": float(nearest_incorrect_distances.mean()),
        "std_nearest_incorrect_distance": float(nearest_incorrect_distances.std()),
        "mean_separation_ratio": float(separation_ratios.mean()),
        "median_separation_ratio": float(np.median(separation_ratios)),
        "fraction_separated": float((separation_ratios < 1.0).mean()),
        "per_pair": per_pair,
    }


def run_threshold_sweep(storage_vecs, retrieval_vecs, n_pairs):
    """Sweep distance thresholds and compute precision/recall.

    At each threshold: retrieve all stored items within that distance.
    Precision = correct retrievals / total retrievals.
    Recall = correct retrievals / total items that should be retrieved (= n_pairs).
    """
    dist_matrix = cosine_distance_matrix(retrieval_vecs, storage_vecs)
    thresholds = np.arange(0.05, 1.0, 0.05)

    results = []
    for thresh in thresholds:
        total_retrieved = 0
        correct_retrieved = 0

        for i in range(n_pairs):
            within_thresh = np.where(dist_matrix[i] < thresh)[0]
            total_retrieved += len(within_thresh)
            if i in within_thresh:
                correct_retrieved += 1

        precision = correct_retrieved / max(total_retrieved, 1)
        recall = correct_retrieved / n_pairs

        results.append({
            "threshold": float(thresh),
            "precision": float(precision),
            "recall": float(recall),
            "total_retrieved": int(total_retrieved),
            "correct_retrieved": int(correct_retrieved),
        })

    return results


def plot_precision_recall(threshold_results, layer, output_dir):
    """Plot precision-recall curve for a given layer."""
    thresholds = [r["threshold"] for r in threshold_results]
    precisions = [r["precision"] for r in threshold_results]
    recalls = [r["recall"] for r in threshold_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Precision-Recall curve
    ax1.plot(recalls, precisions, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Precision-Recall Curve (Layer {layer})', fontsize=13)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    # Annotate threshold values
    for t, p, r in zip(thresholds, precisions, recalls):
        if t in [0.1, 0.2, 0.3, 0.5, 0.7]:
            ax1.annotate(f't={t:.1f}', (r, p), textcoords="offset points",
                        xytext=(8, 5), fontsize=8, color='gray')

    # Precision and Recall vs Threshold
    ax2.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2, markersize=4)
    ax2.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2, markersize=4)
    ax2.set_xlabel('Distance Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'Precision & Recall vs Threshold (Layer {layer})', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"precision_recall_layer{layer}.png", dpi=150)
    plt.close()


def plot_layer_comparison(layer_results, output_dir):
    """Plot retrieval accuracy across layers."""
    layers = sorted(layer_results.keys())
    top1 = [layer_results[l]["top1_accuracy"] for l in layers]
    top3 = [layer_results[l]["top3_accuracy"] for l in layers]
    top5 = [layer_results[l]["top5_accuracy"] for l in layers]
    sep_ratios = [layer_results[l]["mean_separation_ratio"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(layers))
    width = 0.25

    ax1.bar(x - width, top1, width, label='Top-1', color='#2196F3')
    ax1.bar(x, top3, width, label='Top-3', color='#4CAF50')
    ax1.bar(x + width, top5, width, label='Top-5', color='#FF9800')
    ax1.axhline(y=1/50, color='gray', linestyle='--', alpha=0.5, label='Chance (2%)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Retrieval Accuracy by Layer', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in layers])
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (v1, v3, v5) in enumerate(zip(top1, top3, top5)):
        ax1.text(i - width, v1 + 0.02, f'{v1:.0%}', ha='center', fontsize=8)
        ax1.text(i, v3 + 0.02, f'{v3:.0%}', ha='center', fontsize=8)
        ax1.text(i + width, v5 + 0.02, f'{v5:.0%}', ha='center', fontsize=8)

    ax2.bar(x, sep_ratios, 0.5, color='#9C27B0')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No separation')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Mean Separation Ratio', fontsize=12)
    ax2.set_title('Separation Ratio by Layer (lower = better)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in layers])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(sep_ratios):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "layer_comparison.png", dpi=150)
    plt.close()


def plot_distance_distributions(storage_vecs, retrieval_vecs, layer, output_dir):
    """Plot distribution of correct vs incorrect distances."""
    n = len(storage_vecs)
    dist_matrix = cosine_distance_matrix(retrieval_vecs, storage_vecs)

    correct_dists = [dist_matrix[i, i] for i in range(n)]
    incorrect_dists = []
    for i in range(n):
        for j in range(n):
            if i != j:
                incorrect_dists.append(dist_matrix[i, j])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(incorrect_dists, bins=50, alpha=0.6, label='Incorrect pairs', color='#F44336', density=True)
    ax.hist(correct_dists, bins=20, alpha=0.8, label='Correct pairs', color='#2196F3', density=True)
    ax.set_xlabel('Cosine Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distance Distributions (Layer {layer})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"distance_distribution_layer{layer}.png", dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MPAR Cross-Prompt Retrieval")
    parser.add_argument("--output-dir", type=str, default="results/mpar_experiment")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9, 12])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load GPT-2 small
    print("Loading GPT-2 small...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Layers: {model.config.n_layer}, Hidden dim: {model.config.n_embd}")

    n_pairs = len(PROMPT_PAIRS)
    storage_prompts = [p[0] for p in PROMPT_PAIRS]
    retrieval_prompts = [p[1] for p in PROMPT_PAIRS]
    fact_tokens = [p[2] for p in PROMPT_PAIRS]
    print(f"  Prompt pairs: {n_pairs}")

    # Layer sweep
    layers_to_test = args.layers
    layer_results = {}
    layer_threshold_results = {}

    print(f"\nRunning layer sweep: {layers_to_test}")
    print("=" * 80)

    for layer in layers_to_test:
        print(f"\n--- Layer {layer} ---")
        t_layer = time.time()

        # Extract MPARs
        storage_vecs = extract_mpar(storage_prompts, model, tokenizer, layer, device)
        retrieval_vecs = extract_mpar(retrieval_prompts, model, tokenizer, layer, device)
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

        # Find best F1 threshold
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

        # Show worst retrieval failures
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
                print(f"      Storage: \"{storage_prompts[idx][:60]}...\"")
                print(f"      Retrieval: \"{retrieval_prompts[idx][:60]}...\"")
                print(f"      Retrieved instead: \"{storage_prompts[ret_idx][:60]}...\"")

        print(f"  Layer {layer} done in {time.time() - t_layer:.1f}s")

    # Layer comparison plot
    plot_layer_comparison(layer_results, output_dir)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
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

    # Save results
    all_results = {
        "n_pairs": n_pairs,
        "layers_tested": layers_to_test,
        "model": "gpt2",
        "hidden_dim": model.config.n_embd,
        "device": str(device),
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "threshold_sweeps": {str(k): v for k, v in layer_threshold_results.items()},
        "prompt_pairs": [
            {"storage": s, "retrieval": r, "fact": f}
            for s, r, f in PROMPT_PAIRS
        ],
    }

    # Remove per_pair detail from top-level for cleaner JSON
    summary = {
        "n_pairs": n_pairs,
        "model": "gpt2",
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

    print(f"\nResults saved to {output_dir}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
