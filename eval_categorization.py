"""Evaluate V18's categorization head accuracy.

Runs WikiText-103 validation and test sets through V18, compares the
categorization head's predictions against ground-truth TF-IDF cluster labels.

Reports: overall accuracy, per-category precision/recall/F1, confusion matrix.

Usage:
    python eval_categorization.py [--split validation] [--device cuda]
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import _extract_articles, _cluster_articles, load_wikitext, _build_category_ids


def load_v18_model(device):
    """Load trained V18 model."""
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt_path = Path("results/v18_cross_attn/best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    print(f"Loaded V18 (step {ckpt.get('step', '?')}, val_ppl {ckpt.get('val_ppl', '?'):.2f})")
    return model, cfg


def get_ground_truth_labels(split_name="validation", n_categories=50):
    """Get ground-truth category labels for a WikiText-103 split.

    Fits TF-IDF + KMeans on training articles, then predicts labels for
    the target split articles using the same model (transform + predict).

    Returns:
        train_articles, train_labels, split_articles, split_labels
    """
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans

    raw = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Extract articles
    print("Extracting training articles...")
    train_articles = _extract_articles(raw["train"]["text"])
    split_articles = _extract_articles(raw[split_name]["text"])
    print(f"Found {len(train_articles)} train, {len(split_articles)} {split_name} articles")

    # Fit TF-IDF + KMeans on training set
    print(f"Clustering into {n_categories} categories...")
    train_texts = [text for _, text in train_articles]
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        stop_words='english', max_df=0.95, min_df=2,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    kmeans = MiniBatchKMeans(n_clusters=n_categories, random_state=42, batch_size=1000)
    train_labels = kmeans.fit_predict(train_tfidf).tolist()

    # Predict labels for split articles using the same fitted model
    split_texts = [text for _, text in split_articles]
    split_tfidf = vectorizer.transform(split_texts)
    split_labels = kmeans.predict(split_tfidf).tolist()
    print(f"Assigned {len(set(split_labels))} unique categories to {split_name} articles")

    return train_articles, train_labels, split_articles, split_labels


@torch.no_grad()
def predict_categories(model, tokenizer, articles, device, max_seq_len=512):
    """Run articles through V18 and get categorization head predictions.

    Args:
        model: V18 model
        tokenizer: GPT-2 tokenizer
        articles: list of (title, text)
        device: torch device
        max_seq_len: max tokens per forward pass

    Returns:
        list of (predicted_category, confidence, title) tuples
    """
    model.eval()
    predictions = []

    for i, (title, text) in enumerate(articles):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 16:
            predictions.append((-1, 0.0, title))  # skip very short
            continue

        # Use first max_seq_len tokens (matching training)
        ids = ids[:max_seq_len]
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        output = model(input_ids, step=0)
        cat_logits = output.categorization_logits  # (1, num_categories)
        probs = F.softmax(cat_logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        conf = probs[0, pred].item()

        predictions.append((pred, conf, title))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(articles)} articles processed")

    return predictions


def evaluate(predictions, articles, title_to_label, n_categories=50):
    """Compute accuracy, per-category metrics, and confusion matrix."""
    correct = 0
    total = 0
    per_cat_tp = Counter()
    per_cat_fp = Counter()
    per_cat_fn = Counter()
    per_cat_total = Counter()
    confusion = defaultdict(Counter)  # confusion[true][pred] = count

    results_per_article = []

    for (pred, conf, title), (art_title, _) in zip(predictions, articles):
        if pred == -1:
            continue  # skipped

        true_label = title_to_label.get(art_title, title_to_label.get(title, -1))
        if true_label == -1:
            continue  # article not in training set clustering

        total += 1
        per_cat_total[true_label] += 1
        confusion[true_label][pred] += 1

        if pred == true_label:
            correct += 1
            per_cat_tp[true_label] += 1
        else:
            per_cat_fp[pred] += 1
            per_cat_fn[true_label] += 1

        results_per_article.append({
            "title": title,
            "true": true_label,
            "pred": pred,
            "correct": pred == true_label,
            "confidence": conf,
        })

    accuracy = correct / total if total > 0 else 0

    # Per-category precision, recall, F1
    per_cat_metrics = {}
    for cat in range(n_categories):
        tp = per_cat_tp[cat]
        fp = per_cat_fp[cat]
        fn = per_cat_fn[cat]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if per_cat_total[cat] > 0 or fp > 0:
            per_cat_metrics[cat] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": per_cat_total[cat],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_category": per_cat_metrics,
        "confusion": {str(k): dict(v) for k, v in confusion.items()},
        "per_article": results_per_article,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate V18 categorization accuracy")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Get ground truth
    train_articles, train_labels, split_articles, split_labels = get_ground_truth_labels(
        split_name=args.split, n_categories=cfg.cross_attn_engram.num_categories,
    )

    # Build label map for split articles: title -> ground truth label
    title_to_label = {}
    for (title, _), label in zip(split_articles, split_labels):
        title_to_label[title] = label

    # Build category name map (top titles per cluster from training set)
    cluster_names = {}
    for (title, _), label in zip(train_articles, train_labels):
        if label not in cluster_names:
            cluster_names[label] = []
        if len(cluster_names[label]) < 3:
            cluster_names[label].append(title)

    # Predict
    print(f"\nPredicting categories for {len(split_articles)} {args.split} articles...")
    t0 = time.time()
    predictions = predict_categories(model, tokenizer, split_articles, device)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")

    # Evaluate
    results = evaluate(predictions, split_articles, title_to_label,
                       n_categories=cfg.cross_attn_engram.num_categories)

    # Print results
    print("\n" + "=" * 70)
    print(f"CATEGORIZATION ACCURACY — {args.split} set")
    print("=" * 70)
    print(f"Overall accuracy: {results['accuracy']:.3f} ({results['correct']}/{results['total']})")
    print(f"Random baseline:  {1/cfg.cross_attn_engram.num_categories:.3f} (1/{cfg.cross_attn_engram.num_categories})")

    # Top and bottom categories by F1
    cats_by_f1 = sorted(results["per_category"].items(), key=lambda x: -x[1]["f1"])

    print(f"\nTop 10 categories by F1:")
    print(f"  {'Cat':>4} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Supp':>5}  Sample titles")
    print(f"  {'-'*60}")
    for cat, m in cats_by_f1[:10]:
        names = cluster_names.get(cat, ["?"])
        name_str = ", ".join(n[:25] for n in names[:2])
        print(f"  {cat:4d} {m['f1']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['support']:5d}  {name_str}")

    print(f"\nBottom 10 categories by F1 (with support > 0):")
    cats_with_support = [(c, m) for c, m in cats_by_f1 if m["support"] > 0]
    for cat, m in cats_with_support[-10:]:
        names = cluster_names.get(cat, ["?"])
        name_str = ", ".join(n[:25] for n in names[:2])
        print(f"  {cat:4d} {m['f1']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['support']:5d}  {name_str}")

    # Confidence analysis
    correct_confs = [r["confidence"] for r in results["per_article"] if r["correct"]]
    wrong_confs = [r["confidence"] for r in results["per_article"] if not r["correct"]]
    if correct_confs:
        print(f"\nMean confidence — correct: {sum(correct_confs)/len(correct_confs):.3f}, "
              f"wrong: {sum(wrong_confs)/len(wrong_confs):.3f}" if wrong_confs else "")

    # Show some misclassifications
    misclassified = [r for r in results["per_article"] if not r["correct"]]
    if misclassified:
        print(f"\nSample misclassifications (first 10):")
        for r in misclassified[:10]:
            true_names = cluster_names.get(r["true"], ["?"])
            pred_names = cluster_names.get(r["pred"], ["?"])
            print(f"  '{r['title'][:40]}' — true: cat {r['true']} ({true_names[0][:20]}), "
                  f"pred: cat {r['pred']} ({pred_names[0][:20]}), conf: {r['confidence']:.3f}")

    # Save
    out_path = Path("results/v18_cross_attn") / f"categorization_{args.split}.json"
    save_data = {
        "split": args.split,
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"],
        "per_category": {str(k): v for k, v in results["per_category"].items()},
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
