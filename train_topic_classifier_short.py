"""Train topic similarity classifier on SHORT prompts.

Same approach as train_topic_classifier.py but generates single-sentence
engram pairs to match the actual use case of topic routing during conversation.

Extracts individual sentences from WikiText-103 articles, computes engrams
for each, then builds same-article/different-article pairs.

Usage:
    python train_topic_classifier_short.py [--n-articles 2000] [--device cuda]
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import _extract_articles
from train_topic_classifier import (
    TopicClassifier, EngramPairDataset, train_classifier,
)


def load_v18_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


def extract_sentences(text, min_words=8, max_words=40):
    """Extract individual sentences from article text."""
    # Split on period followed by space/newline, or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if min_words <= len(words) <= max_words:
            result.append(s)
    return result


@torch.no_grad()
def compute_sentence_engrams(model, tokenizer, articles, device, max_sentences_per_article=5):
    """Compute engrams for individual sentences from articles.

    Returns:
        dict mapping article_index -> list of (engram_tensor, sentence_text) tuples
    """
    n_layers = model.cfg.model.n_layers
    extract_layer = n_layers - 2

    article_data = {}

    for art_idx, (title, text) in enumerate(articles):
        sentences = extract_sentences(text)
        if len(sentences) < 2:
            continue

        # Sample up to max_sentences_per_article
        if len(sentences) > max_sentences_per_article:
            sentences = random.sample(sentences, max_sentences_per_article)

        engrams = []
        for sent in sentences:
            ids = tokenizer.encode(sent, add_special_tokens=False)
            if len(ids) < 5:
                continue
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

            captured = {}
            def hook_fn(module, inp, out):
                captured['h'] = out[0].detach()
            handle = model.blocks[extract_layer].register_forward_hook(hook_fn)
            _ = model(input_ids, step=0)
            handle.remove()

            engram = captured['h'].mean(dim=1).squeeze(0).cpu()
            engrams.append((engram, sent))

        if len(engrams) >= 2:
            article_data[art_idx] = engrams

        if (art_idx + 1) % 500 == 0:
            print(f"  {art_idx + 1}/{len(articles)} articles processed "
                  f"({len(article_data)} with 2+ sentences)")

    return article_data


def build_short_pairs(article_data, n_pairs, pos_ratio=0.5):
    """Build positive and negative sentence-level engram pairs."""
    all_keys = list(article_data.keys())
    multi_keys = [k for k in all_keys if len(article_data[k]) >= 2]

    n_pos = int(n_pairs * pos_ratio)
    n_neg = n_pairs - n_pos

    engrams_a = []
    engrams_b = []
    labels = []
    texts_a = []
    texts_b = []

    # Positive pairs: two sentences from same article
    for _ in range(n_pos):
        art = random.choice(multi_keys)
        items = article_data[art]
        i, j = random.sample(range(len(items)), 2)
        engrams_a.append(items[i][0])
        engrams_b.append(items[j][0])
        texts_a.append(items[i][1])
        texts_b.append(items[j][1])
        labels.append(1.0)

    # Negative pairs: sentences from different articles
    for _ in range(n_neg):
        art1, art2 = random.sample(all_keys, 2)
        item1 = random.choice(article_data[art1])
        item2 = random.choice(article_data[art2])
        engrams_a.append(item1[0])
        engrams_b.append(item2[0])
        texts_a.append(item1[1])
        texts_b.append(item2[1])
        labels.append(0.0)

    return (
        torch.stack(engrams_a),
        torch.stack(engrams_b),
        torch.tensor(labels),
        texts_a,
        texts_b,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-articles", type=int, default=3000)
    parser.add_argument("--n-pairs", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Extract articles
    print("\nLoading WikiText-103 articles...")
    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    articles = _extract_articles(raw["train"]["text"])
    articles = articles[:args.n_articles]
    print(f"Using {len(articles)} articles")

    # Compute sentence-level engrams
    print("\nComputing sentence engrams...")
    t0 = time.time()
    article_data = compute_sentence_engrams(model, tokenizer, articles, device)
    elapsed = time.time() - t0
    n_sentences = sum(len(v) for v in article_data.values())
    print(f"Computed {n_sentences} sentence engrams from {len(article_data)} articles in {elapsed:.0f}s")

    # Build pairs
    print(f"\nBuilding {args.n_pairs} sentence-level pairs...")
    all_a, all_b, all_labels, texts_a, texts_b = build_short_pairs(article_data, args.n_pairs)
    print(f"Positive: {(all_labels == 1).sum().item()}, Negative: {(all_labels == 0).sum().item()}")

    # Analyze similarity distributions
    with torch.no_grad():
        cos_sims = F.cosine_similarity(
            F.normalize(all_a, dim=-1),
            F.normalize(all_b, dim=-1),
            dim=-1,
        )
        pos_sims = cos_sims[all_labels == 1]
        neg_sims = cos_sims[all_labels == 0]
        print(f"\nShort-prompt cosine similarity distributions:")
        print(f"  Positive pairs: mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}, "
              f"min={pos_sims.min():.4f}, max={pos_sims.max():.4f}")
        print(f"  Negative pairs: mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}, "
              f"min={neg_sims.min():.4f}, max={neg_sims.max():.4f}")
        print(f"  Overlap zone: [{neg_sims.quantile(0.75):.4f}, {pos_sims.quantile(0.25):.4f}]")

        # Compare to article-length stats
        print(f"\n  (For reference, article-length segments: pos mean=0.755, neg mean=0.135)")

    # Split train/val
    n_val = min(5000, len(all_labels) // 5)
    perm = torch.randperm(len(all_labels))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = EngramPairDataset(all_a[train_idx], all_b[train_idx], all_labels[train_idx])
    val_ds = EngramPairDataset(all_a[val_idx], all_b[val_idx], all_labels[val_idx])

    # Train classifier on SHORT prompts
    print(f"\nTraining SHORT-PROMPT classifier (hidden=32, epochs={args.epochs})...")
    classifier, best_acc = train_classifier(
        train_ds, val_ds,
        hidden_dim=32,
        epochs=args.epochs,
    )

    # Optimal fixed threshold for short prompts
    print(f"\nOptimal fixed cosine threshold (short prompts):")
    best_thresh_acc = 0
    best_thresh = 0
    for t_val in [i * 0.05 for i in range(1, 20)]:
        preds = (cos_sims[val_idx] > t_val).float()
        acc = (preds == all_labels[val_idx]).float().mean().item()
        if acc > best_thresh_acc:
            best_thresh_acc = acc
            best_thresh = t_val
    print(f"  Best threshold: {best_thresh:.2f}, accuracy: {best_thresh_acc:.3f}")
    print(f"  Learned classifier accuracy: {best_acc:.3f}")
    print(f"  Improvement: {best_acc - best_thresh_acc:+.3f}")

    # Show some hard cases from the overlap zone
    print(f"\nHard cases (cosine similarity in overlap zone):")
    overlap_mask = (cos_sims > neg_sims.quantile(0.75)) & (cos_sims < pos_sims.quantile(0.25))
    overlap_idx = overlap_mask.nonzero().squeeze(-1)
    if len(overlap_idx) > 0:
        sample_idx = overlap_idx[torch.randperm(len(overlap_idx))[:10]]
        for idx in sample_idx:
            i = idx.item()
            label = "SAME" if all_labels[i] == 1 else "DIFF"
            with torch.no_grad():
                pred_prob = classifier.predict(all_a[i:i+1], all_b[i:i+1]).item()
            print(f"  [{label}] sim={cos_sims[i]:.4f} clf={pred_prob:.3f}")
            print(f"    A: {texts_a[i][:80]}...")
            print(f"    B: {texts_b[i][:80]}...")

    # Save
    out_path = Path("results/v18_cross_attn/topic_classifier_short.pt")
    torch.save({
        "model_state_dict": classifier.state_dict(),
        "hidden_dim": 32,
        "best_val_acc": best_acc,
        "best_fixed_threshold": best_thresh,
        "best_fixed_threshold_acc": best_thresh_acc,
        "n_articles": len(article_data),
        "n_pairs": args.n_pairs,
        "pos_sim_stats": {
            "mean": pos_sims.mean().item(),
            "std": pos_sims.std().item(),
            "min": pos_sims.min().item(),
            "max": pos_sims.max().item(),
        },
        "neg_sim_stats": {
            "mean": neg_sims.mean().item(),
            "std": neg_sims.std().item(),
            "min": neg_sims.min().item(),
            "max": neg_sims.max().item(),
        },
    }, out_path)
    print(f"\nSaved short-prompt classifier to {out_path}")
    print(f"Params: {sum(p.numel() for p in classifier.parameters()):,}")


if __name__ == "__main__":
    main()
