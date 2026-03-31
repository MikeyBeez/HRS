"""Train a lightweight binary classifier for topic similarity.

Given two engram vectors, predicts P(same_topic). Replaces the fixed
cosine similarity threshold in TopicContextManager.

Training data: segment pairs from WikiText-103 articles.
- Positive: two segments from the same article
- Negative: two segments from different articles

The classifier operates on features derived from engram pairs, not raw
engrams — keeping it small and fast.

Usage:
    python train_topic_classifier.py [--n-pairs 50000] [--device cuda]
"""

import argparse
import json
import random
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


class PairFeatures(nn.Module):
    """Extract features from a pair of engram vectors.

    Features:
    1. Cosine similarity (scalar)
    2. L2 distance (scalar)
    3. Element-wise product, mean (scalar)
    4. Element-wise product, std (scalar)
    5. Element-wise absolute difference, mean (scalar)
    6. Element-wise absolute difference, std (scalar)
    7. L2 norm of engram A (scalar)
    8. L2 norm of engram B (scalar)
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a, b: (batch, d_model) engram vectors

        Returns:
            (batch, 8) feature vectors
        """
        # Normalize for cosine sim
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)

        cos_sim = (a_norm * b_norm).sum(dim=-1, keepdim=True)
        l2_dist = (a - b).norm(dim=-1, keepdim=True)

        product = a_norm * b_norm
        prod_mean = product.mean(dim=-1, keepdim=True)
        prod_std = product.std(dim=-1, keepdim=True)

        diff = (a_norm - b_norm).abs()
        diff_mean = diff.mean(dim=-1, keepdim=True)
        diff_std = diff.std(dim=-1, keepdim=True)

        norm_a = a.norm(dim=-1, keepdim=True)
        norm_b = b.norm(dim=-1, keepdim=True)

        return torch.cat([
            cos_sim, l2_dist, prod_mean, prod_std,
            diff_mean, diff_std, norm_a, norm_b,
        ], dim=-1)


class TopicClassifier(nn.Module):
    """Binary classifier: P(same_topic | engram_a, engram_b).

    Small MLP on pair features. ~2K parameters.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.features = PairFeatures()
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a, b: (batch, d_model) engram vectors

        Returns:
            (batch,) logits (apply sigmoid for probability)
        """
        feats = self.features(a, b)
        return self.mlp(feats).squeeze(-1)

    def predict(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns P(same_topic)."""
        return torch.sigmoid(self.forward(a, b))


class EngramPairDataset(Dataset):
    """Dataset of engram pairs with same/different topic labels."""

    def __init__(self, engrams_a, engrams_b, labels):
        self.engrams_a = engrams_a
        self.engrams_b = engrams_b
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.engrams_a[idx], self.engrams_b[idx], self.labels[idx]


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


@torch.no_grad()
def compute_article_engrams(model, tokenizer, articles, device, max_segments_per_article=3, seg_len=512):
    """Compute engrams for article segments.

    Returns:
        dict mapping article_index -> list of (D,) engram tensors
    """
    n_layers = model.cfg.model.n_layers
    extract_layer = n_layers - 2

    article_engrams = {}

    for art_idx, (title, text) in enumerate(articles):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 64:
            continue

        segments = []
        for seg_start in range(0, len(ids), seg_len):
            seg = ids[seg_start:seg_start + seg_len]
            if len(seg) < 64:
                continue
            segments.append(seg)
            if len(segments) >= max_segments_per_article:
                break

        if not segments:
            continue

        engrams = []
        for seg in segments:
            input_ids = torch.tensor(seg, dtype=torch.long).unsqueeze(0).to(device)

            captured = {}
            def hook_fn(module, inp, out):
                captured['h'] = out[0].detach()
            handle = model.blocks[extract_layer].register_forward_hook(hook_fn)
            _ = model(input_ids, step=0)
            handle.remove()

            engram = captured['h'].mean(dim=1).squeeze(0).cpu()
            engrams.append(engram)

        article_engrams[art_idx] = engrams

        if (art_idx + 1) % 500 == 0:
            print(f"  {art_idx + 1}/{len(articles)} articles processed")

    return article_engrams


def build_pairs(article_engrams, n_pairs, pos_ratio=0.5):
    """Build positive and negative engram pairs.

    Positive: two segments from same article
    Negative: two segments from different articles
    """
    # Articles with >= 2 segments (for positive pairs)
    multi_seg = {k: v for k, v in article_engrams.items() if len(v) >= 2}
    all_keys = list(article_engrams.keys())

    n_pos = int(n_pairs * pos_ratio)
    n_neg = n_pairs - n_pos

    engrams_a = []
    engrams_b = []
    labels = []

    # Positive pairs
    multi_keys = list(multi_seg.keys())
    for _ in range(n_pos):
        art = random.choice(multi_keys)
        segs = multi_seg[art]
        i, j = random.sample(range(len(segs)), 2)
        engrams_a.append(segs[i])
        engrams_b.append(segs[j])
        labels.append(1.0)

    # Negative pairs
    for _ in range(n_neg):
        art1, art2 = random.sample(all_keys, 2)
        seg1 = random.choice(article_engrams[art1])
        seg2 = random.choice(article_engrams[art2])
        engrams_a.append(seg1)
        engrams_b.append(seg2)
        labels.append(0.0)

    return (
        torch.stack(engrams_a),
        torch.stack(engrams_b),
        torch.tensor(labels),
    )


def train_classifier(train_dataset, val_dataset, hidden_dim=32, lr=1e-3, epochs=20, batch_size=256):
    """Train the topic classifier."""
    model = TopicClassifier(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for a, b, labels in train_loader:
            logits = model(a, b)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = (logits > 0).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_pos_correct = 0
        val_pos_total = 0
        val_neg_correct = 0
        val_neg_total = 0
        with torch.no_grad():
            for a, b, labels in val_loader:
                logits = model(a, b)
                preds = (logits > 0).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                # Per-class accuracy
                pos_mask = labels == 1.0
                neg_mask = labels == 0.0
                val_pos_correct += (preds[pos_mask] == labels[pos_mask]).sum().item()
                val_pos_total += pos_mask.sum().item()
                val_neg_correct += (preds[neg_mask] == labels[neg_mask]).sum().item()
                val_neg_total += neg_mask.sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        pos_acc = val_pos_correct / max(val_pos_total, 1)
        neg_acc = val_neg_correct / max(val_neg_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss/train_total:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
              f"(pos={pos_acc:.3f} neg={neg_acc:.3f})")

    model.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-articles", type=int, default=2000, help="Articles to process for engrams")
    parser.add_argument("--n-pairs", type=int, default=50000, help="Training pairs")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Classifier hidden dim")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load V18
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Extract articles
    print("\nLoading WikiText-103 articles...")
    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    articles = _extract_articles(raw["train"]["text"])
    articles = articles[:args.n_articles]
    print(f"Using {len(articles)} articles")

    # Compute engrams
    print("\nComputing article engrams...")
    t0 = time.time()
    article_engrams = compute_article_engrams(model, tokenizer, articles, device)
    elapsed = time.time() - t0
    n_engrams = sum(len(v) for v in article_engrams.values())
    print(f"Computed {n_engrams} engrams from {len(article_engrams)} articles in {elapsed:.0f}s")

    # Build pairs
    print(f"\nBuilding {args.n_pairs} training pairs...")
    all_a, all_b, all_labels = build_pairs(article_engrams, args.n_pairs)
    print(f"Positive: {(all_labels == 1).sum().item()}, Negative: {(all_labels == 0).sum().item()}")

    # Analyze pair similarity distributions
    with torch.no_grad():
        cos_sims = F.cosine_similarity(
            F.normalize(all_a, dim=-1),
            F.normalize(all_b, dim=-1),
            dim=-1,
        )
        pos_sims = cos_sims[all_labels == 1]
        neg_sims = cos_sims[all_labels == 0]
        print(f"\nCosine similarity distributions:")
        print(f"  Positive pairs: mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}, "
              f"min={pos_sims.min():.4f}, max={pos_sims.max():.4f}")
        print(f"  Negative pairs: mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}, "
              f"min={neg_sims.min():.4f}, max={neg_sims.max():.4f}")
        print(f"  Overlap zone: [{neg_sims.quantile(0.75):.4f}, {pos_sims.quantile(0.25):.4f}]")

    # Split train/val
    n_val = min(5000, len(all_labels) // 5)
    perm = torch.randperm(len(all_labels))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = EngramPairDataset(all_a[train_idx], all_b[train_idx], all_labels[train_idx])
    val_ds = EngramPairDataset(all_a[val_idx], all_b[val_idx], all_labels[val_idx])

    # Train
    print(f"\nTraining classifier (hidden={args.hidden_dim}, epochs={args.epochs})...")
    classifier, best_acc = train_classifier(
        train_ds, val_ds,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )

    # Compute optimal cosine threshold for comparison
    print(f"\nOptimal fixed cosine threshold (for comparison):")
    best_thresh_acc = 0
    best_thresh = 0
    for t in [i * 0.05 for i in range(1, 20)]:
        preds = (cos_sims[val_idx] > t).float()
        acc = (preds == all_labels[val_idx]).float().mean().item()
        if acc > best_thresh_acc:
            best_thresh_acc = acc
            best_thresh = t
    print(f"  Best threshold: {best_thresh:.2f}, accuracy: {best_thresh_acc:.3f}")
    print(f"  Learned classifier accuracy: {best_acc:.3f}")
    print(f"  Improvement: {best_acc - best_thresh_acc:+.3f}")

    # Save
    out_path = Path("results/v18_cross_attn/topic_classifier.pt")
    torch.save({
        "model_state_dict": classifier.state_dict(),
        "hidden_dim": args.hidden_dim,
        "best_val_acc": best_acc,
        "best_fixed_threshold": best_thresh,
        "best_fixed_threshold_acc": best_thresh_acc,
        "n_articles": len(article_engrams),
        "n_pairs": args.n_pairs,
        "pos_sim_stats": {
            "mean": pos_sims.mean().item(),
            "std": pos_sims.std().item(),
        },
        "neg_sim_stats": {
            "mean": neg_sims.mean().item(),
            "std": neg_sims.std().item(),
        },
    }, out_path)
    print(f"\nSaved classifier to {out_path}")
    print(f"Params: {sum(p.numel() for p in classifier.parameters()):,}")


if __name__ == "__main__":
    main()
