"""Experiment: Attention Residuals for Engram Extraction.

Instead of extracting engrams from a single layer, use learned depth-wise
attention to dynamically weight across all layers. Tests whether
depth-attended engrams improve retrieval and grounding.

Three conditions:
A. Standard engram (mean-pooled from layer 4) — baseline
B. Delta 5 engram (last layer transition) — previous best
C. AttnRes engram (learned depth-attention across all layers)
D. AttnRes engram with exponential kernel depth-attention

For the AttnRes conditions, we add a small module that:
1. Collects hidden states from all layers during forward pass
2. Uses a learned query to attend across layers per position
3. Mean-pools the depth-attended states into an engram

This is NOT retraining V18. The AttnRes module is trained separately
on a retrieval objective: maximize cosine similarity between same-article
engrams and minimize it for cross-article engrams.

Usage:
    python exp_attnres.py [--device cuda] [--train-steps 2000]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore, EngramEntry
from niah_egr import NEEDLES, DISTRACTORS
from data import _extract_articles


# ============================================================
# AttnRes Module
# ============================================================

class DepthAttentionEngram(nn.Module):
    """Learned depth-wise attention for engram extraction.

    Given hidden states from all layers, produces a single engram vector
    by attending across layers at each position, then mean-pooling.
    """

    def __init__(self, d_model, n_layers, use_exponential=False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_exponential = use_exponential

        # Learned query for depth attention (one per position is too expensive;
        # use a single shared query that attends across layers)
        self.depth_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Key projection for each layer's hidden states
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

        # Value projection (identity — we want the actual hidden states)
        # but with a learned per-layer scaling
        self.layer_scale = nn.Parameter(torch.ones(n_layers + 1))  # +1 for embedding

        # Temperature for exponential kernel
        self.temperature = float(d_model) if use_exponential else 1.0

    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: list of (1, T, D) tensors, one per layer
                                (including embedding as layer 0)

        Returns:
            engram: (D,) mean-pooled depth-attended vector
            weights: (n_layers+1,) attention weights per layer (for inspection)
        """
        n_layers = len(hidden_states_list)
        T = hidden_states_list[0].shape[1]
        D = self.d_model

        # Stack all layers: (1, L, T, D) where L = n_layers + 1
        stacked = torch.stack(hidden_states_list, dim=1)  # (1, L, T, D)

        # Scale each layer
        scales = self.layer_scale[:n_layers].view(1, n_layers, 1, 1)
        stacked = stacked * scales

        # Mean pool across positions first (cheaper than attending per position)
        layer_means = stacked.mean(dim=2)  # (1, L, D)

        # Depth attention: query attends across L layers
        query = self.depth_query  # (1, 1, D)
        keys = self.key_proj(layer_means)  # (1, L, D)

        if self.use_exponential:
            # Exponential kernel: -||q-k||² / temperature
            q_sq = (query ** 2).sum(dim=-1, keepdim=True)  # (1, 1, 1)
            k_sq = (keys ** 2).sum(dim=-1, keepdim=True)  # (1, L, 1)
            dot = query @ keys.transpose(-2, -1)  # (1, 1, L)
            distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot
            scores = -distances / self.temperature  # (1, 1, L)
        else:
            # Dot product
            scale = 1.0 / math.sqrt(D)
            scores = (query @ keys.transpose(-2, -1)) * scale  # (1, 1, L)

        weights = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)  # (L,)

        # Weighted combination of layer means
        # weights: (L,), layer_means: (1, L, D)
        attended = (weights.unsqueeze(0).unsqueeze(-1) * layer_means).sum(dim=1)  # (1, D)

        engram = F.normalize(attended.squeeze(0), dim=0)
        return engram, weights.detach()


# ============================================================
# Hidden state extraction from frozen V18
# ============================================================

@torch.no_grad()
def extract_all_layers(model, token_ids, device):
    """Run forward pass and capture all layer hidden states."""
    ids = token_ids[:512].unsqueeze(0).to(device)

    hidden_states = []
    x = model.drop(model.tok_emb(ids))
    hidden_states.append(x.detach())

    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        x, _, _, _ = block(x, step=0, engram_buffer=eb)
        hidden_states.append(x.detach())

    return hidden_states  # list of (1, T, D), length = n_layers + 1


# ============================================================
# Training the AttnRes module on retrieval objective
# ============================================================

def train_attnres(model, attnres_module, tokenizer, articles, device,
                  n_steps=2000, lr=1e-3, batch_size=16):
    """Train the depth-attention module on a contrastive retrieval objective.

    Positive pairs: two segments from the same article.
    Negative pairs: segments from different articles.
    Loss: contrastive (cosine similarity).
    """
    optimizer = torch.optim.Adam(attnres_module.parameters(), lr=lr)
    attnres_module.train()

    # Pre-compute hidden states for all article segments
    print("  Pre-computing hidden states for training...")
    article_hidden = {}  # article_idx -> list of hidden_states_list
    for art_idx, (title, text) in enumerate(articles):
        ids_full = tokenizer.encode(text, add_special_tokens=False)
        if len(ids_full) < 64:
            continue
        segments = []
        for start in range(0, len(ids_full), 512):
            seg = ids_full[start:start + 512]
            if len(seg) < 32:
                continue
            seg_ids = torch.tensor(seg, dtype=torch.long)
            hs = extract_all_layers(model, seg_ids, device)
            segments.append(hs)
            if len(segments) >= 3:
                break
        if len(segments) >= 2:
            article_hidden[art_idx] = segments
        if len(article_hidden) >= 200:
            break

    multi_keys = [k for k, v in article_hidden.items() if len(v) >= 2]
    all_keys = list(article_hidden.keys())
    print(f"  {len(article_hidden)} articles, {len(multi_keys)} with 2+ segments")

    # Training loop
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        # Build batch: half positive, half negative
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        half = batch_size // 2

        for _ in range(half):
            art = random.choice(multi_keys)
            i, j = random.sample(range(len(article_hidden[art])), 2)
            pos_a.append(article_hidden[art][i])
            pos_b.append(article_hidden[art][j])

        for _ in range(half):
            a1, a2 = random.sample(all_keys, 2)
            s1 = random.choice(article_hidden[a1])
            s2 = random.choice(article_hidden[a2])
            neg_a.append(s1)
            neg_b.append(s2)

        # Compute engrams
        loss = torch.tensor(0.0, device=device)
        for hs_a, hs_b in zip(pos_a, pos_b):
            eng_a, _ = attnres_module(hs_a)
            eng_b, _ = attnres_module(hs_b)
            loss = loss - F.cosine_similarity(eng_a.unsqueeze(0), eng_b.unsqueeze(0))

        for hs_a, hs_b in zip(neg_a, neg_b):
            eng_a, _ = attnres_module(hs_a)
            eng_b, _ = attnres_module(hs_b)
            # Push negatives apart: maximize distance (minimize negative cosine sim)
            loss = loss + F.cosine_similarity(eng_a.unsqueeze(0), eng_b.unsqueeze(0))

        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg = sum(losses[-500:]) / 500
            elapsed = time.time() - t0
            # Show layer weights
            with torch.no_grad():
                sample_hs = article_hidden[all_keys[0]][0]
                _, weights = attnres_module(sample_hs)
                w_str = " ".join(f"{w:.3f}" for w in weights.tolist())
            print(f"    Step {step+1}: loss={avg:.4f}, weights=[{w_str}] ({elapsed:.0f}s)")

    return losses


# ============================================================
# NIAH evaluation
# ============================================================

def eval_niah(model, attnres_module, tokenizer, device, label):
    """Run NIAH with AttnRes engrams."""
    d_model = model.cfg.model.d_model
    distractors = DISTRACTORS[:20]

    results = []
    for needle in NEEDLES:
        # Build store
        store = EngramStore(d_model)

        all_docs = [(needle.fact, "needle")] + [(d, f"dist_{i}") for i, d in enumerate(distractors)]

        for text, source in all_docs:
            ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
            if ids.shape[0] < 5:
                continue
            hs = extract_all_layers(model, ids, device)

            with torch.no_grad():
                engram, _ = attnres_module(hs)

            store.store(engram.cpu(), EngramEntry(
                text=text, mean_entropy=0.0, condition=label, source=source,
            ))

        # Query
        query_ids = torch.tensor(
            tokenizer.encode(needle.query, add_special_tokens=False), dtype=torch.long
        )
        query_hs = extract_all_layers(model, query_ids, device)
        with torch.no_grad():
            query_engram, weights = attnres_module(query_hs)

        retrieval = store.retrieve(query_engram.cpu(), top_k=5, min_similarity=0.0)
        found = False
        rank = -1
        sim = 0.0
        for r, (s, entry, _) in enumerate(retrieval):
            if entry.source == "needle":
                found = True
                rank = r + 1
                sim = s
                break

        status = f"FOUND rank={rank} sim={sim:.4f}" if found else "NOT FOUND"
        print(f"    {needle.category}: {status}")
        results.append({"needle": needle.category, "found": found, "rank": rank, "sim": sim})

    return results


def eval_grounding(model, attnres_module, tokenizer, device, label):
    """Test generation grounding with AttnRes engrams."""
    d_model = model.cfg.model.d_model
    results = []

    for needle in NEEDLES:
        needle_ids = torch.tensor(
            tokenizer.encode(needle.fact, add_special_tokens=False), dtype=torch.long
        )
        query_ids = torch.tensor(
            tokenizer.encode(needle.query, add_special_tokens=False), dtype=torch.long
        )

        # Get AttnRes engram for needle
        hs = extract_all_layers(model, needle_ids, device)
        with torch.no_grad():
            engram, _ = attnres_module(hs)

        # Inject as cross-attention buffer
        orig_buffer = model.engram_buffer.data.clone()
        orig_init = model._engram_buffer_initialized

        buffer = engram.unsqueeze(0).unsqueeze(0).expand(1, 32, d_model).contiguous()
        model.engram_buffer = nn.Parameter(buffer.to(device), requires_grad=False)
        model._engram_buffer_initialized = True

        # Generate
        input_ids = query_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(150):
                idx = input_ids[:, -512:]
                output = model(idx, step=0)
                logits = output.logits[:, -1, :] / 0.9
                v, _ = torch.topk(logits, 50)
                logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        model.engram_buffer = nn.Parameter(orig_buffer, requires_grad=False)
        model._engram_buffer_initialized = orig_init

        gen_text = tokenizer.decode(input_ids[0, query_ids.shape[0]:], skip_special_tokens=True)
        hits = sum(1 for t in needle.answer_tokens if t.lower() in gen_text.lower())
        results.append({
            "needle": needle.category,
            "hits": hits,
            "total": len(needle.answer_tokens),
            "text": gen_text[:150],
        })
        print(f"    {needle.category}: {hits}/{len(needle.answer_tokens)} — {gen_text[:80]}...")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-steps", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d_model = cfg.model.d_model
    n_layers = cfg.model.n_layers

    # Load training articles
    print("\nLoading WikiText-103 for AttnRes training...")
    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    articles = _extract_articles(raw["validation"]["text"])

    results = {}

    # ============================================================
    # Condition C: AttnRes with dot product depth-attention
    # ============================================================
    print(f"\n{'='*60}")
    print("CONDITION C: AttnRes (dot product depth-attention)")
    print(f"{'='*60}")

    attnres_dot = DepthAttentionEngram(d_model, n_layers, use_exponential=False).to(device)
    print(f"  AttnRes params: {sum(p.numel() for p in attnres_dot.parameters()):,}")

    train_attnres(model, attnres_dot, tokenizer, articles, device,
                  n_steps=args.train_steps)

    print("\n  NIAH Retrieval:")
    niah_c = eval_niah(model, attnres_dot, tokenizer, device, "attnres_dot")
    print("\n  Generation Grounding:")
    ground_c = eval_grounding(model, attnres_dot, tokenizer, device, "attnres_dot")

    results["attnres_dot"] = {
        "niah": niah_c,
        "grounding": ground_c,
        "layer_weights": None,  # filled below
    }

    # Get learned layer weights
    with torch.no_grad():
        sample_ids = torch.tensor(tokenizer.encode(NEEDLES[0].fact, add_special_tokens=False), dtype=torch.long)
        sample_hs = extract_all_layers(model, sample_ids, device)
        _, weights_c = attnres_dot(sample_hs)
        results["attnres_dot"]["layer_weights"] = weights_c.tolist()
        print(f"\n  Learned layer weights: {['%.3f' % w for w in weights_c.tolist()]}")

    # ============================================================
    # Condition D: AttnRes with exponential kernel depth-attention
    # ============================================================
    print(f"\n{'='*60}")
    print("CONDITION D: AttnRes (exponential kernel depth-attention)")
    print(f"{'='*60}")

    attnres_exp = DepthAttentionEngram(d_model, n_layers, use_exponential=True).to(device)

    train_attnres(model, attnres_exp, tokenizer, articles, device,
                  n_steps=args.train_steps)

    print("\n  NIAH Retrieval:")
    niah_d = eval_niah(model, attnres_exp, tokenizer, device, "attnres_exp")
    print("\n  Generation Grounding:")
    ground_d = eval_grounding(model, attnres_exp, tokenizer, device, "attnres_exp")

    results["attnres_exp"] = {
        "niah": niah_d,
        "grounding": ground_d,
    }

    with torch.no_grad():
        _, weights_d = attnres_exp(sample_hs)
        results["attnres_exp"]["layer_weights"] = weights_d.tolist()
        print(f"\n  Learned layer weights: {['%.3f' % w for w in weights_d.tolist()]}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Baselines from previous experiments
    print(f"\n  Retrieval (NIAH, 5 needles):")
    print(f"  {'Method':<30} {'Found':>6} {'Mean Rank':>10} {'Mean Sim':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Standard engram (prev)':<30} {'5/5':>6} {'1.2':>10} {'0.505':>10}")
    print(f"  {'Delta 5 (prev)':<30} {'5/5':>6} {'1.0':>10} {'0.539':>10}")

    for name, data in results.items():
        niah = data["niah"]
        n_found = sum(1 for r in niah if r["found"])
        ranks = [r["rank"] for r in niah if r["found"]]
        sims = [r["sim"] for r in niah if r["found"]]
        mr = f"{sum(ranks)/len(ranks):.1f}" if ranks else "N/A"
        ms = f"{sum(sims)/len(sims):.4f}" if sims else "N/A"
        print(f"  {name:<30} {n_found:>4}/5 {mr:>10} {ms:>10}")

    print(f"\n  Generation grounding:")
    print(f"  {'Method':<30} {'Recall':>8}")
    print(f"  {'-'*40}")
    print(f"  {'Standard engram (prev)':<30} {'3% (1/30)':>8}")
    print(f"  {'Delta 5 (prev)':<30} {'7% (2/30)':>8}")

    for name, data in results.items():
        gr = data["grounding"]
        total_hits = sum(r["hits"] for r in gr)
        total_possible = sum(r["total"] for r in gr)
        pct = total_hits / total_possible * 100 if total_possible > 0 else 0
        print(f"  {name:<30} {f'{pct:.0f}% ({total_hits}/{total_possible})':>8}")

    print(f"\n  Learned layer weights:")
    layer_names = [f"emb"] + [f"L{i+1}" for i in range(n_layers)]
    print(f"  {'Method':<20} {' '.join(f'{n:>6}' for n in layer_names)}")
    for name, data in results.items():
        w = data["layer_weights"]
        print(f"  {name:<20} {' '.join(f'{v:>6.3f}' for v in w)}")

    # Save
    out_path = Path("results/v18_cross_attn/attnres_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    # Freeze V18 — we only train the AttnRes module
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded V18 (step {ckpt.get('step', '?')}, frozen)")
    return model, cfg


if __name__ == "__main__":
    main()
