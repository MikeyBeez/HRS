"""Logarithmic Kernel Depth Attention for Engram Extraction.

Tests whether log-compressed scoring finds richer multi-layer combinations
than exponential's bimodal solution.

Three conditions:
H. Log-similarity: log(1 + relu(dot_score))
I. Log-distance: -log(1 + distance/temp)
J. Power-law: -distance^(p/2) / temp with learnable exponent p

Usage:
    python exp_log_kernel.py [--device cuda] [--train-steps 2000]
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
from exp_attnres import extract_all_layers, eval_niah, eval_grounding


# ============================================================
# Log Kernel Modules
# ============================================================

class LogSimilarityDepthAttention(nn.Module):
    """Depth attention using log-compressed dot product: log(1 + relu(q·k))."""

    def __init__(self, d_model, n_layers):
        super().__init__()
        self.d_model = d_model
        self.total_layers = n_layers + 1
        self.depth_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.ones(self.total_layers))

    def forward(self, hidden_states_list):
        n = len(hidden_states_list)
        stacked = torch.stack(hidden_states_list, dim=1)
        scales = self.layer_scale[:n].view(1, n, 1, 1)
        stacked = stacked * scales
        layer_means = stacked.mean(dim=2)  # (1, L, D)

        query = self.depth_query
        keys = self.key_proj(layer_means)

        scale = 1.0 / math.sqrt(self.d_model)
        raw_scores = (query @ keys.transpose(-2, -1)) * scale  # (1, 1, L)
        scores = torch.log1p(F.relu(raw_scores))  # log(1 + max(0, score))

        weights = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)
        attended = (weights.unsqueeze(0).unsqueeze(-1) * layer_means).sum(dim=1)
        engram = F.normalize(attended.squeeze(0), dim=0)
        return engram, weights.detach()


class LogDistanceDepthAttention(nn.Module):
    """Depth attention using negative log distance: -log(1 + ||q-k||²/temp)."""

    def __init__(self, d_model, n_layers):
        super().__init__()
        self.d_model = d_model
        self.total_layers = n_layers + 1
        self.depth_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.ones(self.total_layers))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(d_model))))

    def forward(self, hidden_states_list):
        n = len(hidden_states_list)
        stacked = torch.stack(hidden_states_list, dim=1)
        scales = self.layer_scale[:n].view(1, n, 1, 1)
        stacked = stacked * scales
        layer_means = stacked.mean(dim=2)

        query = self.depth_query
        keys = self.key_proj(layer_means)

        temperature = self.log_temperature.exp()
        q_sq = (query ** 2).sum(dim=-1, keepdim=True)
        k_sq = (keys ** 2).sum(dim=-1, keepdim=True)
        dot = query @ keys.transpose(-2, -1)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot

        scores = -torch.log1p(distances / temperature)

        weights = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)
        attended = (weights.unsqueeze(0).unsqueeze(-1) * layer_means).sum(dim=1)
        engram = F.normalize(attended.squeeze(0), dim=0)
        return engram, weights.detach()


class PowerLawDepthAttention(nn.Module):
    """Depth attention with learnable power exponent: -d^(p/2) / temp.

    At p=2: Gaussian. At p=1: Laplacian. At p<1: sub-linear (log-like).
    """

    def __init__(self, d_model, n_layers):
        super().__init__()
        self.d_model = d_model
        self.total_layers = n_layers + 1
        self.depth_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.ones(self.total_layers))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(d_model))))
        # Learnable power exponent — start at 1.0 (Laplacian)
        self.raw_power = nn.Parameter(torch.tensor(0.0))  # softplus(0) ≈ 0.693

    def get_power(self):
        return F.softplus(self.raw_power)

    def forward(self, hidden_states_list):
        n = len(hidden_states_list)
        stacked = torch.stack(hidden_states_list, dim=1)
        scales = self.layer_scale[:n].view(1, n, 1, 1)
        stacked = stacked * scales
        layer_means = stacked.mean(dim=2)

        query = self.depth_query
        keys = self.key_proj(layer_means)

        temperature = self.log_temperature.exp()
        q_sq = (query ** 2).sum(dim=-1, keepdim=True)
        k_sq = (keys ** 2).sum(dim=-1, keepdim=True)
        dot = query @ keys.transpose(-2, -1)
        distances = (q_sq + k_sq.transpose(-2, -1) - 2 * dot).clamp(min=1e-8)

        p = self.get_power()
        scores = -distances.pow(p / 2.0) / temperature

        weights = F.softmax(scores, dim=-1).squeeze(0).squeeze(0)
        attended = (weights.unsqueeze(0).unsqueeze(-1) * layer_means).sum(dim=1)
        engram = F.normalize(attended.squeeze(0), dim=0)
        return engram, weights.detach()


# ============================================================
# Training
# ============================================================

def train_module(model, module, tokenizer, articles, device,
                 n_steps=2000, lr=1e-3, batch_size=16, label=""):
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    module.train()

    print("  Pre-computing hidden states...")
    article_hidden = {}
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

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        half = batch_size // 2
        loss = torch.tensor(0.0, device=device)

        for _ in range(half):
            art = random.choice(multi_keys)
            i, j = random.sample(range(len(article_hidden[art])), 2)
            ea, _ = module(article_hidden[art][i])
            eb, _ = module(article_hidden[art][j])
            loss = loss - F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0))

        for _ in range(half):
            a1, a2 = random.sample(all_keys, 2)
            ea, _ = module(random.choice(article_hidden[a1]))
            eb, _ = module(random.choice(article_hidden[a2]))
            loss = loss + F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0))

        loss = loss / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg = sum(losses[-500:]) / 500
            elapsed = time.time() - t0
            with torch.no_grad():
                sample_hs = article_hidden[all_keys[0]][0]
                _, weights = module(sample_hs)
                w_str = " ".join(f"{w:.3f}" for w in weights.tolist())
                extras = ""
                if hasattr(module, 'log_temperature'):
                    extras += f" temp={module.log_temperature.exp().item():.1f}"
                if hasattr(module, 'raw_power'):
                    extras += f" p={module.get_power().item():.3f}"
            print(f"    [{label}] Step {step+1}: loss={avg:.4f}{extras} weights=[{w_str}] ({elapsed:.0f}s)")

    return losses


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-steps", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded V18 (frozen)")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d_model = cfg.model.d_model
    n_layers = cfg.model.n_layers

    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    articles = _extract_articles(raw["validation"]["text"])

    conditions = [
        ("H_log_similarity", LogSimilarityDepthAttention(d_model, n_layers)),
        ("I_log_distance", LogDistanceDepthAttention(d_model, n_layers)),
        ("J_power_law", PowerLawDepthAttention(d_model, n_layers)),
    ]

    all_results = {}

    for cond_name, module in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION {cond_name}")
        print(f"{'='*60}")

        module = module.to(device)
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params:,}")

        train_module(model, module, tokenizer, articles, device,
                     n_steps=args.train_steps, label=cond_name)

        print("\n  NIAH Retrieval:")
        niah = eval_niah(model, module, tokenizer, device, cond_name)

        print("\n  Generation Grounding:")
        ground = eval_grounding(model, module, tokenizer, device, cond_name)

        with torch.no_grad():
            sample_ids = torch.tensor(
                tokenizer.encode(NEEDLES[0].fact, add_special_tokens=False), dtype=torch.long
            )
            sample_hs = extract_all_layers(model, sample_ids, device)
            _, weights = module(sample_hs)

        result = {
            "niah": niah,
            "grounding": ground,
            "layer_weights": weights.tolist(),
        }
        if hasattr(module, 'log_temperature'):
            result["temperature"] = module.log_temperature.exp().item()
        if hasattr(module, 'raw_power'):
            result["power"] = module.get_power().item()

        all_results[cond_name] = result

        layer_names = ["emb"] + [f"L{i+1}" for i in range(n_layers)]
        print(f"\n  Layer weights: {' '.join(f'{w:.3f}' for w in weights.tolist())}")
        if hasattr(module, 'raw_power'):
            print(f"  Power exponent: {module.get_power().item():.3f}")
        if hasattr(module, 'log_temperature'):
            print(f"  Temperature: {module.log_temperature.exp().item():.1f}")

        del module
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY — ALL CONDITIONS")
    print(f"{'='*60}")

    print(f"\n  {'Method':<30} {'Found':>6} {'Rank':>6} {'Sim':>8} {'Special':>12}")
    print(f"  {'-'*65}")
    # Baselines
    baselines = [
        ("Standard engram", "5/5", "1.2", "0.505", "—"),
        ("Delta 5", "5/5", "1.0", "0.539", "—"),
        ("AttnRes dot", "5/5", "1.0", "0.500", "—"),
        ("AttnRes exp", "5/5", "1.6", "0.400", "—"),
    ]
    for name, found, rank, sim, spec in baselines:
        print(f"  {name:<30} {found:>6} {rank:>6} {sim:>8} {spec:>12}")

    for name, data in all_results.items():
        niah = data["niah"]
        n_found = sum(1 for r in niah if r["found"])
        ranks = [r["rank"] for r in niah if r["found"]]
        sims = [r["sim"] for r in niah if r["found"]]
        mr = f"{sum(ranks)/len(ranks):.1f}" if ranks else "N/A"
        ms = f"{sum(sims)/len(sims):.4f}" if sims else "N/A"
        spec = ""
        if "power" in data:
            spec = f"p={data['power']:.3f}"
        elif "temperature" in data:
            spec = f"t={data['temperature']:.0f}"
        print(f"  {name:<30} {n_found:>4}/5 {mr:>6} {ms:>8} {spec:>12}")

    print(f"\n  Grounding:")
    print(f"  {'Method':<30} {'Recall':>10}")
    print(f"  {'-'*42}")
    for name in ["Standard engram", "Delta 5", "AttnRes dot", "AttnRes exp"]:
        recall = {"Standard engram": "3%", "Delta 5": "7%", "AttnRes dot": "3%", "AttnRes exp": "10%"}
        print(f"  {name:<30} {recall[name]:>10}")
    for name, data in all_results.items():
        gr = data["grounding"]
        hits = sum(r["hits"] for r in gr)
        total = sum(r["total"] for r in gr)
        pct = hits / total * 100 if total > 0 else 0
        print(f"  {name:<30} {f'{pct:.0f}% ({hits}/{total})':>10}")

    print(f"\n  Layer weights:")
    layer_names = ["emb"] + [f"L{i+1}" for i in range(n_layers)]
    print(f"  {'Method':<25}" + " ".join(f"{n:>6}" for n in layer_names))
    print(f"  {'AttnRes dot':<25}" + " ".join(f"{w:>6.3f}" for w in [0.001,0,0,0,0,0,0.999]))
    print(f"  {'AttnRes exp':<25}" + " ".join(f"{w:>6.3f}" for w in [0.512,0,0,0,0.01,0.478,0]))
    for name, data in all_results.items():
        w = data["layer_weights"]
        print(f"  {name:<25}" + " ".join(f"{v:>6.3f}" for v in w))

    # Count active layers (weight > 0.05) for each method
    print(f"\n  Active layers (weight > 0.05):")
    for name, data in all_results.items():
        active = sum(1 for w in data["layer_weights"] if w > 0.05)
        top_layers = sorted(enumerate(data["layer_weights"]), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{layer_names[i]}={w:.3f}" for i, w in top_layers if w > 0.01)
        print(f"  {name:<25} {active} active: {top_str}")

    # Save
    out_path = Path("results/v18_cross_attn/log_kernel_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
