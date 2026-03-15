"""Expert Isomorphism in PEER Networks — 5-phase experiment.

Hypothesis: In a PEER network trained on WikiText-2, the 262K single-neuron
experts share a common learned transformation (u vectors) and only truly
specialize in their output projections (v vectors).

Phase 1: Baseline — train standard PEER model, save expert weights
Phase 2: Isomorphism Analysis — SVD decomposition, clustering, variance analysis
Phase 3: Shared Trunk Architecture — shared u, distinct v, retrain
Phase 4: Router Analysis — compare router entropy before/after
Phase 5: Compression Measurement — parameter counts, perplexity deltas

Hardware: RTX 5070 Ti
Dataset: WikiText-2
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ExperimentConfig, AblationConfig, ModelConfig, PEERConfig
from model import HRSTransformer, HRSBlock, HRSOutput
from data import load_wikitext, build_dataloaders
from losses import CombinedHRSLoss
from peer import PEER


# ============================================================
# Shared Trunk PEER — modified PEER with shared u, distinct v
# ============================================================

class SharedTrunkPEER(nn.Module):
    """PEER variant where all experts share input weights (u) but keep
    distinct output weights (v).

    Expert computation: sigma(u_shared^T x) * v_i
    The shared u is initialized from the centroid of clustered u vectors.
    """

    def __init__(self, model_cfg: ModelConfig, peer_cfg: PEERConfig,
                 shared_u: torch.Tensor = None, expert_v: torch.Tensor = None,
                 keys_a: torch.Tensor = None, keys_b: torch.Tensor = None,
                 input_proj_weight: torch.Tensor = None,
                 output_proj_weight: torch.Tensor = None):
        super().__init__()
        d = model_cfg.d_model
        self.n_heads = peer_cfg.n_heads
        self.top_k = peer_cfg.top_k
        self.n_sub_keys = peer_cfg.n_sub_keys
        self.head_dim = d // peer_cfg.n_heads
        n_total = peer_cfg.n_sub_keys ** 2

        # Input/output projections (transferred from baseline)
        self.input_proj = nn.Linear(d, d, bias=False)
        if input_proj_weight is not None:
            self.input_proj.weight.data.copy_(input_proj_weight)

        # Product keys (transferred from baseline)
        self.keys_a = nn.Parameter(keys_a.clone() if keys_a is not None
                                   else torch.empty(self.n_heads, peer_cfg.n_sub_keys, self.head_dim))
        self.keys_b = nn.Parameter(keys_b.clone() if keys_b is not None
                                   else torch.empty(self.n_heads, peer_cfg.n_sub_keys, self.head_dim))

        # SHARED trunk: single u vector for all experts
        if shared_u is not None:
            self.expert_u_shared = nn.Parameter(shared_u.clone())  # (head_dim,)
        else:
            self.expert_u_shared = nn.Parameter(torch.empty(self.head_dim))
            nn.init.normal_(self.expert_u_shared, std=0.01)

        # DISTINCT v per expert (transferred or initialized)
        if expert_v is not None:
            self.expert_v = nn.Parameter(expert_v.clone())  # (n_total, head_dim)
        else:
            self.expert_v = nn.Parameter(torch.empty(n_total, self.head_dim))
            nn.init.normal_(self.expert_v, std=0.01)

        self.output_proj = nn.Linear(d, d, bias=False)
        if output_proj_weight is not None:
            self.output_proj.weight.data.copy_(output_proj_weight)

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(model_cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        K = self.top_k
        H = self.n_heads
        S = self.n_sub_keys
        hd = self.head_dim

        h = self.input_proj(x).reshape(B, T, H, hd)

        # Product key retrieval (same as standard PEER)
        scores_a = torch.einsum('bthd,hsd->bths', h, self.keys_a)
        scores_b = torch.einsum('bthd,hsd->bths', h, self.keys_b)

        top_a_scores, top_a_idx = scores_a.topk(K, dim=-1)
        top_b_scores, top_b_idx = scores_b.topk(K, dim=-1)

        product_scores = top_a_scores.unsqueeze(-1) + top_b_scores.unsqueeze(-2)
        product_idx = top_a_idx.unsqueeze(-1) * S + top_b_idx.unsqueeze(-2)

        product_scores_flat = product_scores.reshape(B, T, H, K * K)
        product_idx_flat = product_idx.reshape(B, T, H, K * K)

        top_scores, top_pos = product_scores_flat.topk(K, dim=-1)
        top_expert_idx = product_idx_flat.gather(-1, top_pos)

        top_weights = F.softmax(top_scores, dim=-1)

        # Gather ONLY v for selected experts
        flat_idx = top_expert_idx.reshape(-1)
        sel_v = self.expert_v[flat_idx].reshape(B, T, H, K, hd)

        # Shared activation: sigma(u_shared^T x) — same for all experts
        # u_shared: (hd,) broadcast over all positions
        h_expanded = h.unsqueeze(3)  # (B, T, H, 1, hd)
        activation = torch.sigmoid((h_expanded * self.expert_u_shared).sum(dim=-1))  # (B, T, H, 1)

        # Output: activation * sum(w_i * v_i)
        weighted_v = (top_weights.unsqueeze(-1) * sel_v).sum(dim=3)  # (B, T, H, hd)
        expert_out = activation * weighted_v  # (B, T, H, hd)

        merged = expert_out.reshape(B, T, D)
        out = self.dropout(self.norm(self.output_proj(merged)))
        return out


# ============================================================
# Training utilities
# ============================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    if step >= max_steps:
        return base_lr * 0.1
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader, device, amp_dtype, max_batches=20):
    """Evaluate perplexity on validation set."""
    model.eval()
    total_ce = 0.0
    total_tokens = 0
    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(x)
        B, T, V = output.logits.shape
        ce = F.cross_entropy(output.logits.reshape(B*T, V), y.reshape(B*T), reduction='sum')
        total_ce += ce.item()
        total_tokens += B * T
    ppl = math.exp(min(total_ce / max(total_tokens, 1), 20))
    model.train()
    return ppl


@torch.no_grad()
def evaluate_test(model, test_loader, device, amp_dtype):
    """Evaluate perplexity on full test set."""
    model.eval()
    total_ce = 0.0
    total_tokens = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(x)
        B, T, V = output.logits.shape
        ce = F.cross_entropy(output.logits.reshape(B*T, V), y.reshape(B*T), reduction='sum')
        total_ce += ce.item()
        total_tokens += B * T
    ppl = math.exp(min(total_ce / max(total_tokens, 1), 20))
    model.train()
    return ppl


@torch.no_grad()
def compute_router_entropy(model, val_loader, device, amp_dtype, max_batches=20):
    """Compute mean routing entropy across validation batches."""
    model.eval()
    entropies = []
    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(x)
        for rw in output.routing_weights:
            rw = rw.float().clamp(min=1e-10)
            ent = -(rw * rw.log()).sum(dim=-1).mean().item()
            entropies.append(ent)
    model.train()
    return np.mean(entropies) if entropies else 0.0


def simple_train(model, loaders, cfg, device, max_steps, output_dir,
                 lr=3e-4, warmup_steps=500, eval_interval=1000,
                 log_interval=200, label_smoothing=0.1, tag=""):
    """Simplified training loop for this experiment."""
    amp_dtype = torch.bfloat16
    loss_fn = CombinedHRSLoss(
        locality_cfg=cfg.locality if cfg.locality.enabled else None,
        label_smoothing=label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.1, betas=(0.9, 0.95),
    )

    train_iter = iter(loaders["train"])
    model.train()
    log = []
    t0 = time.time()

    grad_accum = cfg.training.grad_accum_steps

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        for _ in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                output = model(x, step=step, collect_layer_reps=cfg.locality.enabled)
                loss_dict = loss_fn(
                    output.logits, y,
                    layer_representations=output.layer_representations if cfg.locality.enabled else None,
                    routing_weights=output.routing_weights if cfg.uses_router() else None,
                    routing_balance_loss_val=output.routing_balance_loss if cfg.uses_router() else None,
                    balance_weight=cfg.router.balance_loss_weight,
                    routing_entropy_loss_val=output.routing_entropy_loss if cfg.uses_router() else None,
                    entropy_weight=cfg.router.entropy_loss_weight if cfg.uses_router() else 0.0,
                    engram_recon_loss=output.engram_recon_loss if cfg.uses_engrams() else None,
                    recon_weight=cfg.engram.recon_loss_weight if cfg.uses_engrams() else 0.0,
                )
                loss = loss_dict["loss"] / grad_accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        base_lr = get_lr(step, warmup_steps, max_steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        optimizer.step()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            ce = loss_dict["ce_loss"].item()
            ppl = math.exp(min(ce, 20))
            print(f"  {tag} step {step:5d}/{max_steps} | CE {ce:.4f} | ppl {ppl:.1f} | lr {base_lr:.2e} | {elapsed:.0f}s")
            log.append({"step": step, "ce": ce, "ppl": ppl, "lr": base_lr, "time": elapsed})

        if step % eval_interval == 0:
            val_ppl = evaluate(model, loaders["validation"], device, amp_dtype)
            print(f"  {tag} step {step:5d} | val_ppl = {val_ppl:.2f}")
            log[-1]["val_ppl"] = val_ppl

    # Final eval
    val_ppl = evaluate(model, loaders["validation"], device, amp_dtype)
    test_ppl = evaluate_test(model, loaders["test"], device, amp_dtype)
    print(f"  {tag} FINAL | val_ppl = {val_ppl:.2f} | test_ppl = {test_ppl:.2f}")

    return {"val_ppl": val_ppl, "test_ppl": test_ppl, "log": log}


# ============================================================
# Phase 1: Baseline Training
# ============================================================

def phase1_baseline(device, output_dir, max_steps=15000):
    """Train standard PEER model on WikiText-2."""
    print("\n" + "="*70)
    print("PHASE 1: Baseline PEER Training")
    print("="*70)

    set_seed(42)

    # Use V4-like config on WikiText-2 (PEER as unconditional FFN + 3-tier routing)
    cfg = ExperimentConfig.from_ablation(AblationConfig.V4_FULL)
    cfg.training.dataset = "wikitext-2"
    cfg.training.batch_size = 8
    cfg.training.grad_accum_steps = 4  # effective batch = 32
    cfg.training.max_steps = max_steps
    cfg.run_name = "expert_iso_baseline"
    # Disable phased training for simplicity — single-phase training
    cfg.phased.enabled = False

    print(f"Dataset: WikiText-2")
    print(f"Max steps: {max_steps}")
    print(f"Effective batch: {cfg.training.batch_size * cfg.training.grad_accum_steps}")

    # Data
    splits, tokenizer = load_wikitext("wikitext-2", cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # Model
    model = HRSTransformer(cfg).to(device)
    print(f"Parameters: {model.param_count():,}")
    for name, count in model.component_param_counts().items():
        if count > 0:
            print(f"  {name}: {count:,}")

    # Train
    results = simple_train(
        model, loaders, cfg, device, max_steps, output_dir,
        lr=3e-4, warmup_steps=500, eval_interval=2000,
        log_interval=500, tag="[P1]",
    )

    # Save model
    save_path = output_dir / "phase1_baseline.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "results": results,
    }, save_path)
    print(f"Saved baseline to {save_path}")

    # Measure router entropy
    amp_dtype = torch.bfloat16
    router_ent = compute_router_entropy(model, loaders["validation"], device, amp_dtype)
    results["router_entropy"] = router_ent
    print(f"Baseline router entropy: {router_ent:.4f}")

    return model, cfg, loaders, results


# ============================================================
# Phase 2: Isomorphism Analysis
# ============================================================

def phase2_analysis(model, output_dir):
    """Analyze expert weight structure via SVD."""
    print("\n" + "="*70)
    print("PHASE 2: Expert Isomorphism Analysis")
    print("="*70)

    results = {}
    all_layer_results = []

    for layer_idx, block in enumerate(model.blocks):
        if not hasattr(block, 'peer_ffn'):
            continue

        peer = block.peer_ffn
        u = peer.expert_u.data.cpu().float().numpy()  # (N, hd)
        v = peer.expert_v.data.cpu().float().numpy()  # (N, hd)
        N, hd = u.shape

        print(f"\nLayer {layer_idx}: {N} experts, head_dim={hd}")

        # --- SVD of expert_u (input/activation weights) ---
        print(f"  Analyzing expert_u (input weights)...")
        u_centered = u - u.mean(axis=0, keepdims=True)
        U_u, S_u, Vt_u = np.linalg.svd(u_centered, full_matrices=False)

        # Singular value spectrum
        s_u_norm = S_u / S_u.sum()
        cumvar_u = np.cumsum(S_u**2) / np.sum(S_u**2)
        # Effective rank
        p_u = S_u / S_u.sum()
        p_u = p_u[p_u > 1e-10]
        eff_rank_u = np.exp(-np.sum(p_u * np.log(p_u)))

        # Variance of u across experts
        var_u = np.var(u, axis=0).sum()  # total variance

        print(f"    Effective rank: {eff_rank_u:.1f}")
        print(f"    Top-1 SV explains: {cumvar_u[0]*100:.1f}%")
        print(f"    Top-5 SV explain: {cumvar_u[min(4, len(cumvar_u)-1)]*100:.1f}%")
        print(f"    Top-10 SV explain: {cumvar_u[min(9, len(cumvar_u)-1)]*100:.1f}%")
        print(f"    Total variance: {var_u:.6f}")

        # --- SVD of expert_v (output weights) ---
        print(f"  Analyzing expert_v (output weights)...")
        v_centered = v - v.mean(axis=0, keepdims=True)
        U_v, S_v, Vt_v = np.linalg.svd(v_centered, full_matrices=False)

        s_v_norm = S_v / S_v.sum()
        cumvar_v = np.cumsum(S_v**2) / np.sum(S_v**2)
        p_v = S_v / S_v.sum()
        p_v = p_v[p_v > 1e-10]
        eff_rank_v = np.exp(-np.sum(p_v * np.log(p_v)))

        var_v = np.var(v, axis=0).sum()

        print(f"    Effective rank: {eff_rank_v:.1f}")
        print(f"    Top-1 SV explains: {cumvar_v[0]*100:.1f}%")
        print(f"    Top-5 SV explain: {cumvar_v[min(4, len(cumvar_v)-1)]*100:.1f}%")
        print(f"    Top-10 SV explain: {cumvar_v[min(9, len(cumvar_v)-1)]*100:.1f}%")
        print(f"    Total variance: {var_v:.6f}")

        # --- Pairwise cosine similarity within u and v ---
        print(f"  Computing pairwise cosine similarities (sampled)...")
        n_pairs = 50000
        idx1 = np.random.randint(0, N, n_pairs)
        idx2 = np.random.randint(0, N, n_pairs)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        u_norm = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-10)
        v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)

        cos_u = np.sum(u_norm[idx1] * u_norm[idx2], axis=1)
        cos_v = np.sum(v_norm[idx1] * v_norm[idx2], axis=1)

        print(f"    Cosine similarity (u): mean={cos_u.mean():.4f}, std={cos_u.std():.4f}")
        print(f"    Cosine similarity (v): mean={cos_v.mean():.4f}, std={cos_v.std():.4f}")

        # --- Clustering: k-means on SVD fingerprints ---
        print(f"  Clustering experts by SVD fingerprint...")
        from sklearn.cluster import MiniBatchKMeans

        # Use first 10 SVD components as fingerprint
        n_components = min(10, hd)
        u_fingerprint = U_u[:, :n_components] * S_u[:n_components]
        v_fingerprint = U_v[:, :n_components] * S_v[:n_components]

        for n_clusters in [10, 50, 100, 500]:
            km_u = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
            km_u.fit(u_fingerprint)
            inertia_u = km_u.inertia_

            km_v = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
            km_v.fit(v_fingerprint)
            inertia_v = km_v.inertia_

            print(f"    k={n_clusters}: u_inertia={inertia_u:.2f}, v_inertia={inertia_v:.2f}, "
                  f"ratio(v/u)={inertia_v/max(inertia_u, 1e-10):.2f}")

        # --- Variance ratio: how much total variance lives in v vs u ---
        total_var = var_u + var_v
        var_ratio_v = var_v / total_var if total_var > 0 else 0.5

        print(f"\n  VARIANCE DECOMPOSITION:")
        print(f"    u variance: {var_u:.6f} ({(1-var_ratio_v)*100:.1f}%)")
        print(f"    v variance: {var_v:.6f} ({var_ratio_v*100:.1f}%)")
        print(f"    Variance in output layer (v): {var_ratio_v*100:.1f}%")

        layer_result = {
            "layer_idx": layer_idx,
            "n_experts": N,
            "head_dim": hd,
            "u_effective_rank": float(eff_rank_u),
            "v_effective_rank": float(eff_rank_v),
            "u_cumvar_top1": float(cumvar_u[0]),
            "u_cumvar_top5": float(cumvar_u[min(4, len(cumvar_u)-1)]),
            "u_cumvar_top10": float(cumvar_u[min(9, len(cumvar_u)-1)]),
            "v_cumvar_top1": float(cumvar_v[0]),
            "v_cumvar_top5": float(cumvar_v[min(4, len(cumvar_v)-1)]),
            "v_cumvar_top10": float(cumvar_v[min(9, len(cumvar_v)-1)]),
            "u_cosine_mean": float(cos_u.mean()),
            "u_cosine_std": float(cos_u.std()),
            "v_cosine_mean": float(cos_v.mean()),
            "v_cosine_std": float(cos_v.std()),
            "u_variance": float(var_u),
            "v_variance": float(var_v),
            "variance_ratio_v": float(var_ratio_v),
            "u_singular_values_top20": S_u[:20].tolist(),
            "v_singular_values_top20": S_v[:20].tolist(),
        }
        all_layer_results.append(layer_result)

    results["per_layer"] = all_layer_results

    # Aggregate across layers
    if all_layer_results:
        results["mean_u_effective_rank"] = np.mean([r["u_effective_rank"] for r in all_layer_results])
        results["mean_v_effective_rank"] = np.mean([r["v_effective_rank"] for r in all_layer_results])
        results["mean_u_cosine"] = np.mean([r["u_cosine_mean"] for r in all_layer_results])
        results["mean_v_cosine"] = np.mean([r["v_cosine_mean"] for r in all_layer_results])
        results["mean_variance_ratio_v"] = np.mean([r["variance_ratio_v"] for r in all_layer_results])

        print(f"\n{'='*50}")
        print(f"AGGREGATE RESULTS ACROSS {len(all_layer_results)} PEER LAYERS:")
        print(f"  u effective rank (mean): {results['mean_u_effective_rank']:.1f}")
        print(f"  v effective rank (mean): {results['mean_v_effective_rank']:.1f}")
        print(f"  u cosine similarity (mean): {results['mean_u_cosine']:.4f}")
        print(f"  v cosine similarity (mean): {results['mean_v_cosine']:.4f}")
        print(f"  Variance in v (mean): {results['mean_variance_ratio_v']*100:.1f}%")
        print(f"{'='*50}")

    # Save analysis
    with open(output_dir / "phase2_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# Phase 3: Shared Trunk Architecture
# ============================================================

def phase3_shared_trunk(model, cfg, loaders, device, output_dir, max_steps=10000):
    """Build and train shared trunk PEER variant."""
    print("\n" + "="*70)
    print("PHASE 3: Shared Trunk Architecture")
    print("="*70)

    set_seed(42)

    # Build a new model with shared trunk PEER
    shared_model = HRSTransformer(cfg).to(device)

    # Copy all weights from baseline
    shared_model.load_state_dict(model.state_dict())

    # Replace each PEER FFN with SharedTrunkPEER
    for layer_idx, block in enumerate(shared_model.blocks):
        if not hasattr(block, 'peer_ffn'):
            continue

        peer = block.peer_ffn
        u = peer.expert_u.data  # (N, hd)
        v = peer.expert_v.data  # (N, hd)

        # Compute centroid of u vectors as shared trunk
        u_centroid = u.mean(dim=0)  # (hd,)

        print(f"  Layer {layer_idx}: replacing PEER with SharedTrunkPEER")
        print(f"    u centroid norm: {u_centroid.norm():.4f}")
        print(f"    u mean norm: {u.norm(dim=1).mean():.4f}")

        shared_peer = SharedTrunkPEER(
            cfg.model, cfg.peer,
            shared_u=u_centroid,
            expert_v=v,
            keys_a=peer.keys_a.data,
            keys_b=peer.keys_b.data,
            input_proj_weight=peer.input_proj.weight.data,
            output_proj_weight=peer.output_proj.weight.data,
        ).to(device)

        # Copy norm weights
        shared_peer.norm.weight.data.copy_(peer.norm.weight.data)
        shared_peer.norm.bias.data.copy_(peer.norm.bias.data)

        block.peer_ffn = shared_peer

    # Count parameters
    shared_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    baseline_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Baseline parameters: {baseline_params:,}")
    print(f"  Shared trunk parameters: {shared_params:,}")
    print(f"  Reduction: {baseline_params - shared_params:,} ({(1 - shared_params/baseline_params)*100:.1f}%)")

    # Evaluate before fine-tuning
    amp_dtype = torch.bfloat16
    val_ppl_before = evaluate(shared_model, loaders["validation"], device, amp_dtype)
    print(f"\n  Val PPL before fine-tuning: {val_ppl_before:.2f}")

    # Fine-tune
    print(f"\n  Fine-tuning shared trunk model for {max_steps} steps...")
    results = simple_train(
        shared_model, loaders, cfg, device, max_steps, output_dir,
        lr=1e-4, warmup_steps=300, eval_interval=2000,
        log_interval=500, label_smoothing=0.1, tag="[P3]",
    )

    results["val_ppl_before_finetune"] = val_ppl_before
    results["baseline_params"] = baseline_params
    results["shared_params"] = shared_params
    results["param_reduction_pct"] = (1 - shared_params/baseline_params) * 100

    # Save
    save_path = output_dir / "phase3_shared_trunk.pt"
    torch.save({
        "model_state_dict": shared_model.state_dict(),
        "results": results,
    }, save_path)
    print(f"Saved shared trunk model to {save_path}")

    return shared_model, results


# ============================================================
# Phase 4: Router Analysis
# ============================================================

def phase4_router_analysis(baseline_model, shared_model, loaders, device, output_dir):
    """Compare router entropy before and after trunk sharing."""
    print("\n" + "="*70)
    print("PHASE 4: Router Analysis")
    print("="*70)

    amp_dtype = torch.bfloat16

    # Baseline router entropy
    baseline_ent = compute_router_entropy(baseline_model, loaders["validation"], device, amp_dtype)
    print(f"  Baseline router entropy: {baseline_ent:.4f}")

    # Shared trunk router entropy
    shared_ent = compute_router_entropy(shared_model, loaders["validation"], device, amp_dtype)
    print(f"  Shared trunk router entropy: {shared_ent:.4f}")

    entropy_delta = shared_ent - baseline_ent
    print(f"  Delta: {entropy_delta:+.4f}")
    if entropy_delta < 0:
        print(f"  -> Router became MORE confident (lower entropy) with shared trunk")
    else:
        print(f"  -> Router became LESS confident (higher entropy) with shared trunk")

    # Per-layer analysis
    print(f"\n  Per-layer routing distributions:")
    for model_name, mdl in [("baseline", baseline_model), ("shared", shared_model)]:
        mdl.eval()
        layer_entropies = defaultdict(list)
        layer_fractions = defaultdict(list)

        for batch_idx, (x, y) in enumerate(loaders["validation"]):
            if batch_idx >= 10:
                break
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                output = mdl(x)
            for i, rw in enumerate(output.routing_weights):
                rw = rw.float().clamp(min=1e-10)
                ent = -(rw * rw.log()).sum(dim=-1).mean().item()
                layer_entropies[i].append(ent)
                fracs = rw.detach().mean(dim=(0, 1)).cpu().numpy()
                layer_fractions[i].append(fracs)

        print(f"\n  {model_name}:")
        for i in sorted(layer_entropies.keys()):
            mean_ent = np.mean(layer_entropies[i])
            mean_fracs = np.mean(layer_fractions[i], axis=0)
            frac_str = ", ".join(f"{f:.3f}" for f in mean_fracs)
            print(f"    Layer {i}: entropy={mean_ent:.4f}, tier_weights=[{frac_str}]")

    results = {
        "baseline_router_entropy": baseline_ent,
        "shared_router_entropy": shared_ent,
        "entropy_delta": entropy_delta,
    }

    with open(output_dir / "phase4_router.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# Phase 5: Compression Measurement
# ============================================================

def phase5_compression(baseline_model, shared_model, baseline_results,
                       shared_results, loaders, device, output_dir):
    """Final compression analysis."""
    print("\n" + "="*70)
    print("PHASE 5: Compression Measurement")
    print("="*70)

    amp_dtype = torch.bfloat16

    # Parameter counts
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    shared_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    compression_ratio = baseline_params / shared_params

    print(f"  Baseline parameters: {baseline_params:,}")
    print(f"  Shared trunk parameters: {shared_params:,}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Parameter reduction: {(1 - shared_params/baseline_params)*100:.1f}%")

    # Per-component breakdown
    print(f"\n  Parameter breakdown (PEER layers only):")
    for model_name, mdl in [("baseline", baseline_model), ("shared", shared_model)]:
        peer_params = 0
        peer_u_params = 0
        peer_v_params = 0
        for block in mdl.blocks:
            if hasattr(block, 'peer_ffn'):
                peer = block.peer_ffn
                peer_total = sum(p.numel() for p in peer.parameters())
                peer_params += peer_total
                if hasattr(peer, 'expert_u'):
                    peer_u_params += peer.expert_u.numel()
                elif hasattr(peer, 'expert_u_shared'):
                    peer_u_params += peer.expert_u_shared.numel()
                peer_v_params += peer.expert_v.numel()
        print(f"  {model_name}: PEER total={peer_params:,}, u={peer_u_params:,}, v={peer_v_params:,}")

    # Test set perplexity
    print(f"\n  Test set evaluation:")
    baseline_test_ppl = evaluate_test(baseline_model, loaders["test"], device, amp_dtype)
    shared_test_ppl = evaluate_test(shared_model, loaders["test"], device, amp_dtype)
    ppl_delta = shared_test_ppl - baseline_test_ppl
    ppl_delta_pct = (ppl_delta / baseline_test_ppl) * 100

    print(f"  Baseline test PPL: {baseline_test_ppl:.2f}")
    print(f"  Shared trunk test PPL: {shared_test_ppl:.2f}")
    print(f"  PPL delta: {ppl_delta:+.2f} ({ppl_delta_pct:+.1f}%)")

    results = {
        "baseline_params": baseline_params,
        "shared_params": shared_params,
        "compression_ratio": compression_ratio,
        "param_reduction_pct": (1 - shared_params/baseline_params) * 100,
        "baseline_test_ppl": baseline_test_ppl,
        "shared_test_ppl": shared_test_ppl,
        "ppl_delta": ppl_delta,
        "ppl_delta_pct": ppl_delta_pct,
    }

    # Final summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY: Expert Isomorphism in PEER Networks")
    print(f"{'='*70}")
    print(f"  Baseline test PPL:      {baseline_test_ppl:.2f}")
    print(f"  Shared trunk test PPL:  {shared_test_ppl:.2f} ({ppl_delta:+.2f})")
    print(f"  Compression ratio:      {compression_ratio:.2f}x")
    print(f"  Parameter reduction:    {(1 - shared_params/baseline_params)*100:.1f}%")
    print(f"  Baseline params:        {baseline_params:,}")
    print(f"  Shared params:          {shared_params:,}")
    print(f"{'='*70}")

    with open(output_dir / "phase5_compression.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Expert Isomorphism Experiment")
    parser.add_argument("--baseline-steps", type=int, default=15000,
                        help="Training steps for baseline (Phase 1)")
    parser.add_argument("--finetune-steps", type=int, default=10000,
                        help="Fine-tuning steps for shared trunk (Phase 3)")
    parser.add_argument("--output-dir", type=str, default="results/expert_isomorphism")
    parser.add_argument("--resume-phase", type=int, default=1,
                        help="Resume from this phase (1-5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Baseline
    if args.resume_phase <= 1:
        baseline_model, cfg, loaders, baseline_results = phase1_baseline(
            device, output_dir, max_steps=args.baseline_steps,
        )
    else:
        # Load from checkpoint
        print("Loading baseline from checkpoint...")
        ckpt = torch.load(output_dir / "phase1_baseline.pt", map_location=device, weights_only=False)
        cfg = ckpt["config"]
        baseline_results = ckpt["results"]
        splits, _ = load_wikitext("wikitext-2", cfg.model.max_seq_len)
        loaders = build_dataloaders(splits, cfg.training.batch_size)
        baseline_model = HRSTransformer(cfg).to(device)
        baseline_model.load_state_dict(ckpt["model_state_dict"])

    # Phase 2: Isomorphism Analysis
    if args.resume_phase <= 2:
        analysis_results = phase2_analysis(baseline_model, output_dir)
    else:
        with open(output_dir / "phase2_analysis.json") as f:
            analysis_results = json.load(f)

    # Phase 3: Shared Trunk
    if args.resume_phase <= 3:
        shared_model, shared_results = phase3_shared_trunk(
            baseline_model, cfg, loaders, device, output_dir,
            max_steps=args.finetune_steps,
        )
    else:
        ckpt = torch.load(output_dir / "phase3_shared_trunk.pt", map_location=device, weights_only=False)
        shared_results = ckpt["results"]
        # Rebuild shared model
        shared_model = HRSTransformer(cfg).to(device)
        shared_model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Phase 4: Router Analysis
    if args.resume_phase <= 4:
        router_results = phase4_router_analysis(
            baseline_model, shared_model, loaders, device, output_dir,
        )

    # Phase 5: Compression
    compression_results = phase5_compression(
        baseline_model, shared_model, baseline_results, shared_results,
        loaders, device, output_dir,
    )

    # Save complete results
    all_results = {
        "phase1_baseline": {
            "val_ppl": baseline_results["val_ppl"],
            "test_ppl": baseline_results["test_ppl"],
            "router_entropy": baseline_results.get("router_entropy"),
        },
        "phase2_analysis": {
            "mean_u_effective_rank": analysis_results.get("mean_u_effective_rank"),
            "mean_v_effective_rank": analysis_results.get("mean_v_effective_rank"),
            "mean_u_cosine": analysis_results.get("mean_u_cosine"),
            "mean_v_cosine": analysis_results.get("mean_v_cosine"),
            "mean_variance_ratio_v": analysis_results.get("mean_variance_ratio_v"),
        },
        "phase3_shared_trunk": {
            "val_ppl": shared_results["val_ppl"],
            "test_ppl": shared_results["test_ppl"],
            "val_ppl_before_finetune": shared_results.get("val_ppl_before_finetune"),
        },
        "phase5_compression": compression_results,
    }

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
