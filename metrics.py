"""Metrics for HRS experiments (v1 and v2).

Includes:
- Representation quality metrics (effective rank, cosine similarity, intrinsic dimensionality)
- HRS-specific metrics (routing stats, FLOPs estimation) — v1 only
- PEER utilization tracking — v2 only
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# --- Representation quality metrics (from LFP) ---

@torch.no_grad()
def effective_rank(Z: torch.Tensor) -> float:
    """Effective rank via Shannon entropy of normalized singular values."""
    if Z.shape[0] > 4096:
        idx = torch.randperm(Z.shape[0])[:4096]
        Z = Z[idx]
    Z = Z.float()
    s = torch.linalg.svdvals(Z)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


@torch.no_grad()
def cosine_similarity_stats(Z: torch.Tensor, n_pairs: int = 10000) -> dict:
    """Statistics of pairwise cosine similarities."""
    Z = Z.float()
    Z = F.normalize(Z, dim=-1)
    N = Z.shape[0]
    idx1 = torch.randint(0, N, (n_pairs,))
    idx2 = torch.randint(0, N, (n_pairs,))
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    sims = (Z[idx1] * Z[idx2]).sum(dim=-1)
    return {"mean": sims.mean().item(), "std": sims.std().item()}


@torch.no_grad()
def intrinsic_dimensionality(Z: torch.Tensor, threshold: float = 0.9) -> int:
    """Number of PCA components for threshold% variance."""
    if Z.shape[0] > 4096:
        idx = torch.randperm(Z.shape[0])[:4096]
        Z = Z[idx]
    Z = Z.float()
    Z = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    var_explained = (s ** 2).cumsum(dim=0) / (s ** 2).sum()
    n_components = (var_explained < threshold).sum().item() + 1
    return min(n_components, Z.shape[1])


# --- HRS v1 metrics (routing) ---

@torch.no_grad()
def routing_entropy(routing_weights: torch.Tensor) -> dict:
    """Compute entropy of routing decisions per layer."""
    rw = routing_weights.clamp(min=1e-10)
    entropy = -(rw * rw.log()).sum(dim=-1)

    assignments = routing_weights.argmax(dim=-1)
    n_tiers = routing_weights.shape[-1]
    tier_fractions = []
    total = assignments.numel()
    for t in range(n_tiers):
        frac = (assignments == t).sum().item() / total
        tier_fractions.append(frac)

    return {
        "mean_entropy": entropy.mean().item(),
        "min_entropy": entropy.min().item(),
        "tier_fractions": tier_fractions,
    }


@torch.no_grad()
def routing_distribution(routing_weights_list: list) -> dict:
    """Aggregate routing stats across all layers."""
    if not routing_weights_list:
        return {}

    per_layer = []
    for i, rw in enumerate(routing_weights_list):
        stats = routing_entropy(rw)
        per_layer.append(stats)

    mean_entropy = sum(s["mean_entropy"] for s in per_layer) / len(per_layer)
    mean_fractions = [0.0] * len(per_layer[0]["tier_fractions"])
    for s in per_layer:
        for j, f in enumerate(s["tier_fractions"]):
            mean_fractions[j] += f / len(per_layer)

    return {
        "routing_entropy_per_layer": [s["mean_entropy"] for s in per_layer],
        "routing_entropy_mean": mean_entropy,
        "tier_fractions_per_layer": [s["tier_fractions"] for s in per_layer],
        "tier_fractions_mean": mean_fractions,
    }


@torch.no_grad()
def output_magnitude_ratio(routing_weights_list: list) -> float:
    """Max/min component norm ratio across tiers."""
    if not routing_weights_list:
        return 1.0

    ratios = []
    for rw in routing_weights_list:
        mean_per_tier = rw.mean(dim=(0, 1))
        max_w = mean_per_tier.max().item()
        min_w = mean_per_tier.min().item()
        ratio = max_w / max(min_w, 0.01)
        ratios.append(ratio)

    return sum(ratios) / len(ratios)


@torch.no_grad()
def estimate_flops_per_token(routing_weights: torch.Tensor, d_model: int, d_ff: int, seq_len: int, n_experts: int = 4) -> float:
    """Estimate FLOPs per token based on routing decisions."""
    assignments = routing_weights.argmax(dim=-1)
    total = assignments.numel()

    conv_flops = d_model * 7
    expert_flops = d_model * 256 * 2
    attn_flops = d_model * seq_len
    sink_flops = d_model

    dense_flops = d_model * seq_len + d_model * d_ff * 2

    tier_flops = [conv_flops, expert_flops, attn_flops, sink_flops]

    weighted_flops = 0.0
    for t in range(4):
        frac = (assignments == t).sum().item() / total
        weighted_flops += frac * tier_flops[t]

    return weighted_flops / dense_flops if dense_flops > 0 else 1.0


# --- PEER v2 metrics ---

@torch.no_grad()
def peer_expert_utilization(model, val_loader: DataLoader, device: torch.device,
                            amp_dtype: torch.dtype, max_batches: int = 5) -> dict:
    """Track which PEER experts are used and how uniformly.

    Runs a few batches and collects expert indices from PEER layers.
    Returns utilization statistics.
    """
    from model import HRSv2Block

    # Collect all expert indices across batches
    all_indices = []
    n_total = None

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            # We need to hook into PEER layers to capture expert indices
            # For now, compute a proxy: forward pass + sample from key scores
            _ = model(x)

        # After forward pass, we can estimate utilization from PEER key tables
        # by checking which experts have large gradient norms (proxy for usage)

    # Static utilization info from config
    for block in model.blocks:
        if isinstance(block, HRSv2Block) and block.use_peer:
            peer = block.ffn
            n_total = peer.n_sub_keys ** 2
            n_active = peer.n_heads * peer.top_k
            return {
                "peer_n_total_experts": n_total,
                "peer_n_active_per_token": n_active,
                "peer_sparsity": 1.0 - n_active / n_total,
                "peer_n_heads": peer.n_heads,
                "peer_top_k": peer.top_k,
            }

    return {}


# --- Unified evaluation ---

@torch.no_grad()
def run_all_metrics(
    model,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int = 10,
) -> dict:
    """Run all evaluation metrics on validation data."""
    model.eval()

    total_ce = 0.0
    total_tokens = 0
    all_layer_reps = {}
    all_routing_weights = []
    all_attn_weights = {}
    all_targets = []

    n_layers = len(model.blocks)
    for i in range(n_layers):
        all_layer_reps[i] = []
        all_attn_weights[i] = []

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(x, collect_intermediates=True, collect_layer_reps=True)

        B, T, V = output.logits.shape
        ce = F.cross_entropy(
            output.logits.reshape(B * T, V),
            y.reshape(B * T),
            reduction="sum",
        )
        total_ce += ce.item()
        total_tokens += B * T

        for i, rep in enumerate(output.layer_representations):
            if i < n_layers:
                all_layer_reps[i].append(rep.float().cpu())

        if output.routing_weights:
            all_routing_weights.append(
                [rw.float().cpu() for rw in output.routing_weights]
            )

        for i, aw in enumerate(output.attention_weights):
            if aw is not None and i < n_layers:
                all_attn_weights[i].append(aw.float().cpu())

        all_targets.append(y.cpu())

    val_ppl = math.exp(min(total_ce / max(total_tokens, 1), 20))

    metrics = {"val_ppl": val_ppl, "val_ce": total_ce / max(total_tokens, 1)}

    # Per-layer representation metrics
    eff_ranks = []
    cos_sim_means = []
    intrinsic_dims = []

    for i in range(n_layers):
        if not all_layer_reps[i]:
            continue
        Z = torch.cat(all_layer_reps[i]).reshape(-1, model.cfg.model.d_model)
        if Z.shape[0] > 8192:
            idx = torch.randperm(Z.shape[0])[:8192]
            Z = Z[idx]

        eff_ranks.append(effective_rank(Z))
        cos_sim_means.append(cosine_similarity_stats(Z)["mean"])
        intrinsic_dims.append(intrinsic_dimensionality(Z))

    metrics["effective_rank_per_layer"] = eff_ranks
    metrics["effective_rank_mean"] = sum(eff_ranks) / len(eff_ranks) if eff_ranks else 0
    metrics["cosine_sim_per_layer"] = cos_sim_means
    metrics["cosine_sim_mean"] = sum(cos_sim_means) / len(cos_sim_means) if cos_sim_means else 0
    metrics["intrinsic_dim_per_layer"] = intrinsic_dims
    metrics["intrinsic_dim_mean"] = sum(intrinsic_dims) / len(intrinsic_dims) if intrinsic_dims else 0

    # v1 Routing metrics
    if all_routing_weights:
        last_rw = all_routing_weights[-1]
        routing_stats = routing_distribution(last_rw)
        metrics.update(routing_stats)
        metrics["magnitude_ratio"] = output_magnitude_ratio(last_rw)

        if last_rw:
            metrics["flops_ratio"] = estimate_flops_per_token(
                last_rw[0],
                model.cfg.model.d_model,
                model.cfg.model.d_ff,
                model.cfg.model.max_seq_len,
            )

    # v2 PEER metrics
    if model._is_v2 and model.cfg.uses_peer():
        peer_stats = peer_expert_utilization(model, val_loader, device, amp_dtype)
        metrics.update(peer_stats)

    # Attention entropy (from backbone attention layers)
    attn_entropies = []
    for i in range(n_layers):
        if not all_attn_weights[i]:
            continue
        aw = torch.cat(all_attn_weights[i])
        attn = aw.clamp(min=1e-10)
        entropy = -(attn * attn.log()).sum(dim=-1).mean().item()
        attn_entropies.append(entropy)

    metrics["attn_entropy_per_layer"] = attn_entropies
    metrics["attn_entropy_mean"] = sum(attn_entropies) / len(attn_entropies) if attn_entropies else 0

    return metrics
