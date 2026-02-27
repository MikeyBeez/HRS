"""Metrics for HRS experiments.

Includes HRS-specific metrics (routing stats, FLOPs estimation) plus
reusable representation quality metrics from LFP.
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


# --- HRS-specific metrics ---

@torch.no_grad()
def routing_entropy(routing_weights: torch.Tensor) -> dict:
    """Compute entropy of routing decisions per layer.

    Lower entropy = more decisive routing (good).
    Higher entropy = uniform routing (router hasn't learned).

    Args:
        routing_weights: (B, T, n_tiers) routing weights

    Returns:
        Dict with mean_entropy and per_tier_fraction
    """
    # Per-token routing entropy
    rw = routing_weights.clamp(min=1e-10)
    entropy = -(rw * rw.log()).sum(dim=-1)  # (B, T)

    # Fraction of tokens per tier (hard assignment)
    assignments = routing_weights.argmax(dim=-1)  # (B, T)
    n_tiers = routing_weights.shape[-1]
    tier_fractions = []
    total = assignments.numel()
    for t in range(n_tiers):
        frac = (assignments == t).sum().item() / total
        tier_fractions.append(frac)

    return {
        "mean_entropy": entropy.mean().item(),
        "min_entropy": entropy.min().item(),
        "tier_fractions": tier_fractions,  # [conv%, expert%, attn%, sink%]
    }


@torch.no_grad()
def routing_distribution(routing_weights_list: list) -> dict:
    """Aggregate routing stats across all layers.

    Args:
        routing_weights_list: list of (B, T, n_tiers) per layer

    Returns:
        Dict with per-layer and mean routing stats
    """
    if not routing_weights_list:
        return {}

    per_layer = []
    for i, rw in enumerate(routing_weights_list):
        stats = routing_entropy(rw)
        per_layer.append(stats)

    # Aggregate
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
    """Max/min component norm ratio across tiers.

    Detects if one tier monopolizes output.
    Ratio > 3 = concerning, ratio > 10 = one tier dominates.

    Args:
        routing_weights_list: list of (B, T, n_tiers) per layer

    Returns:
        Mean magnitude ratio across layers
    """
    if not routing_weights_list:
        return 1.0

    ratios = []
    for rw in routing_weights_list:
        # Mean weight per tier
        mean_per_tier = rw.mean(dim=(0, 1))  # (n_tiers,)
        max_w = mean_per_tier.max().item()
        min_w = mean_per_tier.min().item()
        # Clamp min to avoid explosion with peaked routing
        ratio = max_w / max(min_w, 0.01)
        ratios.append(ratio)

    return sum(ratios) / len(ratios)


@torch.no_grad()
def estimate_flops_per_token(routing_weights: torch.Tensor, d_model: int, d_ff: int, seq_len: int, n_experts: int = 4) -> float:
    """Estimate FLOPs per token based on routing decisions.

    Args:
        routing_weights: (B, T, n_tiers) routing weights
        d_model: model dimension
        d_ff: feedforward dimension
        seq_len: sequence length
        n_experts: number of experts

    Returns:
        Estimated FLOPs per token (relative to dense baseline)
    """
    # Hard assignments for FLOPs estimation
    assignments = routing_weights.argmax(dim=-1)  # (B, T)
    total = assignments.numel()

    # Approximate FLOPs per tier per token
    # Conv: O(d * kernel_size) ~ d * 7
    conv_flops = d_model * 7
    # Expert: O(d * expert_hidden * 2) ~ d * 256 * 2 (one expert, 2 layers)
    expert_flops = d_model * 256 * 2
    # Attention: O(d * seq_len) for full attention
    attn_flops = d_model * seq_len
    # Sink: O(d) ~ just scaling
    sink_flops = d_model

    # Dense baseline: attention + FFN per token
    dense_flops = d_model * seq_len + d_model * d_ff * 2

    tier_flops = [conv_flops, expert_flops, attn_flops, sink_flops]

    weighted_flops = 0.0
    for t in range(4):
        frac = (assignments == t).sum().item() / total
        weighted_flops += frac * tier_flops[t]

    # Return ratio relative to dense baseline
    return weighted_flops / dense_flops if dense_flops > 0 else 1.0


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

        # CE loss for perplexity
        B, T, V = output.logits.shape
        ce = F.cross_entropy(
            output.logits.reshape(B * T, V),
            y.reshape(B * T),
            reduction="sum",
        )
        total_ce += ce.item()
        total_tokens += B * T

        # Collect representations
        for i, rep in enumerate(output.layer_representations):
            if i < n_layers:
                all_layer_reps[i].append(rep.float().cpu())

        # Collect routing weights
        if output.routing_weights:
            all_routing_weights.append(
                [rw.float().cpu() for rw in output.routing_weights]
            )

        # Collect attention weights
        for i, aw in enumerate(output.attention_weights):
            if aw is not None and i < n_layers:
                all_attn_weights[i].append(aw.float().cpu())

        all_targets.append(y.cpu())

    targets_flat = torch.cat(all_targets).reshape(-1)
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

    # Routing metrics
    if all_routing_weights:
        # Aggregate routing weights across batches (use last batch for simplicity)
        last_rw = all_routing_weights[-1]
        routing_stats = routing_distribution(last_rw)
        metrics.update(routing_stats)
        metrics["magnitude_ratio"] = output_magnitude_ratio(last_rw)

        # FLOPs estimation
        if last_rw:
            metrics["flops_ratio"] = estimate_flops_per_token(
                last_rw[0],  # first layer
                model.cfg.model.d_model,
                model.cfg.model.d_ff,
                model.cfg.model.max_seq_len,
            )

    # Attention entropy (from backbone attention, not tier attention)
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
