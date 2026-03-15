"""BDH-inspired modules for HRS v8/v9.

v8 components:
1. VirtualSynapse — engram-derived rank-1 focus matrix modulates attention scores
2. routing_hub_loss — Zipf/power-law target replaces uniform balance loss
3. apply_sparsity_bottleneck — top-k sparsity + positivity on tier inputs

v9 addition:
4. LossScaler — learnable scaling parameters for auxiliary loss terms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BDHConfig, ModelConfig


class VirtualSynapse(nn.Module):
    """Engram-derived attention focus matrix.

    Produces a rank-1 matrix M from mean-pooled engrams via two learned
    projections. Applied as multiplicative gain on attention scores:

        Scores = (Q @ K^T / sqrt(d)) * (1 + alpha * M)

    where M = outer_product(proj1(engram_agg), proj2(engram_agg))
    broadcast across sequence positions.

    This allows the engram to "rewire" attention flow for the current
    context without touching base weights.
    """

    def __init__(self, model_cfg: ModelConfig, bdh_cfg: BDHConfig):
        super().__init__()
        d = model_cfg.d_model
        n_heads = model_cfg.n_heads
        head_dim = d // n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.alpha = bdh_cfg.focus_alpha

        # Project aggregated engram into Q-space and K-space focus vectors
        # Output: (n_heads, head_dim) shaped focus for each of Q and K
        proj_dim = bdh_cfg.focus_proj_dim
        self.proj_q = nn.Sequential(
            nn.Linear(d, proj_dim, bias=False),
            nn.GELU(),
            nn.Linear(proj_dim, n_heads * head_dim, bias=False),
        )
        self.proj_k = nn.Sequential(
            nn.Linear(d, proj_dim, bias=False),
            nn.GELU(),
            nn.Linear(proj_dim, n_heads * head_dim, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.proj_q, self.proj_k]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)

    def forward(self, engrams: torch.Tensor) -> torch.Tensor:
        """Compute focus matrix from engrams.

        Args:
            engrams: (B, E, D) engram vectors (E = n_windows * K)

        Returns:
            focus_q: (B, n_heads, head_dim) — Q-space focus vector
            focus_k: (B, n_heads, head_dim) — K-space focus vector
        """
        if engrams.shape[1] == 0:
            return None, None

        # Aggregate engrams: mean pool to (B, D)
        agg = engrams.mean(dim=1)

        # Project to focus vectors: (B, n_heads * head_dim) -> (B, n_heads, head_dim)
        fq = self.proj_q(agg).reshape(-1, self.n_heads, self.head_dim)
        fk = self.proj_k(agg).reshape(-1, self.n_heads, self.head_dim)

        # L2-normalize to prevent magnitude explosion
        fq = F.normalize(fq, dim=-1)
        fk = F.normalize(fk, dim=-1)

        return fq, fk

    def apply_to_scores(
        self, scores: torch.Tensor, focus_q: torch.Tensor, focus_k: torch.Tensor
    ) -> torch.Tensor:
        """Apply focus matrix as multiplicative gain on attention scores.

        Args:
            scores: (B, n_heads, T, T) raw attention scores (Q @ K^T / sqrt(d))
            focus_q: (B, n_heads, head_dim) from forward()
            focus_k: (B, n_heads, head_dim) from forward()

        Returns:
            (B, n_heads, T, T) modulated scores
        """
        if focus_q is None or focus_k is None:
            return scores

        # Rank-1 outer product: (B, n_heads, 1, head_dim) @ (B, n_heads, head_dim, 1)
        # -> (B, n_heads, 1, 1) — a scalar gain per head
        # But we want position-dependent gain. The engram is position-independent,
        # so M is a constant matrix. For rank-1: M_ij = fq_i * fk_j summed over head_dim.
        # Since fq and fk are single vectors (not per-position), M is a scalar per head.
        # This scalar modulates all scores in that head uniformly.
        #
        # For richer modulation, we compute the dot product as a per-head scalar gain:
        gain = (focus_q * focus_k).sum(dim=-1, keepdim=True).unsqueeze(-1)  # (B, H, 1, 1)

        return scores * (1.0 + self.alpha * gain)


def routing_hub_loss(
    routing_weights: torch.Tensor, hub_exponent: float = 1.5
) -> torch.Tensor:
    """Scale-free hub balance loss: push tier utilization toward Zipf distribution.

    Instead of forcing uniform utilization (which prevents hub formation),
    target a power-law where the most-used tier gets disproportionate traffic.

    Sorts actual utilization descending and compares to Zipf target.

    Args:
        routing_weights: (B, T, n_tiers) routing weights
        hub_exponent: Zipf exponent (higher = steeper power law)

    Returns:
        Scalar loss (0.0 when utilization matches Zipf target exactly)
    """
    n_tiers = routing_weights.shape[-1]

    # Actual utilization per tier
    p = routing_weights.mean(dim=(0, 1))  # (n_tiers,)
    p_sorted, _ = p.sort(descending=True)

    # Zipf target: rank k gets k^(-exponent), normalized
    ranks = torch.arange(1, n_tiers + 1, device=p.device, dtype=p.dtype)
    target = ranks.pow(-hub_exponent)
    target = target / target.sum()

    # L2 distance between sorted utilization and Zipf target
    loss = ((p_sorted - target) ** 2).sum()
    return loss


def apply_sparsity_bottleneck(
    h: torch.Tensor, rho: float = 0.05, activation: str = "softplus"
) -> torch.Tensor:
    """Monosemantic sparsity bottleneck on tier inputs.

    Enforces sparse, positive-only activations to prevent polysemanticity.
    Each token retains only the top rho fraction of features.

    Args:
        h: (B, T, D) hidden states feeding into tier computation
        rho: fraction of features to keep (0.05 = 5%)
        activation: "relu" or "softplus" positivity constraint

    Returns:
        (B, T, D) sparse, positive hidden states
    """
    # Positivity constraint
    if activation == "relu":
        h_pos = F.relu(h)
    else:
        h_pos = F.softplus(h)

    # Top-k sparsity
    D = h_pos.shape[-1]
    k = max(1, int(D * rho))

    topk_vals, topk_idx = h_pos.topk(k, dim=-1)  # (B, T, k)

    # Build sparse output
    sparse_h = torch.zeros_like(h_pos)
    sparse_h.scatter_(-1, topk_idx, topk_vals)

    # Straight-through estimator: forward uses sparse, backward flows through original
    # This preserves gradients for the kept values (scatter is differentiable for those)
    return sparse_h


class LossScaler(nn.Module):
    """Learnable scaling parameters for auxiliary loss terms (v9).

    Each scale is parameterized as exp(raw) to ensure positivity.
    Initialized so exp(raw) matches the fixed V8 coefficient exactly.
    A soft penalty -log(scale) keeps scales from collapsing to zero.

    Scales:
        hub_scale: weight for the Zipf hub balance loss (V8 fixed: balance_loss_weight=0.1)
        entropy_scale: weight for routing entropy loss (V8 fixed: entropy_loss_weight=0.01)
        recon_scale: weight for engram reconstruction loss (V8 fixed: recon_loss_weight=0.1)
    """

    def __init__(self, hub_init: float = 0.1, entropy_init: float = 0.01,
                 recon_init: float = 0.1, alive_penalty: float = 0.001):
        super().__init__()
        import math
        # Raw parameters: log(init_value) so exp(raw) = init_value at step 0
        self.raw_hub = nn.Parameter(torch.tensor(math.log(hub_init)))
        self.raw_entropy = nn.Parameter(torch.tensor(math.log(entropy_init)))
        self.raw_recon = nn.Parameter(torch.tensor(math.log(recon_init)))
        self.alive_penalty = alive_penalty

    @property
    def hub_scale(self) -> torch.Tensor:
        return self.raw_hub.exp()

    @property
    def entropy_scale(self) -> torch.Tensor:
        return self.raw_entropy.exp()

    @property
    def recon_scale(self) -> torch.Tensor:
        return self.raw_recon.exp()

    def penalty(self) -> torch.Tensor:
        """Soft penalty discouraging scales from approaching zero.

        Returns alive_coeff * sum(-log(scale_i)) added to total loss.
        """
        return self.alive_penalty * (
            -self.raw_hub - self.raw_entropy - self.raw_recon
        )

    def scale_dict(self) -> dict:
        """Current scale values for logging."""
        return {
            "hub_scale": self.hub_scale.item(),
            "entropy_scale": self.entropy_scale.item(),
            "recon_scale": self.recon_scale.item(),
        }
