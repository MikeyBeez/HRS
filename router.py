"""Learned router for HRS token-to-tier assignment.

Each layer has a small MLP that produces routing weights over 4 tiers:
conv, expert, attention, sink. Uses softmax routing with three auxiliary
losses:

1. Balance loss: prevents routing collapse (pushes toward uniform globally).
2. Entropy loss: prevents over-confident per-token decisions (exploration).
3. FLOPs loss: penalizes expensive routing, making the compute tradeoff
   differentiable and learnable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RouterConfig, ModelConfig


class TokenRouter(nn.Module):
    """Per-layer router that assigns tokens to compute tiers.

    Architecture: 2-layer MLP (d_model -> hidden -> n_tiers)
    Output: soft routing weights via temperature-scaled softmax

    Optional TRC (Temporal Routing Cache): causal moving average over
    router logits, smoothing routing decisions across adjacent tokens.
    """

    def __init__(self, model_cfg: ModelConfig, router_cfg: RouterConfig):
        super().__init__()
        self.n_tiers = router_cfg.n_tiers
        self.gumbel_tau = router_cfg.gumbel_tau
        self.gumbel_tau_min = router_cfg.gumbel_tau_min
        self.gumbel_anneal_steps = router_cfg.gumbel_anneal_steps
        self.trc_enabled = router_cfg.trc_enabled
        self.trc_window = router_cfg.trc_window

        self.mlp = nn.Sequential(
            nn.Linear(model_cfg.d_model, router_cfg.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(router_cfg.hidden_dim, router_cfg.n_tiers, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def get_tau(self, step: int) -> float:
        """Anneal temperature from tau to tau_min over gumbel_anneal_steps."""
        if step >= self.gumbel_anneal_steps:
            return self.gumbel_tau_min
        progress = step / self.gumbel_anneal_steps
        return self.gumbel_tau + (self.gumbel_tau_min - self.gumbel_tau) * progress

    def _trc_smooth(self, logits: torch.Tensor) -> torch.Tensor:
        """Causal moving average over token positions (TRC low-pass filter).

        Smooths router logits so adjacent tokens receive similar routing,
        exploiting temporal coherence in natural language (e.g., code blocks,
        math derivations route similarly).

        Args:
            logits: (B, T, K) raw router logits

        Returns:
            (B, T, K) smoothed logits (causal — only uses past/current positions)
        """
        B, T, K = logits.shape
        W = min(self.trc_window, T)

        # Causal moving average via cumulative sum
        # smoothed[t] = mean(logits[max(0,t-W+1):t+1])
        cumsum = logits.cumsum(dim=1)  # (B, T, K)
        # Shift right by W positions to get cumsum[t-W]
        shifted = F.pad(cumsum, (0, 0, W, 0))[:, :T]  # (B, T, K)
        window_sum = cumsum - shifted

        # Counts: min(t+1, W) for each position t
        counts = torch.arange(1, T + 1, device=logits.device, dtype=logits.dtype)
        counts = counts.clamp(max=W).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        return window_sum / counts

    def forward(
        self, x: torch.Tensor, step: int = 0, hard: bool = False
    ) -> torch.Tensor:
        """Compute routing weights for all tokens.

        Args:
            x: (B, T, d_model) hidden states
            step: current training step (for temperature annealing)
            hard: use hard routing (straight-through estimator)

        Returns:
            (B, T, n_tiers) routing weights summing to 1 per token
        """
        logits = self.mlp(x)  # (B, T, n_tiers)

        # TRC: causal moving average smooths routing across adjacent tokens
        if self.trc_enabled:
            logits = self._trc_smooth(logits)

        tau = self.get_tau(step) if self.training else self.gumbel_tau_min

        if self.training:
            # Gumbel-softmax: differentiable sampling with annealed temperature
            B, T, K = logits.shape
            logits_flat = logits.reshape(B * T, K)
            routing = F.gumbel_softmax(logits_flat, tau=tau, hard=hard, dim=-1)
            routing = routing.reshape(B, T, K)
        else:
            # Deterministic softmax at eval (temperature-scaled)
            routing = F.softmax(logits / tau, dim=-1)

        return routing


def routing_balance_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    """Fully differentiable load-balancing loss (global tier balance).

    Uses squared-mean formulation: n_tiers * sum(p_i^2) where
    p_i = mean routing probability for tier i across all tokens.

    Minimized (= 1.0) when p_i = 1/n_tiers (uniform).

    Args:
        routing_weights: (B, T, n_tiers) routing weights

    Returns:
        Scalar balance loss (1.0 when perfectly balanced)
    """
    n_tiers = routing_weights.shape[-1]
    p = routing_weights.mean(dim=(0, 1))  # (n_tiers,)
    balance_loss = (p * p).sum() * n_tiers
    return balance_loss


def routing_entropy_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    """Negative per-token routing entropy (exploration regularizer).

    Encourages each token to maintain uncertainty about which tier to
    use, preventing the router from committing to hard decisions before
    the tiers have had time to specialize.

    Returns negative mean entropy so that minimizing this loss
    maximizes per-token entropy. Normalized to [-1, 0] range by
    dividing by max_entropy = log(n_tiers).

    Args:
        routing_weights: (B, T, n_tiers) routing weights

    Returns:
        Scalar in [-1, 0]. More negative = higher entropy = more exploration.
    """
    import math
    n_tiers = routing_weights.shape[-1]
    max_entropy = math.log(n_tiers)

    rw = routing_weights.clamp(min=1e-10)
    per_token_entropy = -(rw * rw.log()).sum(dim=-1)  # (B, T)

    # Normalize by max possible entropy and negate
    return -(per_token_entropy.mean() / max_entropy)


# Relative FLOPs cost per tier (normalized so dense baseline = 1.0).
# Architecture-derived constants for d_model=512, d_ff=2048,
# seq_len=512, kernel=7, expert_hidden=256.
#   dense baseline per token = attn(d*T) + ffn(2*d*d_ff) = 262144 + 2097152 = 2359296
#   conv:    d * kernel        =   3584  -> 0.0015
#   expert:  d * expert_h * 2  = 262144  -> 0.111
#   attn:    d * T             = 262144  -> 0.111
#   sink:    d (just scaling)  =    512  -> 0.0002
TIER_FLOPS_COST = [0.0015, 0.111, 0.111, 0.0002]


def routing_flops_loss(
    routing_weights: torch.Tensor,
    tier_costs: list[float] = None,
) -> torch.Tensor:
    """Differentiable FLOPs cost of routing decisions.

    Computes the expected compute cost as a weighted sum of tier costs:
        L_flops = sum_i(p_i * cost_i)

    Minimized by routing everything to sink (cheapest), but combined
    with CE loss, the model learns the quality/compute tradeoff.

    Args:
        routing_weights: (B, T, n_tiers) routing weights
        tier_costs: per-tier relative FLOPs costs. Defaults to TIER_FLOPS_COST.

    Returns:
        Scalar expected FLOPs ratio (0.0 = all sink, ~0.11 = all attention)
    """
    if tier_costs is None:
        tier_costs = TIER_FLOPS_COST

    costs = torch.tensor(tier_costs, device=routing_weights.device, dtype=routing_weights.dtype)
    p = routing_weights.mean(dim=(0, 1))  # (n_tiers,)
    return (p * costs).sum()
