"""Config for saliency-pooling-vs-attention ablation at Shakespeare scale.

Same scaffold dimensions as the head-aggregation ablation:
  d=384, n_heads=6, head_dim=64, n_layers=6, d_ff=1536, ctx=256.
Note n_heads=6 (not 8) per the saliency spec; head_dim=64.

Parameter targets per layer (4d² = 589,824 for V22 baseline):
  baseline : Q + K + V + W_O = 4d² = 589,824
  A        : V + W_O + saliency_A (d → 768 → 1) ≈ 589,440
  B        : V + W_O + saliency_B (2d → d → 1)  = 589,728
  C        : V + W_O + saliency_C (2d → d → 1)  = 589,728
  D        : V + W_O                             = 294,912 (floor by design)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


VARIANTS: List[str] = [
    "baseline",         # V22 Bonsignore-kernel attention (Q-K based, w/ per-head log_tau, head_alphas, head_output_scalars)
    "sdpa",             # standard scaled-dot-product Q-K attention (no per-head extras) — added for follow-up exp 2
    "A",                # shared global saliency: 1 MLP (d→d_inter→1), single pattern
    "B",                # query-conditioned saliency: 1 MLP (2d→d→1), pair pattern
    "C",                # context-summary saliency: 1 MLP (2d→d→1) on (pos, summary)
    "D",                # pure mean pooling, but with V projection and W_O still present (uniform-weight attention)
    "cumulative_mean",  # NO projections at all: replace attn sublayer with causal cumulative mean of LN'd residual
    "dual_projection",                  # two parallel d→d projections of LN'd residual, concat, compress 2d→d
    "dual_projection_with_cumulative",  # same but P2 operates on the causal cumulative mean of the LN'd residual
]


@dataclass
class ModelConfig:
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    ctx_len: int = 256
    dropout: float = 0.0
    vocab_size: int = -1
    # Saliency MLP hidden width for variant A (single-input MLP). Chosen
    # so total saliency params match 2d²: A_hidden = (2d² − d) / (d + 1)
    # ≈ 2d − 1; we use 768 = 2d for round numbers (≈ 0.13% under target).
    saliency_a_hidden: int = 768
    variant: str = "baseline"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_every: int = 200
    seed: int = 0
