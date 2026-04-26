"""Config for the W_O head-aggregation ablation at Shakespeare scale.

Target: ~10.7M params (per spec's 10.8M). At d=384, H=8, 6 layers,
d_ff=1536, tied embeddings, that lands at 10.74M.

Per-head output path is param-matched to baseline W_O (d×d=147,456 per
layer) within exact precision: per-head MLP (dh=48 → d_inter → dh) plus
up-projection (dh → d). Solving for d_inter:
  H * 2 * dh * d_inter + dh * d = d^2
  8 * 96 * d_inter + 18432 = 147456
  d_inter = 168
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


# All variants. "baseline" = learned W_O. A-F keep per-head MLP fixed and
# vary only the cross-head aggregation operator.
VARIANTS: List[str] = [
    "baseline",   # concat heads → W_O
    "A",          # per-head MLP + mean pool
    "B",          # per-head MLP + sum pool
    "C",          # per-head MLP + elementwise max
    "D",          # per-head MLP + attention pool (learned dh-query over heads)
    "E",          # per-head MLP + L2-normalized mean
    "F",          # per-head MLP + top-k mean (by output L2 norm)
]


@dataclass
class ModelConfig:
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    ctx_len: int = 256
    dropout: float = 0.0
    vocab_size: int = -1
    # Per-head MLP hidden size for variants A-F. 168 matches W_O's param
    # budget exactly at d=384, H=8. See module docstring.
    per_head_hidden: int = 168
    # For variant F (top-k): how many heads to keep.
    topk: int = 4
    # Which variant this config describes.
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
