"""Four attention variants with matched V/O pathways.

Only the score computation differs. V and O projections are standard multi-head
in every variant, so the test isolates the pair-interaction structure.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.diagonal_attention.config import ModelConfig


class _AttentionBase(nn.Module):
    """Shared V/O plumbing.

    Subclasses implement compute_scores(x) -> either (B, T, T) shared across
    heads or (B, H, T, T) per-head. The base class handles masking, softmax,
    value projection, and output projection identically in all cases.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)

    def compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H, dh = self.cfg.n_heads, self.cfg.d_head

        scores = self.compute_scores(x)
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        V = self.W_V(x).view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)

        if attn.dim() == 3:
            # Shared attention map across heads.
            out = torch.einsum("bts,bhsd->bhtd", attn, V)
        else:
            out = torch.einsum("bhts,bhsd->bhtd", attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)


class StandardMHA(_AttentionBase):
    """Variant 1: scores = Q K^T / sqrt(d_head), per-head projections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        d = cfg.d_model
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)

    def compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H, dh = self.cfg.n_heads, self.cfg.d_head
        Q = self.W_Q(x).view(B, T, H, dh).transpose(1, 2)
        K = self.W_K(x).view(B, T, H, dh).transpose(1, 2)
        return (Q @ K.transpose(-2, -1)) / math.sqrt(dh)

    def score_params(self) -> int:
        return sum(p.numel() for p in [self.W_Q.weight, self.W_K.weight])


class FullBilinear(_AttentionBase):
    """Variant 2: scores = X W X^T / sqrt(d), shared across heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        d = cfg.d_model
        W = torch.empty(d, d)
        nn.init.xavier_normal_(W)
        self.W = nn.Parameter(W)

    def compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        xW = x @ self.W
        return (xW @ x.transpose(-2, -1)) / math.sqrt(self.cfg.d_model)

    def score_params(self) -> int:
        return self.W.numel()


class DiagonalBilinear(_AttentionBase):
    """Variant 3: scores = (X ⊙ w) X^T / sqrt(d), learned d-length vector.

    Shared diagonal across heads (spec says start with shared).
    Initialized to 1.0 so the initial behavior equals the Identity variant.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.w = nn.Parameter(torch.ones(cfg.d_model))

    def compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        xw = x * self.w
        return (xw @ x.transpose(-2, -1)) / math.sqrt(self.cfg.d_model)

    def score_params(self) -> int:
        return self.w.numel()


class Identity(_AttentionBase):
    """Variant 4: scores = X X^T / sqrt(d). Sanity floor."""

    def compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ x.transpose(-2, -1)) / math.sqrt(self.cfg.d_model)

    def score_params(self) -> int:
        return 0


def build_attention(cfg: ModelConfig) -> _AttentionBase:
    v = cfg.variant.lower()
    if v == "mha":
        return StandardMHA(cfg)
    if v == "bilinear":
        return FullBilinear(cfg)
    if v == "diagonal":
        return DiagonalBilinear(cfg)
    if v == "identity":
        return Identity(cfg)
    raise ValueError(f"unknown variant: {cfg.variant}")
