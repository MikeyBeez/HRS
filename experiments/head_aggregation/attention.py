"""Bonsignore-kernel attention with swappable output aggregation.

Kernel: scores = -||q - k||^2 / tau_h per head, softmax over keys, causal.
Per-head temperature `log_tau`, per-head sharpness `head_alphas` (sigmoid),
per-head output scaling `head_output_scalars` (softplus). These match V22
exactly. The per-head *score-refinement* MLP from V22 is intentionally
omitted — it's orthogonal to the W_O question and keeps debug surface tight.

Baseline: concat heads → W_O (d × d).

Variants A-F: each head's post-scaling output is run through a shared-structure,
parameter-matched per-head MLP (dh → d_inter → dh, GeLU), then the H outputs
are aggregated across the head dimension with a fixed operator (variant-specific),
then a single up-projection (dh → d) writes to residual-stream dim.

Parameter match: baseline W_O has d*d params per layer. The per-head path
has H * 2 * dh * d_inter + dh * d params. With d_inter = 168 at d=384, H=8,
both sides equal 147,456.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.head_aggregation.config import ModelConfig


class BonsignoreAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d = cfg.d_model
        H = cfg.n_heads
        dh = self.head_dim
        self.variant = cfg.variant

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.log_tau = nn.Parameter(torch.full((H,), math.log(float(dh))))
        self.head_alphas = nn.Parameter(torch.ones(H))
        self.head_output_scalars = nn.Parameter(torch.zeros(H))
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        if self.variant == "baseline":
            # Standard: concat heads → W_O
            self.out_proj = nn.Linear(d, d, bias=False)
        else:
            # Per-head path: MLP (dh → d_inter → dh) per head, then aggregate, then up-proj.
            d_inter = cfg.per_head_hidden
            # (H, dh, d_inter) and (H, d_inter, dh); einsum'd per head.
            self.Wh_in = nn.Parameter(torch.empty(H, dh, d_inter))
            self.Wh_out = nn.Parameter(torch.empty(H, d_inter, dh))
            nn.init.normal_(self.Wh_in, std=0.02)
            nn.init.normal_(self.Wh_out, std=0.02)
            self.up_proj = nn.Linear(dh, d, bias=False)
            if self.variant == "D":
                # Single learned dh-dim query vector; dot against each head's
                # output to produce H softmax weights per (B, T) position.
                self.q_pool = nn.Parameter(torch.randn(dh) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Bonsignore kernel
        q_sq = (q ** 2).sum(dim=-1, keepdim=True)              # (B, H, T, 1)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)              # (B, H, T, 1)
        dot = q @ k.transpose(-2, -1)                           # (B, H, T, T)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot     # (B, H, T, T)
        taus = self.log_tau.exp().view(1, H, 1, 1)
        scores = -distances / taus
        alphas = torch.sigmoid(self.head_alphas).view(1, H, 1, 1)
        scores = scores * alphas

        # Causal
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v                                          # (B, H, T, dh)

        # Per-head output scaling (V22 parity)
        head_scales = F.softplus(self.head_output_scalars).view(1, H, 1, 1)
        out = out * head_scales

        if self.variant == "baseline":
            out = out.transpose(1, 2).reshape(B, T, D)          # concat heads
            return self.resid_dropout(self.out_proj(out))

        # Per-head MLP: dh → d_inter → dh (GeLU)
        h1 = torch.einsum("bhtd,hde->bhte", out, self.Wh_in)
        h1 = F.gelu(h1)
        h2 = torch.einsum("bhte,hed->bhtd", h1, self.Wh_out)    # (B, H, T, dh)

        # Aggregate across heads
        v_ = self.variant
        if v_ == "A":
            agg = h2.mean(dim=1)                                # (B, T, dh)
        elif v_ == "B":
            agg = h2.sum(dim=1)
        elif v_ == "C":
            agg = h2.max(dim=1).values
        elif v_ == "D":
            # Attention-pool: softmax over H heads for each (B, T) using dot
            # product against learned dh-dim query.
            pool_scores = torch.einsum("bhtd,d->bht", h2, self.q_pool)   # (B, H, T)
            pool_w = F.softmax(pool_scores, dim=1)                        # softmax over H
            agg = torch.einsum("bht,bhtd->btd", pool_w, h2)
        elif v_ == "E":
            agg = h2.mean(dim=1)
            agg = F.normalize(agg, p=2, dim=-1)
        elif v_ == "F":
            # Top-k mean: for each (B, T), keep top-k heads by output L2 norm.
            norms = h2.norm(dim=-1)                             # (B, H, T)
            _, top_idx = norms.topk(self.cfg.topk, dim=1)       # (B, k, T)
            mask = torch.zeros_like(norms, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
            h2_masked = h2 * mask.unsqueeze(-1).to(h2.dtype)
            agg = h2_masked.sum(dim=1) / float(self.cfg.topk)
        else:
            raise ValueError(f"unknown variant: {v_}")

        return self.resid_dropout(self.up_proj(agg))

    # ------------------------------------------------------------
    # Diagnostics — snapshot norms/scalars for the hypothesis-mechanism log
    # ------------------------------------------------------------
    def diagnostics(self) -> dict:
        out = {
            "head_output_scalars_softplus": F.softplus(
                self.head_output_scalars.detach()
            ).tolist(),
            "head_alphas_sigmoid": torch.sigmoid(self.head_alphas.detach()).tolist(),
            "taus": self.log_tau.exp().detach().tolist(),
        }
        if self.variant != "baseline":
            out["Wh_in_frobenius"] = float(self.Wh_in.detach().norm().item())
            out["Wh_out_frobenius"] = float(self.Wh_out.detach().norm().item())
            out["up_proj_frobenius"] = float(self.up_proj.weight.detach().norm().item())
            # Per-head norms (to see if individual heads are differentiating)
            out["Wh_in_perhead_frob"] = [
                float(self.Wh_in.detach()[h].norm().item()) for h in range(self.n_heads)
            ]
            out["Wh_out_perhead_frob"] = [
                float(self.Wh_out.detach()[h].norm().item()) for h in range(self.n_heads)
            ]
            if self.variant == "D":
                out["q_pool_norm"] = float(self.q_pool.detach().norm().item())
        else:
            out["out_proj_frobenius"] = float(self.out_proj.weight.detach().norm().item())
        return out

    def output_path_params(self) -> int:
        """Parameter count in the output pathway (for budget verification)."""
        if self.variant == "baseline":
            return self.out_proj.weight.numel()
        n = self.Wh_in.numel() + self.Wh_out.numel() + self.up_proj.weight.numel()
        if self.variant == "D":
            n += self.q_pool.numel()
        return n
