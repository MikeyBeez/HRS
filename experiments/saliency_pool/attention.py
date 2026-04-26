"""Saliency-pooling attention variants + V22 Bonsignore baseline.

All variants share the V projection and W_O. Only the scoring mechanism
that produces the position-weighting differs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.saliency_pool.config import ModelConfig


# ============================================================
# Baseline: V22 Bonsignore-kernel attention (same as head_aggregation
# baseline, kept here so the saliency_pool module is self-contained).
# ============================================================
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

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.full((H,), math.log(float(dh))))
        self.head_alphas = nn.Parameter(torch.ones(H))
        self.head_output_scalars = nn.Parameter(torch.zeros(H))
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_sq = (q ** 2).sum(dim=-1, keepdim=True)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)
        dot = q @ k.transpose(-2, -1)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot
        taus = self.log_tau.exp().view(1, H, 1, 1)
        scores = -distances / taus
        alphas = torch.sigmoid(self.head_alphas).view(1, H, 1, 1)
        scores = scores * alphas

        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v

        head_scales = F.softplus(self.head_output_scalars).view(1, H, 1, 1)
        out = out * head_scales

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_dropout(self.out_proj(out))

    def diagnostics(self) -> dict:
        return {
            "head_output_scalars_softplus": F.softplus(
                self.head_output_scalars.detach()
            ).tolist(),
            "head_alphas_sigmoid": torch.sigmoid(self.head_alphas.detach()).tolist(),
            "taus": self.log_tau.exp().detach().tolist(),
        }

    def output_path_params(self) -> int:
        return (self.qkv.weight.numel() + self.out_proj.weight.numel())


# ============================================================
# Saliency variants (A, B, C, D)
# All preserve V (per-head) and W_O.
# ============================================================
class SaliencyAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.variant = cfg.variant
        d = cfg.d_model
        dh = self.head_dim

        self.W_V = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        if self.variant == "A":
            # d → d_inter → 1, applied per-position to produce single saliency
            # vector. Shared across heads; masked causally per query.
            h_inter = cfg.saliency_a_hidden
            self.sal_A = nn.Sequential(
                nn.Linear(d, h_inter, bias=True),
                nn.GELU(),
                nn.Linear(h_inter, 1, bias=True),
            )
        elif self.variant == "B":
            # 2d → d → 1, applied per (q,k) pair. Implemented as two parallel
            # linears (Wq · q + Wk · k + b) followed by GeLU and a final linear
            # — mathematically equivalent to Linear(2d, d) with weight = [Wq;Wk],
            # but lets us avoid materializing the (B, T, T, 2d) input. Param
            # count matches: Wq + Wk + b + final = d² + d² + d + d + 1 ≈ 2d² + 2d.
            self.sal_B_q = nn.Linear(d, d, bias=True)
            self.sal_B_k = nn.Linear(d, d, bias=False)
            self.sal_B_2 = nn.Linear(d, 1, bias=True)
        elif self.variant == "C":
            # 2d → d → 1, takes (position, context_summary). Causal cumulative
            # mean is used as the summary so the result remains autoregressive.
            self.sal_C = nn.Sequential(
                nn.Linear(2 * d, d, bias=True),
                nn.GELU(),
                nn.Linear(d, 1, bias=True),
            )
        elif self.variant == "D":
            # No saliency: pure (causal) mean pool. Nothing to add.
            pass
        else:
            raise ValueError(f"unknown saliency variant: {self.variant}")

        # Scratch for diagnostics — populated on each forward.
        self._last_attn_entropy = None
        self._last_attn = None  # per-batch attention pattern, for diagnostics

    def _sal_B_pairscores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the (B, T_q, T_k) pair-saliency tensor for variant B.

        Implements `Linear(2d, d) -> GeLU -> Linear(d, 1)` via the algebraic
        decomposition Linear([xq; xk]) = W_q xq + W_k xk + b. The (B, T, T, d)
        intermediate is built via broadcast addition rather than concat, which
        avoids materialising the (B, T, T, 2d) tensor.
        """
        hq = self.sal_B_q(x)                                  # (B, T, d)
        hk = self.sal_B_k(x)                                  # (B, T, d)
        combined = hq.unsqueeze(2) + hk.unsqueeze(1)          # (B, T_q, T_k, d)
        combined = F.gelu(combined)
        return self.sal_B_2(combined).squeeze(-1)             # (B, T_q, T_k)

    def _causal_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Causal cumulative mean: out[b, t, :] = mean(x[b, 0..t, :])."""
        # Shape (B, T, d).
        cumsum = x.cumsum(dim=1)
        denom = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        return cumsum / denom.view(1, -1, 1)

    def _entropy(self, attn: torch.Tensor, dim: int) -> torch.Tensor:
        """Entropy of an attention distribution along `dim`. attn must be a
        probability distribution along that axis (post-softmax)."""
        eps = 1e-12
        return -(attn * (attn.clamp_min(eps)).log()).sum(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim

        # V projection (per-head)
        v = self.W_V(x).view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)

        # Build the (per-query, per-context) attention pattern according to variant.
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )

        if self.variant == "A":
            # Per-position saliency, shared across queries (modulo mask).
            sal = self.sal_A(x).squeeze(-1)                  # (B, T)
            # Broadcast to (B, T_q, T_k): every query gets the same raw scores
            # per context position; mask differs per query.
            attn_scores = sal.unsqueeze(1).expand(B, T, T)   # (B, T_q, T_k)
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(attn_scores, dim=-1)            # (B, T_q, T_k)

        elif self.variant == "B":
            # Pair saliency (B, T_q, T_k). Use gradient checkpointing so the
            # 3.2 GB intermediate (B, T_q, T_k, d) tensor doesn't stack across
            # 6 layers in the backward graph (would OOM on a 16 GB card).
            attn_scores = torch.utils.checkpoint.checkpoint(
                self._sal_B_pairscores, x, use_reentrant=False
            )                                                # (B, T_q, T_k)
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(attn_scores, dim=-1)

        elif self.variant == "C":
            # Causal cumulative mean as context summary at each position.
            ctx_summary = self._causal_mean(x)               # (B, T, D)
            # Per-position saliency on (position, summary_at_that_position).
            # Note: summary[t] is the prefix-mean up to t. We use summary[t]
            # as the "context" at position t. The same saliency value is then
            # shared across query positions q ≥ t (mask handles the rest).
            cat = torch.cat([x, ctx_summary], dim=-1)        # (B, T, 2D)
            sal = self.sal_C(cat).squeeze(-1)                # (B, T)
            attn_scores = sal.unsqueeze(1).expand(B, T, T)   # (B, T_q, T_k)
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(attn_scores, dim=-1)

        elif self.variant == "D":
            # Pure causal mean: attn[q, k] = 1/(q+1) if k <= q else 0.
            row_lens = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)  # (T,)
            attn = torch.ones(T, T, device=x.device, dtype=x.dtype)
            attn = attn.masked_fill(causal_mask, 0.0)
            attn = attn / row_lens.view(-1, 1)                # normalize each row
            attn = attn.unsqueeze(0).expand(B, T, T)          # (B, T_q, T_k)

        else:
            raise ValueError(f"unknown variant: {self.variant}")

        attn = self.attn_dropout(attn)

        # Diagnostics: entropy of each query's distribution, averaged over (B, T_q).
        # Done on the post-softmax tensor before V multiplication.
        with torch.no_grad():
            ent = self._entropy(attn.detach(), dim=-1)        # (B, T_q)
            self._last_attn_entropy = float(ent.mean().item())

        # Apply pattern to per-head V. attn: (B, T_q, T_k). v: (B, H, T, dh).
        # out[b, h, q, :] = Σ_k attn[b, q, k] * v[b, h, k, :]
        out = torch.einsum("bqk,bhkd->bhqd", attn, v)         # (B, H, T_q, dh)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_dropout(self.out_proj(out))

    def _sal_modules(self):
        """Return list of submodules holding the saliency-MLP params for the
        current variant (so we can introspect Frobenius / param counts)."""
        if self.variant == "A":
            return [self.sal_A]
        if self.variant == "B":
            return [self.sal_B_q, self.sal_B_k, self.sal_B_2]
        if self.variant == "C":
            return [self.sal_C]
        return []

    def diagnostics(self) -> dict:
        d = {
            "variant": self.variant,
            "last_attn_entropy_mean": self._last_attn_entropy,
        }
        sal = self._sal_modules()
        if sal:
            sq = 0.0
            for m in sal:
                for p in m.parameters():
                    sq += float(p.detach().norm().item()) ** 2
            d["sal_mlp_frob"] = sq ** 0.5
        return d

    def output_path_params(self) -> int:
        n = self.W_V.weight.numel() + self.out_proj.weight.numel()
        for m in self._sal_modules():
            n += sum(p.numel() for p in m.parameters())
        return n


# ============================================================
# Standard scaled-dot-product attention (added for follow-up exp 2).
# Same head/dim configuration and same V/W_O wiring as Bonsignore baseline.
# Differs only in the scoring mechanism: Q@K^T/sqrt(dh) vs -||Q-K||²/τ_h.
# Intentionally drops Bonsignore's per-head extras (log_tau, head_alphas,
# head_output_scalars) so this is a "minimal" SDPA control. The comparison
# vs Bonsignore tests kernel choice + per-head extras together; the
# comparison vs saliency variants tests scoring-mechanism class with
# matched extras (none).
# ============================================================
class StandardAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d = cfg.d_model
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_dropout(self.out_proj(out))

    def diagnostics(self) -> dict:
        return {"variant": "sdpa"}

    def output_path_params(self) -> int:
        return self.qkv.weight.numel() + self.out_proj.weight.numel()


# ============================================================
# Cumulative-mean attention sublayer — radical simplification.
# No Q, K, V, or W_O. The "attention" sublayer's output at position t
# is the causal cumulative mean of its LN'd input over positions 0..t.
# Per-position differentiation comes from causal prefix structure alone.
# ============================================================
class CumulativeMeanAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # No parameters. The whole point is to test what happens
        # without the attention projections.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        cumsum = x.cumsum(dim=1)                                  # (B, T, D)
        pos = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        return cumsum / pos                                       # (B, T, D)

    def diagnostics(self) -> dict:
        return {"variant": "cumulative_mean", "n_params": 0}

    def output_path_params(self) -> int:
        return 0


# ============================================================
# Dual projection sublayer — two parallel d→d projections, concat, compress.
# Tests whether richer per-position feature views (with learned mixture) can
# replace attention. No Q@K^T, no scoring, no per-pair computation.
# Param count = 4d² per block, matching standard attention's Q+K+V+W_O.
# ============================================================
class DualProjectionAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.W_P1 = nn.Linear(d, d, bias=False)
        self.W_P2 = nn.Linear(d, d, bias=False)
        self.W_C = nn.Linear(2 * d, d, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = self.W_P1(x)
        p2 = self.W_P2(x)
        out = self.W_C(torch.cat([p1, p2], dim=-1))
        return self.resid_dropout(out)

    def diagnostics(self) -> dict:
        return {
            "variant": "dual_projection",
            "W_P1_frob": float(self.W_P1.weight.detach().norm().item()),
            "W_P2_frob": float(self.W_P2.weight.detach().norm().item()),
            "W_C_frob": float(self.W_C.weight.detach().norm().item()),
        }

    def output_path_params(self) -> int:
        return (self.W_P1.weight.numel()
                + self.W_P2.weight.numel()
                + self.W_C.weight.numel())


# ============================================================
# Dual projection + cumulative mean — same as DualProjectionAttention but
# P2 operates on the causal cumulative mean of the LN'd residual stream,
# adding linear-cost cross-position information flow.
# ============================================================
class DualProjectionCumulativeAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.W_P1 = nn.Linear(d, d, bias=False)
        self.W_P2 = nn.Linear(d, d, bias=False)
        self.W_C = nn.Linear(2 * d, d, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        cumsum = x.cumsum(dim=1)
        pos = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        x_cum = cumsum / pos
        p1 = self.W_P1(x)
        p2 = self.W_P2(x_cum)
        out = self.W_C(torch.cat([p1, p2], dim=-1))
        return self.resid_dropout(out)

    def diagnostics(self) -> dict:
        return {
            "variant": "dual_projection_with_cumulative",
            "W_P1_frob": float(self.W_P1.weight.detach().norm().item()),
            "W_P2_frob": float(self.W_P2.weight.detach().norm().item()),
            "W_C_frob": float(self.W_C.weight.detach().norm().item()),
        }

    def output_path_params(self) -> int:
        return (self.W_P1.weight.numel()
                + self.W_P2.weight.numel()
                + self.W_C.weight.numel())


def build_attention(cfg: ModelConfig) -> nn.Module:
    if cfg.variant == "baseline":
        return BonsignoreAttention(cfg)
    if cfg.variant == "sdpa":
        return StandardAttention(cfg)
    if cfg.variant == "cumulative_mean":
        return CumulativeMeanAttention(cfg)
    if cfg.variant == "dual_projection":
        return DualProjectionAttention(cfg)
    if cfg.variant == "dual_projection_with_cumulative":
        return DualProjectionCumulativeAttention(cfg)
    return SaliencyAttention(cfg)
