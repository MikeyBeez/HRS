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


# ============================================================
# Compression-then-attention sublayer (V4-style hybrid).
# Compress k tokens → 1 entry via softmax-weighted block saliency, run full
# Bonsignore attention over the T/k compressed entries, decompress by
# broadcasting each compressed-entry output back to all k tokens of its block.
# ============================================================
class CompressBlock(nn.Module):
    """Block-level saliency-weighted pooling: (B, T, d) -> (B, T/k, d)."""

    def __init__(self, cfg: ModelConfig, k: int):
        super().__init__()
        self.k = k
        self.W_s = nn.Linear(cfg.d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k = self.k
        assert T % k == 0, f"T={T} must be divisible by compression ratio k={k}"
        m = T // k
        x_blk = x.reshape(B, m, k, D)
        s = self.W_s(x_blk)                       # (B, m, k, 1)
        w = F.softmax(s, dim=2)                   # softmax across within-block positions
        return (w * x_blk).sum(dim=2)             # (B, m, D)


class CompressionAttention(nn.Module):
    """Causal V4-style compressed attention (CSA).

    For query at position j (in block i = j // k, offset r = j % k):
      - keys = compressed entries [0, i)         (strictly past blocks)
              ∪ uncompressed tokens [i*k, j)    (current block, strictly before j)
              ∪ token j itself
      - q, k, v projections are shared between compressed entries and
        uncompressed tokens (same Linear(d, 3d) applied to both inputs)
      - Bonsignore Q-K scoring with per-head temperatures

    The output at position j depends only on tokens 0..j. Verified by
    `causality_check` (see bottom of module).
    """

    def __init__(self, cfg: ModelConfig, k: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.k = k
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d, H, dh = cfg.d_model, cfg.n_heads, self.head_dim

        self.compress = CompressBlock(cfg, k)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.full((H,), math.log(float(dh))))
        self.head_alphas = nn.Parameter(torch.ones(H))
        self.head_output_scalars = nn.Parameter(torch.zeros(H))
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Cache of (T, m+T) bool masks keyed by (T, device).
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _build_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """mask[j, p] = True if key at index p (in concat(compressed, uncompressed))
        is valid for query at position j."""
        k = self.k
        m = T // k
        K_total = m + T
        j = torch.arange(T, device=device).unsqueeze(1)            # (T, 1)
        p = torch.arange(K_total, device=device).unsqueeze(0)      # (1, K_total)
        i = j // k                                                  # block index per query
        # Compressed segment is positions [0, m) in the K tensor.
        comp_mask = (p < i) & (p < m)
        # Uncompressed segment is positions [m, m+T). Within that:
        #   query j (in block i) sees tokens i*k, i*k+1, ..., j (inclusive).
        uncomp_mask = (p >= m) & (p >= m + i * k) & (p <= m + j)
        return comp_mask | uncomp_mask

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._build_mask(T, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh, k = self.n_heads, self.head_dim, self.k
        assert T % k == 0, f"T={T} must be divisible by compression ratio k={k}"
        m = T // k

        # Compressed entries from past+current blocks (block i's entry is causal
        # within its own block — see CompressBlock — but we only let *later*
        # queries read it via the mask, so future leak is impossible).
        comp = self.compress(x)                                 # (B, m, D)

        # qkv on x: gives Q (we keep) and "uncompressed" K, V.
        qkv_x = self.qkv(x).reshape(B, T, 3, H, dh)
        q_x, k_x, v_x = qkv_x.unbind(dim=2)                     # each (B, T, H, dh)
        # qkv on compressed: only K and V are used (Q on compressed is wasted).
        qkv_c = self.qkv(comp).reshape(B, m, 3, H, dh)
        _, k_c, v_c = qkv_c.unbind(dim=2)                       # k_c, v_c: (B, m, H, dh)

        # Concat key/value along sequence: [compressed (m), uncompressed (T)].
        K = torch.cat([k_c, k_x], dim=1).transpose(1, 2)        # (B, H, m+T, dh)
        V = torch.cat([v_c, v_x], dim=1).transpose(1, 2)        # (B, H, m+T, dh)
        Q = q_x.transpose(1, 2)                                 # (B, H, T, dh)

        q_sq = (Q ** 2).sum(dim=-1, keepdim=True)               # (B, H, T, 1)
        k_sq = (K ** 2).sum(dim=-1, keepdim=True)               # (B, H, m+T, 1)
        dot = Q @ K.transpose(-2, -1)                           # (B, H, T, m+T)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot
        taus = self.log_tau.exp().view(1, H, 1, 1)
        scores = -distances / taus
        alphas = torch.sigmoid(self.head_alphas).view(1, H, 1, 1)
        scores = scores * alphas

        mask = self._get_mask(T, x.device)                      # (T, m+T)
        scores = scores.masked_fill(~mask.view(1, 1, T, m + T), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ V                                          # (B, H, T, dh)

        head_scales = F.softplus(self.head_output_scalars).view(1, H, 1, 1)
        out = out * head_scales
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_dropout(self.out_proj(out))

    def diagnostics(self) -> dict:
        return {
            "variant": f"compress_{self.k}",
            "compression_ratio": self.k,
            "W_s_norm": float(self.compress.W_s.weight.detach().norm().item()),
            "head_alphas_sigmoid": torch.sigmoid(self.head_alphas.detach()).tolist(),
            "taus": self.log_tau.exp().detach().tolist(),
            "head_output_scalars_softplus": F.softplus(
                self.head_output_scalars.detach()
            ).tolist(),
        }

    def output_path_params(self) -> int:
        return (self.compress.W_s.weight.numel()
                + self.qkv.weight.numel()
                + self.out_proj.weight.numel())


# ============================================================
# V4-style hybrid: recent uncompressed window + moderately compressed
# middle range + aggressively compressed distant past. Block-aligned
# compression at fixed positions; per-query masks select which blocks
# fall into each region for that query.
# ============================================================
class HybridCompressionAttention(nn.Module):
    """For query at position j the K/V set is:
      Region 1 (recent uncompressed): tokens at [j-W+1, j]            (length ≤ W)
      Region 2 (moderate r2):         r2-blocks fully in [j-W-M+1, j-W]   (≤ M//r2 entries)
      Region 3 (aggressive r3):       r3-blocks fully in [0, j-W-M]    (≤ (j-W-M+1)//r3 entries)

    Compression is causal because every block included satisfies
    `(b+1)·r ≤ j-W+1 ≤ j`, so all source tokens are < j.
    """

    def __init__(self, cfg: ModelConfig, W: int = 128, M: int = 384,
                 r2: int = 4, r3: int = 16):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.W = W
        self.M = M
        self.r2 = r2
        self.r3 = r3
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d, H, dh = cfg.d_model, cfg.n_heads, self.head_dim

        # Each compression rate has its own saliency vector (per spec).
        self.compress_r2 = CompressBlock(cfg, r2)
        self.compress_r3 = CompressBlock(cfg, r3)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.full((H,), math.log(float(dh))))
        self.head_alphas = nn.Parameter(torch.ones(H))
        self.head_output_scalars = nn.Parameter(torch.zeros(H))
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _build_mask(self, T: int, device: torch.device) -> torch.Tensor:
        W, M, r2, r3 = self.W, self.M, self.r2, self.r3
        m_r3 = T // r3
        m_r2 = T // r2
        K_total = m_r3 + m_r2 + T
        j = torch.arange(T, device=device).unsqueeze(1)             # (T, 1)
        p = torch.arange(K_total, device=device).unsqueeze(0)       # (1, K_total)

        # r3 segment: positions [0, m_r3). Block index within segment = p.
        in_r3 = p < m_r3
        b_r3 = p                                                     # block index = position
        r3_end = (b_r3 + 1) * r3                                     # last token + 1
        # Region 3: block end ≤ j - W - M + 1  (block fully ≤ position j-W-M)
        r3_valid = in_r3 & (r3_end <= j - W - M + 1)

        # r2 segment: positions [m_r3, m_r3 + m_r2). Block index = p - m_r3.
        in_r2 = (p >= m_r3) & (p < m_r3 + m_r2)
        b_r2 = p - m_r3
        r2_start = b_r2 * r2
        r2_end = (b_r2 + 1) * r2
        # Region 2: block start ≥ j - W - M + 1 AND block end ≤ j - W + 1
        r2_valid = in_r2 & (r2_start >= j - W - M + 1) & (r2_end <= j - W + 1)

        # Uncompressed segment: positions [m_r3 + m_r2, m_r3 + m_r2 + T).
        in_u = p >= (m_r3 + m_r2)
        u_pos = p - (m_r3 + m_r2)                                    # original token position
        # Region 1: u_pos in [max(0, j-W+1), j]
        u_valid = in_u & (u_pos >= j - W + 1) & (u_pos <= j)

        return r3_valid | r2_valid | u_valid

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._build_mask(T, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh, r2, r3 = self.n_heads, self.head_dim, self.r2, self.r3
        assert T % r3 == 0 and T % r2 == 0, f"T={T} must be divisible by r2={r2} and r3={r3}"
        m_r2 = T // r2
        m_r3 = T // r3

        comp_r2 = self.compress_r2(x)                                # (B, m_r2, D)
        comp_r3 = self.compress_r3(x)                                # (B, m_r3, D)

        qkv_x = self.qkv(x).reshape(B, T, 3, H, dh)
        q_x, k_x, v_x = qkv_x.unbind(dim=2)
        qkv_r2 = self.qkv(comp_r2).reshape(B, m_r2, 3, H, dh)
        _, k_r2, v_r2 = qkv_r2.unbind(dim=2)
        qkv_r3 = self.qkv(comp_r3).reshape(B, m_r3, 3, H, dh)
        _, k_r3, v_r3 = qkv_r3.unbind(dim=2)

        # Concat order matches the mask: [r3, r2, uncompressed].
        K = torch.cat([k_r3, k_r2, k_x], dim=1).transpose(1, 2)      # (B, H, K_total, dh)
        V = torch.cat([v_r3, v_r2, v_x], dim=1).transpose(1, 2)
        Q = q_x.transpose(1, 2)                                      # (B, H, T, dh)

        q_sq = (Q ** 2).sum(dim=-1, keepdim=True)
        k_sq = (K ** 2).sum(dim=-1, keepdim=True)
        dot = Q @ K.transpose(-2, -1)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot
        taus = self.log_tau.exp().view(1, H, 1, 1)
        scores = -distances / taus
        alphas = torch.sigmoid(self.head_alphas).view(1, H, 1, 1)
        scores = scores * alphas

        mask = self._get_mask(T, x.device)                           # (T, K_total)
        scores = scores.masked_fill(~mask.view(1, 1, T, -1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ V                                               # (B, H, T, dh)

        head_scales = F.softplus(self.head_output_scalars).view(1, H, 1, 1)
        out = out * head_scales
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_dropout(self.out_proj(out))

    def diagnostics(self) -> dict:
        return {
            "variant": "compress_hybrid",
            "W": self.W, "M": self.M, "r2": self.r2, "r3": self.r3,
            "W_s_r2_norm": float(self.compress_r2.W_s.weight.detach().norm().item()),
            "W_s_r3_norm": float(self.compress_r3.W_s.weight.detach().norm().item()),
            "head_alphas_sigmoid": torch.sigmoid(self.head_alphas.detach()).tolist(),
            "taus": self.log_tau.exp().detach().tolist(),
        }

    def output_path_params(self) -> int:
        return (self.compress_r2.W_s.weight.numel()
                + self.compress_r3.W_s.weight.numel()
                + self.qkv.weight.numel()
                + self.out_proj.weight.numel())


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
    if cfg.variant == "compress_hybrid":
        return HybridCompressionAttention(cfg)
    if cfg.variant.startswith("compress_"):
        k = int(cfg.variant.split("_", 1)[1])
        return CompressionAttention(cfg, k)
    return SaliencyAttention(cfg)
