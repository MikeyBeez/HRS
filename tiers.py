"""Tiered compute operators for HRS.

Four tiers with different compute costs:
- Conv: O(n) depthwise 1D convolution for local patterns
- Expert: O(n) MoE with small experts for token-level specialization
- Attention: O(n^2) causal self-attention for high-importance tokens
- Sink: O(1) identity with learned scale-down for low-importance tokens
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, TierConfig


class ConvTier(nn.Module):
    """Causal depthwise 1D convolution tier. O(n) cost.

    Captures local n-gram patterns using only past context.
    Uses left-only padding to maintain causality (no future leakage).
    """

    def __init__(self, model_cfg: ModelConfig, tier_cfg: TierConfig):
        super().__init__()
        d = model_cfg.d_model
        k = tier_cfg.conv_kernel_size
        self.causal_pad = k - 1  # left-only padding for causality
        # Depthwise conv: each channel gets its own filter (no built-in padding)
        self.conv = nn.Conv1d(
            d, d, kernel_size=k, padding=0, groups=d, bias=False
        )
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(model_cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal depthwise convolution.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        # Conv1d expects (B, C, T); causal = pad left only
        h = x.transpose(1, 2)  # (B, D, T)
        h = F.pad(h, (self.causal_pad, 0))  # left-pad
        h = self.conv(h).transpose(1, 2)  # (B, T, D)
        h = self.dropout(self.norm(h))
        return h


class ExpertTier(nn.Module):
    """Mixture-of-Experts tier. O(n) cost per token.

    Small MoE with top-1 routing: each token goes to one expert.
    """

    def __init__(self, model_cfg: ModelConfig, tier_cfg: TierConfig):
        super().__init__()
        d = model_cfg.d_model
        h = tier_cfg.expert_hidden
        self.n_experts = tier_cfg.n_experts
        self.top_k = tier_cfg.expert_top_k

        # Expert gate
        self.gate = nn.Linear(d, self.n_experts, bias=False)

        # Each expert is a 2-layer MLP: d -> h -> d
        self.expert_w1 = nn.Parameter(torch.empty(self.n_experts, d, h))
        self.expert_w2 = nn.Parameter(torch.empty(self.n_experts, h, d))

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(model_cfg.dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.expert_w1, std=0.02)
        nn.init.normal_(self.expert_w2, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MoE with top-1 routing.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # Gate logits and top-k selection
        gate_logits = self.gate(x_flat)  # (B*T, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Top-1 routing
        top_vals, top_idx = gate_weights.topk(self.top_k, dim=-1)  # (B*T, 1)
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize

        # Dispatch to experts
        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = (top_idx.squeeze(-1) == e)  # (B*T,)
            if mask.any():
                x_e = x_flat[mask]  # (n_tokens, D)
                h = F.gelu(x_e @ self.expert_w1[e])  # (n_tokens, h)
                h = h @ self.expert_w2[e]  # (n_tokens, D)
                out[mask] = h * top_vals[mask]

        out = self.dropout(self.norm(out.reshape(B, T, D)))
        return out


class AttentionTier(nn.Module):
    """Standard causal self-attention tier. O(n^2) cost.

    Used for high-importance tokens that need global context.
    Includes RoPE positional embeddings.
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.n_heads = model_cfg.n_heads
        self.head_dim = model_cfg.d_model // model_cfg.n_heads

        self.qkv = nn.Linear(model_cfg.d_model, 3 * model_cfg.d_model, bias=model_cfg.bias)
        self.out_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=model_cfg.bias)
        self.attn_dropout = nn.Dropout(model_cfg.dropout)
        self.resid_dropout = nn.Dropout(model_cfg.dropout)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, model_cfg.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class SinkTier(nn.Module):
    """Sink tier: identity with learned scale-down.

    Tokens pass through but contribute minimally. The learned scalar
    starts at ~0.1 so sink-routed tokens have reduced impact on
    subsequent attention computations.
    """

    def __init__(self, tier_cfg: TierConfig):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(tier_cfg.sink_init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale down tokens (sink channel).

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) scaled down
        """
        return x * self.scale.abs()  # abs to ensure non-negative


# --- RoPE (shared with AttentionTier) ---

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
