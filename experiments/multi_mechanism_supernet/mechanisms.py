"""12 head mechanisms for the multi-mechanism supernet.

Each mechanism takes input (B, T, d_model) and produces (B, T, head_dim).
The supernet concatenates all 12 outputs and applies per-mechanism gating.

Mechanisms (per spec):
  1. Full attention with learned positional embeddings (handled at model-level)
  2. Full attention with learned positional embeddings (duplicate of 1)
  3. Full attention with RoPE
  4. Compression-stack-attention (architecture A) at 16x
  5. Compression-stack-attention (architecture A) at 8x
  6. Compression-stack-attention (architecture A) at 4x
  7. Per-layer K/V compression (architecture B) at 16x
  8. Selection-based top-k attention (k=64)
  9. Mamba block (pure PyTorch)
  10. Sliding-window attention (window=256)
  11. Causal 1D conv mixer (kernel=31)
  12. Gated linear attention
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Common building blocks ----------------

def causal_strided_compress(d, n_layers):
    """Build a stack of n_layers causal Conv1d(k=2, s=2, p=0) + LN + GELU. Total compression 2^n_layers."""
    layers = []
    for _ in range(n_layers):
        layers.append(_CompLayer(d))
    return nn.Sequential(*layers)


class _CompLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=2, stride=2, padding=0)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = F.gelu(x)
        return x


def apply_rope(x, sin, cos):
    """x: (B, H, T, hd) where hd is even. sin, cos: (T, hd)."""
    x_rot = x * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)
    return x_rot


def _rotate_half(x):
    # split last dim into halves and rotate
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def build_rope(seq_len, head_dim, base=10000.0, device=None):
    """Return (sin, cos) of shape (seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                # (T, hd/2)
    emb = torch.cat([freqs, freqs], dim=-1)         # (T, hd)
    return emb.sin(), emb.cos()


# ---------------- Mechanisms ----------------

class FullAttn(nn.Module):
    """Full causal attention. Single-head per mechanism: project to Q/K/V at head_dim
    from d_model_in. Optionally apply RoPE."""

    def __init__(self, d_model, head_dim, use_rope=False, max_seq=1024):
        super().__init__()
        self.head_dim = head_dim
        self.qkv = nn.Linear(d_model, 3 * head_dim, bias=True)
        self.use_rope = use_rope
        if use_rope:
            assert head_dim % 2 == 0
            sin, cos = build_rope(max_seq, head_dim)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)                             # (B, T, 3*hd)
        q, k, v = qkv.chunk(3, dim=-1)                # each (B, T, hd)
        # SDPA expects (B, H, T, hd); add a head dim of 1
        q = q.unsqueeze(1); k = k.unsqueeze(1); v = v.unsqueeze(1)
        if self.use_rope:
            sin = self.rope_sin[:T]; cos = self.rope_cos[:T]
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.squeeze(1)                          # (B, T, hd)


class CompressionA(nn.Module):
    """Architecture A: compress full residual to T/cr, attention on compressed,
    upsample back via nearest-neighbor (broadcast over each cr-window).

    Compression ratio = 2 ** n_compress_layers.
    """
    def __init__(self, d_model, head_dim, n_compress_layers):
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = 2 ** n_compress_layers
        self.in_proj = nn.Linear(d_model, head_dim, bias=False)
        self.compress = causal_strided_compress(head_dim, n_compress_layers)
        self.qkv = nn.Linear(head_dim, 3 * head_dim, bias=True)

    def forward(self, x):
        B, T, D = x.shape
        cr = self.compression_ratio
        # Project to head_dim, compress
        h = self.in_proj(x)                           # (B, T, hd)
        h = self.compress(h)                          # (B, T/cr, hd)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.unsqueeze(1); k = k.unsqueeze(1); v = v.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.squeeze(1)                          # (B, T/cr, hd)
        # Upsample back to T via repeat-interleave (each compressed pos broadcasts to cr)
        out = out.repeat_interleave(cr, dim=1)        # (B, T, hd)
        return out


class CompressionB(nn.Module):
    """Architecture B: per-layer K/V compression. Q stays at full T.
    K, V are pooled to T/16 via compression stack. Causal mask: query at
    position t attends to compressed positions c <= floor(t/16)."""

    def __init__(self, d_model, head_dim, n_compress_layers=4):
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = 2 ** n_compress_layers
        self.qkv = nn.Linear(d_model, 3 * head_dim, bias=True)
        self.compress_k = causal_strided_compress(head_dim, n_compress_layers)
        self.compress_v = causal_strided_compress(head_dim, n_compress_layers)
        self._mask_cache = {}

    def _get_mask(self, T, T_kv, device):
        key = (T, T_kv, device)
        if key not in self._mask_cache:
            t = torch.arange(T, device=device).unsqueeze(1)
            c = torch.arange(T_kv, device=device).unsqueeze(0)
            allowed = c <= (t // self.compression_ratio)
            mask = torch.zeros(T, T_kv, device=device)
            mask.masked_fill_(~allowed, float("-inf"))
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)                # (B, T, hd) each
        # Compress K, V
        k_c = self.compress_k(k)                       # (B, T/cr, hd)
        v_c = self.compress_v(v)
        # SDPA with custom mask
        q = q.unsqueeze(1); k_c = k_c.unsqueeze(1); v_c = v_c.unsqueeze(1)
        mask = self._get_mask(T, k_c.shape[2], q.device)
        out = F.scaled_dot_product_attention(q, k_c, v_c, attn_mask=mask, is_causal=False)
        return out.squeeze(1)                          # (B, T, hd)


class TopKAttn(nn.Module):
    """Top-k selection attention: query at position t attends only to top-k
    causal positions by score. Softmax over selected positions."""

    def __init__(self, d_model, head_dim, k=64):
        super().__init__()
        self.head_dim = head_dim
        self.k = k
        self.qkv = nn.Linear(d_model, 3 * head_dim, bias=True)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)                # (B, T, hd) each
        # Compute full scores (B, T, T)
        scores = torch.einsum("btd,bsd->bts", q, k) / math.sqrt(self.head_dim)
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        # Top-k along last dim (key positions). For early positions (t < k),
        # there are fewer than k causal positions; topk handles this fine since
        # masked positions are -inf.
        top_k = min(self.k, T)
        topk_vals, topk_idx = scores.topk(top_k, dim=-1)        # (B, T, k)
        # Softmax over the k selected
        attn = F.softmax(topk_vals, dim=-1)                       # (B, T, k)
        # Gather V at those positions: V is (B, T, hd); we need (B, T, k, hd)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        v_exp = v.unsqueeze(1).expand(B, T, T, self.head_dim)
        gathered = v_exp.gather(dim=2, index=topk_idx_exp)        # (B, T, k, hd)
        out = (attn.unsqueeze(-1) * gathered).sum(dim=2)          # (B, T, hd)
        return out


class _MambaCore(nn.Module):
    """Pure-PyTorch Mamba S6 core operating on a small d_inner dim.
    For supernet head: input (B, T, d_model_in), output (B, T, head_dim).
    Internally d_inner = 2 * head_dim."""

    def __init__(self, d_model_in, head_dim, d_state=16, d_conv=4):
        super().__init__()
        self.d_inner = 2 * head_dim
        self.head_dim = head_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(1, math.ceil(head_dim / 16))

        self.in_proj = nn.Linear(d_model_in, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Init dt_proj bias such that softplus(bias) ∈ [dt_min, dt_max]
        with torch.no_grad():
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.data.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, head_dim, bias=False)

    def forward(self, hidden_states):
        B, T, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)              # (B, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                     # each (B, T, d_inner)
        # Causal Conv1d
        x_t = x.transpose(1, 2)
        x_t = self.conv1d(x_t)[:, :, :T]
        x = x_t.transpose(1, 2)
        x = F.silu(x)

        # SSM params
        x_dbl = self.x_proj(x)
        dt, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))                          # (B, T, d_inner)
        A = -torch.exp(self.A_log.float())                          # (d_inner, d_state)

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))   # (B, T, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(2)                       # (B, T, d_inner, d_state)

        state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            state = dA[:, t] * state + dB[:, t] * x[:, t, :, None]
            y_t = (state * C_proj[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + self.D * x
        y = y * F.silu(z)
        return self.out_proj(y)                                       # (B, T, head_dim)


class MambaMech(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.core = _MambaCore(d_model, head_dim)

    def forward(self, x):
        return self.core(x)


class SlidingWindowAttn(nn.Module):
    """Causal sliding-window attention with window=window_size."""

    def __init__(self, d_model, head_dim, window_size=256):
        super().__init__()
        self.head_dim = head_dim
        self.window = window_size
        self.qkv = nn.Linear(d_model, 3 * head_dim, bias=True)
        self._mask_cache = {}

    def _get_mask(self, T, device):
        key = (T, device)
        if key not in self._mask_cache:
            t = torch.arange(T, device=device).unsqueeze(1)
            s = torch.arange(T, device=device).unsqueeze(0)
            # Allowed: causal AND within window
            allowed = (s <= t) & ((t - s) < self.window)
            mask = torch.zeros(T, T, device=device)
            mask.masked_fill_(~allowed, float("-inf"))
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.unsqueeze(1); k = k.unsqueeze(1); v = v.unsqueeze(1)
        mask = self._get_mask(T, x.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        return out.squeeze(1)


class CausalConvMixer(nn.Module):
    """Causal 1D conv with kernel=31. No attention."""

    def __init__(self, d_model, head_dim, kernel=31):
        super().__init__()
        self.head_dim = head_dim
        self.in_proj = nn.Linear(d_model, head_dim, bias=False)
        self.conv = nn.Conv1d(head_dim, head_dim, kernel_size=kernel,
                                padding=kernel - 1, bias=True)
        self.kernel = kernel

    def forward(self, x):
        B, T, D = x.shape
        h = self.in_proj(x)                            # (B, T, hd)
        h = h.transpose(1, 2)                          # (B, hd, T)
        h = self.conv(h)[:, :, :T]                      # trim to T (causal)
        h = h.transpose(1, 2)                          # (B, T, hd)
        return F.gelu(h)


class GatedLinearAttn(nn.Module):
    """Gated linear attention: state[t] = α[t] * state[t-1] + K[t]^T V[t];
    out[t] = Q[t] · state[t]. α is per-token learned gate ∈ (0, 1).

    Pure-PyTorch O(T) Python loop (slow but correct).
    """

    def __init__(self, d_model, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        # Per-token gate from input
        self.gate_proj = nn.Linear(d_model, head_dim, bias=True)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x)                            # (B, T, hd)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Per-token, per-channel gate
        g = torch.sigmoid(self.gate_proj(x))          # (B, T, hd) ∈ (0, 1)

        # Recurrent state of shape (B, hd, hd) = K^T V accumulator
        state = torch.zeros(B, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            # Apply per-channel gate to the existing state (decay)
            state = state * g[:, t, :, None]
            # Update: outer product k_t v_t
            state = state + torch.einsum("bd,be->bde", k[:, t], v[:, t])
            # Read: query the state
            y_t = torch.einsum("bd,bde->be", q[:, t], state)
            outs.append(y_t)
        return torch.stack(outs, dim=1)               # (B, T, hd)


def build_mechanisms(d_model, head_dim, max_seq) -> List[nn.Module]:
    """Build the 12 mechanisms in the order specified by the program."""
    return [
        FullAttn(d_model, head_dim, use_rope=False, max_seq=max_seq),     # 1
        FullAttn(d_model, head_dim, use_rope=False, max_seq=max_seq),     # 2 duplicate
        FullAttn(d_model, head_dim, use_rope=True, max_seq=max_seq),      # 3 RoPE
        CompressionA(d_model, head_dim, n_compress_layers=4),              # 4 16x A
        CompressionA(d_model, head_dim, n_compress_layers=3),              # 5 8x A
        CompressionA(d_model, head_dim, n_compress_layers=2),              # 6 4x A
        CompressionB(d_model, head_dim, n_compress_layers=4),              # 7 16x B
        TopKAttn(d_model, head_dim, k=64),                                  # 8 top-k
        MambaMech(d_model, head_dim),                                       # 9 Mamba
        SlidingWindowAttn(d_model, head_dim, window_size=256),              # 10 sliding window
        CausalConvMixer(d_model, head_dim, kernel=31),                      # 11 conv mixer
        GatedLinearAttn(d_model, head_dim),                                 # 12 GLA
    ]


MECHANISM_NAMES = [
    "full_learned_pos_a", "full_learned_pos_b", "full_rope",
    "compression_a_16x", "compression_a_8x", "compression_a_4x",
    "compression_b_16x", "topk_64", "mamba",
    "sliding_window_256", "causal_conv_31", "gated_linear_attn",
]
