"""Gated-compression transformer with M = √N bottleneck.

Design parameter N = 1024 (max context). Bottleneck M = √N = 32.
  if input length k ≤ M:  pad to M positions, mask the padding, attend.
  if input length k > M:  compress to M via mean pooling (no learned params).

Positional encoding: applied AFTER the gate. The compressed M positions get
pos_emb[0..M-1] regardless of what they represent. The model learns to
interpret the encoding based on context.

Compression mechanism: torch.nn.functional.adaptive_avg_pool1d(x, M).
Divides k positions into M groups of ⌈k/M⌉ each, mean-pools within each.

Output: (B, M, V) logits when gate is open, (B, k_padded, V) when gate is
closed (with k_padded = M for spec consistency, but valid tokens only at
positions [0..k-1]).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatedConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    M: int = 32                      # bottleneck size; M² = N for N=1024
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Standard causal SDPA with optional padding mask."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x, key_padding_mask=None):
        """x: (B, T, D). key_padding_mask: (B, T) bool — True means PAD position
        that should be ignored."""
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Build float mask combining causal + padding
            # Allowed: i can attend to j if j <= i AND j is not padded
            t = torch.arange(T, device=x.device)
            causal = t.unsqueeze(0) >= t.unsqueeze(1)              # (T, T) — j ≤ i
            allowed_kj = ~key_padding_mask                          # (B, T) — non-padded keys
            mask = causal.unsqueeze(0) & allowed_kj.unsqueeze(1)    # (B, T, T)
            float_mask = torch.zeros(B, T, T, device=x.device)
            float_mask.masked_fill_(~mask, float("-inf"))
            float_mask = float_mask.unsqueeze(1)                    # (B, 1, T, T)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=float_mask, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
        return self.out(out.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: GatedConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class GatedCompression(nn.Module):
    """Mean-pool to M positions when input length > M; identity otherwise."""
    def __init__(self, M=32):
        super().__init__()
        self.M = M

    def forward(self, x):
        # x: (B, T, D)
        T = x.shape[1]
        if T <= self.M:
            return x
        # compress
        x_t = x.transpose(1, 2)                           # (B, D, T)
        x_p = F.adaptive_avg_pool1d(x_t, self.M)          # (B, D, M)
        return x_p.transpose(1, 2)                        # (B, M, D)


class GatedTransformer(nn.Module):
    """Gated-compression transformer.

    Forward:
      tokens (B, k) -> tok_emb -> gate (compress if k > M, else pass) ->
        pad-to-M if needed -> + pos_emb -> N blocks -> ln_f -> head -> logits.

    When gate is closed (k ≤ M): output shape (B, M, V) but valid only at
    positions [0..k-1]. Padding mask applied during attention.

    When gate is open (k > M): output shape (B, M, V). All M positions valid.
    """
    def __init__(self, cfg: GatedConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.gate = GatedCompression(cfg.M)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.M, cfg.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """x: (B, k). Returns logits (B, M, V) and a flag for gate state.
        For k ≤ M, valid output positions are 0..k-1 (rest are padded).
        For k > M, all M positions are valid; each represents a compressed
        group of ⌈k/M⌉ original tokens."""
        B, k = x.shape
        h = self.tok_emb(x)                               # (B, k, D)
        h = self.gate(h)                                   # (B, M, D) or (B, k, D)
        T_after = h.shape[1]                               # = M if compressed, k otherwise

        key_padding_mask = None
        if T_after < self.cfg.M:
            # gate closed and k < M → pad to M
            pad = torch.zeros(B, self.cfg.M - T_after, self.cfg.d_model,
                                device=h.device, dtype=h.dtype)
            h = torch.cat([h, pad], dim=1)                # (B, M, D)
            # Build padding mask: True at padded positions
            kpm = torch.zeros(B, self.cfg.M, dtype=torch.bool, device=h.device)
            kpm[:, T_after:] = True
            key_padding_mask = kpm

        # Add pos_emb (always to M positions)
        h = h + self.pos_emb[:, :h.shape[1], :]

        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        h = self.ln_f(h)
        return self.head(h)                                # (B, M, V)


class BaselineTransformer(nn.Module):
    """Standard 6-layer transformer (no gate). Used as length-specific baselines."""
    def __init__(self, cfg: GatedConfig, max_seq=1024):
        super().__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, cfg.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)
