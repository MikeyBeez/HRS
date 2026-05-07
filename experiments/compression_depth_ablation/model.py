"""Configurable compression stack: build from arbitrary stride pattern.

  stride_pattern = [2, 2, 2, 2]              → 4-layer (current)
  stride_pattern = [2, 1, 2, 1, 2, 2]        → 6-layer (interleaved)
  stride_pattern = [2, 1, 1, 2, 1, 1, 2, 2]  → 8-layer (interleaved)

Conventions:
  - Stride-2 layers: causal Conv1d(D, D, kernel=2, stride=2, padding=0).
    Output[i] = f(input[2i], input[2i+1]). Naturally causal.
  - Stride-1 layers: causal Conv1d(D, D, kernel=3, stride=1, left-padded 2).
    Output[i] = f(input[i-2], input[i-1], input[i]). Length preserved.

Total compression ratio = product of strides. Asserted to be 16.

State dict for the 4-layer pattern matches `compression_conv/model.py` if
written with the same module names; we use distinct names here ('layers')
to avoid silent compatibility surprises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepthCompressionConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    ctx_len: int = 1024
    stride_pattern: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dropout: float = 0.1


class CausalCompLayer(nn.Module):
    def __init__(self, d_model, stride):
        super().__init__()
        self.stride = stride
        if stride == 2:
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=2,
                                    stride=2, padding=0)
            self._left_pad = 0
        elif stride == 1:
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                                    stride=1, padding=0)
            self._left_pad = 2  # causal padding
        else:
            raise ValueError(f"unsupported stride {stride}")
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T)
        x = x.transpose(1, 2)
        if self._left_pad > 0:
            x = F.pad(x, (self._left_pad, 0))
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = F.gelu(x)
        return x


class CompressionStack(nn.Module):
    def __init__(self, d_model, stride_pattern):
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalCompLayer(d_model, s) for s in stride_pattern]
        )
        ratio = 1
        for s in stride_pattern:
            ratio *= s
        self.ratio = ratio
        assert ratio == 16, f"compression ratio {ratio} != 16"

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DepthCompressedTransformer(nn.Module):
    def __init__(self, cfg: DepthCompressionConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ctx_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.compress = CompressionStack(cfg.d_model, cfg.stride_pattern)
        self.compression_ratio = self.compress.ratio

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
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :T, :]
        h = self.compress(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)
