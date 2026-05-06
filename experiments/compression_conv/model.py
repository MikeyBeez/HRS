"""Hierarchical learned compression + transformer for Tiny Shakespeare.

Architecture:
  tokens -> embed + pos -> [4-layer 2x compression stack, optional]
                        -> N transformer blocks (causal SDPA + FFN)
                        -> ln_f -> linear head -> logits

Compression layer (causal, 2x):
  conv1d(D, D, kernel_size=2, stride=2, padding=0) on (B, D, T) -> (B, D, T/2)
  + LayerNorm + GELU

With kernel=2 stride=2 no padding, output[i] = f(input[2i], input[2i+1]).
This is naturally causal because output[i]'s receptive field is the leftmost
2 input positions of its 2-token window. Stack 4 layers for 16x total
compression. Output[i] at the top of the stack sees input tokens
[16i .. 16i+15] only.

Causality is preserved in the standard sense: predicting the next token
after a compressed group uses only inputs that came before that token.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompressedConfig:
    vocab_size: int = 50257
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 1536
    ctx_len: int = 512                 # original-input length
    n_compress_layers: int = 4         # 0 = no compression; 4 = 16x
    dropout: float = 0.0


class CausalCompressLayer(nn.Module):
    """One 2x causal compression: Conv1d(k=2, s=2) + LN + GELU."""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2,
                                padding=0, bias=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D) — assumes T is even
        x = x.transpose(1, 2)             # (B, D, T)
        x = self.conv(x)                  # (B, D, T/2)
        x = x.transpose(1, 2)             # (B, T/2, D)
        x = self.ln(x)
        x = F.gelu(x)
        return x


class CompressionStack(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalCompressLayer(d_model) for _ in range(n_layers)]
        )
        self.ratio = 2 ** n_layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)               # each (B, T, H, hd)
        q = q.transpose(1, 2)                     # (B, H, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )                                         # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


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


class CompressedTransformer(nn.Module):
    """Drop-in transformer with optional compression stack between
    embedding and the transformer blocks."""

    def __init__(self, cfg: CompressedConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ctx_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        if cfg.n_compress_layers > 0:
            self.compress = CompressionStack(cfg.d_model, cfg.n_compress_layers)
            self.compression_ratio = 2 ** cfg.n_compress_layers
        else:
            self.compress = nn.Identity()
            self.compression_ratio = 1

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie weights
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
        h = self.compress(h)             # (B, T/cr, D) or (B, T, D)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)              # (B, *, V)
