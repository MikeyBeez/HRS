"""Compression + Mamba model.

Architecture:
  tokens -> embed + pos -> [4-layer 2x compression stack]
                        -> N (LN -> Mamba -> LN -> FFN) blocks
                        -> ln_f -> linear head -> logits

Same compression stack as `experiments/compression_conv/model.py`. Each
transformer block is replaced by a Mamba block followed by an FFN. The
Mamba operates on the compressed sequence (T/16 positions at training
ctx 2048 = 128 positions).

Block param budget matches transformer: Mamba ≈ 4 D² + small SSM,
FFN = 8 D², total ≈ 12 D² same as attention(4 D²) + FFN(8 D²).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.compression_mamba.mamba_block import MambaBlock
from experiments.compression_conv.model import (
    CausalCompressLayer, CompressionStack,
)


@dataclass
class CompMambaConfig:
    vocab_size: int = 50257
    d_model: int = 384
    n_layers: int = 4
    d_ff: int = 1536
    ctx_len: int = 2048
    n_compress_layers: int = 4         # 16x total
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0


class MambaPlusFFNBlock(nn.Module):
    """LN → Mamba (residual) → LN → FFN (residual)."""
    def __init__(self, cfg: CompMambaConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mamba = MambaBlock(
            d_model=cfg.d_model, d_state=cfg.d_state,
            d_conv=cfg.d_conv, expand=cfg.expand,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.mamba(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CompressedMambaTransformer(nn.Module):
    def __init__(self, cfg: CompMambaConfig):
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

        self.blocks = nn.ModuleList(
            [MambaPlusFFNBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie weights with embedding
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if not getattr(m, "_no_reinit", False):
                if hasattr(m, "weight") and m.weight is not None:
                    if not getattr(m.weight, "_no_reinit", False):
                        nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                    nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Conv1d):
            if not getattr(m, "_no_reinit", False):
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
