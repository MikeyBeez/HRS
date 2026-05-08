"""Multi-mechanism supernet transformer.

12 head mechanisms competing in each attention layer with learned gates.
Floor-then-release schedule: gates floored at 0.5 for first half of training,
free in second half.

Architecture:
  embed + pos_emb -> N (LayerNorm + SupernetLayer + LayerNorm + FFN) blocks
                   -> ln_f -> tied head -> logits
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.multi_mechanism_supernet.mechanisms import (
    build_mechanisms, MECHANISM_NAMES,
)


@dataclass
class SupernetConfig:
    vocab_size: int = 50257
    d_model: int = 288               # divisible by 12
    n_heads: int = 12                # one per mechanism
    n_layers: int = 6
    d_ff: int = 1152                 # 4 * d_model
    ctx_len: int = 1024
    dropout: float = 0.1
    gate_floor: float = 0.5          # floor for floored period
    gate_init: float = 1.0           # initial gate magnitude


class SupernetLayer(nn.Module):
    """12 head mechanisms in parallel + learned per-mechanism gating."""

    def __init__(self, cfg: SupernetConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.mechanisms = nn.ModuleList(
            build_mechanisms(cfg.d_model, self.head_dim, cfg.ctx_len)
        )
        # One scalar gate per mechanism
        self.gates = nn.Parameter(torch.full((cfg.n_heads,), cfg.gate_init))
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def gate_values(self, floor_active: bool):
        """Return effective gate values (with optional floor applied)."""
        g = self.gates
        if floor_active:
            g = torch.clamp(g, min=self.cfg.gate_floor)
        return g

    def forward(self, x, floor_active=True):
        B, T, D = x.shape
        outs = [m(x) for m in self.mechanisms]      # each (B, T, hd)
        cat = torch.cat(outs, dim=-1)                # (B, T, n_heads * hd) = (B, T, D)
        # Apply per-mechanism gating
        g = self.gate_values(floor_active)
        # Broadcast each gate to its head_dim slice: shape (n_heads * head_dim,) = (D,)
        g_per_dim = g.repeat_interleave(self.head_dim)
        cat = cat * g_per_dim.unsqueeze(0).unsqueeze(0)
        return self.out_proj(cat)


class SupernetBlock(nn.Module):
    def __init__(self, cfg: SupernetConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SupernetLayer(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x, floor_active=True):
        x = x + self.attn(self.ln1(x), floor_active=floor_active)
        x = x + self.ffn(self.ln2(x))
        return x


class SupernetTransformer(nn.Module):
    def __init__(self, cfg: SupernetConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ctx_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([SupernetBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.compression_ratio = 1   # output at full length

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, "_no_init") and m._no_init:
                return
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, floor_active=True):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            h = block(h, floor_active=floor_active)
        h = self.ln_f(h)
        return self.head(h)

    def get_all_gate_values(self, floor_active=False):
        """Return list of (layer_idx, gate_tensor) tuples for inspection."""
        out = []
        for i, block in enumerate(self.blocks):
            out.append((i, block.attn.gate_values(floor_active=floor_active).detach().cpu().clone()))
        return out
