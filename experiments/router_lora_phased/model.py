"""TinyTransformer (BPE) with rank-32 LoRA on layer-4 MLP first linear,
plus a router that produces a per-passage scalar in [0,1] scaling the LoRA.

Three usage modes (set via lora_scale at forward):
  lora_scale = 1.0  -> LoRA fully active (Phase 2 training; Condition B)
  lora_scale = 0.0  -> LoRA disabled         (Phase 1 training; Condition C)
  lora_scale ∈ R    -> set by router         (Phase 3 training; Condition A)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    ctx_len: int = 512
    dropout: float = 0.1
    lora_layer: int = 4
    router_layer: int = 4
    lora_rank: int = 32
    router_hidden: int = 128


class CausalAttn(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.dropout(self.out_proj(out))


class LoRAMLP(nn.Module):
    """MLP with rank-r LoRA on the first linear (d_model → d_ff). Forward
    accepts a `lora_scale` in [0, 1] (or any tensor broadcastable to (B, T, 1))
    that scales the LoRA contribution per token / per passage.
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.lora_A = nn.Parameter(torch.zeros(cfg.lora_rank, cfg.d_model))
        self.lora_B = nn.Parameter(torch.zeros(cfg.d_ff, cfg.lora_rank))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)   # B=0 → initial LoRA contribution = 0

    def forward(self, x, lora_scale=None):
        h = self.fc1(x)
        if lora_scale is not None:
            # lora_scale: scalar, (B,), or (B, T) — broadcast to (B, T, 1).
            if not torch.is_tensor(lora_scale):
                lora_scale = torch.tensor(lora_scale, device=x.device, dtype=x.dtype)
            if lora_scale.ndim == 0:
                gate = lora_scale
            elif lora_scale.ndim == 1:
                gate = lora_scale.view(-1, 1, 1)
            elif lora_scale.ndim == 2:
                gate = lora_scale.unsqueeze(-1)
            else:
                gate = lora_scale
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T  # (B, T, d_ff)
            h = h + gate * lora_out
        h = F.gelu(h)
        return self.fc2(h)


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_lora_layer = (layer_idx == cfg.lora_layer)
        self.is_router_layer = (layer_idx == cfg.router_layer)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalAttn(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if self.is_lora_layer:
            self.ffn = LoRAMLP(cfg)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Linear(cfg.d_ff, cfg.d_model),
            )

    def forward(self, x, lora_scale=None, capture_post_attn=False):
        post_attn = x + self.attn(self.ln1(x))
        captured = post_attn if capture_post_attn else None
        if self.is_lora_layer:
            x = post_attn + self.ffn(self.ln2(post_attn), lora_scale=lora_scale)
        else:
            x = post_attn + self.ffn(self.ln2(post_attn))
        return x, captured


class Router(nn.Module):
    """MLP on (mean-pooled) post-attention hidden states → scalar in [0, 1]."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.router_hidden),
            nn.GELU(),
            nn.Linear(cfg.router_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d_model). Returns (B,) per-passage scalar in [0, 1]."""
        pooled = h.mean(dim=1)        # (B, d_model)
        return torch.sigmoid(self.mlp(pooled)).squeeze(-1)


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        for blk in self.blocks:
            nn.init.normal_(blk.attn.qkv.weight, std=0.02)
            nn.init.normal_(blk.attn.out_proj.weight, std=0.02)
            if not blk.is_lora_layer:
                for layer in blk.ffn:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=0.02)
                        if layer.bias is not None: nn.init.zeros_(layer.bias)
            else:
                nn.init.normal_(blk.ffn.fc1.weight, std=0.02); nn.init.zeros_(blk.ffn.fc1.bias)
                nn.init.normal_(blk.ffn.fc2.weight, std=0.02); nn.init.zeros_(blk.ffn.fc2.bias)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, lora_scale=None, capture_router_layer=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        captured = None
        for blk in self.blocks:
            ls = lora_scale if blk.is_lora_layer else None
            x, c = blk(x, lora_scale=ls,
                        capture_post_attn=(blk.is_router_layer and capture_router_layer))
            if c is not None:
                captured = c
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, captured

    def base_params(self):
        for n, p in self.named_parameters():
            if "lora_A" in n or "lora_B" in n: continue
            yield p

    def lora_params(self):
        for blk in self.blocks:
            if blk.is_lora_layer:
                yield blk.ffn.lora_A
                yield blk.ffn.lora_B
                return

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
