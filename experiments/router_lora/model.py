"""Vanilla TinyTransformer + Router + LoRA + UpdateMechanism.

Two-pass loop semantics:
  pass 1: forward with current LoRA state → first_pass_loss + post-attn hidden
          states at layer `router_layer`.
  routing+update: router produces per-position weights; weighted sum →
          UpdateMechanism produces ΔA, ΔB.
  pass 2: forward with (A + ΔA, B + ΔB) → second_pass_loss.
  loss = second_pass_loss - first_pass_loss + λ * mean(router_weights)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    ctx_len: int = 256
    dropout: float = 0.0
    vocab_size: int = -1
    # Router/LoRA
    router_layer: int = 4              # post-attn hidden states from this block
    lora_layer: int = 4                # LoRA on this block's MLP first-linear (d_model→d_ff)
    lora_rank: int = 8
    router_hidden: int = 128
    sparsity_lambda: float = 0.01


class CausalAttn(nn.Module):
    """Standard causal multi-head self-attention. Plain SDPA."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.dropout(self.out_proj(out))


class LoRAMLP(nn.Module):
    """MLP block where the FIRST linear (d_model→d_ff) supports an additive
    LoRA delta passed at forward time. Used as the lora_layer's FFN."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        # LoRA: A:(rank, d_model), B:(d_ff, rank). LoRA contribution = (x @ A.T) @ B.T.
        # Stored as Parameters so they're persistent and trainable across passages.
        self.lora_A = nn.Parameter(torch.zeros(cfg.lora_rank, cfg.d_model))
        self.lora_B = nn.Parameter(torch.zeros(cfg.d_ff, cfg.lora_rank))
        # Standard LoRA init: A from N(0, 0.02), B as zeros so initial LoRA = 0.
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor,
                delta_A: torch.Tensor | None = None,
                delta_B: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, d_model). Optional delta_A/delta_B add to LoRA matrices for
        this single forward only (used in pass 2)."""
        # Base FFN
        h = self.fc1(x)
        # LoRA contribution
        A_eff = self.lora_A if delta_A is None else self.lora_A + delta_A
        B_eff = self.lora_B if delta_B is None else self.lora_B + delta_B
        lora_out = (x @ A_eff.transpose(-2, -1)) @ B_eff.transpose(-2, -1)  # (B, T, d_ff)
        h = h + lora_out
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

    def forward(self, x: torch.Tensor,
                delta_A: torch.Tensor | None = None,
                delta_B: torch.Tensor | None = None,
                capture_post_attn: bool = False):
        post_attn = x + self.attn(self.ln1(x))   # post-attention residual
        captured = post_attn if capture_post_attn else None
        if self.is_lora_layer:
            x = post_attn + self.ffn(self.ln2(post_attn), delta_A=delta_A, delta_B=delta_B)
        else:
            x = post_attn + self.ffn(self.ln2(post_attn))
        return x, captured


class Router(nn.Module):
    """MLP router: post-attn hidden state → routing weight in [0, 1]."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.router_hidden),
            nn.GELU(),
            nn.Linear(cfg.router_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d_model). Returns (B, T) routing weights in [0, 1]."""
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class UpdateMechanism(nn.Module):
    """Compress routed hidden states → (ΔA, ΔB) for the LoRA layer.

    compressed = sum_t(weights_t * h_t) / max(sum_t(weights_t), 1)
    ΔA flat = U_A(compressed)   # (B, rank * d_model)
    ΔB flat = U_B(compressed)   # (B, d_ff * rank)
    Reshape and squeeze batch (we use batch_size=1 during ingest).
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        d, r, df = cfg.d_model, cfg.lora_rank, cfg.d_ff
        # Output sizes: ΔA is (r, d), ΔB is (df, r). Use small init so initial
        # update is near-zero (model starts as no-op).
        self.U_A = nn.Linear(d, r * d)
        self.U_B = nn.Linear(d, df * r)
        nn.init.normal_(self.U_A.weight, std=0.001)
        nn.init.zeros_(self.U_A.bias)
        nn.init.normal_(self.U_B.weight, std=0.001)
        nn.init.zeros_(self.U_B.bias)
        # Per-update scale that the model can learn to make small.
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, weights: torch.Tensor):
        """h: (B, T, d_model). weights: (B, T) in [0, 1]."""
        cfg = self.cfg
        # Normalize-weighted sum across positions.
        wsum = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        compressed = (h * weights.unsqueeze(-1)).sum(dim=1) / wsum   # (B, d_model)
        delta_A_flat = self.U_A(compressed) * self.scale            # (B, r*d_model)
        delta_B_flat = self.U_B(compressed) * self.scale            # (B, d_ff*r)
        B = h.shape[0]
        delta_A = delta_A_flat.view(B, cfg.lora_rank, cfg.d_model)
        delta_B = delta_B_flat.view(B, cfg.d_ff, cfg.lora_rank)
        # Per-passage update (for ingest we use B=1, so squeeze).
        if B == 1:
            return delta_A.squeeze(0), delta_B.squeeze(0)
        return delta_A, delta_B


class TinyTransformer(nn.Module):
    """6-layer 256-dim transformer with one LoRA-augmented MLP at lora_layer."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        assert cfg.vocab_size > 0
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

    def forward(self, idx: torch.Tensor,
                delta_A: torch.Tensor | None = None,
                delta_B: torch.Tensor | None = None,
                capture_router_layer: bool = False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        captured = None
        for blk in self.blocks:
            x, c = blk(x,
                       delta_A=delta_A if blk.is_lora_layer else None,
                       delta_B=delta_B if blk.is_lora_layer else None,
                       capture_post_attn=(blk.is_router_layer and capture_router_layer))
            if c is not None:
                captured = c
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, captured

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def base_params(self):
        """Generator over base-model parameters (excludes LoRA adapter A/B)."""
        for n, p in self.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                continue
            yield p

    def lora_params(self):
        for blk in self.blocks:
            if blk.is_lora_layer:
                yield blk.ffn.lora_A
                yield blk.ffn.lora_B
                return
