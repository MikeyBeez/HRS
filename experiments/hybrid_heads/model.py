"""Hybrid-head transformer: per-layer K/V compression on a subset of heads.

Architecture B from the program spec discussion. The residual stream stays at
full sequence length T. Within each attention block:

  - Q, K, V projected at full length (B, T, D) → split into n_heads heads.
  - For the first `n_full_heads` heads: standard causal SDPA(Q_h, K_h, V_h).
  - For the remaining `n_compressed_heads` heads: K_h and V_h are pooled
    through a 4-layer strided causal conv stack down to length T/16 to give
    K_c, V_c. Then SDPA(Q_h, K_c, V_c) with a query-dependent mask:
    query position t can attend to compressed positions c such that
    c <= floor(t / 16).
  - All head outputs are concatenated at full length T, projected, residual.

Loss is computed at compressed-aligned positions [15, 31, ..., T-17] predicting
tokens at [16, 32, ..., T-16] (matched task with prior experiments).

The compression conv stack (kernel=2, stride=2, applied 4 times → 16×) is
SHARED across all compressed heads in a layer. Different layers have their own
stack. This keeps parameter overhead modest while letting each layer learn its
own pooling.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_full_heads: int = 1            # 0 (pure-compressed), 1, 2, or n_heads (baseline)
    n_layers: int = 6
    d_ff: int = 1024
    ctx_len: int = 1024
    compression_layers: int = 4       # 4 → 16x; per-layer kv compression
    dropout: float = 0.1


class CompressKV(nn.Module):
    """Reduce a (B, T, D) tensor to (B, T/16, D) via 4 strided causal convs.

    Causal because each conv layer has kernel=2 stride=2 padding=0:
    output[i] = f(input[2i], input[2i+1]). Stack of 4 → 16x reduction.
    """
    def __init__(self, head_dim, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(head_dim, head_dim, kernel_size=2, stride=2, padding=0),
            ))
        self.lns = nn.ModuleList(
            [nn.LayerNorm(head_dim) for _ in range(n_layers)]
        )
        self.ratio = 2 ** n_layers

    def forward(self, x):
        # x: (B, T, D_head)
        for conv, ln in zip(self.layers, self.lns):
            x = x.transpose(1, 2)
            x = conv(x)
            x = x.transpose(1, 2)
            x = ln(x)
            x = F.gelu(x)
        return x   # (B, T/16, D_head)


class HybridAttention(nn.Module):
    """Multi-head attention with first n_full_heads heads doing standard SDPA
    and remaining heads using compressed K/V."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide n_heads"
        self.n_heads = cfg.n_heads
        self.n_full = cfg.n_full_heads
        self.n_comp = cfg.n_heads - cfg.n_full_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.dropout = cfg.dropout

        if self.n_comp > 0:
            # Per-head compression of K/V. We share the compression module
            # across compressed heads to keep param count modest.
            self.compress_k = CompressKV(self.head_dim, cfg.compression_layers)
            self.compress_v = CompressKV(self.head_dim, cfg.compression_layers)
            self.compression_ratio = self.compress_k.ratio
        else:
            self.compress_k = None
            self.compress_v = None
            self.compression_ratio = 1

        # Cache for the (T x T/16) compressed-attention causal mask
        self._compressed_mask_cache = {}

    def _compressed_mask(self, T_q, T_kv, device):
        """For SDPA: float mask (T_q, T_kv) where -inf disallows.
        Query at position t can attend to compressed position c if
        c <= floor(t / compression_ratio)."""
        key = (T_q, T_kv, device)
        if key not in self._compressed_mask_cache:
            # Allowed (True): c <= floor(t / cr)
            t = torch.arange(T_q, device=device).unsqueeze(1)        # (T_q, 1)
            c = torch.arange(T_kv, device=device).unsqueeze(0)       # (1, T_kv)
            allowed = c <= (t // self.compression_ratio)              # (T_q, T_kv) bool
            mask = torch.zeros(T_q, T_kv, device=device)
            mask.masked_fill_(~allowed, float('-inf'))
            self._compressed_mask_cache[key] = mask
        return self._compressed_mask_cache[key]

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                # each (B, T, H, hd)
        # transpose to (B, H, T, hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        outputs = []

        # ---- Full-attention heads ----
        if self.n_full > 0:
            q_f = q[:, :self.n_full]               # (B, n_full, T, hd)
            k_f = k[:, :self.n_full]
            v_f = v[:, :self.n_full]
            o_f = F.scaled_dot_product_attention(
                q_f, k_f, v_f, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )                                       # (B, n_full, T, hd)
            outputs.append(o_f)

        # ---- Compressed-attention heads ----
        if self.n_comp > 0:
            q_c = q[:, self.n_full:]               # (B, n_comp, T, hd)
            k_full = k[:, self.n_full:]            # (B, n_comp, T, hd)
            v_full = v[:, self.n_full:]
            # Compress K/V per head: CompressKV expects (B*, T, hd)
            kc_in = k_full.reshape(B * self.n_comp, T, self.head_dim)
            vc_in = v_full.reshape(B * self.n_comp, T, self.head_dim)
            k_c = self.compress_k(kc_in).reshape(
                B, self.n_comp, T // self.compression_ratio, self.head_dim
            )
            v_c = self.compress_v(vc_in).reshape(
                B, self.n_comp, T // self.compression_ratio, self.head_dim
            )
            # SDPA with custom mask
            T_kv = k_c.shape[2]
            mask = self._compressed_mask(T, T_kv, q_c.device)
            o_c = F.scaled_dot_product_attention(
                q_c, k_c, v_c, attn_mask=mask, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )                                       # (B, n_comp, T, hd)
            outputs.append(o_c)

        # Concatenate full + compressed head outputs along the head axis
        out = torch.cat(outputs, dim=1)             # (B, n_heads, T, hd)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        return self.out(out)


class HybridBlock(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = HybridAttention(cfg)
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


class HybridTransformer(nn.Module):
    """Standard transformer with HybridBlock attention.

    Output is at full length (B, T, V). The training task takes loss at
    compressed-aligned positions [15, 31, ..., T-17] (matched task with
    prior experiments).
    """
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ctx_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([HybridBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        # Mark output as full-length so train code knows to pull
        # baseline-style logits at compressed-aligned positions
        self.compression_ratio = 1   # the residual stream is at full length

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
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)
