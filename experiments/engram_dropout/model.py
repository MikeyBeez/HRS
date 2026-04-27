"""Tiny-Shakespeare-scale reproduction of V23's engram + cross-attention pipeline.

Faithful but slim. Pieces preserved from the WT-103 V23:
- PerHeadBonsignoreAttention: per-head learned MLP refining `-‖q-k‖²/τ_h`
  (initialized as near-identity so initial behavior matches the exponential
  prior; can co-evolve under training)
- EngramEncoder: window mean-pool → 2-layer MLP → K engrams per window
- EngramCrossAttention: Q from sequence, K/V from engrams, learned gate
- engram_reconstruction_loss: cosine similarity vs. window means

Differences from V23 (documented per spec):
- d=256, n_heads=4, n_layers=6, char-level, ctx=256 (vs. WT-103's 30M-param config)
- No 3-phase training schedule; MLPs trainable from step 0 (V23's calibration
  protocol is WT-103-specific and we don't have the budget for it on this scale)
- No KV-cache (eval is short enough that we don't need it)
- No RoPE (use learned position embeddings — simpler, fine at ctx=256)
- Engrams computed from same-batch hidden states each step (no V23-style
  external buffer — char-Shakespeare doesn't have memorable past samples to
  retrieve from in a meaningful way; the experiment is whether the cross-attn
  *channel* is robust to dropout, not whether retrieval works)
- Cross-attention engram injection at layer 2 (V23's layer 3 was found dead;
  spec says preserve mid-stack but avoid the dead layer)
- Engram dropout: per-batch with prob p, the engram tensor passed into the
  cross-attention is zeroed (encoder still runs; recon loss always applied)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EngramDropoutConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024            # 4 * d_model
    ctx_len: int = 256
    dropout: float = 0.0
    vocab_size: int = -1
    # Engram pipeline
    engram_window: int = 64     # W: tokens per window (smaller than V23's 128 to give
                                #     progressive engram access at ctx=256)
    engram_k: int = 4           # K: engrams per window
    engram_layer: int = 2       # cross-attn engram injected here (NOT layer 3)
    engram_extract_layer: int = 1  # encoder reads hidden states after this layer
    engram_dropout_p: float = 0.0   # per-batch zero-out probability
    # Bonsignore Kernel
    mlp_hidden: int = 32        # per-head MLP hidden width
    # Recon loss
    recon_loss_weight: float = 0.1


# ============================================================
# Per-head Bonsignore self-attention (faithful to model.py:PerHeadBonsignoreAttention,
# minus RoPE/KV-cache).
# ============================================================
class PerHeadBonsignoreAttention(nn.Module):
    def __init__(self, cfg: EngramDropoutConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d, H, dh = cfg.d_model, cfg.n_heads, self.head_dim

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.log_tau = nn.Parameter(torch.full((H,), math.log(float(dh))))
        self.head_alphas = nn.Parameter(torch.ones(H))
        self.head_output_scalars = nn.Parameter(torch.zeros(H))

        # Per-head MLP refining the exponential score (1 → mlp_hidden → 1).
        # Init as near-identity so initial behavior is the V23-style exponential.
        self.head_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, cfg.mlp_hidden),
                nn.GELU(),
                nn.Linear(cfg.mlp_hidden, 1),
            )
            for _ in range(H)
        ])
        for mlp in self.head_mlps:
            nn.init.uniform_(mlp[0].weight, -0.01, 0.01)
            nn.init.zeros_(mlp[0].bias)
            nn.init.uniform_(mlp[2].weight, -0.01, 0.01)
            nn.init.zeros_(mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_sq = (q ** 2).sum(dim=-1, keepdim=True)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)
        dot = q @ k.transpose(-2, -1)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot       # (B, H, T, T)

        taus = self.log_tau.exp().view(1, H, 1, 1)
        exp_scores = -distances / taus                            # log-space exponential

        # Per-head MLP refinement on top of the exponential prior.
        # Apply per head: reshape to (B*T*T, 1), mlp, reshape back.
        # (Done head-by-head to preserve per-head MLP weights.)
        refined = torch.empty_like(exp_scores)
        for h in range(H):
            flat = exp_scores[:, h].reshape(-1, 1)
            refined[:, h] = self.head_mlps[h](flat).reshape(exp_scores[:, h].shape)
        # Blend: alpha modulates how much the MLP refines vs. raw exponential.
        # (V23 uses per-head sharpness rather than literal blending; we replicate.)
        alphas = torch.sigmoid(self.head_alphas).view(1, H, 1, 1)
        scores = refined * alphas

        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v

        head_scales = F.softplus(self.head_output_scalars).view(1, H, 1, 1)
        out = out * head_scales

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_dropout(self.out_proj(out))

    def get_diagnostics(self) -> dict:
        with torch.no_grad():
            taus = self.log_tau.exp().detach().tolist()
            alphas = torch.sigmoid(self.head_alphas.detach()).tolist()
            scales = F.softplus(self.head_output_scalars.detach()).tolist()
        return {"taus": taus, "alphas": alphas, "scales": scales}


# ============================================================
# Engram encoder: window mean-pool → 2-layer MLP → K engrams per window.
# Mirrors engram.py:EngramEncoder.
# ============================================================
class EngramEncoder(nn.Module):
    def __init__(self, cfg: EngramDropoutConfig):
        super().__init__()
        self.W = cfg.engram_window
        self.K = cfg.engram_k
        d = cfg.d_model
        self.encoder = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, self.K * d),
        )
        self.norm = nn.LayerNorm(d)
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        n_windows = T // self.W
        if n_windows == 0:
            return torch.zeros(B, 0, D, device=h.device, dtype=h.dtype)
        usable = n_windows * self.W
        windows = h[:, :usable].reshape(B, n_windows, self.W, D)
        pooled = windows.mean(dim=2)                              # (B, n_windows, D)
        encoded = self.encoder(pooled)                            # (B, n_windows, K*D)
        engrams = encoded.reshape(B, n_windows * self.K, D)
        return self.norm(engrams)


# ============================================================
# Engram cross-attention: Q from sequence, K/V from engrams, learnable gate.
# Mirrors engram.py:EngramCrossAttention.
# ============================================================
class EngramCrossAttention(nn.Module):
    def __init__(self, cfg: EngramDropoutConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.ln = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.gate_scalar = nn.Parameter(torch.tensor(0.0))
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=0.02)
        # Near-zero output proj so cross-attn starts as a near-no-op.
        nn.init.normal_(self.out_proj.weight, std=0.001)

    def forward(self, x: torch.Tensor, engram: torch.Tensor,
                window_size: int | None = None,
                k_per_window: int | None = None) -> torch.Tensor:
        """Cross-attend, with per-window causal masking when W and K are given.

        For query at position p in window i = p // window_size, valid engram
        indices are [0, i*K). Without the mask, the engram for window i would
        leak future tokens within window i into the prediction at p.
        """
        B, T, D = x.shape
        E = engram.shape[1]
        if E == 0:
            return torch.zeros_like(x)
        h = self.ln(x)
        q = self.q_proj(h).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(engram).reshape(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(engram).reshape(B, E, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if window_size is not None and k_per_window is not None:
            # Build (T, E) bool mask: True = valid key.
            j = torch.arange(T, device=x.device).unsqueeze(1)            # (T, 1)
            e = torch.arange(E, device=x.device).unsqueeze(0)            # (1, E)
            i = j // window_size                                          # window of query j
            engram_window_idx = e // k_per_window                         # window of engram e
            attn_mask = engram_window_idx < i                             # (T, E)
            # Expand for SDPA's expected shape (B, H, T, E).
            attn_mask = attn_mask.view(1, 1, T, E)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        # SDPA returns NaN when an entire row is masked-out (no valid keys).
        # That happens for queries in window 0 (no past windows). Replace NaN
        # rows with zeros — equivalent to "this query gets no cross-attn output".
        if attn_mask is not None:
            out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.resid_dropout(self.out_proj(out))
        gate = torch.sigmoid(self.gate_logit) * F.softplus(self.gate_scalar)
        return gate * out

    def gate_value(self) -> float:
        with torch.no_grad():
            return float(torch.sigmoid(self.gate_logit) * F.softplus(self.gate_scalar))


# ============================================================
# engram_reconstruction_loss — cosine similarity per window.
# Mirrors engram.py:engram_reconstruction_loss.
# ============================================================
def engram_reconstruction_loss(original_h: torch.Tensor,
                                engrams: torch.Tensor,
                                window_size: int) -> torch.Tensor:
    B, T, D = original_h.shape
    n_windows = T // window_size
    if n_windows == 0 or engrams.shape[1] == 0:
        return torch.tensor(0.0, device=original_h.device)
    usable = n_windows * window_size
    windows = original_h[:, :usable].reshape(B, n_windows, window_size, D)
    pooled = windows.mean(dim=2)                                  # (B, n_windows, D)
    K = engrams.shape[1] // n_windows
    engram_w = engrams[:, :n_windows * K].reshape(B, n_windows, K, D)
    engram_pooled = engram_w.mean(dim=2)
    sim = F.cosine_similarity(pooled, engram_pooled, dim=-1)      # (B, n_windows)
    return (1.0 - sim).mean()


# ============================================================
# Block + full model.
# ============================================================
class Block(nn.Module):
    def __init__(self, cfg: EngramDropoutConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_cross_attn = (layer_idx == cfg.engram_layer)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = PerHeadBonsignoreAttention(cfg)
        if self.use_cross_attn:
            self.cross_attn = EngramCrossAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, engram: torch.Tensor | None = None,
                engram_window: int | None = None,
                engram_k: int | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.use_cross_attn and engram is not None:
            x = x + self.cross_attn(x, engram, window_size=engram_window,
                                     k_per_window=engram_k)
        x = x + self.ffn(self.ln2(x))
        return x


class EngramTinyTransformer(nn.Module):
    def __init__(self, cfg: EngramDropoutConfig):
        super().__init__()
        assert cfg.vocab_size > 0
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        # Initialize block linears explicitly (qkv, out_proj, ffn) — but DO NOT
        # touch the per-head MLPs or cross-attn out_proj which had custom init.
        for blk in self.blocks:
            for name, p in blk.attn.named_parameters(recurse=False):
                # head_mlps / log_tau / head_alphas / head_output_scalars are not
                # named at recurse=False level (they're in submodules / Parameters).
                pass
            nn.init.normal_(blk.attn.qkv.weight, std=0.02)
            nn.init.normal_(blk.attn.out_proj.weight, std=0.02)
            for layer in blk.ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            if blk.use_cross_attn:
                # The Q/K/V projections were N(0, 0.02) in the cross-attn __init__;
                # the out_proj was deliberately near-zero (std=0.001). Leave both.
                pass

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                    # tied (overrides above init)
        self.engram_encoder = EngramEncoder(cfg)

    def forward(self, idx: torch.Tensor,
                drop_engram: bool = False,
                return_recon: bool = False) -> dict:
        B, T = idx.shape
        cfg = self.cfg
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        engram = None
        h_at_extract = None
        for i, blk in enumerate(self.blocks):
            # Compute engrams from the hidden state right after the extract layer,
            # before the engram-injection block consumes them.
            if i == cfg.engram_layer:
                # `x` here is the input to the engram-injection layer (output of layer i-1).
                # Encode engrams from the hidden state at engram_extract_layer's output.
                # Since extract_layer = engram_layer - 1 by default, that's `x`.
                h_at_extract = x
                engram = self.engram_encoder(h_at_extract)
                if drop_engram:
                    engram = torch.zeros_like(engram)
            x = blk(x, engram=engram,
                    engram_window=cfg.engram_window, engram_k=cfg.engram_k)
        x = self.ln_f(x)
        logits = self.head(x)
        out = {"logits": logits}
        if return_recon:
            out["engram"] = engram
            out["h_at_extract"] = h_at_extract
        return out

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_diagnostics(self) -> dict:
        diags = {"layers": []}
        for i, blk in enumerate(self.blocks):
            d = blk.attn.get_diagnostics()
            d["layer"] = i
            if blk.use_cross_attn:
                d["cross_attn_gate"] = blk.cross_attn.gate_value()
            diags["layers"].append(d)
        return diags
