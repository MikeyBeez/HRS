"""Prelude / Recurrent / Coda architecture with three recurrent-stage variants.

Variants:
  A  — LTI refinement baseline (OpenMythos-style)
  B  — MPAR-bias with per-loop LoRA (the hypothesis)
  C  — MPAR-bias without per-loop LoRA (ablation)

All variants share the same Prelude and Coda. Only the RecurrentStage differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.diagonal_attention.config import ModelConfig
from experiments.diagonal_attention.model import Block


# --------------------------------------------------------------------------
# LoRA (adapted from experiments.identity_ae.lora_wrapper). Kept local so we
# can instantiate per-loop copies without touching the base module.
# --------------------------------------------------------------------------

class LoRABranch(nn.Module):
    """Standalone LoRA branch: output = (x @ A @ B) * scaling. No base path.

    Used as an *additive* per-loop correction on top of the shared recurrent
    block. Shape: (in_dim, rank) and (rank, out_dim). Initialized so the
    branch output is zero at init (B = 0)."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A @ self.B) * self.scaling


# --------------------------------------------------------------------------
# LTI injection module for Variant A. Diagonal A, scalar dt, full B.
# --------------------------------------------------------------------------

class LTIInjection(nn.Module):
    """Discretized LTI: h_{t+1} = A_d · h_t + B · e + Recurrent(h_t).

    A = diag(-exp(log_A) · exp(log_dt)); A_d = exp(A) (elementwise since A
    is diagonal). log_A initialized to log(1) = 0 so A ~ -1 and A_d ~ e^{-1}.
    B is a standard nn.Linear.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Linear(dim, dim, bias=False)

    def A_d(self) -> torch.Tensor:
        A = -torch.exp(self.log_A) * torch.exp(self.log_dt)  # (dim,)
        return torch.exp(A)  # (dim,) — elementwise exp for diagonal A

    def forward(self, h_t: torch.Tensor, e: torch.Tensor,
                 recurrent_out: torch.Tensor) -> torch.Tensor:
        # h_{t+1} = A_d * h_t + B · e + recurrent(h_t)
        A_d = self.A_d()  # (dim,)
        return A_d * h_t + self.B(e) + recurrent_out


# --------------------------------------------------------------------------
# RecurrentStage interface + three variants.
# --------------------------------------------------------------------------

class RecurrentStageA(nn.Module):
    """Variant A — LTI refinement baseline."""

    def __init__(self, cfg: ModelConfig, T_default: int = 4):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.block = Block(cfg)
        self.lti = LTIInjection(cfg.d_model)

    def forward(self, e: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        T = T or self.T_default
        h = e
        for _ in range(T):
            rec_out = self.block(h)
            h = self.lti(h, e, rec_out)
        return h


class MPARProjector(nn.Module):
    """mean_pool(h_out): first projects (B, T_seq, d) -> (B, T_seq, rank),
    then mean-pools across sequence to (B, rank)."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        # Init: fan-in scaled by 1/sqrt(rank) per spec.
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.down.weight.mul_(1.0 / (rank ** 0.5))

    def forward(self, h_out: torch.Tensor) -> torch.Tensor:
        # h_out: (B, T_seq, d) -> (B, T_seq, rank) -> mean over T_seq -> (B, rank).
        return self.down(h_out).mean(dim=1)


class MPARUnprojector(nn.Module):
    """MPAR_project: (B, rank) -> (B, 1, d), broadcastable over sequence."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.up.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.up.weight.mul_(1.0 / (rank ** 0.5))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # (B, rank) -> (B, d) -> (B, 1, d) broadcast over sequence.
        return self.up(m).unsqueeze(1)


class RecurrentStageB(nn.Module):
    """Variant B — MPAR-bias with per-loop LoRA.

    Forward supports hooks for Test 5:
      mpar_capture=True: return list of MPARs m_1..m_T alongside final h.
      mpar_override: if provided, replace the MPAR used at the final loop's
                     h_t reconstruction.
    """

    def __init__(self, cfg: ModelConfig, T_default: int = 4, rank_m: int = 128,
                 lora_rank: int = 16, n_loops_for_lora: int = 16):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.rank_m = rank_m
        self.block = Block(cfg)
        self.project_up = MPARUnprojector(cfg.d_model, rank_m)
        self.project_down = MPARProjector(cfg.d_model, rank_m)
        # Per-loop LoRA on the recurrent block's attention output projection.
        # Build enough LoRA branches for any T we expect at inference.
        self.loras = nn.ModuleList([
            LoRABranch(cfg.d_model, cfg.d_model, rank=lora_rank)
            for _ in range(n_loops_for_lora)
        ])

    def forward(self, e: torch.Tensor, T: Optional[int] = None,
                 mpar_capture: bool = False,
                 coda_mpar_override: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        mpar_capture:
            If True, also return the list [m_1..m_T] computed during the
            canonical forward (even when coda_mpar_override is set, so the
            caller can see what the canonical MPARs looked like).
        coda_mpar_override:
            If provided (shape (B, rank_m)), replace m_T for the purpose
            of computing the Coda input. The recurrent loops still run
            normally (to honor causal dependencies in m); only the final
            `e + project_up(m)` bias is swapped. For Test 5 Mode 3, the
            caller may skip the recurrent loops entirely by using T=0
            (see HRSLoop.forward for the externally-exposed knob).
        """
        T = T or self.T_default
        B = e.shape[0]
        m = torch.zeros(B, self.rank_m, device=e.device, dtype=e.dtype)
        captured: List[torch.Tensor] = []

        for t in range(T):
            h_t = e + self.project_up(m)
            block_out = self.block(h_t)
            lora_idx = t if t < len(self.loras) else (t % len(self.loras))
            lora_out = self.loras[lora_idx](h_t)
            h_out = block_out + lora_out
            m = self.project_down(h_out)
            if mpar_capture:
                captured.append(m)

        m_for_coda = coda_mpar_override if coda_mpar_override is not None else m
        final = e + self.project_up(m_for_coda)
        if mpar_capture:
            return final, captured
        return final


class RecurrentStageD(nn.Module):
    """Variant D — concatenation. Each loop cross-attends to all prior
    loop outputs (concatenated along sequence axis) instead of reading
    a compressed MPAR. Per-loop LoRA kept identical to Variant B.

    Architecture (T loops):
        cache = []
        for t in 0..T-1:
            h_t = e + (CrossAttn(q=e, kv=concat(cache)) if cache else 0)
            h_out = block(h_t) + LoRA_t(h_t)
            cache.append(h_out)
        final = e + CrossAttn(q=e, kv=concat(cache))
        output = Coda(final)

    Causal masking on cross-attention: for query position i (0..seq_len-1),
    only attend to cache positions whose local-index within their chunk
    is <= i, to avoid leaking future-position info from prior loops.
    """

    def __init__(self, cfg: ModelConfig, T_default: int = 4,
                 lora_rank: int = 16, n_loops_for_lora: int = 16):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.block = Block(cfg)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=cfg.n_heads,
            batch_first=True, bias=False,
        )
        self.loras = nn.ModuleList([
            LoRABranch(cfg.d_model, cfg.d_model, rank=lora_rank)
            for _ in range(n_loops_for_lora)
        ])

    def _make_causal_mask(self, seq_len: int, cache_len: int,
                           device: torch.device) -> torch.Tensor:
        """Mask shape (seq_len, cache_len). True = blocked.
        Cache is [chunk_0; chunk_1; ...] each of length seq_len. Block key
        position g iff (g % seq_len) > query_i.
        """
        idx = torch.arange(cache_len, device=device)
        local = idx % seq_len                    # (cache_len,)
        q = torch.arange(seq_len, device=device)[:, None]   # (seq_len, 1)
        mask = local[None, :] > q                # (seq_len, cache_len)
        return mask

    def _combine(self, e: torch.Tensor, cache: List[torch.Tensor]) -> torch.Tensor:
        """e + CrossAttn(q=e, kv=concat(cache))."""
        if not cache:
            return e
        kv = torch.cat(cache, dim=1)
        B, seq_len, _ = e.shape
        mask = self._make_causal_mask(seq_len, kv.shape[1], e.device)
        ca_out, _ = self.cross_attn(e, kv, kv, attn_mask=mask,
                                      need_weights=False)
        return e + ca_out

    def forward(self, e: torch.Tensor, T: Optional[int] = None,
                 capture_outputs: bool = False) -> torch.Tensor:
        T = T or self.T_default
        cache: List[torch.Tensor] = []
        captured: List[torch.Tensor] = []

        for t in range(T):
            h_t = self._combine(e, cache)
            block_out = self.block(h_t)
            lora_idx = t if t < len(self.loras) else (t % len(self.loras))
            lora_out = self.loras[lora_idx](h_t)
            h_out = block_out + lora_out
            cache.append(h_out)
            if capture_outputs:
                captured.append(h_out)

        final = self._combine(e, cache)
        if capture_outputs:
            return final, captured
        return final


class RecurrentStageC(nn.Module):
    """Variant C — MPAR-bias, no per-loop differentiation. Identical to B
    minus the LoRA branch."""

    def __init__(self, cfg: ModelConfig, T_default: int = 4, rank_m: int = 128):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.rank_m = rank_m
        self.block = Block(cfg)
        self.project_up = MPARUnprojector(cfg.d_model, rank_m)
        self.project_down = MPARProjector(cfg.d_model, rank_m)

    def forward(self, e: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        T = T or self.T_default
        B = e.shape[0]
        m = torch.zeros(B, self.rank_m, device=e.device, dtype=e.dtype)
        for _ in range(T):
            h_t = e + self.project_up(m)
            h_out = self.block(h_t)
            m = self.project_down(h_out)
        return e + self.project_up(m)


# --------------------------------------------------------------------------
# Aggregation-method ablation variants (built off Variant B's recipe,
# differing only in how cross-sequence information is pooled before the
# rank-128 bottleneck, or whether the bottleneck is used at all).
# --------------------------------------------------------------------------


def _attn_pool(h_out: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Weighted sum across sequence, weights = softmax(h_out @ q / sqrt(d))."""
    d = h_out.shape[-1]
    scores = (h_out @ query) / (d ** 0.5)          # (B, seq)
    weights = torch.softmax(scores, dim=-1)        # (B, seq)
    return (weights.unsqueeze(-1) * h_out).sum(dim=1)   # (B, d)


class RecurrentStageBMax(nn.Module):
    """Variant Bmax — same as B, but aggregation is max-pool over the
    sequence axis instead of mean-pool. Rank-128 bottleneck retained."""

    def __init__(self, cfg: ModelConfig, T_default: int = 4, rank_m: int = 128,
                 lora_rank: int = 16, n_loops_for_lora: int = 16):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.rank_m = rank_m
        self.block = Block(cfg)
        self.project_up = MPARUnprojector(cfg.d_model, rank_m)
        # Pool-then-linear (unlike MPARProjector which is linear-then-pool):
        # max is not commutative with linear.
        self.project_down = nn.Linear(cfg.d_model, rank_m, bias=False)
        self.loras = nn.ModuleList([
            LoRABranch(cfg.d_model, cfg.d_model, rank=lora_rank)
            for _ in range(n_loops_for_lora)
        ])

    def forward(self, e: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        T = T or self.T_default
        B = e.shape[0]
        m = torch.zeros(B, self.rank_m, device=e.device, dtype=e.dtype)
        for t in range(T):
            h_t = e + self.project_up(m)
            block_out = self.block(h_t)
            lora_idx = t if t < len(self.loras) else (t % len(self.loras))
            lora_out = self.loras[lora_idx](h_t)
            h_out = block_out + lora_out
            pooled = h_out.max(dim=1).values      # (B, d)
            m = self.project_down(pooled)         # (B, rank_m)
        return e + self.project_up(m)


class RecurrentStageBAttn(nn.Module):
    """Variant Battn — aggregation is attention-pool with a learned
    d_model-dim query, then rank-128 bottleneck."""

    def __init__(self, cfg: ModelConfig, T_default: int = 4, rank_m: int = 128,
                 lora_rank: int = 16, n_loops_for_lora: int = 16):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.rank_m = rank_m
        self.block = Block(cfg)
        self.project_up = MPARUnprojector(cfg.d_model, rank_m)
        self.project_down = nn.Linear(cfg.d_model, rank_m, bias=False)
        self.attn_query = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.loras = nn.ModuleList([
            LoRABranch(cfg.d_model, cfg.d_model, rank=lora_rank)
            for _ in range(n_loops_for_lora)
        ])

    def forward(self, e: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        T = T or self.T_default
        B = e.shape[0]
        m = torch.zeros(B, self.rank_m, device=e.device, dtype=e.dtype)
        for t in range(T):
            h_t = e + self.project_up(m)
            block_out = self.block(h_t)
            lora_idx = t if t < len(self.loras) else (t % len(self.loras))
            lora_out = self.loras[lora_idx](h_t)
            h_out = block_out + lora_out
            pooled = _attn_pool(h_out, self.attn_query)   # (B, d)
            m = self.project_down(pooled)                 # (B, rank_m)
        return e + self.project_up(m)


class RecurrentStageBFullrank(nn.Module):
    """Variant Bfull — mean-pool over sequence, no compression bottleneck.
    The pooled d_model vector is broadcast back directly as the MPAR bias."""

    def __init__(self, cfg: ModelConfig, T_default: int = 4,
                 lora_rank: int = 16, n_loops_for_lora: int = 16):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.block = Block(cfg)
        self.loras = nn.ModuleList([
            LoRABranch(cfg.d_model, cfg.d_model, rank=lora_rank)
            for _ in range(n_loops_for_lora)
        ])

    def forward(self, e: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        T = T or self.T_default
        B = e.shape[0]
        m = torch.zeros(B, self.cfg.d_model, device=e.device, dtype=e.dtype)
        for t in range(T):
            h_t = e + m.unsqueeze(1)                     # broadcast (B,1,d)
            block_out = self.block(h_t)
            lora_idx = t if t < len(self.loras) else (t % len(self.loras))
            lora_out = self.loras[lora_idx](h_t)
            h_out = block_out + lora_out
            m = h_out.mean(dim=1)                        # (B, d)
        return e + m.unsqueeze(1)


class RecurrentStageBCombined(nn.Module):
    """Variant Bcomb — three parallel aggregation channels (mean / max /
    attention), each with its own rank-128 project_down, concatenated to
    rank 3·rank_m, then one project_up back to d_model for broadcast."""

    def __init__(self, cfg: ModelConfig, T_default: int = 4, rank_m: int = 128,
                 lora_rank: int = 16, n_loops_for_lora: int = 16):
        super().__init__()
        self.cfg = cfg
        self.T_default = T_default
        self.rank_m = rank_m
        self.block = Block(cfg)
        self.project_down_mean = nn.Linear(cfg.d_model, rank_m, bias=False)
        self.project_down_max = nn.Linear(cfg.d_model, rank_m, bias=False)
        self.project_down_attn = nn.Linear(cfg.d_model, rank_m, bias=False)
        self.project_up_combined = nn.Linear(3 * rank_m, cfg.d_model, bias=False)
        self.attn_query = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.loras = nn.ModuleList([
            LoRABranch(cfg.d_model, cfg.d_model, rank=lora_rank)
            for _ in range(n_loops_for_lora)
        ])

    def forward(self, e: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        T = T or self.T_default
        B = e.shape[0]
        m = torch.zeros(B, 3 * self.rank_m, device=e.device, dtype=e.dtype)
        for t in range(T):
            h_t = e + self.project_up_combined(m).unsqueeze(1)
            block_out = self.block(h_t)
            lora_idx = t if t < len(self.loras) else (t % len(self.loras))
            lora_out = self.loras[lora_idx](h_t)
            h_out = block_out + lora_out
            mean_m = self.project_down_mean(h_out.mean(dim=1))
            max_m = self.project_down_max(h_out.max(dim=1).values)
            attn_m = self.project_down_attn(_attn_pool(h_out, self.attn_query))
            m = torch.cat([mean_m, max_m, attn_m], dim=-1)   # (B, 3·rank_m)
        return e + self.project_up_combined(m).unsqueeze(1)


# --------------------------------------------------------------------------
# Full model wrapping prelude + recurrent + coda.
# --------------------------------------------------------------------------

@dataclass
class HRSLoopConfig:
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    ctx_len: int = 512
    vocab_size: int = -1       # set by caller
    prelude_layers: int = 2
    coda_layers: int = 2
    T_default: int = 4
    rank_m: int = 128
    lora_rank: int = 16
    variant: str = "B"         # "A"/"B"/"C"/"D" or aggregation ablations:
                               # "Bmax"/"Battn"/"Bfull"/"Bcomb"
    dropout: float = 0.0

    def to_block_cfg(self) -> ModelConfig:
        """The underlying TinyTransformer Block() takes ModelConfig — reuse it."""
        return ModelConfig(
            d_model=self.d_model, n_heads=self.n_heads, n_layers=1,
            d_ff=self.d_ff, ctx_len=self.ctx_len, dropout=self.dropout,
            vocab_size=self.vocab_size, variant="mha",
        )


class HRSLoop(nn.Module):
    def __init__(self, cfg: HRSLoopConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "set cfg.vocab_size before building"
        assert cfg.variant in ("A", "B", "C", "D",
                                 "Bmax", "Battn", "Bfull", "Bcomb")
        self.cfg = cfg
        bcfg = cfg.to_block_cfg()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.prelude = nn.ModuleList([Block(bcfg) for _ in range(cfg.prelude_layers)])

        if cfg.variant == "A":
            self.recurrent = RecurrentStageA(bcfg, T_default=cfg.T_default)
        elif cfg.variant == "B":
            self.recurrent = RecurrentStageB(bcfg, T_default=cfg.T_default,
                                               rank_m=cfg.rank_m,
                                               lora_rank=cfg.lora_rank)
        elif cfg.variant == "C":
            self.recurrent = RecurrentStageC(bcfg, T_default=cfg.T_default,
                                               rank_m=cfg.rank_m)
        elif cfg.variant == "D":
            self.recurrent = RecurrentStageD(bcfg, T_default=cfg.T_default,
                                               lora_rank=cfg.lora_rank)
        elif cfg.variant == "Bmax":
            self.recurrent = RecurrentStageBMax(bcfg, T_default=cfg.T_default,
                                                  rank_m=cfg.rank_m,
                                                  lora_rank=cfg.lora_rank)
        elif cfg.variant == "Battn":
            self.recurrent = RecurrentStageBAttn(bcfg, T_default=cfg.T_default,
                                                   rank_m=cfg.rank_m,
                                                   lora_rank=cfg.lora_rank)
        elif cfg.variant == "Bfull":
            self.recurrent = RecurrentStageBFullrank(bcfg,
                                                       T_default=cfg.T_default,
                                                       lora_rank=cfg.lora_rank)
        else:  # Bcomb
            self.recurrent = RecurrentStageBCombined(bcfg,
                                                       T_default=cfg.T_default,
                                                       rank_m=cfg.rank_m,
                                                       lora_rank=cfg.lora_rank)

        self.coda = nn.ModuleList([Block(bcfg) for _ in range(cfg.coda_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, T: Optional[int] = None, **kwargs):
        """
        Accepted kwargs (Variant B only):
          mpar_capture: bool — return list of m_1..m_T alongside logits.
          coda_mpar_override: Tensor (B, rank_m) — replace m_T at Coda input.
          skip_recurrent_with_mpar: Tensor (B, rank_m) — skip the recurrent
              stage entirely; use `e + project_up(mpar)` as Coda input.
              Useful for Test 5 Mode 3 ("bias final Coda pass with mean of
              independently-computed MPARs").
        """
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for blk in self.prelude:
            x = blk(x)
        e = x

        if self.cfg.variant == "B":
            skip = kwargs.pop("skip_recurrent_with_mpar", None)
            if skip is not None:
                h = e + self.recurrent.project_up(skip)
                captured = None
            else:
                rec_out = self.recurrent(e, T=T, **kwargs)
                if isinstance(rec_out, tuple):
                    h, captured = rec_out
                else:
                    h, captured = rec_out, None
        elif self.cfg.variant == "D":
            rec_out = self.recurrent(e, T=T, **kwargs)
            if isinstance(rec_out, tuple):
                h, captured = rec_out
            else:
                h, captured = rec_out, None
        else:
            h = self.recurrent(e, T=T)
            captured = None

        for blk in self.coda:
            h = blk(h)
        h = self.ln_f(h)
        logits = self.head(h)
        if captured is not None:
            return logits, captured
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
