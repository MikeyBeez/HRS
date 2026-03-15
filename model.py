"""HRS model — supports v1 (routed), v2 (attention->conv + PEER), and v8 (BDH) architectures.

v1: Backbone transformer with router + tiered compute (conv, expert, attn, sink)
v2: Fixed attention->conv backbone with PEER FFN replacing standard MLP
v8: v4 base + BDH: virtual synapse, hub routing, monosemantic sparsity

Shared components: dual heads, engram encoder/injector, RoPE.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig, ModelConfig, PEERConfig, MemoryMLPTrainConfig, BDHConfig
from tiers import ConvTier, ExpertTier, AttentionTier, SinkTier, RotaryEmbedding, apply_rotary_emb, rotate_half
from router import TokenRouter, routing_balance_loss, routing_entropy_loss, routing_flops_loss
from engram import EngramEncoder, EngramInjector, GatedEngramInjector, EngramReplacer, engram_reconstruction_loss
from peer import PEER
from bdh import VirtualSynapse, routing_hub_loss, apply_sparsity_bottleneck


@dataclass
class HRSOutput:
    logits: torch.Tensor
    layer_representations: list       # per-layer hidden states for locality loss
    routing_weights: list             # per-layer routing decisions (v1 only)
    routing_balance_loss: torch.Tensor  # aggregate balance loss across layers
    routing_entropy_loss: torch.Tensor  # negative per-token entropy (exploration)
    routing_flops_loss: torch.Tensor  # expected FLOPs cost of routing decisions
    engrams: torch.Tensor             # engram vectors (if applicable)
    engram_recon_loss: torch.Tensor   # engram reconstruction loss
    attention_weights: list           # for evaluation metrics (optional)
    peer_indices: list                # PEER expert indices (v2, for utilization tracking)
    per_token_loss: torch.Tensor = None      # (B, T) for v5 loss cache update
    replacement_gates: torch.Tensor = None   # (B, n_windows) for v5 gate logging
    remember_gates: torch.Tensor = None      # (B, n_windows) for v6 gate logging
    hidden_states: torch.Tensor = None       # (B, T, D) ln_f output for v7 Memory MLP
    memory_logits: torch.Tensor = None       # (B, T, V) from Memory MLP (v7)
    v7_router_weights: torch.Tensor = None   # (B, T, 2) base/memory blend weights (v7)
    # v8 BDH metrics
    bdh_focus_magnitude: float = None        # mean |alpha * gain| across layers
    bdh_sparsity_level: float = None         # actual fraction of zeros after bottleneck
    bdh_hub_distribution: list = None        # sorted tier utilization for logging


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention with RoPE (backbone attention)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)

    def forward(self, x, return_weights=False, focus_qk=None):
        """Forward pass with optional BDH virtual synapse focus.

        Args:
            x: (B, T, D) input hidden states
            return_weights: collect attention weights for metrics
            focus_qk: optional (focus_q, focus_k) tuple from VirtualSynapse
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # v8 BDH: apply virtual synapse as additive bias on Q/K (SDPA compatible)
        if focus_qk is not None:
            focus_q, focus_k = focus_qk
            if focus_q is not None and focus_k is not None:
                # Per-head scalar gain applied as Q/K scaling
                # gain = dot(fq, fk) per head -> (B, H, 1, 1)
                gain = (focus_q * focus_k).sum(dim=-1, keepdim=True).unsqueeze(-1)
                # Scale Q by sqrt(1 + alpha * gain) to approximate multiplicative score gain
                # For small alpha*gain: (Q*s)@K^T/sqrt(d) ≈ Q@K^T/sqrt(d) * (1 + alpha*gain)
                q = q * (1.0 + gain).sqrt()

        if return_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = attn_weights @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            attn_weights = None

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, attn_weights


class MLP(nn.Module):
    """Standard 2-layer MLP (FFN)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class CausalConv(nn.Module):
    """Causal depthwise 1D convolution block (v2 structural layer).

    Same as ConvTier but used as a structural component, not a routed tier.
    """

    def __init__(self, cfg: ModelConfig, kernel_size: int = 7):
        super().__init__()
        d = cfg.d_model
        self.causal_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            d, d, kernel_size=kernel_size, padding=0, groups=d, bias=False
        )
        self.out_proj = nn.Linear(d, d, bias=cfg.bias)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal depthwise convolution.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        h = x.transpose(1, 2)  # (B, D, T)
        h = F.pad(h, (self.causal_pad, 0))  # left-pad for causality
        h = self.conv(h).transpose(1, 2)  # (B, T, D)
        h = self.dropout(self.norm(self.out_proj(h)))
        return h


# ============================================================
# v1 Block (routing + tiered compute)
# ============================================================

class HRSBlock(nn.Module):
    """v1/v8 HRS transformer block.

    For dense_baseline/dual_head: standard pre-norm transformer block.
    For routed configs: adds router + tiered compute after attention.
    v8 BDH adds: virtual synapse on attention, sparsity bottleneck on tier inputs.
    """

    def __init__(self, cfg: ExperimentConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        model_cfg = cfg.model

        # Attention (always present)
        self.ln1 = nn.LayerNorm(model_cfg.d_model)
        self.attn = CausalSelfAttention(model_cfg)
        self.ln2 = nn.LayerNorm(model_cfg.d_model)

        # v8 BDH: virtual synapse per block
        self.use_bdh = cfg.uses_bdh()
        if self.use_bdh and cfg.bdh.virtual_synapse_enabled:
            self.virtual_synapse = VirtualSynapse(model_cfg, cfg.bdh)

        # Router + tiered compute REPLACES MLP
        self.use_router = cfg.uses_router()
        self.use_peer = cfg.uses_peer()
        # v4/v8: PEER as unconditional FFN + 3-tier routing (no expert tier)
        self.use_peer_ffn = self.use_peer and cfg.router.n_tiers == 3
        if self.use_router:
            self.router = TokenRouter(model_cfg, cfg.router)

            self.conv_tier = ConvTier(model_cfg, cfg.tier)
            if self.use_peer_ffn:
                # v4/v8: PEER runs unconditionally, not as a routed tier
                self.peer_ffn = PEER(model_cfg, cfg.peer)
                self.ln_peer = nn.LayerNorm(model_cfg.d_model)
                self.peer_output_gate = nn.Parameter(torch.ones(1) * 0.1)
            elif self.use_peer:
                # v3: PEER replaces ExpertTier in the routing framework
                self.expert_tier = PEER(model_cfg, cfg.peer)
            else:
                self.expert_tier = ExpertTier(model_cfg, cfg.tier)
            self.attn_tier = AttentionTier(model_cfg)
            self.sink_tier = SinkTier(cfg.tier)

            self.tier_output_gate = nn.Parameter(
                torch.full((model_cfg.d_model,), 0.1)
            )

            self.use_sink = cfg.uses_sink()
            if self.use_sink:
                self.sink_kv_scale = nn.Parameter(
                    torch.tensor(cfg.tier.sink_init_scale)
                )
        else:
            self.mlp = MLP(model_cfg)

    def forward(
        self, x, step: int = 0, return_weights: bool = False,
        engrams: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # v8 BDH: compute virtual synapse focus from engrams
        focus_qk = None
        if self.use_bdh and hasattr(self, 'virtual_synapse') and engrams is not None:
            focus_qk = self.virtual_synapse(engrams)

        attn_out, attn_w = self.attn(
            self.ln1(x), return_weights=return_weights, focus_qk=focus_qk,
        )
        x = x + attn_out

        routing_w = None

        if self.use_router:
            h = self.ln2(x)

            # v8 BDH: monosemantic sparsity bottleneck on tier inputs
            if self.use_bdh and self.cfg.bdh.sparsity_enabled:
                h_tier = apply_sparsity_bottleneck(
                    h, rho=self.cfg.bdh.sparsity_rho,
                    activation=self.cfg.bdh.sparsity_activation,
                )
            else:
                h_tier = h

            routing_w = self.router(h, step=step)

            if self.use_peer_ffn:
                # v4/v8: 3-tier routing (conv, attn, sink) + unconditional PEER
                conv_out = self.conv_tier(h_tier)
                attn_out2 = self.attn_tier(h_tier)
                sink_out = self.sink_tier(h_tier)

                tier_outputs = torch.stack(
                    [conv_out, attn_out2, sink_out], dim=-1
                )
                w = routing_w.unsqueeze(2)
                combined = (tier_outputs * w).sum(dim=-1)
                x = x + combined * self.tier_output_gate

                if self.use_sink:
                    sink_weight = routing_w[:, :, 2]  # sink is index 2 in 3-tier
                    kv_scale = 1.0 - sink_weight * (1.0 - self.sink_kv_scale.abs())
                    x = x * kv_scale.unsqueeze(-1)

                # Unconditional PEER FFN (runs for every token)
                peer_out = self.peer_ffn(self.ln_peer(x))
                x = x + peer_out * self.peer_output_gate
            else:
                # v1/v3: 4-tier routing (conv, expert, attn, sink)
                conv_out = self.conv_tier(h_tier)
                expert_out = self.expert_tier(h_tier)
                attn_out2 = self.attn_tier(h_tier)
                sink_out = self.sink_tier(h_tier)

                tier_outputs = torch.stack(
                    [conv_out, expert_out, attn_out2, sink_out], dim=-1
                )
                w = routing_w.unsqueeze(2)
                combined = (tier_outputs * w).sum(dim=-1)
                x = x + combined * self.tier_output_gate

                if self.use_sink:
                    sink_weight = routing_w[:, :, 3]
                    kv_scale = 1.0 - sink_weight * (1.0 - self.sink_kv_scale.abs())
                    x = x * kv_scale.unsqueeze(-1)
        else:
            x = x + self.mlp(self.ln2(x))

        return x, routing_w, attn_w


# ============================================================
# v2 Block (attention->conv backbone + PEER FFN)
# ============================================================

class HRSv2Block(nn.Module):
    """v2 HRS block: structural attention or conv + PEER or MLP FFN.

    Layer structure determined by layer_idx:
    - Layers 0..n_attn-1: CausalSelfAttention + FFN
    - Layers n_attn..n_layers-1: CausalConv + FFN

    FFN is PEER when peer_cfg.enabled, else standard MLP.
    """

    def __init__(self, cfg: ExperimentConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        model_cfg = cfg.model

        n_attn = cfg.n_attention_layers()
        self.use_attention = (layer_idx < n_attn)

        # Block 1: Attention or Conv
        self.ln1 = nn.LayerNorm(model_cfg.d_model)
        if self.use_attention:
            self.attn = CausalSelfAttention(model_cfg)
        else:
            self.conv = CausalConv(model_cfg, kernel_size=cfg.tier.conv_kernel_size)

        # Block 2: PEER or MLP
        self.ln2 = nn.LayerNorm(model_cfg.d_model)
        self.use_peer = cfg.uses_peer()
        if self.use_peer:
            self.ffn = PEER(model_cfg, cfg.peer)
        else:
            self.ffn = MLP(model_cfg)

    def forward(
        self, x, step: int = 0, return_weights: bool = False
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Forward pass.

        Returns:
            x: (B, T, D) output
            routing_w: None (v2 has no routing)
            attn_w: attention weights or None
        """
        attn_w = None

        # Block 1: Attention or Conv
        if self.use_attention:
            h, attn_w = self.attn(self.ln1(x), return_weights=return_weights)
            x = x + h
        else:
            x = x + self.conv(self.ln1(x))

        # Block 2: PEER or MLP
        x = x + self.ffn(self.ln2(x))

        return x, None, attn_w


# ============================================================
# Unified Transformer (supports both v1 and v2)
# ============================================================

class HRSTransformer(nn.Module):
    """Unified model supporting v1 (routed) and v2 (attention->conv + PEER) architectures."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        self._is_v2 = cfg.is_v2()
        self._uses_replacement = cfg.uses_engram_replacement()
        self._uses_remember_gate = cfg.uses_remember_gate()
        self.use_memory_mlp = cfg.uses_memory_mlp()
        self._uses_bdh = cfg.uses_bdh()

        # If engrams are enabled and NOT using replacement, blocks after extraction
        # need a longer max_seq_len for prepended engrams.
        # v5 replacement keeps seq_len constant — no augmented config needed.
        if cfg.uses_engrams() and not self._uses_replacement:
            max_engrams = (model_cfg.max_seq_len // cfg.engram.window_size) * cfg.engram.n_engrams
            self._max_total_seq = model_cfg.max_seq_len + max_engrams
            from dataclasses import replace
            augmented_model_cfg = replace(model_cfg, max_seq_len=self._max_total_seq)
            augmented_cfg = replace(cfg, model=augmented_model_cfg)
        else:
            self._max_total_seq = model_cfg.max_seq_len
            augmented_cfg = cfg

        # Token embedding
        self.tok_emb = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.drop = nn.Dropout(model_cfg.dropout)

        # Build blocks (v1 or v2)
        BlockClass = HRSv2Block if self._is_v2 else HRSBlock
        blocks = []
        for i in range(model_cfg.n_layers):
            if cfg.uses_engrams() and not self._uses_replacement and i > cfg.engram.extract_layer:
                blocks.append(BlockClass(augmented_cfg, layer_idx=i))
            else:
                blocks.append(BlockClass(cfg, layer_idx=i))
        self.blocks = nn.ModuleList(blocks)

        # Final layer norm
        self.ln_f = nn.LayerNorm(model_cfg.d_model)

        # Head A: Generation (next-token prediction)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Head B: Locality projection (for contrastive loss)
        self.use_locality = cfg.locality.enabled
        if self.use_locality:
            self.locality_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)

        # Engram encoder
        self.use_engrams = cfg.uses_engrams()
        if self.use_engrams:
            self.engram_encoder = EngramEncoder(model_cfg, cfg.engram)
            self.engram_extract_layer = cfg.engram.extract_layer

            if self._uses_replacement:
                # v5: in-place blend instead of prepend
                self.engram_replacer = EngramReplacer(model_cfg, cfg.engram)
                self.register_buffer(
                    'per_token_loss_cache',
                    torch.zeros(cfg.training.batch_size, model_cfg.max_seq_len),
                )
            elif self._uses_remember_gate:
                # v6: learned gate before prepend
                self.engram_injector = GatedEngramInjector(model_cfg, cfg.engram)
                self.register_buffer(
                    'per_token_loss_cache',
                    torch.zeros(cfg.training.batch_size, model_cfg.max_seq_len),
                )
            else:
                # v1-v4: prepend engrams
                self.engram_injector = EngramInjector(model_cfg)

        # v7: Memory MLP + V7Router
        if self.use_memory_mlp:
            from memory_mlp_experiment.config import MemoryMLPConfig as InfMemCfg
            from memory_mlp_experiment.model.memory_mlp import MemoryMLP as InferenceMemoryMLP
            from memory_mlp_experiment.model.v7_router import V7Router

            mem_cfg = cfg.memory_mlp
            inf_cfg = InfMemCfg(
                d_input=cfg.model.d_model,
                d_hidden=mem_cfg.d_hidden,
                vocab_size=cfg.model.vocab_size,
                lr=mem_cfg.lr,
                train_steps_per_window=mem_cfg.train_steps_per_batch,
                replay_batch_size=mem_cfg.replay_batch_size,
                replay_buffer_max=mem_cfg.replay_buffer_max,
                expansion_check_interval=mem_cfg.expansion_check_interval,
                expansion_sim_threshold=mem_cfg.expansion_sim_threshold,
                max_hidden=mem_cfg.max_hidden,
            )
            self.memory_mlp = InferenceMemoryMLP(inf_cfg)

            self.v7_router = V7Router(
                d_model=cfg.model.d_model,
                d_hidden=mem_cfg.router_d_hidden,
                n_sources=2,  # base + memory (no KNN during training)
                init_base_bias=mem_cfg.router_init_base_bias,
            )

            self.loss_gate_theta = mem_cfg.loss_gate_theta
            self._loss_gate_calibrated = (mem_cfg.loss_gate_theta > 0)
            self._loss_accumulator = []

        self._init_weights()

        # v6: re-apply gate bias init after global _init_weights zeroes all biases
        if self.use_engrams and self._uses_remember_gate:
            self.engram_injector.remember_gate._init_weights(cfg.engram.gate_bias_init)

        # v7: re-apply V7Router base bias after global _init_weights
        if self.use_memory_mlp:
            with torch.no_grad():
                nn.init.zeros_(self.v7_router.fc2.weight)
                nn.init.zeros_(self.v7_router.fc2.bias)
                self.v7_router.fc2.bias[0] = cfg.memory_mlp.router_init_base_bias

    def _init_weights(self):
        # Collect modules to skip (Memory MLP has its own init)
        skip_modules = set()
        if self.use_memory_mlp:
            for m in self.memory_mlp.modules():
                skip_modules.add(id(m))

        for module in self.modules():
            if id(module) in skip_modules:
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # GPT-2 style residual scaling
        n_residuals = 2 * self.cfg.model.n_layers
        scale = 0.02 / math.sqrt(n_residuals)

        for block in self.blocks:
            if isinstance(block, HRSBlock):
                nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scale)
                if block.use_router:
                    if hasattr(block.attn_tier, 'out_proj'):
                        nn.init.normal_(block.attn_tier.out_proj.weight, mean=0.0, std=scale)
                    # v4: scale unconditional PEER FFN output projection
                    if block.use_peer_ffn and hasattr(block.peer_ffn, 'output_proj'):
                        nn.init.normal_(block.peer_ffn.output_proj.weight, mean=0.0, std=scale)
                    # v3: scale PEER output projection for residual stability
                    elif block.use_peer and hasattr(block, 'expert_tier') and hasattr(block.expert_tier, 'output_proj'):
                        nn.init.normal_(block.expert_tier.output_proj.weight, mean=0.0, std=scale)
                else:
                    nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=scale)
            elif isinstance(block, HRSv2Block):
                if block.use_attention:
                    nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scale)
                else:
                    nn.init.normal_(block.conv.out_proj.weight, mean=0.0, std=scale)
                if not block.use_peer:
                    nn.init.normal_(block.ffn.fc2.weight, mean=0.0, std=scale)
                else:
                    nn.init.normal_(block.ffn.output_proj.weight, mean=0.0, std=scale)

    def forward(
        self,
        idx: torch.Tensor,
        step: int = 0,
        collect_layer_reps: bool = False,
        collect_intermediates: bool = False,
        targets: torch.Tensor = None,
    ) -> HRSOutput:
        x = self.drop(self.tok_emb(idx))
        B, T, D = x.shape

        layer_reps = []
        routing_weights_list = []
        attn_weights_list = []
        peer_indices_list = []
        balance_loss_total = torch.tensor(0.0, device=x.device)
        entropy_loss_total = torch.tensor(0.0, device=x.device)
        flops_loss_total = torch.tensor(0.0, device=x.device)
        engrams = torch.zeros(B, 0, D, device=x.device)
        engram_recon_loss = torch.tensor(0.0, device=x.device)
        replacement_gates = None
        remember_gate_vals = None

        store_reps = collect_intermediates or collect_layer_reps

        # v5/v6: get cached loss for gating (from previous step)
        if self._uses_replacement or self._uses_remember_gate:
            cached_loss = self.per_token_loss_cache[:B, :T]
            # If cache is all zeros (first step), signal None
            if cached_loss.sum() == 0:
                cached_loss = None

        for i, block in enumerate(self.blocks):
            if self.use_engrams and i > self.engram_extract_layer and engrams.shape[1] > 0:
                if self._uses_replacement:
                    # v5: blend engrams in-place (seq_len unchanged)
                    x, replacement_gates = self.engram_replacer(x, engrams, cached_loss)
                elif self._uses_remember_gate:
                    # v6: gated prepend
                    x, n_engrams, remember_gate_vals = self.engram_injector(x, engrams, cached_loss)
                else:
                    # v1-v4: prepend engrams
                    x, n_engrams = self.engram_injector(x, engrams)
            else:
                n_engrams = 0

            # v8 BDH: pass engrams to block for virtual synapse focus
            block_engrams = engrams if (self._uses_bdh and i > self.engram_extract_layer and engrams.shape[1] > 0) else None

            x, routing_w, attn_w = block(
                x, step=step, return_weights=collect_intermediates,
                engrams=block_engrams,
            )

            # Strip prepended engrams from output (v1-v4 only)
            if not self._uses_replacement and n_engrams > 0:
                x = x[:, n_engrams:]

            # Store layer representations
            if store_reps:
                if self.use_locality:
                    layer_reps.append(self.locality_proj(x))
                else:
                    layer_reps.append(x)

            # Routing losses
            if routing_w is not None:
                routing_weights_list.append(routing_w)
                # v8 BDH: use Zipf hub loss instead of uniform balance loss
                if self._uses_bdh and self.cfg.bdh.hub_routing_enabled:
                    balance_loss_total = balance_loss_total + routing_hub_loss(
                        routing_w, self.cfg.bdh.hub_exponent)
                else:
                    balance_loss_total = balance_loss_total + routing_balance_loss(routing_w)
                entropy_loss_total = entropy_loss_total + routing_entropy_loss(routing_w)
                flops_loss_total = flops_loss_total + routing_flops_loss(routing_w)

            if collect_intermediates and attn_w is not None:
                attn_weights_list.append(attn_w)

            # Engram extraction
            if self.use_engrams and i == self.engram_extract_layer:
                engrams = self.engram_encoder(x)
                engram_recon_loss = engram_reconstruction_loss(
                    x.detach(), engrams, self.cfg.engram.window_size,
                )

        # Average losses across layers
        if routing_weights_list:
            n_routed = len(routing_weights_list)
            balance_loss_total = balance_loss_total / n_routed
            entropy_loss_total = entropy_loss_total / n_routed
            flops_loss_total = flops_loss_total / n_routed

        x = self.ln_f(x)
        hidden_for_output = x  # save for v7 Memory MLP
        logits = self.lm_head(x)

        # v5/v6: update EMA loss cache for next step's gating signal
        per_token_loss = None
        if (self._uses_replacement or self._uses_remember_gate) and targets is not None and self.training:
            with torch.no_grad():
                V = logits.shape[-1]
                ptl = F.cross_entropy(
                    logits.detach().reshape(B * T, V),
                    targets.reshape(B * T),
                    reduction='none',
                ).reshape(B, T)
                decay = self.cfg.engram.loss_ema_decay
                self.per_token_loss_cache[:B, :T] = (
                    decay * self.per_token_loss_cache[:B, :T] + (1.0 - decay) * ptl
                )
                per_token_loss = ptl

        # v7: Memory MLP + V7Router
        memory_logits = None
        v7_router_weights = None
        if self.use_memory_mlp:
            hs = hidden_for_output.detach().float()  # (B, T, D) — detached
            memory_logits = self.memory_mlp(hs)  # (B, T, V)

            # Router features
            with torch.no_grad():
                base_entropy = -(F.softmax(logits.float(), -1) * F.log_softmax(logits.float(), -1)).sum(-1, keepdim=True)
                mem_confidence = F.softmax(memory_logits.float(), -1).max(-1, keepdim=True).values

            # V7Router: 2-source (base + memory), no KNN during training
            # Reshape for router: (B*T, D+2)
            B_cur, T_cur = hs.shape[0], hs.shape[1]
            v7_router_weights = self.v7_router(
                hs.reshape(B_cur * T_cur, -1),
                base_entropy.reshape(B_cur * T_cur, 1),
                mem_confidence.reshape(B_cur * T_cur, 1),
            ).reshape(B_cur, T_cur, 2)  # (B, T, 2)

        # v8 BDH: compute diagnostic metrics
        bdh_focus_mag = None
        bdh_sparsity = None
        bdh_hub_dist = None
        if self._uses_bdh and routing_weights_list:
            # Hub distribution: sorted tier utilization from last layer
            with torch.no_grad():
                last_rw = routing_weights_list[-1]
                p = last_rw.float().mean(dim=(0, 1))
                p_sorted, _ = p.sort(descending=True)
                bdh_hub_dist = p_sorted.tolist()

            # Sparsity level: fraction of zeros after bottleneck
            # (computed as 1 - rho since bottleneck keeps exactly rho fraction)
            if self.cfg.bdh.sparsity_enabled:
                bdh_sparsity = 1.0 - self.cfg.bdh.sparsity_rho

            # Focus magnitude: alpha * mean|gain| across virtual synapses
            if self.cfg.bdh.virtual_synapse_enabled and engrams.shape[1] > 0:
                gains = []
                for block in self.blocks:
                    if hasattr(block, 'virtual_synapse'):
                        fq, fk = block.virtual_synapse(engrams.detach())
                        if fq is not None:
                            gain = (fq * fk).sum(dim=-1).abs().mean().item()
                            gains.append(gain * self.cfg.bdh.focus_alpha)
                if gains:
                    bdh_focus_mag = sum(gains) / len(gains)

        return HRSOutput(
            logits=logits,
            layer_representations=layer_reps,
            routing_weights=routing_weights_list,
            routing_balance_loss=balance_loss_total,
            routing_entropy_loss=entropy_loss_total,
            routing_flops_loss=flops_loss_total,
            engrams=engrams,
            engram_recon_loss=engram_recon_loss,
            attention_weights=attn_weights_list,
            peer_indices=peer_indices_list,
            per_token_loss=per_token_loss,
            replacement_gates=replacement_gates,
            remember_gates=remember_gate_vals,
            hidden_states=hidden_for_output if self.use_memory_mlp else None,
            memory_logits=memory_logits,
            v7_router_weights=v7_router_weights,
            bdh_focus_magnitude=bdh_focus_mag,
            bdh_sparsity_level=bdh_sparsity,
            bdh_hub_distribution=bdh_hub_dist,
        )

    def apply_engram_refinement(self):
        """Phase 5 engram refinement: freeze encoder, reinitialize injector/replacer."""
        if not self.use_engrams:
            return

        frozen_count = 0
        for param in self.engram_encoder.parameters():
            param.requires_grad_(False)
            frozen_count += param.numel()

        if self._uses_replacement:
            # v5: reinitialize gate parameters for fine-tuning
            nn.init.constant_(self.engram_replacer.threshold, self.cfg.engram.gate_init_threshold)
            nn.init.constant_(self.engram_replacer.sharpness, self.cfg.engram.gate_sharpness_init)
            nn.init.normal_(self.engram_replacer.replace_type_emb, std=0.02)
            print(f"  Engram refinement: froze {frozen_count:,} encoder params, "
                  f"reinitialized replacer gate params")
        elif self._uses_remember_gate:
            # v6: reinit gate bias and type embedding
            self.engram_injector.remember_gate._init_weights(self.cfg.engram.gate_bias_init)
            nn.init.normal_(self.engram_injector.engram_type_emb, std=0.02)
            print(f"  Engram refinement: froze {frozen_count:,} encoder params, "
                  f"reinitialized remember gate + type embedding")
        else:
            nn.init.normal_(self.engram_injector.engram_type_emb, std=0.02)
            print(f"  Engram refinement: froze {frozen_count:,} encoder params, "
                  f"reinitialized injector type embedding")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def component_param_counts(self) -> dict:
        """Count parameters by component group for logging."""
        counts = {}

        # Backbone (embedding + attn/conv blocks + LN + final LN)
        backbone = sum(p.numel() for p in self.tok_emb.parameters())
        backbone += sum(p.numel() for p in self.ln_f.parameters())
        for block in self.blocks:
            backbone += sum(p.numel() for p in block.ln1.parameters())
            backbone += sum(p.numel() for p in block.ln2.parameters())
            if isinstance(block, HRSv2Block):
                if block.use_attention:
                    backbone += sum(p.numel() for p in block.attn.parameters())
                else:
                    backbone += sum(p.numel() for p in block.conv.parameters())
            elif isinstance(block, HRSBlock):
                backbone += sum(p.numel() for p in block.attn.parameters())
                if not block.use_router:
                    backbone += sum(p.numel() for p in block.mlp.parameters())
        counts["backbone"] = backbone

        # Generation head (weight-tied, so 0 additional)
        counts["gen_head"] = 0

        # Locality projection
        if self.use_locality:
            counts["locality_head"] = sum(p.numel() for p in self.locality_proj.parameters())

        # PEER FFN (v2)
        peer_params = 0
        for block in self.blocks:
            if isinstance(block, HRSv2Block) and block.use_peer:
                peer_params += sum(p.numel() for p in block.ffn.parameters())
        if peer_params > 0:
            counts["peer"] = peer_params

        # MLP FFN (v2 non-PEER or v1 non-routed counted in backbone above)
        mlp_params = 0
        for block in self.blocks:
            if isinstance(block, HRSv2Block) and not block.use_peer:
                mlp_params += sum(p.numel() for p in block.ffn.parameters())
        if mlp_params > 0:
            counts["mlp"] = mlp_params

        # v1 Router
        router_params = 0
        for block in self.blocks:
            if hasattr(block, "router"):
                router_params += sum(p.numel() for p in block.router.parameters())
        if router_params > 0:
            counts["router"] = router_params

        # v1/v3 Tiers
        for tier_name in ["conv_tier", "attn_tier", "sink_tier"]:
            total = 0
            for block in self.blocks:
                if hasattr(block, tier_name):
                    total += sum(p.numel() for p in getattr(block, tier_name).parameters())
            if total > 0:
                counts[tier_name] = total

        # v4: unconditional PEER FFN (separate from tiers)
        peer_ffn_total = 0
        for block in self.blocks:
            if hasattr(block, "peer_ffn"):
                peer_ffn_total += sum(p.numel() for p in block.peer_ffn.parameters())
                if hasattr(block, "ln_peer"):
                    peer_ffn_total += sum(p.numel() for p in block.ln_peer.parameters())
                if hasattr(block, "peer_output_gate"):
                    peer_ffn_total += block.peer_output_gate.numel()
        if peer_ffn_total > 0:
            counts["peer"] = peer_ffn_total

        # expert_tier: count as "peer" when PEER (v3), else "expert_tier"
        expert_total = 0
        expert_is_peer = False
        for block in self.blocks:
            if hasattr(block, "expert_tier"):
                expert_total += sum(p.numel() for p in block.expert_tier.parameters())
                if isinstance(block, HRSBlock) and block.use_peer:
                    expert_is_peer = True
        if expert_total > 0:
            counts["peer" if expert_is_peer else "expert_tier"] = expert_total

        # Engrams
        if self.use_engrams:
            counts["engram_encoder"] = sum(p.numel() for p in self.engram_encoder.parameters())
            if self._uses_replacement:
                counts["engram_replacer"] = sum(p.numel() for p in self.engram_replacer.parameters())
            else:
                counts["engram_injector"] = sum(p.numel() for p in self.engram_injector.parameters())
                if self._uses_remember_gate and hasattr(self.engram_injector, 'remember_gate'):
                    counts["remember_gate"] = sum(p.numel() for p in self.engram_injector.remember_gate.parameters())

        # v7: Memory MLP + V7Router
        if self.use_memory_mlp:
            counts["memory_mlp"] = sum(p.numel() for p in self.memory_mlp.parameters())
            counts["v7_router"] = sum(p.numel() for p in self.v7_router.parameters())

        # v8 BDH: Virtual Synapse params
        if self._uses_bdh:
            vs_params = 0
            for block in self.blocks:
                if hasattr(block, 'virtual_synapse'):
                    vs_params += sum(p.numel() for p in block.virtual_synapse.parameters())
            if vs_params > 0:
                counts["virtual_synapse"] = vs_params

        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts

    def get_param_groups(self) -> dict:
        """Return named parameter groups for phased training.

        v1 groups: backbone, gen_head, locality_head, router, conv, expert, attention_tier, sink, engram
        v2 groups: backbone, gen_head, locality_head, peer, engram
        """
        if self._is_v2:
            return self._get_v2_param_groups()
        return self._get_v1_param_groups()

    def _get_v1_param_groups(self) -> dict:
        groups = {
            "backbone": [], "gen_head": [], "locality_head": [],
            "router": [], "conv": [], "expert": [], "attention_tier": [],
            "sink": [], "engram": [], "v7_router": [],
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # v7: Memory MLP params are trained separately via SGD, skip them
            if "memory_mlp" in name:
                continue
            if "v7_router" in name:
                groups["v7_router"].append(param)
            elif "locality_proj" in name:
                groups["locality_head"].append(param)
            elif "engram" in name:
                groups["engram"].append(param)
            elif "router" in name:
                groups["router"].append(param)
            elif "conv_tier" in name:
                groups["conv"].append(param)
            elif "expert_tier" in name:
                groups["expert"].append(param)
            elif "peer_ffn" in name or "ln_peer" in name or "peer_output_gate" in name:
                # v4/v8: unconditional PEER FFN uses "expert" LR schedule slot
                groups["expert"].append(param)
            elif "virtual_synapse" in name:
                # v8 BDH: virtual synapse uses "expert" LR schedule slot
                groups["expert"].append(param)
            elif "attn_tier" in name:
                groups["attention_tier"].append(param)
            elif "sink_tier" in name or "sink_kv_scale" in name:
                groups["sink"].append(param)
            elif "lm_head" in name:
                groups["gen_head"].append(param)
            else:
                groups["backbone"].append(param)

        return groups

    def _get_v2_param_groups(self) -> dict:
        groups = {
            "backbone": [], "gen_head": [], "locality_head": [],
            "peer": [], "engram": [],
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "locality_proj" in name:
                groups["locality_head"].append(param)
            elif "engram" in name:
                groups["engram"].append(param)
            elif "lm_head" in name:
                groups["gen_head"].append(param)
            elif ".ffn." in name and any(
                isinstance(block, HRSv2Block) and block.use_peer
                for block in self.blocks
            ):
                # PEER FFN parameters (keys, expert weights, projections)
                groups["peer"].append(param)
            else:
                groups["backbone"].append(param)

        return groups
