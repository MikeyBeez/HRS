"""HRS model — supports both v1 (routed) and v2 (attention->conv + PEER) architectures.

v1: Backbone transformer with router + tiered compute (conv, expert, attn, sink)
v2: Fixed attention->conv backbone with PEER FFN replacing standard MLP

Shared components: dual heads, engram encoder/injector, RoPE.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig, ModelConfig, PEERConfig
from tiers import ConvTier, ExpertTier, AttentionTier, SinkTier, RotaryEmbedding, apply_rotary_emb, rotate_half
from router import TokenRouter, routing_balance_loss, routing_entropy_loss, routing_flops_loss
from engram import EngramEncoder, EngramInjector, engram_reconstruction_loss
from peer import PEER


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

    def forward(self, x, return_weights=False):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

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
    """v1 HRS transformer block.

    For dense_baseline/dual_head: standard pre-norm transformer block.
    For routed configs: adds router + tiered compute after attention.
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

        # Router + tiered compute REPLACES MLP
        self.use_router = cfg.uses_router()
        self.use_peer = cfg.uses_peer()
        # v4: PEER as unconditional FFN + 3-tier routing (no expert tier)
        self.use_peer_ffn = self.use_peer and cfg.router.n_tiers == 3
        if self.use_router:
            self.router = TokenRouter(model_cfg, cfg.router)

            self.conv_tier = ConvTier(model_cfg, cfg.tier)
            if self.use_peer_ffn:
                # v4: PEER runs unconditionally, not as a routed tier
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
        self, x, step: int = 0, return_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(self.ln1(x), return_weights=return_weights)
        x = x + attn_out

        routing_w = None

        if self.use_router:
            h = self.ln2(x)
            routing_w = self.router(h, step=step)

            if self.use_peer_ffn:
                # v4: 3-tier routing (conv, attn, sink) + unconditional PEER
                conv_out = self.conv_tier(h)
                attn_out2 = self.attn_tier(h)
                sink_out = self.sink_tier(h)

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
                conv_out = self.conv_tier(h)
                expert_out = self.expert_tier(h)
                attn_out2 = self.attn_tier(h)
                sink_out = self.sink_tier(h)

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

        # If engrams are enabled, blocks after extraction need a longer max_seq_len
        if cfg.uses_engrams():
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
            if cfg.uses_engrams() and i > cfg.engram.extract_layer:
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
            self.engram_injector = EngramInjector(model_cfg)
            self.engram_extract_layer = cfg.engram.extract_layer

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
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
    ) -> HRSOutput:
        x = self.drop(self.tok_emb(idx))

        layer_reps = []
        routing_weights_list = []
        attn_weights_list = []
        peer_indices_list = []
        balance_loss_total = torch.tensor(0.0, device=x.device)
        entropy_loss_total = torch.tensor(0.0, device=x.device)
        flops_loss_total = torch.tensor(0.0, device=x.device)
        engrams = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)
        engram_recon_loss = torch.tensor(0.0, device=x.device)

        store_reps = collect_intermediates or collect_layer_reps

        for i, block in enumerate(self.blocks):
            # Engram injection: prepend engrams before processing (for layers after extraction)
            if self.use_engrams and i > self.engram_extract_layer and engrams.shape[1] > 0:
                x, n_engrams = self.engram_injector(x, engrams)
            else:
                n_engrams = 0

            x, routing_w, attn_w = block(
                x, step=step, return_weights=collect_intermediates,
            )

            # Strip prepended engrams from output
            if n_engrams > 0:
                x = x[:, n_engrams:]

            # Store layer representations
            if store_reps:
                if self.use_locality:
                    layer_reps.append(self.locality_proj(x))
                else:
                    layer_reps.append(x)

            # v1 routing losses
            if routing_w is not None:
                routing_weights_list.append(routing_w)
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
        logits = self.lm_head(x)

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
        )

    def apply_engram_refinement(self):
        """Phase 5 engram refinement (v1): freeze encoder, reinitialize injector."""
        if not self.use_engrams:
            return

        frozen_count = 0
        for param in self.engram_encoder.parameters():
            param.requires_grad_(False)
            frozen_count += param.numel()

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
            counts["engram_injector"] = sum(p.numel() for p in self.engram_injector.parameters())

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
            "sink": [], "engram": [],
        }

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "locality_proj" in name:
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
                # v4: unconditional PEER FFN uses "expert" LR schedule slot
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
