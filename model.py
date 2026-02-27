"""Hierarchical Routed Sinkformer (HRS) model.

Integrates all HRS components:
- Backbone transformer with RoPE
- Dual heads (generation + locality)
- Learned router with Sinkhorn normalization
- Tiered compute (conv, expert, attention, sink)
- Engram encoder for long-range compression

The model adapts to different ablation configurations, disabling
components that aren't needed for a given experiment.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig, ModelConfig, AblationConfig
from tiers import ConvTier, ExpertTier, AttentionTier, SinkTier, RotaryEmbedding, apply_rotary_emb, rotate_half
from router import TokenRouter, routing_balance_loss, routing_entropy_loss, routing_flops_loss
from engram import EngramEncoder, EngramInjector, engram_reconstruction_loss


@dataclass
class HRSOutput:
    logits: torch.Tensor
    layer_representations: list       # per-layer hidden states for locality loss
    routing_weights: list             # per-layer routing decisions
    routing_balance_loss: torch.Tensor  # aggregate balance loss across layers
    routing_entropy_loss: torch.Tensor  # negative per-token entropy (exploration)
    routing_flops_loss: torch.Tensor  # expected FLOPs cost of routing decisions
    engrams: torch.Tensor             # engram vectors (if applicable)
    engram_recon_loss: torch.Tensor   # engram reconstruction loss
    attention_weights: list           # for evaluation metrics (optional)


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


class HRSBlock(nn.Module):
    """Single HRS transformer block.

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
        if self.use_router:
            self.router = TokenRouter(model_cfg, cfg.router)

            # Tiers replace the MLP — same residual slot
            self.conv_tier = ConvTier(model_cfg, cfg.tier)
            self.expert_tier = ExpertTier(model_cfg, cfg.tier)
            self.attn_tier = AttentionTier(model_cfg)
            self.sink_tier = SinkTier(cfg.tier)

            # Output gate: scales down tier outputs for stable residual addition.
            # Without this, conv/expert LayerNorm produces ~1.0 scale outputs,
            # while baseline MLP fc2 is scaled to ~0.007. This mismatch causes
            # the residual stream to explode and effective rank to collapse.
            # Initialized to small value (like GPT-2 residual scaling).
            self.tier_output_gate = nn.Parameter(
                torch.full((model_cfg.d_model,), 0.1)
            )

            # Sink KV scaling (for sink channel effect on attention)
            self.use_sink = cfg.uses_sink()
            if self.use_sink:
                self.sink_kv_scale = nn.Parameter(
                    torch.tensor(cfg.tier.sink_init_scale)
                )
        else:
            # Standard MLP when no routing
            self.mlp = MLP(model_cfg)

    def forward(
        self, x, step: int = 0, return_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through HRS block.

        When routing is enabled, tiers REPLACE the MLP (not add on top).
        This keeps total compute comparable to baseline for most tokens,
        and cheaper for conv/sink-routed tokens.

        Returns:
            x: (B, T, D) output hidden states
            routing_w: (B, T, n_tiers) or None
            attn_w: (B, H, T, T) or None
        """
        # Attention (always)
        attn_out, attn_w = self.attn(self.ln1(x), return_weights=return_weights)
        x = x + attn_out

        routing_w = None

        if self.use_router:
            # Router decides how to process each token (replaces MLP)
            h = self.ln2(x)
            routing_w = self.router(h, step=step)  # (B, T, n_tiers)

            # Apply each tier to the pre-norm input
            conv_out = self.conv_tier(h)      # (B, T, D) — O(n)
            expert_out = self.expert_tier(h)  # (B, T, D) — O(n)
            attn_out2 = self.attn_tier(h)     # (B, T, D) — O(n^2)
            sink_out = self.sink_tier(h)      # (B, T, D) — O(1)

            # Weighted combination: out = sum(w_i * tier_i(x))
            tier_outputs = torch.stack(
                [conv_out, expert_out, attn_out2, sink_out], dim=-1
            )  # (B, T, D, 4)
            w = routing_w.unsqueeze(2)  # (B, T, 1, 4)
            combined = (tier_outputs * w).sum(dim=-1)  # (B, T, D)

            # Scale tier output for stable residual addition
            x = x + combined * self.tier_output_gate

            # Sink KV scaling: scale down KV entries for sink-routed tokens
            if self.use_sink:
                sink_weight = routing_w[:, :, 3]  # (B, T) - sink tier is index 3
                kv_scale = 1.0 - sink_weight * (1.0 - self.sink_kv_scale.abs())
                x = x * kv_scale.unsqueeze(-1)
        else:
            # Standard MLP path (baseline / dual_head)
            x = x + self.mlp(self.ln2(x))

        return x, routing_w, attn_w


class HRSTransformer(nn.Module):
    """Hierarchical Routed Sinkformer.

    Supports all ablation configurations from dense baseline to full HRS.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        # If engrams are enabled, blocks after extraction need a longer max_seq_len
        # to accommodate prepended engrams in the RoPE cache
        if cfg.uses_engrams():
            max_engrams = (model_cfg.max_seq_len // cfg.engram.window_size) * cfg.engram.n_engrams
            self._max_total_seq = model_cfg.max_seq_len + max_engrams
            # Create an adjusted model config for blocks that see engrams
            from dataclasses import replace
            augmented_cfg = replace(cfg, model=replace(model_cfg, max_seq_len=self._max_total_seq))
        else:
            self._max_total_seq = model_cfg.max_seq_len
            augmented_cfg = cfg

        # Token embedding
        self.tok_emb = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.drop = nn.Dropout(model_cfg.dropout)

        # Transformer blocks — blocks after engram extraction use augmented max_seq_len
        blocks = []
        for i in range(model_cfg.n_layers):
            if cfg.uses_engrams() and i > cfg.engram.extract_layer:
                blocks.append(HRSBlock(augmented_cfg, layer_idx=i))
            else:
                blocks.append(HRSBlock(cfg, layer_idx=i))
        self.blocks = nn.ModuleList(blocks)

        # Final layer norm
        self.ln_f = nn.LayerNorm(model_cfg.d_model)

        # Head A: Generation (next-token prediction)
        self.lm_head = nn.Linear(model_cfg.vocab_size, model_cfg.d_model, bias=False)
        # Weight tying
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
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

        # GPT-2 style residual scaling: 2 residual connections per layer (attn + ffn/tiers)
        n_residuals = 2 * self.cfg.model.n_layers
        scale = 0.02 / math.sqrt(n_residuals)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scale)
            if block.use_router:
                # Scale down tier output projections
                if hasattr(block.attn_tier, 'out_proj'):
                    nn.init.normal_(block.attn_tier.out_proj.weight, mean=0.0, std=scale)
            else:
                nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=scale)

    def forward(
        self,
        idx: torch.Tensor,
        step: int = 0,
        collect_layer_reps: bool = False,
        collect_intermediates: bool = False,
    ) -> HRSOutput:
        """Forward pass.

        Args:
            idx: (B, T) token indices
            step: training step (for router temperature annealing)
            collect_layer_reps: store per-layer reps for locality loss
            collect_intermediates: store attention weights for evaluation
        """
        x = self.drop(self.tok_emb(idx))

        layer_reps = []
        routing_weights_list = []
        attn_weights_list = []
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

            # Store layer representations (locality loss uses projected versions)
            if store_reps:
                if self.use_locality:
                    layer_reps.append(self.locality_proj(x))
                else:
                    layer_reps.append(x)

            if routing_w is not None:
                routing_weights_list.append(routing_w)
                balance_loss_total = balance_loss_total + routing_balance_loss(routing_w)
                entropy_loss_total = entropy_loss_total + routing_entropy_loss(routing_w)
                flops_loss_total = flops_loss_total + routing_flops_loss(routing_w)

            if collect_intermediates and attn_w is not None:
                attn_weights_list.append(attn_w)

            # Engram extraction: encode hidden states at the extraction layer
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
        )

    def apply_engram_refinement(self):
        """Phase 5 engram refinement: freeze encoder, reinitialize injector.

        Freezes engram_encoder weights (requires_grad=False) so they produce
        fixed compressed representations. Reinitializes engram_injector type
        embedding so downstream consumers learn fresh connections to the
        frozen engrams.
        """
        if not self.use_engrams:
            return

        # Freeze encoder
        frozen_count = 0
        for param in self.engram_encoder.parameters():
            param.requires_grad_(False)
            frozen_count += param.numel()

        # Reinitialize injector type embedding
        nn.init.normal_(self.engram_injector.engram_type_emb, std=0.02)

        print(f"  Engram refinement: froze {frozen_count:,} encoder params, "
              f"reinitialized injector type embedding")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def component_param_counts(self) -> dict:
        """Count parameters by component group for logging."""
        counts = {}

        # Backbone (embedding + attn + MLP for non-routed blocks + LN + final LN)
        backbone = sum(p.numel() for p in self.tok_emb.parameters())
        backbone += sum(p.numel() for p in self.ln_f.parameters())
        for block in self.blocks:
            backbone += sum(p.numel() for p in block.attn.parameters())
            backbone += sum(p.numel() for p in block.ln1.parameters())
            backbone += sum(p.numel() for p in block.ln2.parameters())
            if not block.use_router:
                backbone += sum(p.numel() for p in block.mlp.parameters())
        counts["backbone"] = backbone

        # Generation head (weight-tied, so 0 additional)
        counts["gen_head"] = 0

        # Locality projection
        if self.use_locality:
            counts["locality_head"] = sum(p.numel() for p in self.locality_proj.parameters())

        # Router
        router_params = 0
        for block in self.blocks:
            if hasattr(block, "router"):
                router_params += sum(p.numel() for p in block.router.parameters())
        counts["router"] = router_params

        # Tiers
        for tier_name in ["conv_tier", "expert_tier", "attn_tier", "sink_tier"]:
            total = 0
            for block in self.blocks:
                if hasattr(block, tier_name):
                    total += sum(p.numel() for p in getattr(block, tier_name).parameters())
            if total > 0:
                counts[tier_name] = total

        # Engrams
        if self.use_engrams:
            counts["engram_encoder"] = sum(p.numel() for p in self.engram_encoder.parameters())
            counts["engram_injector"] = sum(p.numel() for p in self.engram_injector.parameters())

        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts

    def get_param_groups(self) -> dict:
        """Return named parameter groups for phased training.

        Groups: backbone, gen_head, locality_head, router, conv, expert, attention, sink, engram
        """
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
            elif "attn_tier" in name:
                groups["attention_tier"].append(param)
            elif "sink_tier" in name or "sink_kv_scale" in name:
                groups["sink"].append(param)
            elif "lm_head" in name:
                groups["gen_head"].append(param)
            else:
                groups["backbone"].append(param)

        return groups
