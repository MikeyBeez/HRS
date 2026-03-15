"""Configuration for HRS v1, v2, v7, and v8 experiments.

v1: Hierarchical Routed Sinkformer (routing + tiered compute)
v2: Attention->Conv backbone + PEER FFN + Engrams (no routing)
v7: v4 base + Memory MLP + V7Router (logit blending, from-scratch training)
v8: v4 base + BDH-inspired upgrades (virtual synapse, hub routing, sparsity bottleneck)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class AblationConfig(Enum):
    # --- v1 configs (preserved for reproducibility) ---
    DENSE_BASELINE = "dense_baseline"           # Standard transformer, CE only
    DUAL_HEAD = "dual_head"                     # + locality head (dual objective)
    DUAL_HEAD_ROUTER = "dual_head_router"       # + learned router + tiered compute
    DUAL_HEAD_ROUTER_SINK = "dual_head_router_sink"  # + sink channel
    FULL_CORE = "full_core"                     # All core + phased training
    FULL_HRS = "full_hrs"                       # Core + engrams (all 5 phases)
    FULL_HRS_REFINED = "full_hrs_refined"       # Full HRS + engram refinement

    # --- v6 configs ---
    V4_1024 = "v4_1024"                        # config 16: v4 at seq_len=1024 (baseline)
    V6_GATE = "v6_gate"                        # config 17: v4 + learned remember gate + seq_len=1024

    # --- v5 configs ---
    V5_REPLACE = "v5_replace"                  # v4 + learnable engram-based context replacement (no prepend)

    # --- v4 configs ---
    V4_FULL = "v4_full"                        # PEER as universal FFN + 3-tier routing (conv, attn, sink) + engrams

    # --- v3 configs ---
    V3_FULL = "v3_full"                        # v1 routing + PEER as expert tier + engrams

    # --- v7 configs ---
    V7_FULL = "v7_full"                        # v4 base + Memory MLP + V7Router (logit blending)

    # --- v8 configs ---
    V8_BDH = "v8_bdh"                          # v4 base + BDH: virtual synapse, hub routing, sparsity bottleneck

    # --- v9 configs ---
    V9_LEARNABLE = "v9_learnable"              # v8 + learnable loss scaling for auxiliary objectives

    # --- v2 configs ---
    V2_ATTN_CONV = "v2_attn_conv"              # Attention->Conv backbone, standard MLP, no dual-head
    V2_ATTN_CONV_DUAL = "v2_attn_conv_dual"    # + dual-head
    V2_ATTN_CONV_PEER = "v2_attn_conv_peer"    # + PEER FFN (replaces MLP)
    V2_FULL = "v2_full"                        # + Engrams + phased training


@dataclass
class ModelConfig:
    vocab_size: int = 50257       # GPT-2 BPE tokenizer
    n_layers: int = 4
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False


@dataclass
class RouterConfig:
    n_tiers: int = 4              # conv, expert, attention, sink
    hidden_dim: int = 64          # router MLP hidden size
    gumbel_tau: float = 1.0       # Gumbel-softmax temperature (annealed during training)
    gumbel_tau_min: float = 0.3   # minimum temperature (0.3 keeps some softness)
    gumbel_anneal_steps: int = 40000  # steps to anneal from tau to tau_min (80% of training)
    balance_loss_weight: float = 0.1  # load-balancing auxiliary loss weight
    entropy_loss_weight: float = 0.01  # per-token entropy regularization (exploration)
    flops_loss_weight: float = 0.0    # FLOPs cost penalty (0 = off, try 0.1-1.0)
    trc_enabled: bool = False         # Temporal Routing Cache (low-pass filter)
    trc_window: int = 8               # causal moving average window size


@dataclass
class TierConfig:
    # Conv tier
    conv_kernel_size: int = 7
    # Expert tier (MoE)
    n_experts: int = 4
    expert_hidden: int = 256      # each expert: d_model -> expert_hidden -> d_model
    expert_top_k: int = 1         # top-k routing within expert tier
    # Sink tier
    sink_init_scale: float = 0.1  # initial scale-down factor for sink channel


@dataclass
class PEERConfig:
    """Configuration for Parameter Efficient Expert Retrieval."""
    enabled: bool = False
    n_sub_keys: int = 512         # 2 sets of 512 sub-keys -> 262,144 experts
    n_heads: int = 8              # number of retrieval heads
    top_k: int = 16               # experts per head -> 128 active per token


@dataclass
class EngramConfig:
    enabled: bool = False
    window_size: int = 128        # N tokens compressed into K engrams
    n_engrams: int = 4            # K engram vectors per window
    extract_layer: int = 1        # extract from this layer (0-indexed, layer 2 of 4 = index 1)
    engram_dim: int = 512         # engram vector dimension (= d_model)
    recon_loss_weight: float = 0.1  # weight for reconstruction loss
    # v5: in-place replacement instead of prepend
    replacement_enabled: bool = False   # blend instead of prepend
    gate_init_threshold: float = 3.0    # initial θ (high = keep originals early)
    gate_sharpness_init: float = 1.0    # initial α
    gate_entropy_weight: float = 0.01   # regularization weight
    loss_ema_decay: float = 0.99        # EMA decay for loss cache
    # v6: learned remember gate before prepend
    remember_gate_enabled: bool = False
    gate_bias_init: float = 2.0         # sigmoid(2.0) ≈ 0.88
    gate_hidden_dim: int = 32           # hidden dim in gate MLP


@dataclass
class MemoryMLPTrainConfig:
    """Configuration for Memory MLP + V7Router during from-scratch training."""
    enabled: bool = False
    d_hidden: int = 2048
    lr: float = 0.01                          # SGD learning rate for Memory MLP
    train_steps_per_batch: int = 3            # SGD steps per training batch
    replay_buffer_max: int = 50_000           # max (hidden, target) pairs
    replay_batch_size: int = 128              # replay samples per SGD step
    expansion_check_interval: int = 200       # check expansion every N train steps
    expansion_sim_threshold: float = 0.90     # cosine sim threshold for expansion
    max_hidden: int = 4096                    # max hidden units after expansion
    loss_gate_theta: float = 0.0              # 0 = auto-calibrate from first 100 steps
    loss_gate_target_flag_rate: float = 0.15  # target ~15% of tokens flagged
    router_d_hidden: int = 128                # V7Router hidden dim
    router_init_base_bias: float = 2.0        # softmax([2,0]) ~ [0.88, 0.12]
    router_entropy_weight: float = 0.01       # entropy regularization on router weights


@dataclass
class BDHConfig:
    """Configuration for BDH-inspired upgrades (v8)."""
    enabled: bool = False
    # Experiment 1: Virtual Synapse — engram-derived attention focus matrix
    virtual_synapse_enabled: bool = True
    focus_alpha: float = 0.1              # multiplicative gain scaling on attention scores
    focus_proj_dim: int = 64              # projection dim for rank-1 focus factors
    # Experiment 2: Scale-Free Hub Routing — Zipf distribution target
    hub_routing_enabled: bool = True
    hub_exponent: float = 1.5             # Zipf exponent: tier k target = k^(-exponent)
    # Experiment 3: Monosemantic Sparsity Bottleneck
    sparsity_enabled: bool = True
    sparsity_rho: float = 0.05            # fraction of features kept active (top-k)
    sparsity_activation: str = "softplus" # "relu" or "softplus" positivity constraint
    # v9: Learnable loss scaling
    learnable_loss_scaling: bool = False
    alive_penalty: float = 0.001          # coefficient for -log(scale) keep-alive penalty
    loss_scaler_lr_mult: float = 0.1      # LR multiplier for loss scaler params (1/10 of base)


@dataclass
class PhasedTrainingConfig:
    enabled: bool = False
    # v1 phase durations (in steps) - 0 means use metric-triggered transitions
    phase1_steps: int = 0         # Foundation
    phase2_steps: int = 0         # Geometry (v1) / Expert Specialization (v2)
    phase3_steps: int = 0         # Specialization (v1) / Compression (v2)
    phase4_steps: int = 0         # Compression (v1) / Joint Fine-tuning (v2)
    phase5_steps: int = 0         # Refinement (v1 only)

    # v1 LR multipliers per phase (relative to base LR)
    # Format: [backbone, gen_head, locality_head, router, conv, expert, attention, sink, engram, v7_router]
    phase1_lr_mult: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.0, 0.0])
    phase2_lr_mult: List[float] = field(default_factory=lambda: [0.5, 0.5, 1.0, 1.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.0])
    phase3_lr_mult: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 1.0, 0.5, 1.0, 0.5, 1.0, 0.0, 0.0])
    phase4_lr_mult: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 1.0, 0.0])
    phase5_lr_mult: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.0, 0.0])

    # v2 LR multipliers per phase (relative to base LR)
    # Format: [backbone, gen_head, locality_head, peer, engram]
    v2_phase1_lr_mult: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.1, 0.0])
    v2_phase2_lr_mult: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 1.0, 0.0])
    v2_phase3_lr_mult: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.3, 1.0])
    v2_phase4_lr_mult: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 0.5, 0.5])

    # Metric thresholds for phase transitions
    val_loss_patience: int = 3
    rank_stability_window: int = 3
    routing_entropy_patience: int = 3


@dataclass
class LocalityConfig:
    enabled: bool = False
    window_size: int = 16         # W: tokens within window are positives
    neg_distance: int = 32        # 2W: tokens beyond this are negatives
    temperature: float = 0.1     # InfoNCE temperature
    n_negatives: int = 64         # number of negative samples
    loss_weight: float = 0.1      # weight for locality loss


@dataclass
class TrainingConfig:
    ablation: AblationConfig = AblationConfig.DENSE_BASELINE
    # Dataset
    dataset: str = "wikitext-103"
    # Optimization
    batch_size: int = 24
    grad_accum_steps: int = 2     # effective batch = 48
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 50_000
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 2500
    # Paths
    output_dir: str = "results"
    # Hardware
    use_bf16: bool = True
    compile_model: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    tier: TierConfig = field(default_factory=TierConfig)
    peer: PEERConfig = field(default_factory=PEERConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)
    memory_mlp: MemoryMLPTrainConfig = field(default_factory=MemoryMLPTrainConfig)
    bdh: BDHConfig = field(default_factory=BDHConfig)
    phased: PhasedTrainingConfig = field(default_factory=PhasedTrainingConfig)
    locality: LocalityConfig = field(default_factory=LocalityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    run_name: Optional[str] = None

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = self.training.ablation.value

    @staticmethod
    def from_ablation(ablation: AblationConfig, **overrides) -> "ExperimentConfig":
        """Create config for a specific ablation run."""
        cfg = ExperimentConfig()
        cfg.training.ablation = ablation
        cfg.run_name = ablation.value

        # === v1 configs ===

        if ablation == AblationConfig.DENSE_BASELINE:
            pass

        elif ablation == AblationConfig.DUAL_HEAD:
            cfg.locality.enabled = True

        elif ablation == AblationConfig.DUAL_HEAD_ROUTER:
            cfg.locality.enabled = True

        elif ablation == AblationConfig.DUAL_HEAD_ROUTER_SINK:
            cfg.locality.enabled = True

        elif ablation == AblationConfig.FULL_CORE:
            cfg.locality.enabled = True
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 10000
            cfg.phased.phase2_steps = 10000
            cfg.phased.phase3_steps = 10000

        elif ablation == AblationConfig.FULL_HRS:
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 12000

        elif ablation == AblationConfig.FULL_HRS_REFINED:
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 12000
            cfg.phased.phase5_lr_mult = [0.3, 0.5, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.5]

        # === v6 configs ===

        elif ablation == AblationConfig.V4_1024:
            # v4 at seq_len=1024 — baseline for longer sequences
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.peer.enabled = True
            cfg.router.n_tiers = 3
            cfg.model.max_seq_len = 1024
            cfg.training.batch_size = 4
            cfg.training.grad_accum_steps = 6  # 4*6*1024 = 24,576 tokens/step
            cfg.training.max_steps = 38000  # P1-P4 only
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 0  # skip P5

        elif ablation == AblationConfig.V6_GATE:
            # v4 + learned remember gate at seq_len=1024
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.engram.remember_gate_enabled = True
            cfg.peer.enabled = True
            cfg.router.n_tiers = 3
            cfg.model.max_seq_len = 1024
            cfg.training.batch_size = 4
            cfg.training.grad_accum_steps = 6
            cfg.training.max_steps = 38000
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 0

        # === v7 configs ===

        elif ablation == AblationConfig.V7_FULL:
            # v4 base + Memory MLP + V7Router (logit blending, from-scratch training)
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.peer.enabled = True
            cfg.router.n_tiers = 3  # conv, attn, sink (same as v4)
            cfg.memory_mlp.enabled = True
            cfg.training.batch_size = 8
            cfg.training.grad_accum_steps = 6  # effective batch = 48
            cfg.training.max_steps = 38000  # P1(8K)+P2(8K)+P3(10K)+P4(12K), skip P5
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 0  # skip P5 (always regresses)
            # v7 phase schedule: v7_router frozen P1-P2, wakes P3, full P4
            # [backbone, gen_head, locality_head, router, conv, expert, attention, sink, engram, v7_router]
            cfg.phased.phase1_lr_mult = [1.0, 1.0, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.0, 0.0]
            cfg.phased.phase2_lr_mult = [0.5, 0.5, 1.0, 1.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.0]
            cfg.phased.phase3_lr_mult = [0.3, 0.3, 0.3, 1.0, 0.5, 1.0, 0.5, 1.0, 0.0, 0.5]
            cfg.phased.phase4_lr_mult = [0.1, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 1.0, 1.0]

        # === v8 configs ===

        elif ablation == AblationConfig.V8_BDH:
            # v4 base + BDH-inspired upgrades
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.peer.enabled = True
            cfg.router.n_tiers = 3  # conv, attn, sink (same as v4)
            cfg.bdh.enabled = True
            cfg.training.batch_size = 4
            cfg.training.grad_accum_steps = 8  # effective batch = 32
            cfg.training.max_steps = 38000
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 0  # skip P5

        # === v9 configs ===

        elif ablation == AblationConfig.V9_LEARNABLE:
            # v8 base + learnable loss scaling
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.peer.enabled = True
            cfg.router.n_tiers = 3
            cfg.bdh.enabled = True
            cfg.bdh.learnable_loss_scaling = True
            cfg.training.batch_size = 4
            cfg.training.grad_accum_steps = 8  # effective batch = 32
            cfg.training.max_steps = 50000
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 12000

        # === v5 configs ===

        elif ablation == AblationConfig.V5_REPLACE:
            # v4 base + learnable engram-based context replacement
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.engram.replacement_enabled = True
            cfg.peer.enabled = True
            cfg.router.n_tiers = 3  # conv, attn, sink
            cfg.training.grad_accum_steps = 6  # effective batch = 8*6=48
            cfg.training.max_steps = 38000  # P1(8K)+P2(8K)+P3(10K)+P4(12K), skip P5
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 0  # skip P5 (always regresses)
            cfg.phased.phase4_lr_mult = [0.1, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 1.0]

        # === v4 configs ===

        elif ablation == AblationConfig.V4_FULL:
            # PEER as universal FFN (unconditional) + 3-tier routing (conv, attn, sink)
            # Same schedule as v3/v1 (proven best)
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.peer.enabled = True
            cfg.router.n_tiers = 3  # conv, attn, sink (no expert tier — PEER is unconditional)
            cfg.training.grad_accum_steps = 6  # effective batch = 8*6=48 (same as v1's 24*2)
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 12000
            cfg.phased.phase5_lr_mult = [0.3, 0.5, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.5]

        # === v3 configs ===

        elif ablation == AblationConfig.V3_FULL:
            # v1 routing framework + PEER as expert tier + engrams
            # Same schedule as full_hrs_refined (proven best)
            cfg.locality.enabled = True
            cfg.engram.enabled = True
            cfg.peer.enabled = True
            cfg.training.grad_accum_steps = 6  # effective batch = 8*6=48 (same as v1's 24*2)
            cfg.phased.enabled = True
            cfg.phased.phase1_steps = 8000
            cfg.phased.phase2_steps = 8000
            cfg.phased.phase3_steps = 10000
            cfg.phased.phase4_steps = 12000
            cfg.phased.phase5_steps = 12000
            cfg.phased.phase5_lr_mult = [0.3, 0.5, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.5]

        # === v2 configs ===

        elif ablation == AblationConfig.V2_ATTN_CONV:
            # Attention->Conv backbone, standard MLP, no dual-head
            pass

        elif ablation == AblationConfig.V2_ATTN_CONV_DUAL:
            # + dual-head (locality)
            cfg.locality.enabled = True

        elif ablation == AblationConfig.V2_ATTN_CONV_PEER:
            # + PEER FFN (replaces standard MLP)
            cfg.locality.enabled = True
            cfg.peer.enabled = True

        elif ablation == AblationConfig.V2_FULL:
            # Full v2: backbone + PEER + engrams + dual-head + phased training
            cfg.locality.enabled = True
            cfg.peer.enabled = True
            cfg.engram.enabled = True
            cfg.phased.enabled = True
            # 3-4 phase schedule for v2
            cfg.phased.phase1_steps = 12000   # Foundation: backbone + dual-head + conv
            cfg.phased.phase2_steps = 12000   # Expert Specialization: PEER full LR
            cfg.phased.phase3_steps = 14000   # Compression: engram activation
            cfg.phased.phase4_steps = 12000   # Joint fine-tuning

        # Apply any overrides
        for key, val in overrides.items():
            parts = key.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

        return cfg

    def is_v2(self) -> bool:
        """Check if this is a v2 config."""
        return self.training.ablation.value.startswith("v2_")

    def uses_router(self) -> bool:
        """v1/v3/v4/v5/v6/v7/v8: uses routing machinery."""
        return self.training.ablation.value in (
            "dual_head_router", "dual_head_router_sink",
            "full_core", "full_hrs", "full_hrs_refined",
            "v3_full", "v4_full", "v5_replace",
            "v4_1024", "v6_gate", "v7_full", "v8_bdh", "v9_learnable",
        )

    def uses_sink(self) -> bool:
        return self.training.ablation.value in (
            "dual_head_router_sink",
            "full_core", "full_hrs", "full_hrs_refined",
            "v3_full", "v4_full", "v5_replace",
            "v4_1024", "v6_gate", "v7_full", "v8_bdh", "v9_learnable",
        )

    def uses_engrams(self) -> bool:
        return self.engram.enabled and self.training.ablation.value in (
            "full_hrs", "full_hrs_refined", "v2_full",
            "v3_full", "v4_full", "v5_replace",
            "v4_1024", "v6_gate", "v7_full", "v8_bdh", "v9_learnable",
        )

    def uses_peer(self) -> bool:
        return self.peer.enabled and self.training.ablation.value in (
            "v2_attn_conv_peer", "v2_full",
            "v3_full", "v4_full", "v5_replace",
            "v4_1024", "v6_gate", "v7_full", "v8_bdh", "v9_learnable",
        )

    def uses_memory_mlp(self) -> bool:
        """v7: Memory MLP + V7Router logit blending."""
        return self.memory_mlp.enabled and self.training.ablation.value in (
            "v7_full",
        )

    def uses_attn_conv_backbone(self) -> bool:
        """v2: fixed attention->conv layer structure."""
        return self.is_v2()

    def uses_phased_training(self) -> bool:
        return self.phased.enabled and self.training.ablation.value in (
            "full_core", "full_hrs", "full_hrs_refined", "v2_full",
            "v3_full", "v4_full", "v5_replace",
            "v4_1024", "v6_gate", "v7_full", "v8_bdh", "v9_learnable",
        )

    def uses_bdh(self) -> bool:
        """v8/v9: BDH-inspired upgrades."""
        return self.bdh.enabled and self.training.ablation.value in ("v8_bdh", "v9_learnable")

    def uses_learnable_loss_scaling(self) -> bool:
        """v9: learnable auxiliary loss scales."""
        return self.uses_bdh() and self.bdh.learnable_loss_scaling

    def uses_engram_replacement(self) -> bool:
        """v5: blend engrams in-place instead of prepending."""
        return self.engram.replacement_enabled and self.training.ablation.value == "v5_replace"

    def uses_remember_gate(self) -> bool:
        """v6: learned remember gate before prepend."""
        return self.engram.remember_gate_enabled and self.training.ablation.value == "v6_gate"

    def n_attention_layers(self) -> int:
        """Number of layers using attention (first half for v2, all for v1)."""
        if self.is_v2():
            return self.model.n_layers // 2
        return self.model.n_layers

    def n_conv_layers(self) -> int:
        """Number of layers using conv (second half for v2, 0 for v1)."""
        if self.is_v2():
            return self.model.n_layers - self.model.n_layers // 2
        return 0
