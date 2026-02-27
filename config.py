"""Configuration for Hierarchical Routed Sinkformer experiments."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class AblationConfig(Enum):
    DENSE_BASELINE = "dense_baseline"           # Standard transformer, CE only
    DUAL_HEAD = "dual_head"                     # + locality head (dual objective)
    DUAL_HEAD_ROUTER = "dual_head_router"       # + learned router + tiered compute
    DUAL_HEAD_ROUTER_SINK = "dual_head_router_sink"  # + sink channel
    FULL_CORE = "full_core"                     # All core + phased training
    FULL_HRS = "full_hrs"                       # Core + engrams (all 5 phases)
    FULL_HRS_REFINED = "full_hrs_refined"       # Full HRS + engram refinement


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
class EngramConfig:
    enabled: bool = False
    window_size: int = 128        # N tokens compressed into K engrams
    n_engrams: int = 4            # K engram vectors per window
    extract_layer: int = 1        # extract from this layer (0-indexed, layer 2 of 4 = index 1)
    engram_dim: int = 512         # engram vector dimension (= d_model)
    recon_loss_weight: float = 0.1  # weight for reconstruction loss


@dataclass
class PhasedTrainingConfig:
    enabled: bool = False
    # Phase durations (in steps) - 0 means use metric-triggered transitions
    phase1_steps: int = 0         # Foundation
    phase2_steps: int = 0         # Geometry
    phase3_steps: int = 0         # Specialization
    phase4_steps: int = 0         # Compression
    phase5_steps: int = 0         # Refinement
    # LR multipliers per phase (relative to base LR)
    # Format: [backbone, gen_head, locality_head, router, conv, expert, attention, sink, engram]
    # Phase 1 (Foundation): backbone + gen_head + conv at full LR
    phase1_lr_mult: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.0])
    # Phase 2 (Geometry): add locality head + router
    phase2_lr_mult: List[float] = field(default_factory=lambda: [0.5, 0.5, 1.0, 1.0, 0.5, 0.1, 0.1, 0.1, 0.0])
    # Phase 3 (Specialization): router + experts + sink
    phase3_lr_mult: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3, 1.0, 0.5, 1.0, 0.5, 1.0, 0.0])
    # Phase 4 (Compression): engram encoder
    phase4_lr_mult: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 1.0])
    # Phase 5 (Refinement): freeze engram, retrain consumers
    phase5_lr_mult: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.0])
    # Metric thresholds for phase transitions
    val_loss_patience: int = 3    # evals without improvement to trigger transition
    rank_stability_window: int = 3  # evals to check rank stability
    routing_entropy_patience: int = 3  # evals for routing entropy stability


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
    engram: EngramConfig = field(default_factory=EngramConfig)
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

        if ablation == AblationConfig.DENSE_BASELINE:
            # Pure transformer, CE only
            pass

        elif ablation == AblationConfig.DUAL_HEAD:
            cfg.locality.enabled = True

        elif ablation == AblationConfig.DUAL_HEAD_ROUTER:
            cfg.locality.enabled = True
            # Router and tiers enabled implicitly when ablation >= DUAL_HEAD_ROUTER

        elif ablation == AblationConfig.DUAL_HEAD_ROUTER_SINK:
            cfg.locality.enabled = True
            # Router + tiers + sink channel

        elif ablation == AblationConfig.FULL_CORE:
            cfg.locality.enabled = True
            cfg.phased.enabled = True
            # Set default phase durations for non-metric-triggered runs
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
            # Phase 5: engram LR = 0.5 so injector retrains
            # (encoder frozen via requires_grad=False in train.py)
            cfg.phased.phase5_lr_mult = [0.3, 0.5, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.5]

        # Apply any overrides
        for key, val in overrides.items():
            parts = key.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

        return cfg

    def uses_router(self) -> bool:
        return self.training.ablation.value in (
            "dual_head_router", "dual_head_router_sink",
            "full_core", "full_hrs", "full_hrs_refined",
        )

    def uses_sink(self) -> bool:
        return self.training.ablation.value in (
            "dual_head_router_sink",
            "full_core", "full_hrs", "full_hrs_refined",
        )

    def uses_engrams(self) -> bool:
        return self.engram.enabled and self.training.ablation.value in (
            "full_hrs", "full_hrs_refined",
        )

    def uses_phased_training(self) -> bool:
        return self.phased.enabled and self.training.ablation.value in (
            "full_core", "full_hrs", "full_hrs_refined",
        )
