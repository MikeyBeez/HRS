"""Config 7: Full HRS refined — full HRS + engram refinement (Phase 5)."""
from config import ExperimentConfig, AblationConfig

def get_config():
    return ExperimentConfig.from_ablation(AblationConfig.FULL_HRS_REFINED)
