"""Config 5: Full core — all core components + phased training."""
from config import ExperimentConfig, AblationConfig

def get_config():
    return ExperimentConfig.from_ablation(AblationConfig.FULL_CORE)
