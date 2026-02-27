"""Config 2: Dual head — + locality head (contrastive objective)."""
from config import ExperimentConfig, AblationConfig

def get_config():
    return ExperimentConfig.from_ablation(AblationConfig.DUAL_HEAD)
