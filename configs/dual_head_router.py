"""Config 3: Dual head + router — + learned router + tiered compute."""
from config import ExperimentConfig, AblationConfig

def get_config():
    return ExperimentConfig.from_ablation(AblationConfig.DUAL_HEAD_ROUTER)
