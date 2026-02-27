"""Config 4: Dual head + router + sink channel."""
from config import ExperimentConfig, AblationConfig

def get_config():
    return ExperimentConfig.from_ablation(AblationConfig.DUAL_HEAD_ROUTER_SINK)
