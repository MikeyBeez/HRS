"""Config 1: Dense baseline — standard transformer, CE only."""
from config import ExperimentConfig, AblationConfig

def get_config():
    return ExperimentConfig.from_ablation(AblationConfig.DENSE_BASELINE)
