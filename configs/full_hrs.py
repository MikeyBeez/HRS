"""Config 6: Full HRS — core + engrams (all 5 phases)."""
from config import ExperimentConfig, AblationConfig

def get_config():
    return ExperimentConfig.from_ablation(AblationConfig.FULL_HRS)
