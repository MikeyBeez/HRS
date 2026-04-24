"""Shared utilities for the amplitude sweep phases.

Loads the baseline LM and passkey models once, installs the dynamic
scaling hook, and exposes an `eval_with_scales` helper that sets the
scale map, runs eval on both models, and returns metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.diagonal_attention.config import ModelConfig, PasskeyConfig
from experiments.diagonal_attention.model import TinyTransformer
from experiments.pruning.eval import eval_lm, eval_passkey

from experiments.amplitude_sweep.alpha_hooks import (
    install_dynamic_scaling,
    set_scales,
    clear_scales,
)


CKPT_DIR = Path("/mnt/data/Code/HRS/experiments/pruning/checkpoints")


@dataclass
class BaselinesAS:
    lm_model: TinyTransformer
    lm_mcfg: ModelConfig
    pk_model: TinyTransformer
    pk_mcfg: ModelConfig
    pkcfg: PasskeyConfig


def load_baselines(device: torch.device) -> BaselinesAS:
    lm_ckpt = torch.load(CKPT_DIR / "mha_lm.pt", map_location=device,
                          weights_only=False)
    pk_ckpt = torch.load(CKPT_DIR / "mha_passkey.pt", map_location=device,
                          weights_only=False)
    lm_mcfg = ModelConfig(**lm_ckpt["mcfg"])
    lm_model = TinyTransformer(lm_mcfg).to(device)
    lm_model.load_state_dict(lm_ckpt["state_dict"])
    lm_model.eval()
    install_dynamic_scaling(lm_model)

    pk_mcfg = ModelConfig(**pk_ckpt["mcfg"])
    pk_model = TinyTransformer(pk_mcfg).to(device)
    pk_model.load_state_dict(pk_ckpt["state_dict"])
    pk_model.eval()
    install_dynamic_scaling(pk_model)

    pkcfg = PasskeyConfig(**pk_ckpt["pkcfg"])
    return BaselinesAS(lm_model, lm_mcfg, pk_model, pk_mcfg, pkcfg)


def eval_point(bl: BaselinesAS, scales: dict, *, measure_lm: bool = True) -> dict:
    """Run eval with the given scale map applied on both models.

    scales: {(layer_idx, head_idx): float}. 1.0 = unchanged.
    """
    set_scales(scales)

    pk_metrics = eval_passkey(bl.pk_model, bl.pkcfg)
    rec = {
        "passkey_exact": pk_metrics["overall_exact_acc"],
        "passkey_digit": pk_metrics["overall_digit_acc"],
    }
    if measure_lm:
        rec["val_ppl"] = eval_lm(bl.lm_model, bl.lm_mcfg)

    clear_scales()
    return rec
