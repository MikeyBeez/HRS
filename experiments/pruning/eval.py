"""Wraps the diagonal_attention eval functions for pruned models."""
from __future__ import annotations

import numpy as np
import torch

from experiments.diagonal_attention.config import (
    ModelConfig,
    PasskeyConfig,
    TrainConfig,
)
from experiments.diagonal_attention.data import (
    load_shakespeare,
    make_passkey_eval_set,
)
from experiments.diagonal_attention.eval_passkey import evaluate as _eval_passkey
from experiments.diagonal_attention.train import eval_ppl as _eval_ppl


def eval_lm(model, mcfg: ModelConfig, n_batches: int = 80) -> float:
    _train, val, _info = load_shakespeare()
    tcfg = TrainConfig()
    device = next(model.parameters()).device
    return _eval_ppl(model, val, tcfg, mcfg, device, n_batches=n_batches)


def eval_passkey(model, pkcfg: PasskeyConfig, seed: int = 9999) -> dict:
    device = next(model.parameters()).device
    eval_set = make_passkey_eval_set(pkcfg, np.random.default_rng(seed))
    return _eval_passkey(model, pkcfg, eval_set, device)
