"""Fine-tune a pruned model with sticky masks.

The LM and passkey training loops are separate because they use different
data pipelines and loss masks. Both call apply_masks() after every
optimizer step so pruned positions stay zero even though the optimizer
(with decoupled weight decay and momentum) would otherwise nudge them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from experiments.diagonal_attention.config import (
    ModelConfig,
    PasskeyConfig,
    TrainConfig,
)
from experiments.diagonal_attention.data import (
    load_shakespeare,
    loss_mask_for_answer,
    sample_lm_batch,
    sample_passkey_batch,
)
from experiments.diagonal_attention.model import TinyTransformer

from experiments.pruning.prune import PruneState, apply_masks


def finetune_lm(
    model: TinyTransformer,
    mcfg: ModelConfig,
    st: PruneState,
    steps: int,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 1234,
) -> float:
    """Fine-tune on Shakespeare. Returns mean loss over the last 50 steps."""
    if steps <= 0:
        return float("nan")
    device = next(model.parameters()).device
    train_data, _val, _info = load_shakespeare()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                             betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    model.train()
    losses = []
    for _ in range(steps):
        x, y = sample_lm_batch(train_data, batch_size, mcfg.ctx_len, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        apply_masks(st)
        losses.append(float(loss.item()))
    return float(np.mean(losses[-50:]))


def finetune_passkey(
    model: TinyTransformer,
    mcfg: ModelConfig,
    pkcfg: PasskeyConfig,
    st: PruneState,
    steps: int,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 1234,
) -> float:
    if steps <= 0:
        return float("nan")
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                             betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    ans_mask = loss_mask_for_answer(pkcfg, device)
    model.train()
    losses = []
    for _ in range(steps):
        x, y = sample_passkey_batch(pkcfg, batch_size, device, rng)
        logits = model(x)
        B, T, V = logits.shape
        fl = logits.reshape(-1, V)
        ft = y.reshape(-1)
        fm = ans_mask[None].expand(B, T).reshape(-1)
        loss = F.cross_entropy(fl[fm], ft[fm])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        apply_masks(st)
        losses.append(float(loss.item()))
    return float(np.mean(losses[-50:]))
