"""MultiLoRALayer for additive k-adapter composition.

Drop-in extension of experiments/identity_ae/lora_wrapper.py's LoRALayer
that holds up to max_k stacked (A, B) parameter pairs and computes:

  y = W_0 x + sum_{i=0..n_active-1} (x A_i B_i) * (alpha / rank)

For n_active = 1, this is bit-equivalent to the original LoRALayer
(both compute (x A_0 B_0) * scaling). For n_active = 2, this is the
additive k=2 composition: contributions from both adapters sum at every
LoRA-augmented layer, before nonlinearities downstream.

Slots that are not currently loaded have B = 0, so their A/B product
contributes 0 — the sum-then-scale order means contributions sum
exactly, no double-scaling.

Usage:
  apply_multi_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS, max_k=2)
  set_active_adapters(model, [sd_i])           # k=1
  set_active_adapters(model, [sd_i, sd_j])     # k=2
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiLoRALayer(nn.Module):
    """N stacked LoRA adapters, additively composed."""

    def __init__(self, base_layer, rank=128, alpha=256, max_k=2):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.max_k = max_k

        in_f = base_layer.in_features
        out_f = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # max_k stacked adapter slots; B's start at zero so unloaded slots
        # contribute nothing.
        self.lora_As = nn.ParameterList([
            nn.Parameter(torch.randn(in_f, rank, device=device, dtype=dtype) * 0.01)
            for _ in range(max_k)
        ])
        self.lora_Bs = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=dtype))
            for _ in range(max_k)
        ])
        self.n_active = 0

        for p in base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        if self.n_active == 0:
            return base_out
        # Sum all active adapter contributions, then scale once.
        # For n_active=1 this is bit-equivalent to single-LoRA.
        contributions = 0
        for i in range(self.n_active):
            contributions = contributions + (x @ self.lora_As[i] @ self.lora_Bs[i])
        return base_out + contributions * self.scaling

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


def apply_multi_lora(model, rank=128, alpha=256, target_modules=None, max_k=2):
    """Replace target Linear layers with MultiLoRALayer."""
    if target_modules is None:
        target_modules = ['qkv', 'out_proj']

    n_total = 0
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                if 'autoencoder' in name or 'tok_emb' in name or 'lm_head' in name:
                    continue
                if 'encoder' in name or 'decoder' in name:
                    continue
                replacements.append((name, module))

    for name, module in replacements:
        lora = MultiLoRALayer(module, rank=rank, alpha=alpha, max_k=max_k)
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lora)
        n_total += sum(p.numel() for p in lora.lora_As) + sum(p.numel() for p in lora.lora_Bs)

    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    return n_total


@torch.no_grad()
def set_active_adapters(model, state_dicts):
    """Load a list of single-adapter state_dicts into the multi-LoRA slots.

    Each state_dict was saved by the original LoRA wrapper with keys like
      "transformer.blocks.4.attn.qkv.lora_A"
      "transformer.blocks.4.attn.qkv.lora_B"

    We map them onto MultiLoRALayer's slot i:
      "transformer.blocks.4.attn.qkv.lora_As.{slot}"
      "transformer.blocks.4.attn.qkv.lora_Bs.{slot}"

    Other slots have their B zeroed (so their contribution is 0).
    """
    n = len(state_dicts)

    # Set n_active and zero ALL B's in ALL slots
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            assert n <= module.max_k, f"n={n} exceeds max_k={module.max_k}"
            module.n_active = n
            for slot in range(module.max_k):
                module.lora_Bs[slot].zero_()

    # For each adapter, copy A and B into its slot
    for slot, sd in enumerate(state_dicts):
        for key, val in sd.items():
            if key.endswith(".lora_A"):
                base = key[: -len(".lora_A")]
                target_key = f"{base}.lora_As.{slot}"
            elif key.endswith(".lora_B"):
                base = key[: -len(".lora_B")]
                target_key = f"{base}.lora_Bs.{slot}"
            else:
                continue
            # Walk to the parameter
            obj = model
            for p in target_key.split('.'):
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            obj.data.copy_(val)


@torch.no_grad()
def reset_multi_lora(model):
    """Clear all slots — n_active = 0, all B's zeroed."""
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            module.n_active = 0
            for slot in range(module.max_k):
                module.lora_Bs[slot].zero_()
