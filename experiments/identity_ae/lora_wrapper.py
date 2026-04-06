"""Minimal LoRA wrapper for test-time training.

Adds low-rank adapters to specified linear layers. Base weights frozen.
All learning happens in the small A/B matrices.
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-rank adapter wrapping a frozen linear layer."""

    def __init__(self, base_layer, rank=16, alpha=32):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))

        for p in base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


def apply_lora(model, rank=16, alpha=32, target_modules=None):
    """Apply LoRA to specified modules in the model.

    Args:
        model: the transformer model
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: list of substrings to match (e.g., ['qkv', 'out_proj'])
                        If None, applies to all attention projections.

    Returns:
        n_lora_params: number of trainable LoRA parameters added
    """
    if target_modules is None:
        target_modules = ['qkv', 'out_proj']

    n_lora = 0
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                # Don't apply to autoencoder, embedding, or lm_head
                if 'autoencoder' in name or 'tok_emb' in name or 'lm_head' in name:
                    continue
                if 'encoder' in name or 'decoder' in name:
                    continue
                replacements.append((name, module))

    for name, module in replacements:
        lora = LoRALayer(module, rank=rank, alpha=alpha)
        # Navigate to parent and replace
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lora)
        n_lora += lora.lora_A.numel() + lora.lora_B.numel()

    # Freeze everything except LoRA
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    return n_lora


def get_lora_state_dict(model):
    """Extract only LoRA parameters."""
    return {k: v.clone() for k, v in model.named_parameters() if 'lora_' in k}


def load_lora_state_dict(model, state_dict):
    """Load LoRA parameters."""
    current = dict(model.named_parameters())
    for k, v in state_dict.items():
        if k in current:
            current[k].data.copy_(v)


def reset_lora(model):
    """Reset all LoRA parameters to zero (fresh adapter)."""
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            nn.init.normal_(param, std=0.01)
        elif 'lora_B' in name:
            nn.init.zeros_(param)


def lora_weight_stats(model):
    """Get LoRA adapter statistics."""
    norms_a, norms_b = [], []
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            norms_a.append(param.data.norm().item())
        elif 'lora_B' in name:
            norms_b.append(param.data.norm().item())
    return {
        "n_adapters": len(norms_a),
        "mean_norm_A": sum(norms_a) / len(norms_a) if norms_a else 0,
        "mean_norm_B": sum(norms_b) / len(norms_b) if norms_b else 0,
        "max_norm_A": max(norms_a) if norms_a else 0,
        "max_norm_B": max(norms_b) if norms_b else 0,
    }
