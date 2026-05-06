"""SwitchingLMHead — drop-in replacement for V22's tied lm_head that
routes between a frozen original (kept for base-model queries) and a
trainable copy (used when adapters are loaded).

Design notes:
  - The frozen original keeps its tied weight reference to tok_emb.
    Inference with no adapter active still uses base-model weights
    directly.
  - The trainable copy is a fresh nn.Linear(d_model, vocab_size,
    bias=False) initialized from the frozen original at construction
    time. It diverges as adapters are trained.
  - The `use_trainable` flag is set externally (during training and
    during adapter-loaded inference). When False, we run the frozen
    decoder; when True, the trainable one.

Usage:
  decoder = SwitchingLMHead(model.lm_head, d_model, vocab_size)
  model.lm_head = decoder
  decoder.use_trainable = True   # activate trainable path
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SwitchingLMHead(nn.Module):
    def __init__(self, frozen_lm_head: nn.Linear, d_model: int, vocab_size: int):
        super().__init__()
        self.frozen = frozen_lm_head  # weight stays tied to tok_emb if it was

        self.trainable = nn.Linear(d_model, vocab_size, bias=False,
                                     device=frozen_lm_head.weight.device,
                                     dtype=frozen_lm_head.weight.dtype)
        with torch.no_grad():
            self.trainable.weight.data.copy_(frozen_lm_head.weight.data)

        # Freeze the frozen path
        for p in self.frozen.parameters():
            p.requires_grad = False
        # Trainable path is trainable by default
        for p in self.trainable.parameters():
            p.requires_grad = True

        self.use_trainable = False

    def forward(self, x):
        if self.use_trainable:
            return self.trainable(x)
        return self.frozen(x)


def install_switching_lm_head(model):
    """Replace model.lm_head with a SwitchingLMHead.
    Returns the new SwitchingLMHead so the caller can configure it.
    Preserves the original frozen head behavior when use_trainable=False."""
    frozen = model.lm_head
    d_model = frozen.in_features
    vocab_size = frozen.out_features
    sw = SwitchingLMHead(frozen, d_model, vocab_size)
    model.lm_head = sw
    return sw


def get_decoder_state(model):
    """Extract the trainable decoder weight (CPU tensor)."""
    return model.lm_head.trainable.weight.detach().cpu().clone()


@torch.no_grad()
def load_decoder_state(model, weight):
    """Copy a saved decoder weight back into the model's trainable head."""
    model.lm_head.trainable.weight.data.copy_(weight.to(
        model.lm_head.trainable.weight.device,
        dtype=model.lm_head.trainable.weight.dtype,
    ))


def trainable_decoder_params(model):
    """Yield the parameters of the trainable decoder (for the optimizer)."""
    return list(model.lm_head.trainable.parameters())
