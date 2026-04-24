"""Structured head pruning: zero chosen (layer, head) slices of W_Q/K/V/O.

Reuses experiments.pruning.prune's PruneState machinery so the same grad
hooks keep pruned positions at zero during fine-tuning.

Layout:
  For an MHA layer with d_model=d, n_heads=H, d_head=dh=d/H:
    - W_Q, W_K, W_V: Linear(d, d). Output dim is split into (H, dh).
      Zeroing head h => zero rows [h*dh : (h+1)*dh, :] of .weight.
    - W_O: Linear(d, d). Input dim is split into (H, dh).
      Zeroing head h => zero columns [:, h*dh : (h+1)*dh] of .weight.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from experiments.diagonal_attention.model import TinyTransformer
from experiments.pruning.prune import PruneState, _install_grad_hooks


HeadId = Tuple[int, int]  # (layer_idx, head_idx)


def _head_bounds(head_idx: int, d_head: int) -> Tuple[int, int]:
    return head_idx * d_head, (head_idx + 1) * d_head


def build_head_prune_state(
    model: TinyTransformer,
    heads_to_zero: Iterable[HeadId],
) -> PruneState:
    """Build a PruneState whose masks zero exactly the specified heads.

    If called with an empty list, returns a state with all-ones masks
    (effectively a no-op, but grad hooks are still installed).
    """
    heads = list(heads_to_zero)
    H = model.cfg.n_heads
    d = model.cfg.d_model
    dh = model.cfg.d_head

    st = PruneState()
    for layer_idx, blk in enumerate(model.blocks):
        attn = blk.attn
        layer_heads = [h for (l, h) in heads if l == layer_idx]

        # W_Q, W_K, W_V: zero rows for each pruned head.
        for proj_name in ("W_Q", "W_K", "W_V"):
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue
            w = proj.weight
            mask = torch.ones_like(w, dtype=torch.bool)
            for h in layer_heads:
                lo, hi = _head_bounds(h, dh)
                mask[lo:hi, :] = False
            st.targets.append((proj, "weight"))
            st.masks[id(w)] = mask
            w.data.mul_(mask)

        # W_O: zero columns for each pruned head.
        wo = attn.W_O
        w = wo.weight
        mask = torch.ones_like(w, dtype=torch.bool)
        for h in layer_heads:
            lo, hi = _head_bounds(h, dh)
            mask[:, lo:hi] = False
        st.targets.append((wo, "weight"))
        st.masks[id(w)] = mask
        w.data.mul_(mask)

    _install_grad_hooks(st)
    return st


def all_heads(model: TinyTransformer) -> List[HeadId]:
    return [(l, h)
             for l in range(model.cfg.n_layers)
             for h in range(model.cfg.n_heads)]


def compose_with_mlp_mask_state(head_state: PruneState,
                                  mlp_state: PruneState) -> PruneState:
    """Combine an existing head-pruning state with an MLP-pruning state into
    a single PruneState. The two states target disjoint parameters, so it is
    safe to union their masks and hooks."""
    combined = PruneState()
    combined.targets.extend(head_state.targets)
    combined.targets.extend(mlp_state.targets)
    combined.masks.update(head_state.masks)
    combined.masks.update(mlp_state.masks)
    # Re-install hooks against the combined state so closures reference it.
    _install_grad_hooks(combined)
    # Release the original hooks to avoid double-zeroing.
    head_state.release()
    mlp_state.release()
    return combined
