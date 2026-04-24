"""Per-layer magnitude pruning with sticky masks.

Mask lifecycle:
  1. Pick eligible (module, param_name) pairs by `scope`.
  2. At each prune step, compute per-matrix magnitude threshold, build a binary
     mask where |w| >= threshold. New masks AND with any existing mask, so
     sparsity is monotonically non-decreasing across iterative prune steps.
  3. After every optimizer step, multiply weight data by mask (in place) to
     zero out pruned positions. We also register a backward-hook to zero grads
     at pruned positions so Adam's moment estimates don't drift there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# A single pruning target: a parameter tensor referenced by (module, attribute).
Target = Tuple[nn.Module, str]

SCOPES = ("mlp", "attn", "all")


def collect_targets(model: nn.Module, scope: str) -> List[Target]:
    """Enumerate weight matrices to prune.

    Scopes:
      - mlp:  both FFN Linears inside each Block (ffn[0], ffn[2]).
      - attn: W_Q, W_K, W_V, W_O in each Block's MHA attention.
      - all:  both of the above.
    Embeddings, LayerNorm, tied LM head, and biases are never pruned.
    """
    assert scope in SCOPES, f"unknown scope {scope}"
    targets: List[Target] = []
    for blk in model.blocks:
        if scope in ("mlp", "all"):
            # blk.ffn = Sequential(Linear, GELU, Linear)
            targets.append((blk.ffn[0], "weight"))
            targets.append((blk.ffn[2], "weight"))
        if scope in ("attn", "all"):
            attn = blk.attn
            for name in ("W_Q", "W_K", "W_V", "W_O"):
                mod = getattr(attn, name, None)
                if mod is not None:
                    targets.append((mod, "weight"))
    return targets


@dataclass
class PruneState:
    """Holds masks + bookkeeping. One per model."""
    masks: Dict[int, torch.Tensor] = field(default_factory=dict)  # id(param) -> mask
    targets: List[Target] = field(default_factory=list)
    grad_hooks: list = field(default_factory=list)

    def release(self):
        for h in self.grad_hooks:
            h.remove()
        self.grad_hooks.clear()


def _param_of(t: Target) -> nn.Parameter:
    mod, attr = t
    return getattr(mod, attr)


def init_prune_state(model: nn.Module, scope: str) -> PruneState:
    st = PruneState(targets=collect_targets(model, scope))
    for t in st.targets:
        p = _param_of(t)
        st.masks[id(p)] = torch.ones_like(p, dtype=torch.bool)
    _install_grad_hooks(st)
    return st


def _install_grad_hooks(st: PruneState):
    for t in st.targets:
        p = _param_of(t)
        mask_id = id(p)

        def make_hook(mask_id=mask_id):
            def hook(grad):
                m = st.masks[mask_id]
                return grad * m
            return hook

        st.grad_hooks.append(p.register_hook(make_hook()))


@torch.no_grad()
def prune_to_sparsity(st: PruneState, target_sparsity: float):
    """Advance every target's mask so at least `target_sparsity` of weights are
    zero. Uses per-matrix magnitude threshold on the *currently-live* weights.

    Monotonic: each new mask ANDs with the prior one, so previously-pruned
    positions stay pruned.
    """
    assert 0.0 <= target_sparsity < 1.0
    for t in st.targets:
        p = _param_of(t)
        m = st.masks[id(p)]
        w = p.data
        mag = w.abs()
        # Only consider currently-live weights for threshold computation — but
        # already-pruned weights have magnitude ~0 anyway so they drop first
        # naturally. Using all weights gives a stable threshold across calls.
        k = int(round(target_sparsity * w.numel()))
        if k <= 0:
            continue
        # torch.kthvalue returns the k-th smallest; everything below the result
        # gets zeroed.
        thresh = torch.kthvalue(mag.flatten(), k).values
        new_m = mag > thresh
        # Keep earlier zeros zeroed.
        new_m = new_m & m
        st.masks[id(p)] = new_m
        p.data.mul_(new_m)


@torch.no_grad()
def apply_masks(st: PruneState):
    """Zero out pruned positions on the live weights. Call after opt.step()."""
    for t in st.targets:
        p = _param_of(t)
        p.data.mul_(st.masks[id(p)])


def sparsity_report(st: PruneState, model: nn.Module) -> dict:
    per_layer = []
    total = 0
    zeroed = 0
    for t in st.targets:
        p = _param_of(t)
        m = st.masks[id(p)]
        n = m.numel()
        z = n - int(m.sum().item())
        per_layer.append({
            "module": type(t[0]).__name__,
            "shape": list(p.shape),
            "sparsity": z / n,
            "n": n,
            "nonzero": n - z,
        })
        total += n
        zeroed += z
    # Total nonzero param count including non-pruned params too.
    all_total = sum(q.numel() for q in model.parameters())
    all_nonzero = all_total - zeroed
    return {
        "per_layer": per_layer,
        "scope_sparsity": zeroed / total if total else 0.0,
        "scope_total_params": total,
        "scope_zeroed_params": zeroed,
        "effective_total_params": all_nonzero,
        "all_params": all_total,
    }
