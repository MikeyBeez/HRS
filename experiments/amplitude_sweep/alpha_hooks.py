"""Dynamic head-output scaling via monkey-patched MHA forward.

Install once on a loaded model; then mutate the shared SCALES dict
between evals to change which heads are scaled by what. No need to
reload the model per sweep point.

Usage:
    install_dynamic_scaling(model)
    set_scales({(2, 0): 0.5})
    ... run eval ...
    set_scales({(2, 0): 0.0, (2, 1): 5.0})
    ... run eval ...
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


SCALES: Dict[Tuple[int, int], float] = {}


def set_scales(new_scales: Dict[Tuple[int, int], float]):
    """Replace the current scale map. Keys are (layer_idx, head_idx)."""
    SCALES.clear()
    SCALES.update(new_scales)


def clear_scales():
    SCALES.clear()


def install_dynamic_scaling(model):
    """Patch every Block.attn.forward so that the head-output tensor is
    scaled element-wise by SCALES[(layer, head)] (defaulting to 1.0).

    The scaling is applied AFTER the attention×V einsum and BEFORE the
    concatenation + W_O projection — i.e., the scaled quantity is exactly
    the contribution that head makes to the residual stream.
    """
    for layer_idx, blk in enumerate(model.blocks):
        attn_module = blk.attn

        def make_patched(module, layer_idx):
            def patched(x):
                B, T, d = x.shape
                H = module.cfg.n_heads
                dh = module.cfg.d_head
                scores = module.compute_scores(x)
                causal = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal, float("-inf"))
                attn = F.softmax(scores, dim=-1)
                V = module.W_V(x).view(B, T, H, dh).transpose(1, 2)  # (B,H,T,dh)

                if attn.dim() == 3:
                    out = torch.einsum("bts,bhsd->bhtd", attn, V)
                else:
                    out = torch.einsum("bhts,bhsd->bhtd", attn, V)

                # Per-head scaling on the head-output contribution.
                for (l, h), s in SCALES.items():
                    if l == layer_idx and s != 1.0:
                        out[:, h, :, :] = out[:, h, :, :] * s

                out = out.transpose(1, 2).contiguous().view(B, T, d)
                return module.W_O(out)
            return patched

        attn_module.forward = make_patched(attn_module, layer_idx)


def fit_sigmoid(alphas, passkeys):
    """Fit passkey(α) = 1 / (1 + exp(-k * (α - α*))) via minimal gradient
    descent. Returns (alpha_star, k, fit_quality_r2). Used for phase-1
    summary only."""
    import numpy as np
    a = np.asarray(alphas, dtype=np.float64)
    y = np.asarray(passkeys, dtype=np.float64)

    # Initialize α* at the α closest to passkey=0.5, k at 10.
    astar = float(a[np.argmin(np.abs(y - 0.5))])
    k = 10.0

    astar_t = torch.tensor(astar, requires_grad=True)
    k_t = torch.tensor(k, requires_grad=True)
    a_t = torch.tensor(a, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    opt = torch.optim.Adam([astar_t, k_t], lr=0.05)
    for _ in range(800):
        pred = torch.sigmoid(k_t * (a_t - astar_t))
        loss = ((pred - y_t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = torch.sigmoid(k_t * (a_t - astar_t)).numpy()
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return float(astar_t.item()), float(k_t.item()), float(r2)


def transition_width(alphas, passkeys, lo: float = 0.1, hi: float = 0.9):
    """α range where passkey crosses from hi to lo (assumes monotonic decrease
    as α decreases). Returns (alpha_at_hi, alpha_at_lo, width) by linear
    interpolation."""
    import numpy as np
    a = np.asarray(alphas, dtype=np.float64)
    y = np.asarray(passkeys, dtype=np.float64)
    order = np.argsort(-a)  # descending α
    a = a[order]; y = y[order]

    def _interp(target):
        # Find first i where y[i] <= target (scanning descending α).
        for i in range(1, len(a)):
            if y[i] <= target <= y[i - 1]:
                # Linear interp between (a[i-1], y[i-1]) and (a[i], y[i]).
                if y[i - 1] == y[i]:
                    return a[i]
                t = (y[i - 1] - target) / (y[i - 1] - y[i])
                return a[i - 1] + t * (a[i] - a[i - 1])
        return None

    a_hi = _interp(hi)
    a_lo = _interp(lo)
    if a_hi is None or a_lo is None:
        return (a_hi, a_lo, None)
    return (a_hi, a_lo, float(a_hi - a_lo))
