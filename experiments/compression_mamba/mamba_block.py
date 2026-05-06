"""Pure-PyTorch Mamba (S6) block.

Implements the selective state-space block from the Mamba paper:
  - Input: (B, T, D)
  - In-projection: D -> 2*d_inner, split into x and gate
  - Causal Conv1d on x (kernel 4, depthwise = grouped by d_inner)
  - SiLU activation on x
  - SSM parameters from x: dt, B, C (selective)
  - State A is a learned per-channel matrix (-exp(A_log) keeps eigenvalues
    in left half-plane → stable)
  - Discretization: dA = exp(dt * A); dB = dt * B
  - Selective scan: state[t] = dA[t] * state[t-1] + dB[t] * x[t];
                    y[t] = (state[t] * C[t]).sum(-1) + D * x[t]
  - Gating: y *= silu(gate)
  - Out-projection: d_inner -> D

CUDA-kernel-free; runs the scan as a Python loop with vectorized inner
ops. For short compressed sequences (T/16 ≤ 256) this is acceptable. Not
suitable for very long T without the parallel-scan kernel.

Causality: convolution is left-padded so output[t] only sees inputs ≤ t;
selective scan is naturally causal (state evolves t-by-t).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Single S6/Mamba block.

    Parameters approximately match the spec's "expand=2" config:
      d_inner = expand * d_model
      d_state = 16 (hidden state dim per channel)
      d_conv = 4 (causal conv kernel)
      dt_rank = ceil(d_model / 16)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None,
                  dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Up-projection to (x, gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise causal conv1d on x_main (groups = d_inner)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True,
        )

        # SSM parameter projections (selective: depend on x)
        # x_proj: from x_main -> (dt_rank, d_state, d_state)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state,
                                  bias=False)
        # dt_proj: dt_rank -> d_inner (the per-channel timestep)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias such that softplus(bias) is in [dt_min, dt_max]
        with torch.no_grad():
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.data.copy_(inv_dt)
            # Mark the bias as not getting reinitialized
            self.dt_proj.bias._no_reinit = True

        # State matrix A (negative): A_log is learned, A = -exp(A_log) ∈ (−inf, 0)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # (d_inner, d_state)
        self.A_log._no_weight_decay = True

        # Skip connection scalar per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):
        """hidden_states: (B, T, D)
        Returns: (B, T, D)."""
        B, T, D = hidden_states.shape

        # Up-projection
        xz = self.in_proj(hidden_states)              # (B, T, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)                     # each (B, T, d_inner)

        # Causal Conv1d
        x = x.transpose(1, 2)                          # (B, d_inner, T)
        x = self.conv1d(x)[:, :, :T]                    # left pad → trim right
        x = x.transpose(1, 2)                          # (B, T, d_inner)
        x = F.silu(x)

        # SSM parameters
        x_dbl = self.x_proj(x)                         # (B, T, dt_rank + 2*d_state)
        dt, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )
        dt = self.dt_proj(dt)                          # (B, T, d_inner)
        dt = F.softplus(dt)                            # ensure positive

        A = -torch.exp(self.A_log.float())             # (d_inner, d_state)

        # Discretize
        # dA: (B, T, d_inner, d_state) = exp(dt[..., None] * A[None, None, ...])
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: (B, T, d_inner, d_state) = dt[..., None] * B[:, :, None, :]
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(2)

        # Selective scan (sequential)
        state = torch.zeros(B, self.d_inner, self.d_state,
                            device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            state = dA[:, t] * state + dB[:, t] * x[:, t, :, None]
            y_t = (state * C_proj[:, t, None, :]).sum(-1)   # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                          # (B, T, d_inner)

        # Skip connection
        y = y + self.D * x

        # Gating
        y = y * F.silu(z)

        # Out-projection
        return self.out_proj(y)
