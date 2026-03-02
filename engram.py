"""Engram encoder/decoder for HRS.

Compresses windows of hidden states into compact engram vectors that
can be prepended to context as additional KV entries. Achieves 32x
compression (128 tokens -> 4 engrams).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EngramConfig, ModelConfig


class EngramEncoder(nn.Module):
    """Compress a window of hidden states into engram vectors.

    Architecture: mean-pool within window, then 2-layer MLP to engram vectors.
    Input: (B, W, d_model) window of W hidden states
    Output: (B, K, d_model) K engram vectors
    """

    def __init__(self, model_cfg: ModelConfig, engram_cfg: EngramConfig):
        super().__init__()
        self.window_size = engram_cfg.window_size  # N = 128
        self.n_engrams = engram_cfg.n_engrams      # K = 4
        d = model_cfg.d_model

        # 2-layer MLP: d_model -> 2*d_model -> K * d_model
        self.encoder = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, self.n_engrams * d),
        )

        self.norm = nn.LayerNorm(d)
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of hidden states into engrams.

        Args:
            hidden_states: (B, T, d_model) full sequence hidden states

        Returns:
            (B, n_engrams_total, d_model) engram vectors for all windows
        """
        B, T, D = hidden_states.shape
        W = self.window_size
        K = self.n_engrams

        # Number of complete windows
        n_windows = T // W
        if n_windows == 0:
            # Sequence too short for engrams, return empty
            return torch.zeros(B, 0, D, device=hidden_states.device, dtype=hidden_states.dtype)

        # Reshape into windows: (B, n_windows, W, D)
        usable = n_windows * W
        windows = hidden_states[:, :usable].reshape(B, n_windows, W, D)

        # Mean-pool within each window: (B, n_windows, D)
        pooled = windows.mean(dim=2)

        # Encode each pooled window into K engrams: (B, n_windows, K * D)
        encoded = self.encoder(pooled)

        # Reshape: (B, n_windows * K, D)
        engrams = encoded.reshape(B, n_windows * K, D)
        engrams = self.norm(engrams)

        return engrams


class EngramInjector(nn.Module):
    """Prepend engrams to context as additional KV entries.

    During forward pass, engram vectors are concatenated before
    the regular token embeddings so attention can attend to them.
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        # Learnable type embedding to distinguish engrams from tokens
        self.engram_type_emb = nn.Parameter(torch.zeros(1, 1, model_cfg.d_model))
        nn.init.normal_(self.engram_type_emb, std=0.02)

    def forward(
        self, x: torch.Tensor, engrams: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Prepend engrams to token sequence.

        Args:
            x: (B, T, d_model) token hidden states
            engrams: (B, E, d_model) engram vectors

        Returns:
            (B, E+T, d_model) combined sequence, number of engrams prepended
        """
        if engrams.shape[1] == 0:
            return x, 0

        # Add type embedding to engrams
        engrams = engrams + self.engram_type_emb

        # Concatenate: engrams first, then tokens
        combined = torch.cat([engrams, x], dim=1)
        return combined, engrams.shape[1]


class EngramUpsampler(nn.Module):
    """Expand K engrams back to W positions for in-place replacement.

    Each engram covers W/K = 32 positions via repeat_interleave,
    then a learnable positional embedding + refinement MLP restores
    per-position detail.

    Input: (B, n_windows, K, d_model)
    Output: (B, n_windows, W, d_model)
    """

    def __init__(self, model_cfg: ModelConfig, engram_cfg: EngramConfig):
        super().__init__()
        self.window_size = engram_cfg.window_size  # W = 128
        self.n_engrams = engram_cfg.n_engrams      # K = 4
        self.positions_per_engram = self.window_size // self.n_engrams  # 32
        d = model_cfg.d_model

        # Learnable within-window positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, self.window_size, d))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Refinement MLP with residual connection
        self.refine = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)
        self._init_weights()

    def _init_weights(self):
        for m in self.refine:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, engrams: torch.Tensor) -> torch.Tensor:
        """Upsample engrams to window-sized representations.

        Args:
            engrams: (B, n_windows, K, d_model)

        Returns:
            (B, n_windows, W, d_model)
        """
        # Repeat each engram to cover its positions: (B, n_windows, W, D)
        upsampled = engrams.repeat_interleave(self.positions_per_engram, dim=2)

        # Add positional embedding
        upsampled = upsampled + self.pos_emb

        # Refinement with residual
        upsampled = upsampled + self.refine(upsampled)
        upsampled = self.norm(upsampled)

        return upsampled


class RememberGate(nn.Module):
    """Learned per-window gate that modulates engram strength before prepending.

    Computes 4 per-window features:
    1. Normalized per-window loss (z-scored across windows within batch)
    2. Cosine similarity between mean window hidden state and mean window engram
    3. Window hidden state L2 norm (divided by sqrt(D))
    4. Engram L2 norm (divided by sqrt(D))

    MLP: Linear(4, hidden) + GELU + Linear(hidden, 1) -> sigmoid
    Output: (B, n_windows) gate values in [0, 1]
    """

    def __init__(self, engram_cfg: EngramConfig):
        super().__init__()
        self.window_size = engram_cfg.window_size
        self.n_engrams = engram_cfg.n_engrams
        hidden = engram_cfg.gate_hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights(engram_cfg.gate_bias_init)

    def _init_weights(self, bias_init: float):
        # Zero all weights so gate starts as constant sigmoid(bias_init)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Set final bias so sigmoid(bias_init) ≈ 0.88
        self.mlp[-1].bias.data.fill_(bias_init)

    def forward(
        self,
        x: torch.Tensor,
        engrams: torch.Tensor,
        per_token_loss: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute per-window gate values.

        Args:
            x: (B, T, D) token hidden states
            engrams: (B, E, D) engram vectors (E = n_windows * K)
            per_token_loss: (B, T) cached loss from previous step, or None

        Returns:
            (B, n_windows) gate values in [0, 1]
        """
        B, T, D = x.shape
        W = self.window_size
        K = self.n_engrams
        n_windows = T // W

        if n_windows == 0 or engrams.shape[1] == 0:
            return torch.ones(B, max(n_windows, 1), device=x.device)

        usable = n_windows * W

        # Reshape tokens into windows: (B, n_windows, W, D)
        x_windows = x[:, :usable].reshape(B, n_windows, W, D)
        # Mean-pool per window: (B, n_windows, D)
        x_mean = x_windows.mean(dim=2)

        # Reshape engrams into windows: (B, n_windows, K, D)
        engram_windows = engrams[:, :n_windows * K].reshape(B, n_windows, K, D)
        # Mean-pool per window: (B, n_windows, D)
        engram_mean = engram_windows.mean(dim=2)

        # Feature 1: Normalized per-window loss (z-scored)
        if per_token_loss is not None:
            loss_windows = per_token_loss[:, :usable].reshape(B, n_windows, W)
            window_loss = loss_windows.mean(dim=2)  # (B, n_windows)
            # Z-score across windows within each batch element
            loss_std = window_loss.std(dim=1, keepdim=True).clamp(min=1e-6)
            loss_mean = window_loss.mean(dim=1, keepdim=True)
            feat_loss = (window_loss - loss_mean) / loss_std  # (B, n_windows)
        else:
            feat_loss = torch.zeros(B, n_windows, device=x.device)

        # Feature 2: Cosine similarity between window hidden and engram
        feat_cos = F.cosine_similarity(x_mean, engram_mean, dim=-1)  # (B, n_windows)

        # Feature 3: Window hidden state L2 norm / sqrt(D)
        inv_sqrt_d = 1.0 / (D ** 0.5)
        feat_x_norm = x_mean.norm(dim=-1) * inv_sqrt_d  # (B, n_windows)

        # Feature 4: Engram L2 norm / sqrt(D)
        feat_e_norm = engram_mean.norm(dim=-1) * inv_sqrt_d  # (B, n_windows)

        # Stack features: (B, n_windows, 4)
        features = torch.stack([feat_loss, feat_cos, feat_x_norm, feat_e_norm], dim=-1)

        # MLP -> sigmoid: (B, n_windows, 1) -> (B, n_windows)
        gate = torch.sigmoid(self.mlp(features)).squeeze(-1)

        return gate


class GatedEngramInjector(nn.Module):
    """Prepend engrams scaled by a learned per-window remember gate.

    Gate ≈ 1: engrams fully active (v4 behavior)
    Gate ≈ 0: engrams zeroed out
    Strictly additive — no information is destroyed.
    """

    def __init__(self, model_cfg: ModelConfig, engram_cfg: EngramConfig):
        super().__init__()
        self.n_engrams = engram_cfg.n_engrams
        self.remember_gate = RememberGate(engram_cfg)
        self.engram_type_emb = nn.Parameter(torch.zeros(1, 1, model_cfg.d_model))
        nn.init.normal_(self.engram_type_emb, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        engrams: torch.Tensor,
        per_token_loss: torch.Tensor = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Prepend gated engrams to token sequence.

        Args:
            x: (B, T, D) token hidden states
            engrams: (B, E, D) engram vectors
            per_token_loss: (B, T) cached loss or None

        Returns:
            combined: (B, E+T, D) with gated engrams prepended
            n_engrams: number of engrams prepended
            gate_values: (B, n_windows) gate values for logging
        """
        if engrams.shape[1] == 0:
            return x, 0, torch.ones(x.shape[0], 1, device=x.device)

        # Compute per-window gate: (B, n_windows)
        gate_values = self.remember_gate(x, engrams, per_token_loss)

        # Expand gate to per-engram: (B, E, 1)
        K = self.n_engrams
        gate_expanded = gate_values.repeat_interleave(K, dim=1).unsqueeze(-1)  # (B, E, 1)

        # Scale engrams
        scaled_engrams = (engrams + self.engram_type_emb) * gate_expanded

        # Prepend
        combined = torch.cat([scaled_engrams, x], dim=1)
        return combined, engrams.shape[1], gate_values


class EngramReplacer(nn.Module):
    """Soft gating for in-place engram replacement based on per-token loss.

    When a window has high loss (stale/OOD context), gate → 1 and
    upsampled engrams replace the original tokens. When loss is low
    (useful context), gate → 0 and originals are kept.

    Uses previous step's EMA-cached loss to avoid chicken-and-egg.
    """

    def __init__(self, model_cfg: ModelConfig, engram_cfg: EngramConfig):
        super().__init__()
        self.window_size = engram_cfg.window_size  # W = 128
        self.n_engrams = engram_cfg.n_engrams      # K = 4
        d = model_cfg.d_model

        # Learnable gate parameters
        self.threshold = nn.Parameter(torch.tensor(engram_cfg.gate_init_threshold))
        self.sharpness = nn.Parameter(torch.tensor(engram_cfg.gate_sharpness_init))

        # Type embedding for replaced positions
        self.replace_type_emb = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.replace_type_emb, std=0.02)

        # Upsampler
        self.upsampler = EngramUpsampler(model_cfg, engram_cfg)

    def forward(
        self,
        x: torch.Tensor,
        engrams: torch.Tensor,
        per_token_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend upsampled engrams into token sequence based on loss gate.

        Args:
            x: (B, T, d_model) token hidden states
            engrams: (B, n_engrams_total, d_model) engram vectors
            per_token_loss: (B, T) cached loss from previous step, or None

        Returns:
            blended_x: (B, T, d_model) with stale windows replaced
            gate_values: (B, n_windows) soft gate per window
        """
        B, T, D = x.shape
        W = self.window_size
        K = self.n_engrams
        n_windows = T // W

        if per_token_loss is None or engrams.shape[1] == 0 or n_windows == 0:
            return x, torch.zeros(B, max(n_windows, 1), device=x.device)

        # Compute per-window mean loss: (B, n_windows)
        usable = n_windows * W
        loss_windows = per_token_loss[:, :usable].reshape(B, n_windows, W)
        window_loss = loss_windows.mean(dim=2)  # (B, n_windows)

        # Soft gate via sigmoid: high loss -> gate ≈ 1 (replace)
        gate = torch.sigmoid(self.sharpness * (window_loss - self.threshold))  # (B, n_windows)

        # Reshape engrams into windows: (B, n_windows, K, D)
        engram_windows = engrams[:, :n_windows * K].reshape(B, n_windows, K, D)

        # Upsample: (B, n_windows, W, D)
        expanded = self.upsampler(engram_windows)

        # Add type embedding to replacement positions
        expanded = expanded + self.replace_type_emb

        # Blend: gate expands to (B, n_windows, W, 1) for broadcasting
        gate_expanded = gate.unsqueeze(2).unsqueeze(3).expand(B, n_windows, W, 1)

        # Reshape original tokens into windows
        x_windows = x[:, :usable].reshape(B, n_windows, W, D)

        # Soft blend
        blended = gate_expanded * expanded + (1.0 - gate_expanded) * x_windows

        # Reconstruct full sequence
        out = x.clone()
        out[:, :usable] = blended.reshape(B, usable, D)

        return out, gate


def engram_reconstruction_loss(
    original_hidden: torch.Tensor,
    engrams: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Measure how well engrams preserve information from original windows.

    Uses cosine similarity between engram reconstructions and original
    mean-pooled windows.

    Args:
        original_hidden: (B, T, D) original hidden states
        engrams: (B, n_engrams_total, D) compressed engrams
        window_size: size of compression windows

    Returns:
        Scalar reconstruction loss (lower = better preservation)
    """
    B, T, D = original_hidden.shape
    n_windows = T // window_size

    if n_windows == 0 or engrams.shape[1] == 0:
        return torch.tensor(0.0, device=original_hidden.device)

    usable = n_windows * window_size
    # Original mean-pooled windows: (B, n_windows, D)
    windows = original_hidden[:, :usable].reshape(B, n_windows, window_size, D)
    pooled = windows.mean(dim=2)

    # Engrams per window (K engrams per window, average them for comparison)
    K = engrams.shape[1] // n_windows
    engram_windows = engrams[:, :n_windows * K].reshape(B, n_windows, K, D)
    engram_pooled = engram_windows.mean(dim=2)  # (B, n_windows, D)

    # Cosine similarity loss (1 - similarity)
    similarity = F.cosine_similarity(pooled, engram_pooled, dim=-1)  # (B, n_windows)
    loss = (1.0 - similarity).mean()

    return loss
