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
