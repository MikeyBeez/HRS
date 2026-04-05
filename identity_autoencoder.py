"""Identity Autoencoder: test-time learning via reconstruction error.

In-distribution inputs pass through unchanged (skip connection).
Out-of-distribution inputs produce high reconstruction error,
triggering test-time training before the signal proceeds.

The autoencoder's weight delta (current - init) IS the engram:
a compressed representation of everything novel encountered.

Architecture: 1024 → 768 → 256 → 768 → 1024 (~2M params)
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityAutoencoder(nn.Module):
    """Small autoencoder with identity objective.

    When reconstruction is good: acts as skip connection (invisible).
    When reconstruction is poor: signals OOD input.

    Args:
        d_model: transformer hidden dimension (1024 for V22)
        hidden_dim: encoder/decoder hidden width (768)
        bottleneck_dim: compression dimension (256)
    """

    def __init__(self, d_model=1024, hidden_dim=768, bottleneck_dim=256):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder: d_model → hidden → bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Decoder: bottleneck → hidden → d_model
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Gate: controls how much the autoencoder path contributes
        # At init: gate ≈ 0, so output = input + 0 = identity
        self.gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

        self._init_near_zero()

    def _init_near_zero(self):
        """Initialize encoder/decoder with very small weights.

        Combined with the skip connection and near-zero gate,
        the autoencoder starts as perfect identity.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass with skip connection.

        The autoencoder learns: decoder(encoder(x)) ≈ x
        The output uses a skip: output = (1 - g) * x + g * decoder(encoder(x))

        At init: g ≈ 0 → output = x (perfect identity, lossless splice)
        After training: g grows → output = decoder(encoder(x)) ≈ x (learned identity)
        OOD detection: ||decoder(encoder(x)) - x||² is high when bottleneck can't compress

        Args:
            x: (B, T, D) hidden states from transformer layer

        Returns:
            output: (B, T, D) blended reconstruction
            error: (B, T) per-token MSE reconstruction error through bottleneck
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Skip-blended output
        g = torch.sigmoid(self.gate)
        output = (1 - g) * x + g * decoded

        # Reconstruction error: how well does the bottleneck round-trip?
        # This is the OOD signal — high error = can't compress = novel input
        error = ((decoded - x) ** 2).mean(dim=-1)  # (B, T)

        return output, error

    def encode(self, x):
        """Get bottleneck representation only."""
        return self.encoder(x)

    def reconstruction_error(self, x):
        """Compute reconstruction error without returning reconstruction."""
        with torch.no_grad():
            _, error = self.forward(x)
        return error

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class OODDetector:
    """Detect out-of-distribution inputs and train until learned.

    When reconstruction error exceeds threshold, trains the autoencoder
    on the input until error drops below threshold or max steps reached.
    """

    def __init__(self, autoencoder, threshold=0.01, max_train_steps=30,
                 lr=1e-4):
        self.autoencoder = autoencoder
        self.threshold = threshold
        self.max_train_steps = max_train_steps
        self.lr = lr
        self.optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

        # Stats
        self.total_ood_steps = 0
        self.total_checks = 0
        self.total_ood_detections = 0

    def check_and_train(self, x):
        """Check if input is OOD and train if necessary.

        Args:
            x: (B, T, D) hidden states (DETACHED from main graph)

        Returns:
            output: (B, T, D) reconstructed hidden states
            steps_taken: int, number of training steps (0 = in-distribution)
            mean_error: float, final reconstruction error
        """
        x_input = x.detach().requires_grad_(False)
        self.total_checks += 1

        # Check reconstruction error
        with torch.no_grad():
            recon, error = self.autoencoder(x_input)
            mean_error = error.mean().item()

        if mean_error < self.threshold:
            # In-distribution: pass through
            return recon.detach(), 0, mean_error

        # Out-of-distribution: train until learned
        # Must enable gradients for TTT even if caller is in no_grad context
        self.total_ood_detections += 1
        self.autoencoder.train()
        steps_taken = 0

        with torch.enable_grad():
            for step in range(self.max_train_steps):
                encoded = self.autoencoder.encoder(x_input)
                decoded = self.autoencoder.decoder(encoded)
                loss = F.mse_loss(decoded, x_input)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                steps_taken += 1
                self.total_ood_steps += 1

                if loss.item() < self.threshold:
                    break

        # Final pass with learned representation
        self.autoencoder.eval()
        with torch.no_grad():
            recon, error = self.autoencoder(x_input)
            mean_error = error.mean().item()

        return recon.detach(), steps_taken, mean_error

    def get_stats(self):
        return {
            "total_checks": self.total_checks,
            "total_ood_detections": self.total_ood_detections,
            "total_ood_steps": self.total_ood_steps,
            "ood_rate": self.total_ood_detections / max(self.total_checks, 1),
            "avg_steps_per_ood": self.total_ood_steps / max(self.total_ood_detections, 1),
        }


class EngramRecurrence:
    """Manage engram state for RNN-like recurrence.

    The pipeline output at the final layer becomes context for the next step,
    replacing the KV cache with a fixed-size engram.
    """

    def __init__(self, update_method="replace", ema_alpha=0.9):
        self.engram_state = None
        self.update_method = update_method
        self.ema_alpha = ema_alpha
        self.step_count = 0

    def update(self, pipeline_output, max_len=64):
        """Update engram with pipeline output.

        Args:
            pipeline_output: (B, T, D) final layer hidden states
            max_len: max engram sequence length to keep
        """
        new_state = pipeline_output[:, -max_len:, :].detach()
        self.step_count += 1

        if self.update_method == "replace":
            self.engram_state = new_state
        elif self.update_method == "ema":
            if self.engram_state is not None and self.engram_state.shape[1] == new_state.shape[1]:
                self.engram_state = (
                    self.ema_alpha * self.engram_state +
                    (1 - self.ema_alpha) * new_state
                )
            else:
                self.engram_state = new_state
        else:
            self.engram_state = new_state

    def get_context(self):
        """Return current engram for use as context."""
        return self.engram_state

    def reset(self):
        """Clear engram state."""
        self.engram_state = None
        self.step_count = 0


class EngramLibrary:
    """Store and retrieve named engram deltas.

    Each delta is the weight change from init, caused by absorbing
    specific information through test-time training.
    """

    def __init__(self, init_weights_path):
        self.init_weights = torch.load(init_weights_path, weights_only=True)
        self.library = {}

    def capture(self, name, autoencoder):
        """Capture current memory state as a named engram."""
        delta = {}
        current = autoencoder.state_dict()
        for key in current:
            delta[key] = (current[key] - self.init_weights[key]).cpu()
        self.library[name] = delta

    def apply(self, autoencoder, *names):
        """Apply one or more named engrams to the base autoencoder."""
        combined_delta = {}
        for key in self.init_weights:
            combined_delta[key] = torch.zeros_like(self.init_weights[key])
            for name in names:
                combined_delta[key] = combined_delta[key] + self.library[name][key].to(
                    self.init_weights[key].device)
            combined_delta[key] = combined_delta[key] / len(names)

        new_state = {}
        for key in self.init_weights:
            new_state[key] = self.init_weights[key] + combined_delta[key]
        autoencoder.load_state_dict(new_state)

    def save_engram(self, name, path):
        torch.save(self.library[name], path)

    def load_engram(self, name, path):
        self.library[name] = torch.load(path, weights_only=True)

    def delta_stats(self, name):
        """Statistics about a named engram's weight delta."""
        delta = self.library[name]
        total_params = sum(d.numel() for d in delta.values())
        nonzero = sum((d.abs() > 1e-6).sum().item() for d in delta.values())
        max_delta = max(d.abs().max().item() for d in delta.values())
        return {
            "total_params": total_params,
            "nonzero_params": nonzero,
            "sparsity": 1 - nonzero / total_params,
            "max_delta": max_delta,
        }
