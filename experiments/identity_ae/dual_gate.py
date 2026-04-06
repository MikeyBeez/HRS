"""Dual Gate: selective LoRA activation via two autoencoders.

Base gate (frozen): recognizes in-distribution content → skip adapter
Novel gate (learned during TTT): recognizes absorbed content → use adapter
Neither recognizes → trigger TTT

The adapter only fires for content it was trained on.
In-distribution content never sees the adapter. Zero contamination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from identity_autoencoder import IdentityAutoencoder


class DualGate(nn.Module):
    """Two-autoencoder gate for selective LoRA activation."""

    def __init__(self, d_model=1024, hidden_dim=768, bottleneck_dim=256,
                 base_threshold=0.241, novel_threshold=0.241):
        super().__init__()
        self.base_gate = IdentityAutoencoder(d_model, hidden_dim, bottleneck_dim)
        self.novel_gate = IdentityAutoencoder(d_model, hidden_dim, bottleneck_dim)
        self.base_threshold = base_threshold
        self.novel_threshold = novel_threshold

        # Stats
        self.counts = {"in_distribution": 0, "learned_novel": 0, "unknown": 0}

    def load_base_gate(self, path):
        """Load the frozen base gate from Phase 0."""
        self.base_gate.load_state_dict(torch.load(path, weights_only=True))
        for p in self.base_gate.parameters():
            p.requires_grad = False

    def classify(self, x):
        """Classify hidden states into one of three categories.

        Args:
            x: (B, T, D) hidden states

        Returns:
            category: "in_distribution", "learned_novel", or "unknown"
            base_error: float
            novel_error: float
        """
        with torch.no_grad():
            _, base_error = self.base_gate(x)
            base_err = base_error.mean().item()

            if base_err < self.base_threshold:
                self.counts["in_distribution"] += 1
                return "in_distribution", base_err, 0.0

            _, novel_error = self.novel_gate(x)
            novel_err = novel_error.mean().item()

            if novel_err < self.novel_threshold:
                self.counts["learned_novel"] += 1
                return "learned_novel", base_err, novel_err

            self.counts["unknown"] += 1
            return "unknown", base_err, novel_err

    def get_stats(self):
        total = sum(self.counts.values())
        return {
            **self.counts,
            "total": total,
            "adapter_activation_rate": (self.counts["learned_novel"]) / max(total, 1),
            "ttt_trigger_rate": self.counts["unknown"] / max(total, 1),
        }

    def reset_stats(self):
        self.counts = {"in_distribution": 0, "learned_novel": 0, "unknown": 0}
