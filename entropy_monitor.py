"""Entropy Monitor: compute and track per-token Shannon entropy.

Provides rolling window entropy monitoring for write (storage) and
read (retrieval) triggers in the entropy-gated engram system.
"""

import torch
import torch.nn.functional as F
from collections import deque


class EntropyMonitor:
    """Compute and track per-token entropy from model output logits.

    Supports:
    - Segment-level mean entropy (for write/store decisions)
    - Rolling window entropy (for read/retrieve triggers during generation)
    """

    def __init__(
        self,
        write_threshold: float = 4.0,
        read_threshold: float = 4.0,
        read_window: int = 10,
    ):
        self.write_threshold = write_threshold
        self.read_threshold = read_threshold
        self.read_window = read_window
        self._rolling_entropies: deque = deque(maxlen=read_window)

    @staticmethod
    def token_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of next-token distribution at each position.

        Args:
            logits: (B, T, V) or (T, V) raw logits from model

        Returns:
            (B, T) or (T,) entropy in bits at each position
        """
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # Shannon entropy in bits: -sum(p * log2(p))
        entropy = -(probs * log_probs).sum(dim=-1) / torch.log(torch.tensor(2.0))
        return entropy

    @staticmethod
    def segment_mean_entropy(logits: torch.Tensor) -> float:
        """Compute mean entropy across all positions in a segment.

        Args:
            logits: (B, T, V) or (T, V) raw logits

        Returns:
            Scalar mean entropy in bits
        """
        ent = EntropyMonitor.token_entropy(logits)
        return ent.mean().item()

    def should_store(self, logits: torch.Tensor) -> tuple[bool, float]:
        """Check if a segment's entropy exceeds the write threshold.

        Args:
            logits: (B, T, V) or (T, V) raw logits for the segment

        Returns:
            (should_store, mean_entropy)
        """
        mean_ent = self.segment_mean_entropy(logits)
        return mean_ent > self.write_threshold, mean_ent

    def update_rolling(self, token_entropy: float):
        """Update the rolling entropy window with a new token's entropy.

        Args:
            token_entropy: entropy in bits for the latest generated token
        """
        self._rolling_entropies.append(token_entropy)

    def should_retrieve(self) -> tuple[bool, float]:
        """Check if rolling window mean entropy exceeds the read threshold.

        Returns:
            (should_retrieve, rolling_mean_entropy)
            Returns (False, 0.0) if window isn't full yet.
        """
        if len(self._rolling_entropies) < self.read_window:
            return False, 0.0
        mean_ent = sum(self._rolling_entropies) / len(self._rolling_entropies)
        return mean_ent > self.read_threshold, mean_ent

    def reset_rolling(self):
        """Reset the rolling entropy window (e.g., at start of new generation)."""
        self._rolling_entropies.clear()
