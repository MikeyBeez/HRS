"""Shared config for the diagonal-attention ablation."""
from dataclasses import dataclass, field
from typing import List


VARIANTS: List[str] = ["mha", "bilinear", "diagonal", "identity"]


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    ctx_len: int = 256
    dropout: float = 0.0
    vocab_size: int = -1  # set by caller
    variant: str = "mha"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_every: int = 200
    seed: int = 0


@dataclass
class PasskeyConfig:
    """Synthetic passkey retrieval task."""
    passkey_len: int = 4           # digits in the passkey
    ctx_len: int = 256             # must match model ctx
    digit_vocab: int = 10
    # Special tokens appended after the 10 digits:
    # 10 = MARKER, 11 = QUERY, 12 = PAD
    marker_tok: int = 10
    query_tok: int = 11
    pad_tok: int = 12
    vocab_size: int = 13
    # Positions to bucket the passkey into during eval.
    # "position" = index of the MARKER token.
    eval_buckets: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_eval_per_bucket: int = 50
