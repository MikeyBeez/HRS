"""Tiny Shakespeare loader + synthetic passkey generator."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from experiments.diagonal_attention.config import PasskeyConfig


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def _tiny_shakespeare_path() -> Path:
    # Store under the project datasets/ dir (which is gitignored).
    root = Path(__file__).resolve().parents[2]
    p = root / "datasets" / "tiny_shakespeare.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        print(f"downloading tiny shakespeare -> {p}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, p)
    return p


def load_shakespeare(val_frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Char-level Tiny Shakespeare.

    Returns (train_ids, val_ids, info) where info has 'vocab_size', 'stoi', 'itos'.
    """
    text = _tiny_shakespeare_path().read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n_val = int(len(data) * val_frac)
    train = data[:-n_val]
    val = data[-n_val:]
    return train, val, {"vocab_size": len(chars), "stoi": stoi, "itos": itos}


def sample_lm_batch(
    data: np.ndarray,
    batch_size: int,
    ctx_len: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a (x, y) LM batch from a flat token array."""
    ix = rng.integers(0, len(data) - ctx_len - 1, size=batch_size)
    x = np.stack([data[i : i + ctx_len] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + ctx_len] for i in ix])
    return (
        torch.from_numpy(x).to(device, non_blocking=True),
        torch.from_numpy(y).to(device, non_blocking=True),
    )


# ---- Synthetic passkey task -------------------------------------------------

def make_passkey_sample(
    cfg: PasskeyConfig,
    rng: np.random.Generator,
    marker_pos: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """One passkey sample.

    Layout (length cfg.ctx_len):
        [filler ...][MARKER][p1..pK][filler ...][QUERY][p1..pK]

    The last K positions hold the answer — loss is computed only there.
    Returns (input_ids, target_ids, marker_pos).

    input_ids and target_ids are length ctx_len; target is shifted by 1
    so predicting input[t] gives a loss at target[t].

    marker_pos is the index of MARKER in input_ids, for eval bucketing.
    """
    K = cfg.passkey_len
    T = cfg.ctx_len

    # We need room for MARKER + K passkey digits + QUERY + K answer digits.
    # The sequence input is length T; the last K positions = answer.
    # MARKER must sit at [0, T - 2*K - 2] inclusive so the passkey and
    # QUERY/answer fit between it and the end.
    max_marker = T - 2 * K - 2
    assert max_marker >= 1, "ctx_len too small for passkey task"

    if marker_pos is None:
        marker_pos = int(rng.integers(0, max_marker))
    else:
        marker_pos = min(max(marker_pos, 0), max_marker)

    passkey = rng.integers(0, cfg.digit_vocab, size=K).astype(np.int64)

    # Fill with random digits as noise.
    seq = rng.integers(0, cfg.digit_vocab, size=T).astype(np.int64)

    # Place MARKER and passkey.
    seq[marker_pos] = cfg.marker_tok
    seq[marker_pos + 1 : marker_pos + 1 + K] = passkey

    # Place QUERY and answer in the final K+1 slots.
    query_idx = T - K - 1
    seq[query_idx] = cfg.query_tok
    seq[query_idx + 1 : query_idx + 1 + K] = passkey

    # Targets: next-token. Only the answer span contributes to loss — the
    # training loop masks everything else. We still build a full next-token
    # array for convenience.
    tgt = np.zeros(T, dtype=np.int64)
    tgt[:-1] = seq[1:]
    tgt[-1] = cfg.pad_tok  # never used; masked out

    return seq, tgt, marker_pos


def loss_mask_for_answer(cfg: PasskeyConfig, device: torch.device) -> torch.Tensor:
    """Mask that is 1 on positions where we want to score the answer.

    For input index t, the model predicts target t (= input t+1). We want
    the prediction at the K positions just before the final token: these
    generate the K answer tokens (QUERY then p1, p1 then p2, ...).

    Specifically: positions query_idx .. query_idx + K - 1 in the input
    are asked to predict p1, p2, ..., pK respectively.
    """
    K = cfg.passkey_len
    T = cfg.ctx_len
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    query_idx = T - K - 1
    mask[query_idx : query_idx + K] = True
    return mask


def sample_passkey_batch(
    cfg: PasskeyConfig,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for _ in range(batch_size):
        x, y, _ = make_passkey_sample(cfg, rng)
        xs.append(x)
        ys.append(y)
    x_t = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)
    y_t = torch.from_numpy(np.stack(ys)).to(device, non_blocking=True)
    return x_t, y_t


def make_passkey_eval_set(cfg: PasskeyConfig, rng: np.random.Generator):
    """Build a fixed eval set: n_eval_per_bucket samples per position bucket.

    Returns a list of (input_ids, passkey, bucket_frac) tuples (np arrays).
    """
    K = cfg.passkey_len
    T = cfg.ctx_len
    max_marker = T - 2 * K - 2
    out = []
    for frac in cfg.eval_buckets:
        target_pos = int(round(frac * max_marker))
        for _ in range(cfg.n_eval_per_bucket):
            seq, _tgt, mpos = make_passkey_sample(cfg, rng, marker_pos=target_pos)
            passkey = seq[mpos + 1 : mpos + 1 + K].copy()
            out.append((seq, passkey, frac))
    return out
