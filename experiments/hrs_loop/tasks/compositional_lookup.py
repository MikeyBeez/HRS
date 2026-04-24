"""Synthetic multi-hop lookup task.

Each example is a shuffled list of (key → value) pairs, followed by a query.
If `chain_length == k`, the model must follow k−1 intermediate pointers to
reach a terminal value.

Vocab (shared across train and eval):

  tokens 0..99      — "key" tokens (100 distinct labels)
  token 100         — ARROW (→)
  token 101         — SEP (between pairs)
  token 102         — QUERY (before the query key)
  token 103         — ANS (before the answer span)
  token 104         — PAD
  tokens 105..124   — "terminal value" tokens (20 symbols the chain ends at)
  vocab_size        — 125

Example serialization (k=3, arbitrary mapping a→b→c→T):

  a ARROW b SEP b ARROW c SEP c ARROW T SEP ... (distractor pairs) ...
  QUERY a ANS T PAD PAD ...

Sequence length is fixed. Loss is masked so only the single ANS-predicted
token contributes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Vocab layout
N_KEYS = 100
N_TERMS = 20
ARROW_TOK = 100
SEP_TOK = 101
QUERY_TOK = 102
ANS_TOK = 103
PAD_TOK = 104
TERM_START = 105
VOCAB_SIZE = TERM_START + N_TERMS   # = 125


@dataclass
class CompositionalConfig:
    seq_len: int = 256
    n_distractor_pairs: int = 12
    k_values: Tuple[int, ...] = (1, 2, 3, 4)
    # Per-example weight when sampling which k to use during training.
    k_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    seed: int = 0


def _build_chain(k: int, rng: random.Random) -> Tuple[List[int], List[int], int]:
    """Build a chain a_0 → a_1 → ... → a_{k-1} → terminal.

    Returns (chain_keys, pair_sources, pair_targets, terminal).
    chain_keys[0] is the query key a_0.
    """
    keys = rng.sample(range(N_KEYS), k)
    terminal = TERM_START + rng.randrange(N_TERMS)
    pair_srcs = keys                               # k keys
    pair_tgts = keys[1:] + [terminal]              # k targets: k-1 keys + 1 term
    return keys, pair_srcs, pair_tgts, terminal


def make_example(cfg: CompositionalConfig, rng: random.Random, k: int):
    """Return (input_ids, target_ids, answer_pos)."""
    chain_keys, src, tgt, terminal = _build_chain(k, rng)

    # Chain pairs + distractor pairs, all shuffled in presentation.
    used = set(src + [t for t in tgt if t < N_KEYS])
    remaining_keys = [x for x in range(N_KEYS) if x not in used]
    rng.shuffle(remaining_keys)
    d_pairs = []
    d_sources = remaining_keys[:cfg.n_distractor_pairs]
    for ds in d_sources:
        # distractor target is a random other key or a random terminal
        if rng.random() < 0.3:
            dt = TERM_START + rng.randrange(N_TERMS)
        else:
            pool = [x for x in range(N_KEYS) if x != ds and x not in used]
            if not pool:
                dt = TERM_START + rng.randrange(N_TERMS)
            else:
                dt = rng.choice(pool)
        d_pairs.append((ds, dt))

    pairs = list(zip(src, tgt)) + d_pairs
    rng.shuffle(pairs)

    # Serialize: for each pair, (src, ARROW, tgt, SEP)
    tokens: List[int] = []
    for s, t in pairs:
        tokens.extend([s, ARROW_TOK, t, SEP_TOK])
    # Query: QUERY, chain_keys[0], ANS, terminal
    tokens.extend([QUERY_TOK, chain_keys[0], ANS_TOK, terminal])

    # Pad or truncate to seq_len.
    if len(tokens) > cfg.seq_len:
        tokens = tokens[: cfg.seq_len]
    answer_pos = None
    for i, t in enumerate(tokens):
        if t == ANS_TOK:
            answer_pos = i    # the ANS token; model predicts token at pos+1
    while len(tokens) < cfg.seq_len:
        tokens.append(PAD_TOK)

    inp = np.array(tokens, dtype=np.int64)
    # Target is shifted: target[t] is inp[t+1]
    tgt_arr = np.zeros_like(inp)
    tgt_arr[:-1] = inp[1:]
    tgt_arr[-1] = PAD_TOK

    return inp, tgt_arr, answer_pos, terminal


class CompositionalDataset(Dataset):
    def __init__(self, cfg: CompositionalConfig, n_samples: int, k_fixed=None,
                 seed: int = 0):
        self.cfg = cfg
        self.n = n_samples
        self.k_fixed = k_fixed
        self.rng = random.Random(seed)
        # Pre-generate so val sets are deterministic.
        self.samples = []
        k_choices = cfg.k_values if k_fixed is None else (k_fixed,)
        k_weights = cfg.k_weights if k_fixed is None else (1.0,)
        for _ in range(n_samples):
            k = self.rng.choices(k_choices, weights=k_weights, k=1)[0]
            inp, tgt, ans_pos, terminal = make_example(cfg, self.rng, k)
            self.samples.append((inp, tgt, ans_pos, terminal, k))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        inp, tgt, ans_pos, terminal, k = self.samples[idx]
        return (torch.from_numpy(inp), torch.from_numpy(tgt),
                torch.tensor(ans_pos, dtype=torch.long),
                torch.tensor(terminal, dtype=torch.long),
                torch.tensor(k, dtype=torch.long))


def make_loaders(cfg: CompositionalConfig, n_train: int = 20000,
                  n_val_per_k: int = 500,
                  eval_ks: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)) -> dict:
    """Return dict with 'train' loader and 'val_k_X' loaders per k."""
    from torch.utils.data import DataLoader
    tr_ds = CompositionalDataset(cfg, n_train, k_fixed=None, seed=cfg.seed)
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, drop_last=True,
                    num_workers=0)
    val_loaders = {}
    for k in eval_ks:
        val_ds = CompositionalDataset(cfg, n_val_per_k, k_fixed=k,
                                         seed=cfg.seed + 1000 + k)
        val_loaders[k] = DataLoader(val_ds, batch_size=32, shuffle=False,
                                      drop_last=True, num_workers=0)
    return {"train": tr, "val": val_loaders}
