"""Phase A pre-step: generate 200 passkey probe examples with varied MARKER
position, and record the marker/passkey/query indices per example.

Saves probe_examples.pt with:
  tokens : LongTensor [N, T]
  meta   : dict with per-example lists of marker_pos, passkey_range,
           query_pos, passkey (the digits), plus constants pkcfg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.diagonal_attention.config import PasskeyConfig
from experiments.diagonal_attention.data import make_passkey_sample


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def build_batch(n: int, seed: int = 1234) -> dict:
    pkcfg = PasskeyConfig()
    K = pkcfg.passkey_len
    T = pkcfg.ctx_len
    max_marker = T - 2 * K - 2

    rng = np.random.default_rng(seed)
    tokens = np.empty((n, T), dtype=np.int64)
    marker_pos = np.empty(n, dtype=np.int64)
    passkey_start = np.empty(n, dtype=np.int64)
    passkey_end = np.empty(n, dtype=np.int64)
    query_pos = np.empty(n, dtype=np.int64)
    passkeys = np.empty((n, K), dtype=np.int64)

    # Spread marker positions approximately uniformly across [0, max_marker].
    # We use equally spaced rather than iid so the figure covers the range.
    target_positions = np.linspace(0, max_marker, n).round().astype(np.int64)
    rng.shuffle(target_positions)  # randomize order to decorrelate with rng

    for i in range(n):
        seq, _tgt, mpos = make_passkey_sample(pkcfg, rng,
                                                 marker_pos=int(target_positions[i]))
        tokens[i] = seq
        marker_pos[i] = mpos
        passkey_start[i] = mpos + 1
        passkey_end[i] = mpos + K  # inclusive
        query_pos[i] = T - K - 1
        passkeys[i] = seq[mpos + 1 : mpos + 1 + K]

    out = {
        "tokens": torch.from_numpy(tokens),
        "marker_pos": torch.from_numpy(marker_pos),
        "passkey_start": torch.from_numpy(passkey_start),
        "passkey_end": torch.from_numpy(passkey_end),
        "query_pos": torch.from_numpy(query_pos),
        "passkeys": torch.from_numpy(passkeys),
        "pkcfg": {
            "passkey_len": pkcfg.passkey_len,
            "ctx_len": pkcfg.ctx_len,
            "digit_vocab": pkcfg.digit_vocab,
            "marker_tok": pkcfg.marker_tok,
            "query_tok": pkcfg.query_tok,
            "pad_tok": pkcfg.pad_tok,
            "vocab_size": pkcfg.vocab_size,
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=str(RESULTS_DIR / "probe_examples.pt"))
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = build_batch(args.n, args.seed)
    torch.save(data, args.out)
    print(f"wrote {args.out}: {args.n} examples, "
          f"marker_pos range [{int(data['marker_pos'].min())}, "
          f"{int(data['marker_pos'].max())}]")


if __name__ == "__main__":
    main()
