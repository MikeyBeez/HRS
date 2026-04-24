"""Follow-up probe: is L2 H0 a binary passkey-position tagger?

The 6-class linear probe reported 0.455 for L2 H0 vs 0.78–0.82 for its
siblings. The PCA showed L2 H0's values form three tight clusters
(MARKER / all-passkey-positions-merged / other) — consistent with
"passkey vs not" being a one-bit tag rather than a positional encoding.

This script tests that hypothesis directly:
  binary classifier: "is this a passkey position (p1..pK) vs not?"
  3-class classifier: "{MARKER, passkey, other}"

Also reruns the PCA for L2 H3 as a contrast — H3's values encode finer
position information, so its PCA should show p1..p4 separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.diagonal_attention.config import ModelConfig
from experiments.diagonal_attention.model import TinyTransformer
from experiments.head_pruning.attention_probe.probe_l2_values import (
    collect_values,
    install_value_cache,
    pca_scatter,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_CKPT = Path(
    "/mnt/data/Code/HRS/experiments/pruning/checkpoints/mha_passkey.pt"
)


def _build_xy(values, marker_pos, passkey_len, head_idx, class_fn):
    """class_fn: (offset) -> int label (or None to drop). offset ∈ {0 = MARKER,
    1..K = passkey, -1 = other}."""
    N = values.shape[0]
    T = values.shape[2]
    K = passkey_len
    X, Y = [], []
    for i in range(N):
        m = int(marker_pos[i].item())
        for offset, pos in [
            (0, m),
            *((k, m + k) for k in range(1, K + 1)),
            (-1, m - 20), (-1, m + 30),
        ]:
            if pos < 0 or pos >= T:
                continue
            lbl = class_fn(offset)
            if lbl is None:
                continue
            X.append(values[i, head_idx, pos].numpy())
            Y.append(lbl)
    return np.stack(X), np.array(Y)


def logistic_probe(X, Y, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    ntr = int(len(Y) * 0.7)
    tr, te = idx[:ntr], idx[ntr:]
    dh = X.shape[1]
    W = torch.zeros((dh, n_classes), requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.1)
    Xt = torch.tensor(X[tr], dtype=torch.float32)
    Yt = torch.tensor(Y[tr], dtype=torch.long)
    for _ in range(400):
        loss = F.cross_entropy(Xt @ W + b, Yt) + 1e-4 * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_tr = (torch.tensor(X[tr], dtype=torch.float32) @ W + b).argmax(-1).numpy()
        pred_te = (torch.tensor(X[te], dtype=torch.float32) @ W + b).argmax(-1).numpy()
    return {
        "train_acc": float((pred_tr == Y[tr]).mean()),
        "test_acc": float((pred_te == Y[te]).mean()),
        "chance": 1.0 / n_classes,
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--examples", default=str(RESULTS_DIR / "probe_examples.pt"))
    ap.add_argument("--layer", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["mcfg"])
    model = TinyTransformer(mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    install_value_cache(model)

    data = torch.load(args.examples, map_location="cpu", weights_only=False)
    tokens = data["tokens"]
    marker_pos = data["marker_pos"]
    K = data["pkcfg"]["passkey_len"]

    values = collect_values(model, tokens, args.layer)  # [N, H, T, dh]
    H = values.shape[1]

    # Probe A: binary "passkey position vs not" per head.
    def cls_binary(offset):
        if 1 <= offset <= K:
            return 1
        if offset == -1 or offset == 0:
            return 0
        return None

    # Probe B: 3-class "MARKER vs passkey vs other".
    def cls_three(offset):
        if offset == 0:
            return 0
        if 1 <= offset <= K:
            return 1
        if offset == -1:
            return 2
        return None

    results = []
    for h in range(H):
        Xb, Yb = _build_xy(values, marker_pos, K, h, cls_binary)
        r_bin = logistic_probe(Xb, Yb, n_classes=2)
        X3, Y3 = _build_xy(values, marker_pos, K, h, cls_three)
        r_tri = logistic_probe(X3, Y3, n_classes=3)
        results.append({
            "head": h,
            "binary_passkey_vs_other": r_bin,
            "three_way_marker_passkey_other": r_tri,
        })
        print(f"L{args.layer} H{h}  "
              f"binary: test={r_bin['test_acc']:.3f} (chance 0.500)   "
              f"3-way: test={r_tri['test_acc']:.3f} (chance 0.333)")

    # PCA for a sibling head as contrast.
    sibling = 3
    pca_scatter(values, marker_pos, K, sibling,
                 RESULTS_DIR / f"l2_values_pca_H{sibling}.png")

    (RESULTS_DIR / "l2_binary_probe.json").write_text(json.dumps({
        "layer": args.layer,
        "heads": results,
    }, indent=2))
    print(f"wrote {RESULTS_DIR / 'l2_binary_probe.json'}")


if __name__ == "__main__":
    main()
