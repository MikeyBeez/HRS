"""Probe L2 H0 (and the other L2 heads for contrast) value projections.

We previously showed L0 H2/H3 do "find MARKER" and L3 heads do "read passkey,"
but L2 H0 is retrieval-critical without a clean attention-pattern signature.
Its role must live in the value pathway: W_V @ LN(residual) computes some
function of each position, and that function presumably tags passkey-
adjacent positions with retrieval-relevant info that L3 then reads.

This script:
  1. Runs the baseline passkey model on 200 probe examples.
  2. Captures W_V output (value vectors per head) at every position.
     V has shape [N, n_layers, n_heads, T, d_head]; we focus on layer 2.
  3. Analyzes L2 H0 (and neighbors for contrast):
       - Magnitude per aligned position (MARKER at center).
       - PCA of values at MARKER and at passkey p1..pK positions.
       - Linear probe: can we predict "MARKER-relative offset"
         ∈ {MARKER, p1, p2, p3, p4, other} from the value vector?
         A head whose value encodes offset-from-MARKER should give
         high accuracy; a generic head should not.

Produces `fig_l2_values.png` and `l2_value_probe.json`.
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


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_CKPT = Path(
    "/mnt/data/Code/HRS/experiments/pruning/checkpoints/mha_passkey.pt"
)


def install_value_cache(model: TinyTransformer):
    """Patch each Block.attn.forward to also cache V (per-head value vectors).

    V has shape (B, H, T, d_head). Cached on the attn module as `_V`.
    """
    for blk in model.blocks:
        attn_module = blk.attn

        def make_patched(module):
            def patched(x):
                B, T, d = x.shape
                H = module.cfg.n_heads
                dh = module.cfg.d_head
                scores = module.compute_scores(x)
                causal = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal, float("-inf"))
                attn = F.softmax(scores, dim=-1)

                V = module.W_V(x).view(B, T, H, dh).transpose(1, 2)  # (B,H,T,dh)
                module._V = V.detach()

                if attn.dim() == 3:
                    out = torch.einsum("bts,bhsd->bhtd", attn, V)
                else:
                    out = torch.einsum("bhts,bhsd->bhtd", attn, V)
                out = out.transpose(1, 2).contiguous().view(B, T, d)
                return module.W_O(out)
            return patched

        attn_module.forward = make_patched(attn_module)


@torch.no_grad()
def collect_values(model, tokens: torch.Tensor, layer_idx: int,
                     batch_size: int = 32) -> torch.Tensor:
    """Return [N, H, T, d_head] of values for the given layer."""
    N, T = tokens.shape
    H = model.cfg.n_heads
    dh = model.cfg.d_head
    device = next(model.parameters()).device
    out = torch.empty((N, H, T, dh), dtype=torch.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = tokens[start:end].to(device)
        _ = model(x)
        V = model.blocks[layer_idx].attn._V.to(torch.float32).cpu()  # (b, H, T, dh)
        out[start:end] = V
    return out


def align_to_marker(values: torch.Tensor, marker_pos: torch.Tensor) -> torch.Tensor:
    """values [N, *, T, dh]; align per example so MARKER is at T/2."""
    N = values.shape[0]
    T = values.shape[-2]
    center = T // 2
    aligned = torch.empty_like(values)
    for i in range(N):
        shift = int(center - marker_pos[i].item())
        aligned[i] = values[i].roll(shifts=shift, dims=-2)
    return aligned


def magnitude_plot(values: torch.Tensor, marker_pos: torch.Tensor,
                    passkey_len: int, out_path: Path):
    """For each of the 4 L2 heads, plot ||V[t]|| (mean across examples),
    aligned to MARKER center."""
    H = values.shape[1]
    T = values.shape[2]
    aligned = align_to_marker(values, marker_pos)              # [N, H, T, dh]
    norms = aligned.norm(dim=-1)                                # [N, H, T]
    mean_norm = norms.mean(0).numpy()                           # [H, T]
    center = T // 2

    fig, axes = plt.subplots(1, H, figsize=(3.5 * H, 3.5), sharey=True)
    for h in range(H):
        ax = axes[h]
        ax.plot(mean_norm[h], lw=1.0)
        ax.axvline(center, color="#00a000", lw=0.8, label="MARKER")
        ax.axvspan(center + 1, center + 1 + passkey_len,
                    color="#0030a0", alpha=0.15, label="passkey")
        ax.set_title(f"L2 H{h}  value-vector magnitude")
        ax.set_xlabel("aligned position")
        if h == 0:
            ax.set_ylabel("mean ‖V[t]‖₂")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    return mean_norm


def pca_scatter(values: torch.Tensor, marker_pos: torch.Tensor,
                  passkey_len: int, head_idx: int, out_path: Path):
    """PCA on L2 Hh value vectors at {MARKER, p1..pK} positions across examples,
    plotted in first 2 PCs colored by offset-from-MARKER."""
    N = values.shape[0]
    T = values.shape[2]
    dh = values.shape[3]
    K = passkey_len

    # Collect: for each example, values at MARKER + p1..pK + a few controls.
    vecs = []
    labels = []
    for i in range(N):
        m = int(marker_pos[i].item())
        vecs.append(values[i, head_idx, m].numpy())
        labels.append(0)  # MARKER
        for k in range(1, K + 1):
            vecs.append(values[i, head_idx, m + k].numpy())
            labels.append(k)
        # Control: two tokens well before MARKER (if possible) and well after passkey.
        for ctrl in (-20, 30):
            pos = m + ctrl
            if 0 <= pos < T and not (m <= pos <= m + K):
                vecs.append(values[i, head_idx, pos].numpy())
                labels.append(-1)  # other
    V = np.stack(vecs)
    L = np.array(labels)

    # PCA via SVD.
    Vc = V - V.mean(0)
    U, s, Vt = np.linalg.svd(Vc, full_matrices=False)
    pcs = Vc @ Vt[:2].T  # [n, 2]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    color_map = {0: "#00a000", -1: "#777777"}
    for k in range(1, K + 1):
        color_map[k] = plt.cm.tab10(k)

    # Plot in reverse (controls first so foreground stays visible).
    for cls in [-1, 0] + list(range(1, K + 1)):
        mask = L == cls
        if mask.sum() == 0:
            continue
        label = ("MARKER" if cls == 0
                 else f"p{cls}" if cls > 0
                 else "other")
        ax.scatter(pcs[mask, 0], pcs[mask, 1],
                    c=[color_map[cls]], alpha=0.55, s=14, label=label)
    ax.set_xlabel(f"PC1 (σ={s[0]:.2f})")
    ax.set_ylabel(f"PC2 (σ={s[1]:.2f})")
    ax.set_title(f"L2 H{head_idx}: W_V outputs at MARKER, passkey, control positions")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    return {
        "explained_variance": (s ** 2 / (s ** 2).sum()).tolist(),
        "singular_values": s.tolist(),
    }


def offset_probe(values: torch.Tensor, marker_pos: torch.Tensor,
                  passkey_len: int, max_head: int):
    """For each head, train a linear multiclass classifier on the value
    vector at a position, predicting the offset-from-MARKER label in
    {MARKER, p1, .., pK, other}. Report train and test accuracy.

    This is an "information-in-the-value" probe: a generic head should
    not let a linear classifier separate these positions; a tagging head
    should."""
    import torch.nn.functional as FF

    N = values.shape[0]
    T = values.shape[2]
    dh = values.shape[3]
    K = passkey_len

    # Build (x, y) pairs for every head.
    per_head = []
    for h in range(max_head):
        X, Y = [], []
        for i in range(N):
            m = int(marker_pos[i].item())
            X.append(values[i, h, m].numpy()); Y.append(0)          # MARKER
            for k in range(1, K + 1):
                X.append(values[i, h, m + k].numpy()); Y.append(k)  # passkey pk
            for ctrl in (-20, 30):
                pos = m + ctrl
                if 0 <= pos < T and not (m <= pos <= m + K):
                    X.append(values[i, h, pos].numpy()); Y.append(K + 1)  # other
        Xn = np.stack(X)
        Yn = np.array(Y)

        # 70/30 split.
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(Yn))
        ntr = int(len(Yn) * 0.7)
        tr, te = idx[:ntr], idx[ntr:]

        # Train a closed-form softmax regression via torch for convenience.
        W = torch.zeros((dh, K + 2), requires_grad=True)
        b = torch.zeros(K + 2, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.1)
        Xt = torch.tensor(Xn[tr], dtype=torch.float32)
        Yt = torch.tensor(Yn[tr], dtype=torch.long)
        for _ in range(300):
            logits = Xt @ W + b
            loss = FF.cross_entropy(logits, Yt) + 1e-4 * (W ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            logits_te = torch.tensor(Xn[te], dtype=torch.float32) @ W + b
            pred = logits_te.argmax(-1).numpy()
            test_acc = float((pred == Yn[te]).mean())
            logits_tr = torch.tensor(Xn[tr], dtype=torch.float32) @ W + b
            pred_tr = logits_tr.argmax(-1).numpy()
            train_acc = float((pred_tr == Yn[tr]).mean())

        per_head.append({
            "head": h,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_classes": K + 2,
            "chance": 1 / (K + 2),
        })
    return per_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--examples", default=str(RESULTS_DIR / "probe_examples.pt"))
    ap.add_argument("--out-prefix", default=str(RESULTS_DIR / "l2_values"))
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--head", type=int, default=0,
                    help="Head to PCA-plot individually.")
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

    print(f"collecting values from layer {args.layer} ({mcfg.n_heads} heads)...")
    values = collect_values(model, tokens, args.layer)   # [N, H, T, dh]
    print(f"  shape = {tuple(values.shape)}")

    mag_path = Path(args.out_prefix + "_magnitude.png")
    mean_norm = magnitude_plot(values, marker_pos, K, mag_path)

    pca_path = Path(args.out_prefix + f"_pca_H{args.head}.png")
    pca_info = pca_scatter(values, marker_pos, K, args.head, pca_path)

    print(f"running offset probe on all {values.shape[1]} heads at layer {args.layer}...")
    probe = offset_probe(values, marker_pos, K, max_head=values.shape[1])
    for r in probe:
        print(f"  L{args.layer} H{r['head']}: train={r['train_acc']:.3f} "
              f"test={r['test_acc']:.3f}  chance={r['chance']:.3f}")

    result = {
        "layer": args.layer,
        "head_probed_for_pca": args.head,
        "offset_probe_accuracy": probe,
        "pca_info": pca_info,
        "magnitude_peaks": {
            f"head_{h}": {
                "max_pos": int(np.argmax(mean_norm[h])),
                "max_value": float(mean_norm[h].max()),
                "mean_overall": float(mean_norm[h].mean()),
            }
            for h in range(mean_norm.shape[0])
        },
    }
    out_json = Path(args.out_prefix + "_probe.json")
    out_json.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
