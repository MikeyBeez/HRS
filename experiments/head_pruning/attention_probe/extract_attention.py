"""Phase A: run the baseline passkey model on the probe batch, capture
per-layer per-head attention weights.

Monkey-patches the MHA forward to cache the softmax'd attention tensor on
the module. Runs forward in chunks, stacks into [N, L, H, T, T], saves.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.diagonal_attention.config import ModelConfig
from experiments.diagonal_attention.model import TinyTransformer


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_CKPT = Path("/mnt/data/Code/HRS/experiments/pruning/checkpoints/mha_passkey.pt")


def _make_patched_forward(module):
    """Return a forward that caches the softmax'd attention in module._attn."""
    def patched(x):
        B, T, d = x.shape
        H = module.cfg.n_heads
        dh = module.cfg.d_head
        scores = module.compute_scores(x)
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # MHA: (B, H, T, T)
        module._attn = attn.detach()
        V = module.W_V(x).view(B, T, H, dh).transpose(1, 2)
        if attn.dim() == 3:
            out = torch.einsum("bts,bhsd->bhtd", attn, V)
        else:
            out = torch.einsum("bhts,bhsd->bhtd", attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return module.W_O(out)
    return patched


def install_cache(model: TinyTransformer):
    for blk in model.blocks:
        blk.attn.forward = _make_patched_forward(blk.attn)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["mcfg"])
    model = TinyTransformer(mcfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mcfg


@torch.no_grad()
def extract(model, tokens: torch.Tensor, device: torch.device,
             batch_size: int = 32) -> torch.Tensor:
    """Return attention tensor of shape [N, L, H, T, T]."""
    N, T = tokens.shape
    L = model.cfg.n_layers
    H = model.cfg.n_heads

    # Stored on CPU; we only keep detached+moved tensors.
    out = torch.empty((N, L, H, T, T), dtype=torch.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = tokens[start:end].to(device)
        _logits = model(x)  # side effect: each block.attn._attn set
        for li, blk in enumerate(model.blocks):
            a = blk.attn._attn.to(torch.float32).cpu()  # (b, H, T, T)
            out[start:end, li] = a
        print(f"  batch {start}-{end}/{N}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--examples", default=str(RESULTS_DIR / "probe_examples.pt"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "attention_tensors.pt"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.examples, map_location="cpu", weights_only=False)
    tokens = data["tokens"]

    model, mcfg = load_model(Path(args.ckpt), device)
    install_cache(model)

    print(f"extracting attention from {args.ckpt}")
    print(f"  shape target: [{tokens.shape[0]}, {mcfg.n_layers}, "
          f"{mcfg.n_heads}, {mcfg.ctx_len}, {mcfg.ctx_len}]")
    attn = extract(model, tokens, device, args.batch_size)

    torch.save({
        "attention": attn,          # [N, L, H, T, T]
        "n_layers": mcfg.n_layers,
        "n_heads": mcfg.n_heads,
        "ctx_len": mcfg.ctx_len,
        "ckpt": str(args.ckpt),
    }, args.out)
    size_mb = attn.numel() * attn.element_size() / (1024 ** 2)
    print(f"wrote {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
