"""Freeze MPAR after iteration N: eval-time depth sweep on a model trained
at T=4. Freezing m at m_N is algebraically equivalent to evaluating at
T_eval=N (Coda input = e + project_up(m_N) in both cases), so this script
simply reads val PPL / accuracy at each T_eval and compares to canonical.

If T_eval=1 matches T_eval=4, iteration beyond the first converged MPAR is
mathematically redundant for the trained model's forward pass.

Runs on:
  - WikiText Variant B  → val PPL over full val split at T_eval ∈ {0..6}
  - Composite Variant B → k-conditional accuracy at T_eval ∈ {0..6}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


@torch.no_grad()
def _wikitext_eval(model, loader, device, amp_dtype, T_eval):
    """Full-val-split cross-entropy at a fixed T_eval. T_eval=0 → bypass
    recurrent entirely via skip_recurrent_with_mpar=zeros."""
    model.eval()
    losses = []
    rank_m = model.cfg.rank_m
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x, _ = batch[0].to(device), batch[1].to(device)
        else:
            x = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            if T_eval == 0:
                B = x.shape[0]
                zero = torch.zeros(B, rank_m, device=device, dtype=amp_dtype)
                out = model(x[:, :-1], skip_recurrent_with_mpar=zero)
            else:
                out = model(x[:, :-1], T=T_eval)
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                    x[:, 1:].reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def _composite_eval(model, loader, device, amp_dtype, T_eval):
    model.eval()
    correct = total = 0
    rank_m = model.cfg.rank_m
    for x, y, ans_pos, terminal, k in loader:
        x = x.to(device); ans_pos = ans_pos.to(device); terminal = terminal.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            if T_eval == 0:
                B = x.shape[0]
                zero = torch.zeros(B, rank_m, device=device, dtype=amp_dtype)
                out = model(x, skip_recurrent_with_mpar=zero)
            else:
                out = model(x, T=T_eval)
            if isinstance(out, tuple):
                out = out[0]
        idx = torch.arange(out.shape[0], device=out.device)
        pred = out[idx, ans_pos].argmax(dim=-1)
        correct += (pred == terminal).sum().item()
        total += terminal.shape[0]
    return correct / max(1, total)


def wikitext_run(T_list: list[int], seed: int = 0,
                  splits=None) -> dict:
    from data import load_wikitext
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    suffix = "" if seed == 0 else f"_seed{seed}"
    ckpt_path = CKPT_DIR / f"variant_B{suffix}_best.pt"
    ckpt = torch.load(ckpt_path,
                        map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B"
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    if splits is None:
        splits, _ = load_wikitext("wikitext-2", seq_len=512)
    val_loader = DataLoader(splits["validation"], batch_size=16, shuffle=False,
                              drop_last=False, num_workers=0)

    out = {"dataset": "wikitext-2", "seed": seed, "ckpt": ckpt_path.name,
           "T_list": T_list, "val_ppl": {}, "val_loss": {}}
    print(f"\n=== WikiText Variant B seed {seed} — "
          f"trained at T_train={cfg.T_default}, eval-time T sweep ===")
    print(f"{'T_eval':>6}  {'val_loss':>9}  {'val_ppl':>9}")
    print("-" * 30)
    for T_eval in T_list:
        loss = _wikitext_eval(model, val_loader, device, amp_dtype, T_eval)
        ppl = math.exp(loss)
        out["val_loss"][str(T_eval)] = loss
        out["val_ppl"][str(T_eval)] = ppl
        tag = " (bypass)" if T_eval == 0 else (" (canonical)" if T_eval == cfg.T_default else "")
        print(f"{T_eval:>6}  {loss:>9.4f}  {ppl:>9.3f}{tag}")
    return out


def composite_run(T_list: list[int], k_list: tuple[int, ...] = (1, 2, 3, 4, 6, 8)) -> dict:
    from experiments.hrs_loop.tasks.compositional_lookup import (
        CompositionalConfig, make_loaders,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    ckpt = torch.load(CKPT_DIR / "composite_B_best.pt",
                        map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B"
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    task_cfg = CompositionalConfig(**ckpt["task_cfg"])
    data = make_loaders(task_cfg, n_train=1, n_val_per_k=500,
                          eval_ks=k_list)

    out = {"dataset": "composite", "T_list": list(T_list),
           "k_list": list(k_list), "acc_grid": {}}
    print(f"\n=== Composite Variant B — eval-time T sweep ===")
    hdr = f"{'T_eval':>6}  " + "  ".join(f"k={k:>1}" for k in k_list)
    print(hdr)
    print("-" * len(hdr))
    for T_eval in T_list:
        row = {}
        for k in k_list:
            row[str(k)] = _composite_eval(model, data["val"][k], device,
                                            amp_dtype, T_eval)
        out["acc_grid"][str(T_eval)] = row
        tag = " (bypass)" if T_eval == 0 else (" (canonical)" if T_eval == cfg.T_default else "")
        print(f"{T_eval:>6}  " + "  ".join(f"{row[str(k)]:.3f}" for k in k_list) + tag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-list", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4, 6])
    ap.add_argument("--wikitext-seeds", nargs="+", type=int, default=[0],
                    help="Seeds whose variant_B_seedN_best.pt to eval on WikiText")
    ap.add_argument("--skip-composite", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "stage3_freeze_mpar.json"))
    args = ap.parse_args()

    results = {"T_list": args.T_list}

    # Load WikiText splits once if any wikitext seeds requested
    wt_seeds = [s for s in args.wikitext_seeds
                 if (CKPT_DIR / (f"variant_B{'' if s == 0 else f'_seed{s}'}_best.pt")).exists()]
    if wt_seeds:
        from data import load_wikitext
        splits, _ = load_wikitext("wikitext-2", seq_len=512)
        results["wikitext_by_seed"] = {}
        for s in wt_seeds:
            results["wikitext_by_seed"][str(s)] = wikitext_run(
                args.T_list, seed=s, splits=splits
            )

    if not args.skip_composite and (CKPT_DIR / "composite_B_best.pt").exists():
        results["composite"] = composite_run(args.T_list)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
