"""Test the address-reinforcement framing on Variant B.

Hypothesis: the MPAR, after projection back to d_model, points in roughly the
same direction as the mean-pooled Prelude output (e_bar). If so, the
recurrent stage is adding an additive reinforcement along the same axis.

Alternative: the MPAR is orthogonal or complementary to e_bar — adding
new-direction information, not reinforcing existing direction.

Run on (1) Variant B WikiText checkpoint, (2) Variant B composite checkpoint.
Both must exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Per-row cosine, then mean across batch."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def _norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """mean over batch of ||a||/||b||."""
    return (a.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)).mean().item()


@torch.no_grad()
def analyze(ckpt_path: Path, get_batch: Callable[[int], torch.Tensor],
            label: str, T: int = 4, n_seqs: int = 50) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B", f"expected Variant B, got {cfg.variant}"
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = get_batch(n_seqs).to(device)
    B, L = x.shape

    # Forward through Prelude explicitly.
    pos = torch.arange(L, device=device)
    h = model.tok_emb(x) + model.pos_emb(pos)[None]
    for blk in model.prelude:
        h = blk(h)
    e = h                                          # (B, L, d)
    e_bar = e.float().mean(dim=1)                  # (B, d)

    # Run recurrent stage manually, capturing m_t after each loop.
    rec = model.recurrent
    m = torch.zeros(B, rec.rank_m, device=device, dtype=e.dtype)
    m_list = []
    for t in range(T):
        h_t = e + rec.project_up(m)
        block_out = rec.block(h_t)
        lora_idx = t if t < len(rec.loras) else (t % len(rec.loras))
        lora_out = rec.loras[lora_idx](h_t)
        h_out = block_out + lora_out
        m = rec.project_down(h_out)
        m_list.append(m.float())                   # (B, rank_m)

    # project_up in fp32 for accurate cosines.
    W_up = rec.project_up.up.weight.detach().float()   # (d, rank_m)
    def up(m_vec: torch.Tensor) -> torch.Tensor:
        return m_vec @ W_up.T                          # (B, d)

    m_up_list = [up(m) for m in m_list]

    # Net effect on Coda input: Coda sees e + project_up(m_T) broadcast.
    # Collapse to per-sequence vector for cosine: (e_bar + m_up_T).
    m_up_T = m_up_list[-1]                            # (B, d)
    coda_input_bar = e_bar + m_up_T

    # Sharper view: per-position cosine(e[b, i, :], m_up_T[b, :]). This asks
    # whether the broadcast MPAR aligns with some positions in the sequence
    # but not others (selective addressing), vs. being uniformly near-zero
    # against every position (truly orthogonal side channel).
    e_f = e.float()                                   # (B, L, d)
    per_pos_cos = torch.nn.functional.cosine_similarity(
        e_f, m_up_T.unsqueeze(1), dim=-1
    )                                                 # (B, L)
    per_pos_flat = per_pos_cos.flatten()
    quantiles = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
                              device=per_pos_flat.device)
    q_vals = torch.quantile(per_pos_flat, quantiles).tolist()

    # Per-position norm ratio: ||m_up_T|| / ||e[b, i]|| at each position.
    e_pos_norm = e_f.norm(dim=-1)                     # (B, L)
    m_norm = m_up_T.norm(dim=-1, keepdim=True)        # (B, 1)
    per_pos_ratio = (m_norm / e_pos_norm.clamp_min(1e-8)).flatten()
    r_q_vals = torch.quantile(per_pos_ratio,
                                torch.tensor([0.05, 0.5, 0.95],
                                             device=per_pos_ratio.device)).tolist()

    result = {
        "label": label,
        "ckpt": str(ckpt_path),
        "n_seqs": n_seqs,
        "seq_len": L,
        "T": T,
        "e_bar_norm_mean": e_bar.norm(dim=-1).mean().item(),
        "per_loop": [
            {
                "t": t + 1,
                "cos_e_bar_m_up": _cos(e_bar, m_up_list[t]),
                "cos_e_bar_abs": _cos(e_bar, m_up_list[t].abs()),   # sanity
                "norm_ratio_m_up_over_e_bar": _norm_ratio(m_up_list[t], e_bar),
                "m_up_norm_mean": m_up_list[t].norm(dim=-1).mean().item(),
            }
            for t in range(T)
        ],
        "loop_to_loop_m_up_cos": [
            {"t_from": t + 1, "t_to": t + 2,
             "cos": _cos(m_up_list[t], m_up_list[t + 1])}
            for t in range(T - 1)
        ],
        "net_effect": {
            "cos_e_bar_coda_input": _cos(e_bar, coda_input_bar),
            "norm_ratio_coda_over_e": _norm_ratio(coda_input_bar, e_bar),
            "cos_e_bar_m_up_T": _cos(e_bar, m_up_T),
            "norm_ratio_m_up_T_over_e_bar": _norm_ratio(m_up_T, e_bar),
        },
        "per_position_cos_e_m_up_T": {
            "n_positions": int(per_pos_flat.numel()),
            "mean": per_pos_flat.mean().item(),
            "std": per_pos_flat.std().item(),
            "min": per_pos_flat.min().item(),
            "max": per_pos_flat.max().item(),
            "quantiles": {f"q{round(q*100):02d}": v
                           for q, v in zip(quantiles.tolist(), q_vals)},
        },
        "per_position_norm_ratio_m_over_e": {
            "q05": r_q_vals[0], "q50": r_q_vals[1], "q95": r_q_vals[2],
        },
    }

    # Pretty print
    print(f"\n=== {label} ===")
    print(f"  ckpt: {ckpt_path.name}   n={n_seqs}, seq_len={L}, T={T}")
    print(f"  ||e_bar||_mean = {result['e_bar_norm_mean']:.3f}")
    print(f"\n  per-loop cosine(e_bar, project_up(m_t)):")
    for r in result["per_loop"]:
        print(f"    t={r['t']}:  cos={r['cos_e_bar_m_up']:+.3f}   "
              f"||m_up_t||/||e_bar||={r['norm_ratio_m_up_over_e_bar']:.3f}   "
              f"||m_up_t||={r['m_up_norm_mean']:.3f}")
    print(f"\n  loop-to-loop cosine(m_up_t, m_up_{{t+1}}):")
    for r in result["loop_to_loop_m_up_cos"]:
        print(f"    t={r['t_from']}→{r['t_to']}:  cos={r['cos']:+.3f}")
    ne = result["net_effect"]
    print(f"\n  net effect (Coda input vs bare Prelude):")
    print(f"    cos(e_bar, e_bar + m_up_T) = {ne['cos_e_bar_coda_input']:+.3f}")
    print(f"    ||e_bar + m_up_T|| / ||e_bar|| = {ne['norm_ratio_coda_over_e']:.3f}")
    print(f"    cos(e_bar, m_up_T) = {ne['cos_e_bar_m_up_T']:+.3f}")
    print(f"    ||m_up_T|| / ||e_bar|| = {ne['norm_ratio_m_up_T_over_e_bar']:.3f}")

    pp = result["per_position_cos_e_m_up_T"]
    print(f"\n  per-position cos(e[b,i,:], m_up_T[b,:])  "
          f"(n_positions = {pp['n_positions']}):")
    print(f"    mean={pp['mean']:+.3f}  std={pp['std']:.3f}  "
          f"min={pp['min']:+.3f}  max={pp['max']:+.3f}")
    qs = pp["quantiles"]
    print(f"    q01={qs['q01']:+.3f}  q05={qs['q05']:+.3f}  "
          f"q25={qs['q25']:+.3f}  q50={qs['q50']:+.3f}  "
          f"q75={qs['q75']:+.3f}  q95={qs['q95']:+.3f}  q99={qs['q99']:+.3f}")
    pnr = result["per_position_norm_ratio_m_over_e"]
    print(f"  per-position ||m_up_T||/||e[b,i]||  "
          f"q05={pnr['q05']:.3f}  median={pnr['q50']:.3f}  q95={pnr['q95']:.3f}")

    return result


def wikitext_batch(n_seqs: int) -> torch.Tensor:
    """Grab n_seqs validation sequences of length 512 from WikiText-2."""
    from data import load_wikitext  # top-level data.py with WikiText loader
    splits, _ = load_wikitext("wikitext-2", seq_len=512)
    val = splits["validation"]
    xs = []
    for i in range(n_seqs):
        item = val[i]
        if isinstance(item, (tuple, list)):
            xs.append(item[0])
        else:
            xs.append(item)
    return torch.stack(xs)


def composite_batch(n_seqs: int, k: int = 4) -> torch.Tensor:
    """Grab n_seqs val sequences at chain length k (in-distribution = k=4)."""
    from experiments.hrs_loop.tasks.compositional_lookup import (
        CompositionalConfig, CompositionalDataset,
    )
    cfg = CompositionalConfig(seq_len=256, seed=7)    # fresh held-out seed
    ds = CompositionalDataset(cfg, n_samples=n_seqs, k_fixed=k, seed=7)
    xs = []
    for i in range(n_seqs):
        inp, _, _, _, _ = ds[i]
        xs.append(inp)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=50)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--wikitext-seeds", nargs="+", type=int, default=[0],
                    help="Seeds of Variant B WikiText checkpoints to analyze.")
    ap.add_argument("--skip-composite", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "stage3_address_reinforcement.json"))
    args = ap.parse_args()

    out = {"n_seqs": args.n_seqs, "T": args.T, "runs": []}

    for s in args.wikitext_seeds:
        suffix = "" if s == 0 else f"_seed{s}"
        wikitext_ckpt = CKPT_DIR / f"variant_B{suffix}_best.pt"
        if wikitext_ckpt.exists():
            r = analyze(wikitext_ckpt, wikitext_batch,
                        f"WikiText Variant B seed {s}",
                        T=args.T, n_seqs=args.n_seqs)
            out["runs"].append(r)
        else:
            print(f"[skip] {wikitext_ckpt} missing")

    composite_ckpt = CKPT_DIR / "composite_B_best.pt"
    if not args.skip_composite and composite_ckpt.exists():
        r = analyze(composite_ckpt,
                    lambda n: composite_batch(n, k=4),
                    "Composite Variant B (k=4 val)",
                    T=args.T, n_seqs=args.n_seqs)
        out["runs"].append(r)
    elif args.skip_composite:
        pass
    else:
        print(f"[skip] {composite_ckpt} missing")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
