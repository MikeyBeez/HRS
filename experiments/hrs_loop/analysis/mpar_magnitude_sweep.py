"""MPAR magnitude sweep on trained Variant B WikiText checkpoint.

At the final step of the recurrent stage, Coda's input is `e + project_up(m_T)`.
This script scales that contribution by k: `e + k * project_up(m_T)` for
k ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} and measures val PPL.

Interpretation:
  - smooth PPL(k) curve with minimum at k=1.0 and graceful degradation
    either side  →  MPAR acts as a modulation/gain control signal
  - threshold/cliff behavior (flat then collapse at some k)
                          →  MPAR acts as content (retrieved information
                             below threshold is useless)
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
def _eval_with_scale(model, loader, device, amp_dtype, scale: float,
                      T: int) -> float:
    """Forward with MPAR contribution at Coda's input scaled by `scale`.

    We replicate HRSLoop.forward manually so we can inject the scale factor
    at the right place: after the recurrent stage produces m_T, the Coda
    input becomes `e + scale * project_up(m_T)` instead of `e + project_up(m_T)`.

    Inside the recurrent loop itself (the per-iteration MPAR bias fed to the
    block), we leave the magnitude unchanged — that's a separate intervention.
    This script only tests the final Coda-facing contribution.
    """
    model.eval()
    losses = []
    rec = model.recurrent
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            idx = x[:, :-1]
            y = x[:, 1:]
            B, L = idx.shape
            pos = torch.arange(L, device=device)
            h = model.tok_emb(idx) + model.pos_emb(pos)[None]
            for blk in model.prelude:
                h = blk(h)
            e = h
            # Recurrent stage, replicating RecurrentStageB.forward
            m = torch.zeros(B, rec.rank_m, device=device, dtype=e.dtype)
            for t in range(T):
                h_t = e + rec.project_up(m)
                block_out = rec.block(h_t)
                lora_idx = t if t < len(rec.loras) else (t % len(rec.loras))
                lora_out = rec.loras[lora_idx](h_t)
                h_out = block_out + lora_out
                m = rec.project_down(h_out)
            # Coda input with MPAR contribution scaled by `scale`.
            coda_in = e + scale * rec.project_up(m)
            for blk in model.coda:
                coda_in = blk(coda_in)
            coda_in = model.ln_f(coda_in)
            logits = model.head(coda_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    y.reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))


def run_one_seed(seed: int, scales: list[float], T: int, val_loader,
                  device, amp_dtype) -> dict:
    suffix = "" if seed == 0 else f"_seed{seed}"
    ckpt_path = CKPT_DIR / f"variant_B{suffix}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B"
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print(f"\n=== seed {seed} ({ckpt_path.name}) T_eval={T} ===")
    print(f"  {'k':>5}  {'val_loss':>9}  {'val_ppl':>9}  {'Δ% vs k=1':>10}")
    print("  " + "-" * 40)

    per_scale = {}
    # Compute k=1.0 first as anchor
    results_list = []
    for k in scales:
        loss = _eval_with_scale(model, val_loader, device, amp_dtype, k, T)
        ppl = math.exp(loss)
        per_scale[str(k)] = {"val_loss": loss, "val_ppl": ppl}
        results_list.append((k, loss, ppl))

    ppl_1 = per_scale["1.0"]["val_ppl"]
    for k, loss, ppl in results_list:
        delta_pct = 100.0 * (ppl - ppl_1) / ppl_1
        print(f"  {k:>5.2f}  {loss:>9.4f}  {ppl:>9.3f}  {delta_pct:>+9.2f}%")

    return {"seed": seed, "ckpt": ckpt_path.name, "T": T,
             "scales": scales, "per_scale": per_scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out",
                    default=str(RESULTS_DIR / "stage3_mpar_magnitude.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from data import load_wikitext
    splits, _ = load_wikitext("wikitext-2", seq_len=512)
    val_loader = DataLoader(splits["validation"], batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    all_runs = []
    for s in args.seeds:
        r = run_one_seed(s, args.scales, args.T, val_loader, device, amp_dtype)
        all_runs.append(r)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"seeds": args.seeds, "scales": args.scales, "T": args.T,
            "runs": all_runs}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
