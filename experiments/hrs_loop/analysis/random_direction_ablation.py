"""Random-direction ablation for Variant B on WikiText.

Replaces the canonical MPAR contribution to Coda with a random direction in
project_up's image, scaled so the d-space norm matches canonical per
sequence. This isolates direction from magnitude:

  canonical:   Coda input = e + project_up(m_T)
  random:      Coda input = e + α * project_up(r),   r ~ N(0, I) in rank_m,
               α chosen per-sequence so ||α * project_up(r)|| = ||project_up(m_T)||
  bypass:      Coda input = e                         (= scale 0 in magnitude sweep)

Outcomes to distinguish:
  random PPL ≈ bypass  →  direction matters strongly; the MPAR is content
  random PPL intermediate  →  partial, both direction and magnitude contribute
  random PPL ≈ canonical  →  magnitude is all that matters; the MPAR acts
                              as pure gain within the project_up image
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
def _eval_variant(model, loader, device, amp_dtype, T: int, mode: str,
                    rand_seed: int = 0) -> float:
    """Return mean cross-entropy loss over loader. mode ∈ {canonical, random, bypass}."""
    model.eval()
    losses = []
    rec = model.recurrent
    gen = torch.Generator(device=device).manual_seed(rand_seed) if mode == "random" else None
    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        idx = x[:, :-1]
        y = x[:, 1:]
        B, L = idx.shape
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            pos = torch.arange(L, device=device)
            h = model.tok_emb(idx) + model.pos_emb(pos)[None]
            for blk in model.prelude:
                h = blk(h)
            e = h
            # Run the recurrent stage canonically to get m_T.
            m = torch.zeros(B, rec.rank_m, device=device, dtype=e.dtype)
            for t in range(T):
                h_t = e + rec.project_up(m)
                block_out = rec.block(h_t)
                lora_idx = t if t < len(rec.loras) else (t % len(rec.loras))
                lora_out = rec.loras[lora_idx](h_t)
                h_out = block_out + lora_out
                m = rec.project_down(h_out)

            if mode == "canonical":
                contribution = rec.project_up(m).squeeze(1)    # (B, d)
            elif mode == "bypass":
                contribution = torch.zeros(B, e.shape[-1], device=device, dtype=e.dtype)
            elif mode == "random":
                # Random rank_m direction → project_up → d-space.
                # Scale so ||scaled|| == ||project_up(m_T)|| per sequence.
                # Resulting vector lives in project_up's image (128-dim subspace
                # of 256-dim d-space).
                v_canon = rec.project_up(m).squeeze(1)         # (B, d)
                canon_norm = v_canon.norm(dim=-1, keepdim=True)    # (B, 1)
                r = torch.randn(B, rec.rank_m, generator=gen,
                                  device=device, dtype=e.dtype)
                v_rand = rec.project_up(r).squeeze(1)          # (B, d)
                rand_norm = v_rand.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                contribution = v_rand * (canon_norm / rand_norm)
            elif mode == "random_full_d":
                # Random direction in full d-space (not confined to project_up's
                # image), scaled to match canonical per-sequence norm. With high
                # probability, this vector has a substantial component orthogonal
                # to project_up's image, which is off-manifold for Coda.
                d = e.shape[-1]
                v_canon = rec.project_up(m).squeeze(1)         # (B, d)
                canon_norm = v_canon.norm(dim=-1, keepdim=True)    # (B, 1)
                r = torch.randn(B, d, generator=gen,
                                  device=device, dtype=e.dtype)
                rand_norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                contribution = r * (canon_norm / rand_norm)
            else:
                raise ValueError(mode)

            coda_in = e + contribution.unsqueeze(1)            # (B, 1, d) broadcast
            for blk in model.coda:
                coda_in = blk(coda_in)
            coda_in = model.ln_f(coda_in)
            logits = model.head(coda_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    y.reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="Model seed")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-random-draws", type=int, default=5,
                    help="Number of random-seed draws for the random-direction eval.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "stage3_random_direction.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    ckpt_path = CKPT_DIR / f"variant_B{suffix}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B"
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    from data import load_wikitext
    splits, _ = load_wikitext("wikitext-2", seq_len=512)
    val_loader = DataLoader(splits["validation"], batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    print(f"\n=== Variant B seed {args.seed} — random-direction ablation ===")
    print(f"  ckpt: {ckpt_path.name}, T={args.T}")

    canonical_loss = _eval_variant(model, val_loader, device, amp_dtype,
                                     T=args.T, mode="canonical")
    bypass_loss = _eval_variant(model, val_loader, device, amp_dtype,
                                  T=args.T, mode="bypass")
    canonical_ppl = math.exp(canonical_loss)
    bypass_ppl = math.exp(bypass_loss)

    def run_random(mode_name: str) -> tuple[list[float], float, float]:
        losses = []
        for i in range(args.n_random_draws):
            rs = 100 + 100 * i
            loss = _eval_variant(model, val_loader, device, amp_dtype,
                                  T=args.T, mode=mode_name, rand_seed=rs)
            losses.append(loss)
            print(f"  [{mode_name}] draw {i+1} (rand_seed={rs}):  "
                  f"val_loss={loss:.4f}  val_ppl={math.exp(loss):.3f}")
        ppls = [math.exp(l) for l in losses]
        return ppls, float(np.mean(ppls)), \
               (float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0)

    rand_in_sub_ppls, rand_in_sub_mean, rand_in_sub_std = run_random("random")
    rand_full_ppls, rand_full_mean, rand_full_std = run_random("random_full_d")

    gap = bypass_ppl - canonical_ppl
    print()
    print(f"  canonical PPL:                     {canonical_ppl:>7.2f}")
    print(f"  bypass PPL:                        {bypass_ppl:>7.2f}   "
          f"(+{100*(bypass_ppl-canonical_ppl)/canonical_ppl:.1f}%)")
    print(f"  random in project_up's image:      {rand_in_sub_mean:>7.2f} ± "
          f"{rand_in_sub_std:.2f}   "
          f"(+{100*(rand_in_sub_mean-canonical_ppl)/canonical_ppl:.1f}%, "
          f"{100*(rand_in_sub_mean-canonical_ppl)/gap:.0f}% of canon→bypass gap)")
    print(f"  random in full d-space:            {rand_full_mean:>7.2f} ± "
          f"{rand_full_std:.2f}   "
          f"(+{100*(rand_full_mean-canonical_ppl)/canonical_ppl:.1f}%, "
          f"{100*(rand_full_mean-canonical_ppl)/gap:.0f}% of canon→bypass gap)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "seed": args.seed, "T": args.T, "n_random_draws": args.n_random_draws,
        "canonical_ppl": canonical_ppl,
        "bypass_ppl": bypass_ppl,
        "random_in_subspace_ppls": rand_in_sub_ppls,
        "random_in_subspace_mean": rand_in_sub_mean,
        "random_in_subspace_std": rand_in_sub_std,
        "random_full_d_ppls": rand_full_ppls,
        "random_full_d_mean": rand_full_mean,
        "random_full_d_std": rand_full_std,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
