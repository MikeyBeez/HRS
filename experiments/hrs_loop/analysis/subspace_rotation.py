"""Subspace-rotation ablation on trained Variant B (WikiText seed 0).

Replace project_up's learned d_model × rank_m weight matrix U with U_rot = R·U,
where R is a random d_model × d_model orthogonal rotation. This preserves U's
singular values and Frobenius norm but rotates U's column space to a different
rank-rank_m subspace of d_model.

Prediction for the emergent-subspace-filter story (Hypothesis 1):
  rotated-U PPL ≈ full-d-space random PPL (~310 range) or worse.
  The network's learned consumption machinery is specialized to U's specific
  column space; rotating to a different rank-128 subspace forfeits that.

Alternative (Hypothesis 2 — generic rank-128 consumption):
  rotated-U PPL ≈ canonical PPL (~195). Coda treats any rank-128 subspace
  with matching singular spectrum the same way.

Intermediate outcomes distinguish partial specialization.

We run 5 rotation seeds (42..46) to rule out rotation-specific artifacts.
Optionally replicate across Variant B model seeds 1,2,3.
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


def _make_rotation(d: int, seed: int, device, dtype) -> torch.Tensor:
    """Random orthogonal d × d matrix via QR of Gaussian, seeded."""
    gen = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(d, d, generator=gen, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    # Ensure determinant-neutral (QR can give reflections + rotations; either
    # is fine for this experiment — both are orthogonal).
    return Q.to(dtype)


def _subspace_cos(U: torch.Tensor, U_rot: torch.Tensor) -> dict:
    """Principal angles between col-span of U and col-span of U_rot.
    Returns max, mean, min of the principal cosines (= singular values of
    orthonormal_cols(U)^T @ orthonormal_cols(U_rot))."""
    Uf = U.float()
    Vf = U_rot.float()
    # Orthonormal bases (QR of column matrix); thin QR.
    W1, _ = torch.linalg.qr(Uf)
    W2, _ = torch.linalg.qr(Vf)
    M = W1.T @ W2                                # (k, k)
    svals = torch.linalg.svdvals(M)
    return {
        "mean_principal_cos": float(svals.mean().item()),
        "max_principal_cos": float(svals.max().item()),
        "min_principal_cos": float(svals.min().item()),
        "sum_sq_principal_cos": float((svals ** 2).sum().item()),
    }


@torch.no_grad()
def _eval_ppl(model, loader, device, amp_dtype, T: int) -> float:
    """Canonical forward, full val split, mean cross-entropy."""
    model.eval()
    losses = []
    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        idx, y = x[:, :-1], x[:, 1:]
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(idx, T=T)
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                    y.reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))


def run_seed(model_seed: int, rot_seeds: list[int], T: int, val_loader,
              device, amp_dtype) -> dict:
    suffix = "" if model_seed == 0 else f"_seed{model_seed}"
    ckpt_path = CKPT_DIR / f"variant_B{suffix}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B"
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # Canonical PPL (sanity check)
    canon_loss = _eval_ppl(model, val_loader, device, amp_dtype, T)
    canon_ppl = math.exp(canon_loss)
    print(f"\n=== Variant B seed {model_seed} ===")
    print(f"  ckpt: {ckpt_path.name}  T={T}")
    print(f"  canonical PPL: {canon_ppl:.2f}")

    U_orig = model.recurrent.project_up.up.weight.data.clone()   # (d, rank_m)
    d, k = U_orig.shape
    u_fnorm = U_orig.norm(p="fro").item()
    u_svs = torch.linalg.svdvals(U_orig.float()).cpu().tolist()

    per_rotation = []
    for rs in rot_seeds:
        R = _make_rotation(d, seed=rs, device=device, dtype=U_orig.dtype)
        U_rot = R @ U_orig                                       # (d, rank_m)

        # Geometry verification
        rot_fnorm = U_rot.norm(p="fro").item()
        rot_svs = torch.linalg.svdvals(U_rot.float()).cpu().tolist()
        fnorm_delta = abs(rot_fnorm - u_fnorm) / u_fnorm
        svs_match = max(abs(a - b) for a, b in zip(u_svs, rot_svs))
        overlap = _subspace_cos(U_orig, U_rot)

        # Replace project_up's weight, evaluate, restore.
        model.recurrent.project_up.up.weight.data.copy_(U_rot)
        loss = _eval_ppl(model, val_loader, device, amp_dtype, T)
        ppl = math.exp(loss)
        model.recurrent.project_up.up.weight.data.copy_(U_orig)

        per_rotation.append({
            "rot_seed": rs,
            "val_loss": loss,
            "val_ppl": ppl,
            "frobenius_delta_rel": fnorm_delta,
            "singular_values_max_abs_delta": svs_match,
            "subspace_overlap": overlap,
        })
        print(f"  rot_seed={rs}: PPL={ppl:.2f}   "
              f"ΔF={fnorm_delta:.2e}  Δsv={svs_match:.2e}  "
              f"mean_princ_cos(U, U_rot)={overlap['mean_principal_cos']:.3f}")

    ppls = [r["val_ppl"] for r in per_rotation]
    return {
        "model_seed": model_seed,
        "ckpt": ckpt_path.name,
        "T": T,
        "canonical_ppl": canon_ppl,
        "U_frobenius_norm": u_fnorm,
        "U_singular_values": u_svs,
        "rot_seeds": rot_seeds,
        "per_rotation": per_rotation,
        "rotated_mean_ppl": float(np.mean(ppls)),
        "rotated_std_ppl": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
        "rotated_min_ppl": float(min(ppls)),
        "rotated_max_ppl": float(max(ppls)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--rot-seeds", nargs="+", type=int,
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out",
                    default=str(RESULTS_DIR / "stage3_subspace_rotation.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from data import load_wikitext
    splits, _ = load_wikitext("wikitext-2", seq_len=512)
    val_loader = DataLoader(splits["validation"], batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    all_runs = []
    for ms in args.model_seeds:
        r = run_seed(ms, args.rot_seeds, args.T, val_loader, device, amp_dtype)
        all_runs.append(r)

    # Reference numbers from prior experiments (for the side-by-side)
    # Pulled from stage3_random_direction.json and stage3_random_direction_seed{1,2,3}.json
    reference = {}
    for ms in args.model_seeds:
        if ms == 0:
            p = RESULTS_DIR / "stage3_random_direction.json"
        else:
            p = RESULTS_DIR / f"stage3_random_direction_seed{ms}.json"
        if p.exists():
            d = json.loads(p.read_text())
            reference[str(ms)] = {
                "canonical_ppl": d.get("canonical_ppl"),
                "bypass_ppl": d.get("bypass_ppl"),
                "random_in_subspace_mean": d.get("random_in_subspace_mean"),
                "random_full_d_mean": d.get("random_full_d_mean"),
            }

    out = {"T": args.T, "rot_seeds": args.rot_seeds,
            "model_seeds": args.model_seeds,
            "runs": all_runs, "reference_from_random_direction": reference}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    # Summary table
    print("\n" + "=" * 76)
    print("Summary: rotated-U PPL vs existing reference points")
    print("=" * 76)
    hdr = f"{'model_seed':>10} {'canon':>8} {'bypass':>8} {'rand_full':>10} {'rand_sub':>10} {'rotated_U':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_runs:
        ref = reference.get(str(r["model_seed"]), {})
        canon = r["canonical_ppl"]
        by = ref.get("bypass_ppl", float("nan"))
        rf = ref.get("random_full_d_mean", float("nan"))
        rs = ref.get("random_in_subspace_mean", float("nan"))
        rot_m, rot_s = r["rotated_mean_ppl"], r["rotated_std_ppl"]
        print(f"{r['model_seed']:>10} {canon:>8.2f} {by:>8.2f} {rf:>10.2f} "
              f"{rs:>10.2f} {rot_m:>7.2f}±{rot_s:>5.2f}")
    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
