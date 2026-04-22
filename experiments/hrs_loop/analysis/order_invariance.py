"""Test 5: Order invariance of MPAR accumulation.

Run Variant B in three inference modes on a held-out batch:
  Mode 1 — canonical (loops 1→2→3→4, each sees prior MPAR).
  Mode 2 — capture m_1..m_4 canonically, then replay Coda biased with
           mean(permutation(m_1..m_4)). 6 permutations: identity, reverse,
           4 random seeds.
  Mode 3 — 4 independent runs with T=1 (each starts from zero MPAR),
           yielding m_1^ind..m_4^ind; average and use as mpar_override.

All three modes go through the same forward path via the `mpar_capture`
and `mpar_override` hooks already wired into RecurrentStageB.

Metrics per mode (vs Mode 1):
  - mean per-token KL divergence
  - top-1 argmax agreement rate
  - val PPL on this batch
  - L2 norm of the biased MPAR (final m used at last loop)

For scale calibration we also include Variant A on the same batch.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.hrs_loop.analysis._shared import (
    RESULTS_DIR, load_checkpoint, load_val_loader,
)


@torch.no_grad()
def _logits_and_mpars_canonical(model, x, T, device, amp_dtype):
    """Return (logits, [m_1..m_T])."""
    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                         enabled=(device.type == "cuda")):
        logits, captured = model(x, T=T, mpar_capture=True)
    return logits.to(torch.float32), [m.to(torch.float32) for m in captured]


@torch.no_grad()
def _logits_with_coda_override(model, x, T, device, amp_dtype, override):
    """Mode 2: run the recurrent loops normally but swap m_T at the Coda input.

    Run the T recurrent iterations canonically, then replace the MPAR that
    feeds the Coda with `override`. This is the spec's "bias the stream
    with mean(perm(m_1..m_4))" — we compute the bias at the point where
    Coda consumes it, not by re-running the final loop.
    """
    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                         enabled=(device.type == "cuda")):
        logits = model(x, T=T, coda_mpar_override=override.to(x.device))
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.to(torch.float32)


@torch.no_grad()
def _logits_with_skip(model, x, device, amp_dtype, override):
    """Mode 3: skip the recurrent stage; use `e + project_up(override)` as
    Coda input directly. This matches the spec's 'Bias the final Coda pass
    with mean(m_indep...).' verbatim."""
    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                         enabled=(device.type == "cuda")):
        logits = model(x, skip_recurrent_with_mpar=override.to(x.device))
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.to(torch.float32)


@torch.no_grad()
def _logits_basic(model, x, T, device, amp_dtype):
    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                         enabled=(device.type == "cuda")):
        logits = model(x, T=T)
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.to(torch.float32)


def kl_per_token(p_logits, q_logits) -> float:
    """Mean KL(P || Q) per token. Chunked along batch to avoid OOM on
    large-vocab tensors (batch 16 × seq 512 × vocab 50k ≈ 1.6 GB/tensor)."""
    total = 0.0
    n = 0
    for pb, qb in zip(p_logits.split(2, dim=0), q_logits.split(2, dim=0)):
        p = F.log_softmax(pb, dim=-1)
        q = F.log_softmax(qb, dim=-1)
        kl = (p.exp() * (p - q)).sum(-1)
        total += float(kl.sum())
        n += kl.numel()
    return total / max(n, 1)


def top1_agreement(p_logits, q_logits) -> float:
    return float((p_logits.argmax(-1) == q_logits.argmax(-1)).float().mean())


def token_loss(logits, target_ids) -> float:
    return float(F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1)
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(RESULTS_DIR / "test5_order_invariance.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_B, cfg_B, _ = load_checkpoint("B", device)
    T = cfg_B.T_default
    val_loader, _tok = load_val_loader(batch_size=args.batch_size)

    # Optional: load Variant A for calibration.
    try:
        model_A, _cfg_A, _ = load_checkpoint("A", device)
        have_A = True
    except FileNotFoundError:
        have_A = False

    perms = [tuple(range(T)), tuple(reversed(range(T)))]
    rng = random.Random(42)
    while len(perms) < 6:
        p = list(range(T))
        rng.shuffle(p)
        if tuple(p) not in perms:
            perms.append(tuple(p))
    perm_labels = ["identity", "reverse"] + [f"random_{i}" for i in range(4)]

    # Accumulators: per-mode sums of KL, top1 agreement counts, loss.
    totals = {}

    def _ensure(key):
        if key not in totals:
            totals[key] = {"kl_sum": 0.0, "top1_sum": 0.0, "loss_sum": 0.0,
                            "mpar_norm_sum": 0.0, "n_tokens": 0, "n_batches": 0}

    _ensure("mode1_canonical")
    for label in perm_labels:
        _ensure(f"mode2_perm_{label}")
    _ensure("mode3_independent_avg")
    if have_A:
        _ensure("variantA_calibration")

    for bi, batch in enumerate(val_loader):
        if bi >= args.n_batches:
            break
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        inp = x[:, :-1]
        tgt = x[:, 1:]

        # Mode 1: canonical Variant B.
        logits_m1, mpars = _logits_and_mpars_canonical(
            model_B, inp, T, device, amp_dtype)
        loss_m1 = token_loss(logits_m1, tgt)

        t1 = totals["mode1_canonical"]
        t1["kl_sum"] += 0.0   # KL(m1||m1)=0
        t1["top1_sum"] += 1.0
        t1["loss_sum"] += loss_m1
        t1["mpar_norm_sum"] += float(mpars[-1].norm(dim=-1).mean())
        t1["n_batches"] += 1

        # Mode 2: replay Coda input with averaged permutation of m_1..m_T.
        # Since arithmetic mean is permutation-invariant in exact arithmetic,
        # the 6 perms probe bf16 reduction-order variance (a calibration
        # floor). The real test is Mode2-mean vs Mode1-latest: does using
        # the average of all m's differ from using only m_T?
        M = torch.stack(mpars, dim=0)   # (T, B, rank)
        for pi, label in zip(perms, perm_labels):
            perm_avg = M[list(pi)].mean(dim=0)
            logits = _logits_with_coda_override(
                model_B, inp, T, device, amp_dtype, perm_avg)
            t = totals[f"mode2_perm_{label}"]
            t["kl_sum"] += kl_per_token(logits_m1, logits)
            t["top1_sum"] += top1_agreement(logits_m1, logits)
            t["loss_sum"] += token_loss(logits, tgt)
            t["mpar_norm_sum"] += float(perm_avg.norm(dim=-1).mean())
            t["n_batches"] += 1

        # Mode 3: independent-loop MPARs. Run B T=1 four times, each with
        # a zero initial MPAR (the default). Average the resulting single
        # MPARs. Then Coda is biased with that average directly — no
        # recurrent passes at the final inference step.
        m_indep = []
        for _ in range(T):
            _lg, cap = _logits_and_mpars_canonical(
                model_B, inp, 1, device, amp_dtype)
            m_indep.append(cap[0])  # the one MPAR produced by T=1
        m_indep_avg = torch.stack(m_indep, dim=0).mean(dim=0)

        logits_m3 = _logits_with_skip(
            model_B, inp, device, amp_dtype, m_indep_avg)
        t3 = totals["mode3_independent_avg"]
        t3["kl_sum"] += kl_per_token(logits_m1, logits_m3)
        t3["top1_sum"] += top1_agreement(logits_m1, logits_m3)
        t3["loss_sum"] += token_loss(logits_m3, tgt)
        t3["mpar_norm_sum"] += float(m_indep_avg.norm(dim=-1).mean())
        t3["n_batches"] += 1

        # Also: the canonical m_T norm for Mode 1 reference.
        # (Recorded into t1["mpar_norm_sum"] above as float(mpars[-1].norm...).)

        # Variant A calibration.
        if have_A:
            logits_A = _logits_basic(model_A, inp, T, device, amp_dtype)
            tA = totals["variantA_calibration"]
            tA["kl_sum"] += kl_per_token(logits_m1, logits_A)
            tA["top1_sum"] += top1_agreement(logits_m1, logits_A)
            tA["loss_sum"] += token_loss(logits_A, tgt)
            tA["mpar_norm_sum"] += 0.0  # n/a
            tA["n_batches"] += 1

        if (bi + 1) % 10 == 0:
            print(f"  processed {bi + 1}/{args.n_batches} batches")

    # Summarize.
    summary = []
    for mode, t in totals.items():
        n = max(1, t["n_batches"])
        row = {
            "mode": mode,
            "mean_kl_nats_per_token": t["kl_sum"] / n,
            "mean_top1_agreement": t["top1_sum"] / n,
            "mean_loss": t["loss_sum"] / n,
            "val_ppl": float(math.exp(t["loss_sum"] / n)),
            "mean_final_mpar_l2": t["mpar_norm_sum"] / n,
            "n_batches": n,
        }
        summary.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {Path(args.out)}\n")

    hdr = (f"{'mode':<30} {'KL/tok':>10} {'top1':>8} {'PPL':>10} {'‖m‖':>9}")
    print(hdr); print("-" * len(hdr))
    for r in summary:
        print(f"{r['mode']:<30} {r['mean_kl_nats_per_token']:>10.4f} "
              f"{r['mean_top1_agreement']:>8.3f} "
              f"{r['val_ppl']:>10.3f} "
              f"{r['mean_final_mpar_l2']:>9.3f}")


if __name__ == "__main__":
    main()
