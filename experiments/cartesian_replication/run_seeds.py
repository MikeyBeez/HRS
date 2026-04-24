"""Re-run the MHA vs Full-Bilinear comparison on Tiny Shakespeare
across N seeds for statistical characterization.

This is the "Cartesian product attention" test — the bilinear variant
computes scores = X W X^T / sqrt(d), which is the Cartesian product
of X with itself mediated by a learned d×d bilinear form W. Reuses
the existing scaffold in experiments/diagonal_attention/.

Outputs mha_vs_bilinear_seeds.json with per-seed val PPL + summary
statistics and effect size.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.diagonal_attention.config import ModelConfig, TrainConfig
from experiments.diagonal_attention.data import load_shakespeare, sample_lm_batch
from experiments.diagonal_attention.model import TinyTransformer
from experiments.diagonal_attention.train import _lr_at, eval_ppl


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def train_one(variant: str, seed: int, steps: int) -> dict:
    tcfg = TrainConfig(steps=steps, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, info = load_shakespeare()
    mcfg = ModelConfig(variant=variant, vocab_size=info["vocab_size"])
    model = TinyTransformer(mcfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr, weight_decay=tcfg.weight_decay, betas=tcfg.betas,
    )
    rng = np.random.default_rng(seed)

    model.train()
    t0 = time.time()
    losses = []
    for step in range(tcfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tcfg)
        x, y = sample_lm_batch(train_data, tcfg.batch_size, mcfg.ctx_len,
                                 device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN at step {step}, variant={variant} seed={seed}")

    final_ppl = eval_ppl(model, val_data, tcfg, mcfg, device, n_batches=80)
    return {
        "variant": variant,
        "seed": seed,
        "steps": steps,
        "final_val_ppl": final_ppl,
        "score_params": model.score_params(),
        "total_params": model.total_params(),
        "train_loss_last_50_mean": float(np.mean(losses[-50:])),
        "wall_seconds": time.time() - t0,
    }


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d with pooled SD (Hedges-style small-sample correction skipped)."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("inf") if ma != mb else 0.0
    return (ma - mb) / pooled


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-statistic + two-sided p via normal approx."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb) if (va or vb) else 0.0
    if se == 0:
        return (float("inf") if ma != mb else 0.0, 0.0 if ma != mb else 1.0)
    t = (ma - mb) / se
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--variants", nargs="+", default=["mha", "bilinear"])
    ap.add_argument("--out", default=str(RESULTS_DIR / "mha_vs_bilinear_seeds.json"))
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in args.variants:
        for seed in args.seeds:
            print(f"\n=== {variant} seed {seed} ===")
            rec = train_one(variant, seed, args.steps)
            print(f"  val_ppl={rec['final_val_ppl']:.3f}  "
                  f"wall={rec['wall_seconds']:.1f}s")
            runs.append(rec)
            Path(args.out).write_text(json.dumps({"runs": runs}, indent=2))

    # Aggregate.
    summary = {"by_variant": {}, "comparison": {}}
    for variant in args.variants:
        vals = [r["final_val_ppl"] for r in runs if r["variant"] == variant]
        summary["by_variant"][variant] = {
            "seeds": [r["seed"] for r in runs if r["variant"] == variant],
            "ppls": vals,
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    if len(args.variants) == 2:
        a = [r["final_val_ppl"] for r in runs if r["variant"] == args.variants[0]]
        b = [r["final_val_ppl"] for r in runs if r["variant"] == args.variants[1]]
        ma = statistics.mean(a); mb = statistics.mean(b)
        diff = mb - ma  # second variant minus first
        rel = diff / ma if ma else 0.0
        d = cohens_d(a, b)   # positive = first variant has lower PPL (better)
        t, p = welch_t(a, b)
        summary["comparison"] = {
            "variant_a": args.variants[0],
            "variant_b": args.variants[1],
            "mean_a": ma, "mean_b": mb,
            "diff_b_minus_a": diff,
            "relative_improvement_b_over_a_pct":
                -rel * 100,   # positive = b better (lower PPL)
            "cohen_d_a_minus_b": d,
            "welch_t": t, "welch_p_two_sided": p,
            "interpretation": (
                "positive relative_improvement_b_over_a_pct means variant b "
                "has lower val PPL than variant a; |d| >= 0.8 is 'large' "
                "effect in the usual Cohen cut-offs."
            ),
        }

    final = {"args": vars(args), "runs": runs, "summary": summary}
    Path(args.out).write_text(json.dumps(final, indent=2))

    # Stdout table.
    print("\n=== Summary ===")
    for v, r in summary["by_variant"].items():
        print(f"  {v:>10}: mean PPL = {r['mean']:.4f}  "
              f"std = {r['std']:.4f}  "
              f"(min {r['min']:.4f}, max {r['max']:.4f}, n={len(r['ppls'])})")
    c = summary["comparison"]
    if c:
        print(f"\n  {c['variant_b']} vs {c['variant_a']}: "
              f"Δmean = {c['diff_b_minus_a']:+.4f} PPL  "
              f"({c['relative_improvement_b_over_a_pct']:+.2f}% relative)")
        print(f"  Cohen's d = {c['cohen_d_a_minus_b']:+.3f}  "
              f"Welch t = {c['welch_t']:.3f}, p ≈ {c['welch_p_two_sided']:.3f}")


if __name__ == "__main__":
    main()
