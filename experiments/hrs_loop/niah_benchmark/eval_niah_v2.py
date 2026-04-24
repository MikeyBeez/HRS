"""Expanded NIAH benchmark eval.

Runs two configurations against a V18-compatible checkpoint:
  A) original 5 needles × 20 hardcoded distractors (reproduces task 2/3)
  B) expanded 25 needles × 40 distractors (20 hardcoded + 20 from WT-103)

Uses task 2's continuation-only recall methodology by reusing
engram_content_ablation.run_variant_on_needle with v1_baseline.

Usage:
    python eval_niah_v2.py --checkpoint results/v18_cross_attn/best.pt --label v18_baseline
    python eval_niah_v2.py --checkpoint results/v18_raft/checkpoint_5000.pt --label raft_final
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer
from niah_egr import DISTRACTORS as HARDCODED_DISTRACTORS  # 20 hardcoded paragraphs
from niah_egr import Needle

from experiments.hrs_loop.engram_content_ablation import (
    VARIANTS, run_variant_on_needle,
)

BENCHMARK_DIR = Path(__file__).parent
NEEDLES_PATH = BENCHMARK_DIR / "needles_v2.json"
RESULTS_DIR = BENCHMARK_DIR / "results"

EXTRACT_LAYER = 4
CROSS_ATTN_LAYERS = (1, 3, 5)
INV_SOFTPLUS_1 = math.log(math.e - 1.0)


# ------------------------------------------------------------
# Checkpoint loader (handles both V18 baseline and RAFT-tuned)
# ------------------------------------------------------------
def load_v18_compat_model(ckpt_path: Path, device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Same gate_scalar compat patch used in tasks 2 & 3: softplus(x)=1
    patched = 0
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_scalar") and any(name == m for m in missing):
            with torch.no_grad():
                p.fill_(INV_SOFTPLUS_1)
            patched += 1
    print(f"Loaded {ckpt_path.name}: step={ckpt.get('step', '?')} "
          f"missing={len(missing)} unexpected={len(unexpected)} patched_gate_scalar={patched}")

    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    return model, cfg


# ------------------------------------------------------------
# Distractors: 20 hardcoded + 20 from WT-103 validation
# ------------------------------------------------------------
def build_distractor_pool(n_wt103_extra: int = 20, seed: int = 42) -> list[str]:
    """Return 20 hardcoded + n_wt103_extra sampled from WT-103 validation."""
    assert len(HARDCODED_DISTRACTORS) == 20, f"expected 20, got {len(HARDCODED_DISTRACTORS)}"
    pool = list(HARDCODED_DISTRACTORS)

    if n_wt103_extra > 0:
        from datasets import load_dataset
        print(f"  Sampling {n_wt103_extra} extra distractors from WT-103 validation...")
        raw = load_dataset("wikitext", "wikitext-103-raw-v1")
        # Extract non-empty text paragraphs (filter out section headers like "= = X = =")
        paragraphs = [
            t.strip() for t in raw["validation"]["text"]
            if len(t.strip()) > 200 and not t.strip().startswith("=")
        ]
        rng = random.Random(seed)
        # Sample and truncate to ~80-word chunks so lengths match hardcoded set
        sampled = rng.sample(paragraphs, n_wt103_extra)
        trimmed = []
        for p in sampled:
            # Keep up to ~80 words
            words = p.split()
            if len(words) > 80:
                p = " ".join(words[:80])
            trimmed.append(p)
        pool.extend(trimmed)
    return pool


# ------------------------------------------------------------
# Load needles from JSON -> Needle dataclass
# ------------------------------------------------------------
def load_needles(path: Path = NEEDLES_PATH) -> list[dict]:
    return json.load(open(path))["needles"]


def needle_from_dict(d: dict, token_set: str = "answer_tokens") -> Needle:
    """Convert JSON entry to niah_egr.Needle dataclass.

    token_set: "answer_tokens" (default; matches task 4) or
               "cleaned_answer_tokens" (task 5; query echoes stripped).
    """
    return Needle(
        fact=d["fact"],
        query=d["query"],
        answer_tokens=d[token_set],
        category=d["category"],
    )


# ------------------------------------------------------------
# Benchmark runner
# ------------------------------------------------------------
def run_benchmark_config(
    model, tokenizer, device, needles: list[dict], distractors: list[str],
    max_new_tokens: int = 100, temperature: float = 0.9, top_k: int = 50,
    seed: int = 42, label: str = "", token_set: str = "answer_tokens",
) -> dict:
    """Run retrieval + continuation-only recall on a needle set."""
    torch.manual_seed(seed)
    build_cfg = {
        "extract_layer": EXTRACT_LAYER,
        "v2_layers": CROSS_ATTN_LAYERS,
        "v3_n": 32, "v4_n": 16,
    }
    build_fn = VARIANTS["v1_baseline"]

    per_needle = []
    t0 = time.time()
    for i, nd in enumerate(needles):
        needle = needle_from_dict(nd, token_set=token_set)
        r = run_variant_on_needle(
            model, tokenizer, device,
            "v1_baseline", build_fn, build_cfg,
            needle, distractors,
            temperature, top_k, max_new_tokens,
        )
        r["id"] = nd["id"]
        r["version"] = nd.get("version", "")
        per_needle.append(r)
        if (i + 1) % 5 == 0 or i == len(needles) - 1:
            print(f"  [{label}] {i+1}/{len(needles)}  elapsed={time.time()-t0:.0f}s")
    elapsed = time.time() - t0

    # Aggregate: mean rank, acc@1, continuation recall, per-needle-mean recall
    ranks = [r["rank"] for r in per_needle]
    mean_rank = sum(ranks) / len(ranks)
    acc_at_1 = sum(1 for r in ranks if r == 1) / len(ranks)
    total_hits = sum(r["n_hits"] for r in per_needle)
    total_possible = sum(r["n_answer_tokens"] for r in per_needle)
    recall_pooled = total_hits / total_possible  # hits / total token slots

    # Per-needle recall (in [0,1])
    per_rec = [r["n_hits"] / r["n_answer_tokens"] for r in per_needle]
    recall_mean = sum(per_rec) / len(per_rec)
    recall_var = sum((x - recall_mean) ** 2 for x in per_rec) / max(1, len(per_rec) - 1)
    recall_sem = math.sqrt(recall_var / len(per_rec))
    # 95% CI on pooled recall (binomial approximation, treating token slots as independent)
    p = recall_pooled
    ci95_half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / total_possible)

    return {
        "label": label,
        "token_set": token_set,
        "n_needles": len(per_needle),
        "n_distractors": len(distractors),
        "mean_rank": mean_rank,
        "acc_at_1": acc_at_1,
        "total_hits": total_hits,
        "total_possible": total_possible,
        "recall_pooled": recall_pooled,
        "recall_pooled_95ci": ci95_half,
        "recall_per_needle_mean": recall_mean,
        "recall_per_needle_sem": recall_sem,
        "elapsed_s": elapsed,
        "per_needle": per_needle,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint (e.g. results/v18_cross_attn/best.pt or results/v18_raft/checkpoint_5000.pt)")
    ap.add_argument("--label", type=str, required=True,
                    help="Short label for the output file (e.g. v18_baseline or raft_5000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--run", choices=["A", "B", "both"], default="both")
    ap.add_argument("--token-set", choices=["answer_tokens", "cleaned_answer_tokens"],
                    default="answer_tokens",
                    help="Which token list to score recall against")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO / args.checkpoint if not Path(args.checkpoint).is_absolute() \
        else Path(args.checkpoint)

    model, _cfg = load_v18_compat_model(ckpt_path, device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    needles_all = load_needles()
    needles_orig = [n for n in needles_all if n.get("version") == "original"]
    assert len(needles_orig) == 5, f"expected 5 originals, got {len(needles_orig)}"
    assert len(needles_all) >= 20, f"expected 20+ expanded, got {len(needles_all)}"

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"checkpoint": str(ckpt_path), "label": args.label, "seed": args.seed,
           "token_set": args.token_set}

    if args.run in ("A", "both"):
        print("\n=== Config A: original 5 needles + 20 hardcoded distractors ===")
        dist_A = build_distractor_pool(n_wt103_extra=0)
        resA = run_benchmark_config(
            model, tokenizer, device, needles_orig, dist_A,
            args.max_new_tokens, seed=args.seed, label=f"A[{args.label}]",
            token_set=args.token_set,
        )
        out["config_A"] = resA
        print(f"\n  Config A summary:")
        print(f"    retrieval acc@1      = {resA['acc_at_1']:.2f} ({int(resA['acc_at_1']*resA['n_needles'])}/{resA['n_needles']})")
        print(f"    retrieval mean rank  = {resA['mean_rank']:.2f}")
        print(f"    continuation recall  = {resA['recall_pooled']:.3f} "
              f"({resA['total_hits']}/{resA['total_possible']}) "
              f"±{resA['recall_pooled_95ci']:.3f} (95% CI)")
        print(f"    per-needle recall    = {resA['recall_per_needle_mean']:.3f} "
              f"±{resA['recall_per_needle_sem']:.3f} (SEM)")

    if args.run in ("B", "both"):
        print("\n=== Config B: expanded 25 needles + 40 distractors ===")
        dist_B = build_distractor_pool(n_wt103_extra=20, seed=args.seed)
        resB = run_benchmark_config(
            model, tokenizer, device, needles_all, dist_B,
            args.max_new_tokens, seed=args.seed, label=f"B[{args.label}]",
            token_set=args.token_set,
        )
        out["config_B"] = resB
        print(f"\n  Config B summary:")
        print(f"    retrieval acc@1      = {resB['acc_at_1']:.2f} ({int(resB['acc_at_1']*resB['n_needles'])}/{resB['n_needles']})")
        print(f"    retrieval mean rank  = {resB['mean_rank']:.2f}")
        print(f"    continuation recall  = {resB['recall_pooled']:.3f} "
              f"({resB['total_hits']}/{resB['total_possible']}) "
              f"±{resB['recall_pooled_95ci']:.3f} (95% CI)")
        print(f"    per-needle recall    = {resB['recall_per_needle_mean']:.3f} "
              f"±{resB['recall_per_needle_sem']:.3f} (SEM)")

    label_suffix = "_cleaned" if args.token_set == "cleaned_answer_tokens" else ""
    out_path = RESULTS_DIR / f"{args.label}{label_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
