"""Phase 71 — Hard-OOD-aware verification head training.

Phase 70 K-scaling found per-pair cross-FPR of 19.6% under deployment-honest
calibration (training-only thresholds), well above the in-sample 4% Phase 70
originally reported. This script tests whether *hard-OOD-aware training
negatives* — template-perturbed queries about entities NOT in the library
(and NOT in the Phase 65 hard-OOD test set) — close that gap.

Three training conditions per adapter, all using h_mean_L5:
  A — baseline:        4 self-positives + 19 cross-adapter train_prompts
                       (replicates Phase 70 protocol exactly)
  B — hard-OOD only:   4 self-positives + 32 hard-OOD-aware queries
  C — combined:        4 self-positives + 19 cross + 32 hard-OOD-aware

Hard-OOD-aware queries are template-perturbed: same template as the adapter's
training query, entity replaced with one drawn from a fresh vocabulary that
is disjoint from the 20 library entities AND from the 20 Phase 65 hard-OOD
test entities. 8 new entities per category × 4 paraphrases = 32 per category.

All thresholds calibrated on each head's training negatives at 95th
percentile (deployment-honest). Evaluated on the SAME held-out test set as
Phase 70 K-scaling: 60 in-library held-out paraphrases + Phase 65 hard-OOD
prompts (3 paraphrases each = 60 hard-OOD test queries — none in the
training distribution).

Pre-committed predictions:
  per-pair cross-FPR <= 5%   (open-vocab solved):     20%
  per-pair cross-FPR 6-12%   (partial improvement):  50%
  per-pair cross-FPR >= 13%  (no meaningful change):  30%

Reuses Phase 65 cache; no LoRA retraining.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase71_hard_ood_aware.py
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.phase10_passkey import load_model, generate_passkeys
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import apply_lora, load_lora_state_dict


SEED = 0
D = 1024
HIDDEN = 256
N_EPOCHS = 100
LR = 1e-3
PHASE65_CACHE = Path("results/identity_ae/phase65/cache")
PHASE71_DIR = Path("results/identity_ae/phase71")


# Disjoint hard-OOD-aware vocabulary (8 per category, all verified disjoint
# from library [0:5] and Phase 65 test [5:10] — see preflight check)
NEW_FACILITIES = ["tropical", "lunar", "arctic", "jungle", "polar", "urban", "glacial", "abyssal"]
NEW_SCIENTISTS = ["Dr. Hiroshi Tanaka", "Professor Maria Castillo", "Agent Rebecca Cho",
                  "Commander Liam O'Brien", "Director Aisha Patel", "Specialist Mateo Diaz",
                  "Operative Yuna Kim", "Researcher Theo Anderson"]
NEW_THINGS_UNITS = [("pendulum", "hertz"), ("calorimeter", "joules"),
                    ("oscilloscope", "decibels"), ("transponder", "lumens"),
                    ("gyroscope", "radians"), ("manometer", "pascals"),
                    ("voltmeter", "volts"), ("thermometer", "celsius")]
NEW_PROTOCOLS = ["Cascade", "Tempest", "Pinnacle", "Cipher", "Halcyon", "Crucible",
                 "Zenith", "Ember"]


def make_fake_test(category, *args):
    """Construct a generate_passkeys-style test dict with new entities,
    so train_paraphrase can templatize correctly."""
    if category == "numeric":
        fac = args[0]
        return {"type": "numeric",
                "prompt": f"What is the system access code for the {fac} facility?",
                "passage": f"The {fac} facility code is dummy.",
                "passkey": "dummy", "id": -1}
    if category == "entity":
        name = args[0]
        return {"type": "entity",
                "prompt": f"When did {name} make their breakthrough discovery?",
                "passage": f"{name} made a discovery on dummy date.",
                "passkey": "dummy", "id": -1}
    if category == "technical":
        thing, unit = args
        return {"type": "technical",
                "prompt": f"What is the critical threshold of the {thing} in {unit}?",
                "passage": f"The {thing} critical threshold is dummy {unit}.",
                "passkey": "dummy", "id": -1}
    if category == "fact":
        proto = args[0]
        return {"type": "fact",
                "prompt": f"How many signatories does the {proto} Protocol require?",
                "passage": f"The {proto} Protocol requires dummy signatories.",
                "passkey": "dummy", "id": -1}
    raise ValueError(category)


def build_hard_ood_aware_pool():
    """Per category: 8 new entities × 4 paraphrases (1 base + 3 train_paraphrase) = 32 prompts."""
    pool = {"numeric": [], "entity": [], "technical": [], "fact": []}
    for fac in NEW_FACILITIES:
        t = make_fake_test("numeric", fac)
        pool["numeric"].append(t["prompt"])
        for p in train_paraphrase(t):
            pool["numeric"].append(p)
    for name in NEW_SCIENTISTS:
        t = make_fake_test("entity", name)
        pool["entity"].append(t["prompt"])
        for p in train_paraphrase(t):
            pool["entity"].append(p)
    for thing, unit in NEW_THINGS_UNITS:
        t = make_fake_test("technical", thing, unit)
        pool["technical"].append(t["prompt"])
        for p in train_paraphrase(t):
            pool["technical"].append(p)
    for proto in NEW_PROTOCOLS:
        t = make_fake_test("fact", proto)
        pool["fact"].append(t["prompt"])
        for p in train_paraphrase(t):
            pool["fact"].append(p)
    return pool


# ============================================================
# L5 features, head, helpers (matches Phase 70)
# ============================================================

@torch.no_grad()
def l5_mean_only(model, ids_t):
    h = hidden_at_layer(model, ids_t, 5)
    return h.mean(dim=1).squeeze(0).cpu()


def cosine(a, b):
    return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))


class VerifHead(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(X_pos, X_neg, n_epochs=N_EPOCHS, lr=LR, device="cuda"):
    X = torch.cat([X_pos, X_neg]).to(device).float()
    y = torch.cat([torch.ones(X_pos.shape[0]),
                    torch.zeros(X_neg.shape[0])]).to(device)
    head = VerifHead().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr)
    pos_weight = torch.tensor([X_neg.shape[0] / max(X_pos.shape[0], 1)]).to(device)
    for _ in range(n_epochs):
        logits = head(X)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optim.zero_grad(); loss.backward(); optim.step()
    return head.eval().cpu()


def quantile(xs, q):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    pos = q * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def build_hard_ood_test_tests():
    """The Phase 65 hard-OOD test entities (entities 5..10 per category)."""
    all_tests = generate_passkeys(50)
    by_type = defaultdict(list)
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][5:10] + by_type["entity"][5:10]
            + by_type["technical"][5:10] + by_type["fact"][5:10])


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE71_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Library ----
    print(f"[A] loading Phase 65 library cache")
    blob = torch.load(PHASE65_CACHE / "library.pt", map_location="cpu", weights_only=False)
    library = blob["library"]
    keys_l0_per_adapter = [[k.cpu() if k.is_cuda else k for k in ks] for ks in blob["keys_l0"]]
    n_adapters = len(library)

    # ---- Build hard-OOD-aware training pool (disjoint from test) ----
    hood_pool = build_hard_ood_aware_pool()
    print(f"    hard-OOD-aware training pool per category: "
          f"{ {k: len(v) for k, v in hood_pool.items()} }")
    # Sanity preflight: no overlap with library or Phase 65 test prompts
    library_prompts = set(p for entry in library for p in entry["train_prompts"])
    hood_test_prompts = set()
    for t in build_hard_ood_test_tests():
        hood_test_prompts.add(t["prompt"])
        for p in held_out_paraphrase(t):
            hood_test_prompts.add(p)
    for cat, prompts in hood_pool.items():
        for p in prompts:
            assert p not in library_prompts, f"contamination: {p!r} in library"
            assert p not in hood_test_prompts, f"contamination: {p!r} in test"
    print(f"    preflight: hard-OOD-aware pool disjoint from library and test ✓")

    # ---- Build full query inventory ----
    queries = {}
    train_prompts_per = []
    held_out_per = []
    for i, entry in enumerate(library):
        tp = entry["train_prompts"]
        train_prompts_per.append(tp)
        for p in tp:
            qkey = ("train", i, p)
            ids = tokenizer.encode(p, add_special_tokens=False)[:512]
            queries[qkey] = {"text": p, "true_idx": i, "kind": "train", "ids": ids}
        ho = held_out_paraphrase(entry["test"])
        held_out_per.append(ho)
        for p in ho:
            qkey = ("ho", i, p)
            ids = tokenizer.encode(p, add_special_tokens=False)[:512]
            queries[qkey] = {"text": p, "true_idx": i, "kind": "in_library", "ids": ids}
    # Hard-OOD-aware TRAINING pool
    for cat, prompts in hood_pool.items():
        for j, p in enumerate(prompts):
            qkey = ("hood_train", cat, j, p)
            ids = tokenizer.encode(p, add_special_tokens=False)[:512]
            queries[qkey] = {"text": p, "category": cat, "kind": "hood_train", "ids": ids}
    # Hard-OOD TEST set (Phase 65) — 20 entities × 3 paraphrases = 60 queries
    test_tests = build_hard_ood_test_tests()
    test_per_category = defaultdict(list)
    for t in test_tests:
        test_per_category[t["type"]].append(t)
    for t in test_tests:
        for p in held_out_paraphrase(t):
            qkey = ("hood_test", t["type"], t["prompt"], p)
            ids = tokenizer.encode(p, add_special_tokens=False)[:512]
            queries[qkey] = {"text": p, "category": t["type"],
                              "kind": "hood_test", "ids": ids}
    print(f"    inventory: {len(queries)} unique queries "
          f"(80 train + 60 in-lib + {sum(len(v) for v in hood_pool.values())} hood-train + 60 hood-test)")

    # ---- Base model in eval mode ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    model.eval()
    reset_lora_to_zero(model)

    # ---- Per-adapter forward passes (mean L5 only) ----
    print(f"\n[B] forward-passing all queries through each adapter")
    t0 = time.time()
    adapter_mean = [dict() for _ in range(n_adapters)]
    for ai, entry in enumerate(library):
        sd_gpu = {k: v.to(device) for k, v in entry["sd"].items()}
        load_lora_state_dict(model, sd_gpu)
        for qkey, q in queries.items():
            ids_t = torch.tensor(q["ids"], dtype=torch.long).unsqueeze(0).to(device)
            adapter_mean[ai][qkey] = l5_mean_only(model, ids_t)
        reset_lora_to_zero(model)
        if (ai + 1) % 5 == 0:
            print(f"    [{ai+1:2d}/{n_adapters}] elapsed {time.time()-t0:.0f}s")

    # ---- Train 3 conditions per adapter ----
    print(f"\n[C] training 3 conditions × {n_adapters} = {3*n_adapters} heads")
    rng = random.Random(SEED)
    cond_heads = {"A": [], "B": [], "C": []}
    cond_taus  = {"A": [], "B": [], "C": []}
    cond_tneg  = {"A": [], "B": [], "C": []}  # for diagnostics

    for ai in range(n_adapters):
        ai_category = library[ai]["test"]["type"]
        # Self-positives (4 per adapter)
        pos_keys = [("train", ai, p) for p in train_prompts_per[ai]]
        X_pos = torch.stack([adapter_mean[ai][k] for k in pos_keys])

        # A: cross-adapter negatives only (matches Phase 70)
        cross_keys = []
        for aj in range(n_adapters):
            if aj == ai: continue
            chosen = rng.choice(train_prompts_per[aj])
            cross_keys.append(("train", aj, chosen))
        X_cross = torch.stack([adapter_mean[ai][k] for k in cross_keys])

        # B: hard-OOD-aware negatives only (same category as ai)
        hood_prompts = hood_pool[ai_category]
        hood_keys = []
        for j, p in enumerate(hood_prompts):
            qkey = ("hood_train", ai_category, j, p)
            hood_keys.append(qkey)
        X_hood = torch.stack([adapter_mean[ai][k] for k in hood_keys])

        # Train A
        head_A = train_head(X_pos, X_cross, device=device)
        with torch.no_grad():
            tneg_A = head_A(X_cross.cpu().float()).tolist()
        tau_A = quantile(tneg_A, 0.95)
        cond_heads["A"].append(head_A); cond_taus["A"].append(tau_A); cond_tneg["A"].append(tneg_A)

        # Train B
        head_B = train_head(X_pos, X_hood, device=device)
        with torch.no_grad():
            tneg_B = head_B(X_hood.cpu().float()).tolist()
        tau_B = quantile(tneg_B, 0.95)
        cond_heads["B"].append(head_B); cond_taus["B"].append(tau_B); cond_tneg["B"].append(tneg_B)

        # Train C
        X_neg_C = torch.cat([X_cross, X_hood])
        head_C = train_head(X_pos, X_neg_C, device=device)
        with torch.no_grad():
            tneg_C = head_C(X_neg_C.cpu().float()).tolist()
        tau_C = quantile(tneg_C, 0.95)
        cond_heads["C"].append(head_C); cond_taus["C"].append(tau_C); cond_tneg["C"].append(tneg_C)

    print(f"    tau ranges (median):  "
          f"A={sorted(cond_taus['A'])[10]:+.3f}  "
          f"B={sorted(cond_taus['B'])[10]:+.3f}  "
          f"C={sorted(cond_taus['C'])[10]:+.3f}")

    # ---- Compute routing matrix R[query, adapter] for end-to-end eval ----
    eval_qkeys = [k for k, q in queries.items() if q["kind"] in ("in_library", "hood_test")]
    n_eval = len(eval_qkeys)
    print(f"\n[D] eval set: {n_eval} queries "
          f"(60 in-lib + 60 hood-test)")
    print(f"    extracting base-L0 means for routing")

    R = torch.zeros(n_eval, n_adapters)
    for qi, qkey in enumerate(eval_qkeys):
        q = queries[qkey]
        ids_t = torch.tensor(q["ids"], dtype=torch.long).unsqueeze(0).to(device)
        h_l0 = model.drop(model.tok_emb(ids_t))
        q_l0 = h_l0.mean(dim=1).squeeze(0).cpu()
        for ai in range(n_adapters):
            best = -2.0
            for k in keys_l0_per_adapter[ai]:
                s = cosine(q_l0, k)
                if s > best:
                    best = s
            R[qi, ai] = best

    # ---- Compute head logit matrix per condition ----
    H = {cond: torch.zeros(n_eval, n_adapters) for cond in ["A", "B", "C"]}
    for qi, qkey in enumerate(eval_qkeys):
        for ai in range(n_adapters):
            feat = adapter_mean[ai][qkey].unsqueeze(0).float()
            for cond in ["A", "B", "C"]:
                with torch.no_grad():
                    H[cond][qi, ai] = cond_heads[cond][ai](feat).item()

    # ---- Per-condition metrics ----
    eval_kind = [queries[k]["kind"] for k in eval_qkeys]
    eval_true_idx = []
    for k in eval_qkeys:
        q = queries[k]
        if q["kind"] == "in_library":
            true_a = -1
            for i, ho in enumerate(held_out_per):
                if q["text"] in ho:
                    true_a = i
                    break
            eval_true_idx.append(true_a)
        else:
            eval_true_idx.append(-1)

    print(f"\n[E] per-condition metrics")
    print(f"  cond | per-pair cross-FPR | hard-OOD FPR | TPR_pool | E2E balanced")
    print(f"  -----+--------------------+--------------+----------+-------------")
    summary = {}
    for cond in ["A", "B", "C"]:
        # Per-pair cross-FPR: for adapter A's head, count its logits on OTHER
        # adapters' held-out queries that exceed tau_A
        cross_fp_count = 0
        cross_total = 0
        for qi, qkey in enumerate(eval_qkeys):
            if eval_kind[qi] != "in_library": continue
            true_a = eval_true_idx[qi]
            for ai in range(n_adapters):
                if ai == true_a: continue
                cross_total += 1
                if H[cond][qi, ai].item() >= cond_taus[cond][ai]:
                    cross_fp_count += 1
        pair_xfpr = cross_fp_count / max(cross_total, 1)

        # Hard-OOD FPR: pool across all adapters' heads on hood-test queries
        # (the heads' false-accept rate on out-of-library queries that passed
        # routing). Pool: for each (q, head) pair, accept = logit > tau.
        ood_fp_count = 0
        ood_total = 0
        for qi, qkey in enumerate(eval_qkeys):
            if eval_kind[qi] != "hood_test": continue
            for ai in range(n_adapters):
                ood_total += 1
                if H[cond][qi, ai].item() >= cond_taus[cond][ai]:
                    ood_fp_count += 1
        ood_fpr_pool = ood_fp_count / max(ood_total, 1)

        # TPR pooled: self-positives (in-library, head of true adapter) above tau
        tp_count = 0
        in_lib_total = 0
        for qi, qkey in enumerate(eval_qkeys):
            if eval_kind[qi] != "in_library": continue
            true_a = eval_true_idx[qi]
            in_lib_total += 1
            if H[cond][qi, true_a].item() >= cond_taus[cond][true_a]:
                tp_count += 1
        tpr_pool = tp_count / max(in_lib_total, 1)

        # End-to-end at K=20 (all adapters): route via C0b -> head -> decide
        n_tp = n_fn = n_fp = n_tn = 0
        for qi, qkey in enumerate(eval_qkeys):
            kind = eval_kind[qi]
            true_a = eval_true_idx[qi]
            routed = int(R[qi].argmax().item())
            logit = H[cond][qi, routed].item()
            accept = logit >= cond_taus[cond][routed]
            if kind == "in_library":
                if routed == true_a:
                    if accept: n_tp += 1
                    else:      n_fn += 1
                else:
                    if accept: n_fp += 1
                    else:      n_tn += 1
            else:  # hood_test (out-of-library always)
                if accept: n_fp += 1
                else:      n_tn += 1
        e2e_tpr = n_tp / max(n_tp + n_fn, 1)
        e2e_fpr = n_fp / max(n_fp + n_tn, 1)
        e2e_balanced = (e2e_tpr + (1 - e2e_fpr)) / 2

        print(f"   {cond}   |  {pair_xfpr:>7.1%}            "
              f"|  {ood_fpr_pool:>5.1%}      "
              f"|  {tpr_pool:>4.0%}    "
              f"|  {e2e_balanced:>5.1%}")

        summary[cond] = {
            "pair_cross_FPR": pair_xfpr,
            "hard_OOD_FPR_pool": ood_fpr_pool,
            "TPR_pool": tpr_pool,
            "end_to_end": {
                "TPR": e2e_tpr, "FPR": e2e_fpr, "balanced": e2e_balanced,
                "tp": n_tp, "fn": n_fn, "fp": n_fp, "tn": n_tn,
            },
        }

    # ---- Save JSON ----
    out_path = PHASE71_DIR / "hard_ood_aware.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"hidden": HIDDEN, "n_epochs": N_EPOCHS, "lr": LR, "seed": SEED,
                       "n_adapters": n_adapters,
                       "hood_pool_per_category": {k: len(v) for k, v in hood_pool.items()},
                       "n_hood_test_queries": 60,
                       "calibration": "training negatives, 95th percentile, per adapter"},
            "tau_per_adapter": {cond: cond_taus[cond] for cond in ["A", "B", "C"]},
            "summary": summary,
            "phase70_kscaling_baseline": {
                "pair_cross_FPR": 0.196,
                "end_to_end_balanced_K20": 0.652,
            },
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot: cross-FPR and balanced acc per condition ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    conds = ["A", "B", "C"]
    cond_labels = ["A: cross-only\n(Phase 70 baseline)", "B: hard-OOD only", "C: combined"]
    pair_xs = [summary[c]["pair_cross_FPR"] for c in conds]
    ood_fs = [summary[c]["hard_OOD_FPR_pool"] for c in conds]
    e2e_b = [summary[c]["end_to_end"]["balanced"] for c in conds]
    e2e_t = [summary[c]["end_to_end"]["TPR"] for c in conds]
    e2e_f = [summary[c]["end_to_end"]["FPR"] for c in conds]

    x = list(range(len(conds)))
    w = 0.35
    axes[0].bar([xi - w/2 for xi in x], pair_xs, width=w, color="C0", label="per-pair cross-FPR")
    axes[0].bar([xi + w/2 for xi in x], ood_fs, width=w, color="C3", label="pool hard-OOD FPR")
    axes[0].axhline(0.196, color="C0", ls="--", lw=0.7, alpha=0.7, label="Phase 70 K-scale x-FPR (0.196)")
    axes[0].axhline(0.05, color="gray", ls=":", lw=0.7, label="5% threshold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(cond_labels, fontsize=9)
    axes[0].set_ylabel("FPR"); axes[0].set_title("False-positive rates by training condition")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].bar([xi - w for xi in x], e2e_t, width=w*0.9, color="C2", label="TPR")
    axes[1].bar(x, e2e_f, width=w*0.9, color="C3", label="FPR")
    axes[1].bar([xi + w for xi in x], e2e_b, width=w*0.9, color="C0", label="balanced")
    axes[1].axhline(0.652, color="C0", ls="--", lw=0.7, alpha=0.7, label="Phase 70 K-scale balanced (0.652)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cond_labels, fontsize=9)
    axes[1].set_ylabel("rate"); axes[1].set_title("End-to-end deployment (K=20)")
    axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PHASE71_DIR / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict against pre-committed predictions ----
    print(f"\n{'='*72}\nVERDICT")
    print(f"{'='*72}")
    print(f"  Phase 70 K-scaling baseline (cond A reproducer): per-pair cross-FPR ~19.6%")
    for cond in ["A", "B", "C"]:
        s = summary[cond]
        print(f"  Cond {cond}: per-pair x-FPR {s['pair_cross_FPR']:.1%}  "
              f"hard-OOD FPR {s['hard_OOD_FPR_pool']:.1%}  "
              f"E2E balanced {s['end_to_end']['balanced']:.1%}")

    best_cond = min(["A", "B", "C"], key=lambda c: summary[c]["pair_cross_FPR"])
    best_xfpr = summary[best_cond]["pair_cross_FPR"]
    print(f"\n  best condition by cross-FPR: {best_cond} ({best_xfpr:.1%})")

    if best_xfpr <= 0.05:
        bucket = "<=5% (open-vocab solved)"
    elif best_xfpr <= 0.12:
        bucket = "6-12% (partial improvement)"
    else:
        bucket = ">=13% (no meaningful improvement)"
    print(f"  outcome bucket (pre-committed): {bucket}")
    print(f"  pre-commitments were: <=5% 20%, 6-12% 50%, >=13% 30%")


if __name__ == "__main__":
    main()
