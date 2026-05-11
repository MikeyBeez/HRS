"""Phase 70 follow-up: K-scaling sub-sample.

Phase 70 measured per-adapter h_mean_L5 verification heads on a 20-adapter
library and got 90% balanced accuracy with cross-FPR 4%, OOD-FPR 8%. The
caveat flagged in the README: "20-adapter library. Cross-adapter discrimination
at 4% FPR is measured against 19 same-template-different-entity sister
adapters. At 100s or 1000s of adapters, the cross-FPR could grow."

This script tests scaling within the data we have. For K in {2, 5, 10, 15, 20}
adapter subsamples, with multiple random seeds:

  (1) Per-adapter cross-FPR vs K: how does adapter A's head's false-accept
      rate against sister adapters' queries scale with the number of sisters?

  (2) End-to-end deployment vs K: subsample K adapters, route each query via
      C0b restricted to K, apply the routed adapter's verification head with
      its calibrated threshold, decide ACCEPT/REJECT. Measure TPR (in-library
      queries whose true adapter IS in K and got correctly verified) and FPR
      (false accepts: in-library queries whose true adapter is NOT in K, plus
      hard-OOD queries). Does end-to-end balanced accuracy hold at K=20?

Re-runs Phase 70's head training deterministically (seed=0) so heads match
the published Phase 70 numbers. Adds K-scaling analysis on top of the same
trained heads.

Reuses Phase 65 cache.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase70_kscaling.py
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
N_KSCALE_SEEDS = 50         # subsamples per K
K_VALUES = [2, 3, 5, 8, 12, 15, 18, 20]
PHASE65_CACHE = Path("results/identity_ae/phase65/cache")
PHASE70K_DIR = Path("results/identity_ae/phase70")


@torch.no_grad()
def l5_features(model, ids_t):
    h = hidden_at_layer(model, ids_t, 5)
    return h[:, -1, :].squeeze(0).cpu(), h.mean(dim=1).squeeze(0).cpu()


def build_hard_ood_tests():
    all_tests = generate_passkeys(50)
    by_type = defaultdict(list)
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][5:10] + by_type["entity"][5:10]
            + by_type["technical"][5:10] + by_type["fact"][5:10])


class VerifHead(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

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


def cosine(a, b):
    return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE70K_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Library ----
    print(f"[A] loading Phase 65 library cache")
    blob = torch.load(PHASE65_CACHE / "library.pt", map_location="cpu", weights_only=False)
    library = blob["library"]
    keys_l0_per_adapter = [[k.cpu() if k.is_cuda else k for k in ks] for ks in blob["keys_l0"]]
    n_adapters = len(library)

    # ---- Build query inventory (same as Phase 70) ----
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
    hard_ood_tests = build_hard_ood_tests()
    hard_ood_prompts = [t["prompt"] for t in hard_ood_tests]
    for j, ood in enumerate(hard_ood_tests):
        qkey = ("ood", j, ood["prompt"])
        ids = tokenizer.encode(ood["prompt"], add_special_tokens=False)[:512]
        queries[qkey] = {"text": ood["prompt"], "true_idx": -1, "kind": "hard_ood", "ids": ids}

    print(f"    {sum(len(p) for p in train_prompts_per)} train + "
          f"{sum(len(p) for p in held_out_per)} held-out + "
          f"{len(hard_ood_prompts)} hard-OOD = {len(queries)} unique queries")

    # ---- Base model in eval mode ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    model.eval()
    reset_lora_to_zero(model)

    # ---- Per-adapter L5 mean features for all queries ----
    print(f"\n[B] forward-passing all queries through each adapter")
    t0 = time.time()
    # adapter_mean_features[ai][qkey] = mean-pool L5 (D,)
    adapter_mean = [dict() for _ in range(n_adapters)]
    for ai, entry in enumerate(library):
        sd = entry["sd"]
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)
        for qkey, q in queries.items():
            ids_t = torch.tensor(q["ids"], dtype=torch.long).unsqueeze(0).to(device)
            _, mean = l5_features(model, ids_t)
            adapter_mean[ai][qkey] = mean
        reset_lora_to_zero(model)
        if (ai + 1) % 5 == 0:
            print(f"    [{ai+1:2d}/{n_adapters}] elapsed {time.time()-t0:.0f}s")

    # ---- Train h_mean_L5 heads (matches Phase 70 protocol) ----
    print(f"\n[C] training {n_adapters} h_mean_L5 verification heads")
    rng = random.Random(SEED)
    heads = []
    train_neg_scores = []  # per-adapter list of negative training scores for tau calibration
    for ai in range(n_adapters):
        pos_keys = [("train", ai, p) for p in train_prompts_per[ai]]
        neg_keys = []
        for aj in range(n_adapters):
            if aj == ai: continue
            chosen = rng.choice(train_prompts_per[aj])
            neg_keys.append(("train", aj, chosen))
        X_pos = torch.stack([adapter_mean[ai][k] for k in pos_keys])
        X_neg = torch.stack([adapter_mean[ai][k] for k in neg_keys])
        head = train_head(X_pos, X_neg, device=device)
        heads.append(head)
        # Calibrate tau on training negatives at 95% specificity
        with torch.no_grad():
            neg_logits = head(X_neg.cpu().float()).tolist()
        train_neg_scores.append(neg_logits)

    # Calibrated thresholds per adapter
    tau_per_adapter = [quantile(s, 0.95) for s in train_neg_scores]
    print(f"    tau range: min {min(tau_per_adapter):+.3f}  "
          f"median {sorted(tau_per_adapter)[n_adapters//2]:+.3f}  "
          f"max {max(tau_per_adapter):+.3f}")

    # ---- Build full routing matrix and head-logit matrix ----
    print(f"\n[D] computing routing and head matrices for all queries")
    qkeys_eval = [k for k, q in queries.items() if q["kind"] in ("in_library", "hard_ood")]
    n_eval = len(qkeys_eval)
    print(f"    eval queries: {n_eval}")

    # Routing matrix: R[query, adapter] = max cosine(query_l0, adapter's L0 keys)
    # We need query L0 features (under base model, LoRA reset) — re-extract
    print(f"    extracting base-L0 means for routing")
    R = torch.zeros(n_eval, n_adapters)
    for qi, qkey in enumerate(qkeys_eval):
        q = queries[qkey]
        ids_t = torch.tensor(q["ids"], dtype=torch.long).unsqueeze(0).to(device)
        h_l0 = model.drop(model.tok_emb(ids_t))  # (1, T, D)
        q_l0 = h_l0.mean(dim=1).squeeze(0).cpu()
        for ai in range(n_adapters):
            best = -2.0
            for k in keys_l0_per_adapter[ai]:
                s = cosine(q_l0, k)
                if s > best:
                    best = s
            R[qi, ai] = best

    # Head logit matrix: H[query, adapter] = head[adapter](adapter_mean[adapter][query])
    H = torch.zeros(n_eval, n_adapters)
    for qi, qkey in enumerate(qkeys_eval):
        for ai in range(n_adapters):
            with torch.no_grad():
                feat = adapter_mean[ai][qkey].unsqueeze(0).float()
                H[qi, ai] = heads[ai](feat).item()

    # Evaluation labels
    eval_true_idx = torch.tensor([queries[k]["true_idx"] for k in qkeys_eval])
    eval_kind = [queries[k]["kind"] for k in qkeys_eval]

    print(f"    routing matrix: {tuple(R.shape)}, head matrix: {tuple(H.shape)}")

    # ---- (1) Per-adapter cross-FPR vs K ----
    print(f"\n[E] per-adapter cross-FPR vs K")
    # For each adapter A, compute its head's logit on every other adapter's
    # held-out queries, then thresholded vs tau_A. As K random sister adapters
    # are sampled, what fraction trigger A's head?
    # Pre-compute: cross_logits_per_adapter[A] = list of (sister_adapter_idx, logit)
    cross_logits = defaultdict(list)
    for qi, qkey in enumerate(qkeys_eval):
        if eval_kind[qi] != "in_library":
            continue
        true_a = int(eval_true_idx[qi].item())
        for ai in range(n_adapters):
            if ai == true_a: continue
            cross_logits[ai].append((true_a, float(H[qi, ai].item())))

    rng_k = random.Random(SEED)
    cross_fpr_by_k = {}
    for K in K_VALUES:
        n_competitors = K - 1
        if n_competitors < 1: continue
        adapter_fprs = []
        for ai in range(n_adapters):
            entries = cross_logits[ai]
            # Group by sister adapter
            by_sister = defaultdict(list)
            for sid, l in entries:
                by_sister[sid].append(l)
            other_sisters = list(by_sister.keys())
            if n_competitors > len(other_sisters):
                n_use = len(other_sisters)
            else:
                n_use = n_competitors
            seed_fprs = []
            for s in range(N_KSCALE_SEEDS):
                rng_local = random.Random(SEED * 1000 + s + K * 10000 + ai)
                sample = rng_local.sample(other_sisters, n_use)
                logits = [l for sid in sample for l in by_sister[sid]]
                fpr = sum(1 for l in logits if l >= tau_per_adapter[ai]) / max(len(logits), 1)
                seed_fprs.append(fpr)
            adapter_fprs.append(sum(seed_fprs) / len(seed_fprs))
        mean_fpr = sum(adapter_fprs) / len(adapter_fprs)
        std_fpr = (sum((f - mean_fpr)**2 for f in adapter_fprs) / len(adapter_fprs))**0.5
        cross_fpr_by_k[K] = {"mean": mean_fpr, "std": std_fpr,
                              "per_adapter": adapter_fprs}
        print(f"    K={K:>2d}  cross-FPR mean {mean_fpr:.1%} +/- {std_fpr:.1%} "
              f"(across {n_adapters} adapters)")

    # ---- (2) End-to-end deployment vs K ----
    print(f"\n[F] end-to-end deployment TPR/FPR vs K")
    # For each (K, seed): subsample K adapters; for each query route via R subset,
    # apply head's threshold, decide. Compute TPR and FPR.
    deployment_by_k = {}
    for K in K_VALUES:
        seed_results = []
        for s in range(N_KSCALE_SEEDS):
            rng_local = random.Random(SEED * 1000 + s + K * 100000)
            subset = sorted(rng_local.sample(range(n_adapters), K))
            n_tp = n_fn = n_fp = n_tn = 0
            n_in_subset_eligible = 0
            n_should_reject = 0
            for qi, qkey in enumerate(qkeys_eval):
                kind = eval_kind[qi]
                true_a = int(eval_true_idx[qi].item())
                # Route within subset
                routing_scores = R[qi, subset]
                routed_local = int(routing_scores.argmax().item())
                routed_a = subset[routed_local]
                # Verification logit + threshold for routed_a
                logit = float(H[qi, routed_a].item())
                accept = logit >= tau_per_adapter[routed_a]
                # True label: TRUST iff in-library AND true_a in subset AND routed_a == true_a
                if kind == "in_library":
                    if true_a in subset:
                        n_in_subset_eligible += 1
                        if routed_a == true_a:
                            # eligible TRUST case
                            if accept: n_tp += 1
                            else:      n_fn += 1
                        else:
                            # mis-routed within subset: should be rejected
                            n_should_reject += 1
                            if accept: n_fp += 1
                            else:      n_tn += 1
                    else:
                        # true adapter not in deployed library: must reject
                        n_should_reject += 1
                        if accept: n_fp += 1
                        else:      n_tn += 1
                else:  # hard_ood
                    n_should_reject += 1
                    if accept: n_fp += 1
                    else:      n_tn += 1
            tpr = n_tp / max(n_tp + n_fn, 1)
            fpr = n_fp / max(n_fp + n_tn, 1)
            balanced = (tpr + (1 - fpr)) / 2
            seed_results.append({"tpr": tpr, "fpr": fpr, "balanced": balanced,
                                  "tp": n_tp, "fn": n_fn, "fp": n_fp, "tn": n_tn,
                                  "n_eligible": n_in_subset_eligible,
                                  "n_should_reject": n_should_reject})
        tprs = [r["tpr"] for r in seed_results]
        fprs = [r["fpr"] for r in seed_results]
        bals = [r["balanced"] for r in seed_results]
        deployment_by_k[K] = {
            "tpr_mean": sum(tprs)/len(tprs), "tpr_std": (sum((x - sum(tprs)/len(tprs))**2 for x in tprs)/len(tprs))**0.5,
            "fpr_mean": sum(fprs)/len(fprs), "fpr_std": (sum((x - sum(fprs)/len(fprs))**2 for x in fprs)/len(fprs))**0.5,
            "balanced_mean": sum(bals)/len(bals), "balanced_std": (sum((x - sum(bals)/len(bals))**2 for x in bals)/len(bals))**0.5,
        }
        d = deployment_by_k[K]
        print(f"    K={K:>2d}  TPR {d['tpr_mean']:.1%} +/- {d['tpr_std']:.1%}  "
              f"FPR {d['fpr_mean']:.1%} +/- {d['fpr_std']:.1%}  "
              f"balanced {d['balanced_mean']:.1%} +/- {d['balanced_std']:.1%}")

    # ---- Save JSON ----
    out_path = PHASE70K_DIR / "kscaling.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"k_values": K_VALUES, "n_seeds_per_k": N_KSCALE_SEEDS,
                       "n_adapters_total": n_adapters, "head": "h_mean_L5"},
            "tau_per_adapter": tau_per_adapter,
            "cross_fpr_by_k": cross_fpr_by_k,
            "deployment_by_k": deployment_by_k,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    Ks = sorted(cross_fpr_by_k.keys())
    cross_means = [cross_fpr_by_k[k]["mean"] for k in Ks]
    cross_stds = [cross_fpr_by_k[k]["std"] for k in Ks]
    axes[0].errorbar(Ks, cross_means, yerr=cross_stds, marker="o", color="C0", capsize=3)
    axes[0].set_xlabel("K (adapter library size)")
    axes[0].set_ylabel("per-adapter cross-FPR")
    axes[0].set_title("Per-adapter cross-FPR vs K\n(head's false-accept rate "
                      "against K-1 sister adapters)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0.05, color="gray", ls="--", lw=0.7, label="5% reference")
    axes[0].legend(fontsize=9)

    Ks_d = sorted(deployment_by_k.keys())
    tpr_m = [deployment_by_k[k]["tpr_mean"] for k in Ks_d]
    tpr_s = [deployment_by_k[k]["tpr_std"]  for k in Ks_d]
    fpr_m = [deployment_by_k[k]["fpr_mean"] for k in Ks_d]
    fpr_s = [deployment_by_k[k]["fpr_std"]  for k in Ks_d]
    bal_m = [deployment_by_k[k]["balanced_mean"] for k in Ks_d]
    bal_s = [deployment_by_k[k]["balanced_std"]  for k in Ks_d]
    axes[1].errorbar(Ks_d, tpr_m, yerr=tpr_s, marker="o", color="C2", capsize=3, label="TPR")
    axes[1].errorbar(Ks_d, fpr_m, yerr=fpr_s, marker="s", color="C3", capsize=3, label="FPR")
    axes[1].errorbar(Ks_d, bal_m, yerr=bal_s, marker="^", color="C0", capsize=3, label="balanced")
    axes[1].set_xlabel("K (adapter library size)")
    axes[1].set_ylabel("rate")
    axes[1].set_title("End-to-end deployment vs K\n"
                      "(route in K-subset -> verify -> decide)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plot_path = PHASE70K_DIR / "kscaling.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict ----
    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    cross_at_k20 = cross_fpr_by_k[20]["mean"] if 20 in cross_fpr_by_k else None
    cross_at_k2  = cross_fpr_by_k[2]["mean"]  if 2  in cross_fpr_by_k else None
    if cross_at_k20 is not None and cross_at_k2 is not None:
        scaling_factor = cross_at_k20 / max(cross_at_k2, 1e-6)
        print(f"  per-adapter cross-FPR: K=2 -> {cross_at_k2:.1%}, "
              f"K=20 -> {cross_at_k20:.1%} ({scaling_factor:.1f}x)")
    bal_at_k20 = deployment_by_k[20]["balanced_mean"] if 20 in deployment_by_k else None
    bal_at_k2  = deployment_by_k[2]["balanced_mean"]  if 2  in deployment_by_k else None
    if bal_at_k20 is not None and bal_at_k2 is not None:
        print(f"  end-to-end balanced acc: K=2 -> {bal_at_k2:.1%}, "
              f"K=20 -> {bal_at_k20:.1%}")


if __name__ == "__main__":
    main()
