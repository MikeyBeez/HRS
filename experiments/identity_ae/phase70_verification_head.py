"""Phase 70 — Per-adapter verification head.

Phase 69 found that metadata entity-match gives 100% balanced accuracy when
entity vocabularies are enumerable. The architectural follow-up: can the same
discrimination be recovered from the adapter's hidden states alone, without
any metadata lookup? This matters for open-vocab deployments where you can't
enumerate entities at route time.

Procedure:

  1. Pre-compute base-model L5 hidden representations for all unique queries
     (180 queries: 80 training prompts + 60 held-out paraphrases + 20 hard-OOD).

  2. For each of the 20 adapters:
     a. Load adapter (frozen base + active LoRA on layers 4-5).
     b. Forward-pass each query, extract three representations:
          h_last_L5             : last-token L5 hidden (adapter active)
          h_mean_L5             : mean-pooled L5 hidden (adapter active)
          h_last_minus_base_L5  : last-token L5 (adapter) minus last-token L5 (base)
                                  i.e. what did this adapter change?
     c. Build a balanced training set:
          4 positives  : adapter's own train_prompts
          19 negatives : one train_prompt sampled from each other adapter
     d. Train 3 small MLPs (256 hidden, BCE, AdamW, 100 epochs) on the
        three representation types.
     e. Score every (query, adapter) pair on the held-out + hard-OOD sets.

  3. Aggregate: for each representation type, compute per-adapter and pooled:
       TPR @ 95% specificity, hard-OOD FPR, cross-adapter FPR, balanced acc.

Reuses Phase 65 cache (library state dicts) and the existing paraphrase
pipeline. Model in eval mode throughout.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase70_verification_head.py
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
PHASE70_DIR = Path("results/identity_ae/phase70")


# ============================================================
# Hidden-state extraction
# ============================================================

@torch.no_grad()
def l5_features(model, ids_t):
    """Return (last_token_l5, mean_l5) as CPU tensors of shape (D,)."""
    h = hidden_at_layer(model, ids_t, 5)  # (1, T, D)
    return h[:, -1, :].squeeze(0).cpu(), h.mean(dim=1).squeeze(0).cpu()


# ============================================================
# Hard-OOD construction (mirrors Phase 65)
# ============================================================

def build_hard_ood_tests():
    all_tests = generate_passkeys(50)
    by_type = defaultdict(list)
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][5:10] + by_type["entity"][5:10]
            + by_type["technical"][5:10] + by_type["fact"][5:10])


# ============================================================
# Verification head
# ============================================================

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
    """Train a head on positive/negative features. Returns the trained head (CPU)."""
    X = torch.cat([X_pos, X_neg]).to(device).float()
    y = torch.cat([
        torch.ones(X_pos.shape[0]),
        torch.zeros(X_neg.shape[0]),
    ]).to(device)

    head = VerifHead().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr)
    # Class-balanced BCE weights to handle the 4-vs-19 imbalance
    pos_weight = torch.tensor([X_neg.shape[0] / max(X_pos.shape[0], 1)]).to(device)
    for epoch in range(n_epochs):
        logits = head(X)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return head.eval().cpu()


# ============================================================
# Calibration helpers
# ============================================================

def quantile(xs, q):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    pos = q * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def tpr_at_specificity(pos_scores, neg_scores, target_spec=0.95):
    """At threshold tau set so target_spec of negatives have score < tau,
    what fraction of positives have score >= tau?"""
    if not pos_scores or not neg_scores:
        return float("nan"), float("nan")
    tau = quantile(neg_scores, target_spec)
    tpr = sum(1 for s in pos_scores if s >= tau) / len(pos_scores)
    return tpr, tau


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE70_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load library ----
    print(f"[A] loading Phase 65 cache from {PHASE65_CACHE}")
    blob = torch.load(PHASE65_CACHE / "library.pt", map_location="cpu", weights_only=False)
    library = blob["library"]
    n_adapters = len(library)
    print(f"    {n_adapters} adapters loaded")

    # ---- Build query inventory (with disjoint train/eval splits) ----
    # train_prompts (per adapter): the 4 prompts used to train the LoRA adapter
    # held_out_paras (per adapter): 3 unseen paraphrases reserved for evaluation
    # hard_ood_prompts: 20 entity-substituted near-neighbors (no paraphrasing)
    queries = {}  # global key -> {"text": str, "category": str, "ids": list}

    train_prompts_per_adapter = []
    for i, entry in enumerate(library):
        adapter_train = entry["train_prompts"]      # 4 strings
        train_prompts_per_adapter.append(adapter_train)
        for p in adapter_train:
            qkey = ("train", i, p)
            ids = tokenizer.encode(p, add_special_tokens=False)[:512]
            queries[qkey] = {"text": p, "category": entry["test"]["type"], "ids": ids}

    held_out_per_adapter = []
    for i, entry in enumerate(library):
        ho = held_out_paraphrase(entry["test"])     # 3 strings
        held_out_per_adapter.append(ho)
        for p in ho:
            qkey = ("ho", i, p)
            ids = tokenizer.encode(p, add_special_tokens=False)[:512]
            queries[qkey] = {"text": p, "category": entry["test"]["type"], "ids": ids}

    hard_ood = build_hard_ood_tests()                # 20 hard-OOD test dicts
    hard_ood_prompts = [t["prompt"] for t in hard_ood]
    for j, ood_test in enumerate(hard_ood):
        qkey = ("ood", j, ood_test["prompt"])
        ids = tokenizer.encode(ood_test["prompt"], add_special_tokens=False)[:512]
        queries[qkey] = {"text": ood_test["prompt"], "category": ood_test["type"], "ids": ids}

    print(f"    inventory: {sum(len(p) for p in train_prompts_per_adapter)} train + "
          f"{sum(len(p) for p in held_out_per_adapter)} held-out + "
          f"{len(hard_ood_prompts)} hard-OOD = {len(queries)} unique queries")

    # ---- Build base model in eval mode ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    model.eval()
    reset_lora_to_zero(model)

    # ---- Phase 1: cache base L5 features for all queries ----
    print(f"\n[B] caching base L5 (last + mean) for {len(queries)} unique queries")
    base_features = {}
    t0 = time.time()
    for qkey, q in queries.items():
        ids_t = torch.tensor(q["ids"], dtype=torch.long).unsqueeze(0).to(device)
        last, mean = l5_features(model, ids_t)
        base_features[qkey] = {"last": last, "mean": mean}
    print(f"    {len(base_features)} base feature pairs in {time.time()-t0:.0f}s")

    # ---- Phase 2: per-adapter loop ----
    # adapter_features[adapter_idx][qkey] = {"last": ..., "mean": ..., "delta_last": ...}
    adapter_features = [dict() for _ in range(n_adapters)]
    print(f"\n[C] per-adapter forward passes (loading 20 adapters)")
    t0 = time.time()
    for ai, entry in enumerate(library):
        sd = entry["sd"]
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)
        for qkey, q in queries.items():
            ids_t = torch.tensor(q["ids"], dtype=torch.long).unsqueeze(0).to(device)
            last, mean = l5_features(model, ids_t)
            delta_last = last - base_features[qkey]["last"]
            adapter_features[ai][qkey] = {
                "last": last, "mean": mean, "delta_last": delta_last,
            }
        reset_lora_to_zero(model)
        if (ai + 1) % 5 == 0:
            print(f"    [{ai+1:2d}/{n_adapters}] elapsed {time.time()-t0:.0f}s")

    # ---- Phase 3: train heads ----
    # For each adapter, train 3 heads (one per representation type)
    print(f"\n[D] training 3 x {n_adapters} = {3*n_adapters} verification heads")
    rng = random.Random(SEED)
    rep_names = ["h_last_L5", "h_mean_L5", "h_last_minus_base_L5"]
    rep_keys  = ["last",      "mean",       "delta_last"]
    heads = [{}  for _ in range(n_adapters)]   # heads[adapter_idx][rep_name] = trained MLP

    for ai in range(n_adapters):
        # Positives: this adapter's 4 train_prompts (under THIS adapter's forward pass)
        pos_keys = [("train", ai, p) for p in train_prompts_per_adapter[ai]]

        # Negatives: 1 train_prompt sampled from each of the other 19 adapters,
        # forward-passed under THIS adapter (i.e. how does adapter ai see them?)
        neg_keys = []
        for aj in range(n_adapters):
            if aj == ai:
                continue
            chosen = rng.choice(train_prompts_per_adapter[aj])
            neg_keys.append(("train", aj, chosen))

        for rep_name, rep_key in zip(rep_names, rep_keys):
            X_pos = torch.stack([adapter_features[ai][k][rep_key] for k in pos_keys])
            X_neg = torch.stack([adapter_features[ai][k][rep_key] for k in neg_keys])
            head = train_head(X_pos, X_neg, device=device)
            heads[ai][rep_name] = head

    # ---- Phase 4: score eval set under each adapter's heads ----
    # For adapter ai, evaluate against all eval queries (held_out + hard_ood),
    # but use ai's forward-pass features for them.
    print(f"\n[E] scoring eval queries under each adapter's heads")
    # results[rep_name] = {"self_pos": [...], "cross_neg": [...], "ood_neg": [...]}
    # Per-adapter breakdowns also kept.
    rep_results = {rn: {"self_pos": [], "cross_neg": [], "ood_neg": [],
                          "per_adapter": []}
                    for rn in rep_names}

    for ai in range(n_adapters):
        per_adapter_record = {ai_name: {"self_pos": [], "cross_neg": [], "ood_neg": []}
                                for ai_name in rep_names}
        # Self-positives: ai's own held-out paraphrases
        for p in held_out_per_adapter[ai]:
            qkey = ("ho", ai, p)
            for rep_name, rep_key in zip(rep_names, rep_keys):
                feat = adapter_features[ai][qkey][rep_key].unsqueeze(0)
                with torch.no_grad():
                    logit = heads[ai][rep_name](feat).item()
                rep_results[rep_name]["self_pos"].append(logit)
                per_adapter_record[rep_name]["self_pos"].append(logit)

        # Cross-adapter negatives: other adapters' held-out paraphrases
        for aj in range(n_adapters):
            if aj == ai:
                continue
            for p in held_out_per_adapter[aj]:
                qkey = ("ho", aj, p)
                for rep_name, rep_key in zip(rep_names, rep_keys):
                    feat = adapter_features[ai][qkey][rep_key].unsqueeze(0)
                    with torch.no_grad():
                        logit = heads[ai][rep_name](feat).item()
                    rep_results[rep_name]["cross_neg"].append(logit)
                    per_adapter_record[rep_name]["cross_neg"].append(logit)

        # Hard-OOD negatives: same 20 OOD prompts, scored under ai
        for j, p in enumerate(hard_ood_prompts):
            qkey = ("ood", j, p)
            for rep_name, rep_key in zip(rep_names, rep_keys):
                feat = adapter_features[ai][qkey][rep_key].unsqueeze(0)
                with torch.no_grad():
                    logit = heads[ai][rep_name](feat).item()
                rep_results[rep_name]["ood_neg"].append(logit)
                per_adapter_record[rep_name]["ood_neg"].append(logit)

        for rep_name in rep_names:
            rep_results[rep_name]["per_adapter"].append(per_adapter_record[rep_name])

    # ---- Phase 5: metrics per representation ----
    print(f"\n{'='*72}\nRESULTS")
    print(f"{'='*72}")
    print(f"  {'rep':<22s} {'pooled_TPR@95':>14s} {'pooled_OOD_FPR':>15s} "
          f"{'pooled_xFPR':>12s} {'pooled_bal':>11s}")

    summaries = {}
    for rep_name in rep_names:
        r = rep_results[rep_name]
        all_neg = r["cross_neg"] + r["ood_neg"]
        # Pooled: TPR at 95% spec on all-negatives
        tpr_pool, tau_pool = tpr_at_specificity(r["self_pos"], all_neg, 0.95)
        # Decompose: at the same pooled tau, what's OOD FPR vs cross FPR?
        ood_fpr = sum(1 for s in r["ood_neg"]   if s >= tau_pool) / max(len(r["ood_neg"]),  1)
        x_fpr   = sum(1 for s in r["cross_neg"] if s >= tau_pool) / max(len(r["cross_neg"]),1)
        # Balanced: (TPR + (1 - all_neg_FPR)) / 2 at this tau
        all_neg_fpr = sum(1 for s in all_neg if s >= tau_pool) / max(len(all_neg), 1)
        bal_acc = (tpr_pool + (1 - all_neg_fpr)) / 2

        # Per-adapter calibration: each adapter sets its own tau on its own negatives
        per_adapter_tprs, per_adapter_oods, per_adapter_xs = [], [], []
        for record in r["per_adapter"]:
            adapter_neg = record["cross_neg"] + record["ood_neg"]
            tpr, tau_a = tpr_at_specificity(record["self_pos"], adapter_neg, 0.95)
            if not (tpr != tpr):  # not nan
                per_adapter_tprs.append(tpr)
            ofpr = sum(1 for s in record["ood_neg"]   if s >= tau_a) / max(len(record["ood_neg"]),  1)
            xfpr = sum(1 for s in record["cross_neg"] if s >= tau_a) / max(len(record["cross_neg"]),1)
            per_adapter_oods.append(ofpr)
            per_adapter_xs.append(xfpr)
        mean_per_adapter_tpr = sum(per_adapter_tprs) / max(len(per_adapter_tprs), 1)
        mean_per_adapter_ofpr = sum(per_adapter_oods) / max(len(per_adapter_oods), 1)
        mean_per_adapter_xfpr = sum(per_adapter_xs) / max(len(per_adapter_xs), 1)

        print(f"  {rep_name:<22s} {tpr_pool:>14.0%} {ood_fpr:>15.0%} "
              f"{x_fpr:>12.0%} {bal_acc:>11.0%}")
        print(f"    per-adapter mean: TPR@95spec {mean_per_adapter_tpr:.0%}  "
              f"OOD-FPR {mean_per_adapter_ofpr:.0%}  cross-FPR {mean_per_adapter_xfpr:.0%}")

        summaries[rep_name] = {
            "pooled": {
                "tau_at_95_spec": tau_pool,
                "TPR_at_95_spec": tpr_pool,
                "OOD_FPR_at_95_spec": ood_fpr,
                "cross_adapter_FPR_at_95_spec": x_fpr,
                "balanced_accuracy": bal_acc,
                "n_self_pos": len(r["self_pos"]),
                "n_cross_neg": len(r["cross_neg"]),
                "n_ood_neg":  len(r["ood_neg"]),
            },
            "per_adapter_mean": {
                "TPR_at_95_spec": mean_per_adapter_tpr,
                "OOD_FPR_at_95_spec": mean_per_adapter_ofpr,
                "cross_adapter_FPR_at_95_spec": mean_per_adapter_xfpr,
            },
            "per_adapter_records": [{
                "tpr_at_95_self": (sum(1 for s in record["self_pos"] if
                                        s >= quantile(record["cross_neg"]+record["ood_neg"], 0.95))
                                    / max(len(record["self_pos"]), 1))
                                    if record["cross_neg"]+record["ood_neg"] else float("nan"),
                "self_pos_scores": record["self_pos"],
                "cross_neg_scores": record["cross_neg"],
                "ood_neg_scores": record["ood_neg"],
            } for record in r["per_adapter"]],
        }

    # ---- Phase 6: plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, rep_name in zip(axes, rep_names):
        r = rep_results[rep_name]
        all_scores = r["self_pos"] + r["cross_neg"] + r["ood_neg"]
        lo, hi = min(all_scores), max(all_scores)
        bins = [lo + i * (hi - lo) / 30 for i in range(31)]
        ax.hist(r["self_pos"],  bins=bins, alpha=0.7, color="C2", label=f"self pos (n={len(r['self_pos'])})")
        ax.hist(r["cross_neg"], bins=bins, alpha=0.5, color="C1", label=f"cross neg (n={len(r['cross_neg'])})")
        ax.hist(r["ood_neg"],   bins=bins, alpha=0.6, color="C3", label=f"hard-OOD neg (n={len(r['ood_neg'])})")
        ax.axvline(summaries[rep_name]["pooled"]["tau_at_95_spec"],
                   color="black", ls="--", lw=1, label="tau@95spec")
        ax.set_title(f"{rep_name}\nTPR@95={summaries[rep_name]['pooled']['TPR_at_95_spec']:.0%}, "
                     f"OOD-FPR={summaries[rep_name]['pooled']['OOD_FPR_at_95_spec']:.0%}, "
                     f"x-FPR={summaries[rep_name]['pooled']['cross_adapter_FPR_at_95_spec']:.0%}",
                     fontsize=9)
        ax.set_xlabel("head logit")
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel("count")
    plt.suptitle("Phase 70: per-adapter verification head — score distributions on hard-OOD", fontsize=11)
    plt.tight_layout()
    plot_path = PHASE70_DIR / "head_calibration.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nSaved {plot_path}")

    # ---- Phase 7: save JSON ----
    out_path = PHASE70_DIR / "verification_head.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"hidden": HIDDEN, "n_epochs": N_EPOCHS, "lr": LR, "seed": SEED,
                       "n_adapters": n_adapters,
                       "n_train_per_adapter_pos": 4,
                       "n_train_per_adapter_neg": n_adapters - 1,
                       "n_eval_self_pos_per_adapter": 3,
                       "n_eval_cross_neg_per_adapter": (n_adapters - 1) * 3,
                       "n_eval_ood_neg_per_adapter": 20},
            "summaries": summaries,
        }, f, indent=2)
    print(f"Saved {out_path}")

    # ---- Verdict ----
    print(f"\n{'='*72}\nVERDICT")
    print(f"{'='*72}")
    best_rep = min(rep_names,
                    key=lambda r: summaries[r]["pooled"]["OOD_FPR_at_95_spec"])
    best_ofpr = summaries[best_rep]["pooled"]["OOD_FPR_at_95_spec"]
    best_xfpr = summaries[best_rep]["pooled"]["cross_adapter_FPR_at_95_spec"]
    best_tpr = summaries[best_rep]["pooled"]["TPR_at_95_spec"]
    print(f"  Best representation: {best_rep}")
    print(f"    pooled TPR@95spec = {best_tpr:.0%}")
    print(f"    pooled OOD FPR    = {best_ofpr:.0%}")
    print(f"    pooled cross FPR  = {best_xfpr:.0%}")

    if best_xfpr <= 0.10 and best_ofpr <= 0.10 and best_tpr >= 0.90:
        verdict = (f"DEPLOYMENT-GRADE: {best_rep} reaches OOD FPR {best_ofpr:.0%}, "
                   f"cross-adapter FPR {best_xfpr:.0%}, TPR {best_tpr:.0%}. "
                   f"Open-vocab verification is viable from hidden states alone.")
    elif best_xfpr <= 0.20 and best_ofpr <= 0.20:
        verdict = (f"PARTIAL: {best_rep} gets cross-FPR {best_xfpr:.0%}, "
                   f"OOD-FPR {best_ofpr:.0%}. Better than nothing but well "
                   f"below metadata's 0% rates. Open-vocab signal is recoverable "
                   f"but not deployment-clean.")
    else:
        verdict = (f"REPRESENTATIONAL CONFIRMED: best representation ({best_rep}) "
                   f"still gets cross-FPR {best_xfpr:.0%} and OOD-FPR {best_ofpr:.0%}. "
                   f"Entity signal is not recoverable from hidden states regardless "
                   f"of pooling or learned head. Closed-vocab metadata check is the "
                   f"only deployment-grade verification mechanism.")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
