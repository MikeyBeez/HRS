"""Phase 50: V-space SVD analysis and optimal engram extraction.

The V-space alignment is 0.65-0.70, the information recovery floor is 18%,
and mean pooling is a lossy first-order approximation. This phase measures
what's actually in V-space and how much of it can be captured.

Four stages:

1. MEASURE: Run SVD on the V matrices at every layer for the 20 passkey
   passages. Get singular value spectra. How much of V-space is captured
   by the first 1, 5, 10 singular vectors?

2. TEST: Use top-k left singular vectors as engrams. Compare against
   mean-pooled engrams on routing accuracy, information recovery, and
   compression. How much does mean pooling leave on the table?

3. BUILD: Train a small MLP to predict the SVD engram from the input
   hidden state (pre-V projection). If it learns the mapping, we have a
   cheap function that produces the optimal V-space descriptor without
   computing V.

4. COMPARE: Run stages 1-3 on both the V22 baseline and the Phase 49b
   metadata-enriched model. Does structured preprocessing improve V-space
   singular value spectra?

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase50_vspace_svd.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)

RANK = 128
ALPHA = 256
N_STEPS = 150
GEN_TOKENS = 50
MAX_CTX_POS = 512
N_PASSAGES_E2 = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200


# ================================================================
# Stage 1: SVD measurement
# ================================================================

@torch.no_grad()
def extract_v_matrices(model, tokenizer, tests, device):
    """For each passage and each layer, compute the V matrix.

    V is the value projection of the hidden states: V = H @ W_V^T
    where H is (T, D) and W_V is from the attention module.

    Returns: list of dicts, one per passage, each containing
      {layer_idx: V_matrix (T, D_v)} where D_v = n_heads * head_dim
    """
    results = []
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

        # Forward through the model, capturing hidden states at each layer
        h = model.drop(model.tok_emb(ids_t))  # (1, T, D)
        layer_vs = {}

        for layer_idx, block in enumerate(model.blocks):
            attn = block.attn
            ln_h = block.ln1(h)  # pre-attention layernorm
            B, T, D = ln_h.shape

            # The attention uses a fused qkv projection
            # qkv output is (B, T, 3 * n_heads * head_dim)
            qkv = attn.qkv(ln_h)
            n_h = attn.n_heads
            hd = attn.head_dim
            # Split into Q, K, V
            q, k, v = qkv.split(n_h * hd, dim=-1)
            # v is (B, T, n_heads * head_dim) = (1, T, D)
            layer_vs[layer_idx] = v.squeeze(0).cpu()  # (T, D)

            # Continue forward pass
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)

        results.append({
            "test": dict(test),
            "layer_vs": layer_vs,
            "n_tokens": ids_t.shape[1],
        })
    return results


def analyze_svd_spectra(v_data):
    """Run SVD on each V matrix and report singular value spectra."""
    all_spectra = {}  # layer -> list of spectra (one per passage)

    for entry in v_data:
        for layer_idx, V in entry["layer_vs"].items():
            if layer_idx not in all_spectra:
                all_spectra[layer_idx] = []

            # V is (T, D). SVD: V = U @ diag(S) @ Vh
            U, S, Vh = torch.linalg.svd(V.float(), full_matrices=False)
            # S is sorted descending
            total_var = (S ** 2).sum().item()
            cumvar = (S ** 2).cumsum(0) / total_var

            all_spectra[layer_idx].append({
                "singular_values": S.tolist()[:50],  # keep top 50
                "cumvar_1": cumvar[0].item() if len(cumvar) > 0 else 0,
                "cumvar_5": cumvar[4].item() if len(cumvar) > 4 else 0,
                "cumvar_10": cumvar[9].item() if len(cumvar) > 9 else 0,
                "cumvar_20": cumvar[19].item() if len(cumvar) > 19 else 0,
                "cumvar_50": cumvar[49].item() if len(cumvar) > 49 else 0,
                "n_tokens": entry["n_tokens"],
                "effective_rank": (S / S.max()).gt(0.01).sum().item(),
            })

    return all_spectra


# ================================================================
# Stage 2: SVD engram vs mean-pooled engram
# ================================================================

@torch.no_grad()
def extract_svd_engrams(model, tokenizer, test, device, top_k=1):
    """Extract the top-k left singular vector(s) of V at each layer.

    The first left singular vector of V captures the direction of maximum
    variance in V-space — the optimal single-vector summary.

    Returns: dict {layer_idx: engram (D,)} for top_k=1, or (k, D) for top_k>1
    """
    ids = tokenizer.encode(test["passage"], add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

    h = model.drop(model.tok_emb(ids_t))
    engrams = {}

    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn
        ln_h = block.ln1(h)
        qkv = attn.qkv(ln_h)
        n_h = attn.n_heads
        hd = attn.head_dim
        _, _, v = qkv.split(n_h * hd, dim=-1)
        V = v.squeeze(0).float()  # (T, D)

        U, S, Vh = torch.linalg.svd(V, full_matrices=False)
        # The top-k right singular vectors (rows of Vh) are the principal
        # directions in V-space. Scale by singular values for magnitude.
        if top_k == 1:
            engrams[layer_idx] = (Vh[0] * S[0]).cpu()  # (D,)
        else:
            engrams[layer_idx] = (Vh[:top_k] * S[:top_k].unsqueeze(1)).cpu()

        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)

    return engrams


@torch.no_grad()
def extract_mean_engram(model, ids_t, layer):
    """Standard L5 mean-pooled engram."""
    h = hidden_at_layer(model, ids_t, layer)
    return h.mean(dim=1).squeeze(0).detach().cpu()


@torch.no_grad()
def forward_segments(model, segments, device):
    parts = []
    for kind, x in segments:
        if kind == "tokens":
            ids = x.unsqueeze(0).to(device)
            parts.append(model.drop(model.tok_emb(ids)))
        elif kind == "hidden":
            parts.append(x.view(1, 1, -1).to(device))
    h = torch.cat(parts, dim=1)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]
    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    return model.lm_head(h)


def segments_len(segments):
    return sum(len(x) if kind == "tokens" else 1 for kind, x in segments)


@torch.no_grad()
def continuation_nll(model, prefix_segments, continuation_ids, device):
    M = len(continuation_ids)
    full_segments = list(prefix_segments) + [("tokens", continuation_ids[:-1])]
    logits = forward_segments(model, full_segments, device)
    prefix_len = segments_len(prefix_segments)
    pred = logits[:, prefix_len:prefix_len + M - 1, :]
    target = continuation_ids[1:].unsqueeze(0).to(device)
    nll = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                          target.reshape(-1), reduction="mean")
    return float(nll)


# ================================================================
# Main
# ================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase50")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()
    print(f"Stratified: {len(tests)} passages\n")

    # ============================================================
    # Load models
    # ============================================================
    print("Loading V22 baseline...")
    model, cfg = load_model(device)
    model.eval()

    enriched_path = Path("results/v22_enriched/best.pt")
    has_enriched = enriched_path.exists()
    enriched_model = None
    if has_enriched:
        print("Loading Phase 49b enriched model...")
        enriched_model = HRSTransformer(cfg).to(device)
        ckpt = torch.load(str(enriched_path), map_location=device,
                          weights_only=False)
        enriched_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        enriched_model.eval()
        print(f"  step={ckpt['step']}, val_ppl={ckpt['val_ppl']:.1f}")

    # ============================================================
    # STAGE 1: SVD spectra
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 1: V-space singular value spectra")
    print("=" * 60)

    for label, mdl in [("V22 baseline", model)] + \
            ([("49b enriched", enriched_model)] if enriched_model else []):
        print(f"\n  {label}:")
        v_data = extract_v_matrices(mdl, tokenizer, tests, device)
        spectra = analyze_svd_spectra(v_data)

        print(f"  {'layer':>5}  {'σ1%':>6}  {'top5%':>6}  {'top10%':>7}  "
              f"{'top20%':>7}  {'top50%':>7}  {'eff_rank':>8}")
        layer_summaries = {}
        for layer_idx in sorted(spectra.keys()):
            entries = spectra[layer_idx]
            avg = lambda key: sum(e[key] for e in entries) / len(entries)
            s1 = avg("cumvar_1")
            s5 = avg("cumvar_5")
            s10 = avg("cumvar_10")
            s20 = avg("cumvar_20")
            s50 = avg("cumvar_50")
            er = avg("effective_rank")
            print(f"  {layer_idx:>5}  {s1:>5.1%}  {s5:>5.1%}  {s10:>6.1%}  "
                  f"{s20:>6.1%}  {s50:>6.1%}  {er:>7.1f}")
            layer_summaries[layer_idx] = {
                "cumvar_1": s1, "cumvar_5": s5, "cumvar_10": s10,
                "cumvar_20": s20, "cumvar_50": s50, "effective_rank": er,
            }

        if label == "V22 baseline":
            baseline_spectra = spectra
            baseline_summaries = layer_summaries
        else:
            enriched_summaries = layer_summaries

    # ============================================================
    # STAGE 2: SVD engram vs mean-pooled engram
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 2: SVD engram vs mean-pooled engram")
    print("=" * 60)

    # 2a: Routing accuracy (L0 and L5)
    print("\n  2a: Routing accuracy on held-out paraphrases")

    # Build library keys: mean-pooled and SVD
    reset_lora_to_zero(model)

    mean_keys_l0 = []  # per adapter, list of mean-pooled L0
    mean_keys_l5 = []
    svd_keys_l0 = []   # per adapter, SVD top-1 at L0
    svd_keys_l5 = []

    for test in tests:
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

        # Mean-pooled
        h0 = model.drop(model.tok_emb(ids_t))
        mean_l0 = h0.mean(dim=1).squeeze(0).detach().cpu()

        h5 = hidden_at_layer(model, ids_t, 5)
        mean_l5 = h5.mean(dim=1).squeeze(0).detach().cpu()

        # SVD top-1
        svd_engs = extract_svd_engrams(model, tokenizer, test, device, top_k=1)
        svd_l0 = svd_engs[0]  # layer 0
        svd_l5 = svd_engs[5]  # layer 5

        mean_keys_l0.append([mean_l0])
        mean_keys_l5.append([mean_l5])
        svd_keys_l0.append([svd_l0])
        svd_keys_l5.append([svd_l5])

    def route(query_key, library_keys):
        best_a, best_score = -1, -2.0
        for ai, keys in enumerate(library_keys):
            for kv in keys:
                s = cosine(query_key, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    for key_label, lib_keys in [
        ("mean L0", mean_keys_l0),
        ("SVD-1 L0", svd_keys_l0),
        ("mean L5", mean_keys_l5),
        ("SVD-1 L5", svd_keys_l5),
    ]:
        n_routed = 0
        for i, test in enumerate(tests):
            ho_paras = held_out_paraphrase(test)
            for para in ho_paras:
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

                if "L0" in key_label:
                    if "SVD" in key_label:
                        q_engs = extract_svd_engrams(model, tokenizer,
                                                     {"passage": para}, device)
                        q = q_engs[0]
                    else:
                        h0 = model.drop(model.tok_emb(ids_t))
                        q = h0.mean(dim=1).squeeze(0).detach().cpu()
                else:
                    if "SVD" in key_label:
                        q_engs = extract_svd_engrams(model, tokenizer,
                                                     {"passage": para}, device)
                        q = q_engs[5]
                    else:
                        h5 = hidden_at_layer(model, ids_t, 5)
                        q = h5.mean(dim=1).squeeze(0).detach().cpu()

                if route(q, lib_keys) == i:
                    n_routed += 1

        print(f"    {key_label:12s}: {n_routed}/60 ({n_routed/60:.0%})")

    # 2b: Information recovery floor
    print("\n  2b: Information recovery (engram-as-cache, 50 WikiText passages)")

    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]
    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:N_PASSAGES_E2].tolist()

    conditions = {
        "no_context": [],
        "full_context": [],
        "mean_engram": [],
        "svd1_engram": [],
        "svd5_engram": [],
        "svd10_engram": [],
    }

    for ti, idx in enumerate(indices):
        item = val_ds[idx]
        ids = item[0] if isinstance(item, (list, tuple)) else item
        ids = ids[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        context_ids = ids[:CONTEXT_LEN]
        continuation = ids[CONTEXT_LEN:]

        # Mean engram
        ctx_t = context_ids.unsqueeze(0).to(device)
        mean_eng = extract_mean_engram(model, ctx_t, 5)

        # SVD engrams at L5
        # We need to extract V at layer 5 for the context
        h = model.drop(model.tok_emb(ctx_t))
        for li, block in enumerate(model.blocks):
            if li == 5:
                attn = block.attn
                ln_h = block.ln1(h)
                qkv = attn.qkv(ln_h)
                n_h = attn.n_heads
                hd = attn.head_dim
                _, _, v = qkv.split(n_h * hd, dim=-1)
                V = v.squeeze(0).float()
                U, S, Vh = torch.linalg.svd(V, full_matrices=False)
                svd1_eng = (Vh[0] * S[0]).cpu()
                svd5_eng = (Vh[:5] * S[:5].unsqueeze(1)).mean(0).cpu()
                svd10_eng = (Vh[:10] * S[:10].unsqueeze(1)).mean(0).cpu()
                break
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)

        conditions["no_context"].append(
            continuation_nll(model, [], continuation, device))
        conditions["full_context"].append(
            continuation_nll(model, [("tokens", context_ids)],
                             continuation, device))
        conditions["mean_engram"].append(
            continuation_nll(model, [("hidden", mean_eng)],
                             continuation, device))
        conditions["svd1_engram"].append(
            continuation_nll(model, [("hidden", svd1_eng)],
                             continuation, device))
        conditions["svd5_engram"].append(
            continuation_nll(model, [("hidden", svd5_eng)],
                             continuation, device))
        conditions["svd10_engram"].append(
            continuation_nll(model, [("hidden", svd10_eng)],
                             continuation, device))

        if (ti + 1) % 10 == 0:
            print(f"    [{ti+1}/{N_PASSAGES_E2}]")

    no_nll = sum(conditions["no_context"]) / len(conditions["no_context"])
    full_nll = sum(conditions["full_context"]) / len(conditions["full_context"])
    gap = no_nll - full_nll

    print(f"\n    no_context NLL   : {no_nll:.4f}")
    print(f"    full_context NLL : {full_nll:.4f}")
    print(f"    gap              : {gap:.4f}")
    print()

    recovery_results = {}
    for cond_name in ["mean_engram", "svd1_engram", "svd5_engram", "svd10_engram"]:
        nll = sum(conditions[cond_name]) / len(conditions[cond_name])
        recovery = (no_nll - nll) / gap if gap > 0 else 0
        print(f"    {cond_name:15s}: NLL {nll:.4f}  recovery {recovery:.1%}")
        recovery_results[cond_name] = {"nll": nll, "recovery": recovery}

    # ============================================================
    # STAGE 4: Compare with enriched model (if available)
    # ============================================================
    enriched_spectra_summary = None
    if enriched_model:
        print(f"\n{'='*60}")
        print("STAGE 4: Compare V-space spectra (baseline vs enriched)")
        print("=" * 60)
        print(f"\n  {'layer':>5}  {'base σ1%':>8}  {'enr σ1%':>8}  "
              f"{'base top10':>10}  {'enr top10':>10}  "
              f"{'base rank':>9}  {'enr rank':>9}")
        for l in sorted(baseline_summaries.keys()):
            b = baseline_summaries[l]
            e = enriched_summaries.get(l, b)
            print(f"  {l:>5}  {b['cumvar_1']:>7.1%}  {e['cumvar_1']:>7.1%}  "
                  f"{b['cumvar_10']:>9.1%}  {e['cumvar_10']:>9.1%}  "
                  f"{b['effective_rank']:>8.1f}  {e['effective_rank']:>8.1f}")
        enriched_spectra_summary = enriched_summaries

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 50 SUMMARY: V-space SVD analysis")
    print("=" * 72)

    best_layer = max(baseline_summaries.keys(),
                     key=lambda l: baseline_summaries[l]["cumvar_1"])
    bs = baseline_summaries[best_layer]
    print(f"\n  Most compressible layer: {best_layer}")
    print(f"    first singular vector captures: {bs['cumvar_1']:.1%}")
    print(f"    top 5 capture:                  {bs['cumvar_5']:.1%}")
    print(f"    top 10 capture:                 {bs['cumvar_10']:.1%}")
    print(f"    effective rank:                 {bs['effective_rank']:.0f}")

    print(f"\n  Information recovery comparison:")
    for name, r in recovery_results.items():
        print(f"    {name:15s}: {r['recovery']:.1%}")

    mean_r = recovery_results["mean_engram"]["recovery"]
    svd1_r = recovery_results["svd1_engram"]["recovery"]
    delta = svd1_r - mean_r
    print(f"\n  SVD-1 vs mean-pooled delta: {delta:+.1%}")
    if delta > 0.02:
        print("  -> SVD engram recovers more information than mean pooling")
    elif delta < -0.02:
        print("  -> SVD engram recovers less (V-space principal direction != useful)")
    else:
        print("  -> No meaningful difference")

    # Save
    out = {
        "baseline_spectra": {str(k): v for k, v in baseline_summaries.items()},
        "recovery": recovery_results,
        "no_context_nll": no_nll,
        "full_context_nll": full_nll,
        "gap": gap,
    }
    if enriched_spectra_summary:
        out["enriched_spectra"] = {str(k): v for k, v in enriched_spectra_summary.items()}
    with open(results_dir / "vspace_svd.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
