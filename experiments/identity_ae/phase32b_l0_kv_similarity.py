"""Phase 32b: KV cache similarity for the L0 (token embedding) engram.

Phase 32 measured K-space alignment for the L5 mean engram and found cosine
0.82–0.89 against the passage's mean K vector at every layer past a layer-0
bootstrap (where the engram is an L5-shaped vector being projected through
W_k_layer0, which was trained against L0-shaped inputs).

Phase 44 found that the L0 mean engram is a strictly better routing key for
Application 1: 90 percent held-out paraphrase retrieval vs 75 percent for
the L5 nonstop_mean engram. The bag-of-token-embeddings is a more reliable
address than the layer-5 hidden-state mean.

This script measures the K-space alignment for the L0 mean engram. The
prediction: there should be no layer-0 bootstrap problem, because the L0
engram is exactly the kind of vector that W_k_layer0 expects to see (it
*is* a layer-0 hidden state, by construction). The cosine should be high
at every layer including layer 0.

Same setup as Phase 32 — 50 WikiText validation passages, 128 tokens each,
forward hooks on every block's qkv projection — only the engram extraction
changes.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase32b_l0_kv_similarity.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.phase32_kv_similarity import (
    install_qkv_hooks, remove_hooks,
    per_head_cosine, per_head_l2, per_head_norm,
)
from experiments.identity_ae.phase35_engram_after_ttt import forward_from_x


N_PASSAGES = 50
PASSAGE_LEN = 128


@torch.no_grad()
def l0_engram(model, ids_t):
    """L0 mean engram: mean of input token embeddings (no transformer pass)."""
    h = model.drop(model.tok_emb(ids_t))   # (1, T, D)
    return h.mean(dim=1)                    # (1, D)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase32b")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim
    d_model = n_heads * head_dim
    print(f"Model: V22, layers={n_layers}, heads={n_heads}, head_dim={head_dim}, d_model={d_model}")
    print(f"Engram source: L0_mean (token embedding mean, no transformer pass)\n")

    # ============================================================
    # Sample 50 WikiText validation passages
    # ============================================================
    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]
    print(f"Validation: {len(val_ds)} sequences available, sampling {N_PASSAGES}\n")

    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:N_PASSAGES].tolist()

    metrics = {l: {"cos_k": [], "cos_v": [],
                    "l2_k": [], "l2_v": [],
                    "norm_ratio_k": [], "norm_ratio_v": [],
                    "rand_cos_k": [], "rand_cos_v": []}
                for l in range(n_layers)}

    model.eval()
    reset_lora_to_zero(model)

    for ti, idx in enumerate(indices):
        item = val_ds[idx]
        ids = item[0] if isinstance(item, tuple) else item
        ids = ids[:PASSAGE_LEN].unsqueeze(0).to(device)

        # Pass 1: full passage forward, capture K/V at every layer
        passage_store = {}
        handles = install_qkv_hooks(model, passage_store)
        with torch.no_grad():
            _ = model(ids, step=0)
        remove_hooks(handles)

        # Compute the L0 mean engram of this passage
        engram = l0_engram(model, ids)   # (1, D)

        # Pass 2: engram-as-prefix forward, capture K/V at every layer
        engram_store = {}
        handles = install_qkv_hooks(model, engram_store)
        with torch.no_grad():
            x_eng = engram.view(1, 1, d_model)
            _ = forward_from_x(model, x_eng)
        remove_hooks(handles)

        # Per-layer comparisons
        for l in range(n_layers):
            K_p = passage_store[l]["k"].squeeze(0)
            V_p = passage_store[l]["v"].squeeze(0)
            K_e = engram_store[l]["k"].squeeze(0).squeeze(0)
            V_e = engram_store[l]["v"].squeeze(0).squeeze(0)

            mean_K_p = K_p.mean(dim=0)
            mean_V_p = V_p.mean(dim=0)

            metrics[l]["cos_k"].append(per_head_cosine(mean_K_p, K_e))
            metrics[l]["cos_v"].append(per_head_cosine(mean_V_p, V_e))
            metrics[l]["l2_k"].append(per_head_l2(mean_K_p, K_e))
            metrics[l]["l2_v"].append(per_head_l2(mean_V_p, V_e))
            metrics[l]["norm_ratio_k"].append(per_head_norm(K_e) / (per_head_norm(mean_K_p) + 1e-8))
            metrics[l]["norm_ratio_v"].append(per_head_norm(V_e) / (per_head_norm(mean_V_p) + 1e-8))

            rand_k = torch.randn_like(K_e)
            rand_k = rand_k / (rand_k.norm(dim=-1, keepdim=True) + 1e-8) * K_e.norm(dim=-1, keepdim=True)
            rand_v = torch.randn_like(V_e)
            rand_v = rand_v / (rand_v.norm(dim=-1, keepdim=True) + 1e-8) * V_e.norm(dim=-1, keepdim=True)
            metrics[l]["rand_cos_k"].append(per_head_cosine(mean_K_p, rand_k))
            metrics[l]["rand_cos_v"].append(per_head_cosine(mean_V_p, rand_v))

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1:2d}/{N_PASSAGES}] processed")

    # Aggregate
    def avg(xs):
        return float(sum(xs) / len(xs))

    summary_per_layer = []
    for l in range(n_layers):
        m = metrics[l]
        summary_per_layer.append({
            "layer": l,
            "cos_k":         avg(m["cos_k"]),
            "cos_v":         avg(m["cos_v"]),
            "rand_cos_k":    avg(m["rand_cos_k"]),
            "rand_cos_v":    avg(m["rand_cos_v"]),
            "l2_k":          avg(m["l2_k"]),
            "l2_v":          avg(m["l2_v"]),
            "norm_ratio_k":  avg(m["norm_ratio_k"]),
            "norm_ratio_v":  avg(m["norm_ratio_v"]),
        })

    print(f"\n{'='*78}")
    print(f"PHASE 32b SUMMARY: KV centroid similarity for the L0 mean engram")
    print(f"{'='*78}")
    print(f"  In-distribution content (WikiText val), {N_PASSAGES} passages, {PASSAGE_LEN} tokens each")
    print(f"  Engram = L0 mean (token embedding mean, NO transformer pass)")
    print()
    print(f"  {'layer':>5} | {'cos K':>8} {'rand K':>8} | {'cos V':>8} {'rand V':>8} | "
          f"{'L2 K':>8} {'L2 V':>8} | {'|Ke|/|Kp|':>10} {'|Ve|/|Vp|':>10}")
    print(f"  {'-'*5}-+-{'-'*8} {'-'*8}-+-{'-'*8} {'-'*8}-+-{'-'*8} {'-'*8}-+-{'-'*10} {'-'*10}")
    for s in summary_per_layer:
        print(f"  {s['layer']:>5} | "
              f"{s['cos_k']:>8.3f} {s['rand_cos_k']:>8.3f} | "
              f"{s['cos_v']:>8.3f} {s['rand_cos_v']:>8.3f} | "
              f"{s['l2_k']:>8.3f} {s['l2_v']:>8.3f} | "
              f"{s['norm_ratio_k']:>10.3f} {s['norm_ratio_v']:>10.3f}")

    print()
    print(f"  Phase 32 reference (L5 mean engram):")
    print(f"    layer 0: cos K = -0.056 (random -0.005)  ← bootstrap failure")
    print(f"    layer 1: cos K =  0.844 (random -0.012)")
    print(f"    layer 4: cos K =  0.894 (random -0.002)")
    print()
    headline_layers = [0, n_layers // 2, n_layers - 1]
    print(f"  Phase 32b headline (L0 mean engram):")
    for l in headline_layers:
        s = summary_per_layer[l]
        print(f"    layer {l}: cos K = {s['cos_k']:>6.3f}  (random {s['rand_cos_k']:>6.3f})")

    out = {
        "n_passages": N_PASSAGES,
        "passage_len": PASSAGE_LEN,
        "engram_source": "L0_mean",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "per_layer": summary_per_layer,
    }
    with open(results_dir / "l0_kv_similarity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
