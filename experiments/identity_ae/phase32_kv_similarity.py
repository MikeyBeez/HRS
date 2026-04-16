"""Phase 32: KV cache similarity — does the engram land near the K/V centroid?

Phase 35 showed empirically that the engram is an address: it carries no
usable information until the model has been trained on its referent
(condition A: 0/20), but after TTT the same engram retrieves nearly all
passkeys (B: 19/20, C: 20/20).

This script measures the *mechanism* behind that result for in-distribution
content (which the model already knows). The hypothesis: the L5 mean engram,
when injected as a single hidden-state prefix and run through the model,
produces K and V vectors at each layer that are close to the mean K and V
the full passage produces. If true, the attention mechanism cannot tell the
difference between attending over the passage tokens and attending over
the engram pointer — they live in the same region of K/V space.

Procedure (per passage):
  1. Forward the passage tokens. Forward hooks on each block.attn.qkv
     capture the (B, T, 3*D) projection. Split into Q, K, V; we keep K
     and V pre-RoPE (position-agnostic, so averaging across positions
     is meaningful).
  2. Compute the L5 mean engram of the passage hidden states.
  3. Forward a single-position hidden-state input where x[0,0,:] = engram.
     The same hooks capture per-layer K, V at the engram position.
  4. For each layer, compute:
       - cos(mean(K_passage), K_engram)   averaged across heads
       - cos(mean(V_passage), V_engram)   averaged across heads
       - L2 distance, same averaging
       - Norm ratio  ||K_engram|| / ||mean(K_passage)||
  5. Random-vector baseline: cos(mean(K_passage), random_unit) per layer.

Dataset: 50 WikiText validation sequences truncated to ~128 tokens.
WikiText is in-distribution for V22 (trained on WikiText-103), so the
"manifold has been mapped" precondition of the centroid theory holds.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase32_kv_similarity.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import hidden_at_layer, reset_lora_to_zero
from experiments.identity_ae.phase35_engram_after_ttt import forward_from_x


LAYER = 5
N_PASSAGES = 50
PASSAGE_LEN = 128


# ----------------------------------------------------------------
# Hook plumbing.
# ----------------------------------------------------------------
def install_qkv_hooks(model, store: dict):
    """Register forward hooks on every block's qkv projection.

    The hook splits the projection into pre-RoPE Q, K, V per head and
    writes K and V into `store[layer_idx]`. Returns a list of handles
    that the caller must remove when done.
    """
    handles = []
    for i, block in enumerate(model.blocks):
        n_heads = block.attn.n_heads
        head_dim = block.attn.head_dim

        def make_hook(layer_idx, H, Dh):
            def hook(module, inp, out):
                B, T, _ = out.shape
                qkv = out.reshape(B, T, 3, H, Dh)
                _, k, v = qkv.unbind(dim=2)         # each (B, T, H, Dh)
                store[layer_idx] = {
                    "k": k.detach().float().cpu(),  # avoid GPU memory pile-up
                    "v": v.detach().float().cpu(),
                }
            return hook

        handles.append(block.attn.qkv.register_forward_hook(
            make_hook(i, n_heads, head_dim)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ----------------------------------------------------------------
# Per-head cosine and L2 between (1, 1, H, Dh) and (1, 1, H, Dh).
# Returns a Python float averaged across heads.
# ----------------------------------------------------------------
def per_head_cosine(a, b):
    """a, b: (H, Dh) tensors. Returns mean per-head cosine."""
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=-1).mean().item()


def per_head_l2(a, b):
    return (a - b).norm(dim=-1).mean().item()


def per_head_norm(a):
    return a.norm(dim=-1).mean().item()


# ----------------------------------------------------------------
# Main.
# ----------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase32")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    print(f"Model: V22, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim
    d_model = n_heads * head_dim
    print(f"Layers: {n_layers}, heads: {n_heads}, head_dim: {head_dim}, d_model: {d_model}")
    print(f"Engram source: L{LAYER}_mean (hidden-state prefix injection)\n")

    # ============================================================
    # Sample 50 WikiText validation passages, truncated to 128 tokens.
    # ============================================================
    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]
    print(f"Validation: {len(val_ds)} sequences available, sampling {N_PASSAGES}\n")

    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:N_PASSAGES].tolist()

    # ============================================================
    # Per-passage measurement.
    # ============================================================
    # accum[layer] = list of dicts with per-passage measurements
    metrics = {l: {"cos_k": [], "cos_v": [],
                    "l2_k": [], "l2_v": [],
                    "norm_ratio_k": [], "norm_ratio_v": [],
                    "rand_cos_k": [], "rand_cos_v": []}
                for l in range(n_layers)}

    model.eval()
    reset_lora_to_zero(model)

    for ti, idx in enumerate(indices):
        item = val_ds[idx]
        # WikiTextDataset items are typically (input_ids, target_ids) or just ids
        if isinstance(item, tuple):
            ids = item[0]
        elif isinstance(item, dict):
            ids = item.get("input_ids", next(iter(item.values())))
        else:
            ids = item
        ids = ids[:PASSAGE_LEN].unsqueeze(0).to(device)  # (1, T)

        # ----- Pass 1: full passage forward, capture K/V at every layer -----
        passage_store = {}
        handles = install_qkv_hooks(model, passage_store)
        with torch.no_grad():
            _ = model(ids, step=0)
        remove_hooks(handles)

        # ----- Compute the L5 mean engram of this passage -----
        with torch.no_grad():
            h_L = hidden_at_layer(model, ids, LAYER)  # (1, T, D)
            engram = h_L.mean(dim=1)                    # (1, D)

        # ----- Pass 2: engram-as-prefix forward, capture K/V at every layer -----
        engram_store = {}
        handles = install_qkv_hooks(model, engram_store)
        with torch.no_grad():
            x_eng = engram.view(1, 1, d_model)
            _ = forward_from_x(model, x_eng)
        remove_hooks(handles)

        # ----- Per-layer comparisons -----
        for l in range(n_layers):
            K_p = passage_store[l]["k"].squeeze(0)   # (T, H, Dh)
            V_p = passage_store[l]["v"].squeeze(0)
            K_e = engram_store[l]["k"].squeeze(0).squeeze(0)   # (H, Dh)
            V_e = engram_store[l]["v"].squeeze(0).squeeze(0)

            mean_K_p = K_p.mean(dim=0)   # (H, Dh)
            mean_V_p = V_p.mean(dim=0)

            metrics[l]["cos_k"].append(per_head_cosine(mean_K_p, K_e))
            metrics[l]["cos_v"].append(per_head_cosine(mean_V_p, V_e))
            metrics[l]["l2_k"].append(per_head_l2(mean_K_p, K_e))
            metrics[l]["l2_v"].append(per_head_l2(mean_V_p, V_e))
            metrics[l]["norm_ratio_k"].append(per_head_norm(K_e) / (per_head_norm(mean_K_p) + 1e-8))
            metrics[l]["norm_ratio_v"].append(per_head_norm(V_e) / (per_head_norm(mean_V_p) + 1e-8))

            # Random baseline: random unit per-head vector
            rand_k = torch.randn_like(K_e)
            rand_k = rand_k / (rand_k.norm(dim=-1, keepdim=True) + 1e-8) * K_e.norm(dim=-1, keepdim=True)
            rand_v = torch.randn_like(V_e)
            rand_v = rand_v / (rand_v.norm(dim=-1, keepdim=True) + 1e-8) * V_e.norm(dim=-1, keepdim=True)
            metrics[l]["rand_cos_k"].append(per_head_cosine(mean_K_p, rand_k))
            metrics[l]["rand_cos_v"].append(per_head_cosine(mean_V_p, rand_v))

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1:2d}/{N_PASSAGES}] processed")

    # ============================================================
    # Aggregate.
    # ============================================================
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

    # ============================================================
    # Print.
    # ============================================================
    print(f"\n{'='*78}")
    print(f"PHASE 32 SUMMARY: KV centroid similarity (engram vs passage mean)")
    print(f"{'='*78}")
    print(f"  In-distribution content (WikiText val), {N_PASSAGES} passages, "
          f"{PASSAGE_LEN} tokens each")
    print(f"  Engram = L{LAYER} mean of passage hidden states, injected at layer 0")
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

    # Headline
    print()
    headline_layers = [0, n_layers // 2, n_layers - 1]
    print(f"  Headline (cos K vs random):")
    for l in headline_layers:
        s = summary_per_layer[l]
        print(f"    layer {l}: {s['cos_k']:.3f}  (random baseline: {s['rand_cos_k']:.3f})")

    # ============================================================
    # Save.
    # ============================================================
    out = {
        "n_passages": N_PASSAGES,
        "passage_len": PASSAGE_LEN,
        "engram_layer": LAYER,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "per_layer": summary_per_layer,
    }
    with open(results_dir / "kv_similarity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
