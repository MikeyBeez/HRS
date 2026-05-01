"""Extended measurement: also compute cosine-nearest token (not just
lm_head argmax) and post-LayerNorm hidden states.

The lm_head argmax over h @ wte.T finds the dot-product-largest token,
which can be biased by per-token embedding magnitude. The cosine-nearest
token is the geometrically-nearest in cosine space and answers the
spec's "how close is h to any vocabulary point?" question more directly.

Also measure post-ln_f hidden state (the actual input to lm_head),
because GPT-2 normalizes the final hidden before output projection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/position_erosion"
TINY = REPO / "datasets/tiny_shakespeare.txt"
N_TOKENS = 1024
MODEL_NAME = "gpt2"


@torch.no_grad()
def main():
    device = torch.device("cuda")
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    n_layers = len(model.transformer.h)
    d_model = model.config.n_embd
    vocab = model.config.vocab_size
    wte = model.transformer.wte.weight.detach()  # (V, D)
    wte_norm = wte / wte.norm(dim=-1, keepdim=True)  # (V, D), unit-rowed

    text = TINY.read_text()
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    ids = full_ids[:min(N_TOKENS, model.config.n_positions)]
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    print(f"  using {len(ids)} tokens")

    out = model(ids_t, output_hidden_states=True, return_dict=True)
    hs = torch.stack([h.squeeze(0) for h in out.hidden_states], dim=0)
    L_plus_1, T, D = hs.shape

    # Compute the post-ln_f version of the final layer
    ln_f = model.transformer.ln_f
    h_final_ln = ln_f(hs[-1])  # (T, D)

    def measure(h, label):
        """h: (T, D). Returns dict with cos/l2 to lm_head argmax AND to
        cosine-nearest token."""
        # 1) lm_head argmax (dot-product nearest)
        logits = h @ wte.T  # (T, V)
        argmax_dot = logits.argmax(dim=-1)  # (T,)
        emb_dot = wte[argmax_dot]
        cos_dot = F.cosine_similarity(h, emb_dot, dim=-1)
        l2_dot  = (h - emb_dot).norm(dim=-1)

        # 2) cosine-nearest token
        h_norm = h / h.norm(dim=-1, keepdim=True)  # (T, D)
        cos_to_all = h_norm @ wte_norm.T  # (T, V) — true cosine
        argmax_cos = cos_to_all.argmax(dim=-1)
        emb_cos = wte[argmax_cos]
        cos_cos = cos_to_all.gather(1, argmax_cos.unsqueeze(-1)).squeeze(-1)
        l2_cos  = (h - emb_cos).norm(dim=-1)

        h_norm_mean = h.norm(dim=-1).mean().item()
        return {
            "label": label,
            "cos_dot_mean": float(cos_dot.mean().item()),
            "cos_dot_p50":  float(cos_dot.median().item()),
            "l2_dot_mean":  float(l2_dot.mean().item()),
            "cos_cos_mean": float(cos_cos.mean().item()),
            "cos_cos_p50":  float(cos_cos.median().item()),
            "l2_cos_mean":  float(l2_cos.mean().item()),
            "h_norm_mean":  float(h_norm_mean),
            "argmax_dot_eq_argmax_cos_frac":
                float((argmax_dot == argmax_cos).float().mean().item()),
        }

    # All layers (0..n_layers) by block output
    rows = []
    for layer in range(L_plus_1):
        rows.append({"layer": layer, **measure(hs[layer], f"L{layer}")})

    # Final post-LayerNorm
    rows.append({"layer": L_plus_1, **measure(h_final_ln,
                                               f"L{L_plus_1-1}_post_lnf")})

    print(f"\nPer-layer (cos to lm-head-argmax token AND to cosine-nearest token):")
    print(f"  {'layer':>20}  {'cos_dot':>8}  {'cos_cos':>8}  "
          f"{'l2_cos':>9}  {'h_norm':>8}  {'argmax_eq':>9}")
    for r in rows:
        print(f"  {r['label']:>20}  {r['cos_dot_mean']:8.4f}  "
              f"{r['cos_cos_mean']:8.4f}  {r['l2_cos_mean']:9.2f}  "
              f"{r['h_norm_mean']:8.2f}  "
              f"{r['argmax_dot_eq_argmax_cos_frac']:9.4f}")

    # Random control with per-layer matched magnitude — true cosine-nearest
    rng = torch.Generator(device=device); rng.manual_seed(0)
    control_random = []
    for r in rows:
        target = r["h_norm_mean"]
        rand = torch.randn(T, D, generator=rng, device=device)
        rand = rand / rand.norm(dim=-1, keepdim=True) * target
        rand_norm = rand / rand.norm(dim=-1, keepdim=True)
        cos_to_all = rand_norm @ wte_norm.T
        argmax_cos = cos_to_all.argmax(dim=-1)
        cos_cos = cos_to_all.gather(1, argmax_cos.unsqueeze(-1)).squeeze(-1)
        control_random.append({
            "label": r["label"],
            "cos_cos_mean": float(cos_cos.mean().item()),
        })

    print(f"\nControl (random vectors, cosine-nearest token):")
    for r in control_random:
        print(f"  {r['label']:>20}  cos={r['cos_cos_mean']:.4f}")

    # Embedding self-control
    sample_emb = wte[ids_t.squeeze(0)]  # (T, D)
    cos_self = F.cosine_similarity(sample_emb,
                                     wte[(sample_emb @ wte.T).argmax(dim=-1)],
                                     dim=-1).mean().item()
    print(f"\nEmbedding self-control (token embeddings projected back): "
          f"cos_dot={cos_self:.4f}")

    # Examples: trace 3 positions through depth, showing top-1 (cosine-nearest)
    EXAMPLES = (5, 100, 500)
    layer_examples = {pos: [] for pos in EXAMPLES if pos < T}
    for pos in EXAMPLES:
        if pos >= T: continue
        in_tok = tokenizer.decode([ids[pos]])
        for layer in range(L_plus_1):
            h = hs[layer, pos]
            h_n = h / h.norm()
            cos_to_all = h_n @ wte_norm.T
            top_id = cos_to_all.argmax().item()
            top_str = tokenizer.decode([top_id])
            top_cos = cos_to_all[top_id].item()
            layer_examples[pos].append({
                "layer": layer, "input_token": in_tok,
                "top1_cos_token": top_str,
                "top1_cos": top_cos,
                "h_norm": h.norm().item(),
            })
        # Post-LN
        h = h_final_ln[pos]
        h_n = h / h.norm()
        cos_to_all = h_n @ wte_norm.T
        top_id = cos_to_all.argmax().item()
        layer_examples[pos].append({
            "layer": "L11_post_lnf", "input_token": in_tok,
            "top1_cos_token": tokenizer.decode([top_id]),
            "top1_cos": cos_to_all[top_id].item(),
            "h_norm": h.norm().item(),
        })

    print(f"\nExample positions (cosine-nearest token by layer):")
    for pos, trace in layer_examples.items():
        in_tok = trace[0]["input_token"]
        print(f"  pos={pos} input={in_tok!r}:")
        for t in trace:
            print(f"    {str(t['layer']):>13}: top1={t['top1_cos_token']!r:30s} "
                  f"cos={t['top1_cos']:.3f}  ||h||={t['h_norm']:7.2f}")

    out_data = {
        "model": MODEL_NAME, "n_tokens": T,
        "rows": rows,
        "control_random": control_random,
        "control_embedding_self_cos": cos_self,
        "layer_examples": layer_examples,
    }
    out_path = EXP / "results/measure2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
