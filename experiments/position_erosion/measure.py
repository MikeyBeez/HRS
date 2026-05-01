"""Measure how close late-layer hidden states are to vocabulary points.

For each (layer, position):
  h = hidden_states[layer, position]
  logits = h @ wte.T
  top1 = logits.argmax()
  cos  = cosine(h, wte[top1])
  l2   = ||h - wte[top1]||

Aggregate per layer; compare to two controls:
  * random vectors with the same per-layer magnitude
  * token embeddings themselves (the fully-collapsed extreme)

Substrate: GPT-2 small (12 layers, 768d, learned absolute position
embeddings — the canonical case the experiment is asking about).
Input: first 1024 GPT-2 tokens of Tiny Shakespeare.
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
    print(f"  layers={n_layers} d_model={d_model} vocab={vocab}")

    # wte: (vocab, d_model). lm_head weight tied.
    wte = model.transformer.wte.weight.detach()  # (V, D)

    # Tokenize first N_TOKENS of Tiny Shakespeare
    text = TINY.read_text()
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    ctx_max = model.config.n_positions  # 1024 for gpt2
    ids = full_ids[:min(N_TOKENS, ctx_max)]
    print(f"  text bytes={len(text):,}  total_tokens={len(full_ids):,}  "
          f"using={len(ids)}")
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    # Forward with output_hidden_states
    t0 = time.time()
    out = model(ids_t, output_hidden_states=True, return_dict=True)
    print(f"  forward: {time.time()-t0:.1f}s, hidden_states tuple len="
          f"{len(out.hidden_states)}")
    # hidden_states[0] = post-embedding (after wte+wpe+dropout)
    # hidden_states[1..n_layers] = output of each transformer block

    # Build per-layer measurements
    # hidden_states is tuple of (B, T, D); we have B=1, T=len(ids)
    hs = torch.stack([h.squeeze(0) for h in out.hidden_states], dim=0)  # (L+1, T, D)
    print(f"  hidden stack shape: {hs.shape}")

    # Project all (L+1, T) hidden states through lm_head (= wte^T) to find
    # nearest tokens.
    # logits shape: (L+1, T, V)
    L_plus_1, T, D = hs.shape
    per_layer = []
    layer_examples = {}  # for tracking 3 positions through depth

    for layer in range(L_plus_1):
        h = hs[layer]  # (T, D)
        logits = h @ wte.T  # (T, V)
        top1 = logits.argmax(dim=-1)  # (T,)
        top1_emb = wte[top1]  # (T, D)

        cos = F.cosine_similarity(h, top1_emb, dim=-1)  # (T,)
        l2 = (h - top1_emb).norm(dim=-1)  # (T,)
        h_norm = h.norm(dim=-1)  # (T,)

        per_layer.append({
            "layer": layer,
            "cos_mean": float(cos.mean().item()),
            "cos_std":  float(cos.std().item()),
            "cos_p25":  float(cos.quantile(0.25).item()),
            "cos_p50":  float(cos.median().item()),
            "cos_p75":  float(cos.quantile(0.75).item()),
            "l2_mean":  float(l2.mean().item()),
            "l2_std":   float(l2.std().item()),
            "h_norm_mean": float(h_norm.mean().item()),
            "h_norm_std":  float(h_norm.std().item()),
        })

        # Examples: positions 5, 100, 500
        for pos in (5, 100, 500):
            if pos >= T: continue
            tok_id = top1[pos].item()
            tok_str = tokenizer.decode([tok_id])
            layer_examples.setdefault(pos, []).append({
                "layer": layer,
                "input_token": tokenizer.decode([ids[pos]]),
                "top1_token": tok_str,
                "cos": float(cos[pos].item()),
                "l2": float(l2[pos].item()),
                "h_norm": float(h_norm[pos].item()),
            })

    print(f"\nPer-layer summary:")
    print(f"  {'layer':>5}  {'cos_mean':>9}  {'cos_p50':>9}  "
          f"{'l2_mean':>9}  {'h_norm':>9}")
    for r in per_layer:
        print(f"  {r['layer']:5d}  {r['cos_mean']:9.4f}  {r['cos_p50']:9.4f}  "
              f"{r['l2_mean']:9.3f}  {r['h_norm_mean']:9.3f}")

    # ---------- Controls ----------
    # Control 1: random vectors with per-layer magnitude
    # For each layer, sample T random vectors with the same mean ||h|| as
    # that layer's hidden states. Find nearest token, measure.
    rng = torch.Generator(device=device); rng.manual_seed(0)
    control_random = []
    for layer in range(L_plus_1):
        h_norm_target = per_layer[layer]["h_norm_mean"]
        # Random unit vectors scaled to that magnitude
        rand = torch.randn(T, D, generator=rng, device=device)
        rand = rand / rand.norm(dim=-1, keepdim=True) * h_norm_target
        logits = rand @ wte.T
        top1 = logits.argmax(dim=-1)
        top1_emb = wte[top1]
        cos = F.cosine_similarity(rand, top1_emb, dim=-1)
        l2 = (rand - top1_emb).norm(dim=-1)
        control_random.append({
            "layer": layer,
            "cos_mean": float(cos.mean().item()),
            "l2_mean":  float(l2.mean().item()),
        })

    # Control 2: token embeddings themselves projected back through lm_head.
    # For a sample of T tokens (use the same input ids), compute lm_head
    # argmax of the embedding itself, then cosine to that argmax token's
    # embedding.
    sample_emb = wte[ids_t.squeeze(0)]  # (T, D)
    logits = sample_emb @ wte.T
    top1 = logits.argmax(dim=-1)
    top1_emb = wte[top1]
    cos = F.cosine_similarity(sample_emb, top1_emb, dim=-1)
    l2 = (sample_emb - top1_emb).norm(dim=-1)
    same_token_frac = (top1 == ids_t.squeeze(0)).float().mean().item()
    control_emb = {
        "cos_mean": float(cos.mean().item()),
        "l2_mean":  float(l2.mean().item()),
        "self_top1_frac": same_token_frac,
    }

    print(f"\nControl 1 (random vectors at matched magnitude):")
    print(f"  {'layer':>5}  {'cos_mean':>9}  {'l2_mean':>9}")
    for r in control_random:
        print(f"  {r['layer']:5d}  {r['cos_mean']:9.4f}  {r['l2_mean']:9.3f}")
    print(f"\nControl 2 (token embeddings self-projection):")
    print(f"  cos_mean={control_emb['cos_mean']:.4f}  "
          f"l2_mean={control_emb['l2_mean']:.3f}  "
          f"self-top1-frac={control_emb['self_top1_frac']:.4f}")

    out_data = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab": vocab,
        "n_tokens_used": T,
        "per_layer": per_layer,
        "control_random": control_random,
        "control_embedding_self": control_emb,
        "layer_examples": layer_examples,
    }
    out_path = EXP / "results/measure.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
