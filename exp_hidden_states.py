"""Experiment: Full Hidden States vs Mean-Pooled Engram.

Tests where in-context information lives by comparing four conditions:
A. Mean-pooled engram (standard V18-EGR — already shown to fail at grounding)
B. Full hidden states injected via cross-attention (preserves token-level info)
C. Text prepend (standard in-context learning — the control)
D. Random hidden states (negative control)

Usage:
    python exp_hidden_states.py [--device cuda]
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from niah_egr import NEEDLES, DISTRACTORS


def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


@torch.no_grad()
def extract_hidden_states(model, token_ids, device, extract_layer):
    """Extract full hidden states (all positions) from a layer.

    Returns:
        full: (1, T, D) — all position hidden states
        pooled: (1, 1, D) — mean-pooled
    """
    ids = token_ids[:512].unsqueeze(0).to(device)

    captured = {}
    def hook_fn(module, inp, out):
        captured['h'] = out[0].detach()

    handle = model.blocks[extract_layer].register_forward_hook(hook_fn)
    _ = model(ids, step=0)
    handle.remove()

    full = captured['h']  # (1, T, D)
    pooled = full.mean(dim=1, keepdim=True)  # (1, 1, D)
    return full, pooled


@torch.no_grad()
def generate_with_engram(model, prompt_ids, engram_buffer, device,
                         max_new_tokens=150, temperature=0.9, top_k=50):
    """Generate with a custom engram buffer injected into cross-attention.

    Args:
        model: V18 model
        prompt_ids: (T,) token ids
        engram_buffer: (1, E, D) engram to inject, or None for no injection
        device: torch device
        max_new_tokens: tokens to generate
    """
    model.eval()
    orig_buffer = model.engram_buffer.data.clone()
    orig_init = model._engram_buffer_initialized

    if engram_buffer is not None:
        model.engram_buffer = torch.nn.Parameter(engram_buffer.to(device), requires_grad=False)
        model._engram_buffer_initialized = True
    else:
        model._engram_buffer_initialized = False

    input_ids = prompt_ids.unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        idx = input_ids[:, -512:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Restore original buffer
    model.engram_buffer = torch.nn.Parameter(orig_buffer, requires_grad=False)
    model._engram_buffer_initialized = orig_init

    generated = input_ids[0, prompt_ids.shape[0]:]
    return generated


@torch.no_grad()
def generate_with_prepend(model, prompt_ids, prepend_ids, device,
                          max_new_tokens=150, temperature=0.9, top_k=50):
    """Generate with text prepended to context (standard in-context learning)."""
    model.eval()
    # Disable engram buffer — pure self-attention
    orig_init = model._engram_buffer_initialized
    model._engram_buffer_initialized = False

    # Concatenate prepend + prompt, truncate to 512
    combined = torch.cat([prepend_ids, prompt_ids])[-512:]
    prompt_offset = max(0, combined.shape[0] - prompt_ids.shape[0])

    input_ids = combined.unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        idx = input_ids[:, -512:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    model._engram_buffer_initialized = orig_init

    # Return only the generated part
    generated = input_ids[0, combined.shape[0]:]
    return generated


def count_answer_hits(text, answer_tokens):
    """Count how many answer tokens appear in generated text."""
    text_lower = text.lower()
    hits = sum(1 for t in answer_tokens if t.lower() in text_lower)
    return hits, len(answer_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    extract_layer = cfg.model.n_layers - 2
    d_model = cfg.model.d_model

    print(f"\nRunning 4 conditions on {len(NEEDLES)} needles")
    print(f"  Extract layer: {extract_layer}")
    print(f"  d_model: {d_model}")

    conditions = {
        "A_mean_pooled": {"hits": 0, "total": 0, "details": []},
        "B_full_hidden": {"hits": 0, "total": 0, "details": []},
        "C_text_prepend": {"hits": 0, "total": 0, "details": []},
        "D_random": {"hits": 0, "total": 0, "details": []},
    }

    for needle in NEEDLES:
        print(f"\n{'='*60}")
        print(f"Needle: {needle.category}")
        print(f"  Fact: {needle.fact[:80]}...")
        print(f"  Query: {needle.query}")
        print(f"  Answer tokens: {needle.answer_tokens}")

        # Tokenize needle and query
        needle_ids = torch.tensor(
            tokenizer.encode(needle.fact, add_special_tokens=False), dtype=torch.long
        )
        query_ids = torch.tensor(
            tokenizer.encode(needle.query, add_special_tokens=False), dtype=torch.long
        )

        # Extract hidden states from needle
        full_hidden, pooled_hidden = extract_hidden_states(
            model, needle_ids, device, extract_layer
        )
        print(f"  Needle tokens: {needle_ids.shape[0]}")
        print(f"  Full hidden: {full_hidden.shape}")
        print(f"  Pooled hidden: {pooled_hidden.shape}")

        # ---- Condition A: Mean-pooled engram (expanded to 32 tokens) ----
        # Repeat the single pooled vector to fill the 32-slot buffer
        engram_a = pooled_hidden.expand(1, 32, d_model).contiguous()
        gen_a = generate_with_engram(model, query_ids, engram_a, device, args.max_new_tokens)
        text_a = tokenizer.decode(gen_a, skip_special_tokens=True)
        hits_a, total_a = count_answer_hits(text_a, needle.answer_tokens)

        # ---- Condition B: Full hidden states ----
        # Inject all position hidden states through cross-attention
        # Need to temporarily resize the buffer
        gen_b = generate_with_engram(model, query_ids, full_hidden, device, args.max_new_tokens)
        text_b = tokenizer.decode(gen_b, skip_special_tokens=True)
        hits_b, total_b = count_answer_hits(text_b, needle.answer_tokens)

        # ---- Condition C: Text prepend ----
        gen_c = generate_with_prepend(model, query_ids, needle_ids, device, args.max_new_tokens)
        text_c = tokenizer.decode(gen_c, skip_special_tokens=True)
        hits_c, total_c = count_answer_hits(text_c, needle.answer_tokens)

        # ---- Condition D: Random hidden states ----
        random_hidden = torch.randn_like(full_hidden)
        gen_d = generate_with_engram(model, query_ids, random_hidden, device, args.max_new_tokens)
        text_d = tokenizer.decode(gen_d, skip_special_tokens=True)
        hits_d, total_d = count_answer_hits(text_d, needle.answer_tokens)

        # Record results
        for cond, hits, total, text in [
            ("A_mean_pooled", hits_a, total_a, text_a),
            ("B_full_hidden", hits_b, total_b, text_b),
            ("C_text_prepend", hits_c, total_c, text_c),
            ("D_random", hits_d, total_d, text_d),
        ]:
            conditions[cond]["hits"] += hits
            conditions[cond]["total"] += total
            conditions[cond]["details"].append({
                "needle": needle.category,
                "hits": hits,
                "total": total,
                "text": text[:200],
            })

        print(f"\n  Results:")
        print(f"    A (mean-pooled):  {hits_a}/{total_a} — {text_a[:100]}...")
        print(f"    B (full hidden):  {hits_b}/{total_b} — {text_b[:100]}...")
        print(f"    C (text prepend): {hits_c}/{total_c} — {text_c[:100]}...")
        print(f"    D (random):       {hits_d}/{total_d} — {text_d[:100]}...")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<25} {'Hits':>6} {'Total':>6} {'Recall':>8}")
    print(f"  {'-'*50}")
    for cond_name, cond in conditions.items():
        recall = cond["hits"] / cond["total"] if cond["total"] > 0 else 0
        print(f"  {cond_name:<25} {cond['hits']:>6} {cond['total']:>6} {recall:>8.1%}")

    # Diagnosis
    print(f"\n  Interpretation:")
    recall_b = conditions["B_full_hidden"]["hits"] / conditions["B_full_hidden"]["total"]
    recall_a = conditions["A_mean_pooled"]["hits"] / conditions["A_mean_pooled"]["total"]
    recall_c = conditions["C_text_prepend"]["hits"] / conditions["C_text_prepend"]["total"]
    recall_d = conditions["D_random"]["hits"] / conditions["D_random"]["total"]

    if recall_b > recall_a + 0.1:
        print(f"  → H1 SUPPORTED: Full hidden states ({recall_b:.0%}) >> mean pooled ({recall_a:.0%})")
        print(f"    Mean pooling destroys token-level information.")
    elif recall_b <= recall_a + 0.05:
        print(f"  → H2 SUPPORTED: Full hidden states ({recall_b:.0%}) ≈ mean pooled ({recall_a:.0%})")
        print(f"    Information isn't extractable from hidden states regardless of pooling.")

    if recall_c > recall_b + 0.1:
        print(f"  → Cross-attention can't replicate in-context learning (C={recall_c:.0%} >> B={recall_b:.0%})")
        print(f"    The autoregressive attention path carries something cross-attention doesn't.")
    elif recall_c <= recall_a + 0.05:
        print(f"  → Model can't ground from ANY source at 510M params (C={recall_c:.0%} ≈ A={recall_a:.0%})")
        print(f"    The grounding problem is a capacity problem, not a mechanism problem.")

    if recall_d >= recall_b:
        print(f"  → WARNING: Random hidden states ({recall_d:.0%}) ≥ full hidden ({recall_b:.0%})")
        print(f"    The cross-attention may not be using the injected content at all.")

    # Save
    out_path = Path("results/v18_cross_attn/exp_hidden_states.json")
    with open(out_path, "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
