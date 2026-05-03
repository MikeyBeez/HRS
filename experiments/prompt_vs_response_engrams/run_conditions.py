"""Run all conditions for the prompt-vs-response-engrams experiment.

Conditions:
  1. full_context     — concat all 100 turns; ceiling
  2. random_engrams   — k random engrams as inputs_embeds prefix
  3. prompts_as_context — concat 100 prompts (no responses); KEY new condition
  4. recent_only      — last N turns (full text); truncation baseline
  5..7. prompt+response engrams at L8/L16/L24 (last-token)
  8..10. prompt-only engrams at L8/L16/L24 (last-token)
  11. uniform_pool    — average all engrams of best (layer, content) → single vector

For engram conditions, train a fresh W via multi-label InfoNCE on the
5 validation probes (same as prior experiment).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/prompt_vs_response_engrams"
PRIOR = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.probes import (
    VAL_PROBES, TEST_PROBES, build_probe_records,
)

BASE = "mistralai/Mistral-7B-v0.1"
LAYERS = [8, 16, 24]
TOP_K = 10
TEMP = 0.1
GEN_TOKENS = 180
RECENT_N = 8


def get_layer_pool(model, ids, layer):
    """Last-token hidden state at given layer. ids: (1, T)."""
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
        return out.hidden_states[layer][0, -1, :].float()  # (D,)


def train_W(probe_engrams, probe_relevant_ids, library_engrams,
            n_steps=500, lr=1e-3, temp=0.05, device="cuda"):
    """Multi-label InfoNCE: each (probe, positive_engram) pair classifies
    against the engram bank."""
    D = library_engrams.shape[-1]
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(library_engrams, dim=-1)
    pairs = []
    for pi, rel in enumerate(probe_relevant_ids):
        for tid in rel:
            pairs.append((pi, tid))
    if not pairs:
        return W, 0.0
    pairs_p = torch.tensor([p[0] for p in pairs], device=device)
    pairs_t = torch.tensor([p[1] for p in pairs], device=device)
    for _ in range(n_steps):
        proj_n = F.normalize(W(probe_engrams), dim=-1)
        sim = (proj_n[pairs_p] @ keys_n.T) / temp
        loss = F.cross_entropy(sim, pairs_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        proj_n = F.normalize(W(probe_engrams), dim=-1)
        sim = (proj_n[pairs_p] @ keys_n.T) / temp
        train_acc = (sim.argmax(dim=-1) == pairs_t).float().mean().item()
    return W, train_acc


def generate_text(model, tokenizer, prompt, inputs_embeds=None,
                   max_new_tokens=GEN_TOKENS, device="cuda",
                   max_input_tokens=2000):
    """Generate. If inputs_embeds is None, generate from text prompt
    (truncated to max_input_tokens). Else prepend inputs_embeds to the
    embedded probe."""
    with torch.no_grad():
        if inputs_embeds is None:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if ids.shape[1] > max_input_tokens:
                ids = ids[:, -max_input_tokens:]
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            return tokenizer.decode(out[0, ids.shape[1]:],
                                      skip_special_tokens=True), ids.shape[1]
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        emb = model.model.embed_tokens(ids)
        prefix = inputs_embeds.unsqueeze(0).to(emb.dtype)
        full = torch.cat([prefix, emb], dim=1)
        n_input_tokens = full.shape[1]
        out = model.generate(
            inputs_embeds=full, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True), n_input_tokens


def routing_pr(top_k_ids, gt_ids, k):
    if k == 0 or not gt_ids: return 0.0, 0.0
    pred = set(top_k_ids[:k]); gt = set(gt_ids)
    overlap = pred & gt
    p = len(overlap) / k
    r = len(overlap) / len(gt)
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--temp", type=float, default=TEMP)
    args = ap.parse_args()

    device = torch.device("cuda")
    rng = np.random.default_rng(42)

    print(f"Loading {BASE} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    d_model = model.config.hidden_size

    turns = json.loads((PRIOR / "data/turns.json").read_text())["turns"]
    print(f"  {len(turns)} turns")

    # Load engrams (6 sets)
    eng_npz = np.load(EXP / "data/engrams_v2.npz")
    eng_sets = {key: torch.tensor(eng_npz[key], dtype=torch.float32, device=device)
                for key in eng_npz.files}
    print(f"  engram sets: {list(eng_sets.keys())}")

    # Anisotropy from saved
    aniso = json.loads((EXP / "data/anisotropy.json").read_text())

    # Probes
    val = build_probe_records(VAL_PROBES)
    test = build_probe_records(TEST_PROBES)

    # Compute probe-side embeddings for each (layer) at last-token using
    # the PROBE TEXT itself.
    print("\nComputing probe-side last-token embeddings at L8, L16, L24 ...")
    probe_emb = {}
    for layer in LAYERS:
        val_e = []
        for p in val:
            ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
            val_e.append(get_layer_pool(model, ids, layer))
        val_e = torch.stack(val_e)
        test_e = []
        for p in test:
            ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
            test_e.append(get_layer_pool(model, ids, layer))
        test_e = torch.stack(test_e)
        probe_emb[layer] = {"val": val_e, "test": test_e}
        print(f"  L{layer}: val {val_e.shape}, test {test_e.shape}")

    # Train 6 W projections (one per engram-type / layer)
    print("\nTraining 6 W projections ...")
    val_relevant = [p["relevant_ids"] for p in val]
    Ws = {}
    train_accs = {}
    for layer in LAYERS:
        for kind in ("p", "pr"):
            key = f"{layer}_{kind}"
            W, ta = train_W(probe_emb[layer]["val"], val_relevant,
                              eng_sets[key], n_steps=500, lr=1e-3,
                              temp=0.05, device=device)
            # Convert to fp16 to save memory; routing math is tolerant.
            W = W.half()
            Ws[key] = W; train_accs[key] = ta
            print(f"  W[{key}]: train_acc={ta:.3f}")
    torch.cuda.empty_cache()

    # Pre-compute concatenated text variants
    full_context_text = "\n\n".join(t["full_text"] for t in turns)
    prompts_only_text = "\n".join(t["prompt"] for t in turns)
    recent_only_text  = "\n\n".join(t["full_text"] for t in turns[-RECENT_N:])

    n_full = len(tokenizer.encode(full_context_text))
    n_prompts = len(tokenizer.encode(prompts_only_text))
    n_recent = len(tokenizer.encode(recent_only_text))
    print(f"\nToken budgets:")
    print(f"  full_context:        {n_full:5d}")
    print(f"  prompts_as_context:  {n_prompts:5d}")
    print(f"  recent_only ({RECENT_N}):     {n_recent:5d}")
    print(f"  engram conditions:   {args.top_k:5d} (10 prefix vectors)")

    # Run conditions
    print(f"\n=== Running conditions × {len(test)} test probes ===")
    results = []
    t_total = time.time()

    for pi, p in enumerate(test):
        probe_text = p["prompt"]
        gt_ids = p["relevant_ids"]
        rec = {"probe_idx": pi, "prompt": probe_text,
               "n_relevant": len(gt_ids), "relevant_ids": gt_ids}
        print(f"\n[{pi+1}/{len(test)}] {probe_text[:60]}...  "
              f"({len(gt_ids)} relevant)")

        short_probe = probe_text + "\n\nReport: "

        # Cond 1: full_context
        full_prompt = full_context_text + "\n\n" + probe_text + "\n\nReport: "
        gen, n_in = generate_text(model, tokenizer, full_prompt)
        rec["gen_full_context"] = gen
        rec["tokens_full_context"] = n_in

        # Cond 2: random engrams (use 16_pr engrams as the bank)
        rand_ids = rng.choice(100, size=args.top_k, replace=False).tolist()
        rand_engrams = eng_sets["16_pr"][rand_ids]
        gen, n_in = generate_text(model, tokenizer, short_probe,
                                    inputs_embeds=rand_engrams)
        rec["gen_random_engrams"] = gen
        rec["tokens_random_engrams"] = n_in
        rec["random_ids"] = rand_ids

        # Cond 3: prompts_as_context (KEY new condition)
        prompts_prompt = prompts_only_text + "\n\n" + probe_text + "\n\nReport: "
        gen, n_in = generate_text(model, tokenizer, prompts_prompt)
        rec["gen_prompts_as_context"] = gen
        rec["tokens_prompts_as_context"] = n_in

        # Cond 4: recent_only
        recent_prompt = recent_only_text + "\n\n" + probe_text + "\n\nReport: "
        gen, n_in = generate_text(model, tokenizer, recent_prompt)
        rec["gen_recent_only"] = gen
        rec["tokens_recent_only"] = n_in

        # Conds 5-10: 6 engram routings (3 layers × 2 content)
        for layer in LAYERS:
            for kind in ("p", "pr"):
                key = f"{layer}_{kind}"
                W = Ws[key]
                eng_bank = eng_sets[key]
                keys_n = F.normalize(eng_bank, dim=-1)
                # Project probe through W and route (fp16-safe)
                pe = probe_emb[layer]["test"][pi].half()
                with torch.no_grad():
                    proj = W(pe.unsqueeze(0))
                    proj_n = F.normalize(proj.float(), dim=-1)
                    sim = (proj_n @ keys_n.T).squeeze(0)
                topk = sim.argsort(descending=True)[:args.top_k].cpu().tolist()
                p_at_k, r_at_k = routing_pr(topk, gt_ids, args.top_k)
                rec[f"routing_p_at_k_{key}"] = p_at_k
                rec[f"routing_r_at_k_{key}"] = r_at_k
                rec[f"topk_{key}"] = topk
                # Generate
                topk_e = eng_bank[topk]
                gen, n_in = generate_text(model, tokenizer, short_probe,
                                            inputs_embeds=topk_e)
                rec[f"gen_engram_{key}"] = gen
                rec[f"tokens_engram_{key}"] = n_in

        # Cond 11: uniform pool (use 16_pr as the bank — best layer guess)
        uniform = eng_sets["16_pr"].mean(dim=0, keepdim=True)
        gen, n_in = generate_text(model, tokenizer, short_probe,
                                    inputs_embeds=uniform)
        rec["gen_uniform_pool"] = gen
        rec["tokens_uniform_pool"] = n_in

        results.append(rec)

    out = {
        "results": results,
        "anisotropy": aniso,
        "train_accs": train_accs,
        "config": {"top_k": args.top_k, "temp": args.temp,
                    "recent_n": RECENT_N, "gen_tokens": GEN_TOKENS,
                    "layers": LAYERS},
        "wall_total_s": time.time() - t_total,
    }
    out_path = EXP / "results/conditions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Total wall: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
