"""Train W and run all 5 conditions × 20 test probes on Mistral-7B.

W is trained on the 5 validation probes via InfoNCE: for each val probe
with R relevant turn-ids, we form (probe_l16, target_engram) pairs by
treating each relevant engram as a positive for that probe. (Multi-label
InfoNCE: each probe's loss is mean over its positive engrams of the
standard cross-entropy with the engram-bank as classes.)

Conditions (per probe):
  1. full_context: concatenate all 100 full_texts as one big context;
                   ask the probe; generate.
  2. recent_only: take only the LAST N turns; concatenate; ask; generate.
  3. random_engrams: pick K random engrams; inject mean as a context-
                     replacement vector; generate.
  4. uniform_pool: average ALL 100 engrams; inject as a single vector
                   prefix; generate.
  5. engram_routing: probe → W → softmax(temp) over engram bank → top-K
                     engrams → weighted sum → inject as prefix; generate.

For "inject as prefix", we use Mistral's `inputs_embeds` interface and
prepend the injected vector(s) to the embedded probe.

Generation: greedy, 200 new tokens.
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
EXP = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.probes import (
    VAL_PROBES, TEST_PROBES, build_probe_records,
)

BASE = "mistralai/Mistral-7B-v0.1"
LAYER = 16
GEN_TOKENS = 180
RECENT_N = 8        # how many recent turns the truncation baseline keeps
TOP_K = 10          # for engram routing


def get_l16_pool(model, ids):
    """Run forward, take layer 16 hidden state mean."""
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[LAYER]  # (1, T, D)
        return h.mean(dim=1).squeeze(0).float()  # (D,)


def train_W(probe_engrams, probe_relevant_ids, library_engrams,
            n_steps=500, lr=1e-3, temp=0.05, device="cuda"):
    """probe_engrams: (P, D) probe-side hidden mid-layer means.
    probe_relevant_ids: list of P lists of int ids (positives in library).
    library_engrams: (N, D) engram bank.

    Multi-label InfoNCE: for each (probe, relevant_id), classify against
    the full library. Loss is mean over (probe × positive) pairs.
    """
    D = library_engrams.shape[-1]
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(library_engrams, dim=-1)

    # Build flat (probe_idx, target_id) pairs for training
    pairs = []
    for pi, rel in enumerate(probe_relevant_ids):
        for tid in rel:
            pairs.append((pi, tid))
    if not pairs:
        raise ValueError("No relevant pairs to train W")
    pairs_p = torch.tensor([p[0] for p in pairs], device=device)
    pairs_t = torch.tensor([p[1] for p in pairs], device=device)

    for step in range(n_steps):
        proj_n = F.normalize(W(probe_engrams), dim=-1)  # (P, D)
        # For each (probe_idx, target_id) pair, compute logits
        sim = (proj_n[pairs_p] @ keys_n.T) / temp  # (n_pairs, N)
        loss = F.cross_entropy(sim, pairs_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        proj_n = F.normalize(W(probe_engrams), dim=-1)
        sim = (proj_n[pairs_p] @ keys_n.T) / temp
        train_acc = (sim.argmax(dim=-1) == pairs_t).float().mean().item()
    return W, train_acc


def truncate_concat(turns, n_recent, max_tokens=3500):
    """Concatenate the last n_recent turns; truncate to max_tokens."""
    txt = "\n\n".join(t["full_text"] for t in turns[-n_recent:])
    return txt


def generate_answer(model, tokenizer, prompt, inputs_embeds=None,
                     max_new_tokens=GEN_TOKENS, device="cuda"):
    """If inputs_embeds is None, generate from prompt (text). Else
    embed prompt and PREPEND inputs_embeds (shape (P_prefix, D))."""
    if inputs_embeds is None:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return gen
    # Embed-mode: prepend prefix vectors
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    emb = model.model.embed_tokens(ids)  # (1, T, D)
    prefix = inputs_embeds.unsqueeze(0).to(emb.dtype)  # (1, P_prefix, D)
    full = torch.cat([prefix, emb], dim=1)
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=full, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    # When using inputs_embeds, generate returns just the new tokens
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    return gen


def routing_precision_recall(top_k_ids, gt_ids, k):
    if k == 0 or len(gt_ids) == 0:
        return 0.0, 0.0
    pred_set = set(top_k_ids[:k])
    gt_set = set(gt_ids)
    overlap = pred_set & gt_set
    p = len(overlap) / k
    r = len(overlap) / len(gt_set)
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--gen_tokens", type=int, default=GEN_TOKENS)
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

    # Load turns + engrams
    turns = json.loads((EXP / "data/turns.json").read_text())["turns"]
    engrams_np = np.load(EXP / "data/engrams.npy")
    engrams = torch.tensor(engrams_np, dtype=torch.float32, device=device)
    print(f"  {len(turns)} turns, engrams shape={engrams.shape}")

    # Probes
    val = build_probe_records(VAL_PROBES)
    test = build_probe_records(TEST_PROBES)
    print(f"  val probes: {len(val)}, test probes: {len(test)}")

    # Compute probe-side L16 means for val + test probes
    print("\nComputing probe-side L16 means ...")
    val_probe_emb = []
    for p in val:
        ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
        val_probe_emb.append(get_l16_pool(model, ids))
    val_probe_emb = torch.stack(val_probe_emb)
    test_probe_emb = []
    for p in test:
        ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
        test_probe_emb.append(get_l16_pool(model, ids))
    test_probe_emb = torch.stack(test_probe_emb)
    print(f"  val_probe_emb: {val_probe_emb.shape}, "
          f"test_probe_emb: {test_probe_emb.shape}")

    # Train W
    print("\nTraining W on validation probes ...")
    val_relevant = [p["relevant_ids"] for p in val]
    W, train_acc = train_W(val_probe_emb, val_relevant, engrams,
                              n_steps=500, lr=1e-3, temp=0.05, device=device)
    print(f"  W train_acc={train_acc:.3f}")

    # Pre-compute the FULL-context concatenation (ceiling input)
    full_context_text = "\n\n".join(t["full_text"] for t in turns)
    n_full_tokens = len(tokenizer.encode(full_context_text))
    print(f"\nFull-context length: {n_full_tokens} tokens")

    # Recent-only baseline: pick number of turns whose token count
    # roughly matches the engram budget. Engram budget = TOP_K positions
    # in input space. Recent-only with TOP_K turns ≈ token-count match.
    # For fairness with engram conditions, use a small recent-N (~8 turns).
    recent_only_text = truncate_concat(turns, RECENT_N)
    n_recent_tokens = len(tokenizer.encode(recent_only_text))
    print(f"Recent-only length (last {RECENT_N} turns): {n_recent_tokens} tokens")

    # Run all conditions × all test probes
    print(f"\n=== Running {len(test)} test probes × 5 conditions ===")
    results = []
    t_total = time.time()

    keys_n = F.normalize(engrams, dim=-1)

    for pi, p in enumerate(test):
        probe_text = p["prompt"]
        gt_ids = p["relevant_ids"]
        print(f"\n[{pi+1}/{len(test)}] {probe_text[:60]}...  "
              f"({len(gt_ids)} relevant)")

        rec = {"probe_idx": pi, "prompt": probe_text,
               "n_relevant": len(gt_ids), "relevant_ids": gt_ids}

        # Routing decision (used by conditions 3, 4, 5)
        proj = W(test_probe_emb[pi].unsqueeze(0))
        proj_n = F.normalize(proj, dim=-1)
        sim = (proj_n @ keys_n.T).squeeze(0)        # (N,)
        sim_softmax = F.softmax(sim / args.temp, dim=-1)
        topk_ids = sim.argsort(descending=True)[:args.top_k].cpu().tolist()
        # Routing P@k and R@k
        p_at_k, r_at_k = routing_precision_recall(topk_ids, gt_ids, args.top_k)
        rec["routing_p_at_k"] = p_at_k
        rec["routing_r_at_k"] = r_at_k
        rec["topk_ids"] = topk_ids
        # Recall at k=20 too (more permissive)
        topk_20 = sim.argsort(descending=True)[:20].cpu().tolist()
        _, r_at_20 = routing_precision_recall(topk_20, gt_ids, 20)
        rec["routing_r_at_20"] = r_at_20

        # ---- Condition 1: full context ----
        t0 = time.time()
        prompt_full = full_context_text + "\n\n" + probe_text + "\n\nReport: "
        # If too long, this will OOM; we'll just generate from the last 4000 tokens
        ids = tokenizer.encode(prompt_full, return_tensors="pt")
        if ids.shape[1] > 4000:
            # Truncate the *front* of the context, keep the probe at the end
            ids = ids[:, -4000:]
            ids = ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=args.gen_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            full_gen = tokenizer.decode(out[0, ids.shape[1]:],
                                          skip_special_tokens=True)
        else:
            full_gen = generate_answer(model, tokenizer, prompt_full,
                                          max_new_tokens=args.gen_tokens)
        rec["gen_full_context"] = full_gen
        print(f"  full_context wall={time.time()-t0:.0f}s")

        # ---- Condition 2: recent only (last N turns) ----
        t0 = time.time()
        prompt_recent = recent_only_text + "\n\n" + probe_text + "\n\nReport: "
        rec["gen_recent_only"] = generate_answer(
            model, tokenizer, prompt_recent, max_new_tokens=args.gen_tokens,
        )
        print(f"  recent_only wall={time.time()-t0:.0f}s")

        # ---- Condition 3: random engrams (K) ----
        t0 = time.time()
        rand_ids = rng.choice(engrams.shape[0], size=args.top_k,
                                replace=False).tolist()
        rand_engrams = engrams[rand_ids]
        prompt_short = probe_text + "\n\nReport: "
        rec["gen_random_engrams"] = generate_answer(
            model, tokenizer, prompt_short,
            inputs_embeds=rand_engrams,
            max_new_tokens=args.gen_tokens,
        )
        rec["random_ids"] = rand_ids
        print(f"  random_engrams wall={time.time()-t0:.0f}s")

        # ---- Condition 4: uniform pooling (mean of all 100) ----
        t0 = time.time()
        uniform_pool = engrams.mean(dim=0, keepdim=True)  # (1, D)
        rec["gen_uniform_pool"] = generate_answer(
            model, tokenizer, prompt_short,
            inputs_embeds=uniform_pool,
            max_new_tokens=args.gen_tokens,
        )
        print(f"  uniform_pool wall={time.time()-t0:.0f}s")

        # ---- Condition 5: engram routing (top-K weighted by softmax) ----
        t0 = time.time()
        # Re-rank by softmax over the top-K (or use the topk-only weights)
        topk_w = sim_softmax[topk_ids]
        topk_w = topk_w / topk_w.sum()           # renormalize over top-K
        topk_e = engrams[topk_ids]               # (K, D)
        # Use the K engrams individually (not as a single pooled vector)
        rec["gen_engram_routing"] = generate_answer(
            model, tokenizer, prompt_short,
            inputs_embeds=topk_e,
            max_new_tokens=args.gen_tokens,
        )
        rec["topk_weights"] = topk_w.cpu().tolist()
        print(f"  engram_routing wall={time.time()-t0:.0f}s")

        results.append(rec)

    # Save raw results
    out_path = EXP / "results/conditions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "results": results,
        "config": {"top_k": args.top_k, "temp": args.temp,
                    "recent_n": RECENT_N, "gen_tokens": args.gen_tokens,
                    "layer": LAYER},
        "wall_total_s": time.time() - t_total,
    }, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Total wall: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
