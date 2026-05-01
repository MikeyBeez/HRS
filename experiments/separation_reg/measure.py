"""Measure separation, routing, retrieval at library sizes {10, 20, 50,
100, 200} for both procedures.

For each (procedure, size N):
  1. Take first N engrams from engrams_{A|B}.pt (stored adapter-active).
  2. Train W (1024x1024 linear, identity-init, InfoNCE 500 steps) on
     (L0_para_train, target_id) pairs across the N adapters' training
     paraphrases.
  3. Separation stats: pairwise cosine similarity of W(stored) (the
     post-projection view, since routing happens here).
     Actually — simpler reading is on stored engrams DIRECTLY. We report
     both: raw L5 stored cos AND post-W cos.
  4. Routing accuracy: for each adapter's held-out paraphrases, compute
     L0(no LoRA), project via W, cosine to stored engrams, argmax.
  5. Retrieval accuracy: load argmax adapter, generate, substring-check.

Subsamples: at size 200, eval retrieval on 50 random adapters (faster).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, hidden_at_layer,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)

PPD = REPO / "experiments/per_passage_dickens"
SR = REPO / "experiments/separation_reg"

D = 1024
RANK = 128
ALPHA = RANK * 2
SIZES = [10, 20, 50, 100, 200]


def encode(tokenizer, text, device, ctx=512):
    ids = tokenizer.encode(text, add_special_tokens=False)[:ctx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach()


@torch.no_grad()
def generate(model, ids_t, n_tokens=30, gen_seed=0,
              temperature=0.6, top_k=20):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, gen):
    if answer.lower() in gen.lower():
        return True
    a = answer.replace(",", "").replace(" ", "").lower()
    g = gen.replace(",", "").replace(" ", "").lower()
    return bool(a) and a in g


def train_W(L0_mat, lib_engrams, target_ids, n_steps=500, lr=1e-3,
            temp=0.05, device=None):
    """InfoNCE: L0_mat (M, D), lib_engrams (N, D), target_ids (M,)."""
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(lib_engrams, dim=-1)
    for step in range(n_steps):
        proj_n = F.normalize(W(L0_mat), dim=-1)
        sim = (proj_n @ keys_n.T) / temp
        loss = F.cross_entropy(sim, target_ids)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return W


def pairwise_stats(vecs):
    """vecs: (N, D). Returns mean, min, max, p10/50/90 of pairwise cos."""
    n = vecs.shape[0]
    if n < 2:
        return {"n": n, "mean": float("nan"), "min": float("nan"),
                "max": float("nan"), "p10": float("nan"), "p50": float("nan"),
                "p90": float("nan")}
    vn = F.normalize(vecs, dim=-1)
    sim = vn @ vn.T  # (n, n)
    # Take upper triangle excluding diagonal
    iu = torch.triu_indices(n, n, offset=1)
    pairs = sim[iu[0], iu[1]]
    return {
        "n": n,
        "mean": float(pairs.mean().item()),
        "min":  float(pairs.min().item()),
        "max":  float(pairs.max().item()),
        "p10":  float(pairs.quantile(0.10).item()),
        "p50":  float(pairs.median().item()),
        "p90":  float(pairs.quantile(0.90).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", choices=["A", "B"], required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((SR / "data/library_200.json").read_text())

    print(f"=== Procedure {args.procedure} measurement ===")

    # Load engrams
    eng = torch.load(SR / f"results/engrams_{args.procedure}.pt",
                      map_location=device, weights_only=False)
    engrams_all = eng["engrams"].to(device)  # (200, D)
    print(f"  loaded engrams shape: {engrams_all.shape}")

    # Build base model with LoRA structure
    print("  loading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)

    # Adapter dir
    adir = SR / ("adapters_a" if args.procedure == "A" else "adapters_b")

    # Pre-compute L0 means (base, no LoRA) for ALL training paraphrases and
    # all held-out paraphrases of all 200 adapters. These are the same
    # regardless of library size.
    print("  computing L0 (base, no-LoRA) for train+held-out paraphrases ...")
    train_L0 = {}      # adapter_k -> list of (4, D) tensors
    held_L0  = {}      # adapter_k -> list of (3, D) tensors
    held_text = {}     # adapter_k -> list of held-out paraphrase strings
    answer_by_k = {}
    for k, entry in enumerate(library):
        ts = []
        for p in entry["paraphrases_train"]:
            ts.append(l0_mean(model, encode(tokenizer, p, device)))
        train_L0[k] = torch.stack(ts)
        hs = []
        for p in entry["paraphrases_held_out"]:
            hs.append(l0_mean(model, encode(tokenizer, p, device)))
        held_L0[k] = torch.stack(hs)
        held_text[k] = entry["paraphrases_held_out"]
        answer_by_k[k] = entry["answer"]

    results = []
    rng_global = random.Random(42)

    for N in SIZES:
        t_size = time.time()
        print(f"\n--- size N={N} ---")
        # Stored engrams: first N
        stored = engrams_all[:N]  # (N, D)

        # Build W training data: per adapter k in 0..N-1, per training paraphrase
        L0_train = []; ids_train = []
        for k in range(N):
            for vec in train_L0[k]:
                L0_train.append(vec)
                ids_train.append(k)
        L0_train_t = torch.stack(L0_train).to(device)
        ids_train_t = torch.tensor(ids_train, dtype=torch.long, device=device)

        # Train W
        t0 = time.time()
        W = train_W(L0_train_t, stored, ids_train_t,
                     n_steps=500, lr=1e-3, temp=0.05, device=device)
        w_wall = time.time() - t0

        # Separation stats: stored engrams (raw L5) AND W-projected engrams
        # ... actually post-W is (W @ L0_query) space, not (W @ stored).
        # The comparison space is: query=W(L0), stored=L5_active (no W).
        # So separation should be measured on stored DIRECTLY.
        sep_stored = pairwise_stats(stored)
        # Also report post-W view of stored — useful as a sanity check
        # (but routing doesn't actually project stored).
        sep_post_W = pairwise_stats(W(stored))

        # Routing accuracy: for each adapter k in 0..N-1, for each held-out
        # paraphrase, project L0 through W and find argmax cos vs stored.
        keys_n = F.normalize(stored, dim=-1)
        n_correct = 0; n_total = 0
        argmax_collected = []
        for k in range(N):
            for vec in held_L0[k]:
                with torch.no_grad():
                    proj = W(vec.unsqueeze(0))
                    pn = F.normalize(proj, dim=-1)
                    sim = (pn @ keys_n.T).squeeze(0)
                    am = sim.argmax().item()
                argmax_collected.append((k, am))
                n_total += 1
                if am == k: n_correct += 1
        routing_rate = n_correct / n_total

        # Retrieval accuracy. For time budget, evaluate on min(N, 50)
        # randomly-sampled adapters; per adapter, all 3 held-out × 1 seed.
        retrieval_n = 0; retrieval_hit = 0
        sample_ks = rng_global.sample(range(N), min(N, 50))
        for k in sample_ks:
            for j, hp in enumerate(held_text[k]):
                # Route
                vec = held_L0[k][j]
                with torch.no_grad():
                    proj = W(vec.unsqueeze(0))
                    pn = F.normalize(proj, dim=-1)
                    sim = (pn @ keys_n.T).squeeze(0)
                    routed = sim.argmax().item()
                # Load that adapter
                sd = torch.load(adir / f"adapter_{routed:03d}.pt",
                                  map_location=device, weights_only=False)
                load_lora_state_dict(model, sd)
                # Generate
                ids_t = encode(tokenizer, hp, device)
                gen = generate(model, ids_t, n_tokens=20, gen_seed=0)
                cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                          skip_special_tokens=True)
                hit = check_match(answer_by_k[k], cont)
                retrieval_n += 1
                if hit: retrieval_hit += 1
        retrieval_rate = retrieval_hit / max(1, retrieval_n)
        # Reset adapter to zero for next iteration
        reset_lora_to_zero(model)

        rec = {
            "N": N,
            "sep_stored": sep_stored,
            "sep_post_W": sep_post_W,
            "routing_acc": routing_rate,
            "routing_n": n_total,
            "retrieval_acc": retrieval_rate,
            "retrieval_n": retrieval_n,
            "w_wall_s": w_wall,
            "size_wall_s": time.time() - t_size,
        }
        results.append(rec)
        print(f"  N={N}  stored cos: mean={sep_stored['mean']:.3f} "
              f"min={sep_stored['min']:.3f} max={sep_stored['max']:.3f}  "
              f"routing={routing_rate:.3f} ({n_correct}/{n_total})  "
              f"retrieval={retrieval_rate:.3f}  "
              f"wall={time.time()-t_size:.0f}s")

    out_path = SR / f"results/measure_{args.procedure}.json"
    out_path.write_text(json.dumps({
        "procedure": args.procedure,
        "lambda": float(eng.get("lambda", 0.0)),
        "sizes": SIZES,
        "results": results,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
