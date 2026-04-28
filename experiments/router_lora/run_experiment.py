"""Two-pass ingest + answer-recall evaluation for router_lora experiment.

Conditions:
  A: learned router + LoRA (full system)
  B: random router + LoRA (random uniform routing weights)
  C: no router, no LoRA (frozen base only)

For each (condition, seed):
  1. Load base from `results/base_shakespeare.pt`. Freeze base.
  2. Init fresh Router, UpdateMechanism, and LoRA matrices.
  3. (A/B only) Run ingest: N_EPOCHS over shuffled Dickens passages, two-pass
     loop per passage, backprop loss = pass2 - pass1 + λ * mean(weights).
  4. Evaluate on the 50 queries via greedy generation + substring match.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.diagonal_attention.data import load_shakespeare
from experiments.router_lora.model import (
    TinyTransformer, TinyConfig, Router, UpdateMechanism,
)


N_EPOCHS = 5             # passes through the Dickens passage list during ingest
INGEST_LR = 1e-3
SPARSITY_LAMBDA = 0.001  # was 0.01; lower so router doesn't collapse
DELTA_NORM_CAP = 0.05    # cap each step's delta L2 norm so the running LoRA doesn't run away
EVAL_GEN_LEN = 60        # chars to generate after each probe


def encode(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids: torch.Tensor, itos: dict) -> str:
    return "".join(itos[int(i)] for i in ids)


def load_base(device, vocab_size, ctx_len=512) -> TinyTransformer:
    cfg = TinyConfig(vocab_size=vocab_size, ctx_len=ctx_len)
    model = TinyTransformer(cfg).to(device)
    ckpt = torch.load(REPO / "experiments/router_lora/results/base_shakespeare.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def reset_lora_and_freeze_base(model: TinyTransformer):
    """Re-init LoRA matrices to (Normal, zero) and freeze EVERYTHING in the
    base model — including the LoRA matrices. The LoRA matrices are accumulated
    as detached deltas during ingest, but never directly optimized — that
    prevents the optimizer from gaming the differential loss by inflating
    pass-1 baselines.
    """
    for blk in model.blocks:
        if blk.is_lora_layer:
            torch.nn.init.normal_(blk.ffn.lora_A, std=0.02)
            torch.nn.init.zeros_(blk.ffn.lora_B)
    for p in model.parameters():
        p.requires_grad_(False)


def run_two_pass_step(model, router, update_mech, passage_ids,
                        condition, lambda_sparsity, rng_torch):
    """One backward pass on router + update_mech for one passage.
    LoRA matrices stay frozen during the backward; accumulation of the delta
    happens AFTER the optimizer step (see the caller).

    condition ∈ {'learned', 'random'}.
    Returns (loss1, loss2, mean_router_weight, total_loss, delta_A, delta_B).
    """
    device = passage_ids.device
    x = passage_ids[:-1].unsqueeze(0)              # (1, T-1)
    y = passage_ids[1:].unsqueeze(0)               # (1, T-1)
    # Pass 1: current LoRA, capture router-layer post-attn hidden states.
    logits1, hidden = model(x, capture_router_layer=True)
    loss1 = F.cross_entropy(logits1.reshape(-1, logits1.shape[-1]), y.reshape(-1))

    if condition == "learned":
        weights = router(hidden)
    elif condition == "random":
        weights = torch.rand(hidden.shape[:2], device=device, generator=rng_torch)
    else:
        raise ValueError(condition)

    delta_A, delta_B = update_mech(hidden, weights)

    logits2, _ = model(x, delta_A=delta_A, delta_B=delta_B)
    loss2 = F.cross_entropy(logits2.reshape(-1, logits2.shape[-1]), y.reshape(-1))

    total = (loss2 - loss1) + lambda_sparsity * weights.mean()
    return (loss1.detach(), loss2.detach(), weights.mean().detach(), total,
             delta_A, delta_B)


@torch.no_grad()
def generate_continuation(model, prompt_ids, n_tokens, stoi_size, ctx_len,
                            temperature=1.0, top_k=0):
    """Greedy/argmax continuation by default (temperature=1, top_k=0 → argmax
    at temperature=0 isn't exposed; use top-1 deterministic via argmax)."""
    model.eval()
    x = prompt_ids.clone().unsqueeze(0)
    for _ in range(n_tokens):
        idx = x[:, -ctx_len:]
        logits, _ = model(idx)
        # Greedy
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, next_tok], dim=1)
    return x[0]


def evaluate_recall(model, queries, stoi, itos, ctx_len, gen_len=EVAL_GEN_LEN):
    """Return (recall_fraction, per_query_results)."""
    device = next(model.parameters()).device
    model.eval()
    hits = 0
    results = []
    for q in queries:
        probe_ids = encode(q["probe"], stoi).to(device)
        out_ids = generate_continuation(model, probe_ids, gen_len, len(stoi), ctx_len)
        full = decode(out_ids, itos)
        cont = full[len(q["probe"]):]
        ans = q["answer"]
        # Case-insensitive substring match in continuation only
        hit = ans.lower() in cont.lower()
        if hit:
            hits += 1
        results.append({"id": q["id"], "probe": q["probe"], "answer": ans,
                        "continuation": cont, "hit": hit})
    return hits / len(queries), results


def run_one_seed(condition: str, seed: int, passages: list[str],
                  queries: list[dict], stoi, itos):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_base(device, vocab_size=len(stoi), ctx_len=512)
    reset_lora_and_freeze_base(model)
    cfg.sparsity_lambda = SPARSITY_LAMBDA
    router = Router(cfg).to(device)
    update_mech = UpdateMechanism(cfg).to(device)

    history = []

    if condition == "no_lora":
        # No ingest. Frozen base only.
        recall, per_q = evaluate_recall(model, queries, stoi, itos, cfg.ctx_len)
        return {
            "condition": "no_lora", "seed": seed,
            "recall": recall, "n_queries": len(queries),
            "per_query": per_q, "history": [],
        }

    # Optimizer over router + update_mech (NOT the LoRA matrices, which
    # accumulate as detached deltas after each opt step).
    if condition == "random":
        params = list(update_mech.parameters())
    else:
        params = list(router.parameters()) + list(update_mech.parameters())
    opt = torch.optim.AdamW(params, lr=INGEST_LR, weight_decay=0.0,
                              betas=(0.9, 0.95))

    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(seed + 7919)

    encoded = [encode(p, stoi).to(device) for p in passages]
    rng_np = np.random.default_rng(seed)
    n_passages = len(encoded)

    model.train()
    step = 0
    for epoch in range(N_EPOCHS):
        order = rng_np.permutation(n_passages)
        for pi in order:
            ids = encoded[pi]
            # Truncate to ctx_len if needed
            if ids.shape[0] > cfg.ctx_len:
                ids = ids[:cfg.ctx_len]
            l1, l2, mw, total, dA, dB = run_two_pass_step(
                model, router, update_mech, ids,
                condition="learned" if condition == "learned" else "random",
                lambda_sparsity=SPARSITY_LAMBDA,
                rng_torch=rng_torch,
            )
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            # Accumulate delta into LoRA matrices (detached, no autograd).
            # Cap each step's delta L2 norm so accumulated LoRA doesn't run away
            # to gibberish — first attempt corrupted outputs at ~scale 1.
            with torch.no_grad():
                dA_d = dA.detach(); dB_d = dB.detach()
                nA = dA_d.norm(); nB = dB_d.norm()
                if nA > DELTA_NORM_CAP: dA_d = dA_d * (DELTA_NORM_CAP / nA)
                if nB > DELTA_NORM_CAP: dB_d = dB_d * (DELTA_NORM_CAP / nB)
                for blk in model.blocks:
                    if blk.is_lora_layer:
                        blk.ffn.lora_A.data.add_(dA_d)
                        blk.ffn.lora_B.data.add_(dB_d)
            history.append({
                "step": step, "epoch": epoch, "passage_idx": int(pi),
                "loss1": float(l1), "loss2": float(l2),
                "delta": float(l2 - l1),
                "mean_router_weight": float(mw),
                "total_loss": float(total.detach()),
            })
            step += 1

    recall, per_q = evaluate_recall(model, queries, stoi, itos, cfg.ctx_len)
    lora_A_norm = lora_B_norm = float("nan")
    for blk in model.blocks:
        if blk.is_lora_layer:
            lora_A_norm = float(blk.ffn.lora_A.norm().item())
            lora_B_norm = float(blk.ffn.lora_B.norm().item())
    return {
        "condition": condition, "seed": seed,
        "recall": recall, "n_queries": len(queries),
        "per_query": per_q,
        "history": history,
        "lora_A_norm": lora_A_norm,
        "lora_B_norm": lora_B_norm,
    }


def main():
    # Load Shakespeare to get the same stoi/itos used for tokenization
    train, val, info = load_shakespeare()
    stoi = info["stoi"]; itos = info["itos"]

    with open(REPO / "experiments/router_lora/data/ge_ch12_passages.json") as f:
        data = json.load(f)
    passages = data["passages"]

    with open(REPO / "experiments/router_lora/data/queries.json") as f:
        qf = json.load(f)
    queries = qf["queries"]

    print(f"Loaded {len(passages)} Dickens passages and {len(queries)} queries.")
    print(f"Vocab size: {len(stoi)}.")
    out_dir = REPO / "experiments/router_lora/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t0 = time.time()
    for condition in ["learned", "random", "no_lora"]:
        for seed in [0, 1, 2]:
            print(f"\n=== condition={condition}  seed={seed} ===")
            t_start = time.time()
            r = run_one_seed(condition, seed, passages, queries, stoi, itos)
            print(f"  recall={r['recall']:.3f}  n_steps={len(r['history'])}  "
                  f"wall={time.time()-t_start:.0f}s")
            if r["history"]:
                final = r["history"][-1]
                init = r["history"][0]
                print(f"  router_w start→end: {init['mean_router_weight']:.3f} → "
                      f"{final['mean_router_weight']:.3f}")
                print(f"  loss1 start→end:    {init['loss1']:.3f} → {final['loss1']:.3f}")
                print(f"  Δ(loss2-loss1) start→end: {init['delta']:+.3f} → {final['delta']:+.3f}")
            (out_dir / f"{condition}_seed{seed}.json").write_text(json.dumps(r, indent=2))
            all_results.append({
                "condition": condition, "seed": seed,
                "recall": r["recall"], "n_queries": r["n_queries"],
            })

    print(f"\n{'='*70}\nSUMMARY")
    print(f"{'='*70}")
    summary = {}
    for c in ["learned", "random", "no_lora"]:
        rs = [x["recall"] for x in all_results if x["condition"] == c]
        m = sum(rs) / len(rs)
        std = (sum((x - m) ** 2 for x in rs) / len(rs)) ** 0.5
        summary[c] = {"mean": m, "std": std, "seeds": rs}
        print(f"  {c:>10s}  mean={m:.3f}±{std:.3f}  seeds={rs}")
    (out_dir / "summary.json").write_text(json.dumps({
        "all_results": all_results, "summary": summary,
        "total_wall_s": time.time() - t0,
    }, indent=2))
    print(f"\nTotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
