"""Evaluate the four conditions on Dickens recall queries + Shakespeare quality.

Conditions:
  A: full system    (router-controlled LoRA)
  B: LoRA always-on (router bypassed)
  C: base only      (no LoRA)
  D: random router  (uniform [0,1] gate)

Three seeds per condition for evaluation: same trained Phase 1/2/3 (saving
8 minutes of retraining), three independent stochastic-decoding seeds. The
LoRA + router were trained once each; this measures the spread of the
recall metric under sampling noise. (Per-pipeline-seed retraining would be
a follow-up.)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.router_lora_phased.model import (
    TinyTransformer, TinyConfig, Router,
)


DATA_DIR = REPO / "experiments/router_lora_phased/data"
RESULT_DIR = REPO / "experiments/router_lora_phased/results"

GEN_TOKENS = 30          # generate this many tokens after each probe
TEMPERATURE = 0.8
TOP_K = 50


def load_full_system(device):
    info = json.loads((DATA_DIR / "info.json").read_text())
    cfg = TinyConfig(vocab_size=info["vocab_size"], dropout=0.0)
    model = TinyTransformer(cfg).to(device)
    ck2 = torch.load(RESULT_DIR / "phase2_lora.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck2["model_state_dict"])
    model.eval()
    router = Router(cfg).to(device)
    ck3 = torch.load(RESULT_DIR / "phase3_router.pt", map_location=device, weights_only=False)
    router.load_state_dict(ck3["router_state_dict"])
    router.eval()
    return model, router, cfg


@torch.no_grad()
def generate_with_lora_scale(model, ids, n_tokens, lora_scale, gen_seed,
                                temperature=TEMPERATURE, top_k=TOP_K):
    rng = torch.Generator(device=ids.device)
    rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        ctx = ids[:, -model.cfg.ctx_len:]
        logits, _ = model(ctx, lora_scale=lora_scale)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1, generator=rng)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def get_router_activation(model, router, prompt_ids):
    _, hidden = model(prompt_ids, lora_scale=0.0, capture_router_layer=True)
    return router(hidden)  # (B,)


def evaluate_condition(model, router, queries, tokenizer, condition, seed,
                         device):
    rng_np = np.random.default_rng(seed)
    activations = []
    hits = 0
    raw_hits = 0
    results = []
    for q in queries:
        ids = torch.tensor([tokenizer.encode(q["probe"])], device=device)
        if condition == "A_full":
            with torch.no_grad():
                act = get_router_activation(model, router, ids).item()
            scale = act
        elif condition == "B_always_on":
            scale = 1.0
            act = 1.0
        elif condition == "C_no_lora":
            scale = 0.0
            act = 0.0
        elif condition == "D_random":
            act = float(rng_np.random())
            scale = act
        activations.append(act)

        out = generate_with_lora_scale(model, ids, GEN_TOKENS, scale, gen_seed=seed * 1000 + q["id"])
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        cont = full[len(q["probe"]):]
        ans = q["answer"]
        # Substring match (case-insensitive)
        raw_hit = ans.lower() in cont.lower()
        # "Real" hit: answer must be ≥2 chars (filter trivial false positives)
        real_hit = raw_hit and len(ans) >= 2
        if raw_hit: raw_hits += 1
        if real_hit: hits += 1
        results.append({
            "id": q["id"], "probe": q["probe"], "answer": ans,
            "router_activation": act, "lora_scale_used": scale,
            "continuation": cont, "raw_hit": raw_hit, "real_hit": real_hit,
        })

    return {
        "condition": condition, "seed": seed,
        "recall_real": hits / len(queries),
        "recall_raw": raw_hits / len(queries),
        "n_queries": len(queries),
        "activation_mean": float(np.mean(activations)),
        "activation_std": float(np.std(activations)),
        "per_query": results,
    }


def evaluate_shakespeare_quality(model, router, tokenizer, sh_val_tokens,
                                    device, seeds=[0, 1, 2]):
    """For each seed, take a few Shakespeare-style prompts (from sh_val) and
    generate under both router-controlled and forced-zero LoRA. Measure the
    per-token CE on a continuation drawn from the same val text — lower is
    better. The expectation: router-controlled ≈ forced-zero (router knows to
    disable LoRA on Shakespeare)."""
    out = []
    n_prompts = 20
    prompt_len = 60
    cont_len = 60
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ce_router_controlled = []
        ce_forced_zero = []
        ce_forced_one = []
        for _ in range(n_prompts):
            start = int(rng.integers(0, len(sh_val_tokens) - prompt_len - cont_len))
            ids = torch.tensor([sh_val_tokens[start:start + prompt_len + cont_len].tolist()],
                                 device=device)
            x = ids[:, :-1]; y = ids[:, 1:]
            # Router-controlled
            with torch.no_grad():
                act = get_router_activation(model, router, x).item()
                logits, _ = model(x, lora_scale=act)
                ce_router_controlled.append(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
                logits, _ = model(x, lora_scale=0.0)
                ce_forced_zero.append(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
                logits, _ = model(x, lora_scale=1.0)
                ce_forced_one.append(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
        out.append({
            "seed": seed,
            "shakespeare_ce_router": float(np.mean(ce_router_controlled)),
            "shakespeare_ce_lora_off": float(np.mean(ce_forced_zero)),
            "shakespeare_ce_lora_on": float(np.mean(ce_forced_one)),
        })
    return out


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Loading Phase 1+2+3 system...")
    model, router, cfg = load_full_system(device)
    print(f"  loaded; total params {model.total_params():,}")

    queries = json.loads((DATA_DIR / "queries.json").read_text())["queries"]
    print(f"Loaded {len(queries)} queries")
    sh_val = torch.load(DATA_DIR / "shakespeare_val.pt", weights_only=False)

    all_results = []
    t0 = time.time()
    for condition in ["A_full", "B_always_on", "C_no_lora", "D_random"]:
        for seed in [0, 1, 2]:
            r = evaluate_condition(model, router, queries, tokenizer,
                                     condition, seed, device)
            print(f"  {condition:>14s}  seed={seed}  recall_real={r['recall_real']:.3f}  "
                  f"recall_raw={r['recall_raw']:.3f}  "
                  f"act_mean={r['activation_mean']:.3f}±{r['activation_std']:.3f}")
            all_results.append({
                "condition": condition, "seed": seed,
                "recall_real": r["recall_real"], "recall_raw": r["recall_raw"],
                "activation_mean": r["activation_mean"],
                "activation_std": r["activation_std"],
            })
            (RESULT_DIR / f"eval_{condition}_seed{seed}.json").write_text(
                json.dumps(r, indent=2))

    # Shakespeare quality
    print("\nShakespeare quality eval...")
    sh_results = evaluate_shakespeare_quality(model, router, tokenizer, sh_val, device)
    for r in sh_results:
        print(f"  seed={r['seed']}: CE_router={r['shakespeare_ce_router']:.3f}  "
              f"CE_lora_off={r['shakespeare_ce_lora_off']:.3f}  "
              f"CE_lora_on={r['shakespeare_ce_lora_on']:.3f}")

    # Aggregate
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    summary = {}
    for c in ["A_full", "B_always_on", "C_no_lora", "D_random"]:
        rs = [x for x in all_results if x["condition"] == c]
        recalls_real = [x["recall_real"] for x in rs]
        recalls_raw = [x["recall_raw"] for x in rs]
        acts = [x["activation_mean"] for x in rs]
        m_real = np.mean(recalls_real)
        s_real = np.std(recalls_real)
        m_raw = np.mean(recalls_raw)
        m_act = np.mean(acts)
        summary[c] = {
            "recall_real_mean": float(m_real), "recall_real_std": float(s_real),
            "recall_raw_mean": float(m_raw),
            "activation_mean": float(m_act),
            "seeds_real": recalls_real, "seeds_raw": recalls_raw,
        }
        print(f"  {c:>14s}  real={m_real:.3f}±{s_real:.3f}  raw={m_raw:.3f}  "
              f"act_mean={m_act:.3f}")

    (RESULT_DIR / "evaluation_summary.json").write_text(json.dumps({
        "all_results": all_results, "summary": summary,
        "shakespeare_quality": sh_results,
        "total_wall_s": time.time() - t0,
    }, indent=2))
    print(f"\nTotal eval wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
