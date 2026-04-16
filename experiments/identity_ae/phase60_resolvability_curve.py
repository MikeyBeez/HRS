"""Phase 60: Resolvability curve — how many TTT steps make an engram functional?

Measures two signals as a function of training-step count for the same 5
passages drawn from stratified_tests():

  1. Same-prompt retrieval:  generate_greedy (50 tokens) → check_passkey.
     This is the "does the adapter work at all" signal.

  2. K-space cosine alignment at L5:  compute the L5 mean engram of the
     passage, inject it as a single hidden-state position via forward_from_x,
     capture K projections at L5 via install_qkv_hooks, compute
     per_head_cosine between the engram's K and the passage's mean K.
     This is the "does the engram address resolve in attention K-space" signal.

Step counts: [0, 15, 38, 75, 113, 150, 225, 300].

For each (passage, step_count) pair:
  - reset_lora_to_zero to clear any residue
  - train_adapter_multipara for exactly that many steps
  - measure retrieval and K-cosine alignment

LoRA: RANK=128, ALPHA=256 (from phase38b/phase30b, the rank-128 sweet spot).
Targets: L45_TARGETS.
paraphrase() is called to build prompts_with_answers for multi-para training,
exactly as phase 26 does — the adapter sees the passage + 4 Q/A pairs.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase60_resolvability_curve.py
"""

import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    load_model, check_passkey, generate_greedy,
)
from experiments.identity_ae.lora_wrapper import apply_lora
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests, hidden_at_layer,
)
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase32_kv_similarity import (
    install_qkv_hooks, remove_hooks, per_head_cosine,
)
from experiments.identity_ae.phase35_engram_after_ttt import forward_from_x


RANK = 128
ALPHA = 256
STEP_COUNTS = [0, 15, 38, 75, 113, 150, 225, 300]
N_PASSAGES = 5
ENGRAM_LAYER = 5      # L5 mean engram (matches the paper's main key source)
GEN_TOKENS = 50
D_MODEL = 1024


# ----------------------------------------------------------------
# K-space cosine alignment: engram vs passage mean at layer ENGRAM_LAYER.
# ----------------------------------------------------------------

@torch.no_grad()
def make_engram_vec(model, ids_t):
    """L5 mean-pooled hidden state of the passage tokens. Returns (1, 1, D)."""
    h = hidden_at_layer(model, ids_t, ENGRAM_LAYER)   # (1, T, D)
    return h.mean(dim=1, keepdim=True)                 # (1, 1, D)


@torch.no_grad()
def k_cosine_at_l5(model, ids_t, device):
    """Compute per-head cosine between mean(K_passage) and K_engram at L5.

    Steps:
      1. Full passage forward; hooks capture K at every layer; keep L5.
      2. Compute L5 mean engram from passage hidden states.
      3. Inject engram as single-position hidden input; hooks capture K at L5.
      4. Return per_head_cosine(mean_K_passage[L5], K_engram[L5]).
    """
    n_heads = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim

    # Pass 1: passage forward
    passage_store = {}
    handles = install_qkv_hooks(model, passage_store)
    with torch.no_grad():
        _ = model(ids_t, step=0)
    remove_hooks(handles)

    # Build mean engram from L5 hidden states (re-uses hidden_at_layer)
    x_eng = make_engram_vec(model, ids_t).to(device)  # (1, 1, D)

    # Pass 2: engram-as-prefix forward
    engram_store = {}
    handles = install_qkv_hooks(model, engram_store)
    with torch.no_grad():
        forward_from_x(model, x_eng)
    remove_hooks(handles)

    # L5 comparison
    K_p = passage_store[ENGRAM_LAYER]["k"].squeeze(0)    # (T, H, Dh)
    K_e = engram_store[ENGRAM_LAYER]["k"].squeeze(0).squeeze(0)  # (H, Dh)
    mean_K_p = K_p.mean(dim=0)                           # (H, Dh)

    return per_head_cosine(mean_K_p, K_e)


# ----------------------------------------------------------------
# Main experiment.
# ----------------------------------------------------------------

def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase60")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("PHASE 60: RESOLVABILITY CURVE")
    print("=" * 68)
    print(f"  RANK={RANK}  ALPHA={ALPHA}  D_MODEL={D_MODEL}")
    print(f"  Step counts: {STEP_COUNTS}")
    print(f"  Passages: {N_PASSAGES} (first 5 of stratified_tests)")
    print(f"  Engram layer: L{ENGRAM_LAYER}")
    print(f"  HIGH_LR={HIGH_LR:.1e}  BASE_LR={BASE_LR:.1e}")
    print()

    # Load model + apply LoRA once; re-use for all passages / step counts.
    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    print(f"  LoRA params: {n_lora:,} across {len(L45_TARGETS)} target modules")

    n_heads = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim
    print(f"  n_heads={n_heads}  head_dim={head_dim}  d_model={n_heads * head_dim}")
    print()

    # First 5 passages from stratified_tests (5 numeric, 5 entity, …; take [:5])
    all_tests = stratified_tests()
    tests = all_tests[:N_PASSAGES]
    print(f"  Selected passages: " + ", ".join(f"[{t['type']}]" for t in tests))
    print()

    # Pre-tokenise passages once.
    passage_ids = []
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        passage_ids.append(
            torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        )

    # ----------------------------------------------------------------
    # Experiment loop.
    # ----------------------------------------------------------------
    # results_per_passage[p][step_idx] = {retrieval, k_cosine}
    results_per_passage = []
    t0 = time.time()

    for pi, test in enumerate(tests):
        ids_t = passage_ids[pi]
        print(f"  Passage {pi+1}/{N_PASSAGES}: [{test['type']}]  passkey={test['passkey']!r}")

        # Build multi-para training inputs (same recipe as phase26)
        para_prompts = train_paraphrase(test)
        all_prompts = [test["prompt"]] + para_prompts            # 4 prompts
        prompts_with_answers = [f"{p} {test['passkey']}" for p in all_prompts]

        passage_results = []

        for n_steps in STEP_COUNTS:
            # Reset adapter to zero before every training run so each step
            # count is measured independently from the base model.
            reset_lora_to_zero(model)

            # Train for exactly n_steps (0 means zero-shot / base model).
            if n_steps > 0:
                train_adapter_multipara(
                    model, test["passage"], prompts_with_answers,
                    tokenizer, device,
                    n_steps=n_steps,
                    high_lr=HIGH_LR,
                    base_lr=BASE_LR,
                )

            model.eval()

            # --- Metric 1: same-prompt retrieval ---
            gen = generate_greedy(model, test["prompt"], tokenizer, device, GEN_TOKENS)
            retrieved = check_passkey(gen, test["passkey"])

            # --- Metric 2: K-space cosine alignment at L5 ---
            k_cos = k_cosine_at_l5(model, ids_t, device)

            passage_results.append({
                "n_steps": n_steps,
                "retrieval": int(retrieved),
                "k_cosine_l5": round(k_cos, 5),
                "gen_prefix": gen[:80],
            })

            print(f"    steps={n_steps:4d}  retrieval={int(retrieved)}  "
                  f"k_cos_L5={k_cos:.4f}  gen={gen[:60]!r}")

        results_per_passage.append({
            "passage_idx": pi,
            "type": test["type"],
            "passkey": test["passkey"],
            "prompt": test["prompt"],
            "curve": passage_results,
        })

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")

    # ----------------------------------------------------------------
    # Aggregate across passages.
    # ----------------------------------------------------------------
    # For each step count: mean retrieval rate and mean k_cosine.
    aggregated = []
    for si, n_steps in enumerate(STEP_COUNTS):
        retrieval_vals = [r["curve"][si]["retrieval"] for r in results_per_passage]
        k_cos_vals     = [r["curve"][si]["k_cosine_l5"] for r in results_per_passage]
        aggregated.append({
            "n_steps": n_steps,
            "retrieval_mean": round(sum(retrieval_vals) / len(retrieval_vals), 4),
            "retrieval_sum": sum(retrieval_vals),
            "k_cosine_mean": round(sum(k_cos_vals) / len(k_cos_vals), 5),
            "k_cosine_std": round(
                float(torch.tensor(k_cos_vals, dtype=torch.float).std().item()), 5
            ),
        })

    # ----------------------------------------------------------------
    # Summary table.
    # ----------------------------------------------------------------
    print()
    print("=" * 68)
    print("PHASE 60 SUMMARY — RESOLVABILITY CURVE")
    print("=" * 68)
    print(f"  {'steps':>6} | {'retrieval':>9} | {'k_cos_L5 mean':>14} | {'k_cos_L5 std':>13}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*14}-+-{'-'*13}")
    for row in aggregated:
        print(f"  {row['n_steps']:>6} | "
              f"{row['retrieval_sum']:>3d}/{N_PASSAGES}={row['retrieval_mean']:>5.0%} | "
              f"{row['k_cosine_mean']:>14.5f} | "
              f"{row['k_cosine_std']:>13.5f}")

    print()
    print("  Per-passage breakdown:")
    print(f"  {'passage':>7} {'type':>9} |  " + "  ".join(f"{s:>4}" for s in STEP_COUNTS))
    print(f"  {'-'*7} {'-'*9}-+--" + "--".join("-" * 4 for _ in STEP_COUNTS))
    for r in results_per_passage:
        retrievals = [str(c["retrieval"]) for c in r["curve"]]
        print(f"  {r['passage_idx']+1:>7} {r['type']:>9} |  " + "  ".join(f"{v:>4}" for v in retrievals))

    print()
    print("  K-cosine at L5 per passage:")
    print(f"  {'passage':>7} {'type':>9} |  " + "  ".join(f"{s:>6}" for s in STEP_COUNTS))
    print(f"  {'-'*7} {'-'*9}-+--" + "--".join("-" * 6 for _ in STEP_COUNTS))
    for r in results_per_passage:
        k_coss = [f"{c['k_cosine_l5']:.4f}" for c in r["curve"]]
        print(f"  {r['passage_idx']+1:>7} {r['type']:>9} |  " + "  ".join(f"{v:>6}" for v in k_coss))

    # ----------------------------------------------------------------
    # Save JSON.
    # ----------------------------------------------------------------
    out = {
        "config": {
            "rank": RANK,
            "alpha": ALPHA,
            "d_model": D_MODEL,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "engram_layer": ENGRAM_LAYER,
            "step_counts": STEP_COUNTS,
            "n_passages": N_PASSAGES,
            "high_lr": HIGH_LR,
            "base_lr": BASE_LR,
            "gen_tokens": GEN_TOKENS,
            "lora_targets": L45_TARGETS,
            "elapsed_s": round(elapsed, 1),
        },
        "aggregated": aggregated,
        "per_passage": results_per_passage,
    }
    out_path = results_dir / "resolvability_curve.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
