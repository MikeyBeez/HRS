"""Phase 61: Basin validation — does engram injection re-enter the passage trajectory?

The central claim of the engram-as-address framework is that the trained
adapter maps the compressed engram back onto the *same hidden-state
trajectory* the model would compute if it re-read the full passage.  If
true, then at every layer the last-token hidden state produced by
"engram + prompt" should cosine-align with the last-token hidden state
produced by "full passage + prompt".

This script tests that alignment under three conditions:

  (a) base — no LoRA adapter loaded for either pass
  (b) trained (unloaded) — adapter trained but zeroed before measurement;
      tests whether the engram vector itself (not the weights) carries the
      trajectory information after training
  (c) trained (loaded) — adapter trained AND loaded for both passes;
      this is the intended operating mode

Protocol (10 passages from stratified_tests()[:10]):
  For conditions (b) and (c): train_adapter_multipara for 150 steps,
      save the state dict.
  For each condition:
    Pass A: forward the full passage tokens through all blocks, collect
            last-token hidden state h_A[i] at each layer i.
    Pass B: compute L5 mean engram of the passage; prepend as a single
            hidden position before the embedded prompt tokens; forward
            through all blocks, collect last-token hidden state h_B[i].
    Metric: cosine(h_A[i], h_B[i]) for every layer i.

LoRA: RANK=128, ALPHA=256, D_MODEL=1024 (matches phase38b / phase60).
Targets: L45_TARGETS.
Results saved to results/identity_ae/phase61/.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase61_basin_validation.py
"""

import copy
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model, check_passkey, generate_greedy
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests, hidden_at_layer,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.lora_wrapper import apply_lora, get_lora_state_dict, load_lora_state_dict
from experiments.identity_ae.phase31_weighted_pool import cosine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANK = 128
ALPHA = 256
D_MODEL = 1024
ENGRAM_LAYER = 5       # L5 mean engram — the canonical key source
N_PASSAGES = 10
N_STEPS = 150
GEN_TOKENS = 50


# ---------------------------------------------------------------------------
# Manual block-by-block forward helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_collect_last(model, ids_t):
    """Full passage forward.  Returns dict {layer_idx: tensor(D)} of last-token
    hidden states at each layer, collected *after* each block.

    Args:
        model: the GPT-style model with .drop, .tok_emb, .blocks attributes.
        ids_t: LongTensor of shape (1, T).

    Returns:
        layer_hidden: dict[int -> cpu tensor (D,)]
    """
    h = model.drop(model.tok_emb(ids_t))          # (1, T, D)
    layer_hidden = {}
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        layer_hidden[i] = h[:, -1, :].detach().cpu().squeeze(0)   # (D,)
    return layer_hidden


@torch.no_grad()
def forward_engram_collect_last(model, passage_ids_t, prompt_ids_t):
    """Engram + prompt forward.

    1. Compute L5 mean engram of the passage (shape: (1, 1, D)).
    2. Embed prompt tokens (shape: (1, T2, D)).
    3. Concatenate engram as position-0 prefix: h = [eng | prompt_emb].
    4. Forward through all blocks; collect last-token hidden state per layer.

    Returns:
        layer_hidden: dict[int -> cpu tensor (D,)]
    """
    # Build the L5 mean engram: (1, D) -> unsqueeze -> (1, 1, D)
    eng = hidden_at_layer(model, passage_ids_t, ENGRAM_LAYER)  # (1, T, D)
    eng = eng.mean(dim=1, keepdim=True)                         # (1, 1, D)

    # Embed prompt tokens
    prompt_emb = model.drop(model.tok_emb(prompt_ids_t))        # (1, T2, D)

    # Concatenate: engram is position-0, prompt follows
    h = torch.cat([eng, prompt_emb], dim=1)                     # (1, 1+T2, D)

    layer_hidden = {}
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        layer_hidden[i] = h[:, -1, :].detach().cpu().squeeze(0)   # (D,)
    return layer_hidden


# ---------------------------------------------------------------------------
# Per-condition measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_cosine_per_layer(model, passage_ids_t, prompt_ids_t):
    """Compute per-layer cosine between Pass A and Pass B last-token hidden states.

    Returns list of floats, one per layer.
    """
    lh_a = forward_collect_last(model, passage_ids_t)
    lh_b = forward_engram_collect_last(model, passage_ids_t, prompt_ids_t)
    n_layers = len(lh_a)
    return [float(cosine(lh_a[i], lh_b[i])) for i in range(n_layers)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase61")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 61: BASIN VALIDATION — ENGRAM INJECTION vs FULL PASSAGE TRAJECTORY")
    print("=" * 72)
    print(f"  RANK={RANK}  ALPHA={ALPHA}  D_MODEL={D_MODEL}")
    print(f"  N_PASSAGES={N_PASSAGES}  N_STEPS={N_STEPS}  ENGRAM_LAYER=L{ENGRAM_LAYER}")
    print(f"  HIGH_LR={HIGH_LR:.1e}  BASE_LR={BASE_LR:.1e}")
    print()

    # Load model + apply LoRA once; weights shared across all passages.
    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    n_layers = len(model.blocks)
    print(f"  LoRA params: {n_lora:,}  Layers: {n_layers}  Targets: {len(L45_TARGETS)}")
    print()

    # Select 10 passages from the stratified test pool.
    all_tests = stratified_tests()
    tests = all_tests[:N_PASSAGES]
    print("  Passages: " + ", ".join(f"[{t['type']}]" for t in tests))
    print()

    # Pre-tokenise passages and prompts.
    passage_ids = []
    prompt_ids = []
    for test in tests:
        p_ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        passage_ids.append(
            torch.tensor(p_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        )
        q_ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        prompt_ids.append(
            torch.tensor(q_ids, dtype=torch.long).unsqueeze(0).to(device)
        )

    # Storage: per_passage[pi] = {a, b, c} each a list-of-floats (per layer)
    per_passage = []

    t0 = time.time()

    for pi, test in enumerate(tests):
        ids_t = passage_ids[pi]
        q_ids_t = prompt_ids[pi]
        print(f"  Passage {pi+1}/{N_PASSAGES}: [{test['type']}]  passkey={test['passkey']!r}")

        # Build multi-paraphrase training inputs (same recipe as phase26/60).
        para_prompts = train_paraphrase(test)
        all_prompts = [test["prompt"]] + para_prompts
        prompts_with_answers = [f"{p} {test['passkey']}" for p in all_prompts]

        # ---- Train adapter (used for conditions b and c) ----
        reset_lora_to_zero(model)
        train_adapter_multipara(
            model, test["passage"], prompts_with_answers,
            tokenizer, device,
            n_steps=N_STEPS,
            high_lr=HIGH_LR,
            base_lr=BASE_LR,
        )
        model.eval()
        trained_state = get_lora_state_dict(model)

        # ---- Condition (a): base model, no adapter ----
        reset_lora_to_zero(model)
        cos_a = measure_cosine_per_layer(model, ids_t, q_ids_t)
        print(f"    (a) base:              avg={sum(cos_a)/len(cos_a):.4f}  "
              f"L0={cos_a[0]:.4f}  L{n_layers-1}={cos_a[-1]:.4f}")

        # ---- Condition (b): adapter trained, then zeroed (engram only) ----
        # Load trained weights first so the *model's forward* is the trained
        # one when we compute the L5 engram, but then zero lora_B so the
        # adapter has no effect during the block-by-block passes.
        load_lora_state_dict(model, trained_state)
        reset_lora_to_zero(model)   # zero weights — measures engram vector alone
        cos_b = measure_cosine_per_layer(model, ids_t, q_ids_t)
        print(f"    (b) trained+zeroed:    avg={sum(cos_b)/len(cos_b):.4f}  "
              f"L0={cos_b[0]:.4f}  L{n_layers-1}={cos_b[-1]:.4f}")

        # ---- Condition (c): adapter trained and loaded ----
        load_lora_state_dict(model, trained_state)
        model.eval()
        cos_c = measure_cosine_per_layer(model, ids_t, q_ids_t)
        print(f"    (c) trained+loaded:    avg={sum(cos_c)/len(cos_c):.4f}  "
              f"L0={cos_c[0]:.4f}  L{n_layers-1}={cos_c[-1]:.4f}")

        per_passage.append({
            "passage_idx": pi,
            "type": test["type"],
            "passkey": test["passkey"],
            "prompt": test["prompt"],
            "cos_a": [round(v, 6) for v in cos_a],
            "cos_b": [round(v, 6) for v in cos_b],
            "cos_c": [round(v, 6) for v in cos_c],
        })

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")

    # -------------------------------------------------------------------------
    # Aggregate: mean per-layer cosine across passages for each condition.
    # -------------------------------------------------------------------------
    def mean_per_layer(key):
        return [
            round(sum(r[key][li] for r in per_passage) / N_PASSAGES, 6)
            for li in range(n_layers)
        ]

    avg_a = mean_per_layer("cos_a")
    avg_b = mean_per_layer("cos_b")
    avg_c = mean_per_layer("cos_c")

    # -------------------------------------------------------------------------
    # Print summary table.
    # -------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("PHASE 61 SUMMARY — PER-LAYER COSINE (mean over passages)")
    print("=" * 72)
    print(f"  {'layer':>5} | {'(a) base':>10} | {'(b) trained+zeroed':>18} | {'(c) trained+loaded':>18}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*18}-+-{'-'*18}")
    for li in range(n_layers):
        print(f"  {li:>5} | {avg_a[li]:>10.5f} | {avg_b[li]:>18.5f} | {avg_c[li]:>18.5f}")

    print()
    print(f"  Mean across all layers:")
    print(f"    (a) base:           {sum(avg_a)/n_layers:.5f}")
    print(f"    (b) trained+zeroed: {sum(avg_b)/n_layers:.5f}")
    print(f"    (c) trained+loaded: {sum(avg_c)/n_layers:.5f}")

    # Per-passage summary.
    print()
    print("  Per-passage mean cosine (all layers):")
    print(f"  {'pi':>3} {'type':>10} | {'(a)':>8} {'(b)':>8} {'(c)':>8}")
    print(f"  {'-'*3} {'-'*10}-+-{'-'*8} {'-'*8} {'-'*8}")
    for r in per_passage:
        ma = sum(r["cos_a"]) / n_layers
        mb = sum(r["cos_b"]) / n_layers
        mc = sum(r["cos_c"]) / n_layers
        print(f"  {r['passage_idx']+1:>3} {r['type']:>10} | {ma:>8.5f} {mb:>8.5f} {mc:>8.5f}")

    # -------------------------------------------------------------------------
    # Save JSON.
    # -------------------------------------------------------------------------
    out = {
        "config": {
            "rank": RANK,
            "alpha": ALPHA,
            "d_model": D_MODEL,
            "n_layers": n_layers,
            "n_passages": N_PASSAGES,
            "n_steps": N_STEPS,
            "engram_layer": ENGRAM_LAYER,
            "high_lr": HIGH_LR,
            "base_lr": BASE_LR,
            "gen_tokens": GEN_TOKENS,
            "lora_targets": L45_TARGETS,
            "elapsed_s": round(elapsed, 1),
        },
        "aggregated": {
            "avg_cos_a": avg_a,
            "avg_cos_b": avg_b,
            "avg_cos_c": avg_c,
            "grand_mean_a": round(sum(avg_a) / n_layers, 6),
            "grand_mean_b": round(sum(avg_b) / n_layers, 6),
            "grand_mean_c": round(sum(avg_c) / n_layers, 6),
        },
        "per_passage": per_passage,
    }
    out_path = results_dir / "basin_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
