"""Phase 35: Engram-as-context after TTT (centroid theory test).

The centroid view of context says: an engram is a pointer into the
representation manifold; whether the pointer "works" depends on whether
the model has learned the territory it points to. This script tests
that claim directly. We compute the L5 mean-pooled engram of a passage
and use it as a single-token prefix in front of the query prompt — no
passage tokens in the context. We then compare retrieval rates across:

  B0 no_context             prompt only, base model         (lower bound)
  B1 full_context           passage + prompt, base model    (upper bound)
  B2 lora_ttt_only          prompt only, LoRA adapter loaded (paper's main)
  A  engram_before_ttt      engram_prefix + prompt, base model
  B  engram_after_full_ttt  engram_prefix + prompt, after full-model TTT,
                            engram re-extracted under the TTT'd model
  C  engram_after_lora_ttt  engram_prefix + prompt, after LoRA TTT,
                            engram re-extracted under the loaded adapter

If condition A fails and B/C succeed, the claim is validated empirically:
TTT made the engram functional. The model learned the territory, so the
pointer now resolves.

Injection method: prepend the engram as a single hidden-state vector at
position 0 of the input embedding. The model uses RoPE inside attention
and has no input positional table, so this is a clean substitution.

Full-model TTT is destructive — we save the base state_dict on CPU and
restore it between passages. LoRA TTT is non-destructive (zero the
adapter between passages).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase35_engram_after_ttt.py
"""

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy, run_ttt_full,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, RANK, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase21_per_passage_adapters import train_passage_adapter
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


LAYER = 5            # final-layer engram (matches paper's L5_mean)
N_LORA_STEPS = 150   # phase 24 winner
N_FULL_STEPS = 100   # full-model TTT (matches phase 10 lora_100 / full_100)
FULL_LR = 1e-5       # phase 10 full-TTT learning rate
GEN_TOKENS = 50


# ----------------------------------------------------------------
# Engram extraction (L5 mean) — same as paper's main key source.
# ----------------------------------------------------------------
@torch.no_grad()
def make_engram(model, ids_t):
    h = hidden_at_layer(model, ids_t, LAYER)  # (1, T, D)
    return h.mean(dim=1).squeeze(0).detach()   # (D,) on device


# ----------------------------------------------------------------
# Forward from a precomputed hidden-state input (no token embedding).
# Mirrors model.forward, but skips internal engram-injector / engram-
# extractor machinery — same simplification as phase22.hidden_at_layer.
# ----------------------------------------------------------------
@torch.no_grad()
def forward_from_x(model, x):
    """Run x through all blocks → ln_f → lm_head. Returns logits."""
    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        x, _, _, _ = block(x, step=0, engram_buffer=eb)
    x = model.ln_f(x)
    return model.lm_head(x)


@torch.no_grad()
def generate_with_engram_prefix(model, engram, prompt, tokenizer, device,
                                  n_tokens=GEN_TOKENS):
    """Greedy decode with a single engram vector prepended at position 0.

    The engram lives in the model's hidden-state space (D = d_model). At
    each step we build x = [engram, tok_emb(prompt+generated)] and run
    one forward pass. Slow but correct (no KV cache reuse).
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()

    eng_x = engram.to(device).view(1, 1, -1)  # (1, 1, D)
    max_ctx = 512  # match RoPE max_seq_len in v22

    n_prompt = input_ids.shape[1]
    for _ in range(n_tokens):
        tok_x = model.drop(model.tok_emb(input_ids))            # (1, T, D)
        x = torch.cat([eng_x, tok_x], dim=1)                     # (1, 1+T, D)
        if x.shape[1] > max_ctx:
            # keep engram at position 0, drop oldest tokens
            x = torch.cat([eng_x, x[:, -(max_ctx - 1):, :]], dim=1)
        logits = forward_from_x(model, x)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0, n_prompt:], skip_special_tokens=True)


def per_type_zeros():
    return {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase35")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"Engram source: L{LAYER}_mean (hidden-state prefix injection)\n")

    # Snapshot of pristine model state (LoRA applied + zeroed) on CPU.
    # We'll restore this between passages for the destructive full-TTT path.
    reset_lora_to_zero(model)
    base_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"Saved base state ({len(base_state_cpu)} tensors) to CPU\n")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}\n")

    # Drift baseline (sanity)
    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL (zero adapter): {baseline_ppl:.3f}\n")

    # ============================================================
    # Per-passage trial. Six conditions per passage.
    # ============================================================
    counters = {k: 0 for k in [
        "B0_no_context", "B1_full_context", "B2_lora_ttt_only",
        "A_engram_before_ttt", "B_engram_after_full_ttt", "C_engram_after_lora_ttt",
    ]}
    per_type = {k: per_type_zeros() for k in counters}
    rows = []

    t0 = time.time()
    for ti, test in enumerate(tests):
        passage = test["passage"]
        prompt  = test["prompt"]
        passkey = test["passkey"]
        ptype   = test["type"]

        # Restore pristine base
        model.load_state_dict(base_state_cpu)
        model.to(device)

        # ---------- B0: no context, base model ----------
        gen_b0 = generate_greedy(model, prompt, tokenizer, device, GEN_TOKENS)
        b0_hit = check_passkey(gen_b0, passkey)

        # ---------- B1: full context, base model ----------
        gen_b1 = generate_greedy(model, passage + " " + prompt, tokenizer, device, GEN_TOKENS)
        b1_hit = check_passkey(gen_b1, passkey)

        # ---------- A: engram (base) + prompt ----------
        ids = tokenizer.encode(passage, add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        eng_base = make_engram(model, ids_t)
        gen_a = generate_with_engram_prefix(model, eng_base, prompt, tokenizer, device)
        a_hit = check_passkey(gen_a, passkey)

        # ---------- C path: LoRA TTT, then engram + prompt and prompt-only ----------
        # (Do this BEFORE full-TTT, since LoRA leaves base weights pristine.)
        reset_lora_to_zero(model)
        train_passage_adapter(model, passage, tokenizer, device,
                              n_steps=N_LORA_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)

        # B2: prompt-only with adapter loaded (the paper's main retrieval condition)
        gen_b2 = generate_greedy(model, prompt, tokenizer, device, GEN_TOKENS)
        b2_hit = check_passkey(gen_b2, passkey)

        # C: engram re-extracted under loaded adapter, then engram + prompt
        eng_lora = make_engram(model, ids_t)
        gen_c = generate_with_engram_prefix(model, eng_lora, prompt, tokenizer, device)
        c_hit = check_passkey(gen_c, passkey)

        reset_lora_to_zero(model)

        # ---------- B path: full-model TTT, engram re-extracted, engram + prompt ----------
        # Destructive — we'll restore base_state at the next loop iteration.
        # Enable grad on all params (load_model leaves them in default state)
        for p in model.parameters():
            p.requires_grad = True
        run_ttt_full(model, passage, tokenizer, device, N_FULL_STEPS, FULL_LR)

        eng_full = make_engram(model, ids_t)
        gen_b = generate_with_engram_prefix(model, eng_full, prompt, tokenizer, device)
        b_hit = check_passkey(gen_b, passkey)

        # Tally
        record = {
            "B0_no_context":          b0_hit,
            "B1_full_context":        b1_hit,
            "B2_lora_ttt_only":       b2_hit,
            "A_engram_before_ttt":    a_hit,
            "B_engram_after_full_ttt": b_hit,
            "C_engram_after_lora_ttt": c_hit,
        }
        for k, v in record.items():
            if v:
                counters[k] += 1
                per_type[k][ptype] += 1

        rows.append({
            "id": test["id"], "type": ptype, "passkey": passkey,
            **record,
            "gen_A": gen_a[:120],
            "gen_B": gen_b[:120],
            "gen_C": gen_c[:120],
        })

        if (ti + 1) % 5 == 0 or ti == len(tests) - 1:
            elapsed = time.time() - t0
            print(f"  [{ti+1:2d}/20] {ptype:9s}  "
                  f"B0={counters['B0_no_context']:2d} "
                  f"B1={counters['B1_full_context']:2d} "
                  f"B2={counters['B2_lora_ttt_only']:2d} "
                  f"A={counters['A_engram_before_ttt']:2d} "
                  f"B={counters['B_engram_after_full_ttt']:2d} "
                  f"C={counters['C_engram_after_lora_ttt']:2d}  "
                  f"({elapsed:.0f}s)")

    # Restore once more, drift check
    model.load_state_dict(base_state_cpu)
    model.to(device)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*64}")
    print("PHASE 35 SUMMARY (engram-as-context after TTT)")
    print(f"{'='*64}")
    print(f"  {'Condition':28s} {'overall':>8} {'numeric':>8} {'entity':>8} {'tech':>8} {'fact':>8}")
    print(f"  {'-'*28}{'-'*44}")
    order = [
        ("B0 no_context",            "B0_no_context"),
        ("B1 full_context",          "B1_full_context"),
        ("B2 lora_ttt_only",         "B2_lora_ttt_only"),
        ("A  engram_before_ttt",     "A_engram_before_ttt"),
        ("B  engram_after_full_ttt", "B_engram_after_full_ttt"),
        ("C  engram_after_lora_ttt", "C_engram_after_lora_ttt"),
    ]
    for label, key in order:
        c = counters[key]
        pt = per_type[key]
        print(f"  {label:28s} {c:>3}/20  "
              f"{pt['numeric']:>3}/5  {pt['entity']:>3}/5  "
              f"{pt['technical']:>3}/5  {pt['fact']:>3}/5")

    print(f"\n  Val PPL drift after restore: {drift:+.3f}%")
    print(f"  (Centroid claim validated if A is small but B and/or C are large.)")

    summary = {
        "layer": LAYER,
        "n_lora_steps": N_LORA_STEPS,
        "n_full_steps": N_FULL_STEPS,
        "full_lr": FULL_LR,
        "drift_pct": drift,
        "counters": counters,
        "per_type": per_type,
        "rows": rows,
    }
    with open(results_dir / "engram_after_ttt.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
