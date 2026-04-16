"""Phase 39: Does the identity-AE gate add value over cosine threshold alone?

Background. The paper §3.1 / §3.5 / §3.6 describes a two-stage router for
the per-passage adapter library: stage 1 is an identity-autoencoder gate
that decides *whether* memory is needed at all (in-distribution queries
bypass the library), and stage 2 is engram cosine lookup that decides
*which* adapter to load. The gate is loaded from the checkpoint trained
in Phase 0; the dual-gate code lives in `dual_gate.py` and was last
exercised in Phase 16. Every per-passage library experiment from Phase
21 onward uses the engram lookup *without* the gate, on benchmarks where
every query is by construction off-manifold (the passkey set), so the
gate would fire on all of them and add no signal.

This script measures the gate empirically, against three query streams:

  1. in_distribution    50 short WikiText validation sentences
                        (the model can answer these from pretraining;
                        no adapter should be loaded)
  2. same_prompt        20 absorbed passkey prompts
                        (off-manifold, the correct adapter should fire)
  3. held_out           60 held-out paraphrases (Phase 27 set)
                        (off-manifold, generalization test)

For each query we record two signals:
  - **gate signal**:  mean per-token reconstruction error of the layer-3
                      hidden states through the base IAE
                      (loaded from results/identity_ae/phase0/...)
  - **routing signal**: max cosine similarity of the L5 nonstop_mean
                        engram against any stored adapter key

We then ask three questions:

  Q1. Does the gate cleanly separate in-distribution queries from
      off-manifold queries?  (Does the recon-error histogram show
      a gap between the two distributions?)

  Q2. Does the routing signal alone do the same job?  (Does the
      max-cosine histogram show a gap?)

  Q3. Does combining them help?  (Joint distribution, ROC analysis,
      operating-point comparison.)

If Q2 alone is sufficient — i.e., a single cosine threshold cleanly
separates in-distribution from absorbed queries — then the gate is
removable from the deployed architecture and the paper can be
simplified to a one-stage cosine-threshold router. If the gate's
recon error adds separating power that cosine doesn't have, then the
two-stage architecture is justified and we report the operating points.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase39_gate_value.py
"""

import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import (
    make_key_weighted, cosine,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict,
)

from identity_autoencoder import IdentityAutoencoder


RANK = 128
ALPHA = RANK * 2
N_STEPS = 150
GATE_LAYER = 3            # The IAE was trained on layer-3 hidden states (Phase 16)
KEY_LAYER  = 5            # Engram routing key layer (rest of paper)
STRATEGY   = "nonstop_mean"
N_WIKITEXT = 50
WIKITEXT_LEN = 32         # short sentences, comparable in length to passkey prompts

GATE_CKPT = "results/identity_ae/phase0/autoencoder_init_20ep.pt"


@torch.no_grad()
def gate_recon_error(gate, model, ids_t):
    """Mean per-token reconstruction error of layer-3 hidden states
    through the base identity AE."""
    h = hidden_at_layer(model, ids_t, GATE_LAYER)   # (1, T, D)
    _, error = gate(h)                               # error: (1, T)
    return float(error.mean().item())


@torch.no_grad()
def lookup_max_cosine(model, tokenizer, ids_t, library_keys):
    """Max cosine similarity over all library keys (any entry, any key)."""
    q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
    best_a, best_score = -1, -2.0
    for ai, keys in enumerate(library_keys):
        for k in keys:
            s = cosine(q, k)
            if s > best_score:
                best_score = s
                best_a = ai
    return best_a, best_score


# ----------------------------------------------------------------
# Sampling helpers.
# ----------------------------------------------------------------
def sample_wikitext_queries(val_ds, tokenizer, n, target_len):
    """Sample n short token windows from WikiText validation, decode to text,
    return list of text strings."""
    indices = torch.randperm(len(val_ds))[:n * 4].tolist()  # oversample
    queries = []
    for idx in indices:
        item = val_ds[idx]
        ids = item[0] if isinstance(item, tuple) else item
        if len(ids) < target_len:
            continue
        # Take a random window of target_len tokens
        start = random.randint(0, len(ids) - target_len)
        window = ids[start:start + target_len].tolist()
        text = tokenizer.decode(window, skip_special_tokens=True)
        queries.append(text)
        if len(queries) >= n:
            break
    return queries


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase39")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}  (rank {RANK}, alpha {ALPHA})")
    print(f"Engram source: L{KEY_LAYER}_{STRATEGY}")
    print(f"Gate source:   L{GATE_LAYER} hidden states → base IAE\n")

    # ----- Load base IAE gate -----
    gate = IdentityAutoencoder(d_model=1024, hidden_dim=768, bottleneck_dim=256)
    gate.load_state_dict(torch.load(GATE_CKPT, weights_only=True))
    gate.to(device).eval()
    print(f"Loaded base IAE gate from {GATE_CKPT}")
    print(f"  param count: {gate.param_count():,}\n")

    # ----- Build the library (Phase 38b protocol: rank 128, multi-prompt) -----
    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}\n")

    print("=" * 60)
    print("ABSORPTION PHASE")
    print("=" * 60)
    library = []
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library.append({"sd": sd, "test": dict(test), "train_prompts": train_prompts})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}")

    # ----- Build library keys -----
    reset_lora_to_zero(model)
    library_keys = []
    for entry in library:
        keys_for_entry = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys_for_entry.append(make_key_weighted(model, tokenizer, ids_t, STRATEGY))
        library_keys.append(keys_for_entry)
    print(f"\nLibrary built: {len(library)} adapters, "
          f"{sum(len(k) for k in library_keys)} keys total\n")

    # ----- Build query streams -----
    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]

    wikitext_queries = sample_wikitext_queries(
        val_ds, tokenizer, N_WIKITEXT, WIKITEXT_LEN)
    print(f"In-distribution queries: {len(wikitext_queries)} short WikiText windows")
    print(f"  example: {wikitext_queries[0][:80]!r}\n")

    same_prompt_queries = [t["prompt"] for t in tests]
    held_out_queries = []
    for t in tests:
        held_out_queries.extend(held_out_paraphrase(t))
    print(f"Same-prompt queries:    {len(same_prompt_queries)}")
    print(f"Held-out queries:       {len(held_out_queries)}\n")

    # ----- Score every query under both signals -----
    def score_queries(queries, label):
        out = []
        for q in queries:
            ids = tokenizer.encode(q, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            recon = gate_recon_error(gate, model, ids_t)
            best_a, best_sim = lookup_max_cosine(model, tokenizer, ids_t, library_keys)
            out.append({
                "label": label, "text": q[:80],
                "recon_error": recon, "max_cosine": best_sim, "argmax": best_a,
            })
        return out

    print("=" * 60)
    print("SCORING QUERIES")
    print("=" * 60)
    scores_id = score_queries(wikitext_queries,    "in_distribution")
    scores_sp = score_queries(same_prompt_queries, "same_prompt")
    scores_ho = score_queries(held_out_queries,    "held_out")

    # ----- Aggregate stats per stream -----
    def stats(rows, key):
        vals = [r[key] for r in rows]
        vals.sort()
        n = len(vals)
        return {
            "n":    n,
            "mean": sum(vals) / n,
            "min":  vals[0],
            "p10":  vals[n // 10],
            "p50":  vals[n // 2],
            "p90":  vals[(9 * n) // 10],
            "max":  vals[-1],
        }

    print(f"\n{'='*72}")
    print("DISTRIBUTIONS BY QUERY TYPE")
    print(f"{'='*72}")
    print(f"\n  Recon error (gate signal — lower = more in-distribution):")
    print(f"  {'stream':18s} {'n':>4} {'min':>9} {'p10':>9} {'p50':>9} {'p90':>9} {'max':>9}")
    print(f"  {'-'*18} {'-'*4} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for label, rows in [("in_distribution", scores_id),
                        ("same_prompt",     scores_sp),
                        ("held_out",        scores_ho)]:
        s = stats(rows, "recon_error")
        print(f"  {label:18s} {s['n']:>4d} {s['min']:>9.4f} {s['p10']:>9.4f} "
              f"{s['p50']:>9.4f} {s['p90']:>9.4f} {s['max']:>9.4f}")

    print(f"\n  Max cosine (routing signal — higher = closer to a library key):")
    print(f"  {'stream':18s} {'n':>4} {'min':>9} {'p10':>9} {'p50':>9} {'p90':>9} {'max':>9}")
    print(f"  {'-'*18} {'-'*4} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for label, rows in [("in_distribution", scores_id),
                        ("same_prompt",     scores_sp),
                        ("held_out",        scores_ho)]:
        s = stats(rows, "max_cosine")
        print(f"  {label:18s} {s['n']:>4d} {s['min']:>9.4f} {s['p10']:>9.4f} "
              f"{s['p50']:>9.4f} {s['p90']:>9.4f} {s['max']:>9.4f}")

    # ----- Threshold sweep: pure cosine vs gate-only vs combined -----
    # We want to filter: in_distribution → "no library", absorbed → "load library"
    # Treat held_out as "absorbed" (positive class) since the user wants those routed.
    positives = scores_sp + scores_ho       # should be loaded
    negatives = scores_id                    # should not be loaded

    def evaluate_decision(rule):
        """rule: row -> bool (True = load library, False = bypass)"""
        tp = sum(1 for r in positives if rule(r))
        fn = len(positives) - tp
        fp = sum(1 for r in negatives if rule(r))
        tn = len(negatives) - fp
        return tp, fn, fp, tn

    def fmt_cm(label, tp, fn, fp, tn):
        n_pos = tp + fn
        n_neg = fp + tn
        recall = tp / n_pos if n_pos else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        spec = tn / n_neg if n_neg else 0
        return (f"  {label:38s} TP={tp:>3} FN={fn:>3} FP={fp:>3} TN={tn:>3}  "
                f"recall={recall:.0%} precision={precision:.0%} specificity={spec:.0%}")

    print(f"\n{'='*72}")
    print("DECISION RULE COMPARISON")
    print(f"{'='*72}")
    print(f"  positives (should load library): "
          f"{len(positives)} (same_prompt {len(scores_sp)} + held_out {len(scores_ho)})")
    print(f"  negatives (should NOT load):     {len(negatives)} (in_distribution)\n")

    # Sweep cosine thresholds
    print("  ── Cosine threshold only (no gate) ──")
    for tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        tp, fn, fp, tn = evaluate_decision(lambda r, tau=tau: r["max_cosine"] >= tau)
        print(fmt_cm(f"cosine >= {tau:.2f}", tp, fn, fp, tn))

    # Sweep gate thresholds
    print("\n  ── Gate (recon error) threshold only ──")
    for theta in [0.10, 0.15, 0.20, 0.241, 0.30, 0.40]:
        tp, fn, fp, tn = evaluate_decision(lambda r, theta=theta: r["recon_error"] >= theta)
        print(fmt_cm(f"recon_error >= {theta:.3f}", tp, fn, fp, tn))

    # Combined: gate fires AND cosine match
    print("\n  ── Combined: gate fires AND cosine above threshold ──")
    for theta, tau in [(0.241, 0.50), (0.241, 0.70), (0.241, 0.80),
                        (0.30, 0.70), (0.30, 0.80)]:
        tp, fn, fp, tn = evaluate_decision(
            lambda r, theta=theta, tau=tau: r["recon_error"] >= theta and r["max_cosine"] >= tau)
        print(fmt_cm(f"recon >= {theta:.2f} AND cos >= {tau:.2f}", tp, fn, fp, tn))

    # ----- Save raw rows -----
    out = {
        "rank": RANK,
        "n_steps": N_STEPS,
        "n_wikitext": N_WIKITEXT,
        "wikitext_len": WIKITEXT_LEN,
        "gate_ckpt": GATE_CKPT,
        "scores": {
            "in_distribution": scores_id,
            "same_prompt":     scores_sp,
            "held_out":        scores_ho,
        },
    }
    with open(results_dir / "gate_value.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
