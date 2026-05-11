"""Phase 73 — Does the trained base use other frozen adapters better?

Phase 72c showed the base trained against frozen adapter A (synthetic
"QZ9K7M facility" passage) developed a 40x retrieval improvement over the
pristine base. The architectural question Phase 73 tests: does the trained
base also use OTHER frozen adapters (different passages, different content
types) better than the pristine base does?

If yes (≥3/4 held-out adapters PASS-strong): the base learned a general
adapter-using skill. Major architectural finding.
If no: the base learned to use adapter A specifically. Per-adapter result.
If partial (1-2 PASS-strong, or several PASS-weak with mean improvement):
  some general capability, not deployment-grade.

Procedure (no training of bases; only adapter pretraining + evaluation):

  Stage 1: Pretrain 4 fresh rank-8 adapters (B/C/D/E) on library passages,
    one per content type, using the same procedure that produced adapter A
    (rank 8, 30 steps, lr 3e-4). Save each. Validate pk_ce in 1.5-3.0
    "underperforming" regime; otherwise retrain with adjusted steps.

  Stage 2: 5x2 evaluation matrix.
    For each adapter X in {A, B, C, D, E} × each base Y in {pristine, trained}:
      - Attach X to Y, measure attached passkey CE + greedy retrieval pass/fail.
      - Detach X (lora_active=False), measure base-only per-position CE.
    Plus WikiText PPL per base.

Adapter A serves as positive control — Phase 72c reported pk_ce 0.05 with
trained base + adapter A. If this run reproduces that, the eval pipeline
is correct and the B/C/D/E numbers are trustworthy.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase73_generalization.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.phase63_softmax_baseline import StandardTransformer
from experiments.identity_ae.lora_wrapper import (
    LoRALayer, apply_lora, get_lora_state_dict, load_lora_state_dict,
)


SEED = 0
RANK = 8
ALPHA = 16
PRETRAIN_STEPS = 30
LR_LORA_PRETRAIN = 3e-4
GEN_TOKENS = 30
PHASE73_DIR = Path("results/identity_ae/phase73")
MODELS_DIR = Path("models")

LORA_TARGETS = [
    'blocks.4.attn.qkv', 'blocks.4.attn.out_proj',
    'blocks.4.mlp.fc1',  'blocks.4.mlp.fc2',
    'blocks.5.attn.qkv', 'blocks.5.attn.out_proj',
    'blocks.5.mlp.fc1',  'blocks.5.mlp.fc2',
]

# Adapter A: from Phase 72c. Synthetic passage.
ADAPTER_A = {
    "name": "A_synthetic_QZ9K7M",
    "type": "numeric_synthetic",
    "passage":  "The access code for the QZ9K7M facility is 47281639.",
    "prompt":   "The access code for the QZ9K7M facility is",
    "passkey":  "47281639",
    "adapter_path": "models/phase72c_frozen_adapter.pt",
    "is_pretrained": True,  # Already saved in Phase 72c
}

# Adapters B-E: library passages, one of each content type.
# Statement format matches A's training input.
ADAPTERS_BCDE = [
    {
        "name": "B_library_northern",
        "type": "numeric",
        "passage":  "The system access code for the northern facility is 10433218.",
        "prompt":   "The system access code for the northern facility is",
        "passkey":  "10433218",
        "adapter_path": "models/phase73_frozen_adapter_B.pt",
        "is_pretrained": False,
    },
    {
        "name": "C_library_voss",
        "type": "entity",
        "passage":  "Dr. Elara Voss made a breakthrough discovery on September 9, 2017.",
        "prompt":   "Dr. Elara Voss made a breakthrough discovery on",
        "passkey":  "September 9, 2017",
        "adapter_path": "models/phase73_frozen_adapter_C.pt",
        "is_pretrained": False,
    },
    {
        "name": "D_library_reactor",
        "type": "technical",
        "passage":  "The reactor operates at a critical threshold of 8937 kelvin.",
        "prompt":   "The reactor operates at a critical threshold of",
        "passkey":  "8937",
        "adapter_path": "models/phase73_frozen_adapter_D.pt",
        "is_pretrained": False,
    },
    {
        "name": "E_library_thornfield",
        "type": "fact",
        "passage":  "The Thornfield Protocol requires exactly 18 signatories.",
        "prompt":   "The Thornfield Protocol requires exactly",
        "passkey":  "18",
        "adapter_path": "models/phase73_frozen_adapter_E.pt",
        "is_pretrained": False,
    },
]

ALL_ADAPTERS = [ADAPTER_A] + ADAPTERS_BCDE


# ============================================================
# LoRA active-flag (matches Phase 72c)
# ============================================================

_orig_lora_forward = LoRALayer.forward


def _patched_lora_forward(self, x):
    base_out = self.base_layer(x)
    if not getattr(self, "active", True):
        return base_out
    lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
    return base_out + lora_out


def patch_lora_class():
    LoRALayer.forward = _patched_lora_forward


def set_lora_active(model, active: bool):
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.active = active


# ============================================================
# Helpers
# ============================================================

def load_pristine_base(device):
    ckpt = torch.load("results/identity_ae/phase63/best.pt",
                       map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StandardTransformer(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                                 cfg["n_layers"], cfg["d_ff"], cfg["max_seq_len"],
                                 cfg["dropout"], cfg["bias"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def load_trained_base(device):
    """Load the Phase 72c step-200 base. Phase 72c saved its state_dict AFTER
    apply_lora(), so LoRA-wrapped layer names contain '.base_layer.' suffixes.
    Strip those to fit the vanilla StandardTransformer naming."""
    ckpt = torch.load("models/phase72c_trained_base.pt",
                       map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StandardTransformer(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                                 cfg["n_layers"], cfg["d_ff"], cfg["max_seq_len"],
                                 cfg["dropout"], cfg["bias"]).to(device)
    sd = ckpt["model_state_dict"]
    # Remap '.base_layer.' -> '.' for any LoRA-wrapped keys
    remapped = {}
    for k, v in sd.items():
        if ".base_layer." in k:
            remapped[k.replace(".base_layer.", ".")] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped)
    return model, cfg


def per_token_ce(model, ids_t):
    out = model(ids_t[:, :-1])
    targets = ids_t[:, 1:]
    log_probs = F.log_softmax(out, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.squeeze(0).tolist()


def passkey_positions(tokenizer, full_text, passkey_text):
    """Find positions in the next-token-prediction loss list (length T-1)
    that correspond to the passkey tokens in full_text."""
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    for start in range(len(full_ids)):
        for end in range(start + 1, len(full_ids) + 1):
            if tokenizer.decode(full_ids[start:end]).strip() == passkey_text:
                return [i for i in range(start - 1, end - 1)]
    return []


def greedy_generate(model, prompt, tokenizer, device, n_tokens=GEN_TOKENS):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(n_tokens):
        idx = input_ids[:, -512:]
        out = model(idx)
        next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
    return tokenizer.decode(input_ids[0, len(ids):], skip_special_tokens=True)


def passkey_in_text(text, passkey):
    """Robust passkey containment check (handles whitespace and comma variants
    matching Phase 47's check_passkey logic)."""
    if passkey in text: return True
    clean_pk = passkey.replace(",", "").replace(" ", "")
    clean_text = text.replace(",", "").replace(" ", "")
    return clean_pk in clean_text


@torch.no_grad()
def measure_wikitext_ppl(model, val_batches, device):
    total_nll, total_tok = 0.0, 0
    for ids in val_batches:
        ids_t = ids.to(device)
        out = model(ids_t[:, :-1])
        nll = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                ids_t[:, 1:].reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_tok += ids_t[:, 1:].numel()
    return math.exp(total_nll / max(total_tok, 1))


def build_wikitext_val(tokenizer, seq_len=256, batch_size=4, n_max=16, seed=SEED):
    from datasets import load_dataset
    val_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    ids = []
    for t in val_raw["text"]:
        if not t.strip(): continue
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len):
        chunks.append(torch.tensor(ids[i:i + seq_len], dtype=torch.long))
    val_batches = []
    for i in range(0, min(len(chunks), n_max * batch_size) - batch_size, batch_size):
        val_batches.append(torch.stack(chunks[i:i + batch_size]))
    return val_batches


# ============================================================
# Stage 1: Pretrain or load each frozen adapter
# ============================================================

def pretrain_adapter(adapter_spec, device, tokenizer, n_steps=None):
    """Train rank-8 LoRA on this adapter's passage; save and return baseline metrics.
    Adapter A's cached blob (from Phase 72c) lacks a 'spec' field; we add it back."""
    path = Path(adapter_spec["adapter_path"])
    if path.exists():
        print(f"[A] {adapter_spec['name']}: cached, loading {path}")
        blob = torch.load(path, map_location=device, weights_only=False)
        if "spec" not in blob:
            # Phase 72c format: rebuild spec from top-level fields + adapter_spec
            blob["spec"] = adapter_spec
            # Also ensure passkey_positions present (recompute)
            blob["passkey_positions"] = passkey_positions(
                tokenizer, adapter_spec["passage"], adapter_spec["passkey"])
        return blob

    n_steps = n_steps if n_steps is not None else PRETRAIN_STEPS
    print(f"[A] {adapter_spec['name']}: pretraining (rank {RANK}, {n_steps} steps, lr {LR_LORA_PRETRAIN})")
    base, cfg = load_pristine_base(device)
    base.train()
    n_lora = apply_lora(base, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(base, True)
    for n, p in base.named_parameters():
        p.requires_grad_("lora_" in n)
    lora_params = [p for n, p in base.named_parameters() if "lora_" in n]
    optim = torch.optim.AdamW(lora_params, lr=LR_LORA_PRETRAIN, weight_decay=0.0)

    ids = tokenizer.encode(adapter_spec["passage"], add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for step in range(n_steps):
        out = base(ids_t[:, :-1])
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optim.zero_grad(); loss.backward(); optim.step()

    base.eval()
    pk_pos = passkey_positions(tokenizer, adapter_spec["passage"], adapter_spec["passkey"])
    with torch.no_grad():
        ce = per_token_ce(base, ids_t)
    pk_ce = sum(ce[i] for i in pk_pos) / len(pk_pos) if pk_pos else float("nan")
    gen = greedy_generate(base, adapter_spec["prompt"], tokenizer, device)
    retrieves = passkey_in_text(gen, adapter_spec["passkey"])

    blob = {
        "lora_state_dict": {k: v.cpu() for k, v in get_lora_state_dict(base).items()},
        "rank": RANK, "alpha": ALPHA, "targets": LORA_TARGETS,
        "spec": adapter_spec,
        "passkey_positions": pk_pos,
        "baseline_attached_passkey_ce": pk_ce,
        "baseline_attached_generation": gen,
        "baseline_attached_retrieval_pass": retrieves,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    print(f"    pk_ce={pk_ce:.3f}  retrieval={'PASS' if retrieves else 'FAIL'}  "
          f"gen={gen[:60]!r}")
    print(f"    saved -> {path}")
    return blob


# ============================================================
# Stage 2: Evaluate (adapter, base) combinations
# ============================================================

def evaluate_combination(base_model, base_label, adapter_blob, tokenizer, device):
    """Attach adapter_blob to base_model and measure attached/detached metrics.
    base_model is mutated (LoRA applied + weights loaded). Caller restores."""
    spec = adapter_blob["spec"]
    apply_lora(base_model, rank=adapter_blob["rank"], alpha=adapter_blob["alpha"],
                target_modules=adapter_blob["targets"])
    load_lora_state_dict(base_model, {k: v.to(device) for k, v in
                                        adapter_blob["lora_state_dict"].items()})
    set_lora_active(base_model, True)
    base_model.eval()

    ids = tokenizer.encode(spec["passage"], add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    pk_pos = adapter_blob["passkey_positions"]

    with torch.no_grad():
        # Attached
        set_lora_active(base_model, True)
        ce_att = per_token_ce(base_model, ids_t)
        att_pk_ce = sum(ce_att[i] for i in pk_pos) / len(pk_pos)
        gen_att = greedy_generate(base_model, spec["prompt"], tokenizer, device)
        retrieves = passkey_in_text(gen_att, spec["passkey"])
        # Detached
        set_lora_active(base_model, False)
        ce_det = per_token_ce(base_model, ids_t)
        det_pk_ce = sum(ce_det[i] for i in pk_pos) / len(pk_pos)
        det_mean_ce = sum(ce_det) / len(ce_det)
        set_lora_active(base_model, True)

    return {
        "base": base_label, "adapter": spec["name"], "adapter_type": spec["type"],
        "attached_passkey_ce": att_pk_ce,
        "attached_generation": gen_att,
        "attached_retrieval_pass": retrieves,
        "detached_passkey_ce": det_pk_ce,
        "detached_mean_ce": det_mean_ce,
    }


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE73_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    patch_lora_class()

    # ---- Stage 1: pretrain B-E (A is already cached from Phase 72c) ----
    # Per-adapter step overrides: E's passkey ("18") is single-token so
    # 30 steps overfits to perfect retrieval; use 10 to keep partial regime.
    STEPS_OVERRIDE = {"E_library_thornfield": 10}

    print("=" * 72)
    print("STAGE 1 — pretrain frozen adapters")
    print("=" * 72)
    adapter_blobs = []
    for spec in ALL_ADAPTERS:
        n_steps = STEPS_OVERRIDE.get(spec["name"], PRETRAIN_STEPS)
        blob = pretrain_adapter(spec, device, tokenizer, n_steps=n_steps)
        adapter_blobs.append(blob)

    # ---- Validate pk_ce ranges ----
    print("\nadapter calibration:")
    for blob in adapter_blobs:
        spec = blob["spec"]
        pk = blob["baseline_attached_passkey_ce"]
        ret = blob["baseline_attached_retrieval_pass"]
        regime = "PERFECT" if pk < 0.5 else ("PARTIAL" if pk < 3.5 else "WEAK")
        print(f"  {spec['name']:30s}  pk_ce {pk:.3f}  retrieval={'PASS' if ret else 'FAIL'}  [{regime}]")

    # ---- Stage 2: 5x2 evaluation matrix ----
    print("\n" + "=" * 72)
    print("STAGE 2 — 5×2 evaluation matrix")
    print("=" * 72)
    val_batches = build_wikitext_val(tokenizer)
    results = []

    for base_label in ["pristine", "trained"]:
        # Load fresh base each time (LoRA application mutates)
        if base_label == "pristine":
            base, cfg = load_pristine_base(device)
        else:
            base, cfg = load_trained_base(device)
        base.eval()
        ppl = measure_wikitext_ppl(base, val_batches, device)
        print(f"\n--- base={base_label}  WikiText PPL: {ppl:.3f} ---")

        for blob in adapter_blobs:
            # Re-load fresh base for each adapter so LoRA application doesn't carry over
            if base_label == "pristine":
                base, _ = load_pristine_base(device)
            else:
                base, _ = load_trained_base(device)
            base.eval()
            r = evaluate_combination(base, base_label, blob, tokenizer, device)
            r["wikitext_ppl"] = ppl
            r["pretrain_baseline_passkey_ce"] = blob["baseline_attached_passkey_ce"]
            r["pretrain_baseline_retrieval"] = blob["baseline_attached_retrieval_pass"]
            results.append(r)
            print(f"  {r['adapter']:30s}  att_pk_ce {r['attached_passkey_ce']:.3f}  "
                  f"ret={'P' if r['attached_retrieval_pass'] else 'F'}  "
                  f"det_pk_ce {r['detached_passkey_ce']:.3f}  "
                  f"det_mean {r['detached_mean_ce']:.3f}")

    # ---- Compare pristine vs trained for each adapter ----
    print("\n" + "=" * 72)
    print("STAGE 3 — comparison & generalization classification")
    print("=" * 72)

    by_adapter = {}
    for r in results:
        by_adapter.setdefault(r["adapter"], {})[r["base"]] = r

    classifications = {}
    print(f"\n  {'adapter':30s}  {'pristine pk_ce':>14s}  {'trained pk_ce':>13s}  "
          f"{'Δ':>7s}  {'pristine ret':>12s}  {'trained ret':>11s}  classification")
    for adapter_name, by_base in by_adapter.items():
        p, t = by_base["pristine"], by_base["trained"]
        delta = t["attached_passkey_ce"] - p["attached_passkey_ce"]
        # Classification
        if t["attached_retrieval_pass"] and not p["attached_retrieval_pass"]:
            cls = "PASS-strong"
        elif delta < -1.0 and not t["attached_retrieval_pass"]:
            cls = "PASS-weak"
        elif abs(delta) <= 0.3:
            cls = "NEUTRAL"
        elif delta > 0.3:
            cls = "REGRESS"
        else:
            cls = "PASS-weak"  # delta in (-1.0, -0.3]
        classifications[adapter_name] = cls
        print(f"  {adapter_name:30s}  {p['attached_passkey_ce']:>14.3f}  "
              f"{t['attached_passkey_ce']:>13.3f}  {delta:>+7.3f}  "
              f"{'PASS' if p['attached_retrieval_pass'] else 'FAIL':>12s}  "
              f"{'PASS' if t['attached_retrieval_pass'] else 'FAIL':>11s}  {cls}")

    # ---- Headline summary ----
    held_out = [n for n in classifications if not n.startswith("A_")]
    n_strong = sum(1 for n in held_out if classifications[n] == "PASS-strong")
    n_weak = sum(1 for n in held_out if classifications[n] == "PASS-weak")
    n_neutral = sum(1 for n in held_out if classifications[n] == "NEUTRAL")
    n_regress = sum(1 for n in held_out if classifications[n] == "REGRESS")
    mean_delta_held = sum(by_adapter[n]["trained"]["attached_passkey_ce"] -
                            by_adapter[n]["pristine"]["attached_passkey_ce"]
                            for n in held_out) / len(held_out)

    print(f"\nHeld-out (B/C/D/E) generalization:")
    print(f"  PASS-strong: {n_strong}/4  PASS-weak: {n_weak}/4  "
          f"NEUTRAL: {n_neutral}/4  REGRESS: {n_regress}/4")
    print(f"  mean Δ pk_ce (trained - pristine): {mean_delta_held:+.3f}  "
          f"(negative = trained better)")

    if n_strong >= 3:
        verdict = "STRONG_GENERALIZATION"
    elif n_strong >= 1 or (n_weak >= 2 and n_regress == 0):
        verdict = "PARTIAL_GENERALIZATION"
    elif n_regress > n_strong + n_weak:
        verdict = "REGRESSION"
    else:
        verdict = "NO_GENERALIZATION"
    print(f"\n  VERDICT: {verdict}")

    # ---- Adapter A positive control check ----
    a_p = by_adapter["A_synthetic_QZ9K7M"]["pristine"]
    a_t = by_adapter["A_synthetic_QZ9K7M"]["trained"]
    print(f"\nAdapter A positive control:")
    print(f"  pristine pk_ce {a_p['attached_passkey_ce']:.3f}  "
          f"trained pk_ce {a_t['attached_passkey_ce']:.3f}  "
          f"Δ {a_t['attached_passkey_ce'] - a_p['attached_passkey_ce']:+.3f}")
    print(f"  Phase 72c reported: trained pk_ce 0.05 (Δ from pristine 2.14 → -2.09)")

    # ---- Save JSON ----
    out_path = PHASE73_DIR / "generalization.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "rank": RANK, "alpha": ALPHA,
                "pretrain_steps": PRETRAIN_STEPS, "pretrain_lr": LR_LORA_PRETRAIN,
                "lora_targets": LORA_TARGETS, "seed": SEED,
            },
            "adapters": [{
                "name": b["spec"]["name"], "type": b["spec"]["type"],
                "passage": b["spec"]["passage"], "passkey": b["spec"]["passkey"],
                "pretrain_passkey_ce": b["baseline_attached_passkey_ce"],
                "pretrain_retrieval_pass": b["baseline_attached_retrieval_pass"],
                "pretrain_generation": b["baseline_attached_generation"],
            } for b in adapter_blobs],
            "results": results,
            "classifications": classifications,
            "summary": {
                "held_out_pass_strong": n_strong,
                "held_out_pass_weak": n_weak,
                "held_out_neutral": n_neutral,
                "held_out_regress": n_regress,
                "mean_delta_held_out_pk_ce": mean_delta_held,
                "verdict": verdict,
                "adapter_A_pristine_pk_ce": a_p["attached_passkey_ce"],
                "adapter_A_trained_pk_ce": a_t["attached_passkey_ce"],
                "phase72c_reported_trained_pk_ce": 0.051,
            },
            "predictions": {
                "strong_generalization":  {"P": 0.25, "outcome": verdict == "STRONG_GENERALIZATION"},
                "partial_generalization": {"P": 0.40, "outcome": verdict == "PARTIAL_GENERALIZATION"},
                "no_generalization":      {"P": 0.30, "outcome": verdict == "NO_GENERALIZATION"},
                "regression":             {"P": 0.05, "outcome": verdict == "REGRESSION"},
            },
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot 5×2 matrix ----
    adapter_names = [b["spec"]["name"] for b in adapter_blobs]
    pristine_pk = [by_adapter[n]["pristine"]["attached_passkey_ce"] for n in adapter_names]
    trained_pk = [by_adapter[n]["trained"]["attached_passkey_ce"] for n in adapter_names]
    pristine_ret = [by_adapter[n]["pristine"]["attached_retrieval_pass"] for n in adapter_names]
    trained_ret = [by_adapter[n]["trained"]["attached_retrieval_pass"] for n in adapter_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = list(range(len(adapter_names)))
    w = 0.38
    pristine_color = ["C2" if r else "C3" for r in pristine_ret]
    trained_color  = ["C2" if r else "C3" for r in trained_ret]
    axes[0].bar([xi - w/2 for xi in x], pristine_pk, width=w, color="C0",
                  edgecolor=pristine_color, linewidth=2.5, label="pristine base")
    axes[0].bar([xi + w/2 for xi in x], trained_pk, width=w, color="C1",
                  edgecolor=trained_color, linewidth=2.5, label="trained base (Phase 72c step 200)")
    for xi, p, t in zip(x, pristine_pk, trained_pk):
        axes[0].annotate(f"{p:.2f}", (xi - w/2, p), ha="center", va="bottom", fontsize=7)
        axes[0].annotate(f"{t:.2f}", (xi + w/2, t), ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x); axes[0].set_xticklabels([n.split("_")[0] for n in adapter_names], fontsize=10)
    axes[0].set_ylabel("attached passkey CE (nats; lower = better retrieval)")
    axes[0].set_title("5×2 matrix: passkey CE per (adapter, base)\n"
                      f"green edge = greedy retrieval PASS, red edge = FAIL")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis="y")

    deltas = [t - p for p, t in zip(pristine_pk, trained_pk)]
    colors = ["C2" if d < -0.3 else ("C3" if d > 0.3 else "C7") for d in deltas]
    axes[1].bar(x, deltas, color=colors)
    for xi, d, n in zip(x, deltas, adapter_names):
        axes[1].annotate(f"{d:+.2f}\n[{classifications[n]}]", (xi, d),
                          ha="center", va="bottom" if d < 0 else "top", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([n.split("_")[0] for n in adapter_names], fontsize=10)
    axes[1].set_ylabel("Δ pk_ce (trained - pristine; negative = trained better)")
    axes[1].set_title(f"Generalization deltas\n"
                      f"verdict: {verdict}  (held-out: {n_strong}S/{n_weak}W/{n_neutral}N/{n_regress}R)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline(-1.0, color="C2", ls=":", lw=0.7, label="-1 nat (PASS-weak threshold)")
    axes[1].axhline(0.3, color="C3", ls=":", lw=0.7, label="±0.3 nat (NEUTRAL band)")
    axes[1].axhline(-0.3, color="C3", ls=":", lw=0.7)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = PHASE73_DIR / "generalization_table.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
