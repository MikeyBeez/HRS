"""Task 11: OOD content decay through V18's layer stack.

For each of the 25 needles, extract the residual activation at layer 3
at the position of the needle's most distinctive token, then inject
that activation (norm-matched) at each of layers 0..5 of a neutral
prompt and measure the resulting logit lift on the target BPE token.

This is a read-only diagnostic. No training, no weight changes. All
state changes are via PyTorch forward hooks that are removed after
each measurement.

Outputs: `results/ood_decay/report.json` and `results/ood_decay/REPORT.md`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer

# Distinctive tokens per needle (first invented name from spot_check.py)
from experiments.hrs_loop.niah_benchmark.spot_check import DISTINCTIVE as DISTINCTIVE_ALL
from experiments.hrs_loop.niah_benchmark.eval_niah_v2 import load_needles

INV_SOFTPLUS_1 = math.log(math.e - 1.0)
L_EXTRACT = 3
N_LAYERS = 6  # V18 has 6 blocks
NEUTRAL_PROMPT = "The following is a description of a concept. The concept is called"
OUT_DIR = REPO / "results/ood_decay"


def load_v18(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load(str(REPO / "results/v18_cross_attn/best.pt"),
                      map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_scalar") and any(name == m for m in missing):
            with torch.no_grad():
                p.fill_(INV_SOFTPLUS_1)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V18. missing={len(missing)} unexpected={len(unexpected)}")
    return model, cfg


def pick_target_token(tokenizer, needle) -> tuple[str, int, str]:
    """Return (target_word, target_bpe_id, token_str).

    Uses DISTINCTIVE[nid][0] as the distinctive word. target_bpe_id is the
    first BPE token of " {word}" (GPT-2 tokenization with leading space).
    """
    nid = needle["id"]
    word = DISTINCTIVE_ALL[nid][0]
    # Tokenize with leading space (natural mid-sentence form)
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    first_id = ids[0]
    token_str = tokenizer.decode([first_id])
    return word, first_id, token_str


def find_target_position_in_fact(tokenizer, fact_text: str, target_word: str,
                                 target_bpe_id: int, max_len: int = 512) -> Optional[int]:
    """Return index of first occurrence of target_bpe_id in fact's tokenization.
    If not found, search for a re-tokenized version without leading space.
    """
    ids = tokenizer.encode(fact_text, add_special_tokens=False)[:max_len]
    if target_bpe_id in ids:
        return ids.index(target_bpe_id)
    # Fallback: try without leading space
    alt_ids = tokenizer.encode(target_word, add_special_tokens=False)
    if alt_ids and alt_ids[0] in ids:
        return ids.index(alt_ids[0])
    return None


@torch.no_grad()
def extract_activation(model, tokenizer, device, fact_text: str, target_pos: int,
                       L_extract: int = L_EXTRACT, max_len: int = 512) -> torch.Tensor:
    """Forward fact through V18; capture residual at block L_extract INPUT
    at position target_pos. Returns (D,) tensor (fp32)."""
    ids = tokenizer.encode(fact_text, add_special_tokens=False)[:max_len]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    capture = {}

    def pre_hook(module, args):
        h = args[0]
        capture["h"] = h.detach()[0, target_pos].float().clone()
        return None  # don't modify

    handle = model.blocks[L_extract].register_forward_pre_hook(pre_hook)
    try:
        _ = model(x, step=0)
    finally:
        handle.remove()
    return capture["h"]


@torch.no_grad()
def forward_neutral_logit(model, tokenizer, device, prompt: str,
                          target_bpe_id: int,
                          inject_vector: Optional[torch.Tensor] = None,
                          inject_layer: Optional[int] = None,
                          inject_pos: int = -1) -> float:
    """Forward the neutral prompt and return the logit for target_bpe_id at the
    final position.

    If inject_vector is not None, add it to the residual stream at the INPUT of
    block `inject_layer`, at position `inject_pos` (−1 = last token).
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    T = x.shape[1]
    pos = inject_pos if inject_pos >= 0 else (T + inject_pos)

    handle = None
    if inject_vector is not None and inject_layer is not None:
        vec = inject_vector.to(x.device)

        def pre_hook(module, args):
            h = args[0]
            # Clone to avoid in-place on cached activations
            h2 = h.clone()
            h2[0, pos] = h2[0, pos] + vec
            return (h2,) + args[1:]

        handle = model.blocks[inject_layer].register_forward_pre_hook(pre_hook)

    try:
        output = model(x, step=0)
    finally:
        if handle is not None:
            handle.remove()

    logits = output.logits if hasattr(output, "logits") else output[0]
    return float(logits[0, -1, target_bpe_id].item())


@torch.no_grad()
def residual_norm_at_layer(model, tokenizer, device, prompt: str,
                           layer: int, pos: int = -1) -> float:
    """Measure the L2 norm of the residual stream at block `layer` INPUT,
    at the given position (−1 = last)."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    T = x.shape[1]
    p = pos if pos >= 0 else (T + pos)
    captured = {}

    def pre_hook(module, args):
        h = args[0]
        captured["n"] = float(h[0, p].norm().item())
        return None

    handle = model.blocks[layer].register_forward_pre_hook(pre_hook)
    try:
        _ = model(x, step=0)
    finally:
        handle.remove()
    return captured["n"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_v18(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    needles = load_needles()
    print(f"Loaded {len(needles)} needles\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Sanity check 1: zero-injection equals no-injection
    # =========================================================
    print("=== Sanity check 1: zero-injection vs no-injection ===")
    # Use needle 0's target as the scoring target (any token works)
    w0, tid0, tstr0 = pick_target_token(tokenizer, needles[0])
    base_logit = forward_neutral_logit(model, tokenizer, device, NEUTRAL_PROMPT, tid0)
    zero_vec = torch.zeros(model.cfg.model.d_model, device=device)
    sanity_max_diff = 0.0
    for L in range(N_LAYERS):
        zero_logit = forward_neutral_logit(model, tokenizer, device, NEUTRAL_PROMPT,
                                            tid0, inject_vector=zero_vec, inject_layer=L)
        diff = abs(zero_logit - base_logit)
        sanity_max_diff = max(sanity_max_diff, diff)
        print(f"  L={L}: zero-logit={zero_logit:.6f} base={base_logit:.6f} diff={diff:.2e}")
    print(f"  Sanity: max abs diff = {sanity_max_diff:.2e} (expected ~0)")
    if sanity_max_diff > 1e-4:
        print(f"  *** FAIL: zero-injection does not reproduce baseline. Hook bug. Aborting. ***")
        sys.exit(1)
    print("  PASS\n")

    # =========================================================
    # Sanity check 2: home-position positive control (needle 0)
    # Extract at L3, inject at L3, same position -> lift must be > 0
    # =========================================================
    print("=== Sanity check 2: home-position positive control (needle 0) ===")
    n0 = needles[0]
    w0, tid0, tstr0 = pick_target_token(tokenizer, n0)
    pos0 = find_target_position_in_fact(tokenizer, n0["fact"], w0, tid0)
    if pos0 is None:
        print(f"  *** FAIL: needle 0 target token {tstr0!r} not found in fact. Aborting. ***")
        sys.exit(1)
    base_logit_n0 = forward_neutral_logit(model, tokenizer, device, NEUTRAL_PROMPT, tid0)
    a_t_n0 = extract_activation(model, tokenizer, device, n0["fact"], pos0, L_extract=L_EXTRACT)
    a_t_n0_norm = float(a_t_n0.norm().item())
    # Norm-match at L_EXTRACT
    tgt_norm_L3 = residual_norm_at_layer(model, tokenizer, device, NEUTRAL_PROMPT, L_EXTRACT)
    scale_n0 = tgt_norm_L3 / max(a_t_n0_norm, 1e-8)
    scaled_n0 = (a_t_n0 * scale_n0).to(device)
    home_logit = forward_neutral_logit(
        model, tokenizer, device, NEUTRAL_PROMPT, tid0,
        inject_vector=scaled_n0, inject_layer=L_EXTRACT,
    )
    home_lift = home_logit - base_logit_n0
    print(f"  needle={n0['id']} word={w0!r} target_bpe={tid0} ({tstr0!r})")
    print(f"  a_t_norm={a_t_n0_norm:.3f}  tgt_norm_L{L_EXTRACT}={tgt_norm_L3:.3f}  scale={scale_n0:.3f}")
    print(f"  base_logit={base_logit_n0:+.3f}  home_logit={home_logit:+.3f}  lift={home_lift:+.3f}")
    if home_lift < 0.3:
        print(f"  *** FAIL: home-position lift {home_lift:.3f} < 0.3. The extract/inject ")
        print(f"      pipeline cannot reproduce target-token signal at its own layer. ")
        print(f"      Either extraction captures the wrong activation, the model can't ")
        print(f"      decode residual activations to tokens in this configuration, or ")
        print(f"      the neutral prompt's geometry doesn't align with the extracted ")
        print(f"      activation's geometry. Aborting — the decay curve would be ")
        print(f"      uninterpretable without a working positive control. ***")
        sys.exit(1)
    print("  PASS\n")

    # =========================================================
    # Per-needle decay curves
    # =========================================================
    print("=== Per-needle decay: extract at L3, inject at L0..L5 ===")
    D = model.cfg.model.d_model

    # Pre-compute residual norms at each layer for the neutral prompt
    print("  Pre-computing neutral-prompt residual norms per layer (final position)...")
    neutral_norms = {L: residual_norm_at_layer(model, tokenizer, device, NEUTRAL_PROMPT, L)
                     for L in range(N_LAYERS)}
    for L, n in neutral_norms.items():
        print(f"    L{L}: ||residual|| = {n:.3f}")

    results = []
    skipped = []
    for nd_idx, nd in enumerate(needles):
        nid = nd["id"]
        if nid not in DISTINCTIVE_ALL:
            skipped.append((nid, "no distinctive token"))
            continue
        word, tid, tstr = pick_target_token(tokenizer, nd)
        target_pos = find_target_position_in_fact(tokenizer, nd["fact"], word, tid)
        if target_pos is None:
            skipped.append((nid, f"target token {tid} ({word!r}) not in fact"))
            continue

        # Baseline logit of target t at final position (no injection)
        base_logit = forward_neutral_logit(model, tokenizer, device, NEUTRAL_PROMPT, tid)
        # Extract activation a_t
        a_t = extract_activation(model, tokenizer, device, nd["fact"], target_pos,
                                 L_extract=L_EXTRACT)
        a_t_norm = float(a_t.norm().item())

        # Inject at each layer, norm-matched to that layer's typical residual norm
        layer_logits = {}
        scales = {}
        for L in range(N_LAYERS):
            target_norm = neutral_norms[L]
            if a_t_norm < 1e-8:
                scale = 0.0
            else:
                scale = target_norm / a_t_norm
            scales[L] = scale
            scaled = (a_t * scale).to(device)
            inj_logit = forward_neutral_logit(
                model, tokenizer, device, NEUTRAL_PROMPT, tid,
                inject_vector=scaled, inject_layer=L,
            )
            layer_logits[L] = inj_logit

        lifts = {L: layer_logits[L] - base_logit for L in range(N_LAYERS)}

        row = {
            "id": nid,
            "target_word": word,
            "target_bpe_id": tid,
            "target_str": tstr,
            "target_pos": target_pos,
            "a_t_norm": a_t_norm,
            "baseline_logit": base_logit,
            "layer_logits": layer_logits,
            "layer_lifts": lifts,
            "scales": scales,
        }
        results.append(row)
        # Format per-needle row
        lifts_str = "  ".join(f"L{L}:{lifts[L]:+.2f}" for L in range(N_LAYERS))
        print(f"  [{nd_idx+1:2d}] {nid:35s} t={tstr!r:>12s} base={base_logit:+.2f}  "
              f"{lifts_str}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} needles:")
        for nid, reason in skipped:
            print(f"    {nid}: {reason}")

    # =========================================================
    # Aggregate across needles
    # =========================================================
    print("\n=== Aggregate decay curve (mean ± std across needles) ===")
    agg_layers = {L: [r["layer_lifts"][L] for r in results] for L in range(N_LAYERS)}
    agg_stats = {}
    for L in range(N_LAYERS):
        xs = agg_layers[L]
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        std = math.sqrt(var)
        agg_stats[L] = {"mean_lift": m, "std_lift": std, "n": len(xs),
                        "min": min(xs), "max": max(xs)}
        print(f"  L{L}: mean={m:+.3f}  std={std:.3f}  min={min(xs):+.2f}  max={max(xs):+.2f}  n={len(xs)}")

    home_layer = L_EXTRACT
    print(f"\n  Home-position check (L_inject = L_extract = {home_layer}): "
          f"mean lift {agg_stats[home_layer]['mean_lift']:+.3f}")
    if agg_stats[home_layer]['mean_lift'] < 0.5:
        print(f"  *** WARNING: home-position lift < 0.5 — extraction/injection pipeline weak ***")

    # =========================================================
    # Classify decay pattern per needle
    # =========================================================
    print("\n=== Per-needle pattern classification ===")
    # Pattern buckets:
    #   flat_high: lift at L5 > 1.0 AND lift at L0 > 1.0
    #   late_only: lift at L5 > 1.0 AND lift at L0 < 0.3
    #   full_suppression: max lift across layers < 0.5
    #   other: anything else
    buckets = {"flat_high": 0, "late_only": 0, "full_suppression": 0, "other": 0}
    pattern_by_id = {}
    for r in results:
        lifts = r["layer_lifts"]
        L0_lift = lifts[0]
        L5_lift = lifts[5]
        max_lift = max(lifts.values())
        if L5_lift > 1.0 and L0_lift > 1.0:
            pattern = "flat_high"
        elif L5_lift > 1.0 and L0_lift < 0.3:
            pattern = "late_only"
        elif max_lift < 0.5:
            pattern = "full_suppression"
        else:
            pattern = "other"
        buckets[pattern] += 1
        pattern_by_id[r["id"]] = pattern
    for k, v in buckets.items():
        print(f"  {k:20s}: {v:3d} / {len(results)}")

    # =========================================================
    # Layer-specific suppression heuristic
    # =========================================================
    # Difference mean_lift[L] - mean_lift[L-1]: if strongly negative, layer L
    # suppresses signal entering from L-1's output.
    print("\n=== Layer-to-layer suppression (lift[L] - lift[L-1]) ===")
    layer_deltas = {}
    for L in range(1, N_LAYERS):
        d = agg_stats[L]["mean_lift"] - agg_stats[L - 1]["mean_lift"]
        layer_deltas[L] = d
        arrow = "↓" if d < 0 else "↑"
        print(f"  L{L-1}→L{L}: {d:+.3f} {arrow}")

    # =========================================================
    # Save
    # =========================================================
    out = {
        "sanity_zero_max_diff": sanity_max_diff,
        "neutral_norms": neutral_norms,
        "per_needle": results,
        "aggregate": agg_stats,
        "patterns": {"buckets": buckets, "by_id": pattern_by_id},
        "layer_deltas": layer_deltas,
        "skipped": skipped,
    }
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
