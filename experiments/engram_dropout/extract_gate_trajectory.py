"""Extract V20→V22 cross-attention gate trajectory from saved checkpoints.

Read-only: opens each .pt, pulls gate_logit / gate_scalar at active cross-attn
layers, computes effective gate = sigmoid(gate_logit) * softplus(gate_scalar),
and tabulates over training steps.

V20 owns steps 0–43000 (phase1, phase2). V22 continues from V20's mature
checkpoint at step 43000 through step 63000. Layer 3 cross-attn is disabled
in V22 (use_cross_attn_engram=False) but its parameters remain in the
state_dict; we still report them since they show V20's pre-V22-disabling
trajectory.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

# Make `config`, `model`, etc. importable so torch.load can unpickle
# checkpoints saved with custom classes.
REPO_FOR_IMPORT = Path("/mnt/data/Code/HRS")
if str(REPO_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_FOR_IMPORT))


REPO = Path("/mnt/data/Code/HRS")

# Canonical chain: V20 early-to-mature, then V22 continuation.
CHECKPOINTS = [
    ("V20 phase1_end (after frozen-MLP phase, step ~20000)",
     REPO / "results/v20_bonsignore/phase1_end.pt"),
    ("V20 checkpoint_40000",
     REPO / "results/v20_bonsignore/checkpoint_40000.pt"),
    ("V20 checkpoint_42500",
     REPO / "results/v20_bonsignore/checkpoint_42500.pt"),
    ("V20 best (within phase 2 run)",
     REPO / "results/v20_bonsignore/best.pt"),
    ("V20 final (end of V20 training, step ~43000)",
     REPO / "results/v20_bonsignore/final.pt"),
    ("V22 final (end of V22 phase 1 continuation, step ~53000)",
     REPO / "results/v22_learned_kernel/final.pt"),
    ("V22 best (canonical 17.07 ckpt, step 61000)",
     REPO / "results/v22_learned_kernel/best.pt"),
    ("V22 final_63k (end of V22 training)",
     REPO / "results/v22_learned_kernel/final_63k.pt"),
]


def effective_gate(gate_logit: float, gate_scalar: float | None) -> float:
    """V20-era gate is sigmoid(gate_logit) only (no gate_scalar yet).
    V21+ adds softplus(gate_scalar) as a multiplier."""
    base = float(torch.sigmoid(torch.tensor(gate_logit)))
    if gate_scalar is None:
        return base
    return base * float(torch.nn.functional.softplus(torch.tensor(gate_scalar)))


def extract_one(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    step = ck.get("step", None) if isinstance(ck, dict) else None
    val_ppl = ck.get("val_ppl", None) if isinstance(ck, dict) else None

    layers = sorted(set(
        int(k.split(".")[1]) for k in sd
        if k.startswith("blocks.") and "cross_attn" in k
    ))
    per_layer = {}
    for li in layers:
        gl_k = f"blocks.{li}.cross_attn.gate_logit"
        gs_k = f"blocks.{li}.cross_attn.gate_scalar"
        op_k = f"blocks.{li}.cross_attn.out_proj.weight"
        if gl_k in sd:
            gl = float(sd[gl_k].item())
            gs = float(sd[gs_k].item()) if gs_k in sd else None
            eff = effective_gate(gl, gs)
            row = {"gate_logit": gl,
                   "gate_scalar": gs,           # may be None for V20
                   "effective_gate": eff,
                   "has_gate_scalar": gs is not None}
            if op_k in sd:
                row["out_proj_norm"] = float(sd[op_k].norm().item())
            per_layer[li] = row

    eb = sd.get("engram_buffer", None)
    eb_summary = None
    if eb is not None:
        eb_summary = {
            "shape": list(eb.shape),
            "norm": float(eb.norm().item()),
            "mean_abs": float(eb.abs().mean().item()),
            "std": float(eb.std().item()),
            "all_zero": bool((eb == 0).all().item()),
        }
    return {
        "ckpt_path": str(path),
        "step": step, "val_ppl": val_ppl,
        "per_layer": per_layer,
        "engram_buffer": eb_summary,
    }


def main():
    out_path = REPO / "experiments/engram_dropout/results_stage_a/gate_trajectory.json"
    rows = []
    print(f"{'label':<60s}  {'step':>6s}  {'val_ppl':>8s}  layer1_gate  layer3_gate  layer5_gate  eb_norm")
    print("-" * 130)
    for label, path in CHECKPOINTS:
        if not path.exists():
            print(f"{label:<60s}  -- FILE MISSING --")
            continue
        d = extract_one(path)
        rows.append({"label": label, **d})
        step_s = str(d["step"]) if d["step"] is not None else "—"
        ppl_s = f"{d['val_ppl']:.3f}" if d["val_ppl"] is not None else "—"
        l1 = d["per_layer"].get(1, {}).get("effective_gate", float("nan"))
        l3 = d["per_layer"].get(3, {}).get("effective_gate", float("nan"))
        l5 = d["per_layer"].get(5, {}).get("effective_gate", float("nan"))
        eb_n = d["engram_buffer"]["norm"] if d["engram_buffer"] else float("nan")
        print(f"{label:<60s}  {step_s:>6}  {ppl_s:>8}  {l1:>11.4f}  {l3:>11.4f}  {l5:>11.4f}  {eb_n:>7.2f}")

    print("\n--- gate_logit / gate_scalar pairs per active layer ---")
    print(f"{'label':<60s}  layer  gate_logit  gate_scalar  effective  out_proj_norm")
    for r in rows:
        for li in sorted(r["per_layer"]):
            d = r["per_layer"][li]
            gs = d["gate_scalar"]
            gs_s = f"{gs:>+11.4f}" if gs is not None else f"{'(none)':>11}"
            print(f"{r['label']:<60s}  {li:>5}  {d['gate_logit']:>+10.4f}  "
                  f"{gs_s}  {d['effective_gate']:>9.5f}  "
                  f"{d.get('out_proj_norm', float('nan')):>13.4f}")

    # Print "raw sigmoid(gate_logit)" comparison so V20 and V22 are like-for-like
    # (V20 has no gate_scalar; V22 multiplies in softplus(gate_scalar)).
    print("\n--- raw sigmoid(gate_logit) per layer (V20 vs V22 comparable) ---")
    print(f"{'label':<60s}  {'step':>6s}  L1_sig    L3_sig    L5_sig")
    for r in rows:
        sigs = {}
        for li in [1, 3, 5]:
            d = r["per_layer"].get(li)
            if d:
                sigs[li] = float(torch.sigmoid(torch.tensor(d["gate_logit"])))
        s1 = f"{sigs.get(1, float('nan')):.4f}"
        s3 = f"{sigs.get(3, float('nan')):.4f}"
        s5 = f"{sigs.get(5, float('nan')):.4f}"
        step_s = str(r["step"]) if r["step"] is not None else "—"
        print(f"{r['label']:<60s}  {step_s:>6}  {s1}    {s3}    {s5}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"trajectory": rows}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
