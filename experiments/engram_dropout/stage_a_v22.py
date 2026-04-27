"""Stage A pre-flight: V22-as-shipped engram pathway gate / ablation check.

Goal: confirm v22_learned_kernel exhibits non-trivial engram reliance, so
the dropout sweep has signal to measure against.

Pass criteria (per Phase-2 spec):
  - ablation gap (ppl_off - ppl_on) >= 1.0 PPL
  - cross-attn gate magnitude is non-trivial (not collapsed near zero)

If pass: proceed to Stage B (3 single-seed runs at p ∈ {0.0, 0.1, 0.25}).
If fail: stop and report — pathway isn't carrying the load we assumed.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, build_dataloaders


CKPT_PATH = REPO / "results/v22_learned_kernel/best.pt"
OUT_DIR = REPO / "experiments/engram_dropout/results_stage_a"


def build_model(device):
    """Recreate the V22 model architecture and load best.pt.

    V22 = V20_BONSIGNORE config + disable layer-3 cross-attention (per train_v22.py:133).
    """
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    splits, _ = load_wikitext(
        cfg.training.dataset, cfg.model.max_seq_len,
        with_categories=cfg.uses_categorization(),
        n_categories=cfg.cross_attn_engram.num_categories
            if cfg.uses_categorization() else 50,
    )
    model = HRSTransformer(cfg).to(device)
    # V22 modification (matches train_v22.py:133-137): disable layer-3 cross-attn
    for block in model.blocks:
        if hasattr(block, "cross_attn") and block.use_cross_attn_engram \
                and block.layer_idx == 3:
            block.use_cross_attn_engram = False

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load_state_dict] missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  [load_state_dict] unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")
    print(f"  Loaded checkpoint: step={ckpt.get('step')}, val_ppl_in_ckpt="
          f"{ckpt.get('val_ppl', float('nan')):.3f}")

    # Force the engram_buffer to be marked initialized (it should already be
    # populated from training; otherwise eb=None defeats the point).
    if hasattr(model, "_engram_buffer_initialized"):
        model._engram_buffer_initialized = True
    return model, cfg, splits


@torch.no_grad()
def eval_ppl(model, loader, device, mode: str, n_batches: int = 50,
              amp_dtype=torch.bfloat16) -> dict:
    """Eval with engram pathway either active ('on') or disabled ('off').

    'off' is implemented by temporarily setting _engram_buffer_initialized=False,
    which causes model.forward to pass eb=None into each block — bypassing
    the cross-attn add at line 447-448.
    """
    model.eval()
    losses = []
    gate_values_per_layer: list[list[float]] = []
    eb_norms: list[float] = []

    # For 'off' mode, monkey-flip the init flag.
    saved_init = getattr(model, "_engram_buffer_initialized", None)
    if mode == "off":
        model._engram_buffer_initialized = False

    if mode == "on" and hasattr(model, "engram_buffer"):
        eb_norms.append(float(model.engram_buffer.norm().item()))

    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            output = model(x, step=0)
        loss = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]),
                                y.reshape(-1))
        losses.append(loss.item())
        if mode == "on" and output.cross_attn_gate_values is not None:
            gate_values_per_layer.append([float(g) for g in output.cross_attn_gate_values])

    if saved_init is not None:
        model._engram_buffer_initialized = saved_init

    avg_ce = sum(losses) / len(losses)
    ppl = math.exp(avg_ce)
    out = {"mode": mode, "ce": avg_ce, "ppl": ppl, "n_batches": len(losses)}
    if gate_values_per_layer:
        # Per-layer mean across batches
        n_layers = len(gate_values_per_layer[0])
        per_layer_mean = [
            sum(g[li] for g in gate_values_per_layer) / len(gate_values_per_layer)
            for li in range(n_layers)
        ]
        out["gate_per_layer_mean"] = per_layer_mean
    if eb_norms:
        out["engram_buffer_l2"] = sum(eb_norms) / len(eb_norms)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Stage A: loading V22 (v22_learned_kernel/best.pt) on {device}")
    t0 = time.time()
    model, cfg, splits = build_model(device)
    print(f"  Model ready in {time.time()-t0:.0f}s, params={sum(p.numel() for p in model.parameters()):,}")

    loaders = build_dataloaders(splits, cfg.training.batch_size)
    val_loader = loaders["validation"]

    print("\n=== Eval: engram ON ===")
    on = eval_ppl(model, val_loader, device, mode="on", n_batches=50)
    print(f"  ppl_on  = {on['ppl']:.3f}  (n_batches={on['n_batches']})")
    if "gate_per_layer_mean" in on:
        print(f"  gate values per layer (where active): {on['gate_per_layer_mean']}")
    if "engram_buffer_l2" in on:
        print(f"  engram_buffer L2 norm: {on['engram_buffer_l2']:.3f}")

    print("\n=== Eval: engram OFF (cross-attn bypassed) ===")
    off = eval_ppl(model, val_loader, device, mode="off", n_batches=50)
    print(f"  ppl_off = {off['ppl']:.3f}  (n_batches={off['n_batches']})")

    gap = off["ppl"] - on["ppl"]
    print(f"\n=== Result ===")
    print(f"  ppl_on  = {on['ppl']:.3f}")
    print(f"  ppl_off = {off['ppl']:.3f}")
    print(f"  ablation gap = {gap:+.3f} PPL")
    nontrivial_gate = False
    if "gate_per_layer_mean" in on:
        nontrivial_gate = any(g > 0.05 for g in on["gate_per_layer_mean"])
        print(f"  any gate > 0.05? {nontrivial_gate}")
    pass_criteria = gap >= 1.0 and nontrivial_gate
    print(f"  PASS = {pass_criteria}  (need gap >= 1.0 AND gate non-trivial)")

    out_path = OUT_DIR / "stage_a_result.json"
    out_path.write_text(json.dumps({
        "checkpoint": str(CKPT_PATH),
        "on": on, "off": off, "gap": gap,
        "pass": pass_criteria,
    }, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Total wall: {time.time()-t0:.0f}s")
    return pass_criteria


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
