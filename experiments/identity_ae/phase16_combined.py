"""Phase 16: Combined config — L4-5 LoRA + scheduled LR + dual gate.

Combines the winning ingredients from phases 9, 11, and 14:
  - Layers 4-5 attn + peer_ffn LoRA (phase 14: capacity for 100% retrieval)
  - Rank 512, scheduled LR 3e-4 -> 1e-4 over 100 steps (phase 11/14)
  - Dual gate routing (phase 9: adapter bypassed for ID content -> 0% drift)

Hypothesis: 100% retrieval AND ~0% cumulative val PPL drift across 20 passkeys.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase16_combined.py
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.lora_wrapper import apply_lora
from experiments.identity_ae.dual_gate import DualGate


INSERT_LAYER = 3
RANK = 512
N_STEPS = 100
HIGH_LR = 3e-4
BASE_LR = 1e-4
BASE_THRESHOLD = 0.241
N_PASSAGES = 20


@torch.no_grad()
def get_hidden_at_layer(model, ids_t, layer_idx=INSERT_LAYER):
    """Run forward through layers 0..layer_idx and return hidden state."""
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == layer_idx:
            return h
    return h


@torch.no_grad()
def val_ppl_gated(model, dual_gate, loader, device, n_batches=20):
    """Val PPL with dual gate: adapter zeroed for in_distribution batches."""
    model.eval()
    total = 0.0
    nb = 0
    dual_gate.reset_stats()
    for batch in loader:
        if nb >= n_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        h = get_hidden_at_layer(model, x)
        category, _, _ = dual_gate.classify(h)

        if category == "in_distribution":
            saved = {}
            for n, p in model.named_parameters():
                if 'lora_B' in n:
                    saved[n] = p.data.clone()
                    p.data.zero_()
            out = model(x, step=0)
            for n, p in model.named_parameters():
                if n in saved:
                    p.data.copy_(saved[n])
        else:
            out = model(x, step=0)

        B, T, V = out.logits.shape
        total += F.cross_entropy(out.logits.reshape(B*T, V), y.reshape(B*T)).item()
        nb += 1
    return math.exp(min(total / nb, 20))


@torch.no_grad()
def val_ppl_ungated(model, loader, device, n_batches=20):
    model.eval()
    total = 0.0
    nb = 0
    for batch in loader:
        if nb >= n_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x, step=0)
        B, T, V = out.logits.shape
        total += F.cross_entropy(out.logits.reshape(B*T, V), y.reshape(B*T)).item()
        nb += 1
    return math.exp(min(total / nb, 20))


def run_dual_gate_ttt_scheduled(model, dual_gate, passage, tokenizer, device,
                                 n_steps, high_lr, base_lr):
    """Train LoRA + novel gate on passage with scheduled LR."""
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    novel_params = list(dual_gate.novel_gate.parameters())
    all_params = lora_params + novel_params

    optimizer = torch.optim.Adam(all_params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )

    model.train()
    dual_gate.novel_gate.train()
    for _ in range(n_steps):
        out = model(ids_t[:, :-1], step=0)
        lm_loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                   ids_t[:, 1:].reshape(-1))

        with torch.no_grad():
            h = get_hidden_at_layer(model, ids_t)
        h_input = h.detach()
        encoded = dual_gate.novel_gate.encoder(h_input)
        decoded = dual_gate.novel_gate.decoder(encoded)
        gate_loss = F.mse_loss(decoded, h_input)

        loss = lm_loss + 0.1 * gate_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    dual_gate.novel_gate.eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase16")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params: {n_lora:,} on {len(L45_TARGETS)} layers (4-5)")

    dual_gate = DualGate(d_model=1024, base_threshold=BASE_THRESHOLD,
                         novel_threshold=BASE_THRESHOLD)
    dual_gate.load_base_gate("results/identity_ae/phase0/autoencoder_init_20ep.pt")
    dual_gate.to(device)
    for p in dual_gate.novel_gate.parameters():
        p.requires_grad = True
    print("Dual gate loaded")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    baseline_ungated = val_ppl_ungated(model, loaders["validation"], device)
    baseline_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
    print(f"Baseline val PPL (ungated): {baseline_ungated:.3f}")
    print(f"Baseline val PPL (gated):   {baseline_gated:.3f}")
    print(f"Schedule: {N_STEPS} steps, {HIGH_LR:.1e} -> {BASE_LR:.1e}")
    print(f"Absorbing {N_PASSAGES} passkeys cumulatively\n")

    tests = generate_passkeys(50)[:N_PASSAGES]

    history = [{"i": 0, "val_gated": baseline_gated, "val_ungated": baseline_ungated,
                "delta_gated": 0.0, "delta_ungated": 0.0,
                "passkey_found": None, "passkey_id": None}]

    n_found = 0
    t0 = time.time()
    for i, test in enumerate(tests):
        run_dual_gate_ttt_scheduled(model, dual_gate, test["passage"],
                                     tokenizer, device, N_STEPS, HIGH_LR, BASE_LR)

        # Retrieval — adapter active (no gating during generation)
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        found = check_passkey(gen, test["passkey"])
        if found:
            n_found += 1

        # Val PPL: gated (the real claim) and ungated (for comparison)
        v_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
        v_ungated = val_ppl_ungated(model, loaders["validation"], device)
        d_gated = (v_gated - baseline_gated) / baseline_gated * 100
        d_ungated = (v_ungated - baseline_ungated) / baseline_ungated * 100

        history.append({
            "i": i + 1,
            "val_gated": v_gated,
            "val_ungated": v_ungated,
            "delta_gated": d_gated,
            "delta_ungated": d_ungated,
            "passkey_found": found,
            "passkey_id": test["id"],
            "passkey_type": test["type"],
        })

        elapsed = time.time() - t0
        flag = "OK" if found else "MISS"
        print(f"  [{i+1:2d}/{N_PASSAGES}] {test['type']:9s} {flag:4s}  "
              f"gated={v_gated:.3f} ({d_gated:+.2f}%)  "
              f"ungated={v_ungated:.3f} ({d_ungated:+.2f}%)  ({elapsed:.0f}s)")

    final = history[-1]
    per_passage_gated = final["delta_gated"] / N_PASSAGES
    per_passage_ungated = final["delta_ungated"] / N_PASSAGES

    # Gate classification accuracy on absorbed passkeys
    print("\nMeasuring gate classification on absorbed passkeys...")
    dual_gate.reset_stats()
    novel_correct = 0
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        h = get_hidden_at_layer(model, ids_t)
        cat, _, _ = dual_gate.classify(h)
        if cat == "learned_novel":
            novel_correct += 1

    print()
    print(f"{'='*60}")
    print(f"PHASE 16: COMBINED L4-5 + DUAL GATE ({N_PASSAGES} passages)")
    print(f"{'='*60}")
    print(f"  Retrieval:                   {n_found}/{N_PASSAGES} ({n_found/N_PASSAGES:.0%})")
    print(f"  Gate -> learned_novel:       {novel_correct}/{N_PASSAGES}")
    print(f"  Val PPL (gated):             {baseline_gated:.3f} -> {final['val_gated']:.3f} "
          f"({final['delta_gated']:+.2f}%)")
    print(f"  Val PPL (ungated, comparison): {baseline_ungated:.3f} -> {final['val_ungated']:.3f} "
          f"({final['delta_ungated']:+.2f}%)")
    print(f"  Per-passage drift (gated):   {per_passage_gated:+.3f}%")
    print(f"  Per-passage drift (ungated): {per_passage_ungated:+.3f}%")

    summary = {
        "config": {"rank": RANK, "n_steps": N_STEPS, "high_lr": HIGH_LR,
                   "base_lr": BASE_LR, "targets": L45_TARGETS},
        "n_passages": N_PASSAGES,
        "retrieval": n_found / N_PASSAGES,
        "gate_novel_correct": novel_correct,
        "baseline_gated": baseline_gated,
        "baseline_ungated": baseline_ungated,
        "final_gated": final["val_gated"],
        "final_ungated": final["val_ungated"],
        "delta_gated_pct": final["delta_gated"],
        "delta_ungated_pct": final["delta_ungated"],
        "per_passage_gated_pct": per_passage_gated,
        "per_passage_ungated_pct": per_passage_ungated,
        "history": history,
    }
    with open(results_dir / "combined.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
