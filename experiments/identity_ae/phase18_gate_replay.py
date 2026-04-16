"""Phase 18: Novel gate with replay buffer.

Phase 17 showed the novel gate forgets all but the most recent absorbed type
(4/20 recognized). The fix: keep a replay buffer of past absorbed hidden
states and train the novel gate against current + replay each absorption.

Combines phase16/17 setup but replaces the novel-gate training step with
a replay-augmented version.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase18_gate_replay.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_gated, val_ppl_ungated, get_hidden_at_layer,
    RANK, N_STEPS, HIGH_LR, BASE_LR, BASE_THRESHOLD,
)
from experiments.identity_ae.lora_wrapper import apply_lora
from experiments.identity_ae.dual_gate import DualGate


def run_dual_gate_ttt_with_replay(model, dual_gate, passage, tokenizer, device,
                                   replay_buffer, n_steps, high_lr, base_lr,
                                   replay_weight=1.0):
    """TTT with replay-augmented novel gate training.

    LoRA learns the new passage. Novel gate learns to reconstruct
    current passage hidden state + every past absorbed hidden state.
    """
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

        # Current passage hidden state (with current LoRA active)
        with torch.no_grad():
            h_curr = get_hidden_at_layer(model, ids_t).detach()

        # Novel gate loss: current + replay
        encoded = dual_gate.novel_gate.encoder(h_curr)
        decoded = dual_gate.novel_gate.decoder(encoded)
        gate_loss = F.mse_loss(decoded, h_curr)

        if replay_buffer:
            replay_loss = 0.0
            for h_past in replay_buffer:
                enc_p = dual_gate.novel_gate.encoder(h_past)
                dec_p = dual_gate.novel_gate.decoder(enc_p)
                replay_loss = replay_loss + F.mse_loss(dec_p, h_past)
            replay_loss = replay_loss / len(replay_buffer)
            gate_loss = gate_loss + replay_weight * replay_loss

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

    results_dir = Path("results/identity_ae/phase18")
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

    # Stratified 20: 5 of each type (same as phase 17)
    all_tests = generate_passkeys(50)
    by_type = {"numeric": [], "entity": [], "technical": [], "fact": []}
    for t in all_tests:
        by_type[t["type"]].append(t)
    tests = (by_type["numeric"][:5] + by_type["entity"][:5]
             + by_type["technical"][:5] + by_type["fact"][:5])
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")
    print(f"Schedule: {N_STEPS} steps, {HIGH_LR:.1e} -> {BASE_LR:.1e}")
    print(f"Replay buffer: capped at 20 hidden states (one per absorbed passage)\n")

    replay_buffer = []  # list of detached hidden states from past passages

    history = [{"i": 0, "val_gated": baseline_gated, "val_ungated": baseline_ungated,
                "delta_gated": 0.0, "delta_ungated": 0.0,
                "passkey_found": None, "passkey_id": None, "passkey_type": None}]

    n_found = 0
    found_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    t0 = time.time()
    for i, test in enumerate(tests):
        run_dual_gate_ttt_with_replay(model, dual_gate, test["passage"],
                                       tokenizer, device, replay_buffer,
                                       N_STEPS, HIGH_LR, BASE_LR)

        # Save the hidden state of this passage AFTER absorbing it (with current LoRA)
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        with torch.no_grad():
            h_new = get_hidden_at_layer(model, ids_t).detach().clone()
        replay_buffer.append(h_new)

        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        found = check_passkey(gen, test["passkey"])
        if found:
            n_found += 1
            found_by_type[test["type"]] += 1

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
        print(f"  [{i+1:2d}/20] {test['type']:9s} {flag:4s}  "
              f"gated={v_gated:.3f} ({d_gated:+.3f}%)  "
              f"ungated={v_ungated:.2f} ({d_ungated:+.1f}%)  "
              f"replay={len(replay_buffer):2d}  ({elapsed:.0f}s)")

    final = history[-1]

    # Gate classification on absorbed passkeys (using FINAL novel gate weights)
    print("\nMeasuring gate classification on absorbed passkeys...")
    novel_correct = 0
    novel_correct_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        h = get_hidden_at_layer(model, ids_t)
        cat, _, _ = dual_gate.classify(h)
        if cat == "learned_novel":
            novel_correct += 1
            novel_correct_by_type[test["type"]] += 1

    print()
    print(f"{'='*60}")
    print(f"PHASE 18: REPLAY-AUGMENTED NOVEL GATE")
    print(f"{'='*60}")
    print(f"  Retrieval (overall):         {n_found}/20 ({n_found/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {found_by_type[ptype]}/5")
    print(f"  Gate -> learned_novel:       {novel_correct}/20")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {novel_correct_by_type[ptype]}/5")
    print(f"  Phase 17 (no replay) was:    4/20")
    print(f"  Val PPL (gated):             {baseline_gated:.3f} -> {final['val_gated']:.3f} "
          f"({final['delta_gated']:+.3f}%)")
    print(f"  Val PPL (ungated):           {baseline_ungated:.3f} -> {final['val_ungated']:.3f} "
          f"({final['delta_ungated']:+.2f}%)")

    summary = {
        "config": {"rank": RANK, "n_steps": N_STEPS, "high_lr": HIGH_LR,
                   "base_lr": BASE_LR, "targets": L45_TARGETS, "replay": True},
        "n_passages": 20,
        "stratified": True,
        "retrieval": n_found / 20,
        "retrieval_by_type": {k: v / 5 for k, v in found_by_type.items()},
        "gate_novel_correct": novel_correct,
        "gate_novel_by_type": novel_correct_by_type,
        "baseline_gated": baseline_gated,
        "baseline_ungated": baseline_ungated,
        "final_gated": final["val_gated"],
        "final_ungated": final["val_ungated"],
        "delta_gated_pct": final["delta_gated"],
        "delta_ungated_pct": final["delta_ungated"],
        "history": history,
    }
    with open(results_dir / "replay.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
