"""Phase 19: LoRA rehearsal across all accumulated passages.

The earlier phases only trained LoRA on one passage per absorption, then
tested retrieval on that same passage. They never checked whether *prior*
absorbed passkeys were still retrievable. With single-passage training,
the LoRA almost certainly forgets older content as it absorbs new content.

The fix: when absorbing passage N, train on the union {p_1, ..., p_N}.
Test retrieval on all absorbed passages after each absorption.

Step budget per absorption: 100 steps. New-biased: with prob 0.8 train on
the new passage, prob 0.2 train on a random older one. The new passage gets
~80 fresh steps; older passages share ~20 maintenance steps. First absorption
is all-new (no rehearsal possible).

Combined with: L4-5 LoRA rank 512, scheduled 3e-4 -> 1e-4, dual gate (replay).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase19_rehearsal.py
"""

import json
import random
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


random.seed(0)


def run_rehearsal_ttt(model, dual_gate, accumulated_ids, replay_buffer,
                      tokenizer, device, n_steps, high_lr, base_lr,
                      new_prob=0.8):
    """TTT with rehearsal over all accumulated passages.

    Args:
        accumulated_ids: list of token tensors, last entry is the new passage
        replay_buffer: list of past hidden states for the novel gate
    """
    new_idx = len(accumulated_ids) - 1

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
        # Pick which passage to train on this step
        if new_idx == 0 or random.random() < new_prob:
            ids_t = accumulated_ids[new_idx]
        else:
            ids_t = accumulated_ids[random.randint(0, new_idx - 1)]

        out = model(ids_t[:, :-1], step=0)
        lm_loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                   ids_t[:, 1:].reshape(-1))

        # Novel gate: current step's passage hidden state + replay
        with torch.no_grad():
            h_curr = get_hidden_at_layer(model, ids_t).detach()
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
            gate_loss = gate_loss + replay_loss

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

    results_dir = Path("results/identity_ae/phase19")
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

    # Stratified 20: 5 of each type
    all_tests = generate_passkeys(50)
    by_type = {"numeric": [], "entity": [], "technical": [], "fact": []}
    for t in all_tests:
        by_type[t["type"]].append(t)
    tests = (by_type["numeric"][:5] + by_type["entity"][:5]
             + by_type["technical"][:5] + by_type["fact"][:5])
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")
    print(f"Schedule: {N_STEPS} steps/absorption, {HIGH_LR:.1e} -> {BASE_LR:.1e}")
    print(f"Rehearsal: 80% new / 20% old per step\n")

    # Pre-tokenize all passages
    accumulated_ids = []
    replay_buffer = []

    history = []
    cumulative_retrievals = []  # n_retrieved_so_far after each absorption

    t0 = time.time()
    for i, test in enumerate(tests):
        # Add new passage tokens
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        accumulated_ids.append(ids_t)

        # Rehearsal-augmented training
        run_rehearsal_ttt(model, dual_gate, accumulated_ids, replay_buffer,
                          tokenizer, device, N_STEPS, HIGH_LR, BASE_LR)

        # Save hidden state for novel-gate replay
        with torch.no_grad():
            h_new = get_hidden_at_layer(model, ids_t).detach().clone()
        replay_buffer.append(h_new)

        # Test retrieval on ALL accumulated passages (cumulative check)
        retrieved = 0
        per_test = []
        for j, prev_test in enumerate(tests[: i + 1]):
            gen = generate_greedy(model, prev_test["prompt"], tokenizer, device, 50)
            found = check_passkey(gen, prev_test["passkey"])
            if found:
                retrieved += 1
            per_test.append({"i": j, "type": prev_test["type"], "found": found})
        cumulative_retrievals.append(retrieved)

        # Val PPL
        v_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
        v_ungated = val_ppl_ungated(model, loaders["validation"], device)
        d_gated = (v_gated - baseline_gated) / baseline_gated * 100
        d_ungated = (v_ungated - baseline_ungated) / baseline_ungated * 100

        history.append({
            "i": i + 1,
            "new_type": test["type"],
            "retrieved": retrieved,
            "of_total": i + 1,
            "per_test": per_test,
            "val_gated": v_gated,
            "val_ungated": v_ungated,
            "delta_gated": d_gated,
            "delta_ungated": d_ungated,
        })

        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/20] new={test['type']:9s}  "
              f"retrieved={retrieved:2d}/{i+1:<2d}  "
              f"gated=({d_gated:+.3f}%)  ungated=({d_ungated:+.1f}%)  "
              f"({elapsed:.0f}s)")

    final = history[-1]

    # Final per-type retrieval
    final_per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    for p in final["per_test"]:
        if p["found"]:
            final_per_type[p["type"]] += 1

    # Final gate classification
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
    print(f"PHASE 19: REHEARSAL LoRA + REPLAY GATE")
    print(f"{'='*60}")
    print(f"  Final cumulative retrieval:  {final['retrieved']}/20 ({final['retrieved']/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {final_per_type[ptype]}/5")
    print(f"  Cumulative trace:            {cumulative_retrievals}")
    print(f"  Gate -> learned_novel:       {novel_correct}/20")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {novel_correct_by_type[ptype]}/5")
    print(f"  Val PPL (gated):             {baseline_gated:.3f} -> {final['val_gated']:.3f} "
          f"({final['delta_gated']:+.3f}%)")
    print(f"  Val PPL (ungated):           {baseline_ungated:.3f} -> {final['val_ungated']:.3f} "
          f"({final['delta_ungated']:+.2f}%)")

    summary = {
        "config": {"rank": RANK, "n_steps": N_STEPS, "high_lr": HIGH_LR,
                   "base_lr": BASE_LR, "targets": L45_TARGETS,
                   "rehearsal": True, "new_prob": 0.8},
        "n_passages": 20,
        "stratified": True,
        "final_retrieval": final["retrieved"] / 20,
        "final_per_type": {k: v / 5 for k, v in final_per_type.items()},
        "cumulative_retrievals": cumulative_retrievals,
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
    with open(results_dir / "rehearsal.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
