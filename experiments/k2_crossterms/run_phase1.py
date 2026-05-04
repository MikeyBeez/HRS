"""Phase 1: diagnostic baseline for k=2 additive LoRA composition on Dickens-50.

Phase 1.2 — N=1 sanity check
  Run all 50 Dickens-50 adapters one at a time through MultiLoRALayer
  with n_active=1, score retrieval on each adapter's 3 held-out
  paraphrases × 3 seeds. Should match the published evaluate.py number
  (~0.929). If it doesn't, MultiLoRALayer has a bug — halt before k=2.

Phase 1.3 — N=2 baseline
  Sample 50 (i, j) pairs uniformly with seed 42. For each pair, load
  adapters i and j additively (n_active=2) and score retrieval on
  passage i's probes (companion = j) and passage j's probes (companion =
  i). Capture per-block last-token hidden-state drift relative to the
  N=1 reference for the target adapter.

Decision after Phase 1:
  gap = N=1 retrieval - N=2 retrieval
  - gap < 0.05  → null result (cross-term effect minor); stop here.
  - gap >= 0.05 → proceed to Phase 2 with interventions.

Output:
  results/phase1_n1_baseline.csv          per-adapter N=1 retrieval
  results/phase1_n2_baseline.csv          per-pair N=2 retrieval
  results/phase1_layer_drift.csv          per-pair, per-layer drift
  results/phase1_summary.json             aggregates + decision
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/k2_crossterms"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.k2_crossterms.multi_lora import (
    apply_multi_lora, set_active_adapters, reset_multi_lora,
    MultiLoRALayer,
)


RANK = 128
ALPHA = 256
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50
MAX_CTX_POS = 512
N_PAIRS = 50
PAIR_SEED = 42


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed,
              temperature=TEMPERATURE, top_k=TOP_K):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -MAX_CTX_POS:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :].float() / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, generation):
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    if clean_a and clean_a in clean_g:
        return True
    return False


# ---------------- Drift capture via forward hooks ----------------
class LayerCapture:
    """Forward hooks on each block, grab last-token output."""
    def __init__(self, model):
        self.model = model
        self.last_token_outputs = []  # list of (D,) tensors per block
        self._hooks = []

    def __enter__(self):
        self.last_token_outputs = []
        for block in self.model.blocks:
            h = block.register_forward_hook(self._hook)
            self._hooks.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _hook(self, module, inp, out):
        # block returns a tuple (h, routing_w, attn_w, ...)
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
        # h is (B, T, D); grab last-token of batch 0
        self.last_token_outputs.append(h[0, -1, :].detach().float().cpu().clone())


@torch.no_grad()
def capture_layer_states(model, ids_t):
    """Run a forward pass and return list of last-token hidden states per block."""
    with LayerCapture(model) as cap:
        out = model(ids_t, step=0)
    last_logits = out.logits[0, -1, :].detach().float().cpu().clone()
    return cap.last_token_outputs, last_logits


def compute_drift(states_a, states_b):
    """||a - b|| / ||a|| per layer."""
    out = []
    for ha, hb in zip(states_a, states_b):
        denom = ha.norm().item() + 1e-12
        out.append((hb - ha).norm().item() / denom)
    return out


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())
    print(f"Library: {len(library)} entries")

    # Load model + Dickens checkpoint
    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    n_lora_params = apply_multi_lora(model, rank=RANK, alpha=ALPHA,
                                       target_modules=L45_TARGETS, max_k=2)
    reset_multi_lora(model)
    model.eval()
    print(f"MultiLoRA params (max_k=2): {n_lora_params:,}")

    # Load all 50 adapter state_dicts
    adapter_sds = {}
    for e in keys:
        sd = torch.load(REPO / e["sd_path"], map_location=device, weights_only=False)
        adapter_sds[e["id"]] = sd
    print(f"Loaded {len(adapter_sds)} adapter state_dicts.")

    held_out_per_adapter = {}  # id -> list of (probe, answer, fact_type)
    for entry in library:
        held_out_per_adapter[entry["id"]] = [
            {"probe": q, "answer": entry["answer"],
             "fact_type": entry["fact_type"]}
            for q in entry["paraphrases_held_out"]
        ]

    # ====================================================================
    # PHASE 1.2 — N=1 baseline
    # ====================================================================
    print(f"\n{'='*70}\nPHASE 1.2 — N=1 baseline (sanity check vs published 0.929)\n{'='*70}")

    n1_results = []  # (adapter_id, fact_type, seed, retrieval_per_probe)
    n1_state_cache = {}  # adapter_id -> {probe_text -> (states, logits)}
    t0 = time.time()
    for aid in sorted(adapter_sds.keys()):
        set_active_adapters(model, [adapter_sds[aid]])
        per_probe_hits = []
        n1_state_cache[aid] = {}

        for probe_rec in held_out_per_adapter[aid]:
            probe = probe_rec["probe"]
            ids = tokenizer.encode(probe, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

            # Capture hidden states ONCE (deterministic forward, no sampling)
            states, logits = capture_layer_states(model, ids_t)
            n1_state_cache[aid][probe] = {"states": states, "logits": logits,
                                            "n_input_tokens": ids_t.shape[1]}

            # Score retrieval over 3 seeds
            for seed in [0, 1, 2]:
                gen_seed = seed * 10000 + aid * 100 + len(per_probe_hits)
                gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed=gen_seed)
                full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                cont = full[len(probe):]
                hit = check_match(probe_rec["answer"], cont)
                per_probe_hits.append({
                    "adapter_id": aid, "fact_type": probe_rec["fact_type"],
                    "seed": seed, "probe": probe, "answer": probe_rec["answer"],
                    "continuation": cont, "retrieved": hit,
                })

        adapter_acc = sum(r["retrieved"] for r in per_probe_hits) / len(per_probe_hits)
        n1_results.extend(per_probe_hits)
        if (aid + 1) % 10 == 0 or aid == 0:
            print(f"  N=1 adapter {aid:2d}  retrieval={adapter_acc:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # Aggregate N=1
    n1_acc = sum(r["retrieved"] for r in n1_results) / len(n1_results)
    print(f"\n  N=1 OVERALL retrieval = {n1_acc:.3f}  (target ~0.929 from published evaluate.py)")
    n1_per_type = defaultdict(list)
    for r in n1_results:
        n1_per_type[r["fact_type"]].append(r["retrieved"])
    for ft, hits in sorted(n1_per_type.items()):
        print(f"    {ft:>8s}: {sum(hits)/len(hits):.3f} (n={len(hits)})")

    # Halt criterion
    if abs(n1_acc - 0.929) > 0.05:
        print(f"\n  WARNING: N=1 ({n1_acc:.3f}) deviates from published 0.929 by "
              f"{abs(n1_acc-0.929):.3f}. MultiLoRALayer may have a bug. "
              f"Investigate before trusting N=2.")

    # Save N=1 baseline CSV
    with (EXP / "results/phase1_n1_baseline.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adapter_id", "fact_type", "seed", "probe", "answer",
                     "continuation", "retrieved"])
        for r in n1_results:
            w.writerow([r["adapter_id"], r["fact_type"], r["seed"], r["probe"],
                         r["answer"], r["continuation"][:200], int(r["retrieved"])])

    # ====================================================================
    # PHASE 1.3 — N=2 baseline (50 random pairs)
    # ====================================================================
    print(f"\n{'='*70}\nPHASE 1.3 — N=2 baseline ({N_PAIRS} pairs, seed {PAIR_SEED})\n{'='*70}")

    rng = random.Random(PAIR_SEED)
    aids = sorted(adapter_sds.keys())
    pairs = []
    while len(pairs) < N_PAIRS:
        i, j = rng.sample(aids, 2)
        pairs.append((i, j))
    print(f"Sampled {len(pairs)} pairs.")

    # Type distribution of targets (each pair contributes both i and j as targets)
    target_types = defaultdict(int)
    for entry in library:
        target_types[entry["fact_type"]] += 0
    aid_to_type = {e["id"]: e["fact_type"] for e in library}
    for i, j in pairs:
        target_types[aid_to_type[i]] += 1
        target_types[aid_to_type[j]] += 1
    print(f"Target type distribution (out of {2*N_PAIRS}): {dict(target_types)}")

    n2_results = []
    drift_records = []
    t0 = time.time()
    for pi, (i, j) in enumerate(pairs):
        set_active_adapters(model, [adapter_sds[i], adapter_sds[j]])

        for target_aid, companion_aid in [(i, j), (j, i)]:
            for probe_rec in held_out_per_adapter[target_aid]:
                probe = probe_rec["probe"]
                ids = tokenizer.encode(probe, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

                # Capture N=2 hidden states (single deterministic forward)
                states_n2, logits_n2 = capture_layer_states(model, ids_t)

                # Diff vs cached N=1 states for this target+probe
                cached = n1_state_cache[target_aid][probe]
                states_n1 = cached["states"]
                logits_n1 = cached["logits"]
                per_layer = compute_drift(states_n1, states_n2)
                logit_drift = (logits_n2 - logits_n1).norm().item() / (
                    logits_n1.norm().item() + 1e-12)
                drift_records.append({
                    "pair_id": pi, "target_adapter": target_aid,
                    "companion_adapter": companion_aid, "probe": probe,
                    "fact_type": probe_rec["fact_type"],
                    "per_layer_drift": per_layer,
                    "mean_layer_drift": float(np.mean(per_layer)),
                    "max_layer_drift": float(np.max(per_layer)),
                    "logit_drift": logit_drift,
                })

                # Score retrieval over 3 seeds (same gen_seed scheme as N=1
                # would NOT match because adapter context differs; that's
                # fine — we want a fresh sampling at N=2)
                for seed in [0, 1, 2]:
                    gen_seed = seed * 100000 + pi * 1000 + target_aid
                    gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed=gen_seed)
                    full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    cont = full[len(probe):]
                    hit = check_match(probe_rec["answer"], cont)
                    n2_results.append({
                        "pair_id": pi, "target_adapter": target_aid,
                        "companion_adapter": companion_aid,
                        "fact_type": probe_rec["fact_type"], "seed": seed,
                        "probe": probe, "answer": probe_rec["answer"],
                        "continuation": cont, "retrieved": hit,
                    })

        if (pi + 1) % 10 == 0 or pi == 0:
            n2_so_far = (sum(r["retrieved"] for r in n2_results)
                          / max(1, len(n2_results)))
            print(f"  pair {pi+1:2d}/{N_PAIRS}  N=2 running mean = {n2_so_far:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # Aggregate N=2
    n2_acc = sum(r["retrieved"] for r in n2_results) / len(n2_results)
    n2_per_type = defaultdict(list)
    for r in n2_results:
        n2_per_type[r["fact_type"]].append(r["retrieved"])

    print(f"\n  N=2 OVERALL retrieval = {n2_acc:.3f}  ({len(n2_results)} measurements)")
    for ft, hits in sorted(n2_per_type.items()):
        print(f"    {ft:>8s}: {sum(hits)/len(hits):.3f} (n={len(hits)})")

    # Save N=2 results
    with (EXP / "results/phase1_n2_baseline.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "target_adapter", "companion_adapter",
                     "fact_type", "seed", "probe", "answer",
                     "continuation", "retrieved"])
        for r in n2_results:
            w.writerow([r["pair_id"], r["target_adapter"], r["companion_adapter"],
                         r["fact_type"], r["seed"], r["probe"], r["answer"],
                         r["continuation"][:200], int(r["retrieved"])])

    # Save drift records
    with (EXP / "results/phase1_layer_drift.csv").open("w", newline="") as f:
        w = csv.writer(f)
        n_blocks = len(drift_records[0]["per_layer_drift"])
        w.writerow(["pair_id", "target_adapter", "companion_adapter", "probe",
                     "fact_type", "mean_layer_drift", "max_layer_drift",
                     "logit_drift"] + [f"L{i}" for i in range(n_blocks)])
        for d in drift_records:
            w.writerow([d["pair_id"], d["target_adapter"], d["companion_adapter"],
                         d["probe"][:80], d["fact_type"],
                         f"{d['mean_layer_drift']:.4f}",
                         f"{d['max_layer_drift']:.4f}",
                         f"{d['logit_drift']:.4f}"]
                        + [f"{x:.4f}" for x in d["per_layer_drift"]])

    # Layer drift profile
    n_blocks = len(drift_records[0]["per_layer_drift"])
    layer_drift_profile = []
    for L in range(n_blocks):
        vals = [d["per_layer_drift"][L] for d in drift_records]
        layer_drift_profile.append({"layer": L, "mean": float(np.mean(vals)),
                                      "std": float(np.std(vals))})

    print(f"\n  Per-layer mean drift profile (averaged over {len(drift_records)} measurements):")
    for L, prof in enumerate(layer_drift_profile):
        bar = "█" * int(prof["mean"] * 200)
        print(f"    L{L:2d}: {prof['mean']:.4f} ± {prof['std']:.4f}  {bar}")

    mean_logit_drift = float(np.mean([d["logit_drift"] for d in drift_records]))
    print(f"\n  Mean output logit drift = {mean_logit_drift:.4f}")

    # ====================================================================
    # DECISION
    # ====================================================================
    gap = n1_acc - n2_acc
    print(f"\n{'='*70}\nDECISION\n{'='*70}")
    print(f"  N=1 retrieval: {n1_acc:.3f}")
    print(f"  N=2 retrieval: {n2_acc:.3f}")
    print(f"  Gap          : {gap:.3f}")
    if gap < 0.05:
        decision = "NULL_RESULT_STOP"
        print(f"  → gap < 0.05 → NULL RESULT. Cross-term effect is empirically minor.")
        print(f"     Stop at Phase 1; no Phase 2 interventions warranted.")
    else:
        decision = "PROCEED_PHASE_2"
        print(f"  → gap >= 0.05 → proceed to Phase 2 with interventions.")

    summary = {
        "phase": 1,
        "n1_overall_retrieval": n1_acc,
        "n2_overall_retrieval": n2_acc,
        "gap": gap,
        "decision": decision,
        "n1_per_fact_type": {ft: sum(h)/len(h) for ft, h in n1_per_type.items()},
        "n2_per_fact_type": {ft: sum(h)/len(h) for ft, h in n2_per_type.items()},
        "target_type_distribution_n2": dict(target_types),
        "layer_drift_profile": layer_drift_profile,
        "mean_logit_drift": mean_logit_drift,
        "n_pairs": N_PAIRS,
        "pair_seed": PAIR_SEED,
        "config": {"rank": RANK, "alpha": ALPHA, "gen_tokens": GEN_TOKENS,
                    "temperature": TEMPERATURE, "top_k": TOP_K},
        "wall_total_s": time.time() - t0,
    }
    (EXP / "results/phase1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {EXP/'results/phase1_summary.json'}")
    print(f"Total Phase 1 wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
