"""Phase 2 (no-retraining interventions): Taylor linearization (A) and
discrete per-layer routing (D).

Both reuse the existing 50 Dickens-50 adapters. No new training.

Intervention A — Taylor linearization
  At each generation step, instead of running model(N=2), run:
    logits_taylor = logits(adapter_i alone) + logits(adapter_j alone)
                    - logits(base, no adapters)
  This is the first-order Taylor expansion of the model output in the
  LoRA contributions, evaluated at the no-adapter base point. Cross
  terms above first order are excluded by construction.

  Interpretations:
    retrieval(taylor) ≈ retrieval(N=1 single):  cross terms are
        higher-order and removable; Taylor is a free fix at 3x forward cost.
    retrieval(taylor) ≈ retrieval(N=2 vanilla): cross terms are
        first-order in the adapter contributions; the gap to N=1 is
        intrinsic to additive composition at THIS rank/scaling.
    retrieval(taylor) << retrieval(N=2 vanilla): the linear approximation
        is broken; adapters are too large for it to be valid.

Intervention D — Discrete per-layer routing
  V22 has 6 blocks, LoRA at L4-L5 only. We split:
    L4 → adapter i only (other slot zeroed)
    L5 → adapter j only (other slot zeroed)
  Same split applied for both target adapters in each pair (so adapter
  i is at L4 when its probes are scored, adapter j is at L4 when its
  probes are scored — symmetric, but on opposite layers each time).

  This trivially eliminates cross terms (no layer ever has both
  adapters active simultaneously) at the cost of giving each adapter
  only half its trained representational pathway.

Output:
  results/phase2a_taylor.csv     per-(pair, target, probe, seed) outcomes
  results/phase2d_discrete.csv   same shape
  results/phase2_no_retrain_summary.json   aggregates
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
import torch.nn as nn
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


def check_match(answer, generation):
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    if clean_a and clean_a in clean_g:
        return True
    return False


@torch.no_grad()
def _step_logits(model, idx):
    return model(idx, step=0).logits[:, -1, :]


@torch.no_grad()
def generate_taylor(model, ids_t, sd_i, sd_j, n_tokens, gen_seed,
                     temperature=TEMPERATURE, top_k=TOP_K):
    """Generation under first-order Taylor linearization in adapter contributions.

    At each step:
        logits_taylor = logits(adapter_i) + logits(adapter_j) - logits(base)
    Then sample as usual.
    """
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -MAX_CTX_POS:]
        set_active_adapters(model, [sd_i])
        l_i = _step_logits(model, idx)
        set_active_adapters(model, [sd_j])
        l_j = _step_logits(model, idx)
        reset_multi_lora(model)
        l_base = _step_logits(model, idx)

        logits = (l_i + l_j - l_base).float() / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


# ---------------- Discrete per-layer routing ----------------

def _layer_index_from_name(name):
    """Module names look like 'blocks.4.attn.qkv' or
    'transformer.blocks.4.attn.qkv'. Pull out the integer after 'blocks.'."""
    marker = "blocks."
    idx = name.find(marker)
    if idx == -1:
        return None
    tail = name[idx + len(marker):]
    head = tail.split(".", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


@torch.no_grad()
def set_per_layer_adapters(model, layer_to_sd):
    """Load a different adapter into the slot for each LoRA-augmented layer.

    layer_to_sd: dict mapping layer index -> single-adapter state_dict
    All MultiLoRALayer modules at layer L get their slot 0 set to
    layer_to_sd[L]'s A and B (with slot 1 zeroed). Layers not in the
    dict are zeroed (no contribution).
    """
    # First: zero everything, set n_active = 1 (one slot used)
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            module.n_active = 1
            for slot in range(module.max_k):
                module.lora_Bs[slot].zero_()

    # For each LoRA-augmented module, look up its layer and load the right adapter
    for full_name, module in model.named_modules():
        if not isinstance(module, MultiLoRALayer):
            continue
        L = _layer_index_from_name(full_name)
        if L is None or L not in layer_to_sd:
            continue
        sd = layer_to_sd[L]
        # Find the (lora_A, lora_B) tensors in sd that belong to THIS module
        # by matching the prefix.
        prefix = full_name + "."
        for key, val in sd.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix == "lora_A":
                module.lora_As[0].data.copy_(val)
            elif suffix == "lora_B":
                module.lora_Bs[0].data.copy_(val)


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


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())
    print(f"Library: {len(library)} entries")

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_multi_lora(model, rank=RANK, alpha=ALPHA,
                      target_modules=L45_TARGETS, max_k=2)
    reset_multi_lora(model)
    model.eval()

    # Confirm the LoRA-augmented layers (introspect from model)
    lora_layers_present = sorted({
        _layer_index_from_name(n)
        for n, m in model.named_modules() if isinstance(m, MultiLoRALayer)
        if _layer_index_from_name(n) is not None
    })
    print(f"LoRA-augmented layers: {lora_layers_present}")

    # Load all 50 adapter state_dicts
    adapter_sds = {}
    for e in keys:
        sd = torch.load(REPO / e["sd_path"], map_location=device, weights_only=False)
        adapter_sds[e["id"]] = sd
    print(f"Loaded {len(adapter_sds)} adapter state_dicts.")

    held_out_per_adapter = {}
    aid_to_type = {}
    for entry in library:
        held_out_per_adapter[entry["id"]] = [
            {"probe": q, "answer": entry["answer"],
             "fact_type": entry["fact_type"]}
            for q in entry["paraphrases_held_out"]
        ]
        aid_to_type[entry["id"]] = entry["fact_type"]

    rng = random.Random(PAIR_SEED)
    aids = sorted(adapter_sds.keys())
    pairs = []
    while len(pairs) < N_PAIRS:
        i, j = rng.sample(aids, 2)
        pairs.append((i, j))
    print(f"Sampled {len(pairs)} pairs (seed {PAIR_SEED}).")

    # ====================================================================
    # INTERVENTION A — Taylor linearization
    # ====================================================================
    print(f"\n{'='*72}\nINTERVENTION A — Taylor linearization\n{'='*72}")
    a_results = []
    t0 = time.time()
    for pi, (i, j) in enumerate(pairs):
        sd_i, sd_j = adapter_sds[i], adapter_sds[j]
        for target_aid, companion_aid, sd_t, sd_c in [
            (i, j, sd_i, sd_j), (j, i, sd_j, sd_i)
        ]:
            for probe_rec in held_out_per_adapter[target_aid]:
                probe = probe_rec["probe"]
                ids = tokenizer.encode(probe, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                for seed in [0, 1, 2]:
                    gen_seed = seed * 100000 + pi * 1000 + target_aid
                    gen_ids = generate_taylor(
                        model, ids_t, sd_t, sd_c, GEN_TOKENS, gen_seed
                    )
                    full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    cont = full[len(probe):]
                    hit = check_match(probe_rec["answer"], cont)
                    a_results.append({
                        "pair_id": pi, "target_adapter": target_aid,
                        "companion_adapter": companion_aid,
                        "fact_type": probe_rec["fact_type"], "seed": seed,
                        "probe": probe, "answer": probe_rec["answer"],
                        "continuation": cont, "retrieved": hit,
                    })
        if (pi + 1) % 10 == 0 or pi == 0:
            sf = sum(r["retrieved"] for r in a_results) / max(1, len(a_results))
            print(f"  taylor pair {pi+1:2d}/{N_PAIRS}  running mean = {sf:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    a_acc = sum(r["retrieved"] for r in a_results) / len(a_results)
    a_per_type = defaultdict(list)
    for r in a_results:
        a_per_type[r["fact_type"]].append(r["retrieved"])
    print(f"\n  TAYLOR retrieval = {a_acc:.3f}")
    for ft, hits in sorted(a_per_type.items()):
        print(f"    {ft:>8s}: {sum(hits)/len(hits):.3f} (n={len(hits)})")

    with (EXP / "results/phase2a_taylor.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "target_adapter", "companion_adapter",
                     "fact_type", "seed", "probe", "answer",
                     "continuation", "retrieved"])
        for r in a_results:
            w.writerow([r["pair_id"], r["target_adapter"], r["companion_adapter"],
                         r["fact_type"], r["seed"], r["probe"], r["answer"],
                         r["continuation"][:200], int(r["retrieved"])])

    # ====================================================================
    # INTERVENTION D — Discrete per-layer routing
    # ====================================================================
    print(f"\n{'='*72}\nINTERVENTION D — Discrete per-layer routing\n{'='*72}")
    print(f"  Layer assignment: L{lora_layers_present[0]} = target, "
          f"L{lora_layers_present[1]} = companion")

    d_results = []
    t0 = time.time()
    for pi, (i, j) in enumerate(pairs):
        sd_i, sd_j = adapter_sds[i], adapter_sds[j]
        for target_aid, companion_aid, sd_t, sd_c in [
            (i, j, sd_i, sd_j), (j, i, sd_j, sd_i)
        ]:
            # When measuring TARGET, give it the EARLIER LoRA layer; companion gets the later
            layer_to_sd = {
                lora_layers_present[0]: sd_t,
                lora_layers_present[1]: sd_c,
            }
            set_per_layer_adapters(model, layer_to_sd)
            for probe_rec in held_out_per_adapter[target_aid]:
                probe = probe_rec["probe"]
                ids = tokenizer.encode(probe, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                for seed in [0, 1, 2]:
                    gen_seed = seed * 100000 + pi * 1000 + target_aid
                    gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed)
                    full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    cont = full[len(probe):]
                    hit = check_match(probe_rec["answer"], cont)
                    d_results.append({
                        "pair_id": pi, "target_adapter": target_aid,
                        "companion_adapter": companion_aid,
                        "fact_type": probe_rec["fact_type"], "seed": seed,
                        "probe": probe, "answer": probe_rec["answer"],
                        "continuation": cont, "retrieved": hit,
                        "target_layer": lora_layers_present[0],
                        "companion_layer": lora_layers_present[1],
                    })
        if (pi + 1) % 10 == 0 or pi == 0:
            sf = sum(r["retrieved"] for r in d_results) / max(1, len(d_results))
            print(f"  discrete pair {pi+1:2d}/{N_PAIRS}  running mean = {sf:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    d_acc = sum(r["retrieved"] for r in d_results) / len(d_results)
    d_per_type = defaultdict(list)
    for r in d_results:
        d_per_type[r["fact_type"]].append(r["retrieved"])
    print(f"\n  DISCRETE retrieval = {d_acc:.3f}")
    for ft, hits in sorted(d_per_type.items()):
        print(f"    {ft:>8s}: {sum(hits)/len(hits):.3f} (n={len(hits)})")

    with (EXP / "results/phase2d_discrete.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "target_adapter", "companion_adapter",
                     "fact_type", "seed", "probe", "answer",
                     "continuation", "retrieved", "target_layer", "companion_layer"])
        for r in d_results:
            w.writerow([r["pair_id"], r["target_adapter"], r["companion_adapter"],
                         r["fact_type"], r["seed"], r["probe"], r["answer"],
                         r["continuation"][:200], int(r["retrieved"]),
                         r["target_layer"], r["companion_layer"]])

    # ---- Summary ----
    summary = {
        "phase": "2_no_retrain",
        "taylor": {
            "overall": a_acc,
            "per_fact_type": {ft: sum(h)/len(h) for ft, h in a_per_type.items()},
        },
        "discrete": {
            "overall": d_acc,
            "per_fact_type": {ft: sum(h)/len(h) for ft, h in d_per_type.items()},
            "target_layer": lora_layers_present[0],
            "companion_layer": lora_layers_present[1],
        },
        "context": {
            "phase1_n1": 0.924,
            "phase1_n2_vanilla": 0.651,
            "lora_layers": lora_layers_present,
        },
        "n_pairs": N_PAIRS,
        "pair_seed": PAIR_SEED,
        "wall_total_s": time.time() - t0,
    }
    (EXP / "results/phase2_no_retrain_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSaved {EXP/'results/phase2_no_retrain_summary.json'}")
    print(f"\n  vs N=1 baseline (0.924) and N=2 vanilla (0.651):")
    print(f"     Taylor   = {a_acc:.3f}  (gap-to-N=1 closed: {(a_acc - 0.651)/(0.924 - 0.651) * 100:.0f}%)")
    print(f"     Discrete = {d_acc:.3f}  (gap-to-N=1 closed: {(d_acc - 0.651)/(0.924 - 0.651) * 100:.0f}%)")


if __name__ == "__main__":
    main()
