"""Phase 2D — discrete per-layer routing (standalone, skips Taylor which already ran)."""
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
    apply_multi_lora, set_active_adapters, reset_multi_lora, MultiLoRALayer,
)
from experiments.k2_crossterms.run_phase2_no_retrain import (
    _layer_index_from_name, set_per_layer_adapters, generate, check_match,
    GEN_TOKENS, RANK, ALPHA, N_PAIRS, PAIR_SEED,
)


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())

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

    lora_layers_present = sorted({
        _layer_index_from_name(n)
        for n, m in model.named_modules() if isinstance(m, MultiLoRALayer)
        if _layer_index_from_name(n) is not None
    })
    print(f"LoRA-augmented layers: {lora_layers_present}")
    assert len(lora_layers_present) >= 2, "discrete routing needs at least 2 LoRA layers"

    adapter_sds = {}
    for e in keys:
        sd = torch.load(REPO / e["sd_path"], map_location=device, weights_only=False)
        adapter_sds[e["id"]] = sd

    held_out_per_adapter = {}
    for entry in library:
        held_out_per_adapter[entry["id"]] = [
            {"probe": q, "answer": entry["answer"],
             "fact_type": entry["fact_type"]}
            for q in entry["paraphrases_held_out"]
        ]

    rng = random.Random(PAIR_SEED)
    aids = sorted(adapter_sds.keys())
    pairs = []
    while len(pairs) < N_PAIRS:
        i, j = rng.sample(aids, 2)
        pairs.append((i, j))

    L_target, L_companion = lora_layers_present[0], lora_layers_present[1]
    print(f"Layer assignment: L{L_target} = target, L{L_companion} = companion")

    d_results = []
    t0 = time.time()
    for pi, (i, j) in enumerate(pairs):
        sd_i, sd_j = adapter_sds[i], adapter_sds[j]
        for target_aid, companion_aid, sd_t, sd_c in [
            (i, j, sd_i, sd_j), (j, i, sd_j, sd_i)
        ]:
            layer_to_sd = {L_target: sd_t, L_companion: sd_c}
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
                        "target_layer": L_target, "companion_layer": L_companion,
                    })
        if (pi + 1) % 10 == 0 or pi == 0:
            sf = sum(r["retrieved"] for r in d_results) / max(1, len(d_results))
            print(f"  discrete pair {pi+1:2d}/{N_PAIRS}  running mean = {sf:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    d_acc = sum(r["retrieved"] for r in d_results) / len(d_results)
    d_per_type = defaultdict(list)
    for r in d_results:
        d_per_type[r["fact_type"]].append(r["retrieved"])
    print(f"\nDISCRETE retrieval = {d_acc:.3f}")
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

    # Merge with prior Taylor result into combined summary
    taylor_acc = 0.520
    taylor_per_type = {"entity": None, "numeric": None, "place": None,
                       "relation": 0.037}  # only relation visible in monitor; recompute
    # Recompute Taylor per-type from CSV
    taylor_csv = EXP / "results/phase2a_taylor.csv"
    if taylor_csv.exists():
        from collections import defaultdict as dd
        per_t = dd(list); total = []
        with taylor_csv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                per_t[row["fact_type"]].append(int(row["retrieved"]))
                total.append(int(row["retrieved"]))
        taylor_acc = sum(total)/len(total)
        taylor_per_type = {ft: sum(v)/len(v) for ft, v in per_t.items()}

    summary = {
        "phase": "2_no_retrain",
        "taylor": {
            "overall": taylor_acc,
            "per_fact_type": taylor_per_type,
        },
        "discrete": {
            "overall": d_acc,
            "per_fact_type": {ft: sum(h)/len(h) for ft, h in d_per_type.items()},
            "target_layer": L_target,
            "companion_layer": L_companion,
        },
        "context": {
            "phase1_n1": 0.924,
            "phase1_n2_vanilla": 0.651,
            "lora_layers": lora_layers_present,
        },
        "n_pairs": N_PAIRS, "pair_seed": PAIR_SEED,
        "wall_total_s": time.time() - t0,
    }
    (EXP / "results/phase2_no_retrain_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSaved {EXP/'results/phase2_no_retrain_summary.json'}")
    print(f"\n  vs N=1 baseline (0.924) and N=2 vanilla (0.651):")
    print(f"     Taylor   = {taylor_acc:.3f}  ({(taylor_acc - 0.651):+.3f} vs vanilla)")
    print(f"     Discrete = {d_acc:.3f}  ({(d_acc - 0.651):+.3f} vs vanilla)")


if __name__ == "__main__":
    main()
