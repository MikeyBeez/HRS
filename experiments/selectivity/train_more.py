"""Train 4 additional baseline + selectivity adapters at λ=0.1 for
domains D (chocolate cake), E (pizza), F (soup), G (cookies).

Negatives for each new adapter = a mix of probes drawn from the OTHER 5
domains (10 per other domain × 5 = 50 total). Existing sel_A_lam0.1 and
sel_B_lam0.1 are reused as-is even though they were trained with paired
(single-domain) negatives — Tests 2/3 already showed their selectivity
generalizes.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict,
)
from experiments.selectivity.data import domain_data
from experiments.selectivity.train import (
    build_pos_sources, build_neg_sources, train_dual,
    RANK, ALPHA, N_STEPS,
)

PPD = REPO / "experiments/per_passage_dickens"
LAMBDA = 0.1


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = domain_data()

    print("Loading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    new_domains = ("D", "E", "F", "G")
    all_domains = ("A", "B", "D", "E", "F", "G")

    out_dir = REPO / "experiments/selectivity/adapters"
    train_log = json.loads(
        (REPO / "experiments/selectivity/results/train_log.json").read_text()
    )
    rng = random.Random(0)

    t_total = time.time()

    for d in new_domains:
        # Build positive: this domain's training data
        pos = build_pos_sources(data[d]["train"], tokenizer, device)

        # Build negative: mix of 10 probes from each OTHER domain
        neg_examples = []
        for other in all_domains:
            if other == d:
                continue
            sample = rng.sample(data[other]["train"], 10)
            neg_examples.extend(sample)
        neg = build_neg_sources(neg_examples, tokenizer, device)
        print(f"\n=== Domain {d} ({data[d]['name']}): pos={len(pos)} "
              f"neg={len(neg)} ===")

        # Baseline (positive only)
        for label, lam in [("baseline", 0.0), ("sel", LAMBDA)]:
            name = f"{label}_{d}" if label == "baseline" else f"sel_{d}_lam{LAMBDA}"
            print(f"  Training {name} (λ={lam}) ...")
            reset_lora_to_zero(model)
            t0 = time.time()
            history = train_dual(model, pos, neg if lam > 0 else [],
                                  N_STEPS, HIGH_LR, BASE_LR, lambda_=lam, seed=0)
            wall = time.time() - t0
            sd = {k: v.detach().cpu().clone()
                  for k, v in get_lora_state_dict(model).items()}
            torch.save(sd, out_dir / f"{name}.pt")
            print(f"    final: pos_loss={history[-1]['pos_loss']:.3f} "
                  f"kl={history[-1]['kl']:.3f}  wall={wall:.0f}s")
            train_log[name] = {
                "domain": d, "lambda": lam, "history": history,
                "wall_s": wall,
            }

    out_log_path = REPO / "experiments/selectivity/results/train_log.json"
    out_log_path.write_text(json.dumps(train_log, indent=2))
    print(f"\nTotal additional training wall: {time.time()-t_total:.0f}s")
    print(f"Saved {out_log_path}")


if __name__ == "__main__":
    main()
