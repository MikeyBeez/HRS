"""Build K=2 engrams from the existing 100-turn 16_pr (last-token L16,
prompt+response) bank. Two splits:

  Topic split (engram_2_separated):
    A = MILITARY: battles + generals + logistics + campaigns
        indices 0-24 (25 battles), 25-44 (20 generals),
                60-74 (15 logistics), 90-99 (10 campaigns)
        70 turns total
    B = POLITICAL+SOCIAL: political + social
        indices 45-59 (15 political), 75-89 (15 social)
        30 turns total

  Random split (engram_2_random_split):
    R1 = first half indices 0-49
    R2 = second half indices 50-99

For each pair, mean-pool the existing 16_pr engrams and compute
cosine(A, B) — the key diagnostic.

Predicted routing target per probe is determined by which split holds
more of the probe's manually-annotated relevant turn IDs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/k2_engrams"
PRIOR_ME = REPO / "experiments/multi_engram"
PRIOR_PRE = REPO / "experiments/prompt_vs_response_engrams"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.probes import (
    VAL_PROBES, TEST_PROBES, build_probe_records,
)

MILITARY_IDX = (
    list(range(0, 25))     # battles
    + list(range(25, 45))  # generals
    + list(range(60, 75))  # logistics
    + list(range(90, 100)) # campaigns
)
POLSOC_IDX = (
    list(range(45, 60))    # political
    + list(range(75, 90))  # social
)
assert len(MILITARY_IDX) + len(POLSOC_IDX) == 100
assert set(MILITARY_IDX).isdisjoint(set(POLSOC_IDX))

R1_IDX = list(range(0, 50))
R2_IDX = list(range(50, 100))


def main():
    print("Loading 16_pr engrams from prior experiment ...")
    eng_npz = np.load(PRIOR_PRE / "data/engrams_v2.npz")
    bank = eng_npz["16_pr"]   # (100, 4096) — last-token L16, prompt+response
    print(f"  bank shape: {bank.shape}")

    # ---- Topic split ----
    eng_A = bank[MILITARY_IDX].mean(axis=0)
    eng_B = bank[POLSOC_IDX].mean(axis=0)
    a_t = torch.tensor(eng_A); b_t = torch.tensor(eng_B)
    cos_AB = F.cosine_similarity(a_t.unsqueeze(0),
                                   b_t.unsqueeze(0)).item()
    print(f"\nTopic-split engrams:")
    print(f"  Engram A (military, {len(MILITARY_IDX)} turns)")
    print(f"  Engram B (political+social, {len(POLSOC_IDX)} turns)")
    print(f"  cos(A, B) = {cos_AB:.3f}")
    print(f"  (compare: 100-engram bank pairwise mean cos = 0.472 — "
          f"key diagnostic)")

    # ---- Random split ----
    eng_R1 = bank[R1_IDX].mean(axis=0)
    eng_R2 = bank[R2_IDX].mean(axis=0)
    r1_t = torch.tensor(eng_R1); r2_t = torch.tensor(eng_R2)
    cos_R1R2 = F.cosine_similarity(r1_t.unsqueeze(0),
                                      r2_t.unsqueeze(0)).item()
    print(f"\nRandom-split engrams:")
    print(f"  Engram R1 (turns 0-49)")
    print(f"  Engram R2 (turns 50-99)")
    print(f"  cos(R1, R2) = {cos_R1R2:.3f}")

    # ---- Predicted routing target per probe ----
    test = build_probe_records(TEST_PROBES)
    predictions = []
    for p in test:
        rel = p["relevant_ids"]
        n_in_A = sum(1 for r in rel if r in MILITARY_IDX)
        n_in_B = sum(1 for r in rel if r in POLSOC_IDX)
        if n_in_A > n_in_B:
            target = "A"
        elif n_in_B > n_in_A:
            target = "B"
        else:
            target = "AMBIGUOUS"
        predictions.append({
            "probe_idx": test.index(p),
            "prompt": p["prompt"],
            "n_relevant_in_A": n_in_A,
            "n_relevant_in_B": n_in_B,
            "predicted_target": target,
        })

    print("\nPer-probe predicted target (based on relevant-turn split):")
    for pr in predictions:
        print(f"  [{pr['probe_idx']:2d}] target={pr['predicted_target']:<10s} "
              f"A={pr['n_relevant_in_A']:2d} B={pr['n_relevant_in_B']:2d}  "
              f"| {pr['prompt'][:55]}")

    # Save artifacts
    np.savez(
        EXP / "data/k2_engrams.npz",
        topic_A=eng_A, topic_B=eng_B,
        random_R1=eng_R1, random_R2=eng_R2,
    )
    out = {
        "cos_topic_AB": cos_AB,
        "cos_random_R1R2": cos_R1R2,
        "military_idx": MILITARY_IDX,
        "polsoc_idx": POLSOC_IDX,
        "random_split": {"R1": R1_IDX, "R2": R2_IDX},
        "predictions": predictions,
    }
    (EXP / "data/setup.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {EXP}/data/k2_engrams.npz")
    print(f"Saved {EXP}/data/setup.json")


if __name__ == "__main__":
    main()
