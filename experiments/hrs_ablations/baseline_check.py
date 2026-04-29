"""Quick baseline reproduction: routing-only on all 150 held-out queries.

Uses the canonical (L0_mean → W → cosine vs L5_aggregate) Phase 47 routing.
Should yield 150/150 correct.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.hrs_ablations.util import (
    PPD, D, load_baseline_model, get_tokenizer, hidden_at_embedding,
    load_library_and_adapters, held_out_queries,
)


def main():
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    model, cfg = load_baseline_model(device)
    library, keys, _ = load_library_and_adapters(device)

    # Load library L5 aggregate keys
    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    library_l5_n = F.normalize(library_l5, dim=-1)

    # Load projection
    proj_ck = torch.load(PPD / "results/projection_W.pt",
                         map_location=device, weights_only=False)
    W = nn.Linear(D, D, bias=False).to(device)
    W.load_state_dict(proj_ck["W_state"])
    W.eval()

    queries = held_out_queries(library)
    print(f"Held-out queries: {len(queries)}")

    n_correct = 0
    t0 = time.time()
    for q in queries:
        ids = tokenizer.encode(q["probe"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            l0 = hidden_at_embedding(model, ids_t).mean(dim=1).squeeze(0)
            proj = W(l0.unsqueeze(0))
            proj_n = F.normalize(proj, dim=-1)
            sim = (proj_n @ library_l5_n.T).squeeze(0)
            top = sim.argmax().item()
        if top == q["adapter_id"]:
            n_correct += 1
    print(f"Routing accuracy: {n_correct}/{len(queries)} = "
          f"{n_correct/len(queries):.3f}  wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
