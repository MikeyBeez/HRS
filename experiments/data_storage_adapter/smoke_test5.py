"""Smoke test 5: scale n_steps with training set size to keep visits/source
roughly constant. Find the right scaling for size=10."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.data_storage_adapter.harness import (
    load_base, reset_lora, train_adapter, eval_retrieval,
    make_training_sources,
)


def main():
    device = torch.device("cuda")
    groups = json.loads((REPO / "experiments/data_storage_adapter/data/groups.json").read_text())
    g1 = groups["groups"]["G1_pip_childhood"]

    HIGH_LR = 5e-3
    BASE_LR = 1e-4
    EVAL_KW = {"temperature": 0.6, "top_k": 20, "seeds": (0, 1, 2)}

    # At size=10 with rank 256, sweep step counts.
    for n_steps in [2000, 4000, 8000]:
        t0 = time.time()
        model, cfg, tok = load_base(256, device)
        reset_lora(model)
        sources = make_training_sources(g1[:10], tok, cfg.ctx_len, device)
        info = train_adapter(model, sources, n_steps=n_steps,
                              high_lr=HIGH_LR, base_lr=BASE_LR, seed=0)
        ev = eval_retrieval(model, g1[:10], tok, device, lora_scale=1.0,
                            **EVAL_KW)
        visits_per_src = n_steps / len(sources)
        print(f"rank=256 size=10 n_steps={n_steps} (~{visits_per_src:.0f} visits/src)  "
              f"loss {info['loss_init']:.2f}->{info['loss_mean_last10']:.2f}  "
              f"recall {ev['mean_rate']:.3f} ({ev['hits']}/{ev['n']})  "
              f"wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
