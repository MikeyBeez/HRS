"""Smoke test 2: vary n_steps, LR, max_new_tokens to find a working config."""
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
    make_training_sources, generate_completion,
)


def show_per_para(model, entries, tokenizer, device, cfg, lora_scale=1.0):
    for e in entries:
        for p in e["paraphrases_held_out"]:
            ids = tokenizer.encode(p, add_special_tokens=False)[:cfg.ctx_len]
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            torch.manual_seed(42)
            out = generate_completion(model, ids_t, max_new_tokens=16,
                                      ctx_len=cfg.ctx_len, lora_scale=lora_scale)
            new_ids = out[0, ids_t.shape[1]:].tolist()
            print(f"    '{p}' -> '{tokenizer.decode(new_ids)}'")


def main():
    device = torch.device("cuda")
    groups = json.loads((REPO / "experiments/data_storage_adapter/data/groups.json").read_text())
    pip = groups["groups"]["G1_pip_childhood"][:5]

    # Try a few configs
    configs = [
        {"rank": 64,  "n_steps": 150, "high_lr": 5e-3, "base_lr": 1e-4},
        {"rank": 64,  "n_steps": 300, "high_lr": 5e-3, "base_lr": 1e-4},
        {"rank": 128, "n_steps": 300, "high_lr": 1e-2, "base_lr": 5e-4},
        {"rank": 128, "n_steps": 500, "high_lr": 5e-3, "base_lr": 1e-4},
    ]
    for c in configs:
        print(f"\n--- {c} ---")
        t0 = time.time()
        model, cfg, tok = load_base(c["rank"], device)
        reset_lora(model)
        sources = make_training_sources([pip[0]], tok, cfg.ctx_len, device)
        info = train_adapter(model, sources, n_steps=c["n_steps"],
                              high_lr=c["high_lr"], base_lr=c["base_lr"], seed=0)
        ev = eval_retrieval(model, [pip[0]], tok, device, seeds=(0, 1, 2),
                            lora_scale=1.0)
        print(f"  loss {info['loss_init']:.3f} -> {info['loss_mean_last10']:.3f}  "
              f"recall {ev['mean_rate']:.3f} ({ev['hits']}/{ev['n']})  "
              f"wall {time.time()-t0:.1f}s")
        show_per_para(model, [pip[0]], tok, device, cfg, lora_scale=1.0)


if __name__ == "__main__":
    main()
