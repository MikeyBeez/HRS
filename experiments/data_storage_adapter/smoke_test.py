"""Smoke test: train a rank-128 adapter on one Pip passage, evaluate retrieval.

Spec sanity check: 'After Part A's first adapter trains: passage 1's facts
retrieve correctly. If not, adapter training is broken.'
"""
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
    e0 = groups["groups"]["G1_pip_childhood"][0]
    print(f"Entry 0 fact: {e0['fact']}")
    print(f"  answer: {e0['answer']}")
    print(f"  train paras: {e0['paraphrases_train']}")
    print(f"  heldout paras: {e0['paraphrases_held_out']}")

    for rank in [32, 128]:
        print(f"\n[rank {rank}]")
        t0 = time.time()
        model, cfg, tok = load_base(rank, device)
        reset_lora(model)
        sources = make_training_sources([e0], tok, cfg.ctx_len, device)
        print(f"  sources: {len(sources)} sequences")

        # Eval baseline (LoRA off-equivalent: scale=0)
        base_eval = eval_retrieval(model, [e0], tok, device, seeds=(0, 1, 2),
                                   lora_scale=0.0)
        print(f"  base (scale=0) recall: {base_eval['mean_rate']:.3f}")

        info = train_adapter(model, sources, n_steps=200,
                              high_lr=5e-3, base_lr=1e-4, seed=0)
        print(f"  train: loss {info['loss_init']:.3f} -> {info['loss_mean_last10']:.3f}")

        adapter_eval = eval_retrieval(model, [e0], tok, device,
                                      seeds=(0, 1, 2), lora_scale=1.0)
        print(f"  adapter (scale=1) recall: {adapter_eval['mean_rate']:.3f}")
        print(f"  hits: {adapter_eval['hits']}/{adapter_eval['n']}")
        # Show one example generation
        from transformers import AutoTokenizer
        tokenizer = tok
        for p in e0["paraphrases_held_out"][:1]:
            ids = tokenizer.encode(p, add_special_tokens=False)[:cfg.ctx_len]
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            torch.manual_seed(42)
            from experiments.data_storage_adapter.harness import generate_completion
            out = generate_completion(model, ids_t, max_new_tokens=16,
                                      ctx_len=cfg.ctx_len, lora_scale=1.0)
            new_ids = out[0, ids_t.shape[1]:].tolist()
            print(f"  '{p}' -> '{tokenizer.decode(new_ids)}'")
        print(f"  wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
