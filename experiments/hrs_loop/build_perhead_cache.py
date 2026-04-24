"""Precompute V18 layer-4 hidden states for each engram-store entry.

Produces `engram_store_data/engram_hiddens_v18_layer4.pt`, a dict
{"lengths": LongTensor[870], "hiddens": Float16Tensor[870, max_len, 1024]}.

Per-entry hidden states are the extract-layer (layer 4) output on the
entry's full text. These are used by task 9's attention-pooled per-head
engram computation at training/eval time — the pooling re-projects with
the current W_k, W_v, so training can update the projections.

Run once:
    python experiments/hrs_loop/build_perhead_cache.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer
from engram_store import EngramStore

EXTRACT_LAYER = 4
INV_SOFTPLUS_1 = math.log(math.e - 1.0)
OUT_PATH = REPO / "engram_store_data" / "engram_hiddens_v18_layer4.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt_path = REPO / "results/v18_cross_attn/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    patched = 0
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_scalar") and any(name == m for m in missing):
            with torch.no_grad():
                p.fill_(INV_SOFTPLUS_1)
            patched += 1
    print(f"Loaded V18 (val_ppl {ckpt.get('val_ppl', float('nan')):.2f}). "
          f"missing={len(missing)} unexpected={len(unexpected)} gate_scalar_patched={patched}")
    model.eval()

    store = EngramStore.load(str(REPO / "engram_store_data"))
    print(f"Store: {len(store)} entries")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Capture hidden states at layer 4 via forward hook
    capture = {}

    def hook(module, _inputs, outputs):
        capture["h"] = outputs[0].detach().half().cpu()

    handle = model.blocks[EXTRACT_LAYER].register_forward_hook(hook)

    lengths: list[int] = []
    hiddens: list[torch.Tensor] = []
    max_len = 512  # cap matches V18 pretrain seq_len

    with torch.no_grad():
        for i, e in enumerate(store.entries):
            ids = tokenizer.encode(e.text, add_special_tokens=False)[:max_len]
            x = torch.tensor([ids], device=device, dtype=torch.long)
            _ = model(x, step=0)
            h = capture["h"][0]  # (L, D) fp16 cpu
            lengths.append(h.shape[0])
            hiddens.append(h)
            if (i + 1) % 100 == 0 or i == len(store.entries) - 1:
                print(f"  {i+1}/{len(store.entries)}  avg_len={sum(lengths)/len(lengths):.1f}")

    handle.remove()

    n = len(hiddens)
    max_obs = max(lengths)
    D = hiddens[0].shape[-1]
    print(f"\nPacking {n} entries into ({n}, {max_obs}, {D}) fp16 padded tensor")
    padded = torch.zeros((n, max_obs, D), dtype=torch.float16)
    for i, h in enumerate(hiddens):
        padded[i, :lengths[i]] = h

    torch.save({"lengths": torch.tensor(lengths, dtype=torch.long),
                "hiddens": padded}, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
