"""Shared loader for trained Variant checkpoints + val data."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def load_checkpoint(variant: str, device: torch.device):
    from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig
    ckpt = torch.load(CKPT_DIR / f"variant_{variant}.pt", map_location=device,
                       weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


def load_val_loader(batch_size: int = 16):
    from data import load_wikitext
    splits, tok = load_wikitext("wikitext-2", seq_len=512)
    return DataLoader(splits["validation"], batch_size=batch_size,
                        shuffle=False, drop_last=True, num_workers=0), tok


@torch.no_grad()
def eval_ppl_with_T(model, val_loader, device, T: int | None = None,
                      n_batches: int = 40) -> float:
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(x[:, :-1], T=T) if T is not None else model(x[:, :-1])
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                    x[:, 1:].reshape(-1))
        losses.append(loss.item())
    return float(math.exp(float(np.mean(losses))))
