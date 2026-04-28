"""Continuation-pretrain V22 (results/v22_learned_kernel/best.pt) on the full
Great Expectations corpus until Dickens-fluent. All base params trainable.

Save: experiments/per_passage_dickens/results/v22_dickens_base.pt
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model, generate_greedy
from transformers import AutoTokenizer


GE_TOKENS_PATH = REPO / "experiments/router_lora_phased/data"  # has dickens_lora/router/eval splits


def load_ge_tokens(device):
    """Concat all three GE splits to get the full corpus as one token tensor."""
    parts = []
    for name in ["dickens_lora", "dickens_router", "dickens_eval"]:
        t = torch.load(GE_TOKENS_PATH / f"{name}.pt", weights_only=False)
        parts.append(t)
    full = torch.cat(parts)
    n = full.shape[0]
    val_split = int(0.95 * n)
    return full[:val_split].to(device), full[val_split:].to(device)


def sample_batch(tokens, batch_size, seq_len, rng):
    n = tokens.shape[0]
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    x = torch.stack([tokens[s:s + seq_len] for s in starts])
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def lr_at(step, total, peak, warmup=100):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_lm(model, val_tokens, batch_size, seq_len, n_batches=10):
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(val_tokens, batch_size, seq_len, rng)
        out = model(x, step=0)
        l = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1))
        losses.append(l.item())
    model.train()
    return float(math.exp(np.mean(losses)))


def main():
    steps = 3000
    batch_size = 2        # reduced from 8: V22's PEER FFN OOMs at higher batch
    grad_accum = 4        # effective batch 8
    seq_len = 512
    lr = 3e-5             # low LR to retain general LM capability
    seed = 0

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda")
    model, cfg = load_model(device)
    print(f"V22 loaded, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    train_tok, val_tok = load_ge_tokens(device)
    print(f"GE corpus: train {train_tok.shape[0]:,} tokens, val {val_tok.shape[0]:,} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                              betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = sample_batch(train_tok, batch_size, seq_len, rng)
            out = model(x, step=0)
            loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1))
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 200 == 0 or step == 0:
            ppl = eval_lm(model, val_tok, batch_size, seq_len)
            history.append({"step": step + 1, "train_loss": float(loss.item()),
                            "ge_val_ppl": ppl})
            print(f"  step {step+1:5d}/{steps}  loss={loss.item():.3f}  "
                  f"GE_val_ppl={ppl:.2f}  elapsed={time.time()-t0:.0f}s")

    final_ppl = eval_lm(model, val_tok, batch_size, seq_len, n_batches=40)
    print(f"\n[Pretrain] DONE  GE_val_ppl={final_ppl:.3f}  wall={time.time()-t0:.0f}s")

    # Sanity: generate from prompts
    tok = AutoTokenizer.from_pretrained("gpt2")
    model.eval()
    print("\nGeneration samples post-pretraining:")
    for prompt in [
        "My father's family name being Pirrip, and my Christian name Philip,",
        "When I had returned home that evening, I sat by the fire",
        "Mr. Jaggers stood, according to his wont, before the",
    ]:
        out = generate_greedy(model, prompt, tok, device, n_tokens=50)
        print(f"  PROMPT: {prompt!r}")
        print(f"  OUTPUT: {out[:180]!r}\n")

    out_dir = REPO / "experiments/per_passage_dickens/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "ge_val_ppl": final_ppl,
        "history": history,
        "wall_seconds": time.time() - t0,
    }, out_dir / "v22_dickens_base.pt")
    print(f"Saved {out_dir / 'v22_dickens_base.pt'}")


if __name__ == "__main__":
    main()
