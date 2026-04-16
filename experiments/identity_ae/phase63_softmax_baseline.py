"""Phase 63: Pre-train a standard softmax transformer for comparison.

Standard dot-product softmax attention + dense MLP FFN. Same d_model,
n_layers, n_heads, tokenizer, data, and training schedule as V22.
No PEER, no Sinkhorn, no learned exponential kernel.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase63_softmax_baseline.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from data import load_wikitext, build_dataloaders

# ================================================================
# Standard Transformer Components
# ================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, bias=False, max_seq_len=512):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(T)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, bias=False,
                 max_seq_len=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, bias,
                                         max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len=512, dropout=0.1, bias=False):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, bias, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ================================================================
# Training
# ================================================================

def get_lr(step, warmup, max_steps, base_lr):
    if step < warmup:
        return base_lr * step / warmup
    if step >= max_steps:
        return base_lr * 0.1
    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("results/identity_ae/phase63")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Match V22 config
    D_MODEL = 1024
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 4096
    MAX_SEQ = 512
    DROPOUT = 0.1
    BIAS = False
    VOCAB = 50257
    MAX_STEPS = 43000
    LR = 3e-4
    WARMUP = 1000
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
    WEIGHT_DECAY = 0.1
    MAX_GRAD_NORM = 1.0
    LOG_INTERVAL = 100
    EVAL_INTERVAL = 1000
    SAVE_INTERVAL = 5000

    random.seed(42)
    torch.manual_seed(42)

    print("Phase 63: Standard softmax transformer pre-training")
    print(f"Device: {device}")

    # Data
    splits, tokenizer = load_wikitext("wikitext/wikitext-103-raw-v1", MAX_SEQ)
    loaders = build_dataloaders(splits, BATCH_SIZE)

    # Model
    model = StandardTransformer(VOCAB, D_MODEL, N_HEADS, N_LAYERS, D_FF,
                                 MAX_SEQ, DROPOUT, BIAS).to(device)
    n_params = model.param_count()
    print(f"Model params: {n_params:,}")
    print(f"  (V22 PEER model: 511,901,438 — this is {n_params/511901438:.0%})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Training
    train_iter = iter(loaders["train"])
    step = 0
    micro_step = 0
    accum_loss = 0.0
    best_val_ppl = float("inf")
    t0 = time.time()

    print(f"\nTraining: {MAX_STEPS} steps")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Params: {n_params:,}\n")

    optimizer.zero_grad()
    model.train()

    while step < MAX_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        x, y = batch[0].to(device), batch[1].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            logits = model(x)
            V = logits.shape[-1]
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                    y[:, :-1].reshape(-1)) / GRAD_ACCUM

        loss.backward()
        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            base_lr = get_lr(step, WARMUP, MAX_STEPS, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr

            optimizer.step()
            optimizer.zero_grad()
            step += 1
            accum_loss += loss.item() * GRAD_ACCUM
        else:
            continue

        if step % LOG_INTERVAL == 0:
            avg = accum_loss / LOG_INTERVAL
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - t0
            print(f"step {step:6d}  loss {avg:.4f}  ppl {ppl:.1f}  "
                  f"lr {base_lr:.2e}  ({elapsed:.0f}s)")
            accum_loss = 0.0

        if step % EVAL_INTERVAL == 0:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in loaders["validation"]:
                    if n_val >= 20:
                        break
                    x, y = batch[0].to(device), batch[1].to(device)
                    logits = model(x)
                    V = logits.shape[-1]
                    val_loss += F.cross_entropy(
                        logits[:, :-1].reshape(-1, V),
                        y[:, :-1].reshape(-1)).item()
                    n_val += 1
            val_ppl = math.exp(min(val_loss / n_val, 20))
            print(f"  val PPL: {val_ppl:.2f}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_ppl": val_ppl,
                    "config": {
                        "d_model": D_MODEL, "n_heads": N_HEADS,
                        "n_layers": N_LAYERS, "d_ff": D_FF,
                        "max_seq_len": MAX_SEQ, "dropout": DROPOUT,
                        "bias": BIAS, "vocab_size": VOCAB,
                    },
                }, run_dir / "best.pt")
                print(f"  ** New best: {val_ppl:.2f}")

            model.train()

        if step % SAVE_INTERVAL == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ppl": best_val_ppl,
            }, run_dir / f"checkpoint_{step}.pt")
            existing = sorted(run_dir.glob("checkpoint_*.pt"),
                              key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]:
                old.unlink()

    elapsed = time.time() - t0
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "val_ppl": best_val_ppl,
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "d_ff": D_FF,
            "max_seq_len": MAX_SEQ, "dropout": DROPOUT,
            "bias": BIAS, "vocab_size": VOCAB,
        },
    }, run_dir / "final.pt")

    print(f"\n{'='*60}")
    print(f"Phase 63 COMPLETE: {step} steps in {elapsed/3600:.1f}h")
    print(f"Best val PPL: {best_val_ppl:.2f}")
    print(f"Params: {n_params:,}")
    print(f"Results saved to {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
