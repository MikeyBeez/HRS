"""Phase 03 (D2L) — Perceiver training with 6x budget and fp16-stability fix.

Three changes from Phase 02:

1. Training horizon retargeted to 30K. The cosine schedule explicitly uses
   `args.steps` for the horizon, so passing --steps 30000 produces a 30K
   cosine, not a 5K cosine truncated.

2. fp32 LoRA math so alpha can stay at the spec's value of 16. Phase 02
   went NaN around step 300 with alpha=16 because the LoRA contribution
   (x @ A @ B * scaling) overflowed fp16 dynamic range inside the base
   FFN's c_proj output. Here the LoRA matrices live in fp32 (Perceiver is
   fp32 anyway) and the LoRA forward computes the residual in fp32 and
   only casts to fp16 for the final additive step into the frozen base.
   Extra activation memory is ~12 MB per layer per step at seq 256, well
   within budget on the 5070 Ti.

3. Trailing-window loss logger. The per-step KL on a single (passage,
   query) pair has natural variance from sub-token boundary cases (loss
   ~0 when teacher and student happen to agree exactly, loss ~10 when
   they disagree on the first token). A trailing mean over the last 200
   steps is what "is loss still descending" actually needs.

The Perceiver architecture, hypernet decoder, length curriculum, and
training-data holdout are the same as Phase 02.

Usage
-----
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/d2l/phase03_perceiver_train.py [--steps 30000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Config
# ============================================================
MODEL_ID = "bigcode/starcoder2-3b"
LORA_RANK = 8
LORA_ALPHA = 16
SCALING = LORA_ALPHA / LORA_RANK     # 2.0 (back to spec)
LATENT_N = 32
LATENT_D = 512
N_CROSS = 2
N_SELF = 4
ANSWER_LEN = 4
TRAILING_WINDOW = 200
DEFAULT_STEPS = 30000


# ============================================================
# Perceiver (same as Phase 02)
# ============================================================
class PerceiverBlockCross(nn.Module):
    def __init__(self, d, kv_d, heads=8):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(kv_d)
        self.attn = nn.MultiheadAttention(d, heads, kdim=kv_d, vdim=kv_d, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, latents, kv):
        h, _ = self.attn(self.ln_q(latents), self.ln_kv(kv), self.ln_kv(kv))
        latents = latents + h
        latents = latents + self.mlp(self.ln2(latents))
        return latents


class PerceiverBlockSelf(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, latents):
        h, _ = self.attn(self.ln1(latents), self.ln1(latents), self.ln1(latents))
        latents = latents + h
        latents = latents + self.mlp(self.ln2(latents))
        return latents


class LoRAHypernet(nn.Module):
    def __init__(self, latent_d, n_layers, rank, dim_in, dim_out, heads=8):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer_embed = nn.Embedding(n_layers, latent_d)
        self.rank_query = nn.Parameter(torch.randn(rank, latent_d) * 0.02)
        self.ln_q = nn.LayerNorm(latent_d)
        self.ln_kv = nn.LayerNorm(latent_d)
        self.pool_attn = nn.MultiheadAttention(latent_d, heads, batch_first=True)
        self.proj_a = nn.Linear(latent_d, dim_in)
        self.proj_b = nn.Linear(latent_d, dim_out)
        nn.init.normal_(self.proj_a.weight, std=0.005)
        nn.init.zeros_(self.proj_b.weight)
        nn.init.zeros_(self.proj_b.bias)
        nn.init.zeros_(self.proj_a.bias)

    def forward(self, latents):
        layer_ids = torch.arange(self.n_layers, device=latents.device)
        le = self.layer_embed(layer_ids)
        queries = (le.unsqueeze(1) + self.rank_query.unsqueeze(0)).reshape(-1, latents.size(-1))
        q = self.ln_q(queries).unsqueeze(0)
        kv = self.ln_kv(latents)
        pooled, _ = self.pool_attn(q, kv, kv)
        pooled = pooled[0].view(self.n_layers, self.rank, -1)
        a_cols = self.proj_a(pooled)
        b_rows = self.proj_b(pooled)
        a_mats = a_cols.transpose(1, 2)
        b_mats = b_rows
        return a_mats, b_mats


class Perceiver(nn.Module):
    def __init__(self, base_hidden, latent_n, latent_d, n_cross, n_self,
                 n_layers, rank, dim_in, dim_out):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, latent_n, latent_d) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [PerceiverBlockCross(latent_d, base_hidden) for _ in range(n_cross)]
        )
        self.self_blocks = nn.ModuleList(
            [PerceiverBlockSelf(latent_d) for _ in range(n_self)]
        )
        self.head = LoRAHypernet(latent_d, n_layers, rank, dim_in, dim_out)

    def forward(self, base_hidden):
        latents = self.latents.expand(base_hidden.size(0), -1, -1)
        for blk in self.cross_blocks:
            latents = blk(latents, base_hidden)
        for blk in self.self_blocks:
            latents = blk(latents)
        return self.head(latents)


# ============================================================
# fp32 LoRA injection — the Phase 03 stability fix
# ============================================================
class LoRAInjectedLinear(nn.Module):
    """Frozen base linear with an additive rank-r LoRA contribution. A and B
    matrices are stored as fp32; the LoRA forward computes in fp32 and casts
    only at the residual add, keeping the per-layer LoRA dynamic range safely
    inside fp32's headroom even at alpha=16.
    """

    def __init__(self, base_linear, scaling):
        super().__init__()
        self.base_linear = base_linear
        self.scaling = scaling
        self.A = None       # fp32, (in_features, rank)
        self.B = None       # fp32, (rank, out_features)
        for p in base_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base_linear(x)       # fp16
        if self.A is not None and self.B is not None:
            x32 = x.to(torch.float32)
            lora_out = (x32 @ self.A @ self.B) * self.scaling     # fp32
            out = out + lora_out.to(out.dtype)                     # cast for the add
        return out


def install_lora_wrappers(base_model):
    wrappers = []
    for i, layer in enumerate(base_model.model.layers):
        old = layer.mlp.c_proj
        w = LoRAInjectedLinear(old, SCALING)
        layer.mlp.c_proj = w
        wrappers.append(w)
    return wrappers


def set_lora(wrappers, A_mats, B_mats):
    for i, w in enumerate(wrappers):
        w.A = A_mats[i]
        w.B = B_mats[i]


def clear_lora(wrappers):
    for w in wrappers:
        w.A = None
        w.B = None


# ============================================================
# Data (same as Phase 02)
# ============================================================
def load_text():
    return Path("results/d2l/phase01/bleak_house.txt").read_text(encoding="utf-8")


def load_holdout_sentences():
    items = json.load(open("results/d2l/phase01/cloze_items.json"))
    out = set()
    for it in items:
        s = it.get("full_sentence_for_context", "")
        if s:
            out.add(s[:80])
    return out


def split_into_passages(text, tokenizer, target_token_len, max_passages=4000):
    holdout = load_holdout_sentences()
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    for s in range(0, len(ids) - target_token_len, target_token_len):
        chunk = ids[s:s + target_token_len]
        decoded = tokenizer.decode(chunk)
        if any(h[:80] in decoded for h in holdout):
            continue
        out.append(chunk)
        if len(out) >= max_passages:
            break
    return out


PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,15})\b")
_PRONOUNS_BLOCK = {"The", "And", "But", "For", "In", "On", "At", "Of", "To", "By",
                    "When", "Where", "Why", "How", "What", "Who", "If", "It",
                    "He", "She", "We", "They", "I", "Mr", "Mrs", "Sir",
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"}


def extract_query_answer_pairs(passage_text, tokenizer, n=10, seed=0):
    rng = random.Random(seed)
    nouns = list(PROPER_NOUN_RE.finditer(passage_text))
    if not nouns:
        return []
    rng.shuffle(nouns)
    pairs = []
    for m in nouns[: n * 3]:
        word = m.group(1)
        if word in _PRONOUNS_BLOCK:
            continue
        start = m.start(1)
        if start < 40:
            continue
        prefix = passage_text[:start].rstrip()
        answer = " " + word
        pairs.append((prefix, answer))
        if len(pairs) >= n:
            break
    return pairs


# ============================================================
# Training step
# ============================================================
@torch.no_grad()
def extract_passage_hidden(base, tokenizer, passage_ids, device, layer_idx=None):
    ids = torch.tensor([passage_ids], dtype=torch.long, device=device)
    out = base(ids, output_hidden_states=True, use_cache=False)
    if layer_idx is None:
        h = out.hidden_states[-1]
    else:
        h = out.hidden_states[layer_idx]
    return h


def kl_step(base, wrappers, perceiver, optimizer, tokenizer, passage_ids,
            prefix_text, answer_text, device, log=False):
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)[:ANSWER_LEN]
    if not answer_ids:
        return None

    passage_hidden = extract_passage_hidden(base, tokenizer, passage_ids, device)
    passage_hidden = passage_hidden.float()

    A_mats, B_mats = perceiver(passage_hidden)
    # fp32 LoRA — keep them in fp32, don't cast.
    A_list = [A_mats[i] for i in range(A_mats.size(0))]
    B_list = [B_mats[i] for i in range(B_mats.size(0))]

    clear_lora(wrappers)
    teacher_input = passage_ids + prefix_ids + answer_ids
    teacher_ids = torch.tensor([teacher_input], dtype=torch.long, device=device)
    with torch.no_grad():
        teacher_out = base(teacher_ids, use_cache=False)
    teacher_logits = teacher_out.logits[0]
    ans_start = len(teacher_input) - len(answer_ids) - 1
    teacher_ans_logits = teacher_logits[ans_start: ans_start + len(answer_ids)]

    set_lora(wrappers, A_list, B_list)
    student_input = prefix_ids + answer_ids
    student_ids = torch.tensor([student_input], dtype=torch.long, device=device)
    student_out = base(student_ids, use_cache=False)
    student_logits = student_out.logits[0]
    ans_start_s = len(student_input) - len(answer_ids) - 1
    student_ans_logits = student_logits[ans_start_s: ans_start_s + len(answer_ids)]
    clear_lora(wrappers)

    t_lp = F.log_softmax(teacher_ans_logits.float(), dim=-1)
    s_lp = F.log_softmax(student_ans_logits.float(), dim=-1)
    t_p = t_lp.exp()
    kl = (t_p * (t_lp - s_lp)).sum(dim=-1).mean()

    if not torch.isfinite(kl):
        optimizer.zero_grad(set_to_none=True)
        return None

    optimizer.zero_grad(set_to_none=True)
    kl.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(perceiver.parameters(), 0.5)
    if torch.isfinite(total_norm):
        optimizer.step()

    if log:
        with torch.no_grad():
            t_top = tokenizer.decode([int(teacher_ans_logits[0].argmax().item())])
            s_top = tokenizer.decode([int(student_ans_logits[0].argmax().item())])
            print(f"   teacher top: {t_top!r}  student top: {s_top!r}  "
                  f"target: {tokenizer.decode([answer_ids[0]])!r}",
                  flush=True)

    return float(kl.detach().item())


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--passage-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("results/d2l/phase03")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 03 — Perceiver training (30K-step rerun)", flush=True)
    print(f"  base: {MODEL_ID}", flush=True)
    print(f"  steps: {args.steps}  warmup: {args.warmup}  lr: {args.lr:.0e}", flush=True)
    print(f"  rank: {LORA_RANK}  alpha: {LORA_ALPHA}  scaling: {SCALING}", flush=True)
    print(f"  fp32 LoRA math: ON", flush=True)
    print()

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in base.parameters():
        p.requires_grad = False
    base_hidden = base.config.hidden_size
    n_layers = base.config.num_hidden_layers
    ffn_in = base.config.intermediate_size
    ffn_out = base.config.hidden_size
    print(f"  base loaded in {time.time() - t_load:.1f}s  "
          f"hidden={base_hidden} layers={n_layers}", flush=True)

    wrappers = install_lora_wrappers(base)
    print(f"  installed LoRA wrappers on {len(wrappers)} c_proj modules", flush=True)

    perceiver = Perceiver(
        base_hidden=base_hidden, latent_n=LATENT_N, latent_d=LATENT_D,
        n_cross=N_CROSS, n_self=N_SELF, n_layers=n_layers,
        rank=LORA_RANK, dim_in=ffn_in, dim_out=ffn_out,
    ).to(device).float()
    n_perc = sum(p.numel() for p in perceiver.parameters())
    print(f"  Perceiver: {n_perc / 1e6:.1f}M params", flush=True)

    optimizer = torch.optim.AdamW(perceiver.parameters(), lr=args.lr, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    text = load_text()
    print("  building training passages ...", flush=True)
    passages = split_into_passages(text, tokenizer, args.passage_len, max_passages=2000)
    print(f"  {len(passages)} non-overlapping passages of {args.passage_len} tokens",
          flush=True)

    rng = random.Random(args.seed)
    log = []
    trailing = deque(maxlen=TRAILING_WINDOW)
    t_train = time.time()
    step = 0
    skipped = 0
    consecutive_skips = 0
    max_consecutive_skips = 200    # hard stop if NaN cascade — no more silent burning
    total_iters = 0
    skip_log_every = 100           # print a heartbeat even when only skipping

    while step < args.steps:
        total_iters += 1
        p_idx = rng.randrange(len(passages))
        passage_ids = passages[p_idx]
        passage_text = tokenizer.decode(passage_ids)
        qa_pairs = extract_query_answer_pairs(
            passage_text, tokenizer, n=5, seed=rng.randrange(10**9),
        )
        if not qa_pairs:
            continue
        prefix, answer = qa_pairs[rng.randrange(len(qa_pairs))]

        do_log = (step % args.log_every == 0)
        loss = kl_step(base, wrappers, perceiver, optimizer, tokenizer,
                       passage_ids, prefix, answer, device, log=do_log)
        scheduler.step()
        if loss is None:
            skipped += 1
            consecutive_skips += 1
            # Heartbeat so a NaN cascade is visible immediately, not after an hour.
            if consecutive_skips % skip_log_every == 0:
                print(f"  [skip-burst] step {step}/{args.steps}  "
                      f"consecutive_skips={consecutive_skips}  total_skipped={skipped}  "
                      f"lr={scheduler.get_last_lr()[0]:.1e}",
                      flush=True)
            if consecutive_skips >= max_consecutive_skips:
                print(f"  ERROR: {consecutive_skips} consecutive NaN/Inf steps at "
                      f"step {step}/{args.steps}; halting to avoid silent burn.",
                      flush=True)
                break
            continue
        consecutive_skips = 0
        step += 1
        trailing.append(loss)
        log.append({"step": step, "loss": loss, "lr": scheduler.get_last_lr()[0]})

        if do_log:
            elapsed = time.time() - t_train
            trail_mean = sum(trailing) / len(trailing)
            steps_per_s = step / max(1.0, elapsed)
            eta_s = (args.steps - step) / max(0.1, steps_per_s)
            print(f"  step {step:6d}/{args.steps}  "
                  f"loss={loss:6.3f}  trail@{len(trailing)}={trail_mean:6.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"{steps_per_s:.2f} step/s  elapsed {elapsed:.0f}s  eta {eta_s:.0f}s  "
                  f"skipped {skipped}",
                  flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "perceiver_state_dict": perceiver.state_dict(),
                "config": {
                    "latent_n": LATENT_N, "latent_d": LATENT_D,
                    "n_cross": N_CROSS, "n_self": N_SELF,
                    "rank": LORA_RANK, "alpha": LORA_ALPHA,
                    "n_layers": n_layers, "ffn_in": ffn_in, "ffn_out": ffn_out,
                    "base_hidden": base_hidden, "model_id": MODEL_ID,
                    "fp32_lora": True,
                },
            }
            torch.save(ckpt, results_dir / "perceiver_checkpoint.pt")
            print(f"  -> saved checkpoint at step {step}", flush=True)

    (results_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nTraining done in {time.time() - t_train:.0f}s "
          f"(skipped {skipped} NaN/Inf steps)", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        steps_arr = [r["step"] for r in log]
        losses_arr = [r["loss"] for r in log]
        ax.plot(steps_arr, losses_arr, alpha=0.15, label="step loss", color="steelblue")
        if len(losses_arr) >= TRAILING_WINDOW:
            ma = []
            for i in range(len(losses_arr)):
                lo = max(0, i - TRAILING_WINDOW + 1)
                ma.append(sum(losses_arr[lo:i + 1]) / (i + 1 - lo))
            ax.plot(steps_arr, ma, color="crimson", linewidth=2.2,
                    label=f"trailing mean ({TRAILING_WINDOW})")
        ax.set_xlabel("step")
        ax.set_ylabel("KL(teacher || student)")
        ax.set_title("Phase 03 Perceiver training (30K, α=16, fp32 LoRA)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "training_curve.png", dpi=110)
        print("wrote training_curve.png", flush=True)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
