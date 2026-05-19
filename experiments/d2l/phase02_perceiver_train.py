"""Phase 02 (D2L) — Perceiver training on Bleak House.

Trains a small Perceiver that ingests StarCoder2-3B hidden states on a Bleak
House passage and outputs a rank-8 LoRA adapter applied to every FFN
down-projection (mlp.c_proj) of the base. KL distillation objective:

  teacher = base(passage || query)            -> logits over answer tokens
  student = base_with_LoRA(query)             -> logits over answer tokens
  loss    = KL(softmax(teacher) || softmax(student)) over the answer.

The base is frozen in fp16. The Perceiver is fp32 with Adam. Gradients flow
loss → student logits → base FFN c_proj layers → LoRA(A, B) → Perceiver.

Run-scale calibration
---------------------
Spec asks for 20K steps with 30-50M Perceiver params and full 30-layer LoRA at
rank 8. On RTX 5070 Ti this is heavy. We:
  * Use latent_dim 512, 32 latents — ~25M Perceiver params (in spec band).
  * Default to 2000 steps (pilot/short run). Override with --steps for longer.
  * Length curriculum: 256 / 384 / 512 (vs spec's 256/512/1024) to control
    activation memory. 1024 OOMs at batch 1 on 16 GB without checkpointing.

Holdout: any training passage whose token span overlaps with a Phase 01
cloze item's source sentence is filtered out before training.

Usage
-----
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/d2l/phase02_perceiver_train.py [--steps 2000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Config
# ============================================================
MODEL_ID = "bigcode/starcoder2-3b"
LORA_RANK = 8
LORA_ALPHA = 4              # scaling=0.5 — keeps LoRA contribution inside fp16
SCALING = LORA_ALPHA / LORA_RANK     # 0.5
LATENT_N = 32
LATENT_D = 512
N_CROSS = 2
N_SELF = 4
ANSWER_LEN = 4                       # tokens predicted under KL
PILOT_STEPS = 2000


# ============================================================
# Perceiver
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
    """Produces rank-r LoRA A (in×r) and B (r×out) per layer from a pool of
    latents. Uses layer + rank-slot queries cross-attending into the latent pool.
    """

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
        # Output scaling: keep A/B small at init so initial LoRA is near zero.
        # Tight std on A; B is identically zero so the initial LoRA term is exactly 0.
        nn.init.normal_(self.proj_a.weight, std=0.005)
        nn.init.zeros_(self.proj_b.weight)
        nn.init.zeros_(self.proj_b.bias)
        nn.init.zeros_(self.proj_a.bias)

    def forward(self, latents):
        # latents: (1, N, latent_d)
        # Build (n_layers * rank, latent_d) of queries
        layer_ids = torch.arange(self.n_layers, device=latents.device)
        le = self.layer_embed(layer_ids)                       # (L, D)
        # Combine: per (layer, rank_slot) query.
        # queries shape: (L * R, D)
        queries = (le.unsqueeze(1) + self.rank_query.unsqueeze(0)).reshape(-1, latents.size(-1))
        q = self.ln_q(queries).unsqueeze(0)                    # (1, L*R, D)
        kv = self.ln_kv(latents)                                # (1, N, D)
        pooled, _ = self.pool_attn(q, kv, kv)                  # (1, L*R, D)
        pooled = pooled[0].view(self.n_layers, self.rank, -1)  # (L, R, D)
        a_cols = self.proj_a(pooled)                            # (L, R, dim_in)
        b_rows = self.proj_b(pooled)                            # (L, R, dim_out)
        # A shape per layer is (dim_in, rank) so transpose
        a_mats = a_cols.transpose(1, 2)                         # (L, dim_in, R)
        b_mats = b_rows                                          # (L, R, dim_out)
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
        # base_hidden: (1, T, H) — passage hidden states from frozen base
        latents = self.latents.expand(base_hidden.size(0), -1, -1)
        for blk in self.cross_blocks:
            latents = blk(latents, base_hidden)
        for blk in self.self_blocks:
            latents = blk(latents)
        return self.head(latents)


# ============================================================
# Base wrapping — install LoRA on c_proj
# ============================================================
class LoRAInjectedLinear(nn.Module):
    """Wraps an nn.Linear and adds LoRA outputs when A_buf/B_buf are set."""

    def __init__(self, base_linear, scaling):
        super().__init__()
        self.base_linear = base_linear
        self.scaling = scaling
        self.A = None       # (in_features, rank)
        self.B = None       # (rank, out_features)
        for p in base_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base_linear(x)
        if self.A is not None and self.B is not None:
            # x: (..., in_features). LoRA forward: x @ A @ B * scaling
            xA = x.to(self.A.dtype) @ self.A
            lora_out = (xA @ self.B) * self.scaling
            out = out + lora_out.to(out.dtype)
        return out


def install_lora_wrappers(base_model):
    """Replace mlp.c_proj on every layer with LoRAInjectedLinear. Returns the
    list of wrappers in layer order so we can assign A/B per forward pass."""
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
# Data pipeline
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
    """Tokenize the text once; iterate non-overlapping windows of length
    target_token_len. Drop windows whose decoded text overlaps a holdout
    sentence. Return list of token_id tensors."""
    holdout = load_holdout_sentences()
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    for s in range(0, len(ids) - target_token_len, target_token_len):
        chunk = ids[s:s + target_token_len]
        decoded = tokenizer.decode(chunk)
        # holdout filter: if any holdout-sentence-prefix appears in decoded, skip
        if any(h[:80] in decoded for h in holdout):
            continue
        out.append(chunk)
        if len(out) >= max_passages:
            break
    return out


PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,15})\b")


def extract_query_answer_pairs(passage_text, tokenizer, n=10, seed=0):
    """From a passage text, find proper nouns; build cloze items by masking
    each. Returns list of (prefix_text, answer_text) pairs."""
    rng = random.Random(seed)
    nouns = list(PROPER_NOUN_RE.finditer(passage_text))
    if not nouns:
        return []
    rng.shuffle(nouns)
    pairs = []
    for m in nouns[: n * 3]:
        # Skip very common false-positive starts ("The", "A", "I", names of months, ...)
        word = m.group(1)
        if word in {"The", "And", "But", "For", "In", "On", "At", "Of", "To", "By",
                     "When", "Where", "Why", "How", "What", "Who", "If", "It",
                     "He", "She", "We", "They", "I", "We", "Mr", "Mrs", "Sir",
                     "January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"}:
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
    """Run frozen base on the passage. Returns hidden states at the chosen
    layer (default: last). Shape (1, T, H)."""
    ids = torch.tensor([passage_ids], dtype=torch.long, device=device)
    out = base(ids, output_hidden_states=True, use_cache=False)
    if layer_idx is None:
        h = out.hidden_states[-1]
    else:
        h = out.hidden_states[layer_idx]
    return h


def kl_step(base, wrappers, perceiver, optimizer, scaler, tokenizer, passage_ids,
            prefix_text, answer_text, device, log=False):
    """One training step. Returns the KL loss as a float."""
    # 1. Encode passage and query
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)[:ANSWER_LEN]
    if not answer_ids:
        return None

    # 2. Extract passage hidden states (no grad) for Perceiver input
    passage_hidden = extract_passage_hidden(base, tokenizer, passage_ids, device)
    passage_hidden = passage_hidden.float()

    # 3. Perceiver forward
    A_mats, B_mats = perceiver(passage_hidden)
    # A_mats: (L, dim_in, R) ; B_mats: (L, R, dim_out)
    # Cast to fp16 for base forward
    A_list = [A_mats[i].to(torch.float16) for i in range(A_mats.size(0))]
    B_list = [B_mats[i].to(torch.float16) for i in range(B_mats.size(0))]

    # 4. Teacher forward: passage || query, no LoRA, no grad
    clear_lora(wrappers)
    teacher_input = passage_ids + prefix_ids + answer_ids
    teacher_ids = torch.tensor([teacher_input], dtype=torch.long, device=device)
    with torch.no_grad():
        teacher_out = base(teacher_ids, use_cache=False)
    teacher_logits = teacher_out.logits[0]
    # Answer positions in teacher: last ANSWER_LEN positions are the answer tokens; the
    # logits that PREDICT them sit at positions (len-ANSWER_LEN-1 .. len-2).
    ans_start = len(teacher_input) - len(answer_ids) - 1
    teacher_ans_logits = teacher_logits[ans_start: ans_start + len(answer_ids)]

    # 5. Student forward: query only with LoRA from Perceiver
    set_lora(wrappers, A_list, B_list)
    student_input = prefix_ids + answer_ids
    student_ids = torch.tensor([student_input], dtype=torch.long, device=device)
    student_out = base(student_ids, use_cache=False)
    student_logits = student_out.logits[0]
    ans_start_s = len(student_input) - len(answer_ids) - 1
    student_ans_logits = student_logits[ans_start_s: ans_start_s + len(answer_ids)]
    clear_lora(wrappers)

    # 6. KL loss: KL(teacher || student)
    t_lp = F.log_softmax(teacher_ans_logits.float(), dim=-1)
    s_lp = F.log_softmax(student_ans_logits.float(), dim=-1)
    t_p = t_lp.exp()
    kl = (t_p * (t_lp - s_lp)).sum(dim=-1).mean()

    # NaN/Inf guard. If the student logits overflowed (fp16), skip this step
    # rather than poison the Perceiver weights.
    if not torch.isfinite(kl):
        optimizer.zero_grad(set_to_none=True)
        return None

    # 7. Backward
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(kl).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(perceiver.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
    else:
        kl.backward()
        # Skip the optimizer step if any gradient is non-finite (catches the
        # tail end of an fp16 overflow that didn't poison kl itself yet).
        total_norm = torch.nn.utils.clip_grad_norm_(perceiver.parameters(), 0.5)
        if torch.isfinite(total_norm):
            optimizer.step()

    if log:
        # Print one teacher and student top-token at the first answer position.
        with torch.no_grad():
            t_top = tokenizer.decode([int(teacher_ans_logits[0].argmax().item())])
            s_top = tokenizer.decode([int(student_ans_logits[0].argmax().item())])
            print(f"   teacher top: {t_top!r}  student top: {s_top!r}  "
                  f"target: {tokenizer.decode([answer_ids[0]])!r}")

    return float(kl.detach().item())


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=PILOT_STEPS)
    ap.add_argument("--passage-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("results/d2l/phase02")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 02 — Perceiver training")
    print(f"  base: {MODEL_ID}")
    print(f"  steps: {args.steps}  passage_len: {args.passage_len}  "
          f"lr: {args.lr:.0e}  warmup: {args.warmup}")
    print()

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16,
    ).to(device).eval()
    for p in base.parameters():
        p.requires_grad = False
    base_hidden = base.config.hidden_size
    n_layers = base.config.num_hidden_layers
    ffn_in = base.config.intermediate_size
    ffn_out = base.config.hidden_size
    print(f"  base loaded in {time.time() - t_load:.1f}s  "
          f"hidden={base_hidden} layers={n_layers} ffn_in={ffn_in} ffn_out={ffn_out}")

    wrappers = install_lora_wrappers(base)
    print(f"  installed LoRA wrappers on {len(wrappers)} c_proj modules")

    perceiver = Perceiver(
        base_hidden=base_hidden,
        latent_n=LATENT_N,
        latent_d=LATENT_D,
        n_cross=N_CROSS,
        n_self=N_SELF,
        n_layers=n_layers,
        rank=LORA_RANK,
        dim_in=ffn_in,
        dim_out=ffn_out,
    ).to(device).float()
    n_perc = sum(p.numel() for p in perceiver.parameters())
    print(f"  Perceiver: {n_perc / 1e6:.1f}M params  "
          f"(latents {LATENT_N}x{LATENT_D}, cross {N_CROSS}, self {N_SELF})")

    optimizer = torch.optim.AdamW(perceiver.parameters(), lr=args.lr, betas=(0.9, 0.95))
    # Linear warmup, cosine decay
    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = None  # mixed precision off; base is already fp16, Perceiver fp32

    # Build training passages and queries
    text = load_text()
    print("  building training passages ...")
    passages = split_into_passages(text, tokenizer, args.passage_len, max_passages=2000)
    print(f"  {len(passages)} non-overlapping passages of {args.passage_len} tokens")

    # Pre-build queries per passage (lazy: do on demand to save memory)
    rng = random.Random(args.seed)

    log = []
    t_train = time.time()
    losses_window = []
    step = 0

    while step < args.steps:
        # Sample passage and query
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
        loss = kl_step(base, wrappers, perceiver, optimizer, scaler, tokenizer,
                       passage_ids, prefix, answer, device, log=do_log)
        scheduler.step()
        if loss is None:
            continue
        step += 1
        losses_window.append(loss)
        log.append({"step": step, "loss": loss, "lr": scheduler.get_last_lr()[0]})

        if do_log:
            recent = losses_window[-args.log_every:]
            elapsed = time.time() - t_train
            mean_recent = sum(recent) / len(recent)
            steps_per_s = step / max(1.0, elapsed)
            print(f"  step {step:6d}/{args.steps}  "
                  f"loss={loss:6.3f}  mean@{len(recent)}={mean_recent:6.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"{steps_per_s:.2f} step/s  elapsed {elapsed:.0f}s")

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
                },
            }
            torch.save(ckpt, results_dir / "perceiver_checkpoint.pt")
            print(f"  -> saved checkpoint at step {step}")

    # Save log + plot
    (results_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nTraining done in {time.time() - t_train:.0f}s")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        steps_arr = [r["step"] for r in log]
        losses_arr = [r["loss"] for r in log]
        ax.plot(steps_arr, losses_arr, alpha=0.3, label="step loss")
        # moving avg
        if len(losses_arr) >= 50:
            ma = []
            for i in range(len(losses_arr)):
                lo = max(0, i - 49)
                ma.append(sum(losses_arr[lo:i + 1]) / (i + 1 - lo))
            ax.plot(steps_arr, ma, color="crimson", linewidth=2, label="MA-50")
        ax.set_xlabel("step")
        ax.set_ylabel("KL(teacher || student)")
        ax.set_title("Phase 02 Perceiver training")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "training_curve.png", dpi=110)
        print("wrote training_curve.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
