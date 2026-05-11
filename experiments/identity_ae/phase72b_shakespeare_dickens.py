"""Phase 72b — Adapter-aware base training, Shakespeare + Dickens.

Tests the same architectural mechanism as Phase 72 (base hosts adapter content
without absorbing it), but with two key changes designed to fix the failure
modes the original Phase 72 trajectory exposed:

  * BIGGER OOD DISTRIBUTION GAP. Base = TinyTransformer trained on Tiny
    Shakespeare (BPE, vocab 50257, val PPL 121). OOD = Great Expectations
    (Dickens) prose. Shares an alphabet and basic English grammar but
    differs substantially in vocabulary and sentence structure. Spillover
    damage during regularizer correction should be smaller than Phase 72's
    WikiText-vs-passkey setup, where "predict passkey digits" entangled
    heavily with "predict generic technical English."

  * REFINED REGULARIZER. Replaces tanh saturation with a one-sided hinge
    against a per-position cached pristine baseline:

        L_reg = mean over OOD positions of relu(baseline_ce[i] - L_det[i])
        L_joint = L_attached + L_reg

    Zero gradient when L_det >= baseline (the base has no incentive to push
    further into ignorance once it's at-or-above pristine). Positive gradient
    only when L_det drops below baseline (push the base back up). Per-position
    anchoring uses each token's own pristine difficulty, not a uniform scale.

Other tightening: base lr 1e-5 (10x lower than Phase 72's 1e-4), 200 steps
(adapter converged in ~25 in Phase 72; further steps mostly damaged the base).

Setup re-uses the BPE-tokenized Shakespeare/Dickens data already prepared
under experiments/router_lora_phased/data/. Base architecture is the
TinyTransformer from phase1_base.pt (lora_B=0 confirmed, FFN at block 4 is
a pure standard MLP), converted to a clean vanilla model by stripping the
unused LoRA-A buffer at block 4.

Checkpoints every 10 steps to results/identity_ae/phase72b/checkpoints/
(.pt files gitignored).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase72b_shakespeare_dickens.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.lora_wrapper import (
    LoRALayer, apply_lora,
)


SEED = 0
N_STEPS = 200
SHAKESPEARE_PER_OOD = 4
LR_BASE = 1e-5
LR_LORA = 1e-3
MAX_GRAD_NORM = 1.0
RANK = 128
ALPHA = 256
SHAKESPEARE_BATCH_SIZE = 4
SHAKESPEARE_SEQ_LEN = 256
LOG_INTERVAL = 10
CKPT_INTERVAL = 10
PHASE72B_DIR = Path("results/identity_ae/phase72b")
CKPT_DIR = PHASE72B_DIR / "checkpoints"

# OOD passage: opening of Chapter LI from Great Expectations (held-out chapters
# in router_lora_phased/data/dickens_eval_text.txt). Descriptive narrative
# prose typical of Dickens — long sentences, Victorian vocabulary.
OOD_PASSAGE = (
    "What purpose I had in view when I was hot on tracing out and proving "
    "Estella's parentage, I cannot say. It will presently be seen that the "
    "question was not before me in a distinct shape until it was put before "
    "me by a wiser head than my own."
)

# Continuation for the attached-retrieval test: feed the first sentence as
# prefix; measure attached vs detached CE on the next two sentences.
OOD_PREFIX = (
    "What purpose I had in view when I was hot on tracing out and proving "
    "Estella's parentage, I cannot say."
)
OOD_CONTINUATION = (
    " It will presently be seen that the question was not before me in a "
    "distinct shape until it was put before me by a wiser head than my own."
)

LORA_TARGETS = [
    "blocks.4.attn.qkv", "blocks.4.attn.out_proj",
    "blocks.4.ffn.0",    "blocks.4.ffn.2",
    "blocks.5.attn.qkv", "blocks.5.attn.out_proj",
    "blocks.5.ffn.0",    "blocks.5.ffn.2",
]


# ============================================================
# Vanilla TinyTransformer (no built-in LoRA at any layer)
# ============================================================

class CausalAttn(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.dropout(self.out_proj(out))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttn(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class VanillaTinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                  ctx_len, dropout=0.1):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


def load_vanilla_from_phase1(device):
    """Load phase1_base.pt (which has lora_A buffers but lora_B=0) and rename
    the state dict so it fits the vanilla VanillaTinyTransformer layout."""
    ckpt = torch.load("experiments/router_lora_phased/results/phase1_base.pt",
                       map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    # Drop the LoRA-only keys; rename block-4 ffn from fc1/fc2 -> 0/2
    sd_in = ckpt["model_state_dict"]
    sd_out = {}
    for k, v in sd_in.items():
        if k.endswith("blocks.4.ffn.lora_A") or k.endswith("blocks.4.ffn.lora_B"):
            continue
        if k == "blocks.4.ffn.fc1.weight": sd_out["blocks.4.ffn.0.weight"] = v
        elif k == "blocks.4.ffn.fc1.bias":   sd_out["blocks.4.ffn.0.bias"]   = v
        elif k == "blocks.4.ffn.fc2.weight": sd_out["blocks.4.ffn.2.weight"] = v
        elif k == "blocks.4.ffn.fc2.bias":   sd_out["blocks.4.ffn.2.bias"]   = v
        else:
            sd_out[k] = v

    model = VanillaTinyTransformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        ctx_len=cfg["ctx_len"], dropout=cfg["dropout"],
    ).to(device)
    missing, unexpected = model.load_state_dict(sd_out, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not missing, f"missing keys: {missing}"
    return model, cfg


# ============================================================
# LoRA active-flag patching (lets us toggle the adapter)
# ============================================================

_orig_lora_forward = LoRALayer.forward


def _patched_lora_forward(self, x):
    base_out = self.base_layer(x)
    if not getattr(self, "active", True):
        return base_out
    lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
    return base_out + lora_out


def patch_lora_class():
    LoRALayer.forward = _patched_lora_forward


def set_lora_active(model, active: bool):
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.active = active


# ============================================================
# Helpers
# ============================================================

def per_token_ce(model, ids_t):
    """Returns per-position cross-entropy for next-token prediction (length T-1)."""
    out = model(ids_t[:, :-1])
    targets = ids_t[:, 1:]
    log_probs = F.log_softmax(out, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.squeeze(0).tolist()


@torch.no_grad()
def shakespeare_val_ppl(model, val_tokens, n_seq=8, seq_len=SHAKESPEARE_SEQ_LEN, device="cuda"):
    """Mean PPL over a sample of n_seq sequences from validation tokens."""
    nll_total, tok_total = 0.0, 0
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    for i in range(n_seq):
        start = int(torch.randint(0, len(val_tokens) - seq_len - 1, (1,), generator=rng).item())
        seq = val_tokens[start:start + seq_len].unsqueeze(0).to(device)
        out = model(seq[:, :-1])
        nll = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                seq[:, 1:].reshape(-1), reduction="sum")
        nll_total += float(nll.item())
        tok_total += seq[:, 1:].numel()
    return math.exp(nll_total / max(tok_total, 1))


def continuation_ce(model, tokenizer, prefix, continuation, device):
    """Mean CE on continuation tokens given prefix as context."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    full_ids = prefix_ids + cont_ids
    full_t = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to(device)
    out = model(full_t[:, :-1])
    log_probs = F.log_softmax(out, dim=-1)
    targets = full_t[:, 1:]
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
    # CE on continuation positions only (offset accounting for the [:-1] cut)
    cont_start = len(prefix_ids) - 1
    cont_nll = nll[cont_start:].mean().item()
    return float(cont_nll)


def save_checkpoint(step, model, optim, pk_att, pk_det, ppl, model_cfg):
    path = CKPT_DIR / f"step_{step:04d}.pt"
    base_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" not in k}
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({
        "step": step,
        "ood_attached_continuation_ce": pk_att,
        "ood_detached_continuation_ce": pk_det,
        "shakespeare_val_ppl": ppl,
        "base_state_dict": base_state,
        "lora_state_dict": lora_state,
        "optim_state_dict": optim.state_dict() if optim is not None else None,
        "model_config": model_cfg,
        "lora_rank": RANK,
        "lora_alpha": ALPHA,
        "lora_targets": LORA_TARGETS,
    }, path)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE72B_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load Shakespeare base (vanilla form of phase1_base.pt) ----
    print(f"[A] loading Shakespeare base (BPE, d_model 256, n_layers 6)")
    model, cfg = load_vanilla_from_phase1(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {n_params:,} params, ctx_len {cfg['ctx_len']}, vocab {cfg['vocab_size']}")

    # ---- Tokenize OOD passage ----
    ood_ids = tokenizer.encode(OOD_PASSAGE, add_special_tokens=False)
    ood_t = torch.tensor(ood_ids, dtype=torch.long).unsqueeze(0).to(device)
    print(f"    OOD passage: {len(ood_ids)} tokens")
    print(f"    {OOD_PASSAGE!r}")

    # ---- Pre-training baselines ----
    with torch.no_grad():
        pristine_ce_per_pos = per_token_ce(model, ood_t)
    pristine_mean = sum(pristine_ce_per_pos) / len(pristine_ce_per_pos)
    print(f"    pristine OOD per-token mean CE: {pristine_mean:.3f} nats "
          f"(threshold for OOD validity: >= 5.0)")
    assert pristine_mean >= 5.0, "OOD passage not OOD enough for Shakespeare base"

    # Continuation baseline (used as the "adapter learns" success metric)
    pristine_cont_ce = continuation_ce(model, tokenizer, OOD_PREFIX, OOD_CONTINUATION, device)
    print(f"    pristine OOD continuation CE: {pristine_cont_ce:.3f} nats")

    # Shakespeare val tokens for general-competence tracking
    val_tokens = torch.load("experiments/router_lora_phased/data/shakespeare_val.pt",
                             map_location="cpu", weights_only=False)
    pristine_ppl = shakespeare_val_ppl(model, val_tokens, n_seq=8, device=device)
    print(f"    pristine Shakespeare-val PPL (8 seqs): {pristine_ppl:.3f}")

    # Cache per-position pristine CE as anchor (the regularizer's baseline)
    baseline_ce_t = torch.tensor(pristine_ce_per_pos, device=device)

    # ---- Apply LoRA, patch active flag ----
    print(f"\n[B] applying LoRA (rank {RANK}) to blocks 4-5 (8 modules), patching active flag")
    patch_lora_class()
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(model, True)
    print(f"    LoRA params: {n_lora:,}")

    # Unfreeze base
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(True)

    base_params = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    print(f"    base trainable: {sum(p.numel() for p in base_params):,}  "
          f"lora trainable: {sum(p.numel() for p in lora_params):,}")

    optim = torch.optim.AdamW(
        [{"params": base_params, "lr": LR_BASE},
         {"params": lora_params, "lr": LR_LORA}],
        weight_decay=0.0,
    )

    # ---- Load Shakespeare train tokens for protective batches ----
    train_tokens = torch.load("experiments/router_lora_phased/data/shakespeare_train.pt",
                                map_location="cpu", weights_only=False)
    print(f"    Shakespeare train tokens: {train_tokens.numel():,}")

    def sample_shakespeare_batch():
        starts = torch.randint(0, train_tokens.numel() - SHAKESPEARE_SEQ_LEN - 1,
                                (SHAKESPEARE_BATCH_SIZE,))
        return torch.stack([train_tokens[s:s + SHAKESPEARE_SEQ_LEN] for s in starts]).to(device)

    # ---- Step 0 checkpoint ----
    save_checkpoint(0, model, optim, pristine_cont_ce, pristine_cont_ce,
                     pristine_ppl, model_cfg=cfg)
    checkpoint_records = [(0, pristine_cont_ce, pristine_cont_ce, pristine_ppl)]

    # ---- Training loop ----
    print(f"\n[C] training {N_STEPS} steps, {SHAKESPEARE_PER_OOD}:1 shakespeare:ood, "
          f"base lr {LR_BASE}, lora lr {LR_LORA}")
    trajectory = {
        "step": [], "is_ood_step": [],
        "L_attached": [], "L_detached": [], "L_reg": [], "L_joint": [],
        "ood_attached_cont_ce": [], "ood_detached_cont_ce": [],
        "shakespeare_ppl": [],
    }
    composite_passes = []

    t0 = time.time()
    n_ood_steps = 0
    for step in range(N_STEPS):
        is_ood_step = (step % (SHAKESPEARE_PER_OOD + 1) == SHAKESPEARE_PER_OOD)
        model.train()
        optim.zero_grad()

        if is_ood_step:
            set_lora_active(model, True)
            out_att = model(ood_t[:, :-1])
            log_p_att = F.log_softmax(out_att, dim=-1)
            nll_att = -log_p_att.gather(-1, ood_t[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
            L_attached = nll_att.mean()

            set_lora_active(model, False)
            out_det = model(ood_t[:, :-1])
            log_p_det = F.log_softmax(out_det, dim=-1)
            nll_det = -log_p_det.gather(-1, ood_t[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
            set_lora_active(model, True)

            # One-sided hinge per position; mean across positions
            hinge = F.relu(baseline_ce_t - nll_det)
            L_reg = hinge.mean()
            L_joint = L_attached + L_reg

            L_joint.backward()
            torch.nn.utils.clip_grad_norm_(base_params + lora_params, MAX_GRAD_NORM)
            optim.step()
            n_ood_steps += 1
        else:
            set_lora_active(model, False)
            wt_batch = sample_shakespeare_batch()
            out = model(wt_batch[:, :-1])
            L = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                  wt_batch[:, 1:].reshape(-1))
            set_lora_active(model, True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(base_params + lora_params, MAX_GRAD_NORM)
            optim.step()

        # ---- Monitoring + checkpointing ----
        ckpt_step = (step + 1) if (step + 1) % CKPT_INTERVAL == 0 else None
        if ckpt_step is not None or step == 0:
            model.eval()
            with torch.no_grad():
                set_lora_active(model, True)
                cont_att = continuation_ce(model, tokenizer, OOD_PREFIX, OOD_CONTINUATION, device)
                set_lora_active(model, False)
                cont_det = continuation_ce(model, tokenizer, OOD_PREFIX, OOD_CONTINUATION, device)
                set_lora_active(model, True)
                ppl = shakespeare_val_ppl(model, val_tokens, n_seq=8, device=device)
                # Per-position detached CE on the OOD passage (for hinge inspection)
                set_lora_active(model, False)
                det_per_pos = per_token_ce(model, ood_t)
                set_lora_active(model, True)
                det_mean = sum(det_per_pos) / len(det_per_pos)

            trajectory["step"].append(step)
            trajectory["is_ood_step"].append(is_ood_step)
            trajectory["ood_attached_cont_ce"].append(cont_att)
            trajectory["ood_detached_cont_ce"].append(cont_det)
            trajectory["shakespeare_ppl"].append(ppl)
            if is_ood_step:
                trajectory["L_attached"].append(float(L_attached.item()))
                trajectory["L_detached"].append(float(nll_det.mean().item()))
                trajectory["L_reg"].append(float(L_reg.item()))
                trajectory["L_joint"].append(float(L_joint.item()))
            else:
                for k in ("L_attached", "L_detached", "L_reg", "L_joint"):
                    trajectory[k].append(None)

            # Composite check at this checkpoint
            adapter_works = cont_att < pristine_cont_ce - 0.5  # meaningfully below baseline
            det_within_1nat = abs(det_mean - pristine_mean) <= 1.0
            ppl_within_5pct = abs(ppl - pristine_ppl) / pristine_ppl <= 0.05
            composite = adapter_works and det_within_1nat and ppl_within_5pct

            if ckpt_step is not None:
                save_checkpoint(ckpt_step, model, optim, cont_att, cont_det, ppl, model_cfg=cfg)
                checkpoint_records.append((ckpt_step, cont_att, cont_det, ppl))
                if composite:
                    composite_passes.append(ckpt_step)

            ckpt_marker = " [ckpt]" if ckpt_step is not None else ""
            comp_marker = " ★COMPOSITE" if composite else ""
            dt = time.time() - t0
            eta = dt / (step + 1) * (N_STEPS - step - 1)
            print(f"    step {step+1:>3d}/{N_STEPS}  "
                  f"cont_att {cont_att:>5.2f}  cont_det {cont_det:>5.2f}  "
                  f"det_mean {det_mean:>5.2f}  ppl {ppl:>6.2f}  "
                  f"[ood {n_ood_steps}, {dt:.0f}s, eta {eta:.0f}s]{ckpt_marker}{comp_marker}")

    # ---- Final eval ----
    print(f"\n[D] final eval")
    model.eval()
    with torch.no_grad():
        set_lora_active(model, True)
        final_cont_att = continuation_ce(model, tokenizer, OOD_PREFIX, OOD_CONTINUATION, device)
        set_lora_active(model, False)
        final_cont_det = continuation_ce(model, tokenizer, OOD_PREFIX, OOD_CONTINUATION, device)
        set_lora_active(model, False)
        final_det_per_pos = per_token_ce(model, ood_t)
        set_lora_active(model, True)
        final_det_mean = sum(final_det_per_pos) / len(final_det_per_pos)
        final_ppl = shakespeare_val_ppl(model, val_tokens, n_seq=8, device=device)

    adapter_works = final_cont_att < pristine_cont_ce - 0.5
    det_within_1nat = abs(final_det_mean - pristine_mean) <= 1.0
    ppl_within_5pct = abs(final_ppl - pristine_ppl) / pristine_ppl <= 0.05
    composite_final = adapter_works and det_within_1nat and ppl_within_5pct

    print(f"    attached continuation CE : {final_cont_att:.3f}  (pristine {pristine_cont_ce:.3f}, "
          f"{'PASS' if adapter_works else 'FAIL'})")
    print(f"    detached continuation CE : {final_cont_det:.3f}  (pristine {pristine_cont_ce:.3f})")
    print(f"    detached mean CE on OOD  : {final_det_mean:.3f}  (pristine {pristine_mean:.3f}, "
          f"within-1nat={'PASS' if det_within_1nat else 'FAIL'})")
    print(f"    Shakespeare val PPL      : {final_ppl:.3f}  (pristine {pristine_ppl:.3f}, "
          f"within-5pct={'PASS' if ppl_within_5pct else 'FAIL'})")
    print(f"    final composite : {'PASS' if composite_final else 'FAIL'}")
    if composite_passes:
        print(f"    composite PASSED at checkpoints: {composite_passes}")
    else:
        print(f"    composite never passed at any checkpoint")

    # ---- Save JSON ----
    out_path = PHASE72B_DIR / "shakespeare_dickens.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_steps": N_STEPS, "shakespeare_per_ood": SHAKESPEARE_PER_OOD,
                "lr_base": LR_BASE, "lr_lora": LR_LORA,
                "rank": RANK, "alpha": ALPHA, "lora_targets": LORA_TARGETS,
                "shakespeare_seq_len": SHAKESPEARE_SEQ_LEN,
                "shakespeare_batch_size": SHAKESPEARE_BATCH_SIZE,
                "ood_passage_token_count": len(ood_ids),
                "regularizer": "one_sided_hinge_per_position",
                "seed": SEED,
            },
            "ood_passage": OOD_PASSAGE,
            "ood_prefix": OOD_PREFIX,
            "ood_continuation": OOD_CONTINUATION,
            "pristine": {
                "ood_per_pos_ce": pristine_ce_per_pos,
                "ood_mean_ce": pristine_mean,
                "ood_continuation_ce": pristine_cont_ce,
                "shakespeare_val_ppl": pristine_ppl,
            },
            "final": {
                "attached_cont_ce": final_cont_att,
                "detached_cont_ce": final_cont_det,
                "detached_mean_ce": final_det_mean,
                "shakespeare_val_ppl": final_ppl,
                "adapter_works": adapter_works,
                "det_within_1nat": det_within_1nat,
                "ppl_within_5pct": ppl_within_5pct,
                "composite": composite_final,
            },
            "trajectory": trajectory,
            "composite_passes": composite_passes,
            "predictions": {
                "adapter_learns":              {"P": 0.70, "outcome": adapter_works},
                "detached_within_1nat":        {"P": 0.55, "outcome": det_within_1nat},
                "shakespeare_within_5pct":     {"P": 0.50, "outcome": ppl_within_5pct},
                "composite_at_some_checkpoint": {"P": 0.30, "outcome": len(composite_passes) > 0},
                "composite_3_consecutive":     {"P": 0.15, "outcome":
                    any(all(s + 10 * i in composite_passes for i in range(3))
                        for s in composite_passes)},
            },
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    steps = trajectory["step"]
    axes[0].plot(steps, trajectory["ood_attached_cont_ce"], color="C2", marker="o", markersize=3, label="attached")
    axes[0].plot(steps, trajectory["ood_detached_cont_ce"], color="C3", marker="s", markersize=3, label="detached")
    axes[0].axhline(pristine_cont_ce, color="C0", ls="--", lw=1, label=f"pristine ({pristine_cont_ce:.2f})")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("continuation CE (nats)")
    axes[0].set_title("OOD continuation CE (lower = adapter works)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, trajectory["shakespeare_ppl"], color="C0", marker="o", markersize=3)
    axes[1].axhline(pristine_ppl, color="black", ls="--", lw=1, label=f"pristine ({pristine_ppl:.2f})")
    axes[1].axhline(pristine_ppl * 1.05, color="gray", ls=":", lw=0.6, label="±5%")
    axes[1].axhline(pristine_ppl * 0.95, color="gray", ls=":", lw=0.6)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("Shakespeare val PPL")
    axes[1].set_title("Shakespeare competence")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    ood_steps = [s for s, j in zip(steps, trajectory["L_joint"]) if j is not None]
    L_att = [v for v in trajectory["L_attached"] if v is not None]
    L_det = [v for v in trajectory["L_detached"] if v is not None]
    L_reg = [v for v in trajectory["L_reg"] if v is not None]
    L_jnt = [v for v in trajectory["L_joint"] if v is not None]
    axes[2].plot(ood_steps, L_att, color="C2", marker="o", markersize=3, label="L_attached")
    axes[2].plot(ood_steps, L_det, color="C3", marker="s", markersize=3, label="L_detached")
    axes[2].plot(ood_steps, L_reg, color="C1", marker="^", markersize=3, label="L_reg (hinge)")
    axes[2].plot(ood_steps, L_jnt, color="C0", marker="d", markersize=3, label="L_joint")
    axes[2].axhline(pristine_mean, color="gray", ls=":", lw=0.6, label=f"pristine baseline {pristine_mean:.2f}")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("loss value")
    axes[2].set_title("OOD-batch loss components")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = PHASE72B_DIR / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
