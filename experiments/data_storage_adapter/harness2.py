"""Multi-layer-LoRA harness: wraps blocks 4-5 attention + FFN linears with the
identity_ae LoRALayer wrapper. Bypasses TinyTransformer's built-in single-layer
LoRA by always calling forward with lora_scale=0 (no contribution from the
built-in LoRAMLP path).

Public API matches harness.py:
    load_base(rank, device) -> (model, cfg, tokenizer)
    reset_lora(model)
    train_adapter(model, sources, n_steps, high_lr, base_lr) -> dict
    eval_retrieval(model, entries, tokenizer, device, seeds) -> dict
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.router_lora_phased.model import TinyTransformer, TinyConfig
from experiments.identity_ae.lora_wrapper import (
    apply_lora as _apply_lora,
    reset_lora as _reset_lora_wrapper,
)

BASE_CKPT = REPO / "experiments/router_lora_phased/results/phase1_base.pt"

# Multi-layer LoRA targets on tiny transformer: attention + FFN of blocks 4-5.
# Note: block 4 ffn is LoRAMLP (fc1/fc2). Block 5 ffn is Sequential ([0], [2]).
LORA_TARGETS = [
    "blocks.4.attn.qkv",
    "blocks.4.attn.out_proj",
    "blocks.4.ffn.fc1",
    "blocks.4.ffn.fc2",
    "blocks.5.attn.qkv",
    "blocks.5.attn.out_proj",
    "blocks.5.ffn.0",
    "blocks.5.ffn.2",
]


def load_base(rank: int, device: torch.device):
    """Build TinyTransformer, load base weights, then wrap blocks 4-5 with
    multi-layer LoRA at the requested rank. Built-in LoRAMLP path is bypassed
    by always passing lora_scale=0 in forward.
    """
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = TinyConfig(**ckpt["model_config"])
    model = TinyTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    n_lora = _apply_lora(model, rank=rank, alpha=rank * 2,
                         target_modules=LORA_TARGETS)
    model = model.to(device)

    # Freeze the built-in LoRAMLP params on block 4 — we bypass that path with
    # lora_scale=0 and only train the identity_ae-wrapped LoRAs.
    for n, p in model.named_parameters():
        if n in ("blocks.4.ffn.lora_A", "blocks.4.ffn.lora_B"):
            p.requires_grad = False

    n_train = sum(p.numel() for n, p in model.named_parameters()
                  if p.requires_grad)
    assert n_train == n_lora, f"trainable={n_train} != lora={n_lora}"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, cfg, tokenizer


def reset_lora(model):
    """Reset LoRA matrices: lora_A ~ N(0, 0.02), lora_B = 0.
    Matches the harness1 (single-layer) init scale; the wrapper's default is
    0.01 which converges too slowly here.
    """
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.normal_(std=0.02)
            elif "lora_B" in n:
                p.zero_()


def lora_state_dict(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if "lora_" in n}


def load_lora_state(model, sd, device):
    cur = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in sd.items():
            cur[n].data.copy_(v.to(device))


def encode_prompts(prompts, tokenizer, ctx_len, device):
    out = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if len(ids) < 2:
            continue
        ids = ids[:ctx_len]
        out.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return out


def make_training_sources(entries, tokenizer, ctx_len, device):
    out = []
    for e in entries:
        out.extend(encode_prompts([e["passage"]], tokenizer, ctx_len, device))
        for p in e["paraphrases_train"]:
            joined = f"{p}{e['answer']}"
            out.extend(encode_prompts([joined], tokenizer, ctx_len, device))
    return out


def train_adapter(model, sources, n_steps, high_lr=3e-4, base_lr=1e-4, seed=0):
    rng = random.Random(seed)
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    opt = torch.optim.Adam(params, lr=high_lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, n_steps // 2), gamma=base_lr / high_lr,
    )
    model.train()
    losses = []
    for step in range(n_steps):
        ids_t = sources[rng.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        # Built-in LoRAMLP gets lora_scale=0 (bypassed); wrapped LoRAs are
        # always on through the identity_ae LoRALayer.forward.
        logits, _ = model(ids_t[:, :-1], lora_scale=0.0)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids_t[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
    model.eval()
    return {
        "n_steps": n_steps,
        "loss_init": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_mean_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
    }


@torch.no_grad()
def generate_completion(model, prompt_ids, max_new_tokens, ctx_len,
                         temperature=0.6, top_k=20):
    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        ctx = ids[:, -ctx_len:]
        logits, _ = model(ctx, lora_scale=0.0)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
            thresh = v[:, [-1]]
            logits = torch.where(logits < thresh,
                                 torch.full_like(logits, -float("inf")),
                                 logits)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def normalize(s):
    return s.strip().lower().replace(",", " ").replace("  ", " ")


def substring_hit(generated, answer):
    g = normalize(generated)
    a = normalize(answer)
    if not a:
        return False
    return a in g


@torch.no_grad()
def eval_retrieval(model, entries, tokenizer, device, seeds=(0, 1, 2),
                    max_new_tokens=16, temperature=0.6, top_k=20):
    cfg = model.cfg
    per_entry = []
    total = 0
    hits = 0
    for e in entries:
        ent_hits = 0
        ent_total = 0
        for p in e["paraphrases_held_out"]:
            ids = tokenizer.encode(p, add_special_tokens=False)[:cfg.ctx_len]
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            for s in seeds:
                torch.manual_seed(int(s) + 1000 * e["id"])
                out = generate_completion(model, ids_t, max_new_tokens,
                                          cfg.ctx_len, temperature, top_k)
                new_ids = out[0, ids_t.shape[1]:].tolist()
                gen_text = tokenizer.decode(new_ids)
                hit = substring_hit(gen_text, e["answer"])
                ent_total += 1
                total += 1
                if hit:
                    ent_hits += 1
                    hits += 1
        per_entry.append({
            "id": e["id"], "answer": e["answer"], "hits": ent_hits,
            "total": ent_total, "rate": ent_hits / max(1, ent_total),
        })
    return {
        "mean_rate": hits / max(1, total),
        "n": total, "hits": hits,
        "per_entry": per_entry,
    }
