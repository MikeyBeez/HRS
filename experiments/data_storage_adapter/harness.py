"""Adapter training harness for the data-as-storage experiment.

Reuses the BPE tiny Shakespeare base from router_lora_phased
(6L/256d/4h/ctx512, GPT-2 BPE, val_ppl 121.5). Builds a fresh model with
configurable LoRA rank, loads the pretrained base weights, freezes everything
except the layer-4 LoRA matrices, and trains them from scratch on a passage +
prompt+answer corpus.

Public API:
    load_base(rank, device) -> (model, cfg, tokenizer)
    reset_lora(model)
    train_adapter(model, sources, n_steps, high_lr, base_lr) -> dict
    eval_retrieval(model, entries, tokenizer, device, seeds) -> dict
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.router_lora_phased.model import TinyTransformer, TinyConfig

BASE_CKPT = REPO / "experiments/router_lora_phased/results/phase1_base.pt"


def load_base(rank: int, device: torch.device):
    """Build a TinyTransformer with the given LoRA rank and load pretrained
    base weights (excluding lora_A/lora_B which start fresh at this rank).

    Returns (model, cfg, tokenizer).
    """
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg_d = dict(ckpt["model_config"])
    cfg_d["lora_rank"] = rank
    cfg = TinyConfig(**cfg_d)

    model = TinyTransformer(cfg).to(device)
    sd = {k: v for k, v in ckpt["model_state_dict"].items()
          if "lora_A" not in k and "lora_B" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # missing should be only lora_A / lora_B (we kept them at fresh init).
    assert all(("lora_A" in m or "lora_B" in m) for m in missing), \
        f"unexpected missing keys: {missing}"
    assert len(unexpected) == 0, f"unexpected keys: {unexpected}"

    # Freeze base; train only LoRA parameters.
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n or "lora_B" in n)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, cfg, tokenizer


def reset_lora(model):
    """Reset LoRA matrices: lora_A ~ N(0, 0.02), lora_B = 0."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.normal_(std=0.02)
            elif "lora_B" in n:
                p.zero_()


def lora_state_dict(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if "lora_A" in n or "lora_B" in n}


def load_lora_state(model, sd, device):
    cur = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in sd.items():
            cur[n].data.copy_(v.to(device))


def encode_prompts(prompts: list[str], tokenizer, ctx_len: int, device):
    """Tokenize a batch of strings to a list of (1, T) tensors."""
    out = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if len(ids) < 2:
            continue
        ids = ids[:ctx_len]
        out.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return out


def train_adapter(model, sources, n_steps: int, high_lr: float = 5e-3,
                  base_lr: float = 1e-4, seed: int = 0):
    """Train the LoRA adapter on a list of pre-tokenized (1, T) sources.

    Each step: pick one source uniformly, run LM next-token CE.
    StepLR halves at n_steps/2.
    """
    rng = random.Random(seed)
    params = [p for n, p in model.named_parameters()
              if ("lora_A" in n or "lora_B" in n)]
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
        logits, _ = model(ids_t[:, :-1], lora_scale=1.0)
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
        "n_steps": n_steps, "loss_init": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_mean_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
    }


def make_training_sources(entries: list[dict], tokenizer, ctx_len: int,
                          device) -> list[torch.Tensor]:
    """For each entry, build:
        - the passage tokens
        - each (training paraphrase + answer) tokens
    All as (1, T) tensors on device. Returns flat list across all entries.
    """
    out = []
    for e in entries:
        out.extend(encode_prompts([e["passage"]], tokenizer, ctx_len, device))
        for p in e["paraphrases_train"]:
            joined = f"{p}{e['answer']}"
            out.extend(encode_prompts([joined], tokenizer, ctx_len, device))
    return out


@torch.no_grad()
def generate_completion(model, prompt_ids, max_new_tokens: int, ctx_len: int,
                         temperature: float = 0.8, top_k: int = 50,
                         lora_scale: float = 1.0):
    """Stochastic generation. prompt_ids: (1, T) tensor on device."""
    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        ctx = ids[:, -ctx_len:]
        logits, _ = model(ctx, lora_scale=lora_scale)
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


def normalize(s: str) -> str:
    return s.strip().lower().replace(",", " ").replace("  ", " ")


def substring_hit(generated: str, answer: str) -> bool:
    g = normalize(generated)
    a = normalize(answer)
    if not a:
        return False
    return a in g


@torch.no_grad()
def eval_retrieval(model, entries: list[dict], tokenizer, device,
                   seeds=(0, 1, 2), max_new_tokens: int = 16,
                   temperature: float = 0.8, top_k: int = 50,
                   lora_scale: float = 1.0):
    """For each entry × held-out paraphrase × seed, generate and check
    substring match for the answer. Return mean hit rate, per-entry hits.
    """
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
                                          cfg.ctx_len, temperature, top_k,
                                          lora_scale=lora_scale)
                # Decode the *new* tokens only.
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
