"""Shared utilities for HRS architecture ablations.

Reuses the Phase 47 / per_passage_dickens infrastructure:
  base model: V22 + Dickens-pretrained
  adapters:   experiments/per_passage_dickens/adapters/adapter_{000..049}.pt
  library:    experiments/per_passage_dickens/data/library.json

Provides: model loading, hidden_at_layer wrapper, engram pooling variants,
and eval helpers (routing-only and routing+retrieval).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)

PPD = REPO / "experiments/per_passage_dickens"
RANK = 128
D = 1024
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50


def load_baseline_model(device: torch.device, rank: int = RANK,
                        targets=None):
    """Load V22-Dickens base + LoRA structure (zeroed)."""
    if targets is None:
        targets = L45_TARGETS
    model, cfg = load_model(device)
    dickens_ck = torch.load(
        PPD / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=rank, alpha=rank * 2, target_modules=targets)
    reset_lora_to_zero(model)
    model.eval()
    return model, cfg


@torch.no_grad()
def hidden_at_embedding(model, ids_t):
    """L0 = post-embedding, pre-block-0 hidden (the original Phase 47 query)."""
    return model.drop(model.tok_emb(ids_t))


@torch.no_grad()
def get_hidden(model, ids_t, layer_spec: str):
    """layer_spec ∈ {'L0', 'L1', ..., 'L5', 'EMB'}.
    EMB / L0 = embedding-level (post tok_emb + drop, pre block-0)
    L1..L5 = output of that block index
    """
    if layer_spec in ("EMB", "L0"):
        return hidden_at_embedding(model, ids_t)
    idx = int(layer_spec[1:])
    return hidden_at_layer(model, ids_t, idx)


def pool(h: torch.Tensor, op: str, attn_query: torch.Tensor = None):
    """Pool (1, T, D) → (D,) per spec.
    op ∈ {'mean', 'max', 'attn'}
    For 'attn', attn_query is a learned (D,) vector; weights = softmax(h @ q).
    """
    if op == "mean":
        return h.mean(dim=1).squeeze(0)
    elif op == "max":
        return h.max(dim=1).values.squeeze(0)
    elif op == "attn":
        assert attn_query is not None
        scores = (h.squeeze(0) @ attn_query)  # (T,)
        weights = F.softmax(scores, dim=-1)   # (T,)
        return (weights.unsqueeze(-1) * h.squeeze(0)).sum(dim=0)  # (D,)
    else:
        raise ValueError(op)


@torch.no_grad()
def make_engram(model, ids_t, layer_spec: str, pool_op: str,
                attn_query: torch.Tensor = None):
    """Compute (D,) engram per (layer, pool) spec."""
    h = get_hidden(model, ids_t, layer_spec)
    return pool(h, pool_op, attn_query=attn_query)


def load_library_and_adapters(device, keys_path=None):
    """Returns library entries, library keys (with l5_aggregate), and an
    {adapter_id: state_dict} dict moved to device."""
    library = json.loads((PPD / "data/library.json").read_text())
    if keys_path is None:
        keys_path = PPD / "results/library_keys.json"
    keys = json.loads(Path(keys_path).read_text())
    adapter_sds = {}
    for e in keys:
        sd = torch.load(REPO / e["sd_path"], map_location="cpu",
                        weights_only=False)
        adapter_sds[e["id"]] = {k: v.to(device) for k, v in sd.items()}
    return library, keys, adapter_sds


def held_out_queries(library):
    out = []
    for entry in library:
        for q in entry["paraphrases_held_out"]:
            out.append({
                "adapter_id": entry["id"],
                "fact_type": entry["fact_type"],
                "probe": q,
                "answer": entry["answer"],
            })
    return out


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed,
             temperature=TEMPERATURE, top_k=TOP_K):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, generation):
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    if clean_a and clean_a in clean_g:
        return True
    return False


def get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")
