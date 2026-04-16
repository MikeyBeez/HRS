"""Phase 49: Structured preprocessing for attention input enrichment.

Tests whether providing metadata-enriched embeddings (content_type, POS,
entity_type, salience, query_type) to the main model's attention layers
improves V-space quality. The key metric is the 18% information recovery
floor from Application 2 (Phase 33/37) — if structured input helps, this
number rises.

Measurement battery:
  E1. V-space alignment (K-cosine at each layer) — compare Phase 32/32b
  E2. Information recovery floor (six prefix conditions, engram-as-cache)
      — compare Phase 33's 18% floor
  E3. Passkey retrieval with enriched engrams — compare Phase 47's 100/100/97

The tagger is frozen; only the MetadataEmbedder weights are trained.
The main model's base weights are also frozen — the enrichment is purely
additive to the existing token embeddings.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase49_enriched_input.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)
from experiments.identity_ae.preprocessing.metadata_embedder import (
    TaggerPipeline,
)

RANK = 128
ALPHA = 256
N_STEPS = 150
GEN_TOKENS = 50
LAYER = 5
N_PASSAGES = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200
HALF_LEN = 100
MAX_CTX_POS = 512


# ----------------------------------------------------------------
# Helpers for enriched forward pass
# ----------------------------------------------------------------

@torch.no_grad()
def enriched_embed(model, pipeline, ids_t, device):
    """Token embeddings + metadata embeddings (additive)."""
    tok_emb = model.drop(model.tok_emb(ids_t))        # (B, T, D)
    meta_emb = pipeline(ids_t)                         # (B, T, D)
    return tok_emb + meta_emb


@torch.no_grad()
def enriched_hidden_at_layer(model, pipeline, ids_t, layer, device):
    """Hidden states at a specific layer using enriched input."""
    h = enriched_embed(model, pipeline, ids_t, device)
    for i, block in enumerate(model.blocks):
        if i == layer:
            return h
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    return h


@torch.no_grad()
def enriched_l0_mean(model, pipeline, ids_t, device):
    """L0 mean-pooled enriched embedding."""
    h = enriched_embed(model, pipeline, ids_t, device)
    return h.mean(dim=1).squeeze(0).detach().cpu()


@torch.no_grad()
def raw_l0_mean(model, ids_t):
    """L0 mean (standard, no enrichment) for comparison."""
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


# ----------------------------------------------------------------
# Forward from segments (engram-as-cache, like Phase 33)
# ----------------------------------------------------------------

@torch.no_grad()
def forward_segments(model, pipeline, segments, device, enriched=False):
    """Forward from a list of segments.

    Each segment is one of:
      ("tokens", LongTensor[T])  -> enriched or raw embedding
      ("hidden", FloatTensor[D]) -> inject directly as a hidden position
    """
    parts = []
    for kind, x in segments:
        if kind == "tokens":
            ids = x.unsqueeze(0).to(device)
            if enriched and pipeline is not None:
                parts.append(enriched_embed(model, pipeline, ids, device))
            else:
                parts.append(model.drop(model.tok_emb(ids)))
        elif kind == "hidden":
            parts.append(x.view(1, 1, -1).to(device))
        else:
            raise ValueError(kind)
    h = torch.cat(parts, dim=1)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]
    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    return model.lm_head(h)


def segments_len(segments):
    return sum(len(x) if kind == "tokens" else 1 for kind, x in segments)


@torch.no_grad()
def continuation_nll(model, pipeline, prefix_segments, continuation_ids,
                     device, enriched=False):
    M = len(continuation_ids)
    full_segments = list(prefix_segments) + [("tokens", continuation_ids[:-1])]
    logits = forward_segments(model, pipeline, full_segments, device,
                              enriched=enriched)
    prefix_len = segments_len(prefix_segments)
    pred = logits[:, prefix_len:prefix_len + M - 1, :]
    target = continuation_ids[1:].unsqueeze(0).to(device)
    nll = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                          target.reshape(-1), reduction="mean")
    return float(nll)


@torch.no_grad()
def make_engram(model, pipeline, ids_t, device, enriched=False):
    """Mean-pooled hidden state for engram-as-cache injection."""
    if enriched and pipeline is not None:
        h = enriched_embed(model, pipeline, ids_t, device)
    else:
        h = model.drop(model.tok_emb(ids_t))
    # Run through to layer 5 for the engram
    for i, block in enumerate(model.blocks):
        if i == LAYER:
            break
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    return h.mean(dim=1).squeeze(0).detach()


# ----------------------------------------------------------------
# E1: K-space alignment
# ----------------------------------------------------------------

@torch.no_grad()
def measure_layer_alignment(model, pipeline, tokenizer, device, tests,
                            enriched=False):
    """Measure cosine between engram (mean) and per-layer hidden centroid.

    At each layer, the engram's cosine similarity to the actual hidden-state
    centroid of the passage measures how well the engram represents that
    layer's computation.  Phase 32/32b used this for K-space; here we use
    the full hidden state (which captures both K and V information).
    """
    results_per_layer = {l: [] for l in range(6)}

    for test in tests[:10]:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:256].unsqueeze(0).to(device)

        # Get input-level engram (L0 mean)
        if enriched and pipeline is not None:
            eng = enriched_embed(model, pipeline, ids_t, device)
        else:
            eng = model.drop(model.tok_emb(ids_t))
        eng_mean = eng.mean(dim=1).squeeze(0)  # (D,)

        # Forward through each layer, compare engram to hidden centroid
        h = eng.clone()
        for layer_idx, block in enumerate(model.blocks):
            if layer_idx >= 6:
                break
            # hidden centroid at this layer (BEFORE block processes it)
            h_mean = h.mean(dim=1).squeeze(0)
            sim = cosine(eng_mean.cpu(), h_mean.cpu())
            results_per_layer[layer_idx].append(sim)

            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)

    summary = {}
    for l in range(6):
        vals = results_per_layer[l]
        if vals:
            summary[l] = {"h_cos": sum(vals) / len(vals)}
    return summary


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase49")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tagger_path = Path("experiments/identity_ae/preprocessing/tagger_model.pt")
    if not tagger_path.exists():
        print(f"ERROR: tagger not found at {tagger_path}")
        print("Run train_tagger.py first.")
        return

    print("Loading models...")
    model, cfg = load_model(device)
    model.eval()
    reset_lora_to_zero(model)

    pipeline = TaggerPipeline(str(tagger_path), main_dim=1024,
                               meta_dim=128, device=str(device)).to(device)
    n_meta_params = sum(p.numel() for p in pipeline.embedder.parameters())
    print(f"  MetadataEmbedder params: {n_meta_params:,}")
    print(f"  alpha init: {pipeline.embedder.alpha.item():.3f}")

    tests = stratified_tests()
    print(f"  Stratified tests: {len(tests)}")

    # ==============================================================
    # E1: K-space alignment — raw vs enriched
    # ==============================================================
    print(f"\n{'='*60}")
    print("E1: K-space alignment (raw vs enriched)")
    print("=" * 60)

    raw_la = measure_layer_alignment(model, None, tokenizer, device, tests,
                                     enriched=False)
    enr_la = measure_layer_alignment(model, pipeline, tokenizer, device, tests,
                                     enriched=True)

    print(f"  {'layer':>5}  {'raw h-cos':>10}  {'enr h-cos':>10}")
    for l in range(6):
        if l in raw_la and l in enr_la:
            print(f"  {l:>5}  {raw_la[l]['h_cos']:>10.4f}  "
                  f"{enr_la[l]['h_cos']:>10.4f}")

    # ==============================================================
    # E2: Information recovery floor — raw vs enriched engrams
    # ==============================================================
    print(f"\n{'='*60}")
    print("E2: Information recovery floor (engram-as-cache, 50 passages)")
    print("=" * 60)

    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]

    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:N_PASSAGES].tolist()

    conditions = ["no_context", "full_context", "engram_only"]
    nll_raw = {c: [] for c in conditions}
    nll_enr = {c: [] for c in conditions}

    for ti, idx in enumerate(indices):
        item = val_ds[idx]
        ids = item[0] if isinstance(item, tuple) else item
        ids = ids[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        context_ids = ids[:CONTEXT_LEN]
        continuation = ids[CONTEXT_LEN:]

        # Raw engram
        eng_raw = make_engram(model, None,
                              context_ids.unsqueeze(0).to(device), device,
                              enriched=False)
        # Enriched engram
        eng_enr = make_engram(model, pipeline,
                              context_ids.unsqueeze(0).to(device), device,
                              enriched=True)

        # no_context (same for both)
        nc_nll = continuation_nll(model, None, [], continuation, device)
        nll_raw["no_context"].append(nc_nll)
        nll_enr["no_context"].append(nc_nll)

        # full_context — raw
        nll_raw["full_context"].append(
            continuation_nll(model, None, [("tokens", context_ids)],
                             continuation, device, enriched=False))
        # full_context — enriched
        nll_enr["full_context"].append(
            continuation_nll(model, pipeline, [("tokens", context_ids)],
                             continuation, device, enriched=True))

        # engram_only — raw
        nll_raw["engram_only"].append(
            continuation_nll(model, None, [("hidden", eng_raw)],
                             continuation, device))
        # engram_only — enriched
        nll_enr["engram_only"].append(
            continuation_nll(model, None, [("hidden", eng_enr)],
                             continuation, device))

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1}/{N_PASSAGES}]")

    def mean_nll(lst):
        return sum(lst) / max(len(lst), 1)

    raw_no = mean_nll(nll_raw["no_context"])
    raw_full = mean_nll(nll_raw["full_context"])
    raw_eng = mean_nll(nll_raw["engram_only"])
    enr_no = mean_nll(nll_enr["no_context"])
    enr_full = mean_nll(nll_enr["full_context"])
    enr_eng = mean_nll(nll_enr["engram_only"])

    raw_gap = raw_no - raw_full
    raw_recovery = (raw_no - raw_eng) / max(raw_gap, 1e-8)
    enr_gap = enr_no - enr_full
    enr_recovery = (enr_no - enr_eng) / max(enr_gap, 1e-8)

    print(f"\n  Raw embeddings:")
    print(f"    no_context NLL   : {raw_no:.4f}")
    print(f"    full_context NLL : {raw_full:.4f}")
    print(f"    engram_only NLL  : {raw_eng:.4f}")
    print(f"    gap (no - full)  : {raw_gap:.4f}")
    print(f"    recovery         : {raw_recovery:.1%}")

    print(f"\n  Enriched embeddings:")
    print(f"    no_context NLL   : {enr_no:.4f}")
    print(f"    full_context NLL : {enr_full:.4f}")
    print(f"    engram_only NLL  : {enr_eng:.4f}")
    print(f"    gap (no - full)  : {enr_gap:.4f}")
    print(f"    recovery         : {enr_recovery:.1%}")

    delta = enr_recovery - raw_recovery
    print(f"\n  Delta (enriched - raw): {delta:+.1%}")
    if delta > 0.02:
        print("  -> POSITIVE: enriched engrams recover more information")
    elif delta < -0.02:
        print("  -> NEGATIVE: enriched engrams recover less information")
    else:
        print("  -> NEUTRAL: no meaningful difference")

    # ==============================================================
    # E3: Passkey retrieval with enriched L0 routing
    # ==============================================================
    print(f"\n{'='*60}")
    print("E3: Passkey retrieval — enriched L0 routing vs raw L0")
    print("=" * 60)

    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    # Build library (standard protocol)
    print("  Building adapter library...")
    library = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                tokenizer, device,
                                n_steps=N_STEPS, high_lr=HIGH_LR,
                                base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone()
              for k, v in get_lora_state_dict(model).items()}
        library.append({"sd": sd, "test": dict(test),
                        "train_prompts": train_prompts})
        if (i + 1) % 5 == 0:
            print(f"    [{i+1:2d}/20] ({time.time()-t0:.0f}s)")

    # Extract keys — raw and enriched
    reset_lora_to_zero(model)
    raw_keys = []
    enr_keys = []
    for entry in library:
        rk = []
        ek = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            rk.append(raw_l0_mean(model, ids_t))
            ek.append(enriched_l0_mean(model, pipeline, ids_t, device))
        raw_keys.append(rk)
        enr_keys.append(ek)

    def route(query_key, library_keys):
        best_a, best_score = -1, -2.0
        for ai, keys in enumerate(library_keys):
            for kv in keys:
                s = cosine(query_key, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    # Test held-out paraphrase routing + retrieval
    for label, keys, use_enriched in [
        ("raw L0", raw_keys, False),
        ("enriched L0", enr_keys, True),
    ]:
        n_routed = 0
        n_retr = 0
        per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}

        for i, entry in enumerate(library):
            ho_paras = held_out_paraphrase(entry["test"])
            for para in ho_paras:
                reset_lora_to_zero(model)
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

                if use_enriched:
                    q = enriched_l0_mean(model, pipeline, ids_t, device)
                else:
                    q = raw_l0_mean(model, ids_t)

                best_a = route(q, keys)
                if best_a == i:
                    n_routed += 1

                sd = library[best_a]["sd"]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device,
                                      GEN_TOKENS)
                if check_passkey(gen, entry["test"]["passkey"]):
                    n_retr += 1
                    per_type[entry["test"]["type"]] += 1

        print(f"\n  {label}:")
        print(f"    routing:   {n_routed}/60 ({n_routed/60:.0%})")
        print(f"    retrieval: {n_retr}/60 ({n_retr/60:.0%})")
        print(f"    per type:  num={per_type['numeric']}/15  "
              f"ent={per_type['entity']}/15  "
              f"tech={per_type['technical']}/15  "
              f"fact={per_type['fact']}/15")

    # ==============================================================
    # Summary
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 49 SUMMARY")
    print("=" * 72)
    print(f"  MetadataEmbedder alpha: {pipeline.embedder.alpha.item():.3f}")
    print(f"  E2 recovery (raw):      {raw_recovery:.1%}")
    print(f"  E2 recovery (enriched): {enr_recovery:.1%}")
    print(f"  E2 delta:               {delta:+.1%}")

    out = {
        "e1_raw": {str(k): v for k, v in raw_la.items()},
        "e1_enriched": {str(k): v for k, v in enr_la.items()},
        "e2_raw": {"no": raw_no, "full": raw_full, "eng": raw_eng,
                   "recovery": raw_recovery},
        "e2_enriched": {"no": enr_no, "full": enr_full, "eng": enr_eng,
                        "recovery": enr_recovery},
        "e2_delta": delta,
    }
    with open(results_dir / "enriched_input.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
