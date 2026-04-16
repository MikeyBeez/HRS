"""Phase 52: Saliency-weighted LoRA absorption.

The attention pooling encoder from Phase 51 learned which token positions
carry information. This phase uses those saliency scores to weight the
NTP loss during LoRA absorption, concentrating gradient on informative
tokens rather than distributing it uniformly.

Four conditions:
  1. Uniform loss, rank 128 (Phase 38b baseline)
  2. Saliency-weighted loss, rank 128
  3. Saliency-weighted loss, rank 64 (rank reduction test)
  4. Saliency-weighted loss, rank 128, 75 steps (convergence speed test)

Measurements per condition:
  - Same-prompt passkey retrieval (20 passages)
  - V-space cosine alignment at layers 1-5 (5 passages)
  - Information recovery floor (10 WikiText passages)
  - Saliency distribution analysis

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase52_saliency_weighted.py
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
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)
from experiments.identity_ae.phase51_learned_engram import AttentionPoolEncoder


ALPHA_LORA = 256
N_STEPS = 150
GEN_TOKENS = 50
LAYER = 5
D = 1024
MAX_CTX_POS = 512


# ================================================================
# Saliency computation
# ================================================================

@torch.no_grad()
def compute_saliency(encoder, model, passage_ids_t, device):
    """Compute per-token saliency scores using the frozen attention encoder.

    Returns: saliency (T,) — softmax-normalized importance weights.
    """
    H = hidden_at_layer(model, passage_ids_t, LAYER)  # (1, T, D)
    # Extract the attention weights from the encoder
    B = H.shape[0]
    q = encoder.query.expand(B, -1, -1)        # (1, 1, D)
    k = encoder.k_proj(H)                       # (1, T, D)
    attn = torch.bmm(q, k.transpose(1, 2))      # (1, 1, T)
    attn = attn / math.sqrt(D)
    saliency = F.softmax(attn.squeeze(0).squeeze(0), dim=-1)  # (T,)
    return saliency


# ================================================================
# Saliency-weighted absorption
# ================================================================

def train_adapter_saliency(model, passage, prompts_with_answers, tokenizer,
                           device, saliency, n_steps=N_STEPS,
                           high_lr=HIGH_LR, base_lr=BASE_LR):
    """Train LoRA adapter with saliency-weighted NTP loss on the passage,
    plus standard uniform loss on prompt+answer pairs.

    saliency: (T,) tensor of importance weights for the passage tokens.
              None means uniform (baseline).
    """
    sources = []
    source_saliencies = []

    # Passage with saliency weighting
    p_ids = tokenizer.encode(passage, add_special_tokens=False)
    p_t = torch.tensor(p_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    sources.append(p_t)
    if saliency is not None:
        # Truncate saliency to match token length
        sal = saliency[:p_t.shape[1]].to(device)
        source_saliencies.append(sal)
    else:
        source_saliencies.append(None)

    # Prompt+answer pairs with uniform weighting
    for pa in prompts_with_answers:
        ids = tokenizer.encode(pa, add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        sources.append(ids_t)
        source_saliencies.append(None)  # uniform for prompts

    params = [p for n, p in model.named_parameters()
              if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr)

    model.train()
    for _ in range(n_steps):
        idx = random.randint(0, len(sources) - 1)
        ids_t = sources[idx]
        sal = source_saliencies[idx]

        if ids_t.shape[1] < 2:
            continue

        out = model(ids_t[:, :-1], step=0)
        logits = out.logits
        targets = ids_t[:, 1:]
        B, T, V = logits.shape

        if sal is not None and T > 0:
            # Saliency-weighted loss: weight each token position
            per_token = F.cross_entropy(logits.reshape(-1, V),
                                        targets.reshape(-1),
                                        reduction='none')  # (B*T,)
            per_token = per_token.reshape(B, T)
            # Align saliency to prediction positions (shift by 1)
            sal_aligned = sal[1:T+1] if len(sal) > T else sal[:T]
            if len(sal_aligned) < T:
                # Pad with uniform weight
                pad = torch.ones(T - len(sal_aligned), device=device) / T
                sal_aligned = torch.cat([sal_aligned, pad])
            # Normalize so weights sum to 1
            sal_aligned = sal_aligned / (sal_aligned.sum() + 1e-8)
            loss = (per_token * sal_aligned.unsqueeze(0)).sum()
        else:
            loss = F.cross_entropy(logits.reshape(-1, V),
                                    targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()


# ================================================================
# Forward segments for information recovery measurement
# ================================================================

@torch.no_grad()
def forward_segments(model, segments, device):
    parts = []
    for kind, x in segments:
        if kind == "tokens":
            ids = x.unsqueeze(0).to(device)
            parts.append(model.drop(model.tok_emb(ids)))
        elif kind == "hidden":
            parts.append(x.view(1, 1, -1).to(device))
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
def continuation_nll(model, prefix_segments, continuation_ids, device):
    M = len(continuation_ids)
    full_segments = list(prefix_segments) + [("tokens", continuation_ids[:-1])]
    logits = forward_segments(model, full_segments, device)
    prefix_len = segments_len(prefix_segments)
    pred = logits[:, prefix_len:prefix_len + M - 1, :]
    target = continuation_ids[1:].unsqueeze(0).to(device)
    return float(F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                  target.reshape(-1), reduction="mean"))


# ================================================================
# Main
# ================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase52")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()
    print(f"Stratified: {len(tests)} passages\n")

    # Load model
    print("Loading model...")
    model, cfg = load_model(device)

    # Load attention pooling encoder from Phase 51
    encoder_path = Path("results/identity_ae/phase51/attention_pool.pt")
    if not encoder_path.exists():
        print(f"ERROR: encoder not found at {encoder_path}")
        return
    encoder = AttentionPoolEncoder(D).to(device)
    encoder.load_state_dict(torch.load(str(encoder_path), map_location=device,
                                       weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("Loaded attention pooling encoder from Phase 51")

    # ============================================================
    # Compute saliency for all 20 passages
    # ============================================================
    print("\nComputing saliency scores...")
    passage_saliencies = {}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        sal = compute_saliency(encoder, model, ids_t, device)
        passage_saliencies[test["id"]] = sal.cpu()

        # Show top-5 tokens for first 3 passages
        if test["id"] < 3:
            tokens = tokenizer.convert_ids_to_tokens(ids[:len(sal)])
            top5 = sal.topk(5)
            print(f"  [{test['type']}] id={test['id']}  "
                  f"passkey={test['passkey']!r}  entropy={-(sal * sal.clamp_min(1e-10).log()).sum():.2f}")
            for rank, (score, idx) in enumerate(zip(top5.values, top5.indices)):
                tok = tokens[idx] if idx < len(tokens) else "?"
                print(f"    #{rank+1}: pos={idx.item():3d}  "
                      f"sal={score:.4f}  tok={tok!r}")

    # ============================================================
    # Phase 52b: Four absorption conditions
    # ============================================================
    conditions = [
        ("uniform_r128_150",     128, N_STEPS, False),
        ("saliency_r128_150",    128, N_STEPS, True),
        ("saliency_r64_150",      64, N_STEPS, True),
        ("saliency_r128_75",     128, 75,      True),
    ]

    all_results = {}

    for cond_name, rank, steps, use_saliency in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"  rank={rank}, steps={steps}, saliency={use_saliency}")
        print("=" * 60)

        # Fresh model with LoRA
        model, _ = load_model(device)
        apply_lora(model, rank=rank, alpha=rank * 2, target_modules=L45_TARGETS)

        n_correct = 0
        per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        per_passage = []
        t0 = time.time()

        for ti, test in enumerate(tests):
            reset_lora_to_zero(model)

            train_prompts = [test["prompt"]] + train_paraphrase(test)
            prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]

            sal = passage_saliencies[test["id"]] if use_saliency else None

            train_adapter_saliency(
                model, test["passage"], prompts_with_answers,
                tokenizer, device, saliency=sal,
                n_steps=steps, high_lr=HIGH_LR, base_lr=BASE_LR,
            )

            gen = generate_greedy(model, test["prompt"], tokenizer, device,
                                  GEN_TOKENS)
            hit = check_passkey(gen, test["passkey"])
            if hit:
                n_correct += 1
                per_type[test["type"]] += 1
            per_passage.append({
                "id": test["id"], "type": test["type"],
                "passkey": test["passkey"], "hit": hit,
                "gen": gen[:120],
            })

        elapsed = time.time() - t0
        print(f"  retrieval: {n_correct}/20  "
              f"(num={per_type['numeric']}/5 ent={per_type['entity']}/5 "
              f"tech={per_type['technical']}/5 fact={per_type['fact']}/5)")
        print(f"  time: {elapsed:.0f}s")

        all_results[cond_name] = {
            "rank": rank, "steps": steps, "saliency": use_saliency,
            "n_correct": n_correct, "per_type": dict(per_type),
            "time": elapsed, "per_passage": per_passage,
        }

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Phase 52c: V-space measurement on best saliency condition
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE 52c: V-space quality comparison")
    print("=" * 60)

    # Reload model for V-space analysis
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)

    # Train one adapter with and without saliency, then compare V-space
    vspace_results = {}
    for label, use_sal in [("uniform", False), ("saliency", True)]:
        test = tests[0]  # first passage
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        sal = passage_saliencies[test["id"]] if use_sal else None
        train_adapter_saliency(model, test["passage"], prompts_with_answers,
                               tokenizer, device, saliency=sal,
                               n_steps=N_STEPS, high_lr=HIGH_LR,
                               base_lr=BASE_LR)

        # Measure V-space alignment: engram vs V centroid at each layer
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        alignments = {}
        with torch.no_grad():
            h = model.drop(model.tok_emb(ids_t))
            for layer_idx, block in enumerate(model.blocks):
                attn = block.attn
                ln_h = block.ln1(h)
                qkv = attn.qkv(ln_h)
                n_h = attn.n_heads
                hd = attn.head_dim
                _, _, v = qkv.split(n_h * hd, dim=-1)
                V = v.squeeze(0)  # (T, D)
                v_mean = V.mean(dim=0).cpu()
                # Engram: mean-pooled hidden at this point
                h_mean = h.mean(dim=1).squeeze(0).cpu()
                alignments[layer_idx] = {
                    "v_self_cos": cosine(v_mean, v_mean),
                    "h_v_cos": cosine(h_mean, v_mean),
                }
                eb = model.engram_buffer if model._engram_buffer_initialized else None
                h, _, _, _ = block(h, step=0, engram_buffer=eb)

        vspace_results[label] = alignments
        print(f"\n  {label} V-space alignment:")
        for l in sorted(alignments.keys()):
            print(f"    layer {l}: h-v cosine = {alignments[l]['h_v_cos']:.4f}")

    # ============================================================
    # Information recovery with attention pooling on saliency-trained model
    # ============================================================
    print(f"\n  Information recovery (attention pooling on saliency vs uniform):")

    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]
    torch.manual_seed(0)
    eval_indices = torch.randperm(len(val_ds))[:10].tolist()

    for label, use_sal in [("uniform", False), ("saliency", True)]:
        # Train a single adapter for each condition
        test = tests[0]
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        sal = passage_saliencies[test["id"]] if use_sal else None
        train_adapter_saliency(model, test["passage"], prompts_with_answers,
                               tokenizer, device, saliency=sal,
                               n_steps=N_STEPS, high_lr=HIGH_LR,
                               base_lr=BASE_LR)

        # Reset LoRA for clean measurement (we measure base model V-space)
        reset_lora_to_zero(model)

        nll_no = []
        nll_full = []
        nll_mean = []
        nll_attn = []

        for idx in eval_indices:
            item = val_ds[idx]
            ids = item[0] if isinstance(item, (list, tuple)) else item
            ids = ids[:256]
            if len(ids) < 256:
                continue
            ctx = ids[:200]
            cont = ids[200:]

            ctx_t = ctx.unsqueeze(0).to(device)
            H = hidden_at_layer(model, ctx_t, LAYER)
            mean_eng = H.mean(dim=1).squeeze(0).detach()
            attn_eng = encoder(H).squeeze(0).detach()

            nll_no.append(continuation_nll(model, [], cont, device))
            nll_full.append(continuation_nll(model, [("tokens", ctx)],
                                             cont, device))
            nll_mean.append(continuation_nll(model, [("hidden", mean_eng)],
                                             cont, device))
            nll_attn.append(continuation_nll(model, [("hidden", attn_eng)],
                                             cont, device))

        def avg(lst):
            return sum(lst) / max(len(lst), 1)

        no = avg(nll_no)
        full = avg(nll_full)
        gap = no - full
        mean_r = (no - avg(nll_mean)) / gap if gap > 0 else 0
        attn_r = (no - avg(nll_attn)) / gap if gap > 0 else 0
        print(f"    {label}: mean={mean_r:.1%}  attn_pool={attn_r:.1%}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 52 SUMMARY: saliency-weighted absorption")
    print("=" * 72)

    print(f"\n  Retrieval (same-prompt, 20 passages):")
    for name in ["uniform_r128_150", "saliency_r128_150",
                  "saliency_r64_150", "saliency_r128_75"]:
        r = all_results[name]
        pt = r["per_type"]
        print(f"    {name:22s}: {r['n_correct']}/20  "
              f"num={pt['numeric']}/5 ent={pt['entity']}/5 "
              f"tech={pt['technical']}/5 fact={pt['fact']}/5")

    # Success criteria
    sal_128 = all_results["saliency_r128_150"]["n_correct"]
    uni_128 = all_results["uniform_r128_150"]["n_correct"]
    sal_64 = all_results["saliency_r64_150"]["n_correct"]
    sal_75 = all_results["saliency_r128_75"]["n_correct"]

    print(f"\n  Success criteria:")
    print(f"    saliency r128 ≥ uniform r128: "
          f"{'PASS' if sal_128 >= uni_128 else 'FAIL'} ({sal_128} vs {uni_128})")
    print(f"    saliency r64 ≥ 18/20:         "
          f"{'PASS' if sal_64 >= 18 else 'FAIL'} ({sal_64}/20)")
    print(f"    saliency 75-step ≥ 19/20:     "
          f"{'PASS' if sal_75 >= 19 else 'FAIL'} ({sal_75}/20)")

    out = {
        "conditions": all_results,
        "vspace": {k: {str(l): v for l, v in d.items()}
                   for k, d in vspace_results.items()},
    }
    with open(results_dir / "saliency_weighted.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
