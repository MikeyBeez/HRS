"""Evaluation suite for topic-routed context assembly.

Metrics:
1. Held-out perplexity (does topic context help predict real continuations?)
2. Topic coherence (does generation stay on-topic with the prompt?)
3. Repetition/degeneration (does the model parrot context?)
4. Conditional perplexity by routing quality (does routing confidence predict benefit?)
6. Context quantity ablation (relevance vs quantity)

Usage:
    python eval_topic_routing.py [--threshold 0.5] [--n-samples 200] [--device cuda]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, _extract_articles
from topic_context import TopicContextManager
from entropy_monitor import EntropyMonitor


def load_v18_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


# ============================================================
# Helpers
# ============================================================

@torch.no_grad()
def compute_continuation_perplexity(model, context_ids, continuation_ids, device, max_seq_len=512):
    """Compute perplexity of real continuation given context.

    Args:
        model: V18 model
        context_ids: (C,) context token ids (or empty tensor)
        continuation_ids: (T,) real continuation token ids
        device: torch device
        max_seq_len: max sequence length

    Returns:
        perplexity (float)
    """
    if continuation_ids.shape[0] < 2:
        return float('inf')

    # Concatenate context + continuation, truncate to max_seq_len
    if context_ids is not None and context_ids.shape[0] > 0:
        full = torch.cat([context_ids, continuation_ids])
    else:
        full = continuation_ids

    full = full[-max_seq_len:]
    x = full[:-1].unsqueeze(0).to(device)
    y = full[1:].unsqueeze(0).to(device)

    output = model(x, step=0)
    B, T, V = output.logits.shape
    loss = F.cross_entropy(output.logits.reshape(B * T, V), y.reshape(B * T))
    return math.exp(min(loss.item(), 20))


@torch.no_grad()
def generate_continuation(model, context_ids, prompt_ids, num_tokens, device,
                          temperature=0.9, top_k=50, max_seq_len=512):
    """Generate continuation with optional context prepended."""
    model.eval()
    if context_ids is not None and context_ids.shape[0] > 0:
        combined = torch.cat([context_ids, prompt_ids])[-max_seq_len:]
        prompt_offset = max(0, combined.shape[0] - prompt_ids.shape[0])
    else:
        combined = prompt_ids[-max_seq_len:]
        prompt_offset = 0

    input_ids = combined.unsqueeze(0).to(device)

    for _ in range(num_tokens):
        idx = input_ids[:, -max_seq_len:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Return only generated tokens (after prompt)
    all_tokens = input_ids[0, prompt_offset:]
    generated_only = all_tokens[prompt_ids.shape[0]:]
    return generated_only


def compute_ngram_metrics(tokens, context_tokens=None, ns=(2, 3, 4)):
    """Compute repetition and context overlap metrics.

    Returns dict with rep-n, distinct-n, and context_overlap-n.
    """
    results = {}
    token_list = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens

    for n in ns:
        ngrams = [tuple(token_list[i:i+n]) for i in range(len(token_list) - n + 1)]
        if not ngrams:
            results[f"rep_{n}"] = 0.0
            results[f"distinct_{n}"] = 0.0
            continue

        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        results[f"rep_{n}"] = repeated / len(ngrams)
        results[f"distinct_{n}"] = len(counts) / len(ngrams)

        # Context overlap
        if context_tokens is not None and len(context_tokens) > 0:
            ctx_list = context_tokens.tolist() if isinstance(context_tokens, torch.Tensor) else context_tokens
            ctx_ngrams = set(tuple(ctx_list[i:i+n]) for i in range(len(ctx_list) - n + 1))
            if ctx_ngrams:
                gen_ngrams = set(ngrams)
                overlap = len(gen_ngrams & ctx_ngrams) / len(gen_ngrams)
                results[f"ctx_overlap_{n}"] = overlap
            else:
                results[f"ctx_overlap_{n}"] = 0.0
        else:
            results[f"ctx_overlap_{n}"] = 0.0

    return results


@torch.no_grad()
def compute_engram(model, token_ids, device, extract_layer):
    """Compute engram for topic coherence measurement."""
    model.eval()
    if token_ids.shape[0] < 5:
        return None
    ids = token_ids[:512].unsqueeze(0).to(device)

    captured = {}
    def hook_fn(module, inp, out):
        captured['h'] = out[0].detach()
    handle = model.blocks[extract_layer].register_forward_hook(hook_fn)
    _ = model(ids, step=0)
    handle.remove()

    return captured['h'].mean(dim=1).squeeze(0).cpu()


# ============================================================
# Main evaluation
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--continuation-len", type=int, default=256)
    parser.add_argument("--prompt-len", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    extract_layer = cfg.model.n_layers - 2

    # Load data
    print("\nLoading WikiText-103...")
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens
    train_tokens = splits["train"].tokens

    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    val_articles = _extract_articles(raw["validation"]["text"])

    # Populate topic manager
    mgr = TopicContextManager(
        model, tokenizer,
        similarity_threshold=args.threshold,
        max_context_tokens=384,
        max_active=3,
    )
    print(f"Populating topic clusters...")
    for i, (title, text) in enumerate(val_articles[:60]):
        if len(text.strip()) < 100:
            continue
        mgr.process_prompt(text[:500].strip())
    print(f"  {len(mgr.clusters)} clusters from {min(60, len(val_articles))} articles")

    # Prepare samples
    total_per_sample = args.prompt_len + args.continuation_len
    max_start = len(test_tokens) - total_per_sample
    stride = max(1, max_start // args.n_samples)

    # ============================================================
    # Metric 1: Held-out perplexity
    # ============================================================
    print("\n" + "=" * 60)
    print("METRIC 1: Held-Out Perplexity")
    print("=" * 60)

    ppls = {"none": [], "topic": [], "random": []}

    t0 = time.time()
    for i in range(args.n_samples):
        start = i * stride
        if start + total_per_sample > len(test_tokens):
            start = len(test_tokens) - total_per_sample

        prompt_ids = test_tokens[start:start + args.prompt_len]
        continuation_ids = test_tokens[start + args.prompt_len:start + total_per_sample]

        # Condition A: no context
        ppl_none = compute_continuation_perplexity(
            model, None, torch.cat([prompt_ids, continuation_ids]), device
        )
        ppls["none"].append(ppl_none)

        # Condition D: topic-routed context
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        mgr.process_prompt(prompt_text)
        topic_ctx = mgr.get_context()
        ppl_topic = compute_continuation_perplexity(
            model, topic_ctx, torch.cat([prompt_ids, continuation_ids]), device
        )
        ppls["topic"].append(ppl_topic)

        # Condition B: random context (matched token count)
        n_ctx = topic_ctx.shape[0] if topic_ctx.shape[0] > 0 else 384
        rand_start = random.randint(0, len(train_tokens) - n_ctx - 1)
        random_ctx = train_tokens[rand_start:rand_start + n_ctx]
        ppl_random = compute_continuation_perplexity(
            model, random_ctx, torch.cat([prompt_ids, continuation_ids]), device
        )
        ppls["random"].append(ppl_random)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n_samples}: "
                  f"none={sum(ppls['none'])/len(ppls['none']):.1f} "
                  f"topic={sum(ppls['topic'])/len(ppls['topic']):.1f} "
                  f"random={sum(ppls['random'])/len(ppls['random']):.1f}")

    for k in ppls:
        ppls[k] = [p for p in ppls[k] if not math.isinf(p)]

    mean_ppls = {k: sum(v) / len(v) for k, v in ppls.items()}
    print(f"\n  Results:")
    print(f"    No context:     {mean_ppls['none']:.2f}")
    print(f"    Random context: {mean_ppls['random']:.2f}")
    print(f"    Topic context:  {mean_ppls['topic']:.2f}")
    print(f"    Topic vs none:  {mean_ppls['topic'] - mean_ppls['none']:+.2f}")
    print(f"    Topic vs random: {mean_ppls['topic'] - mean_ppls['random']:+.2f}")

    # ============================================================
    # Metric 3: Repetition and degeneration
    # ============================================================
    print("\n" + "=" * 60)
    print("METRIC 3: Repetition and Degeneration")
    print("=" * 60)

    rep_metrics = {"none": [], "topic": []}

    for i in range(min(args.n_samples, 100)):  # 100 samples for generation
        start = i * stride
        if start + total_per_sample > len(test_tokens):
            start = len(test_tokens) - total_per_sample

        prompt_ids = test_tokens[start:start + args.prompt_len]

        # Baseline generation (no context)
        gen_none = generate_continuation(
            model, None, prompt_ids, args.continuation_len, device
        )
        rep_metrics["none"].append(compute_ngram_metrics(gen_none))

        # Topic-routed generation
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        mgr.process_prompt(prompt_text)
        topic_ctx = mgr.get_context()
        gen_topic = generate_continuation(
            model, topic_ctx, prompt_ids, args.continuation_len, device
        )
        rep_metrics["topic"].append(compute_ngram_metrics(gen_topic, context_tokens=topic_ctx))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/100 generated")

    # Average metrics
    for condition in ["none", "topic"]:
        if not rep_metrics[condition]:
            continue
        keys = rep_metrics[condition][0].keys()
        avgs = {k: sum(m[k] for m in rep_metrics[condition]) / len(rep_metrics[condition]) for k in keys}
        print(f"\n  {condition}:")
        for k, v in sorted(avgs.items()):
            print(f"    {k}: {v:.4f}")

    # ============================================================
    # Metric 4: Conditional perplexity by routing quality
    # ============================================================
    print("\n" + "=" * 60)
    print("METRIC 4: Perplexity vs Routing Confidence")
    print("=" * 60)

    # Bin by routing confidence and compute mean perplexity reduction
    routing_data = []
    for i in range(args.n_samples):
        start = i * stride
        if start + total_per_sample > len(test_tokens):
            start = len(test_tokens) - total_per_sample
        prompt_ids = test_tokens[start:start + args.prompt_len]

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        result = mgr.process_prompt(prompt_text)
        similarity = result["similarity"]

        ppl_reduction = ppls["none"][i] - ppls["topic"][i] if i < len(ppls["none"]) else 0

        routing_data.append({
            "similarity": similarity,
            "ppl_reduction": ppl_reduction,
            "ppl_none": ppls["none"][i] if i < len(ppls["none"]) else 0,
            "ppl_topic": ppls["topic"][i] if i < len(ppls["topic"]) else 0,
        })

    # Bin into quartiles
    sorted_data = sorted(routing_data, key=lambda x: x["similarity"])
    n = len(sorted_data)
    quartiles = [
        sorted_data[:n//4],
        sorted_data[n//4:n//2],
        sorted_data[n//2:3*n//4],
        sorted_data[3*n//4:],
    ]

    print(f"\n  Perplexity reduction by routing confidence quartile:")
    print(f"  {'Quartile':>10} {'Sim Range':>15} {'Mean PPL Red':>14} {'N':>5}")
    for qi, q in enumerate(quartiles):
        if not q:
            continue
        sim_lo = q[0]["similarity"]
        sim_hi = q[-1]["similarity"]
        mean_red = sum(d["ppl_reduction"] for d in q) / len(q)
        print(f"  {'Q' + str(qi+1):>10} {f'[{sim_lo:.3f}, {sim_hi:.3f}]':>15} {mean_red:>+14.2f} {len(q):>5}")

    # Correlation
    sims = [d["similarity"] for d in routing_data]
    reds = [d["ppl_reduction"] for d in routing_data]
    if len(sims) > 1:
        sim_t = torch.tensor(sims)
        red_t = torch.tensor(reds)
        corr = torch.corrcoef(torch.stack([sim_t, red_t]))[0, 1].item()
        print(f"\n  Pearson correlation (sim vs ppl_reduction): {corr:.4f}")

    # ============================================================
    # Metric 2: Topic coherence
    # ============================================================
    print("\n" + "=" * 60)
    print("METRIC 2: Topic Coherence (Prompt-Continuation Similarity)")
    print("=" * 60)

    coherence = {"none": [], "topic": []}

    for i in range(min(args.n_samples, 100)):
        start = i * stride
        if start + total_per_sample > len(test_tokens):
            start = len(test_tokens) - total_per_sample
        prompt_ids = test_tokens[start:start + args.prompt_len]

        prompt_engram = compute_engram(model, prompt_ids, device, extract_layer)
        if prompt_engram is None:
            continue

        # No context generation
        gen_none = generate_continuation(
            model, None, prompt_ids, args.continuation_len, device
        )
        gen_engram_none = compute_engram(model, gen_none, device, extract_layer)
        if gen_engram_none is not None:
            sim = F.cosine_similarity(
                prompt_engram.unsqueeze(0), gen_engram_none.unsqueeze(0)
            ).item()
            coherence["none"].append(sim)

        # Topic-routed generation
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        mgr.process_prompt(prompt_text)
        topic_ctx = mgr.get_context()
        gen_topic = generate_continuation(
            model, topic_ctx, prompt_ids, args.continuation_len, device
        )
        gen_engram_topic = compute_engram(model, gen_topic, device, extract_layer)
        if gen_engram_topic is not None:
            sim = F.cosine_similarity(
                prompt_engram.unsqueeze(0), gen_engram_topic.unsqueeze(0)
            ).item()
            coherence["topic"].append(sim)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/100 evaluated")

    for condition in ["none", "topic"]:
        vals = coherence[condition]
        if vals:
            print(f"\n  {condition}: mean={sum(vals)/len(vals):.4f}, "
                  f"std={torch.tensor(vals).std().item():.4f}, n={len(vals)}")

    if coherence["none"] and coherence["topic"]:
        diff = sum(coherence["topic"]) / len(coherence["topic"]) - sum(coherence["none"]) / len(coherence["none"])
        print(f"  Topic vs none: {diff:+.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Metric 1 — Held-out Perplexity:")
    print(f"    No context: {mean_ppls['none']:.2f}")
    print(f"    Random ctx: {mean_ppls['random']:.2f}")
    print(f"    Topic ctx:  {mean_ppls['topic']:.2f}")

    print(f"\n  Metric 2 — Topic Coherence (prompt-continuation sim):")
    for c in ["none", "topic"]:
        if coherence[c]:
            print(f"    {c}: {sum(coherence[c])/len(coherence[c]):.4f}")

    print(f"\n  Metric 3 — Repetition (rep-3):")
    for c in ["none", "topic"]:
        if rep_metrics[c]:
            avg = sum(m["rep_3"] for m in rep_metrics[c]) / len(rep_metrics[c])
            print(f"    {c}: {avg:.4f}")

    print(f"\n  Metric 3 — Context Overlap (3-gram):")
    if rep_metrics["topic"]:
        avg_overlap = sum(m["ctx_overlap_3"] for m in rep_metrics["topic"]) / len(rep_metrics["topic"])
        print(f"    topic: {avg_overlap:.4f}")

    if len(sims) > 1:
        print(f"\n  Metric 4 — Routing confidence correlation: {corr:.4f}")

    # Save
    results = {
        "threshold": args.threshold,
        "n_samples": args.n_samples,
        "prompt_len": args.prompt_len,
        "continuation_len": args.continuation_len,
        "perplexity": mean_ppls,
        "coherence": {k: sum(v)/len(v) if v else 0 for k, v in coherence.items()},
        "repetition": {
            c: {k: sum(m[k] for m in rep_metrics[c]) / len(rep_metrics[c]) for k in rep_metrics[c][0].keys()}
            for c in rep_metrics if rep_metrics[c]
        },
        "routing_correlation": corr if len(sims) > 1 else None,
    }
    out_path = Path("results/v18_cross_attn/eval_topic_routing.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
