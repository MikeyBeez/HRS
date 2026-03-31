"""MAUVE benchmark for topic-routed context assembly.

Compares:
1. Baseline: sliding-window generation (last 512 tokens)
2. Topic-routed: context assembled from relevant topic cluster

Setup: Process a sequence of WikiText-103 articles through the topic
context manager to populate clusters. Then generate continuations from
test set prompts, comparing baseline vs topic-routed context.

Usage:
    python benchmark_mauve_topic.py [--threshold 0.5] [--n-samples 500]
"""

import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import mauve
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, _extract_articles
from topic_context import TopicContextManager


def load_v18_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


def populate_topic_manager(mgr, articles, tokenizer, max_articles=200):
    """Feed articles through the topic manager to populate clusters.

    Uses full article text (not single sentences) for better engram quality.
    """
    print(f"Populating topic clusters from {min(len(articles), max_articles)} articles...")
    for i, (title, text) in enumerate(articles[:max_articles]):
        if len(text.strip()) < 100:
            continue
        # Use first ~500 chars as the prompt (enough for decent engram)
        prompt = text[:500].strip()
        mgr.process_prompt(prompt)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} articles processed, {len(mgr.clusters)} clusters")

    print(f"  Final: {len(mgr.clusters)} clusters, "
          f"{sum(len(c) for c in mgr.clusters)} total prompts")
    return mgr


@torch.no_grad()
def generate_with_context(model, prompt_ids, context_ids, num_tokens,
                          temperature=0.9, top_k=50, max_seq_len=512):
    """Generate with topic-filtered context prepended to prompt.

    Concatenates context + prompt, truncates to max_seq_len, then generates.
    """
    model.eval()
    device = next(model.parameters()).device

    if context_ids is not None and context_ids.shape[0] > 0:
        # Concatenate context + prompt, keep last max_seq_len tokens
        combined = torch.cat([context_ids, prompt_ids])[-max_seq_len:]
        prompt_offset = combined.shape[0] - prompt_ids.shape[0]
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

    # Return only prompt + generated (strip context prefix)
    return input_ids[0, prompt_offset:]


@torch.no_grad()
def generate_baseline(model, prompt_ids, num_tokens, temperature=0.9,
                      top_k=50, max_seq_len=512):
    """Standard sliding-window generation (no topic routing)."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = prompt_ids.unsqueeze(0).to(device)

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

    return input_ids[0]


def run_condition(model, tokenizer, test_tokens, mgr, prompt_len,
                  continuation_len, n_samples, temperature, top_k,
                  device, label, use_topic_context=False):
    """Run one MAUVE condition."""
    print(f"\n{'='*70}")
    print(f"Condition: {label}")
    print(f"  Prompt: {prompt_len} tok, Continuation: {continuation_len} tok, Samples: {n_samples}")
    print(f"{'='*70}")

    total_per_sample = prompt_len + continuation_len
    max_start = len(test_tokens) - total_per_sample
    stride = max(1, max_start // n_samples)

    ref_texts = []
    gen_texts = []
    context_stats = {"n_with_context": 0, "mean_context_tokens": 0, "total_context_tokens": 0}

    t0 = time.time()
    for i in range(n_samples):
        start = i * stride
        if start + total_per_sample > len(test_tokens):
            start = len(test_tokens) - total_per_sample

        prompt_ids = test_tokens[start:start + prompt_len]
        ref_ids = test_tokens[start:start + total_per_sample]
        ref_texts.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

        if use_topic_context:
            # Classify the prompt and get topic-routed context
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            mgr.process_prompt(prompt_text)
            context_tokens = mgr.get_context()

            if context_tokens.shape[0] > 0:
                context_stats["n_with_context"] += 1
                context_stats["total_context_tokens"] += context_tokens.shape[0]

            gen_ids = generate_with_context(
                model, prompt_ids, context_tokens, continuation_len,
                temperature=temperature, top_k=top_k,
            )
        else:
            gen_ids = generate_baseline(
                model, prompt_ids, continuation_len,
                temperature=temperature, top_k=top_k,
            )

        gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

        done = i + 1
        elapsed = time.time() - t0
        if done % 50 == 0 or done == n_samples:
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{n_samples} generated ({rate:.2f} samples/s)")

    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.0f}s")

    if use_topic_context:
        n_ctx = context_stats["n_with_context"]
        mean_ctx = context_stats["total_context_tokens"] / max(n_ctx, 1)
        print(f"  Context provided: {n_ctx}/{n_samples} samples, mean {mean_ctx:.0f} tokens")

    # Sample outputs
    for i in range(min(2, len(gen_texts))):
        print(f"\n  Sample {i}: {gen_texts[i][:150]}...")

    print("\nComputing MAUVE score...")
    t0 = time.time()
    out = mauve.compute_mauve(
        p_text=ref_texts, q_text=gen_texts,
        device_id=0 if device.type == "cuda" else -1,
        verbose=True,
    )
    mauve_time = time.time() - t0
    print(f"MAUVE score: {out.mauve:.4f} (computed in {mauve_time:.0f}s)")

    return {
        "label": label,
        "prompt_len": prompt_len,
        "mauve_score": out.mauve,
        "generation_time_s": elapsed,
        "context_stats": context_stats if use_topic_context else None,
    }


def main():
    threshold = 0.5
    n_samples = 500
    max_active = 3
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
        if arg == "--n-samples" and i + 1 < len(sys.argv):
            n_samples = int(sys.argv[i + 1])
        if arg == "--max-active" and i + 1 < len(sys.argv):
            max_active = int(sys.argv[i + 1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load data
    print("\nLoading WikiText-103...")
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    val_articles = _extract_articles(raw["validation"]["text"])

    # Create and populate topic manager
    mgr = TopicContextManager(
        model, tokenizer,
        similarity_threshold=threshold,
        max_context_tokens=384,  # leave room for prompt in 512 window
        max_active=max_active,
    )
    populate_topic_manager(mgr, val_articles, tokenizer, max_articles=60)
    print(f"\n{mgr.describe_clusters()}")

    # Known baselines from prior runs (benchmark_mauve_v18.py)
    baselines = {
        50: 0.915,   # 50-tok baseline engram ON
        500: 0.919,  # 500-tok baseline engram ON
    }

    # Run topic-routed conditions only
    continuation_len = 256
    temperature = 0.9
    top_k = 50
    results = []

    for prompt_len in [50, 500]:
        r = run_condition(
            model, tokenizer, test_tokens, mgr,
            prompt_len=prompt_len, continuation_len=continuation_len,
            n_samples=n_samples, temperature=temperature, top_k=top_k,
            device=device,
            label=f"{prompt_len}-tok + topic routing (threshold={threshold})",
            use_topic_context=True,
        )
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"MAUVE RESULTS — Topic-Routed Context (threshold={threshold})")
    print("=" * 70)
    print(f"{'Condition':<50} {'MAUVE':>8} {'vs Baseline':>12}")
    print("-" * 72)
    for r in results:
        baseline = baselines.get(r["prompt_len"], 0)
        delta = r["mauve_score"] - baseline
        print(f"{r['label']:<50} {r['mauve_score']:>8.4f} {delta:>+12.4f}")
    print("=" * 72)
    print(f"\nBaselines (from prior runs): 50-tok={baselines[50]:.3f}, 500-tok={baselines[500]:.3f}")

    # Save
    out_path = Path("results/v18_cross_attn/mauve_topic_routing.json")
    with open(out_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "max_active": max_active,
            "n_clusters": len(mgr.clusters),
            "conditions": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
