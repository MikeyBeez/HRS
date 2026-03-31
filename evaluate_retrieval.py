"""Evaluate entropy-gated engram retrieval on WikiText-103 test set.

Compares:
1. Baseline: V18 generation without retrieval
2. With retrieval: V18 generation with entropy-gated engram retrieval

Metrics: perplexity, trigger rate, retrieval quality inspection.

Usage:
    python evaluate_retrieval.py [--store engram_store_data] [--threshold 4.0]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore
from entropy_monitor import EntropyMonitor
from retrieval_engine import RetrievalEngine


def load_v18_model(device):
    """Load trained V18 model."""
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)

    ckpt_path = Path("results/v18_cross_attn/best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True

    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"Loaded V18 (step {step}, val_ppl {val_ppl:.2f})")
    return model, cfg


@torch.no_grad()
def compute_perplexity(model, token_ids, seq_len=512, device=None):
    """Compute perplexity over a token sequence.

    Args:
        model: HRSTransformer
        token_ids: 1D tensor of token ids
        seq_len: sequence length for chunking
        device: torch device

    Returns:
        (perplexity, mean_entropy, per_segment_stats)
    """
    model.eval()
    n_seqs = (len(token_ids) - 1) // seq_len
    if n_seqs == 0:
        return float('inf'), 0.0, []

    total_loss = 0.0
    total_tokens = 0
    total_entropy = 0.0
    segment_stats = []

    for i in range(n_seqs):
        start = i * seq_len
        x = token_ids[start:start + seq_len].unsqueeze(0).to(device)
        y = token_ids[start + 1:start + seq_len + 1].unsqueeze(0).to(device)

        output = model(x, step=0)
        logits = output.logits

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T), reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += T

        # Entropy
        ent = EntropyMonitor.token_entropy(logits).mean().item()
        total_entropy += ent

        segment_stats.append({
            "segment": i,
            "loss": loss.item() / T,
            "ppl": math.exp(min(loss.item() / T, 20)),
            "mean_entropy": ent,
        })

    mean_loss = total_loss / total_tokens
    ppl = math.exp(min(mean_loss, 20))
    mean_entropy = total_entropy / n_seqs

    return ppl, mean_entropy, segment_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate engram retrieval")
    parser.add_argument("--store", type=str, default="engram_store_data", help="Store directory")
    parser.add_argument("--threshold", type=float, default=4.0, help="Read threshold")
    parser.add_argument("--read-window", type=int, default=10, help="Rolling entropy window")
    parser.add_argument("--top-k", type=int, default=1, help="Top-K retrievals")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Min cosine similarity")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of generation samples")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Tokens to generate per sample")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load store
    print(f"\nLoading engram store from {args.store}...")
    store = EngramStore.load(args.store)
    print(f"Store stats: {store.stats()}")

    # Load test data
    print("\nLoading WikiText-103 test set...")
    from data import load_wikitext
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    # ============================================================
    # 1. Baseline perplexity (no retrieval)
    # ============================================================
    print("\n" + "=" * 60)
    print("1. Baseline Perplexity (no retrieval)")
    print("=" * 60)
    ppl_baseline, ent_baseline, _ = compute_perplexity(
        model, test_tokens, device=device
    )
    print(f"  Perplexity: {ppl_baseline:.2f}")
    print(f"  Mean entropy: {ent_baseline:.4f} bits")

    # ============================================================
    # 2. Generation with retrieval
    # ============================================================
    print("\n" + "=" * 60)
    print("2. Generation with Entropy-Gated Retrieval")
    print("=" * 60)

    engine = RetrievalEngine(
        model=model,
        store=store,
        read_threshold=args.threshold,
        read_window=args.read_window,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )

    # Extract prompts from test set
    prompt_len = 50
    n_samples = min(args.n_samples, len(test_tokens) // (prompt_len + args.max_new_tokens))
    stride = max(1, (len(test_tokens) - prompt_len - args.max_new_tokens) // n_samples)

    all_triggers = []
    total_trigger_rate = 0.0
    gen_texts_baseline = []
    gen_texts_retrieval = []

    t0 = time.time()
    for i in range(n_samples):
        start = i * stride
        prompt_ids = test_tokens[start:start + prompt_len]

        # Baseline generation (no retrieval)
        model._engram_buffer_initialized = True
        # Restore original buffer
        ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
        orig_buffer = ckpt["model_state_dict"].get("engram_buffer")
        if orig_buffer is not None:
            model.engram_buffer.copy_(orig_buffer.to(device))

        baseline_ids = _generate_baseline(
            model, prompt_ids.unsqueeze(0).to(device),
            max_new_tokens=args.max_new_tokens,
        )
        gen_texts_baseline.append(tokenizer.decode(baseline_ids[0], skip_special_tokens=True))

        # Restore buffer for retrieval run
        if orig_buffer is not None:
            model.engram_buffer.copy_(orig_buffer.to(device))
            model._engram_buffer_initialized = True

        # Retrieval generation
        gen_ids, stats = engine.generate_with_retrieval(
            prompt_ids, max_new_tokens=args.max_new_tokens,
        )
        gen_texts_retrieval.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))

        all_triggers.extend(stats["triggers"])
        total_trigger_rate += stats["trigger_rate"]

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{n_samples} samples | "
                  f"{len(all_triggers)} total triggers | "
                  f"{elapsed:.0f}s")

    elapsed = time.time() - t0
    avg_trigger_rate = total_trigger_rate / n_samples

    print(f"\nGeneration complete in {elapsed:.0f}s")
    print(f"Average trigger rate: {avg_trigger_rate:.3f} ({avg_trigger_rate * 100:.1f}%)")
    print(f"Total triggers: {len(all_triggers)}")

    # ============================================================
    # 3. Retrieval quality inspection
    # ============================================================
    print("\n" + "=" * 60)
    print("3. Retrieval Quality Inspection")
    print("=" * 60)

    if all_triggers:
        print(f"\nSample triggers (first 10):")
        for t in all_triggers[:10]:
            print(f"  Step {t['step']}: entropy={t['rolling_entropy']:.3f}, "
                  f"sim={t['similarities']}")
            print(f"    Retrieved: {t['retrieved_texts'][0][:80]}...")
            print()
    else:
        print("No triggers fired. Consider lowering the threshold.")

    # ============================================================
    # 4. Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline perplexity: {ppl_baseline:.2f}")
    print(f"Mean test set entropy: {ent_baseline:.4f} bits")
    print(f"Store size: {len(store)} engrams")
    print(f"Read threshold: {args.threshold} bits")
    print(f"Avg trigger rate: {avg_trigger_rate:.3f}")
    print(f"Total triggers: {len(all_triggers)}")
    print(f"Engine stats: {engine.get_stats()}")

    # Save results
    results = {
        "baseline_ppl": ppl_baseline,
        "mean_test_entropy": ent_baseline,
        "store_size": len(store),
        "read_threshold": args.threshold,
        "read_window": args.read_window,
        "top_k": args.top_k,
        "min_similarity": args.min_similarity,
        "n_samples": n_samples,
        "avg_trigger_rate": avg_trigger_rate,
        "total_triggers": len(all_triggers),
        "triggers": all_triggers[:50],  # save first 50 for inspection
    }
    out_path = Path(args.store) / "retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


@torch.no_grad()
def _generate_baseline(model, prompt_ids, max_new_tokens=256, temperature=0.9, top_k=50):
    """Baseline generation without retrieval (sliding window)."""
    model.eval()
    input_ids = prompt_ids.clone()
    max_seq_len = 512

    for _ in range(max_new_tokens):
        idx = input_ids[:, -max_seq_len:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


if __name__ == "__main__":
    main()
