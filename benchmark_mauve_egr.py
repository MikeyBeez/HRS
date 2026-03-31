"""MAUVE benchmark for V18 with entropy-gated retrieval (EGR).

Compares:
1. V18 baseline (no retrieval)
2. V18 + EGR (entropy-gated engram retrieval)

Uses sliding-window generation (matching original benchmark).

Usage:
    python benchmark_mauve_egr.py [--store engram_store_data] [--threshold 4.0]
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
    return model, cfg, ckpt["model_state_dict"].get("engram_buffer")


def extract_prompts_and_refs(tokens, prompt_len, continuation_len, n_samples):
    """Extract prompt/reference pairs from a token tensor."""
    total_per_sample = prompt_len + continuation_len
    max_start = len(tokens) - total_per_sample
    stride = max(1, max_start // n_samples)

    prompts = []
    ref_sequences = []
    for i in range(n_samples):
        start = i * stride
        if start + total_per_sample > len(tokens):
            start = len(tokens) - total_per_sample
        prompts.append(tokens[start:start + prompt_len])
        ref_sequences.append(tokens[start:start + total_per_sample])

    return torch.stack(prompts), torch.stack(ref_sequences)


@torch.no_grad()
def generate_baseline(model, prompt_ids, num_tokens, temperature=0.9, top_k=50):
    """Sliding-window generation without retrieval."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)
    max_seq_len = 512

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

    return input_ids


@torch.no_grad()
def generate_with_retrieval(
    model, engine, prompt_ids, num_tokens, orig_buffer,
    temperature=0.9, top_k=50,
):
    """Sliding-window generation with entropy-gated retrieval.

    Uses the same sliding-window approach as baseline, but monitors entropy
    and injects retrieved engrams via cross-attention when triggered.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)
    max_seq_len = 512

    engine.monitor.reset_rolling()
    retrieval_active = False
    cooldown = 0

    for step in range(num_tokens):
        idx = input_ids[:, -max_seq_len:]
        output = model(idx, step=0)
        logits = output.logits[:, -1:, :]

        # Monitor entropy
        token_ent = EntropyMonitor.token_entropy(logits).item()
        engine.monitor.update_rolling(token_ent)

        # Check retrieval trigger
        if cooldown > 0:
            cooldown -= 1
        else:
            should_retrieve, rolling_ent = engine.monitor.should_retrieve()
            if should_retrieve and not retrieval_active:
                # Compute context engram for query
                hidden_hook = {}
                def hook_fn(module, inp, out):
                    hidden_hook['h'] = out[0].detach()
                handle = model.blocks[engine.extract_layer].register_forward_hook(hook_fn)
                _ = model(idx, step=0)
                handle.remove()
                context_engram = hidden_hook['h'].mean(dim=1).squeeze(0)

                results = engine.store.retrieve(
                    context_engram, top_k=engine.top_k,
                    min_similarity=engine.min_similarity,
                )
                if results:
                    retrieved = torch.stack([r[2] for r in results], dim=0)  # (K, D)
                    # Expand to match expected buffer shape (1, 32, D) by repeating
                    n_slots = model.engram_buffer.shape[1]
                    if retrieved.shape[0] < n_slots:
                        n_repeats = (n_slots + retrieved.shape[0] - 1) // retrieved.shape[0]
                        retrieved = retrieved.repeat(n_repeats, 1)[:n_slots]
                    model.engram_buffer.data = retrieved.unsqueeze(0).to(device)
                    model._engram_buffer_initialized = True
                    retrieval_active = True
                    cooldown = engine.monitor.read_window

        # Restore buffer after cooldown
        if retrieval_active and cooldown == 0:
            if orig_buffer is not None:
                model.engram_buffer.data = orig_buffer.to(device)
            retrieval_active = False

        # Sample
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Restore buffer
    if orig_buffer is not None:
        model.engram_buffer.data = orig_buffer.to(device)

    return input_ids


def run_condition(
    model, tokenizer, test_tokens, prompt_len, continuation_len,
    n_samples, batch_size, temperature, top_k, device, label,
    engine=None, orig_buffer=None,
):
    """Run one MAUVE condition."""
    print(f"\n{'='*70}")
    print(f"Condition: {label}")
    print(f"  Prompt: {prompt_len} tok, Continuation: {continuation_len} tok, Samples: {n_samples}")
    print(f"{'='*70}")

    prompts, ref_ids = extract_prompts_and_refs(
        test_tokens, prompt_len, continuation_len, n_samples
    )

    ref_texts = [tokenizer.decode(ref_ids[i], skip_special_tokens=True)
                 for i in range(n_samples)]

    gen_texts = []
    t0 = time.time()
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_prompts = prompts[batch_start:batch_end]

        for j in range(batch_prompts.shape[0]):
            single_prompt = batch_prompts[j:j+1]
            if engine is not None:
                gen_ids = generate_with_retrieval(
                    model, engine, single_prompt, continuation_len,
                    orig_buffer, temperature=temperature, top_k=top_k,
                )
            else:
                gen_ids = generate_baseline(
                    model, single_prompt, continuation_len,
                    temperature=temperature, top_k=top_k,
                )
            gen_texts.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))

        done = len(gen_texts)
        elapsed = time.time() - t0
        if done % 50 == 0 or done == n_samples:
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{n_samples} generated ({rate:.2f} samples/s)")

    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.0f}s")

    # Sample outputs
    print(f"\nSample generations:")
    for i in range(min(3, len(gen_texts))):
        prompt_text = tokenizer.decode(prompts[i], skip_special_tokens=True)
        continuation = gen_texts[i][len(prompt_text):]
        print(f"  Prompt: ...{prompt_text[-60:]}")
        print(f"  Gen: {continuation[:120]}...")
        print()

    print("Computing MAUVE score...")
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
    }


def main():
    store_path = "engram_store_data"
    threshold = 4.0
    for i, arg in enumerate(sys.argv):
        if arg == "--store" and i + 1 < len(sys.argv):
            store_path = sys.argv[i + 1]
        if arg == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, orig_buffer = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    store = EngramStore.load(store_path)
    engine = RetrievalEngine(
        model=model, store=store,
        read_threshold=threshold, read_window=10,
        top_k=1, min_similarity=0.3,
    )

    print("\nLoading WikiText-103 test set...")
    from data import load_wikitext
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    n_samples = 1000
    continuation_len = 256
    temperature = 0.9
    top_k = 50
    batch_size = 8  # processed one at a time for retrieval, but groups logging

    results = []

    # Condition 1: Baseline 50-tok
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=50, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device, label="50-tok baseline (no retrieval)",
    )
    results.append(r)

    # Condition 2: EGR 50-tok
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=50, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device, label="50-tok + EGR",
        engine=engine, orig_buffer=orig_buffer,
    )
    results.append(r)

    # Condition 3: Baseline 500-tok
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=500, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device, label="500-tok baseline (no retrieval)",
    )
    results.append(r)

    # Condition 4: EGR 500-tok
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=500, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device, label="500-tok + EGR",
        engine=engine, orig_buffer=orig_buffer,
    )
    results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("MAUVE RESULTS — V18 + Entropy-Gated Retrieval")
    print("=" * 70)
    print(f"Store: {len(store)} engrams, threshold: {threshold} bits")
    print("-" * 70)
    print(f"{'Condition':<40} {'MAUVE':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<40} {r['mauve_score']:>8.4f}")
    print("=" * 70)

    if len(results) == 4:
        short_base = results[0]["mauve_score"]
        short_egr = results[1]["mauve_score"]
        long_base = results[2]["mauve_score"]
        long_egr = results[3]["mauve_score"]
        print(f"\nEGR effect at 50 tokens:  {short_egr - short_base:+.4f} ({short_egr:.4f} vs {short_base:.4f})")
        print(f"EGR effect at 500 tokens: {long_egr - long_base:+.4f} ({long_egr:.4f} vs {long_base:.4f})")

    out_path = Path(store_path) / "mauve_egr_results.json"
    with open(out_path, "w") as f:
        json.dump({"conditions": results, "threshold": threshold, "store_size": len(store)}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
