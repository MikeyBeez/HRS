"""MAUVE benchmark for V18 (cross-attention engram + categorization head).

Adapts the standard benchmark for V18's architecture:
- Cross-attention engram buffer (not prepend)
- Toggle via _engram_buffer_initialized flag
- Uses model.generate() with KV cache for speed

Usage:
    python benchmark_mauve_v18.py [--device cuda]
"""

import sys
import json
import time
from pathlib import Path

import torch
import mauve
from transformers import AutoTokenizer
from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext


def load_model(run_dir, device):
    """Load V18 model from best checkpoint."""
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)

    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # The engram buffer is restored from state_dict, but the initialized flag isn't
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True

    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"Loaded {ckpt_path} (step {step}, val_ppl {val_ppl})")
    print(f"Engram buffer initialized: {model._engram_buffer_initialized}")
    print(f"Engram buffer norm: {model.engram_buffer.norm():.4f}")

    # Print gate values
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'cross_attn') and block.use_cross_attn_engram:
            g = torch.sigmoid(block.cross_attn.gate_logit).item()
            print(f"  Layer {i} gate: {g:.4f}")

    return model, cfg, step, val_ppl


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
def generate_continuations(model, prompt_ids, num_tokens, temperature=0.9, top_k=50):
    """Generate continuation tokens using sliding-window forward pass.

    Matches the original benchmark_mauve.py generation method: truncates to
    the last 512 tokens on each step to stay within trained RoPE positions.
    """
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
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


def run_condition(model, tokenizer, test_tokens, prompt_len, continuation_len,
                  n_samples, batch_size, temperature, top_k, device, label):
    """Run one MAUVE condition using sliding-window generation."""
    print(f"\n{'='*70}")
    print(f"Condition: {label}")
    print(f"  Prompt length: {prompt_len}, Continuation: {continuation_len}")
    print(f"  Samples: {n_samples}, Temperature: {temperature}, Top-k: {top_k}")
    print(f"{'='*70}")

    prompts, ref_ids = extract_prompts_and_refs(
        test_tokens, prompt_len, continuation_len, n_samples
    )

    # Decode references
    ref_texts = [tokenizer.decode(ref_ids[i], skip_special_tokens=True)
                 for i in range(n_samples)]

    # Generate continuations in batches using sliding-window forward pass
    gen_texts = []
    t0 = time.time()
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_prompts = prompts[batch_start:batch_end]

        gen_ids = generate_continuations(
            model, batch_prompts, continuation_len,
            temperature=temperature, top_k=top_k,
        )

        for j in range(gen_ids.shape[0]):
            text = tokenizer.decode(gen_ids[j], skip_special_tokens=True)
            gen_texts.append(text)

        done = len(gen_texts)
        elapsed = time.time() - t0
        if done % 50 == 0 or done == n_samples:
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{n_samples} generated ({rate:.1f} samples/s)")

    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.0f}s")

    # Show samples
    print(f"\nSample generations:")
    for i in range(min(3, len(gen_texts))):
        prompt_text = tokenizer.decode(prompts[i], skip_special_tokens=True)
        continuation = gen_texts[i][len(prompt_text):]
        print(f"  Prompt: ...{prompt_text[-80:]}")
        print(f"  Generated: {continuation[:150]}...")
        print()

    # Compute MAUVE
    print("Computing MAUVE score...")
    t0 = time.time()
    out = mauve.compute_mauve(
        p_text=ref_texts,
        q_text=gen_texts,
        device_id=0 if device.type == "cuda" else -1,
        verbose=True,
    )
    mauve_time = time.time() - t0
    print(f"MAUVE score: {out.mauve:.4f} (computed in {mauve_time:.0f}s)")

    return {
        "label": label,
        "prompt_len": prompt_len,
        "continuation_len": continuation_len,
        "n_samples": n_samples,
        "temperature": temperature,
        "top_k": top_k,
        "mauve_score": out.mauve,
        "generation_time_s": elapsed,
        "mauve_compute_time_s": mauve_time,
    }


def main():
    device_str = "cuda"
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            device_str = sys.argv[i + 1]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    run_dir = Path("results/v18_cross_attn")

    # Load model
    model, cfg, step, val_ppl = load_model(run_dir, device)

    # Load tokenizer and test data
    print("\nLoading WikiText-103 test set...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    # Parameters
    n_samples = 1000
    continuation_len = 256
    temperature = 0.9
    top_k = 50
    batch_size = 8

    results = []
    results_meta = {
        "version": "v18_cross_attn",
        "step": step,
        "val_ppl": val_ppl,
        "n_params": sum(p.numel() for p in model.parameters()),
        "conditions": [],
    }

    # Condition 1: 50-token prompts, cross-attention engram ON
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=50, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device,
        label="50-tok prompt, cross-attn engram ON",
    )
    results.append(r)

    # Condition 2: 500-token prompts, cross-attention engram ON
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=500, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device,
        label="500-tok prompt, cross-attn engram ON",
    )
    results.append(r)

    # Condition 3: 50-token prompts, cross-attention engram OFF
    model._engram_buffer_initialized = False
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=50, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device,
        label="50-tok prompt, cross-attn engram OFF",
    )
    results.append(r)

    # Condition 4: 500-token prompts, cross-attention engram OFF
    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=500, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device,
        label="500-tok prompt, cross-attn engram OFF",
    )
    results.append(r)
    model._engram_buffer_initialized = True  # restore

    # Summary
    print("\n" + "=" * 70)
    print("MAUVE BENCHMARK RESULTS — V18 Cross-Attention Engram")
    print("=" * 70)
    print(f"Model: v18_cross_attn (step {step}, val_ppl {val_ppl})")
    print(f"Params: {results_meta['n_params']/1e6:.1f}M")
    print(f"Samples: {n_samples}, Continuation: {continuation_len} tokens")
    print(f"Sampling: temperature={temperature}, top_k={top_k}")
    print("-" * 70)
    print(f"{'Condition':<50} {'MAUVE':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<50} {r['mauve_score']:>8.4f}")
    print("=" * 70)

    # Key comparisons
    if len(results) >= 4:
        short_on = results[0]["mauve_score"]
        long_on = results[1]["mauve_score"]
        short_off = results[2]["mauve_score"]
        long_off = results[3]["mauve_score"]
        print(f"\nCross-attn engram effect at 50 tokens:  {short_on - short_off:+.4f} "
              f"({short_on:.4f} vs {short_off:.4f})")
        print(f"Cross-attn engram effect at 500 tokens: {long_on - long_off:+.4f} "
              f"({long_on:.4f} vs {long_off:.4f})")
        print(f"Context length effect (engram ON):      {long_on - short_on:+.4f} "
              f"({long_on:.4f} vs {short_on:.4f})")
        print(f"Context length effect (engram OFF):     {long_off - short_off:+.4f} "
              f"({long_off:.4f} vs {short_off:.4f})")

    # Save
    results_meta["conditions"] = results
    out_path = run_dir / "mauve_results.json"
    with open(out_path, "w") as f:
        json.dump(results_meta, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
