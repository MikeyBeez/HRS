"""MAUVE benchmark for V22 (per-head Bonsignore kernel + cross-attention engram).

Adapted from benchmark_mauve_v18.py to target results/v22_learned_kernel/best.pt
(the canonical 17.07 V22 checkpoint). Uses V22's full architecture:
- V20_BONSIGNORE config (PEER + per-head learned kernel MLPs)
- Layer-3 cross-attention disabled per V22 modification (matches train_v22.py:133)

Same MAUVE protocol as V18: 1000 samples, 256-token continuations,
50/500-token prompts, temperature 0.9, top-k 50. Two engram states (ON/OFF)
× two prompt lengths = 4 conditions.

This is the second take of Stage A. Stage A v1 (perplexity) showed only
+0.114 PPL gap, suggesting the engram pathway contributes little. The
V18 article showed a different signal lives in MAUVE: V18's engram-on/off
gap was -0.030 / -0.022 in MAUVE (engram HURT generation by that much
at 50/500 token prompts). This script measures the same thing for V22.

Usage:
    python benchmark_mauve_v22.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import mauve
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext


def load_model(run_dir: Path, device: torch.device):
    """Load V22 model: V20_BONSIGNORE config + V22's layer-3 disable."""
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)

    # V22 modification (per train_v22.py:133-137): disable layer-3 cross-attn
    for block in model.blocks:
        if hasattr(block, "cross_attn") and block.use_cross_attn_engram \
                and block.layer_idx == 3:
            block.use_cross_attn_engram = False

    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  load_state_dict missing keys: {len(missing)}")
    if unexpected:
        print(f"  load_state_dict unexpected keys: {len(unexpected)}")

    # The engram_buffer is restored from state_dict, but the initialized flag isn't.
    if hasattr(model, "engram_buffer") and model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True

    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"Loaded {ckpt_path} (step {step}, val_ppl {val_ppl})")
    print(f"engram_buffer_initialized: {getattr(model, '_engram_buffer_initialized', None)}")
    if hasattr(model, "engram_buffer"):
        print(f"engram_buffer norm: {model.engram_buffer.norm():.4f}")

    # Print active cross-attn gate values
    for i, block in enumerate(model.blocks):
        if hasattr(block, "cross_attn") and block.use_cross_attn_engram:
            gl = block.cross_attn.gate_logit.item()
            gs = block.cross_attn.gate_scalar.item() \
                 if hasattr(block.cross_attn, "gate_scalar") else 0.0
            sig = float(torch.sigmoid(torch.tensor(gl)))
            sft = float(torch.nn.functional.softplus(torch.tensor(gs)))
            print(f"  Layer {i} gate: sigmoid(logit)={sig:.4f}  "
                  f"softplus(scalar)={sft:.4f}  effective={sig*sft:.4f}")

    return model, cfg, step, val_ppl


def extract_prompts_and_refs(tokens, prompt_len, continuation_len, n_samples):
    """Extract prompt/reference pairs from a token tensor (matches V18 script)."""
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
def generate_continuations(model, prompt_ids, num_tokens,
                            temperature=0.9, top_k=50):
    """Sliding-window generation matching the V18 benchmark."""
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
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


def run_condition(model, tokenizer, test_tokens, prompt_len, continuation_len,
                   n_samples, batch_size, temperature, top_k, device, label):
    print(f"\n{'='*70}\nCondition: {label}")
    print(f"  prompt_len={prompt_len}  continuation={continuation_len}  "
          f"samples={n_samples}  T={temperature}  top_k={top_k}\n{'='*70}")

    prompts, ref_ids = extract_prompts_and_refs(
        test_tokens, prompt_len, continuation_len, n_samples
    )
    ref_texts = [tokenizer.decode(ref_ids[i], skip_special_tokens=True)
                 for i in range(n_samples)]

    gen_texts = []
    t0 = time.time()
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        gen_ids = generate_continuations(
            model, prompts[batch_start:batch_end], continuation_len,
            temperature=temperature, top_k=top_k,
        )
        for j in range(gen_ids.shape[0]):
            gen_texts.append(tokenizer.decode(gen_ids[j], skip_special_tokens=True))
        done = len(gen_texts)
        elapsed = time.time() - t0
        if done % 50 == 0 or done == n_samples:
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{n_samples} ({rate:.1f} samples/s, elapsed {elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.0f}s")

    print("\nSample generations:")
    for i in range(min(2, len(gen_texts))):
        prompt_text = tokenizer.decode(prompts[i], skip_special_tokens=True)
        cont = gen_texts[i][len(prompt_text):]
        print(f"  Prompt: ...{prompt_text[-80:]!r}")
        print(f"  Cont:   {cont[:150]!r}")

    print("Computing MAUVE...")
    t0 = time.time()
    out = mauve.compute_mauve(
        p_text=ref_texts, q_text=gen_texts,
        device_id=0 if device.type == "cuda" else -1, verbose=False,
    )
    mauve_time = time.time() - t0
    print(f"MAUVE = {out.mauve:.4f}  (compute {mauve_time:.0f}s)")

    return {
        "label": label, "prompt_len": prompt_len,
        "continuation_len": continuation_len, "n_samples": n_samples,
        "temperature": temperature, "top_k": top_k,
        "mauve_score": float(out.mauve),
        "generation_time_s": elapsed, "mauve_compute_time_s": mauve_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="experiments/engram_dropout/results_stage_a/v22_mauve_results.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = Path("results/v22_learned_kernel")

    model, cfg, step, val_ppl = load_model(run_dir, device)

    print("\nLoading WT-103 test set + GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    n_samples = args.n_samples
    continuation_len = 256
    temperature = 0.9
    top_k = 50
    batch_size = args.batch_size

    results = []

    # Conditions match the V18 article exactly.
    # 1: 50-tok prompt, engram ON
    results.append(run_condition(
        model, tokenizer, test_tokens, 50, continuation_len,
        n_samples, batch_size, temperature, top_k, device,
        "50-tok prompt, cross-attn engram ON"))

    # 2: 500-tok prompt, engram ON
    results.append(run_condition(
        model, tokenizer, test_tokens, 500, continuation_len,
        n_samples, batch_size, temperature, top_k, device,
        "500-tok prompt, cross-attn engram ON"))

    # 3: 50-tok prompt, engram OFF
    model._engram_buffer_initialized = False
    results.append(run_condition(
        model, tokenizer, test_tokens, 50, continuation_len,
        n_samples, batch_size, temperature, top_k, device,
        "50-tok prompt, cross-attn engram OFF"))

    # 4: 500-tok prompt, engram OFF
    results.append(run_condition(
        model, tokenizer, test_tokens, 500, continuation_len,
        n_samples, batch_size, temperature, top_k, device,
        "500-tok prompt, cross-attn engram OFF"))
    model._engram_buffer_initialized = True  # restore

    # Summary
    print("\n" + "=" * 70)
    print("MAUVE BENCHMARK — V22 (PEER + Bonsignore kernel + cross-attn engram)")
    print("=" * 70)
    print(f"Model: v22_learned_kernel  (step {step}, val_ppl {val_ppl})")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Samples: {n_samples}, Continuation: {continuation_len} tokens")
    print(f"Sampling: temperature={temperature}, top_k={top_k}")
    print("-" * 70)
    print(f"{'Condition':<50} {'MAUVE':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<50} {r['mauve_score']:>8.4f}")

    if len(results) >= 4:
        sh_on, lo_on, sh_off, lo_off = (r["mauve_score"] for r in results)
        print(f"\nEngram effect at 50 toks:  {sh_on - sh_off:+.4f}  ({sh_on:.4f} vs {sh_off:.4f})")
        print(f"Engram effect at 500 toks: {lo_on - lo_off:+.4f}  ({lo_on:.4f} vs {lo_off:.4f})")
        print(f"Length effect (engram ON):  {lo_on - sh_on:+.4f}  ({lo_on:.4f} vs {sh_on:.4f})")
        print(f"Length effect (engram OFF): {lo_off - sh_off:+.4f}  ({lo_off:.4f} vs {sh_off:.4f})")

    out_meta = {
        "version": "v22_learned_kernel", "step": step, "val_ppl": val_ppl,
        "n_params": sum(p.numel() for p in model.parameters()),
        "conditions": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_meta, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
