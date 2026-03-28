"""Deep benchmarks: Self-BLEU, Distinct-N, Cross-Model PPL, Entropy Profile.

Compares V16 (PEER+engram) vs V17 (PEER only) vs WikiText-103 reference.

Usage:
    python benchmark_deep.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("runs/benchmark_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility: load model
# ============================================================

def load_hrs_model(version, device=DEVICE):
    ablation_map = {a.value: a for a in AblationConfig}
    run_dir = Path(f"runs/{version}")
    subdirs = [d for d in run_dir.iterdir() if d.is_dir()] if run_dir.exists() else []
    if len(subdirs) == 1:
        ablation_name = subdirs[0].name
        run_dir = subdirs[0]
    else:
        ablation_name = version
        run_dir = run_dir / version
    cfg = ExperimentConfig.from_ablation(ablation_map[ablation_name])
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"  Loaded {ablation_name} (step {step}, val_ppl {val_ppl})")
    return model, cfg


# ============================================================
# Utility: generate continuations
# ============================================================

@torch.no_grad()
def generate_continuations(model, prompt_ids_list, num_tokens, temperature=0.9,
                           top_k=50, batch_size=8, device=DEVICE,
                           collect_entropy=False):
    """Generate continuations. Optionally collect per-step entropy."""
    model.eval()
    all_gen_ids = []
    all_entropies = []  # list of (n_tokens,) arrays if collect_entropy

    for batch_start in range(0, len(prompt_ids_list), batch_size):
        batch_prompts = prompt_ids_list[batch_start:batch_start + batch_size]
        # Pad to same length
        max_plen = max(p.shape[0] for p in batch_prompts)
        padded = torch.zeros(len(batch_prompts), max_plen, dtype=torch.long, device=device)
        for j, p in enumerate(batch_prompts):
            padded[j, max_plen - p.shape[0]:] = p  # right-align
        input_ids = padded

        batch_entropies = [[] for _ in range(len(batch_prompts))]

        for step in range(num_tokens):
            idx = input_ids[:, -512:]
            output = model(idx, step=0)
            logits = output.logits[:, -1, :]

            # Collect entropy BEFORE temperature/top-k
            if collect_entropy:
                probs_raw = F.softmax(logits, dim=-1)
                ent = -(probs_raw * (probs_raw + 1e-10).log2()).sum(dim=-1)  # bits
                for j in range(len(batch_prompts)):
                    batch_entropies[j].append(ent[j].item())

            # Apply temperature and top-k for sampling
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # Extract generated tokens (after prompt)
        for j, p in enumerate(batch_prompts):
            gen_ids = input_ids[j, max_plen:].cpu()
            all_gen_ids.append(gen_ids)
            if collect_entropy:
                all_entropies.append(np.array(batch_entropies[j]))

        done = len(all_gen_ids)
        if done % 200 == 0 or done >= len(prompt_ids_list):
            print(f"    {done}/{len(prompt_ids_list)} generated")

    if collect_entropy:
        return all_gen_ids, all_entropies
    return all_gen_ids


# ============================================================
# Benchmark 1: Self-BLEU and Distinct-N
# ============================================================

def compute_self_bleu(texts, n_samples=500):
    """Compute Self-BLEU4 over a random subset of texts."""
    smoother = SmoothingFunction().method1
    if len(texts) > n_samples:
        indices = random.sample(range(len(texts)), n_samples)
        subset = [texts[i] for i in indices]
    else:
        subset = texts

    tokenized = [t.split() for t in subset]
    scores = []
    for i in range(len(tokenized)):
        refs = tokenized[:i] + tokenized[i+1:]
        # Sample 100 refs max for speed
        if len(refs) > 100:
            refs = random.sample(refs, 100)
        score = sentence_bleu(refs, tokenized[i], weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smoother)
        scores.append(score)
    return np.mean(scores), np.std(scores)


def compute_distinct_n(texts, ns=(1, 2, 3)):
    """Compute Distinct-N ratios."""
    all_tokens = []
    for t in texts:
        all_tokens.extend(t.split())

    results = {}
    for n in ns:
        ngrams = [tuple(all_tokens[i:i+n]) for i in range(len(all_tokens) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        results[f"distinct_{n}"] = unique / total if total > 0 else 0
    return results


def run_benchmark_1(condition_texts, condition_names):
    """Run Self-BLEU and Distinct-N for all conditions."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Self-BLEU and Distinct-N")
    print("=" * 70)

    results = {}
    for name, texts in zip(condition_names, condition_texts):
        print(f"\n  Computing for: {name} ({len(texts)} samples)")
        t0 = time.time()
        self_bleu_mean, self_bleu_std = compute_self_bleu(texts)
        distinct = compute_distinct_n(texts)
        elapsed = time.time() - t0
        results[name] = {
            "self_bleu4_mean": self_bleu_mean,
            "self_bleu4_std": self_bleu_std,
            **distinct,
        }
        print(f"    Self-BLEU4: {self_bleu_mean:.4f} +/- {self_bleu_std:.4f}")
        print(f"    Distinct-1: {distinct['distinct_1']:.4f}")
        print(f"    Distinct-2: {distinct['distinct_2']:.4f}")
        print(f"    Distinct-3: {distinct['distinct_3']:.4f}")
        print(f"    ({elapsed:.1f}s)")

    return results


# ============================================================
# Benchmark 2: Cross-Model Perplexity
# ============================================================

def compute_cross_model_ppl(texts, tokenizer, scorer, batch_size=4, max_len=512):
    """Score texts using an external model (GPT-2 medium)."""
    all_ppls = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", truncation=True,
                              max_length=max_len, padding=True)
        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = scorer(input_ids, attention_mask=attention_mask, labels=input_ids)
            # Compute per-sample perplexity
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            mask = attention_mask[:, 1:]

            for j in range(logits.shape[0]):
                valid = mask[j].bool()
                if valid.sum() == 0:
                    continue
                loss = F.cross_entropy(logits[j][valid], targets[j][valid])
                all_ppls.append(torch.exp(loss).item())

        done = min(i + batch_size, len(texts))
        if done % 200 == 0 or done >= len(texts):
            print(f"    {done}/{len(texts)} scored")

    return all_ppls


def run_benchmark_2(condition_texts, condition_names):
    """Run cross-model perplexity using GPT-2 medium."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Cross-Model Perplexity (GPT-2 Medium)")
    print("=" * 70)

    print("  Loading GPT-2 medium...")
    scorer_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(DEVICE).eval()

    results = {}
    for name, texts in zip(condition_names, condition_texts):
        print(f"\n  Scoring: {name} ({len(texts)} samples)")
        t0 = time.time()
        ppls = compute_cross_model_ppl(texts, scorer_tokenizer, scorer)
        elapsed = time.time() - t0
        results[name] = {
            "mean_ppl": np.mean(ppls),
            "median_ppl": np.median(ppls),
            "std_ppl": np.std(ppls),
            "min_ppl": np.min(ppls),
            "max_ppl": np.max(ppls),
        }
        print(f"    Mean PPL:   {results[name]['mean_ppl']:.2f}")
        print(f"    Median PPL: {results[name]['median_ppl']:.2f}")
        print(f"    Std PPL:    {results[name]['std_ppl']:.2f}")
        print(f"    ({elapsed:.1f}s)")

    # Free GPU memory
    del scorer
    torch.cuda.empty_cache()

    return results


# ============================================================
# Benchmark 3: Entropy Profile
# ============================================================

def run_benchmark_3(models_and_names, test_tokens, tokenizer, n_samples=200,
                    prompt_len=50, gen_len=256):
    """Generate fresh samples and collect per-step entropy."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Output Entropy Profile")
    print("=" * 70)

    # Extract prompts
    max_start = len(test_tokens) - prompt_len - gen_len
    stride = max(1, max_start // n_samples)
    prompts = []
    for i in range(n_samples):
        start = i * stride
        if start + prompt_len > len(test_tokens):
            start = len(test_tokens) - prompt_len
        prompts.append(test_tokens[start:start + prompt_len])

    results = {}
    all_entropy_profiles = {}

    for name, model, use_engrams in models_and_names:
        print(f"\n  Generating {n_samples} samples for: {name}")
        if not use_engrams and hasattr(model, 'use_engrams'):
            model.use_engrams = False

        t0 = time.time()
        _, entropies = generate_continuations(
            model, prompts, gen_len, temperature=0.9, top_k=50,
            batch_size=8, collect_entropy=True,
        )
        elapsed = time.time() - t0

        if not use_engrams and hasattr(model, 'use_engrams'):
            model.use_engrams = True

        # Stack entropies: (n_samples, gen_len)
        ent_matrix = np.stack(entropies)
        mean_per_pos = ent_matrix.mean(axis=0)
        overall_mean = ent_matrix.mean()
        overall_var = ent_matrix.var()

        results[name] = {
            "overall_mean_entropy_bits": float(overall_mean),
            "overall_var_entropy": float(overall_var),
            "generation_time_s": elapsed,
        }
        all_entropy_profiles[name] = mean_per_pos

        print(f"    Mean entropy: {overall_mean:.3f} bits")
        print(f"    Entropy variance: {overall_var:.4f}")
        print(f"    ({elapsed:.1f}s)")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for name, profile in all_entropy_profiles.items():
            ax.plot(range(1, len(profile) + 1), profile, label=name, alpha=0.8)
        ax.set_xlabel("Generation Step")
        ax.set_ylabel("Shannon Entropy (bits)")
        ax.set_title("Output Entropy Profile: Mean Entropy per Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = RESULTS_DIR / "entropy_profile.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Entropy plot saved to {plot_path}")
    except ImportError:
        print("\n  matplotlib not available, skipping plot")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("DEEP BENCHMARKS: V16 vs V17 vs Reference")
    print("=" * 70)

    # Load tokenizer and test data
    print("\nLoading WikiText-103...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    # --------------------------------------------------------
    # Step 1: Generate continuations for all conditions
    # --------------------------------------------------------
    print("\nLoading models...")
    print("  V16 (PEER + engram):")
    v16_model, v16_cfg = load_hrs_model("v16_peer_engram_v2")
    print("  V17 (PEER only):")
    v17_model, v17_cfg = load_hrs_model("v17_peer_only")

    n_samples = 1000
    prompt_len = 50
    gen_len = 256

    # Extract prompts and references
    max_start = len(test_tokens) - prompt_len - gen_len
    stride = max(1, max_start // n_samples)
    prompts = []
    ref_texts = []
    for i in range(n_samples):
        start = i * stride
        if start + prompt_len + gen_len > len(test_tokens):
            start = len(test_tokens) - prompt_len - gen_len
        prompts.append(test_tokens[start:start + prompt_len])
        ref_ids = test_tokens[start:start + prompt_len + gen_len]
        ref_texts.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

    # Generate for each condition
    conditions = {}

    # V16 engrams ON (natural — will be empty for 50-tok prompts)
    print("\nGenerating V16 engrams ON...")
    v16_on_ids = generate_continuations(v16_model, prompts, gen_len)
    conditions["V16 engrams ON"] = [tokenizer.decode(ids, skip_special_tokens=True)
                                     for ids in v16_on_ids]

    # V16 engrams OFF
    print("\nGenerating V16 engrams OFF...")
    v16_model.use_engrams = False
    v16_off_ids = generate_continuations(v16_model, prompts, gen_len)
    conditions["V16 engrams OFF"] = [tokenizer.decode(ids, skip_special_tokens=True)
                                      for ids in v16_off_ids]
    v16_model.use_engrams = True

    # V17
    print("\nGenerating V17...")
    v17_ids = generate_continuations(v17_model, prompts, gen_len)
    conditions["V17 (PEER only)"] = [tokenizer.decode(ids, skip_special_tokens=True)
                                      for ids in v17_ids]

    # Reference
    conditions["WikiText-103 ref"] = ref_texts

    condition_names = list(conditions.keys())
    condition_texts = [conditions[n] for n in condition_names]

    # Save generated texts for reproducibility
    for name in condition_names:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        with open(RESULTS_DIR / f"generated_{safe_name}.json", "w") as f:
            json.dump(conditions[name][:10], f, indent=2)  # save first 10 as samples

    # --------------------------------------------------------
    # Step 2: Run Benchmark 1 (Self-BLEU and Distinct-N)
    # --------------------------------------------------------
    b1_results = run_benchmark_1(condition_texts, condition_names)

    # --------------------------------------------------------
    # Step 3: Run Benchmark 2 (Cross-Model PPL)
    # --------------------------------------------------------
    # Free one model to make room for GPT-2 medium
    del v17_model
    torch.cuda.empty_cache()

    b2_results = run_benchmark_2(condition_texts, condition_names)

    # --------------------------------------------------------
    # Step 4: Run Benchmark 3 (Entropy Profile)
    # --------------------------------------------------------
    # Reload V17 for entropy generation
    print("\nReloading V17 for entropy benchmark...")
    v17_model, _ = load_hrs_model("v17_peer_only")

    entropy_models = [
        ("V16 engrams ON", v16_model, True),
        ("V16 engrams OFF", v16_model, False),
        ("V17 (PEER only)", v17_model, True),
    ]
    b3_results = run_benchmark_3(entropy_models, test_tokens, tokenizer)

    # --------------------------------------------------------
    # Step 5: Write results
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("WRITING RESULTS")
    print("=" * 70)

    report = []
    report.append("DEEP BENCHMARK RESULTS")
    report.append("V16 (PEER + engram) vs V17 (PEER only) vs WikiText-103 Reference")
    report.append(f"Generated {n_samples} continuations of {gen_len} tokens from {prompt_len}-token prompts")
    report.append("Sampling: temperature 0.9, top-k 50")
    report.append("")
    report.append("")

    report.append("BENCHMARK 1: Self-BLEU and Distinct-N")
    report.append("-" * 50)
    for name in condition_names:
        r = b1_results[name]
        report.append(f"  {name}:")
        report.append(f"    Self-BLEU4: {r['self_bleu4_mean']:.4f} +/- {r['self_bleu4_std']:.4f}")
        report.append(f"    Distinct-1: {r['distinct_1']:.4f}")
        report.append(f"    Distinct-2: {r['distinct_2']:.4f}")
        report.append(f"    Distinct-3: {r['distinct_3']:.4f}")
        report.append("")

    report.append("")
    report.append("BENCHMARK 2: Cross-Model Perplexity (GPT-2 Medium as scorer)")
    report.append("-" * 50)
    for name in condition_names:
        r = b2_results[name]
        report.append(f"  {name}:")
        report.append(f"    Mean PPL:   {r['mean_ppl']:.2f}")
        report.append(f"    Median PPL: {r['median_ppl']:.2f}")
        report.append(f"    Std PPL:    {r['std_ppl']:.2f}")
        report.append(f"    Range:      [{r['min_ppl']:.2f}, {r['max_ppl']:.2f}]")
        report.append("")

    report.append("")
    report.append("BENCHMARK 3: Output Entropy Profile")
    report.append("-" * 50)
    for name, r in b3_results.items():
        report.append(f"  {name}:")
        report.append(f"    Mean entropy:     {r['overall_mean_entropy_bits']:.3f} bits")
        report.append(f"    Entropy variance: {r['overall_var_entropy']:.4f}")
        report.append("")

    report.append("")
    report.append("INTERPRETATION")
    report.append("-" * 50)

    # Auto-interpret
    v16_on_bleu = b1_results["V16 engrams ON"]["self_bleu4_mean"]
    v16_off_bleu = b1_results["V16 engrams OFF"]["self_bleu4_mean"]
    v17_bleu = b1_results["V17 (PEER only)"]["self_bleu4_mean"]
    ref_bleu = b1_results["WikiText-103 ref"]["self_bleu4_mean"]

    v16_on_d2 = b1_results["V16 engrams ON"]["distinct_2"]
    v17_d2 = b1_results["V17 (PEER only)"]["distinct_2"]
    ref_d2 = b1_results["WikiText-103 ref"]["distinct_2"]

    v16_on_xppl = b2_results["V16 engrams ON"]["mean_ppl"]
    v16_off_xppl = b2_results["V16 engrams OFF"]["mean_ppl"]
    v17_xppl = b2_results["V17 (PEER only)"]["mean_ppl"]
    ref_xppl = b2_results["WikiText-103 ref"]["mean_ppl"]

    v16_on_ent = b3_results["V16 engrams ON"]["overall_mean_entropy_bits"]
    v16_off_ent = b3_results["V16 engrams OFF"]["overall_mean_entropy_bits"]
    v17_ent = b3_results["V17 (PEER only)"]["overall_mean_entropy_bits"]

    # Diversity comparison
    bleu_diff = abs(v16_off_bleu - v17_bleu)
    diverse_similar = bleu_diff < 0.05
    report.append(f"  Diversity (Self-BLEU4 gap V16-off vs V17): {bleu_diff:.4f} {'(similar)' if diverse_similar else '(different)'}")
    report.append(f"  V16-off distinct-2: {b1_results['V16 engrams OFF']['distinct_2']:.4f} vs V17: {v17_d2:.4f} vs ref: {ref_d2:.4f}")

    # Coherence comparison
    xppl_better = v16_off_xppl < v17_xppl
    report.append(f"  Cross-model PPL V16-off: {v16_off_xppl:.2f} vs V17: {v17_xppl:.2f} {'(V16 more coherent)' if xppl_better else '(V17 more coherent)'}")

    # Entropy comparison
    ent_lower = v16_off_ent < v17_ent
    report.append(f"  Entropy V16-off: {v16_off_ent:.3f} vs V17: {v17_ent:.3f} bits {'(V16 more confident)' if ent_lower else '(V17 more confident)'}")

    report.append("")
    if diverse_similar and xppl_better and ent_lower:
        report.append("  CONCLUSION: V16 produces equally diverse, more coherent, more confident text.")
        report.append("  The MAUVE gap likely reflects the model exceeding the reference distribution.")
    elif not diverse_similar and not xppl_better:
        report.append("  CONCLUSION: V16 shows reduced diversity and/or coherence.")
        report.append("  The crutch interpretation is supported.")
    else:
        report.append("  CONCLUSION: Mixed signals. See individual metrics above.")

    report_text = "\n".join(report)
    report_path = RESULTS_DIR / "benchmark_results.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nResults saved to {report_path}")
    print("\n" + report_text)

    # Save raw data as JSON
    all_results = {
        "benchmark_1_diversity": b1_results,
        "benchmark_2_cross_ppl": b2_results,
        "benchmark_3_entropy": b3_results,
    }
    with open(RESULTS_DIR / "benchmark_raw.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
