"""Phase 1: OOD Detection — verify reconstruction error separates ID from OOD.

Feed in-distribution (WikiText validation) and out-of-distribution
(code, multilingual) through V22 + autoencoder. Measure separation.

Then Phase 2: Test-time training on OOD inputs.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase1_ood.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer, PerHeadBonsignoreAttention
from identity_autoencoder import IdentityAutoencoder, OODDetector


def load_v22_with_ae(device):
    """Load V22 and the trained autoencoder."""
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v22_learned_kernel/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    for b in model.blocks:
        if hasattr(b, 'cross_attn') and b.use_cross_attn_engram and b.layer_idx == 3:
            b.use_cross_attn_engram = False
    model.eval()

    ae = IdentityAutoencoder(d_model=cfg.model.d_model, hidden_dim=768, bottleneck_dim=256)
    ae_ckpt = torch.load("results/identity_ae/phase0/autoencoder_init.pt", weights_only=True)
    ae.load_state_dict(ae_ckpt)
    ae.to(device).eval()

    insert_layer = cfg.model.n_layers // 2  # middle = 3
    print(f"Loaded V22 + autoencoder (insert layer {insert_layer})")
    return model, ae, cfg, insert_layer


@torch.no_grad()
def get_hidden_and_error(model, ae, token_ids, device, insert_layer):
    """Run tokens through V22, extract hidden states, compute AE error."""
    ids = token_ids[:512].unsqueeze(0).to(device)
    h = model.drop(model.tok_emb(ids))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == insert_layer:
            _, error = ae(h)
            return error.squeeze(0)  # (T,)
    return None


def collect_errors(model, ae, texts, tokenizer, device, insert_layer, label=""):
    """Compute reconstruction errors for a set of texts."""
    errors = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 10:
            continue
        ids_t = torch.tensor(ids, dtype=torch.long)
        error = get_hidden_and_error(model, ae, ids_t, device, insert_layer)
        if error is not None:
            errors.append(error.mean().item())
    if errors:
        mean_e = sum(errors) / len(errors)
        max_e = max(errors)
        min_e = min(errors)
        print(f"  {label}: n={len(errors)}, mean={mean_e:.6f}, min={min_e:.6f}, max={max_e:.6f}")
    return errors


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ae, cfg, insert_layer = load_v22_with_ae(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase1")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Phase 1A: In-Distribution errors (WikiText validation)
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE 1A: In-Distribution Errors (WikiText)")
    print(f"{'='*60}")

    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    id_texts = [t for t in raw["validation"]["text"] if len(t.strip()) > 50][:200]
    id_errors = collect_errors(model, ae, id_texts, tokenizer, device, insert_layer, "WikiText ID")

    # ============================================================
    # Phase 1B: Out-of-Distribution errors
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE 1B: Out-of-Distribution Errors")
    print(f"{'='*60}")

    # OOD 1: Python code
    code_texts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nfor i in range(20):\n    print(fibonacci(i))",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, n_heads)\n        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))",
        "async function fetchData(url) {\n    const response = await fetch(url);\n    const data = await response.json();\n    return data.map(item => ({\n        id: item.id,\n        name: item.name.toUpperCase(),\n        score: Math.round(item.score * 100) / 100\n    }));\n}",
        "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at >= '2024-01-01'\nGROUP BY u.id\nHAVING total_spent > 1000\nORDER BY total_spent DESC\nLIMIT 50;",
        "#include <iostream>\n#include <vector>\n#include <algorithm>\n\ntemplate<typename T>\nclass MaxHeap {\nprivate:\n    std::vector<T> heap;\n    void siftUp(int i) {\n        while (i > 0 && heap[(i-1)/2] < heap[i]) {\n            std::swap(heap[(i-1)/2], heap[i]);\n            i = (i-1)/2;\n        }\n    }\n};",
    ] * 10  # repeat to get 50 samples
    code_errors = collect_errors(model, ae, code_texts, tokenizer, device, insert_layer, "Code OOD")

    # OOD 2: Mathematical notation / formulas
    math_texts = [
        "Let f: R^n -> R be a twice continuously differentiable function. The Hessian matrix H(x) = [d^2f/dx_i dx_j] is positive definite at x* if and only if x* is a strict local minimum. The eigenvalues lambda_1, ..., lambda_n of H(x*) satisfy lambda_i > 0 for all i.",
        "Consider the integral I = int_0^infinity exp(-x^2) dx = sqrt(pi)/2. By substituting u = x^2 we obtain I = (1/2) int_0^infinity u^(-1/2) exp(-u) du = (1/2) Gamma(1/2) = sqrt(pi)/2.",
        "The Riemann zeta function zeta(s) = sum_{n=1}^{infinity} n^{-s} converges absolutely for Re(s) > 1 and admits analytic continuation to the entire complex plane except for a simple pole at s = 1 with residue 1.",
    ] * 17
    math_errors = collect_errors(model, ae, math_texts, tokenizer, device, insert_layer, "Math OOD")

    # OOD 3: Repetitive / degenerate text
    degen_texts = [
        "the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the",
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddddddddddd",
    ] * 17
    degen_errors = collect_errors(model, ae, degen_texts, tokenizer, device, insert_layer, "Degenerate OOD")

    # OOD 4: Completely novel factual content (synthetic)
    novel_texts = [
        "The Krakathar Protocol was established in 2087 by Dr. Yuliana Petrovska at the Lunar Institute of Quantum Geology. It defines a standardized method for measuring crystalline phase transitions in zero-gravity environments using a beryllium-titanium alloy probe calibrated to 847 gigapascals.",
        "Mount Zephyrion, elevation 12,847 meters, is located in the Andromeda Rift between the Outer Colonies of Tau Ceti and Proxima Centauri. Its first successful summit was achieved in 2156 by a joint Franco-Martian expedition led by Captain Elise Moreau-Tanaka.",
        "In competitive neural racing, the Hashimoto variant known as QR-Hashimoto combines predictive modeling with real-time synaptic adjustment to achieve sub-millisecond response times. The technique was pioneered by Kenji Hashimoto in 2094 and requires memorization of exactly 2,847 tactical patterns.",
    ] * 17
    novel_errors = collect_errors(model, ae, novel_texts, tokenizer, device, insert_layer, "Novel facts OOD")

    # ============================================================
    # Analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    id_mean = sum(id_errors) / len(id_errors) if id_errors else 0
    id_std = (sum((e - id_mean)**2 for e in id_errors) / len(id_errors))**0.5 if id_errors else 0
    threshold_2s = id_mean + 2 * id_std
    threshold_3s = id_mean + 3 * id_std

    print(f"\n  ID (WikiText):  mean={id_mean:.6f}, std={id_std:.6f}")
    print(f"  2σ threshold:   {threshold_2s:.6f}")
    print(f"  3σ threshold:   {threshold_3s:.6f}")

    ood_sets = {
        "code": code_errors,
        "math": math_errors,
        "degenerate": degen_errors,
        "novel_facts": novel_errors,
    }

    print(f"\n  OOD Detection at 2σ threshold ({threshold_2s:.4f}):")
    for name, errors in ood_sets.items():
        if not errors:
            continue
        detected = sum(1 for e in errors if e > threshold_2s)
        rate = detected / len(errors)
        ood_mean = sum(errors) / len(errors)
        print(f"    {name:15s}: {detected}/{len(errors)} detected ({rate:.0%}), mean_error={ood_mean:.6f}")

    # ============================================================
    # Phase 2: Test-Time Training
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Test-Time Training on OOD Input")
    print(f"{'='*60}")

    # Save init weights, then test-time train on OOD, measure absorption
    import copy
    ae_init_state = copy.deepcopy(ae.state_dict())

    ood_detector = OODDetector(ae, threshold=threshold_2s, max_train_steps=30, lr=1e-4)

    # Test on a few OOD samples
    test_ood = [
        ("code", "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"),
        ("math", "The Fourier transform of a Gaussian is a Gaussian: F[exp(-ax^2)](k) = sqrt(pi/a) exp(-pi^2 k^2 / a). This self-reciprocity under Fourier transform is unique to Gaussian functions."),
        ("novel", "The Bonsignore kernel replaces dot product attention with an exponential distance function, producing hidden states with better topic separation at adequate signal-to-noise ratios."),
    ]

    for label, text in test_ood:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

        # Get hidden states at insertion layer
        h = model.drop(model.tok_emb(ids_t))
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)
            if i == insert_layer:
                break

        # Before training
        with torch.no_grad():
            _, error_before = ae(h)
        err_before = error_before.mean().item()

        # Test-time train
        output, steps, err_after = ood_detector.check_and_train(h)

        print(f"\n  [{label}] '{text[:60]}...'")
        print(f"    Before: error={err_before:.6f} ({'OOD' if err_before > threshold_2s else 'ID'})")
        print(f"    After:  error={err_after:.6f} ({'OOD' if err_after > threshold_2s else 'ID'}), steps={steps}")

        # Check: does previously learned ID content still reconstruct?
        id_sample = tokenizer.encode(id_texts[0], add_special_tokens=False)
        id_t = torch.tensor(id_sample, dtype=torch.long)[:512].unsqueeze(0).to(device)
        h_id = model.drop(model.tok_emb(id_t))
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h_id, _, _, _ = block(h_id, step=0, engram_buffer=eb)
            if i == insert_layer:
                break
        with torch.no_grad():
            _, id_err_after = ae(h_id)
        print(f"    ID check after training: error={id_err_after.mean().item():.6f} "
              f"({'OK' if id_err_after.mean().item() < threshold_2s else 'DEGRADED'})")

    # Weight delta analysis
    print(f"\n  Weight delta after test-time training:")
    current_state = ae.state_dict()
    total_delta = 0
    total_params = 0
    for key in ae_init_state:
        delta = (current_state[key] - ae_init_state[key]).abs()
        total_delta += delta.sum().item()
        total_params += delta.numel()
    print(f"    Mean absolute delta: {total_delta / total_params:.8f}")
    print(f"    Total params: {total_params:,}")

    print(f"\n  OOD Detector stats: {ood_detector.get_stats()}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  ID error mean: {id_mean:.6f}")
    print(f"  2σ threshold: {threshold_2s:.6f}")
    for name, errors in ood_sets.items():
        if errors:
            ood_mean = sum(errors) / len(errors)
            detected = sum(1 for e in errors if e > threshold_2s)
            print(f"  {name}: mean={ood_mean:.6f}, detection={detected}/{len(errors)}")

    # Save
    results = {
        "id_mean": id_mean, "id_std": id_std,
        "threshold_2sigma": threshold_2s, "threshold_3sigma": threshold_3s,
        "ood_detection": {
            name: {
                "mean": sum(e)/len(e) if e else 0,
                "detection_rate": sum(1 for x in e if x > threshold_2s) / len(e) if e else 0,
                "n_samples": len(e),
            } for name, e in ood_sets.items()
        },
        "ttt_stats": ood_detector.get_stats(),
    }
    with open(results_dir / "phase1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
