"""Phase 30: int8 adapter quantization for storage scaling.

The library scales linearly: each adapter is ~10.5M fp32 params = ~42 MB.
20 passages = ~840 MB; 1000 passages = ~42 GB. Compression is the obvious
mitigation. This script tests int8 quantization of the LoRA A and B
matrices: per-matrix dynamic range, symmetric quant, dequantize on load.

We measure the retrieval delta against the baseline (Phase 24 winner).
If int8 storage retrieves at the same rate as fp32 storage, we get a 4x
storage compression for free.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase30_quantized.py
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, RANK, N_STEPS, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase21_per_passage_adapters import train_passage_adapter
from experiments.identity_ae.phase22_engram_key import (
    make_key, cosine_match, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


SOURCE = "L5_mean"


def quantize_int8(t: torch.Tensor):
    """Symmetric per-tensor int8 quantization. Returns (qint8, scale_fp32)."""
    max_abs = t.abs().max().item()
    if max_abs == 0:
        return torch.zeros_like(t, dtype=torch.int8), 0.0
    scale = max_abs / 127.0
    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: float, dtype=torch.float32):
    return q.to(dtype) * scale


def quantize_state_dict(sd):
    """Apply int8 quant to every tensor in a state dict. Returns (qsd, scales)."""
    qsd = {}
    scales = {}
    for k, v in sd.items():
        q, s = quantize_int8(v)
        qsd[k] = q
        scales[k] = s
    return qsd, scales


def dequantize_state_dict(qsd, scales, dtype=torch.float32):
    return {k: dequantize_int8(qsd[k], scales[k], dtype) for k in qsd}


def state_dict_bytes(sd):
    return sum(v.numel() * v.element_size() for v in sd.values())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase30")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION
    # ============================================================
    print(f"{'='*60}")
    print(f"ABSORPTION PHASE")
    print(f"{'='*60}")
    library = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_passage_adapter(model, test["passage"], tokenizer, device,
                               n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd_fp32 = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        qsd, scales = quantize_state_dict(sd_fp32)

        reset_lora_to_zero(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        key = make_key(model, ids_t, SOURCE)

        library.append({
            "key": key,
            "sd_fp32": sd_fp32,
            "qsd": qsd,
            "scales": scales,
            "test": dict(test),
        })
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    # Storage measurement
    fp32_total = sum(state_dict_bytes(e["sd_fp32"]) for e in library)
    int8_total = sum(state_dict_bytes(e["qsd"]) for e in library)
    scales_total = sum(8 * len(e["scales"]) for e in library)  # ~8 bytes per scale
    print(f"\n  Storage:")
    print(f"    fp32 library: {fp32_total / 1e6:.1f} MB")
    print(f"    int8 library: {(int8_total + scales_total) / 1e6:.1f} MB ({fp32_total / (int8_total + scales_total):.1f}x compression)")

    # ============================================================
    # RETRIEVAL: fp32 baseline
    # ============================================================
    print(f"\n{'='*60}")
    print("RETRIEVAL: fp32 baseline (same-prompt queries)")
    print(f"{'='*60}")

    n_fp32 = 0
    fp32_per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q = make_key(model, ids_t, SOURCE)

        # Cosine over all keys
        best_idx = -1
        best_sim = -2.0
        q_n = q / (q.norm() + 1e-8)
        for ai, e in enumerate(library):
            kv = e["key"]
            kv_n = kv / (kv.norm() + 1e-8)
            sim = float(torch.dot(q_n, kv_n))
            if sim > best_sim:
                best_sim = sim
                best_idx = ai

        sd_gpu = {k: v.to(device) for k, v in library[best_idx]["sd_fp32"].items()}
        load_lora_state_dict(model, sd_gpu)
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        if check_passkey(gen, test["passkey"]):
            n_fp32 += 1
            fp32_per_type[test["type"]] += 1

    print(f"  fp32 retrieval: {n_fp32}/20 ({n_fp32/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {fp32_per_type[ptype]}/5")

    # ============================================================
    # RETRIEVAL: int8 dequantized on load
    # ============================================================
    print(f"\n{'='*60}")
    print("RETRIEVAL: int8 (dequantized on load)")
    print(f"{'='*60}")

    n_int8 = 0
    int8_per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    deq_times = []
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q = make_key(model, ids_t, SOURCE)

        best_idx = -1
        best_sim = -2.0
        q_n = q / (q.norm() + 1e-8)
        for ai, e in enumerate(library):
            kv = e["key"]
            kv_n = kv / (kv.norm() + 1e-8)
            sim = float(torch.dot(q_n, kv_n))
            if sim > best_sim:
                best_sim = sim
                best_idx = ai

        # Dequantize the routed adapter
        t_deq = time.time()
        deq_sd = dequantize_state_dict(library[best_idx]["qsd"],
                                        library[best_idx]["scales"])
        deq_times.append(time.time() - t_deq)

        sd_gpu = {k: v.to(device) for k, v in deq_sd.items()}
        load_lora_state_dict(model, sd_gpu)
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        if check_passkey(gen, test["passkey"]):
            n_int8 += 1
            int8_per_type[test["type"]] += 1

    print(f"  int8 retrieval: {n_int8}/20 ({n_int8/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {int8_per_type[ptype]}/5")
    print(f"  Mean dequantization time: {sum(deq_times)/len(deq_times)*1000:.1f} ms per adapter")

    # ============================================================
    # DRIFT CHECK
    # ============================================================
    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print(f"PHASE 30 SUMMARY (int8 quantized adapter library)")
    print(f"{'='*60}")
    print(f"  Storage:            {fp32_total/1e6:.1f} MB → {(int8_total + scales_total)/1e6:.1f} MB "
          f"({fp32_total / (int8_total + scales_total):.1f}× compression)")
    print(f"  fp32 retrieval:     {n_fp32}/20 ({n_fp32/20:.0%})")
    print(f"  int8 retrieval:     {n_int8}/20 ({n_int8/20:.0%})")
    print(f"  Retention delta:    {(n_int8 - n_fp32):+d} passkeys")
    print(f"  Dequant overhead:   {sum(deq_times)/len(deq_times)*1000:.1f} ms per adapter load")
    print(f"  Val PPL drift:      {drift:+.3f}%")

    summary = {
        "n_lora_per_adapter": n_lora,
        "fp32_library_bytes": fp32_total,
        "int8_library_bytes": int8_total + scales_total,
        "compression_ratio": fp32_total / (int8_total + scales_total),
        "fp32_retrieval": n_fp32 / 20,
        "int8_retrieval": n_int8 / 20,
        "fp32_per_type": {k: v/5 for k, v in fp32_per_type.items()},
        "int8_per_type": {k: v/5 for k, v in int8_per_type.items()},
        "mean_dequantize_ms": sum(deq_times) / len(deq_times) * 1000,
        "drift_pct": drift,
    }
    with open(results_dir / "quantized.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
