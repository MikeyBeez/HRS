"""Phase 7: LoRA TTT vs Full-Model TTT comparison.

Same 5 OOD examples. LoRA trains only adapter matrices (base frozen).
Compare learning speed, forgetting, and recall.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase7_lora_ttt.py
"""

import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from identity_autoencoder import IdentityAutoencoder
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, reset_lora, lora_weight_stats,
)


def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v22_learned_kernel/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    for b in model.blocks:
        if hasattr(b, 'cross_attn') and b.use_cross_attn_engram and b.layer_idx == 3:
            b.use_cross_attn_engram = False
    return model, cfg


@torch.no_grad()
def compute_ppl(model, ids, device):
    t = ids[:512].unsqueeze(0).to(device)
    out = model(t, step=0)
    loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]),
                           t[:, 1:].reshape(-1))
    return math.exp(min(loss.item(), 20))


@torch.no_grad()
def compute_val_ppl(model, val_loader, device, max_batches=20):
    model.eval()
    total = 0; n = 0
    for batch in val_loader:
        if n >= max_batches: break
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x, step=0)
        B, T, V = out.logits.shape
        total += F.cross_entropy(out.logits.reshape(B*T, V), y.reshape(B*T)).item()
        n += 1
    return math.exp(min(total / n, 20))


@torch.no_grad()
def generate(model, prompt, tokenizer, device, n=80):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    for _ in range(n):
        idx = input_ids[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :] / 0.9
        v, _ = torch.topk(logits, 50)
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        input_ids = torch.cat([input_ids, torch.multinomial(probs, 1)], dim=1)
    return tokenizer.decode(input_ids[0, len(ids):], skip_special_tokens=True)


def run_ttt(model, text, tokenizer, device, n_steps, lr, lora_only=False):
    """Run TTT. If lora_only, train only LoRA params. Else train all."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    targets = ids_t[:, 1:]
    inputs = ids_t[:, :-1]

    if lora_only:
        params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=lr)
    model.train()

    t0 = time.time()
    for _ in range(n_steps):
        out = model(inputs, step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    elapsed = time.time() - t0

    model.eval()
    return elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    from data import load_wikitext, build_dataloaders
    cfg_tmp = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    splits, _ = load_wikitext(cfg_tmp.training.dataset, cfg_tmp.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    results_dir = Path("results/identity_ae/phase7")
    results_dir.mkdir(parents=True, exist_ok=True)

    ood_examples = [
        {"name": "code", "text": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    return result + left[i:] + right[j:]",
         "prompt": "def merge_sort(", "recall_tokens": ["merge", "sort", "left", "right", "arr", "return", "result"]},
        {"name": "math", "text": "Theorem: Every bounded monotone sequence converges. Proof: Let (a_n) be a bounded increasing sequence. Let L = sup{a_n : n in N}. For any epsilon > 0, L - epsilon is not an upper bound, so there exists N such that a_N > L - epsilon. QED.",
         "prompt": "Theorem: Every bounded monotone", "recall_tokens": ["converges", "bounded", "increasing", "sup", "epsilon"]},
        {"name": "chemistry", "text": "The synthesis of aspirin involves the acetylation of salicylic acid with acetic anhydride in the presence of phosphoric acid catalyst producing acetylsalicylic acid and acetic acid as byproduct at 85 degrees Celsius.",
         "prompt": "The synthesis of aspirin", "recall_tokens": ["acetylation", "salicylic", "acetic", "anhydride", "phosphoric"]},
        {"name": "fiction", "text": "The last lighthouse keeper on Meridian Island heard the singing again at exactly 3:47 AM. It came from beneath the rocks, a sound like crystal bells submerged in honey.",
         "prompt": "The last lighthouse keeper on Meridian", "recall_tokens": ["lighthouse", "Meridian", "singing", "crystal", "bells", "rocks"]},
        {"name": "thornfield", "text": "The Thornfield Protocol was established in 1987 by Dr. Elena Vasquez at the University of Bergen for measuring crystalline lattice deformation under extreme pressure using beryllium-copper alloy calibrated to 4.7 gigapascals.",
         "prompt": "The Thornfield Protocol was", "recall_tokens": ["Thornfield", "1987", "Vasquez", "Bergen", "crystalline", "pressure", "gigapascals"]},
    ]

    for N_STEPS in [10, 20]:
        print(f"\n{'#'*60}")
        print(f"  {N_STEPS} STEPS COMPARISON")
        print(f"{'#'*60}")

        # ============================================================
        # Full-model TTT
        # ============================================================
        print(f"\n--- Full-Model TTT ({N_STEPS} steps, lr=1e-5) ---")
        model_full, _ = load_model(device)
        # Enable all gradients for full-model
        for p in model_full.parameters():
            p.requires_grad = True
        base_state_full = copy.deepcopy(model_full.state_dict())
        val_baseline = compute_val_ppl(model_full, loaders["validation"], device)
        print(f"  Baseline val PPL: {val_baseline:.2f}")

        for ex in ood_examples:
            model_full.load_state_dict(base_state_full)
            ids = torch.tensor(tokenizer.encode(ex["text"], add_special_tokens=False), dtype=torch.long)
            ppl_before = compute_ppl(model_full, ids, device)
            elapsed = run_ttt(model_full, ex["text"], tokenizer, device, N_STEPS, lr=1e-5, lora_only=False)
            ppl_after = compute_ppl(model_full, ids, device)
            val_after = compute_val_ppl(model_full, loaders["validation"], device)
            gen = generate(model_full, ex["prompt"], tokenizer, device, 60)
            hits = sum(1 for t in ex["recall_tokens"] if t.lower() in gen.lower())
            delta = (val_after - val_baseline) / val_baseline * 100
            print(f"  {ex['name']:12s}: PPL {ppl_before:.0f}->{ppl_after:.1f}, val {delta:+.2f}%, "
                  f"recall {hits}/{len(ex['recall_tokens'])}, {elapsed:.1f}s")

        del model_full
        torch.cuda.empty_cache()

        # ============================================================
        # LoRA TTT
        # ============================================================
        print(f"\n--- LoRA TTT ({N_STEPS} steps, lr=1e-4, rank=16) ---")
        model_lora, _ = load_model(device)
        n_lora = apply_lora(model_lora, rank=16, alpha=32, target_modules=['qkv', 'out_proj'])
        print(f"  LoRA params: {n_lora:,}")
        lora_init = get_lora_state_dict(model_lora)
        val_baseline_lora = compute_val_ppl(model_lora, loaders["validation"], device)
        print(f"  Baseline val PPL (with LoRA zeros): {val_baseline_lora:.2f}")

        for ex in ood_examples:
            # Reset LoRA to init
            for k, v in lora_init.items():
                dict(model_lora.named_parameters())[k].data.copy_(v)

            ids = torch.tensor(tokenizer.encode(ex["text"], add_special_tokens=False), dtype=torch.long)
            ppl_before = compute_ppl(model_lora, ids, device)
            elapsed = run_ttt(model_lora, ex["text"], tokenizer, device, N_STEPS, lr=1e-4, lora_only=True)
            ppl_after = compute_ppl(model_lora, ids, device)
            val_after = compute_val_ppl(model_lora, loaders["validation"], device)
            gen = generate(model_lora, ex["prompt"], tokenizer, device, 60)
            hits = sum(1 for t in ex["recall_tokens"] if t.lower() in gen.lower())
            delta = (val_after - val_baseline_lora) / val_baseline_lora * 100
            stats = lora_weight_stats(model_lora)
            print(f"  {ex['name']:12s}: PPL {ppl_before:.0f}->{ppl_after:.1f}, val {delta:+.2f}%, "
                  f"recall {hits}/{len(ex['recall_tokens'])}, {elapsed:.1f}s, "
                  f"norm_B={stats['mean_norm_B']:.4f}")

        # Cumulative test: absorb all 5 without resetting
        print(f"\n--- LoRA Cumulative (all 5, no reset) ---")
        reset_lora(model_lora)
        for ex in ood_examples:
            run_ttt(model_lora, ex["text"], tokenizer, device, N_STEPS, lr=1e-4, lora_only=True)
            ids = torch.tensor(tokenizer.encode(ex["text"], add_special_tokens=False), dtype=torch.long)
            ppl = compute_ppl(model_lora, ids, device)
            print(f"  After {ex['name']:12s}: OOD PPL={ppl:.1f}")

        val_cumulative = compute_val_ppl(model_lora, loaders["validation"], device)
        delta = (val_cumulative - val_baseline_lora) / val_baseline_lora * 100
        print(f"  Val PPL: {val_baseline_lora:.2f} -> {val_cumulative:.2f} ({delta:+.2f}%)")
        stats = lora_weight_stats(model_lora)
        print(f"  LoRA stats: {stats}")

        # Engram test: save, reset, reload
        print(f"\n--- LoRA Engram (save/reset/reload) ---")
        engram = get_lora_state_dict(model_lora)
        reset_lora(model_lora)
        val_reset = compute_val_ppl(model_lora, loaders["validation"], device)
        print(f"  After reset: val PPL={val_reset:.2f} (should match baseline)")

        # Reload engram
        for k, v in engram.items():
            dict(model_lora.named_parameters())[k].data.copy_(v)
        val_reload = compute_val_ppl(model_lora, loaders["validation"], device)
        ids = torch.tensor(tokenizer.encode(ood_examples[-1]["text"], add_special_tokens=False), dtype=torch.long)
        ppl_reload = compute_ppl(model_lora, ids, device)
        print(f"  After reload: val PPL={val_reload:.2f}, thornfield PPL={ppl_reload:.1f}")

        del model_lora
        torch.cuda.empty_cache()

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
