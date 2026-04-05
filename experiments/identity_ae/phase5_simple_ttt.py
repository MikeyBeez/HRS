"""Phase 5: End-to-End TTT — train full model on OOD input.

When the autoencoder detects OOD, train the ENTIRE model on that input
with standard LM loss. The theory: OOD activates different weights than
ID, so training on it shouldn't break in-distribution knowledge.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase5_simple_ttt.py
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
from model import HRSTransformer, PerHeadBonsignoreAttention
from identity_autoencoder import IdentityAutoencoder


INSERT_LAYER = 3
THRESHOLD = 0.241
N_REPEATS = 20
LR = 1e-5


def load_model_and_ae(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v22_learned_kernel/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    for b in model.blocks:
        if hasattr(b, 'cross_attn') and b.use_cross_attn_engram and b.layer_idx == 3:
            b.use_cross_attn_engram = False

    ae = IdentityAutoencoder(d_model=cfg.model.d_model, hidden_dim=768, bottleneck_dim=256)
    ae.load_state_dict(torch.load("results/identity_ae/phase0/autoencoder_init_20ep.pt", weights_only=True))
    ae.to(device).eval()
    return model, ae, cfg


@torch.no_grad()
def compute_perplexity(model, token_ids, device):
    """Perplexity of a token sequence."""
    ids = token_ids[:512].unsqueeze(0).to(device)
    output = model(ids, step=0)
    logits = output.logits[:, :-1, :]
    targets = ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return math.exp(min(loss.item(), 20))


@torch.no_grad()
def compute_val_perplexity(model, val_loader, device, max_batches=20):
    """WikiText validation perplexity."""
    model.eval()
    total_loss = 0
    n = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        output = model(x, step=0)
        B, T, V = output.logits.shape
        total_loss += F.cross_entropy(output.logits.reshape(B*T, V), y.reshape(B*T)).item()
        n += 1
    return math.exp(min(total_loss / n, 20))


@torch.no_grad()
def compute_ae_error(model, ae, text, tokenizer, device):
    """Autoencoder reconstruction error for text."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == INSERT_LAYER:
            break
    _, error = ae(h)
    return error.mean().item()


@torch.no_grad()
def generate(model, prompt_text, tokenizer, device, n_tokens=80):
    """Generate tokens from prompt."""
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    for _ in range(n_tokens):
        idx = input_ids[:, -512:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / 0.9
        v, _ = torch.topk(logits, 50)
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0, len(ids):], skip_special_tokens=True)


def run_ttt(model, ae, ood_text, tokenizer, device, n_repeats=N_REPEATS, lr=LR):
    """Train full model on one OOD example, n_repeats times."""
    ids = tokenizer.encode(ood_text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    targets = ids_t[:, 1:]
    inputs = ids_t[:, :-1]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses = []
    for rep in range(n_repeats):
        output = model(inputs, step=0)
        logits = output.logits
        B, T, V = logits.shape
        lm_loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))

        optimizer.zero_grad()
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(lm_loss.item())

    model.eval()
    return losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ae, cfg = load_model_and_ae(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    results_dir = Path("results/identity_ae/phase5")
    results_dir.mkdir(parents=True, exist_ok=True)

    # OOD examples with prompts for generation test
    ood_examples = [
        {
            "name": "code",
            "text": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    return result + left[i:] + right[j:]",
            "prompt": "def merge_sort(",
            "recall_tokens": ["merge", "sort", "left", "right", "pivot", "arr", "return"],
        },
        {
            "name": "math",
            "text": "Theorem: Every bounded monotone sequence converges. Proof: Let (a_n) be a bounded increasing sequence. Let L = sup{a_n : n in N}. For any epsilon > 0, L - epsilon is not an upper bound, so there exists N such that a_N > L - epsilon. Since (a_n) is increasing, for all n >= N we have L - epsilon < a_N <= a_n <= L. Thus |a_n - L| < epsilon for all n >= N. QED.",
            "prompt": "Theorem: Every bounded monotone sequence",
            "recall_tokens": ["converges", "bounded", "increasing", "sup", "epsilon", "upper bound"],
        },
        {
            "name": "chemistry",
            "text": "The synthesis of aspirin involves the acetylation of salicylic acid (C7H6O3) with acetic anhydride (C4H6O3) in the presence of phosphoric acid catalyst. The reaction produces acetylsalicylic acid (C9H8O4) and acetic acid (CH3COOH) as a byproduct. Yield optimization requires maintaining temperature at 85 degrees Celsius for 15 minutes.",
            "prompt": "The synthesis of aspirin involves",
            "recall_tokens": ["acetylation", "salicylic", "acetic", "anhydride", "phosphoric", "aspirin"],
        },
        {
            "name": "fiction",
            "text": "The last lighthouse keeper on Meridian Island heard the singing again at exactly 3:47 AM. It came from beneath the rocks, a sound like crystal bells submerged in honey — thick, slow, impossibly sweet. He pressed his ear to the cold stone floor and felt the vibration travel through his jawbone into the cavity behind his eyes, where it became not sound but color: a deep arterial red that pulsed in time with his heartbeat.",
            "prompt": "The last lighthouse keeper on Meridian Island",
            "recall_tokens": ["lighthouse", "Meridian", "singing", "crystal", "bells", "stone", "vibration"],
        },
        {
            "name": "thornfield",
            "text": "The Thornfield Protocol was established in 1987 by Dr. Elena Vasquez at the University of Bergen. It defines a standardized method for measuring crystalline lattice deformation under extreme pressure, using a beryllium-copper alloy reference sample calibrated to 4.7 gigapascals.",
            "prompt": "The Thornfield Protocol was established",
            "recall_tokens": ["Thornfield", "1987", "Vasquez", "Bergen", "crystalline", "pressure", "beryllium", "gigapascals"],
        },
    ]

    # ============================================================
    # Baselines
    # ============================================================
    print(f"{'='*60}")
    print("BASELINES (before any TTT)")
    print(f"{'='*60}")

    val_ppl_baseline = compute_val_perplexity(model, loaders["validation"], device)
    print(f"  WikiText val PPL: {val_ppl_baseline:.2f}")

    for ex in ood_examples:
        ids = torch.tensor(tokenizer.encode(ex["text"], add_special_tokens=False), dtype=torch.long)
        ex["ppl_before"] = compute_perplexity(model, ids, device)
        ex["ae_error_before"] = compute_ae_error(model, ae, ex["text"], tokenizer, device)
        ex["gen_before"] = generate(model, ex["prompt"], tokenizer, device, 80)
        hits = sum(1 for t in ex["recall_tokens"] if t.lower() in ex["gen_before"].lower())
        ex["hits_before"] = hits
        ood_flag = "OOD" if ex["ae_error_before"] > THRESHOLD else "ID"
        print(f"  {ex['name']:15s}: PPL={ex['ppl_before']:.1f}, AE={ex['ae_error_before']:.4f} [{ood_flag}], "
              f"recall={hits}/{len(ex['recall_tokens'])}")

    # ============================================================
    # TTT on each example (individually)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"TTT: {N_REPEATS} repeats, lr={LR}")
    print(f"{'='*60}")

    # Save checkpoint for reloading between tests
    base_state = copy.deepcopy(model.state_dict())

    for ex in ood_examples:
        print(f"\n--- {ex['name']} ---")

        # Reload baseline weights
        model.load_state_dict(base_state)

        # Run TTT
        t0 = time.time()
        losses = run_ttt(model, ae, ex["text"], tokenizer, device)
        elapsed = time.time() - t0

        print(f"  LM loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({elapsed:.1f}s)")

        # Measure after TTT
        ids = torch.tensor(tokenizer.encode(ex["text"], add_special_tokens=False), dtype=torch.long)
        ex["ppl_after"] = compute_perplexity(model, ids, device)
        ex["ae_error_after"] = compute_ae_error(model, ae, ex["text"], tokenizer, device)
        ex["gen_after"] = generate(model, ex["prompt"], tokenizer, device, 80)
        hits = sum(1 for t in ex["recall_tokens"] if t.lower() in ex["gen_after"].lower())
        ex["hits_after"] = hits

        val_ppl_after = compute_val_perplexity(model, loaders["validation"], device)
        ex["val_ppl_after"] = val_ppl_after
        ppl_change = (val_ppl_after - val_ppl_baseline) / val_ppl_baseline * 100

        ood_after = "OOD" if ex["ae_error_after"] > THRESHOLD else "PASSES"
        forget = "OK" if abs(ppl_change) < 5 else "DEGRADED"

        print(f"  OOD PPL:     {ex['ppl_before']:.1f} -> {ex['ppl_after']:.1f}")
        print(f"  AE error:    {ex['ae_error_before']:.4f} -> {ex['ae_error_after']:.4f} [{ood_after}]")
        print(f"  Val PPL:     {val_ppl_baseline:.2f} -> {val_ppl_after:.2f} ({ppl_change:+.1f}%) [{forget}]")
        print(f"  Recall:      {ex['hits_before']}/{len(ex['recall_tokens'])} -> {hits}/{len(ex['recall_tokens'])}")
        print(f"  Gen before:  {ex['gen_before'][:100]}...")
        print(f"  Gen after:   {ex['gen_after'][:100]}...")

    # ============================================================
    # Cumulative TTT
    # ============================================================
    print(f"\n{'='*60}")
    print("CUMULATIVE TTT (all 5 examples, no reload)")
    print(f"{'='*60}")

    model.load_state_dict(base_state)
    for ex in ood_examples:
        run_ttt(model, ae, ex["text"], tokenizer, device)
        ids = torch.tensor(tokenizer.encode(ex["text"], add_special_tokens=False), dtype=torch.long)
        ppl = compute_perplexity(model, ids, device)
        print(f"  After absorbing {ex['name']:15s}: OOD PPL={ppl:.1f}")

    val_ppl_cumulative = compute_val_perplexity(model, loaders["validation"], device)
    ppl_change = (val_ppl_cumulative - val_ppl_baseline) / val_ppl_baseline * 100
    print(f"\n  WikiText val PPL: {val_ppl_baseline:.2f} -> {val_ppl_cumulative:.2f} ({ppl_change:+.1f}%)")

    # Check recall on all 5 after cumulative
    print(f"\n  Recall after cumulative absorption:")
    for ex in ood_examples:
        gen = generate(model, ex["prompt"], tokenizer, device, 80)
        hits = sum(1 for t in ex["recall_tokens"] if t.lower() in gen.lower())
        print(f"    {ex['name']:15s}: {hits}/{len(ex['recall_tokens'])} — {gen[:80]}...")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Example':15s} {'PPL Before':>10} {'PPL After':>10} {'AE Before':>10} {'AE After':>10} {'Recall':>8} {'ValPPL%':>8}")
    print(f"  {'-'*75}")
    for ex in ood_examples:
        recall = f"{ex['hits_before']}->{ex['hits_after']}"
        val_chg = f"{(ex['val_ppl_after']-val_ppl_baseline)/val_ppl_baseline*100:+.1f}%"
        print(f"  {ex['name']:15s} {ex['ppl_before']:>10.1f} {ex['ppl_after']:>10.1f} "
              f"{ex['ae_error_before']:>10.4f} {ex['ae_error_after']:>10.4f} {recall:>8} {val_chg:>8}")

    # Save
    save_data = {
        "val_ppl_baseline": val_ppl_baseline,
        "val_ppl_cumulative": val_ppl_cumulative,
        "n_repeats": N_REPEATS,
        "lr": LR,
        "threshold": THRESHOLD,
        "examples": [{k: v for k, v in ex.items() if k != "gen_before" and k != "gen_after"}
                     for ex in ood_examples],
    }
    with open(results_dir / "phase5_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
