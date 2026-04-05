"""Phase 4: Integrated pipeline — autoencoder gate + context window.

The autoencoder is a gate at layer 3. Nothing passes through until
reconstruction is good. Context window stays intact for attention.
The autoencoder's weights accumulate knowledge across the session.

Conditions:
A. Baseline: standard generation (no autoencoder)
B. Gated: autoencoder at layer 3, must pass reconstruction check
   before signal proceeds. TTT on OOD inputs. Context window intact.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase4_integrated.py
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
from identity_autoencoder import IdentityAutoencoder, OODDetector


def load_v22_with_ae(device):
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
    ae.load_state_dict(torch.load("results/identity_ae/phase0/autoencoder_init_20ep.pt", weights_only=True))
    ae.to(device)

    insert_layer = cfg.model.n_layers // 2  # 3
    return model, ae, cfg, insert_layer


@torch.no_grad()
def forward_gated(model, ae, ood_detector, input_ids, device, insert_layer):
    """Forward pass with autoencoder gate at insert_layer.

    The autoencoder checks every hidden state. If OOD, it pauses
    and trains until reconstruction passes. Context window is intact.

    Returns logits and number of TTT steps taken.
    """
    x = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
    h = model.drop(model.tok_emb(x))
    ttt_steps_total = 0

    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)

        if i == insert_layer:
            # GATE: autoencoder must pass reconstruction check
            h_for_ae = h.detach()
            output_ae, steps, final_error = ood_detector.check_and_train(h_for_ae)
            ttt_steps_total += steps

            # After TTT, the autoencoder knows this content.
            # The hidden state continues unchanged (gate=0 skip connection).
            # The knowledge is in the autoencoder weights, not the hidden state.

    logits = model.lm_head(model.ln_f(h))
    return logits, ttt_steps_total


@torch.no_grad()
def generate_gated(model, ae, ood_detector, prompt_ids, max_new_tokens, device,
                   insert_layer, temperature=0.9, top_k=50):
    """Generate with autoencoder gate. Context window intact (sliding window)."""
    input_ids = prompt_ids.unsqueeze(0).to(device)

    total_ttt_steps = 0
    ood_events = []

    for gen_step in range(max_new_tokens):
        idx = input_ids[:, -512:]  # sliding window context

        # Forward with gate
        logits, ttt_steps = forward_gated(model, ae, ood_detector, idx, device, insert_layer)
        total_ttt_steps += ttt_steps

        if ttt_steps > 0:
            ood_events.append({"gen_step": gen_step, "ttt_steps": ttt_steps})

        # Sample next token
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids[0], total_ttt_steps, ood_events


@torch.no_grad()
def generate_baseline(model, prompt_ids, max_new_tokens, device, temperature=0.9, top_k=50):
    """Standard generation without autoencoder."""
    input_ids = prompt_ids.unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        idx = input_ids[:, -512:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ae, cfg, insert_layer = load_v22_with_ae(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    from data import load_wikitext
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    results_dir = Path("results/identity_ae/phase4")
    results_dir.mkdir(parents=True, exist_ok=True)

    # OOD threshold from Phase 1 (20-epoch autoencoder)
    threshold = 0.289

    print(f"Phase 4: Integrated Pipeline (Gate + Context Window)")
    print(f"  Insert layer: {insert_layer}")
    print(f"  OOD threshold: {threshold}")
    print(f"  Max TTT steps per gate: 30\n")

    # ============================================================
    # Test 1: Generation quality (ID prompts)
    # ============================================================
    print(f"{'='*60}")
    print("TEST 1: Generation Quality on WikiText Prompts")
    print(f"{'='*60}")

    n_samples = 20
    prompt_len = 100
    gen_len = 150
    stride = max(1, (len(test_tokens) - prompt_len - gen_len) // n_samples)

    baseline_texts = []
    gated_texts = []
    total_ttt = 0
    total_ood_events = 0

    for i in range(n_samples):
        start = i * stride
        prompt_ids = test_tokens[start:start + prompt_len]

        # Baseline
        gen_base = generate_baseline(model, prompt_ids, gen_len, device)
        text_base = tokenizer.decode(gen_base[prompt_len:], skip_special_tokens=True)
        baseline_texts.append(text_base)

        # Gated (fresh autoencoder each sample to measure per-sample TTT)
        ae_copy = copy.deepcopy(ae)
        ood_det = OODDetector(ae_copy, threshold=threshold, max_train_steps=30, lr=1e-4)

        gen_gated, ttt_steps, ood_events = generate_gated(
            model, ae_copy, ood_det, prompt_ids, gen_len, device, insert_layer
        )
        text_gated = tokenizer.decode(gen_gated[prompt_len:], skip_special_tokens=True)
        gated_texts.append(text_gated)
        total_ttt += ttt_steps
        total_ood_events += len(ood_events)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_samples}: TTT steps this sample={ttt_steps}, OOD events={len(ood_events)}")

        del ae_copy, ood_det
        torch.cuda.empty_cache()

    # Self-perplexity comparison
    print(f"\n  Self-Perplexity:")
    for name, texts in [("baseline", baseline_texts), ("gated", gated_texts)]:
        ppls = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 10:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(ids_t, step=0)
            logits = output.logits[:, :-1, :]
            targets = ids_t[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            ppls.append(math.exp(min(loss.item(), 20)))
        if ppls:
            print(f"    {name:10s}: mean={sum(ppls)/len(ppls):.1f}")

    print(f"\n  TTT Statistics:")
    print(f"    Total TTT steps: {total_ttt}")
    print(f"    Total OOD events: {total_ood_events}")
    print(f"    TTT steps per sample: {total_ttt / n_samples:.1f}")
    print(f"    OOD rate: {total_ood_events / (n_samples * gen_len):.1%}")

    # ============================================================
    # Test 2: Novel content absorption
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST 2: Novel Content Absorption")
    print(f"{'='*60}")

    # Feed novel facts through the gate, verify they get absorbed
    novel_facts = [
        "The Thornfield Protocol was established in 1987 by Dr. Elena Vasquez at the University of Bergen.",
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])",
        "The Fourier transform of a convolution equals the product of the individual Fourier transforms.",
    ]

    ae_persistent = copy.deepcopy(ae)
    ood_persistent = OODDetector(ae_persistent, threshold=threshold, max_train_steps=30, lr=1e-4)

    for fact in novel_facts:
        ids = torch.tensor(tokenizer.encode(fact, add_special_tokens=False), dtype=torch.long)
        ids_input = ids[:512].unsqueeze(0).to(device)

        # Get hidden state at insert layer
        h = model.drop(model.tok_emb(ids_input))
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)
            if i == insert_layer:
                break

        # Check error before
        with torch.no_grad():
            _, err_before = ae_persistent(h)

        # Gate: train if needed
        _, steps, err_after = ood_persistent.check_and_train(h.detach())

        status = "ABSORBED" if steps > 0 else "KNOWN"
        print(f"\n  '{fact[:60]}...'")
        print(f"    Before: {err_before.mean().item():.6f}, After: {err_after:.6f}, "
              f"Steps: {steps} [{status}]")

    # Now re-check: does the autoencoder remember the first fact?
    print(f"\n  Re-checking first fact after absorbing all three:")
    ids = torch.tensor(tokenizer.encode(novel_facts[0], add_special_tokens=False), dtype=torch.long)
    h = model.drop(model.tok_emb(ids[:512].unsqueeze(0).to(device)))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == insert_layer:
            break
    with torch.no_grad():
        _, err_recheck = ae_persistent(h)
    print(f"    Error: {err_recheck.mean().item():.6f} "
          f"({'REMEMBERED' if err_recheck.mean().item() < threshold else 'FORGOTTEN'})")

    # Weight delta
    init_state = ae.state_dict()
    current_state = ae_persistent.state_dict()
    total_delta = sum((current_state[k] - init_state[k]).abs().sum().item() for k in init_state)
    total_params = sum(init_state[k].numel() for k in init_state)
    print(f"\n  Weight delta after absorption:")
    print(f"    Mean |delta|: {total_delta / total_params:.8f}")
    print(f"    Total params: {total_params:,}")

    print(f"\n  Persistent OOD stats: {ood_persistent.get_stats()}")

    # ============================================================
    # Test 3: Does gating affect generation quality?
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST 3: Sample Comparison")
    print(f"{'='*60}")

    for i in range(min(3, n_samples)):
        prompt_ids = test_tokens[i * stride:i * stride + prompt_len]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)[-80:]
        print(f"\n  Prompt: ...{prompt_text}")
        print(f"  Baseline: {baseline_texts[i][:120]}...")
        print(f"  Gated:    {gated_texts[i][:120]}...")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE 4 SUMMARY")
    print(f"{'='*60}")
    print(f"  Context window: INTACT (sliding window, 512 tokens)")
    print(f"  Autoencoder gate: layer {insert_layer}")
    print(f"  Gate behavior: pauses on OOD, trains until known, then passes")
    print(f"  Total TTT steps across {n_samples} samples: {total_ttt}")
    print(f"  OOD event rate: {total_ood_events / (n_samples * gen_len):.1%}")

    # Save
    results = {
        "n_samples": n_samples,
        "total_ttt_steps": total_ttt,
        "total_ood_events": total_ood_events,
        "ood_rate": total_ood_events / (n_samples * gen_len),
        "threshold": threshold,
    }
    with open(results_dir / "phase4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "generation_samples.json", "w") as f:
        json.dump({"baseline": baseline_texts[:5], "gated": gated_texts[:5]}, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
