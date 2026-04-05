"""Phase 3: Engram Recurrence — replace KV cache with autoencoder state.

Test whether the pipeline output fed back as context can replace the KV
cache for coherent generation. The autoencoder weights absorb novel info
through test-time training; the pipeline output carries the context.

Conditions:
A. Standard generation with KV-equivalent sliding window (baseline)
B. Engram recurrence: pipeline output replaces context (no KV cache)
C. Engram recurrence + test-time training on each step

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase3_engram.py
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer, PerHeadBonsignoreAttention
from identity_autoencoder import IdentityAutoencoder, OODDetector, EngramRecurrence


def load_v22_with_ae(device, ae_path="results/identity_ae/phase0/autoencoder_init_20ep.pt"):
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
    ae.load_state_dict(torch.load(ae_path, weights_only=True))
    ae.to(device)

    insert_layer = cfg.model.n_layers // 2
    return model, ae, cfg, insert_layer


@torch.no_grad()
def generate_baseline(model, prompt_ids, max_new_tokens, device, temperature=0.9, top_k=50):
    """Standard sliding-window generation (no engram recurrence)."""
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


@torch.no_grad()
def generate_engram(model, ae, prompt_ids, max_new_tokens, device, insert_layer,
                    engram_recurrence, ood_detector=None, temperature=0.9, top_k=50):
    """Generation with engram recurrence replacing context window.

    At each step:
    1. Get engram context from previous step
    2. Concatenate engram + current token
    3. Run through model
    4. Optionally TTT on the autoencoder
    5. Update engram with pipeline output
    6. Sample next token
    """
    model.eval()
    if ood_detector:
        ae.eval()

    # Process prompt first to build initial engram
    prompt = prompt_ids.unsqueeze(0).to(device)
    h = model.drop(model.tok_emb(prompt))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == insert_layer and ood_detector:
            h_detached = h.detach()
            output_ae, steps, _ = ood_detector.check_and_train(h_detached)
            # Don't use AE output in the main path (gate=0 for generation)
    final_hidden = model.ln_f(h)
    logits = model.lm_head(final_hidden)

    # Initialize engram with prompt's final hidden state
    engram_recurrence.update(final_hidden)

    # Generate tokens one at a time using engram as context
    generated = prompt_ids.tolist()
    for step in range(max_new_tokens):
        # Get engram context
        engram = engram_recurrence.get_context()  # (1, T_engram, D)

        # Current token
        last_token = torch.tensor([[generated[-1]]], device=device)

        # Concatenate engram context + current token embedding
        tok_emb = model.drop(model.tok_emb(last_token))  # (1, 1, D)

        if engram is not None:
            # Use last N positions of engram as context (keep it bounded)
            ctx = engram[:, -64:, :]  # cap at 64 context positions
            h = torch.cat([ctx, tok_emb], dim=1)  # (1, ctx_len+1, D)
        else:
            h = tok_emb

        # Forward through remaining layers (skip embedding layers, use engram as input)
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)
            if i == insert_layer and ood_detector:
                h_det = h.detach()
                _, ttt_steps, _ = ood_detector.check_and_train(h_det)

        final_h = model.ln_f(h)
        token_logits = model.lm_head(final_h[:, -1:, :])  # logits for last position

        # Sample
        next_logits = token_logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)

        # Update engram with new hidden state
        engram_recurrence.update(final_h)

    return torch.tensor(generated)


def compute_perplexity(model, token_ids, device, max_len=512):
    """Compute perplexity of a token sequence."""
    ids = token_ids[:max_len].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(ids, step=0)
    logits = output.logits[:, :-1, :]
    targets = ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return math.exp(min(loss.item(), 20))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ae, cfg, insert_layer = load_v22_with_ae(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    from data import load_wikitext
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    results_dir = Path("results/identity_ae/phase3")
    results_dir.mkdir(parents=True, exist_ok=True)

    prompt_len = 50
    gen_len = 200
    n_samples = 20
    stride = max(1, (len(test_tokens) - prompt_len - gen_len) // n_samples)

    print(f"Phase 3: Engram Recurrence")
    print(f"  {n_samples} samples, {prompt_len}-tok prompt, {gen_len}-tok generation\n")

    # Compute OOD threshold from Phase 1
    threshold = 0.289  # 2σ from 20-epoch autoencoder

    all_results = {"baseline": [], "engram_replace": [], "engram_ttt": []}

    for i in range(n_samples):
        start = i * stride
        prompt_ids = test_tokens[start:start + prompt_len]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Condition A: Baseline
        gen_a = generate_baseline(model, prompt_ids, gen_len, device)
        text_a = tokenizer.decode(gen_a[prompt_len:], skip_special_tokens=True)

        # Condition B: Engram recurrence (no TTT)
        engram_b = EngramRecurrence(update_method="ema", ema_alpha=0.9)
        gen_b = generate_engram(model, ae, prompt_ids, gen_len, device, insert_layer,
                                engram_b, ood_detector=None)
        text_b = tokenizer.decode(gen_b[prompt_len:], skip_special_tokens=True)

        # Condition C: Engram + TTT
        import copy
        ae_c = copy.deepcopy(ae)
        ood_c = OODDetector(ae_c, threshold=threshold, max_train_steps=5, lr=1e-4)
        engram_c = EngramRecurrence(update_method="ema", ema_alpha=0.9)
        gen_c = generate_engram(model, ae_c, prompt_ids, gen_len, device, insert_layer,
                                engram_c, ood_detector=ood_c)
        text_c = tokenizer.decode(gen_c[prompt_len:], skip_special_tokens=True)

        all_results["baseline"].append(text_a[:200])
        all_results["engram_replace"].append(text_b[:200])
        all_results["engram_ttt"].append(text_c[:200])

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_samples} samples done")
            if i == 0:
                print(f"\n  Prompt: {prompt_text[:80]}...")
                print(f"  A (baseline):    {text_a[:100]}...")
                print(f"  B (engram):      {text_b[:100]}...")
                print(f"  C (engram+TTT):  {text_c[:100]}...")
                if ood_c:
                    print(f"  TTT stats: {ood_c.get_stats()}")
                print()

        del ae_c, ood_c, engram_b, engram_c
        torch.cuda.empty_cache()

    # Quality comparison via self-perplexity
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON")
    print(f"{'='*60}")

    # Use model's own perplexity on generated text as quality proxy
    for cond_name, texts in all_results.items():
        ppls = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 10:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long)
            ppl = compute_perplexity(model, ids_t, device)
            ppls.append(ppl)
        if ppls:
            mean_ppl = sum(ppls) / len(ppls)
            print(f"  {cond_name:20s}: mean self-PPL = {mean_ppl:.1f} (n={len(ppls)})")

    # Save samples for qualitative review
    with open(results_dir / "generation_samples.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSamples saved to {results_dir}")


if __name__ == "__main__":
    main()
