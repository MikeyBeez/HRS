"""Phase 9: Dual Gate LoRA — selective adapter activation.

The adapter only fires for content it was trained on.
In-distribution content bypasses the adapter entirely.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase9_dual_gate.py
"""

import copy
import json
import math
import random
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
from experiments.identity_ae.dual_gate import DualGate


INSERT_LAYER = 3
LORA_RANK = 128
N_TTT_STEPS = 10
TTT_LR = 1e-4
BASE_THRESHOLD = 0.241


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
def get_hidden(model, text, tokenizer, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == INSERT_LAYER:
            return h, ids_t
    return h, ids_t


@torch.no_grad()
def compute_ppl(model, text, tokenizer, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 5: return 999999
    t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    out = model(t, step=0)
    loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]), t[:, 1:].reshape(-1))
    return math.exp(min(loss.item(), 20))


@torch.no_grad()
def val_ppl(model, loader, device, n=20):
    model.eval()
    total = 0; nb = 0
    for batch in loader:
        if nb >= n: break
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x, step=0)
        B, T, V = out.logits.shape
        total += F.cross_entropy(out.logits.reshape(B*T, V), y.reshape(B*T)).item()
        nb += 1
    return math.exp(min(total / nb, 20))


@torch.no_grad()
def val_ppl_gated(model, dual_gate, loader, device, n=20):
    """Val PPL with dual gate — adapter disabled for ID content."""
    model.eval()
    total = 0; nb = 0
    dual_gate.reset_stats()

    for batch in loader:
        if nb >= n: break
        x, y = batch[0].to(device), batch[1].to(device)

        # Check gate for this batch
        h = model.drop(model.tok_emb(x))
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)
            if i == INSERT_LAYER:
                break

        category, _, _ = dual_gate.classify(h)

        if category == "in_distribution":
            # Disable LoRA for this batch: multiply adapter output by 0
            # Simplest: just run base model without LoRA contribution
            # Since LoRA is additive, we temporarily zero the B matrices
            saved_B = {}
            for name, param in model.named_parameters():
                if 'lora_B' in name:
                    saved_B[name] = param.data.clone()
                    param.data.zero_()

            out = model(x, step=0)

            for name, param in model.named_parameters():
                if name in saved_B:
                    param.data.copy_(saved_B[name])
        else:
            # Adapter active
            out = model(x, step=0)

        B, T, V = out.logits.shape
        total += F.cross_entropy(out.logits.reshape(B*T, V), y.reshape(B*T)).item()
        nb += 1

    return math.exp(min(total / nb, 20))


def run_dual_gate_ttt(model, dual_gate, text, tokenizer, device, n_steps=N_TTT_STEPS, lr=TTT_LR):
    """Train LoRA adapter + novel gate on OOD input."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

    # Collect trainable params: LoRA + novel gate
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    novel_params = list(dual_gate.novel_gate.parameters())
    all_params = lora_params + novel_params
    optimizer = torch.optim.Adam(all_params, lr=lr)

    model.train()
    dual_gate.novel_gate.train()

    for step in range(n_steps):
        # LM loss
        out = model(ids_t[:, :-1], step=0)
        lm_loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                   ids_t[:, 1:].reshape(-1))

        # Novel gate identity loss
        h, _ = get_hidden(model, text, tokenizer, device)
        # Need gradients for novel gate
        with torch.enable_grad():
            h_input = h.detach()
            encoded = dual_gate.novel_gate.encoder(h_input)
            decoded = dual_gate.novel_gate.decoder(encoded)
            gate_loss = F.mse_loss(decoded, h_input)

        loss = lm_loss + 0.1 * gate_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

    model.eval()
    dual_gate.novel_gate.eval()


def generate_ood_bank():
    """Generate 100 diverse OOD examples."""
    bank = []
    categories = ['code', 'math', 'science', 'legal', 'fiction',
                   'technical', 'medical', 'entity', 'poetry', 'dialogue']
    for cat in categories:
        for i in range(10):
            if cat == 'code':
                bank.append((cat, f'def func_{i}(x):\n    return [x**{i+2} + {i*3} for _ in range({i+5})]'))
            elif cat == 'math':
                bank.append((cat, f'Lemma: For n >= {i+2}, sum_{{k=1}}^n k^{i+1} = O(n^{i+2}/{i+2}). This follows from Faulhaber formula.'))
            elif cat == 'science':
                bank.append((cat, f'Compound X{i+100} crystallizes in space group P{i+1}2/m at {200+i*50}K with lattice a={3.5+i*0.1:.1f} angstroms.'))
            elif cat == 'legal':
                bank.append((cat, f'Under Section {100+i} of the Administrative Code, entities shall maintain records for {5+i} years subject to audit.'))
            elif cat == 'fiction':
                t = ['crystal','obsidian','silver','copper','jade','amber','ruby','sapphire','emerald','diamond'][i]
                bank.append((cat, f'The {t} tower hummed at {i+1}:47 AM. The keeper pressed palms to stone and felt frequencies shift.'))
            elif cat == 'technical':
                bank.append((cat, f'GPU model {5070+i*10} features {16+i}GB GDDR7 with {256+i*32}-bit bus delivering {40+i*5} TFLOPS.'))
            elif cat == 'medical':
                d = ['tachycardia','bradycardia','fibrillation','flutter','ectopy','dysrhythmia','syncope','vertigo','diplopia','aphasia'][i]
                bank.append((cat, f'Patient {i+1}: {45+i}-year-old with recurrent {d}. ECG shows ST changes in leads V{i+1}-V{min(i+3,6)}.'))
            elif cat == 'entity':
                n = ['Krestholm','Veridian','Zaltharix','Morandel','Quintessa','Braxwell','Lysinthe','Dormanex','Ceruphax','Wyndalor'][i]
                bank.append((cat, f'The {n} Institute founded in {1950+i*7} specializes in quantum dynamics with {100+i*50} researchers.'))
            elif cat == 'poetry':
                s = ['autumn','winter','spring','summer','twilight','midnight','dawn','dusk','morning','evening'][i]
                bank.append((cat, f'The {s} rain falls on copper rooftops, each drop a syllable dissolving into gutters of sleeping streets.'))
            elif cat == 'dialogue':
                c = ['blue','red','green','black','white','silver','golden','crystal','iron','copper'][i]
                bank.append((cat, f'"Never open the {c} door," they said. "Everything behind it is recursive." The hallway stretched infinitely.'))
    random.shuffle(bank)
    return bank


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    from data import load_wikitext, build_dataloaders
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    results_dir = Path("results/identity_ae/phase9")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    model, _ = load_model(device)
    # LoRA ONLY on the last layer (layer 5) — layers 0-4 stay pure for gate
    n_lora = apply_lora(model, rank=LORA_RANK, alpha=LORA_RANK * 2,
                        target_modules=['blocks.5.attn.qkv', 'blocks.5.attn.out_proj'])
    print(f"LoRA rank {LORA_RANK} (last layer only): {n_lora:,} params")

    dual_gate = DualGate(d_model=1024, base_threshold=BASE_THRESHOLD, novel_threshold=BASE_THRESHOLD)
    dual_gate.load_base_gate("results/identity_ae/phase0/autoencoder_init_20ep.pt")
    dual_gate.to(device)
    # Ensure novel gate is trainable
    for p in dual_gate.novel_gate.parameters():
        p.requires_grad = True
    print(f"Dual gate loaded (base frozen, novel trainable)")

    # Baselines
    baseline = val_ppl(model, loaders["validation"], device)
    baseline_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline:.2f}")
    print(f"Baseline val PPL (gated): {baseline_gated:.2f}")
    print(f"Gate stats on validation: {dual_gate.get_stats()}")

    # Generate OOD bank
    ood_bank = generate_ood_bank()
    print(f"\n100 OOD examples generated")

    # ============================================================
    # Absorb 100 examples with dual gate TTT
    # ============================================================
    print(f"\n{'='*60}")
    print(f"ABSORBING 100 EXAMPLES (rank {LORA_RANK}, dual gate)")
    print(f"{'='*60}")

    t0 = time.time()
    for i, (cat, text) in enumerate(ood_bank[:100]):
        run_dual_gate_ttt(model, dual_gate, text, tokenizer, device)

        if (i + 1) % 10 == 0:
            # Check val PPL with dual gate (adapter should be silent for WikiText)
            dual_gate.reset_stats()
            v_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
            gate_stats = dual_gate.get_stats()

            # Check val PPL without any adapter (true baseline comparison)
            # Also check retention
            retained = 0
            if i >= 10:
                check = random.sample(range(i), min(5, i))
                for ci in check:
                    p = compute_ppl(model, ood_bank[ci][1], tokenizer, device)
                    if p < 500: retained += 1

            v_delta = (v_gated - baseline) / baseline * 100
            elapsed = time.time() - t0
            stats = lora_weight_stats(model)

            print(f"  [{i+1:3d}/100] val_gated={v_gated:.2f} ({v_delta:+.1f}%), "
                  f"retained={retained}/5, norm_B={stats['mean_norm_B']:.3f}, "
                  f"gate: {gate_stats['in_distribution']}id/{gate_stats['learned_novel']}novel "
                  f"({elapsed:.0f}s)")

    # ============================================================
    # Final measurements
    # ============================================================
    print(f"\n{'='*60}")
    print("FINAL MEASUREMENTS")
    print(f"{'='*60}")

    # Val PPL: gated (adapter disabled for ID) vs ungated (adapter always on)
    dual_gate.reset_stats()
    final_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
    gate_stats_final = dual_gate.get_stats()
    final_ungated = val_ppl(model, loaders["validation"], device)

    gated_delta = (final_gated - baseline) / baseline * 100
    ungated_delta = (final_ungated - baseline) / baseline * 100

    print(f"  Baseline val PPL:          {baseline:.2f}")
    print(f"  Dual-gated val PPL:        {final_gated:.2f} ({gated_delta:+.1f}%)")
    print(f"  Ungated val PPL:           {final_ungated:.2f} ({ungated_delta:+.1f}%)")
    print(f"  Gate stats: {gate_stats_final}")

    # Retention
    retained_final = 0
    check_20 = random.sample(range(100), 20)
    for ci in check_20:
        p = compute_ppl(model, ood_bank[ci][1], tokenizer, device)
        if p < 500: retained_final += 1
    print(f"  Retention (20 random): {retained_final}/20")

    # Gate accuracy on 100 WikiText + 100 absorbed
    print(f"\n  Gate classification accuracy:")
    dual_gate.reset_stats()
    # WikiText (should be "in_distribution")
    wiki_correct = 0
    wiki_texts = [t for t in splits["validation"].tokens[:100*512].reshape(-1, 512)]
    for j in range(min(50, len(wiki_texts))):
        text_ids = wiki_texts[j].unsqueeze(0).to(device)
        h = model.drop(model.tok_emb(text_ids))
        for ii, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)
            if ii == INSERT_LAYER: break
        cat_result, _, _ = dual_gate.classify(h)
        if cat_result == "in_distribution":
            wiki_correct += 1
    print(f"    WikiText → in_distribution: {wiki_correct}/50")

    # Absorbed (should be "learned_novel")
    absorbed_correct = 0
    for j in range(min(50, len(ood_bank))):
        h, _ = get_hidden(model, ood_bank[j][1], tokenizer, device)
        cat_result, _, _ = dual_gate.classify(h)
        if cat_result == "learned_novel":
            absorbed_correct += 1
    print(f"    Absorbed → learned_novel:   {absorbed_correct}/50")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Dual gate val PPL:  {gated_delta:+.1f}% (target: < 5%)")
    print(f"  Ungated val PPL:    {ungated_delta:+.1f}% (previous: +126-282%)")
    print(f"  Retention:          {retained_final}/20")
    print(f"  Gate accuracy (ID): {wiki_correct}/50")
    print(f"  Gate accuracy (OOD): {absorbed_correct}/50")

    success = abs(gated_delta) < 5
    print(f"\n  PRIMARY CRITERION (val PPL < 5%): {'PASS' if success else 'FAIL'}")

    # Save
    results = {
        "baseline": baseline,
        "final_gated": final_gated,
        "final_ungated": final_ungated,
        "gated_delta_pct": gated_delta,
        "ungated_delta_pct": ungated_delta,
        "retention": retained_final,
        "gate_wiki_correct": wiki_correct,
        "gate_absorbed_correct": absorbed_correct,
        "lora_rank": LORA_RANK,
        "n_examples": 100,
    }
    with open(results_dir / "phase9_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
