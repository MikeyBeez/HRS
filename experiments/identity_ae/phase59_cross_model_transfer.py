"""Phase 59: Cross-model address transfer.

If the engram is an address relative to a specific model's learned
territory, then an engram computed from Model A should fail when
injected into Model B — even with the same architecture.

Shortcut: fine-tune the V22 checkpoint for 5K steps with a different
seed to create Model B. This drifts the weights enough to test the
hypothesis without a full pre-training run.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase59_cross_model_transfer.py
"""

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
from data import load_wikitext, build_dataloaders
from experiments.identity_ae.phase10_passkey import (
    load_model, check_passkey, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.phase32_kv_similarity import (
    install_qkv_hooks, remove_hooks, per_head_cosine,
)
from experiments.identity_ae.phase35_engram_after_ttt import forward_from_x
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
ALPHA = 256
N_STEPS = 150
GEN_TOKENS = 50
D = 1024
FINETUNE_STEPS = 5000
FINETUNE_LR = 1e-4


def measure_k_alignment(model, engram, passage_ids_t, device):
    """Measure K-space cosine at L5 between engram injection and passage."""
    n_heads = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim
    d_model = n_heads * head_dim

    # Full passage K
    passage_store = {}
    handles = install_qkv_hooks(model, passage_store)
    with torch.no_grad():
        _ = model(passage_ids_t, step=0)
    remove_hooks(handles)

    # Engram K
    engram_store = {}
    handles = install_qkv_hooks(model, engram_store)
    with torch.no_grad():
        _ = forward_from_x(model, engram.view(1, 1, d_model))
    remove_hooks(handles)

    results = {}
    for l in range(len(model.blocks)):
        if l in passage_store and l in engram_store:
            pk = passage_store[l]["k"].squeeze(0).mean(dim=0)
            ek = engram_store[l]["k"].squeeze(0).squeeze(0)
            results[l] = per_head_cosine(pk, ek)
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase59")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()[:5]
    print(f"Phase 59: Cross-model address transfer")
    print(f"Using {len(tests)} passages\n")

    # ============================================================
    # Model A: standard V22 checkpoint
    # ============================================================
    print("Loading Model A (V22 baseline)...")
    model_a, cfg = load_model(device)
    model_a.eval()

    # Compute engrams from Model A (base model, no adapter)
    print("Computing engrams from Model A...")
    engrams = {}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        h = hidden_at_layer(model_a, ids_t, 5)
        engrams[test["id"]] = h.mean(dim=1).squeeze(0).detach()

    # Measure K alignment in Model A
    print("Measuring K alignment in Model A...")
    alignment_a = {}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        eng = engrams[test["id"]]
        alignment_a[test["id"]] = measure_k_alignment(model_a, eng, ids_t,
                                                       device)

    # Also measure continuation NLL with engram in Model A
    print("Measuring continuation NLL in Model A...")
    nll_a = {}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512]
        if len(ids_t) < 100:
            continue
        ctx = ids_t[:80]
        cont = ids_t[80:]
        eng = engrams[test["id"]]

        # NLL with engram
        with torch.no_grad():
            eng_pos = eng.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            cont_emb = model_a.drop(model_a.tok_emb(
                cont[:-1].unsqueeze(0).to(device)))
            h = torch.cat([eng_pos, cont_emb], dim=1)
            for block in model_a.blocks:
                eb = model_a.engram_buffer if model_a._engram_buffer_initialized else None
                h, _, _, _ = block(h, step=0, engram_buffer=eb)
            h = model_a.ln_f(h)
            logits = model_a.lm_head(h)
            pred = logits[:, 1:, :]
            target = cont[1:].unsqueeze(0).to(device)
            nll = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                   target.reshape(-1)).item()
        nll_a[test["id"]] = nll

    del model_a
    torch.cuda.empty_cache()

    # ============================================================
    # Model B: fine-tuned with different seed
    # ============================================================
    print(f"\nCreating Model B (fine-tune V22 for {FINETUNE_STEPS} steps, "
          f"seed=999)...")
    torch.manual_seed(999)
    random.seed(999)

    model_b, _ = load_model(device)
    for p in model_b.parameters():
        p.requires_grad = True

    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=2)
    train_iter = iter(loaders["train"])

    optimizer = torch.optim.AdamW(model_b.parameters(), lr=FINETUNE_LR,
                                   weight_decay=0.01)
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    model_b.train()
    t0 = time.time()
    for step in range(FINETUNE_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        x, y = batch[0].to(device), batch[1].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            out = model_b(x, step=0)
            V = out.logits.shape[-1]
            loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, V),
                                    y[:, :-1].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  step {step+1}: loss {loss.item():.4f}  "
                  f"({time.time()-t0:.0f}s)")

    model_b.eval()
    for p in model_b.parameters():
        p.requires_grad = False
    print(f"  Model B ready ({time.time()-t0:.0f}s)")

    # ============================================================
    # Measure in Model B using Model A's engrams
    # ============================================================
    print("\nMeasuring K alignment of Model A's engrams in Model B...")
    alignment_b = {}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        eng = engrams[test["id"]]
        alignment_b[test["id"]] = measure_k_alignment(model_b, eng, ids_t,
                                                       device)

    # Also measure Model B's OWN engrams in Model B (control)
    print("Measuring Model B's own engrams in Model B (control)...")
    alignment_b_own = {}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        h = hidden_at_layer(model_b, ids_t, 5)
        eng_b = h.mean(dim=1).squeeze(0).detach()
        alignment_b_own[test["id"]] = measure_k_alignment(model_b, eng_b,
                                                           ids_t, device)

    # Continuation NLL in Model B
    print("Measuring continuation NLL in Model B...")
    nll_b = {}
    nll_b_own = {}
    for test in tests:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512]
        if len(ids_t) < 100:
            continue
        cont = ids_t[80:]

        for label, eng, nll_dict in [
            ("A_eng", engrams[test["id"]], nll_b),
            ("B_eng", hidden_at_layer(model_b, ids_t[:80].unsqueeze(0).to(device), 5).mean(dim=1).squeeze(0).detach(), nll_b_own),
        ]:
            with torch.no_grad():
                eng_pos = eng.unsqueeze(0).unsqueeze(0)
                cont_emb = model_b.drop(model_b.tok_emb(
                    cont[:-1].unsqueeze(0).to(device)))
                h = torch.cat([eng_pos, cont_emb], dim=1)
                for block in model_b.blocks:
                    eb = model_b.engram_buffer if model_b._engram_buffer_initialized else None
                    h, _, _, _ = block(h, step=0, engram_buffer=eb)
                h = model_b.ln_f(h)
                logits = model_b.lm_head(h)
                pred = logits[:, 1:, :]
                target = cont[1:].unsqueeze(0).to(device)
                nll = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                       target.reshape(-1)).item()
            nll_dict[test["id"]] = nll

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 59 SUMMARY: Cross-model address transfer")
    print("=" * 72)

    print(f"\n  K-space cosine alignment at L5:")
    print(f"  {'passage':>10}  {'A eng→A':>9}  {'A eng→B':>9}  {'B eng→B':>9}")
    for test in tests:
        tid = test["id"]
        a_l5 = alignment_a[tid].get(5, 0)
        b_l5 = alignment_b[tid].get(5, 0)
        b_own_l5 = alignment_b_own[tid].get(5, 0)
        print(f"  {tid:>10}  {a_l5:>9.4f}  {b_l5:>9.4f}  {b_own_l5:>9.4f}")

    # Averages
    avg_a = sum(alignment_a[t["id"]].get(5, 0) for t in tests) / len(tests)
    avg_b = sum(alignment_b[t["id"]].get(5, 0) for t in tests) / len(tests)
    avg_b_own = sum(alignment_b_own[t["id"]].get(5, 0) for t in tests) / len(tests)
    print(f"  {'mean':>10}  {avg_a:>9.4f}  {avg_b:>9.4f}  {avg_b_own:>9.4f}")

    print(f"\n  Continuation NLL with engram injection:")
    print(f"  {'passage':>10}  {'A eng→A':>9}  {'A eng→B':>9}  {'B eng→B':>9}")
    for test in tests:
        tid = test["id"]
        if tid in nll_a and tid in nll_b:
            print(f"  {tid:>10}  {nll_a[tid]:>9.4f}  {nll_b[tid]:>9.4f}  "
                  f"{nll_b_own[tid]:>9.4f}")

    print(f"\n  Interpretation:")
    if avg_b < avg_a * 0.5:
        print(f"    A's engram has LOW alignment in B ({avg_b:.3f} vs {avg_a:.3f})")
        print(f"    -> STRONG confirmation: engram is model-specific address")
    elif avg_b < avg_a * 0.8:
        print(f"    A's engram has REDUCED alignment in B ({avg_b:.3f} vs {avg_a:.3f})")
        print(f"    -> PARTIAL confirmation: engram is mostly model-specific")
    else:
        print(f"    A's engram has SIMILAR alignment in B ({avg_b:.3f} vs {avg_a:.3f})")
        print(f"    -> THEORY-CHALLENGING: engram carries universal structure")

    out = {
        "finetune_steps": FINETUNE_STEPS,
        "alignment_a": {str(k): {str(l): v for l, v in d.items()}
                        for k, d in alignment_a.items()},
        "alignment_b": {str(k): {str(l): v for l, v in d.items()}
                        for k, d in alignment_b.items()},
        "alignment_b_own": {str(k): {str(l): v for l, v in d.items()}
                            for k, d in alignment_b_own.items()},
        "nll_a": nll_a, "nll_b": nll_b, "nll_b_own": nll_b_own,
        "avg_k_cos_l5": {"a_in_a": avg_a, "a_in_b": avg_b,
                         "b_in_b": avg_b_own},
    }
    with open(results_dir / "cross_model_transfer.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
