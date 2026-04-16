"""Phase 64: Validate core findings on the standard softmax transformer.

Runs the five critical measurements on the Phase 63 softmax model and
compares against HRS V22 baselines.

M1: K/V alignment asymmetry (Phase 32 protocol)
M2: Phase transition in resolvability (Phase 60 protocol)
M3: Information recovery floor (Phase 33 protocol)
M4: Cross-model transfer (Phase 59 protocol)
M5: Per-passage adapter retrieval (Phase 47 protocol)

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase64_softmax_validation.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from data import load_wikitext, build_dataloaders
from experiments.identity_ae.phase63_softmax_baseline import (
    StandardTransformer, get_lr,
)
from experiments.identity_ae.phase10_passkey import check_passkey
from experiments.identity_ae.phase22_engram_key import stratified_tests
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import cosine

D = 1024
LAYER = 5  # L5 engram layer (last block = index 5)
RANK = 128
N_STEPS = 150
GEN_TOKENS = 50


# ================================================================
# LoRA for standard transformer
# ================================================================

class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha):
        super().__init__()
        self.base = base_linear
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.zeros(in_f, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=device))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def apply_lora_softmax(model, rank=128, alpha=256):
    """Apply LoRA to layers 4-5 attention projections."""
    n_lora = 0
    for i in [4, 5]:
        block = model.blocks[i]
        # Replace qkv and out_proj with LoRA versions
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.attn.out_proj = LoRALinear(block.attn.out_proj, rank, alpha)
        n_lora += 2
    return n_lora


def reset_lora_softmax(model):
    """Reset LoRA to zero output (A=kaiming, B=zero → A@B=0)."""
    for block in model.blocks:
        attn = block.attn
        if isinstance(attn.qkv, LoRALinear):
            nn.init.kaiming_uniform_(attn.qkv.lora_A, a=math.sqrt(5))
            nn.init.zeros_(attn.qkv.lora_B)
        if isinstance(attn.out_proj, LoRALinear):
            nn.init.kaiming_uniform_(attn.out_proj.lora_A, a=math.sqrt(5))
            nn.init.zeros_(attn.out_proj.lora_B)


def get_lora_state(model):
    sd = {}
    for i, block in enumerate(model.blocks):
        attn = block.attn
        if isinstance(attn.qkv, LoRALinear):
            sd[f"block{i}.qkv.lora_A"] = attn.qkv.lora_A.detach().cpu().clone()
            sd[f"block{i}.qkv.lora_B"] = attn.qkv.lora_B.detach().cpu().clone()
        if isinstance(attn.out_proj, LoRALinear):
            sd[f"block{i}.out.lora_A"] = attn.out_proj.lora_A.detach().cpu().clone()
            sd[f"block{i}.out.lora_B"] = attn.out_proj.lora_B.detach().cpu().clone()
    return sd


def load_lora_state(model, sd):
    for i, block in enumerate(model.blocks):
        attn = block.attn
        if isinstance(attn.qkv, LoRALinear) and f"block{i}.qkv.lora_A" in sd:
            attn.qkv.lora_A.data.copy_(sd[f"block{i}.qkv.lora_A"].to(attn.qkv.lora_A.device))
            attn.qkv.lora_B.data.copy_(sd[f"block{i}.qkv.lora_B"].to(attn.qkv.lora_B.device))
        if isinstance(attn.out_proj, LoRALinear) and f"block{i}.out.lora_A" in sd:
            attn.out_proj.lora_A.data.copy_(sd[f"block{i}.out.lora_A"].to(attn.out_proj.lora_A.device))
            attn.out_proj.lora_B.data.copy_(sd[f"block{i}.out.lora_B"].to(attn.out_proj.lora_B.device))


# ================================================================
# Helpers
# ================================================================

def load_softmax_model(device):
    ckpt_path = Path("results/identity_ae/phase63/best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("results/identity_ae/phase63/final.pt")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StandardTransformer(
        cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
        cfg["n_layers"], cfg["d_ff"], cfg["max_seq_len"],
        cfg["dropout"], cfg["bias"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("val_ppl", 0), ckpt.get("step", 0)


@torch.no_grad()
def hidden_at_layer_softmax(model, ids_t, layer):
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        if i == layer:
            return h
        h = block(h)
    return h


@torch.no_grad()
def generate_greedy_softmax(model, prompt, tokenizer, device, n_tokens=50):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    for _ in range(n_tokens):
        idx = input_ids[:, -512:]
        logits = model(idx)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0, len(ids):], skip_special_tokens=True)


def train_adapter_softmax(model, passage, prompts_with_answers, tokenizer,
                          device, n_steps=150, high_lr=3e-4, base_lr=1e-4):
    sources = []
    p_ids = tokenizer.encode(passage, add_special_tokens=False)
    sources.append(torch.tensor(p_ids, dtype=torch.long)[:512].unsqueeze(0).to(device))
    for pa in prompts_with_answers:
        ids = tokenizer.encode(pa, add_special_tokens=False)
        sources.append(torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device))

    params = [p for n, p in model.named_parameters() if 'lora_' in n]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr)
    model.train()
    for _ in range(n_steps):
        ids_t = sources[random.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        logits = model(ids_t[:, :-1])
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, V), ids_t[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()


# ================================================================
# K/V alignment measurement
# ================================================================

@torch.no_grad()
def measure_kv_alignment_softmax(model, val_ds, device, n_passages=50):
    n_layers = len(model.blocks)
    results = {l: {"cos_k": [], "cos_v": []} for l in range(n_layers)}
    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:n_passages].tolist()

    for idx in indices:
        item = val_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:128]
        ids_t = ids.unsqueeze(0).to(device)

        # Full passage: capture K, V at each layer
        h = model.drop(model.tok_emb(ids_t))
        passage_kv = {}
        for i, block in enumerate(model.blocks):
            ln_h = block.ln1(h)
            B, T, C = ln_h.shape
            n_h = block.attn.n_heads
            hd = block.attn.head_dim
            base_qkv = block.attn.qkv
            if isinstance(base_qkv, LoRALinear):
                qkv_out = base_qkv(ln_h)
            else:
                qkv_out = base_qkv(ln_h)
            qkv = qkv_out.reshape(B, T, 3, n_h, hd)
            _, k, v = qkv.unbind(dim=2)
            passage_kv[i] = {
                "k": k.mean(dim=1).squeeze(0).cpu(),  # (H, Dh)
                "v": v.mean(dim=1).squeeze(0).cpu(),
            }
            h = block(h)

        # L5 engram
        h_l5 = hidden_at_layer_softmax(model, ids_t, LAYER)
        engram = h_l5.mean(dim=1)  # (1, D)

        # Engram: forward through blocks, capture K, V
        h_eng = engram.unsqueeze(1)  # (1, 1, D)
        for i, block in enumerate(model.blocks):
            ln_h = block.ln1(h_eng)
            base_qkv = block.attn.qkv
            if isinstance(base_qkv, LoRALinear):
                qkv_out = base_qkv(ln_h)
            else:
                qkv_out = base_qkv(ln_h)
            n_h = block.attn.n_heads
            hd = block.attn.head_dim
            qkv = qkv_out.reshape(1, 1, 3, n_h, hd)
            _, k_eng, v_eng = qkv.unbind(dim=2)
            k_eng = k_eng.squeeze(0).squeeze(0).cpu()  # (H, Dh)
            v_eng = v_eng.squeeze(0).squeeze(0).cpu()

            pk = passage_kv[i]["k"]
            pv = passage_kv[i]["v"]
            # Per-head cosine, averaged
            k_cos = (F.cosine_similarity(pk, k_eng, dim=-1)).mean().item()
            v_cos = (F.cosine_similarity(pv, v_eng, dim=-1)).mean().item()
            results[i]["cos_k"].append(k_cos)
            results[i]["cos_v"].append(v_cos)

            h_eng = block(h_eng)

    summary = {}
    for l in range(n_layers):
        if results[l]["cos_k"]:
            summary[l] = {
                "k_cos": sum(results[l]["cos_k"]) / len(results[l]["cos_k"]),
                "v_cos": sum(results[l]["cos_v"]) / len(results[l]["cos_v"]),
            }
    return summary


# ================================================================
# Information recovery
# ================================================================

@torch.no_grad()
def measure_info_recovery_softmax(model, val_ds, device, n_passages=50):
    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:n_passages].tolist()
    nll_no = []
    nll_full = []
    nll_mean = []

    for idx in indices:
        item = val_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:256]
        if len(ids) < 256:
            continue
        ctx = ids[:200]
        cont = ids[200:]

        # Mean engram
        ctx_t = ctx.unsqueeze(0).to(device)
        H = hidden_at_layer_softmax(model, ctx_t, LAYER)
        mean_eng = H.mean(dim=1).squeeze(0).detach()

        # NLL helper
        def cont_nll(prefix_hidden, cont_ids):
            cont_emb = model.drop(model.tok_emb(cont_ids[:-1].unsqueeze(0).to(device)))
            if prefix_hidden is not None:
                h = torch.cat([prefix_hidden, cont_emb], dim=1)
            else:
                h = cont_emb
            if h.shape[1] > 512:
                h = h[:, -512:]
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)
            logits = model.lm_head(h)
            plen = 0 if prefix_hidden is None else prefix_hidden.shape[1]
            pred = logits[:, plen:plen + len(cont_ids) - 1, :]
            target = cont_ids[1:].unsqueeze(0).to(device)
            return F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                    target.reshape(-1)).item()

        nll_no.append(cont_nll(None, cont))
        full_emb = model.drop(model.tok_emb(ctx.unsqueeze(0).to(device)))
        nll_full.append(cont_nll(full_emb, cont))
        nll_mean.append(cont_nll(mean_eng.unsqueeze(0).unsqueeze(0), cont))

    no = sum(nll_no) / len(nll_no)
    full = sum(nll_full) / len(nll_full)
    gap = no - full
    mean_nll = sum(nll_mean) / len(nll_mean)
    return {
        "no": no, "full": full, "gap": gap, "mean_nll": mean_nll,
        "mean_recovery": (no - mean_nll) / gap if gap > 0 else 0,
    }


# ================================================================
# Main
# ================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase64")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    print("Phase 64: Softmax transformer validation")
    print(f"Device: {device}\n")

    model, val_ppl, train_step = load_softmax_model(device)
    print(f"Loaded softmax model: step={train_step}, val_ppl={val_ppl:.2f}")
    print(f"Params: {model.param_count():,}\n")

    splits, _ = load_wikitext("wikitext/wikitext-103-raw-v1", 512)
    val_ds = splits["validation"]
    tests = stratified_tests()

    all_results = {"model_ppl": val_ppl, "model_step": train_step}

    # ============================================================
    # M1: K/V alignment
    # ============================================================
    print("=" * 60)
    print("M1: K/V ALIGNMENT (Phase 32 protocol)")
    print("=" * 60)

    kv = measure_kv_alignment_softmax(model, val_ds, device, n_passages=50)
    print(f"  {'layer':>5}  {'K-cos':>7}  {'V-cos':>7}")
    for l in sorted(kv.keys()):
        print(f"  {l:>5}  {kv[l]['k_cos']:>7.4f}  {kv[l]['v_cos']:>7.4f}")
    all_results["m1_kv"] = {str(l): v for l, v in kv.items()}

    # ============================================================
    # M2: Phase transition (Phase 60 protocol)
    # ============================================================
    print(f"\n{'='*60}")
    print("M2: PHASE TRANSITION (Phase 60 protocol)")
    print("=" * 60)

    apply_lora_softmax(model, rank=RANK, alpha=RANK * 2)
    step_counts = [0, 15, 38, 75, 113, 150]
    m2_results = []

    for test in tests[:5]:
        for n_steps in step_counts:
            reset_lora_softmax(model)
            if n_steps > 0:
                prompts = [test["prompt"]] + train_paraphrase(test)
                pwa = [f"{p} {test['passkey']}" for p in prompts]
                train_adapter_softmax(model, test["passage"], pwa,
                                      tokenizer, device, n_steps=n_steps)

            gen = generate_greedy_softmax(model, test["prompt"], tokenizer,
                                          device, GEN_TOKENS)
            hit = check_passkey(gen, test["passkey"])

            # K-space alignment
            reset_lora_softmax(model)
            ids = tokenizer.encode(test["passage"], add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            # Quick K-cos at last layer
            kv_quick = measure_kv_alignment_softmax(
                model, val_ds, device, n_passages=5)
            k_cos = kv_quick.get(LAYER - 1, {}).get("k_cos", 0)

            m2_results.append({
                "passage_id": test["id"], "steps": n_steps,
                "hit": hit, "k_cos": k_cos,
            })

    # Aggregate
    for ns in step_counts:
        hits = [r for r in m2_results if r["steps"] == ns]
        n_hit = sum(1 for r in hits if r["hit"])
        print(f"  steps={ns:>3}: retrieval {n_hit}/{len(hits)}")
    all_results["m2_phase_transition"] = m2_results

    # ============================================================
    # M3: Information recovery
    # ============================================================
    print(f"\n{'='*60}")
    print("M3: INFORMATION RECOVERY (Phase 33 protocol)")
    print("=" * 60)

    reset_lora_softmax(model)
    info = measure_info_recovery_softmax(model, val_ds, device, n_passages=50)
    print(f"  no_context:  {info['no']:.4f}")
    print(f"  full_context: {info['full']:.4f}")
    print(f"  mean_engram:  {info['mean_nll']:.4f}")
    print(f"  gap:          {info['gap']:.4f}")
    print(f"  mean recovery: {info['mean_recovery']:.1%}")
    all_results["m3_info_recovery"] = info

    # ============================================================
    # M4: Cross-model transfer
    # ============================================================
    print(f"\n{'='*60}")
    print("M4: CROSS-MODEL TRANSFER (Phase 59 protocol)")
    print("=" * 60)

    # Compute engrams from Model A
    engrams_a = {}
    for test in tests[:5]:
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        H = hidden_at_layer_softmax(model, ids_t, LAYER)
        engrams_a[test["id"]] = H.mean(dim=1).squeeze(0).detach()

    # A's engrams in A
    kv_a_in_a = measure_kv_alignment_softmax(model, val_ds, device, n_passages=10)
    a_in_a = kv_a_in_a.get(LAYER - 1, {}).get("k_cos", 0)

    # Fine-tune to create Model B
    print("  Fine-tuning Model B (5K steps, seed=999)...")
    model_b, _, _ = load_softmax_model(device)
    for p in model_b.parameters():
        p.requires_grad = True
    torch.manual_seed(999)
    random.seed(999)
    loaders = build_dataloaders(splits, batch_size=2)
    train_iter = iter(loaders["train"])
    optimizer = torch.optim.AdamW(model_b.parameters(), lr=1e-4, weight_decay=0.01)
    model_b.train()
    for s in range(5000):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model_b(x)
        V = logits.shape[-1]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                y[:, :-1].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        optimizer.step()
        if (s + 1) % 1000 == 0:
            print(f"    step {s+1}")
    model_b.eval()
    for p in model_b.parameters():
        p.requires_grad = False

    # A's engrams in B
    kv_a_in_b = measure_kv_alignment_softmax(model_b, val_ds, device, n_passages=10)
    a_in_b = kv_a_in_b.get(LAYER - 1, {}).get("k_cos", 0)

    # B's own engrams in B
    kv_b_in_b = measure_kv_alignment_softmax(model_b, val_ds, device, n_passages=10)
    b_in_b = kv_b_in_b.get(LAYER - 1, {}).get("k_cos", 0)

    print(f"  A eng→A K-cos L{LAYER-1}: {a_in_a:.4f}")
    print(f"  A eng→B K-cos L{LAYER-1}: {a_in_b:.4f}")
    print(f"  B eng→B K-cos L{LAYER-1}: {b_in_b:.4f}")
    degradation = (a_in_a - a_in_b) / a_in_a * 100 if a_in_a > 0 else 0
    print(f"  degradation: {degradation:.1f}%")
    all_results["m4_transfer"] = {
        "a_in_a": a_in_a, "a_in_b": a_in_b, "b_in_b": b_in_b,
        "degradation_pct": degradation,
    }
    del model_b
    torch.cuda.empty_cache()

    # ============================================================
    # M5: Per-passage adapter retrieval
    # ============================================================
    print(f"\n{'='*60}")
    print("M5: PER-PASSAGE ADAPTER RETRIEVAL")
    print("=" * 60)

    library = []
    for i, test in enumerate(tests[:20]):
        reset_lora_softmax(model)
        prompts = [test["prompt"]] + train_paraphrase(test)
        pwa = [f"{p} {test['passkey']}" for p in prompts]
        train_adapter_softmax(model, test["passage"], pwa, tokenizer, device)
        sd = get_lora_state(model)

        # L0 key
        reset_lora_softmax(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        h0 = model.drop(model.tok_emb(ids_t))
        l0_key = h0.mean(dim=1).squeeze(0).detach().cpu()

        library.append({"sd": sd, "test": dict(test), "l0_key": l0_key})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/20] absorbed")

    # Same-prompt retrieval
    n_same = 0
    for i, entry in enumerate(library):
        load_lora_state(model, entry["sd"])
        gen = generate_greedy_softmax(model, entry["test"]["prompt"],
                                       tokenizer, device, GEN_TOKENS)
        if check_passkey(gen, entry["test"]["passkey"]):
            n_same += 1
    print(f"  same-prompt retrieval: {n_same}/20")

    # L0 routing on held-out paraphrases
    n_routed = 0
    n_retrieved = 0
    for i, entry in enumerate(library):
        for para in held_out_paraphrase(entry["test"]):
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            reset_lora_softmax(model)
            h0 = model.drop(model.tok_emb(ids_t))
            q = h0.mean(dim=1).squeeze(0).detach().cpu()

            best_a, best_s = -1, -2.0
            for ai, lib_entry in enumerate(library):
                s = cosine(q, lib_entry["l0_key"])
                if s > best_s:
                    best_s = s
                    best_a = ai
            if best_a == i:
                n_routed += 1
            load_lora_state(model, library[best_a]["sd"])
            gen = generate_greedy_softmax(model, para, tokenizer, device,
                                           GEN_TOKENS)
            if check_passkey(gen, entry["test"]["passkey"]):
                n_retrieved += 1

    total_ho = sum(len(held_out_paraphrase(e["test"])) for e in library)
    print(f"  held-out routing:   {n_routed}/{total_ho}")
    print(f"  held-out retrieval: {n_retrieved}/{total_ho}")
    all_results["m5_retrieval"] = {
        "same_prompt": n_same, "held_out_routed": n_routed,
        "held_out_retrieved": n_retrieved, "total_held_out": total_ho,
    }

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 64 SUMMARY: Softmax validation vs HRS baselines")
    print("=" * 72)
    print(f"\n  {'metric':30s}  {'HRS V22':>10}  {'Softmax':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")

    hrs_kv5 = {"k": "0.81-0.89", "v": "0.65-0.70"}
    soft_kv = kv.get(LAYER - 1, {})
    print(f"  {'K-cos L5':30s}  {'0.81-0.89':>10}  "
          f"{soft_kv.get('k_cos', 0):>10.4f}")
    print(f"  {'V-cos L5':30s}  {'0.65-0.70':>10}  "
          f"{soft_kv.get('v_cos', 0):>10.4f}")
    print(f"  {'K > V asymmetry':30s}  {'yes':>10}  "
          f"{'yes' if soft_kv.get('k_cos', 0) > soft_kv.get('v_cos', 0) else 'NO':>10}")
    print(f"  {'mean recovery':30s}  {'18.0%':>10}  "
          f"{info['mean_recovery']:>9.1%}")
    print(f"  {'cross-model degradation':30s}  {'2.5%':>10}  "
          f"{degradation:>9.1f}%")
    print(f"  {'same-prompt retrieval':30s}  {'20/20':>10}  "
          f"{n_same:>7}/20")
    print(f"  {'held-out routing':30s}  {'60/60':>10}  "
          f"{n_routed:>5}/{total_ho}")

    with open(results_dir / "softmax_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
