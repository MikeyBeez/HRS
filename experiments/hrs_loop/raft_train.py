"""RAFT fine-tuning for V18 cross-attention grounding.

Trains V18's cross-attention pathway on a 3-way buffer mix:
  50% retrieval-active  — nearest store entry for the current batch item
  25% baseline          — V18's frozen EMA corpus buffer (matches pretrain)
  25% wrong engram      — random store entry (teaches gate to distrust)

Phase 1 (0-2000): backbone frozen, cross-attn + cat head only.
Phase 2 (2000-5000): backbone unfrozen at 10x lower LR.

Eval every 1000 steps: NIAH continuation-only recall, MAUVE-500 (n=200),
per-layer gate values. Checkpoint to results/v18_raft/.

This file intentionally bypasses train.py / CombinedHRSLoss / EngramStore's
built-in retrieval pathway. The buffer swap is done by overwriting
model._buffers['engram_buffer'] directly per step.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]  # /mnt/data/Code/HRS
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer
from engram_store import EngramStore
from data import load_wikitext, build_dataloaders
from niah_egr import NEEDLES, DISTRACTORS

# Local: reuse task 2 NIAH harness
from experiments.hrs_loop.engram_content_ablation import (
    VARIANTS, run_variant_on_needle,
)

N_BUFFER_SLOTS = 32
EXTRACT_LAYER = 4  # V18: n_layers=6, extract_layer=-2 -> index 4
CROSS_ATTN_LAYERS = (1, 3, 5)  # odd-indexed blocks have cross-attn
INV_SOFTPLUS_1 = math.log(math.e - 1.0)  # softplus(x) = 1 at this x


# ------------------------------------------------------------
# Model loading (with V21 gate_scalar compat)
# ------------------------------------------------------------

def load_v18_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load(
        str(REPO / "results/v18_cross_attn/best.pt"),
        map_location=device, weights_only=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    patched = 0
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_scalar") and any(name == m for m in missing):
            with torch.no_grad():
                p.fill_(INV_SOFTPLUS_1)
            patched += 1
    print(f"Loaded V18 (step {ckpt.get('step', '?')}, val_ppl "
          f"{ckpt.get('val_ppl', float('nan')):.2f}). "
          f"missing={len(missing)} unexpected={len(unexpected)} "
          f"patched gate_scalar={patched}")
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    return model, cfg


# ------------------------------------------------------------
# Buffer swap helpers
# ------------------------------------------------------------

def swap_buffer(model, new_buffer: torch.Tensor):
    """Directly replace the tensor at model._buffers['engram_buffer'].

    Returns (orig_tensor, orig_initialized_flag) so caller can restore.
    Works even if new_buffer has a different batch dim than the registered
    shape (1, 32, D) — PyTorch's buffer dict just stores tensor refs.
    """
    orig = model._buffers["engram_buffer"]
    orig_flag = model._engram_buffer_initialized
    model._buffers["engram_buffer"] = new_buffer
    model._engram_buffer_initialized = True
    return orig, orig_flag


def restore_buffer(model, orig: torch.Tensor, orig_flag: bool):
    model._buffers["engram_buffer"] = orig
    model._engram_buffer_initialized = orig_flag


@torch.no_grad()
def compute_query_engrams(model, input_ids: torch.Tensor, extract_layer: int) -> torch.Tensor:
    """Run a forward pass on (B, T) ids, mean-pool hidden states at extract_layer.

    Does NOT train parameters; gradients disabled. Returns (B, D).
    """
    capture = {}

    def hook(module, _inputs, outputs):
        capture["h"] = outputs[0].detach()

    handle = model.blocks[extract_layer].register_forward_hook(hook)
    try:
        _ = model(input_ids, step=0)
    finally:
        handle.remove()
    return capture["h"].mean(dim=1)  # (B, D)


def build_batch_buffer(
    model, x, store_keys_gpu, baseline_buffer, extract_layer, device,
    p_retrieve=0.5, p_baseline=0.25, p_wrong=0.25,
    query_len=64, rng=None,
):
    """Produce a (B, 32, D) cross-attention buffer + record condition per item.

    Uses model in current state (buffer must be restored to baseline when
    called, so the query forward pass is comparable to V18 inference).
    """
    B = x.shape[0]
    rng = rng or random.Random()

    # Per-item condition sample
    conditions = []
    for _ in range(B):
        r = rng.random()
        if r < p_retrieve:
            conditions.append("retrieve")
        elif r < p_retrieve + p_baseline:
            conditions.append("baseline")
        else:
            conditions.append("wrong")

    # Compute query engrams (only if any retrieval needed, but cheap to batch all)
    needs_query = any(c == "retrieve" for c in conditions)
    top_idxs = {}
    if needs_query:
        q_engs = compute_query_engrams(model, x[:, :query_len], extract_layer)  # (B, D)
        q_norm = F.normalize(q_engs.float(), dim=1)
        sims = q_norm @ store_keys_gpu.T  # (B, N_store)
        best = sims.argmax(dim=1)  # (B,)
        for i, c in enumerate(conditions):
            if c == "retrieve":
                top_idxs[i] = int(best[i].item())

    n_store = store_keys_gpu.shape[0]
    slots = []
    for i, c in enumerate(conditions):
        if c == "baseline":
            slot = baseline_buffer.squeeze(0)  # (32, D), norm ~5.3
        elif c == "wrong":
            idx = rng.randint(0, n_store - 1)
            v = store_keys_gpu[idx]  # (D,), unit-norm
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
        else:  # retrieve
            v = store_keys_gpu[top_idxs[i]]  # (D,), unit-norm
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
        slots.append(slot)
    buf = torch.stack(slots, dim=0).to(device).contiguous()  # (B, 32, D)
    return buf, conditions


# ------------------------------------------------------------
# LR schedule
# ------------------------------------------------------------

def get_lr(step, max_steps=5000, warmup=500, base=1e-5, final=1e-6):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return final + (base - final) * cos


# ------------------------------------------------------------
# Gate value readout
# ------------------------------------------------------------

def get_gate_values(model, layers=CROSS_ATTN_LAYERS):
    gates = {}
    for i in layers:
        block = model.blocks[i]
        if hasattr(block, "cross_attn") and getattr(block, "use_cross_attn_engram", False):
            ca = block.cross_attn
            g = torch.sigmoid(ca.gate_logit) * F.softplus(ca.gate_scalar)
            gates[i] = float(g.item())
    return gates


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def eval_niah_recall(model, tokenizer, device, max_new_tokens=100,
                     temperature=0.9, top_k_samp=50, seed=42):
    """Task 2's continuation-only NIAH recall (v1_baseline variant)."""
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    build_cfg = {
        "extract_layer": EXTRACT_LAYER,
        "v2_layers": CROSS_ATTN_LAYERS,
        "v3_n": 32, "v4_n": 16,
    }
    per = []
    for needle in NEEDLES:
        r = run_variant_on_needle(
            model, tokenizer, device,
            "v1_baseline", VARIANTS["v1_baseline"], build_cfg,
            needle, DISTRACTORS[:20],
            temperature, top_k_samp, max_new_tokens,
        )
        per.append(r)
    ranks = [r["rank"] for r in per]
    hits = sum(r["n_hits"] for r in per)
    total = sum(r["n_answer_tokens"] for r in per)
    if was_training:
        model.train()
    return {
        "mean_rank": sum(ranks) / len(ranks),
        "acc_at_1": sum(1 for r in ranks if r == 1) / len(ranks),
        "recall": hits / total,
        "hits": hits, "total": total,
        "per_needle_summary": [
            {"cat": r["needle_category"], "rank": r["rank"], "hits": r["n_hits"]}
            for r in per
        ],
    }


def eval_mauve_500(model, tokenizer, test_tokens, device,
                   n_samples=200, continuation_len=256, prompt_len=500,
                   temperature=0.9, top_k=50, batch_size=8):
    """MAUVE on WikiText-103 test with 500-token prompts. No engram injection
    (uses whatever buffer is currently loaded — should be baseline)."""
    import mauve
    was_training = model.training
    model.eval()

    total = prompt_len + continuation_len
    stride = max(1, (len(test_tokens) - total) // n_samples)
    prompts, refs = [], []
    for i in range(n_samples):
        start = i * stride
        if start + total > len(test_tokens):
            start = len(test_tokens) - total
        prompts.append(test_tokens[start:start + prompt_len])
        refs.append(test_tokens[start:start + total])
    prompts = torch.stack(prompts)
    ref_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in refs]

    gen_texts = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            ids = prompts[start:end].to(device)
            for _ in range(continuation_len):
                idx = ids[:, -512:]
                out = model(idx, step=0)
                logits = out.logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, nxt], dim=1)
            for j in range(ids.shape[0]):
                gen_texts.append(tokenizer.decode(ids[j], skip_special_tokens=True))

    out = mauve.compute_mauve(
        p_text=ref_texts, q_text=gen_texts,
        device_id=0 if device.type == "cuda" else -1,
        verbose=False,
    )
    if was_training:
        model.train()
    return float(out.mauve)


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--freeze-until", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr-ca", type=float, default=1e-5, help="cross-attn base LR")
    ap.add_argument("--lr-final", type=float, default=1e-6, help="cross-attn LR at end of cosine")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--mauve-n", type=int, default=200)
    ap.add_argument("--p-retrieve", type=float, default=0.5)
    ap.add_argument("--p-baseline", type=float, default=0.25)
    ap.add_argument("--p-wrong", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="results/v18_raft")
    ap.add_argument("--max-wall-hours", type=float, default=24.0)
    args = ap.parse_args()
    assert abs(args.p_retrieve + args.p_baseline + args.p_wrong - 1.0) < 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    # ----- Model -----
    model, cfg = load_v18_model(device)
    baseline_buffer = model.engram_buffer.detach().clone().to(device)  # (1,32,D) norm~5.3
    print(f"Baseline buffer norm (per slot, identical): "
          f"{baseline_buffer[0, 0].norm().item():.3f}")

    # ----- Store -----
    store = EngramStore.load(str(REPO / "engram_store_data"))
    store_keys_gpu = store.keys.to(device)  # (N, D), already unit-normalized
    print(f"Store: {len(store)} entries, keys shape {tuple(store_keys_gpu.shape)}")
    # sanity: store was built from validation split (not train) — confirmed by
    # populate_store.py reading raw['validation']
    print(f"First 3 entry sources: {[store.entries[i].source for i in range(3)]}")

    # ----- Data (cached) -----
    cache_dir = REPO / "experiments/hrs_loop/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"wt103_seqlen{args.seq_len}_ncat{cfg.cross_attn_engram.num_categories}.pt"
    if cache_path.exists():
        print(f"Loading cached tokenized WT-103 from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        splits = cached["splits"]
    else:
        print("Loading WikiText-103 train/val/test with categories (will cache after)...")
        splits, _ = load_wikitext(
            "wikitext-103", seq_len=args.seq_len,
            with_categories=True, n_categories=cfg.cross_attn_engram.num_categories,
        )
        torch.save({"splits": splits}, cache_path)
        print(f"Cached to {cache_path}")
    loaders = build_dataloaders(splits, batch_size=args.batch_size, num_workers=2)
    train_iter = iter(loaders["train"])
    test_tokens = splits["test"].tokens

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ----- Param groups -----
    crossattn_params, backbone_params = [], []
    ca_names, bb_names = [], []
    for name, p in model.named_parameters():
        if "cross_attn" in name or "categorization_head" in name:
            crossattn_params.append(p)
            ca_names.append(name)
        else:
            backbone_params.append(p)
            bb_names.append(name)
    print(f"Cross-attn+cat params: {sum(p.numel() for p in crossattn_params):,} "
          f"({len(crossattn_params)} tensors)")
    print(f"Backbone params:       {sum(p.numel() for p in backbone_params):,} "
          f"({len(backbone_params)} tensors)")

    # Phase 1: freeze backbone
    for p in backbone_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": crossattn_params, "lr": args.lr_ca, "weight_decay": 0.0},
            {"params": backbone_params, "lr": 0.0, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        **vars(args),
        "device": str(device),
        "baseline_buffer_slot_norm": float(baseline_buffer[0, 0].norm().item()),
        "store_size": len(store),
        "cross_attn_params": sum(p.numel() for p in crossattn_params),
        "backbone_params": sum(p.numel() for p in backbone_params),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # ----- Baseline eval BEFORE any training -----
    print("\n=== Pre-training eval (should match task 2 results) ===")
    niah0 = eval_niah_recall(model, tokenizer, device)
    print(f"  NIAH: recall={niah0['recall']:.3f} acc@1={niah0['acc_at_1']:.2f} "
          f"mean_rank={niah0['mean_rank']:.2f}")
    gates0 = get_gate_values(model)
    print(f"  gates: {gates0}")

    history = [{
        "step": 0, "phase": "pre",
        "niah_recall": niah0["recall"],
        "niah_acc_at_1": niah0["acc_at_1"],
        "niah_mean_rank": niah0["mean_rank"],
        "mauve_500_n200": None,  # skip to save time, will do at first checkpoint
        "gates": gates0,
        "lr_cross_attn": 0.0,
        "ce_loss": None, "cat_loss": None,
    }]

    # ----- Training loop -----
    print(f"\n=== Starting RAFT: {args.max_steps} steps, phase1 0-{args.freeze_until} "
          f"(backbone frozen), phase2 {args.freeze_until}-{args.max_steps} "
          f"(backbone LR = {args.backbone_lr_mult}x cross-attn LR) ===\n")

    model.train()
    t0 = time.time()
    step = 0
    cond_counts = {"retrieve": 0, "baseline": 0, "wrong": 0}
    accum = {"ce": 0.0, "cat": 0.0, "loss": 0.0, "n": 0}
    backbone_unfrozen = False

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        if len(batch) == 3:
            x, y, cat_ids = batch
            cat_ids = cat_ids.to(device)
        else:
            x, y = batch
            cat_ids = None
        x, y = x.to(device), y.to(device)

        # Phase 2 unfreeze
        if step >= args.freeze_until and not backbone_unfrozen:
            for p in backbone_params:
                p.requires_grad = True
            backbone_unfrozen = True
            print(f"*** Step {step}: unfreezing backbone ***")

        # LR update
        lr_ca = get_lr(step, args.max_steps, args.warmup, args.lr_ca, args.lr_final)
        optimizer.param_groups[0]["lr"] = lr_ca
        optimizer.param_groups[1]["lr"] = lr_ca * args.backbone_lr_mult if backbone_unfrozen else 0.0

        optimizer.zero_grad()

        # Build per-batch buffer. Model buffer must be at baseline for the
        # query forward pass (so query engram is computed the inference way).
        # Before calling build_batch_buffer, the buffer is the checkpoint EMA
        # (= baseline). build_batch_buffer uses it for the query pass then
        # returns a fresh tensor — we haven't touched _buffers yet.
        buf, conditions = build_batch_buffer(
            model, x, store_keys_gpu, baseline_buffer,
            EXTRACT_LAYER, device,
            p_retrieve=args.p_retrieve, p_baseline=args.p_baseline, p_wrong=args.p_wrong,
            query_len=64, rng=rng,
        )
        for c in conditions:
            cond_counts[c] += 1

        # Swap in the per-batch buffer for training forward
        orig_buf, orig_init = swap_buffer(model, buf)
        try:
            output = model(x, step=step)
            logits = output.logits  # (B, T, V)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            )
            loss = ce_loss
            cat_loss_val = 0.0
            if output.categorization_logits is not None and cat_ids is not None:
                cat_loss = F.cross_entropy(output.categorization_logits, cat_ids)
                loss = loss + 0.1 * cat_loss
                cat_loss_val = float(cat_loss.item())
        finally:
            restore_buffer(model, orig_buf, orig_init)

        if not torch.isfinite(loss):
            print(f"!!! NaN/inf at step {step}, loss={loss.item()} — stopping")
            break

        loss.backward()

        # Grad sanity at step 100
        if step == 100:
            print("=== Gradient-flow snapshot at step 100 ===")
            total_ca_gn = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None and ("cross_attn" in name or "categorization" in name):
                    n = p.grad.norm().item()
                    total_ca_gn += n ** 2
                    print(f"  {name}: grad_norm={n:.3e}")
            total_bb_gn = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None and "cross_attn" not in name and "categorization" not in name:
                    total_bb_gn += p.grad.norm().item() ** 2
            print(f"  -- total cross-attn grad norm: {math.sqrt(total_ca_gn):.3e}")
            print(f"  -- total backbone grad norm:   {math.sqrt(total_bb_gn):.3e} "
                  f"(should be 0 in phase 1)")

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()

        step += 1
        accum["ce"] += float(ce_loss.item())
        accum["cat"] += cat_loss_val
        accum["loss"] += float(loss.item())
        accum["n"] += 1

        if step % args.log_interval == 0:
            gates = get_gate_values(model)
            gate_str = " ".join(f"L{k}={v:.3f}" for k, v in gates.items())
            el = time.time() - t0
            rate = step / el if el > 0 else 0
            avg_ce = accum["ce"] / accum["n"]
            avg_cat = accum["cat"] / accum["n"]
            mix = " ".join(f"{k[:3]}={cond_counts[k]}" for k in ("retrieve", "baseline", "wrong"))
            print(f"[step {step:4d}/{args.max_steps}] "
                  f"ce={avg_ce:.3f} cat={avg_cat:.3f} "
                  f"lr={lr_ca:.2e} gates: {gate_str} "
                  f"cond({mix}) {rate:.2f}it/s elapsed={el:.0f}s")
            accum = {"ce": 0.0, "cat": 0.0, "loss": 0.0, "n": 0}

        if step % args.eval_interval == 0 or step == args.max_steps:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
            # Ensure buffer is back to baseline for eval (NIAH sets its own)
            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            niah = eval_niah_recall(model, tokenizer, device)
            # Restore baseline buffer before MAUVE (NIAH mutates it)
            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            mauve_score = eval_mauve_500(
                model, tokenizer, test_tokens, device,
                n_samples=args.mauve_n,
            )
            gates = get_gate_values(model)

            # Restore once more
            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            entry = {
                "step": step,
                "phase": "2" if backbone_unfrozen else "1",
                "niah_recall": niah["recall"],
                "niah_acc_at_1": niah["acc_at_1"],
                "niah_mean_rank": niah["mean_rank"],
                "mauve_500_n200": mauve_score,
                "gates": gates,
                "lr_cross_attn": lr_ca,
                "ce_loss": float(ce_loss.item()),
                "cat_loss": cat_loss_val,
                "elapsed_s": time.time() - t0,
            }
            history.append(entry)
            print(f"  NIAH: recall={niah['recall']:.3f} acc@1={niah['acc_at_1']:.2f} "
                  f"mean_rank={niah['mean_rank']:.2f}")
            print(f"  MAUVE-500 (n={args.mauve_n}): {mauve_score:.4f}")
            print(f"  Gates: {gates}")
            print(f"  elapsed={entry['elapsed_s']:.0f}s")

            if mauve_score < 0.90:
                print(f"  *** WARNING: MAUVE below 0.90 floor ({mauve_score:.4f}) ***")

            # Save checkpoint
            ckpt_path = out_dir / f"checkpoint_{step}.pt"
            # Ensure buffer at save time is the (1,32,D) baseline
            save_state = model.state_dict()
            torch.save({
                "step": step,
                "model_state_dict": save_state,
                "lr_cross_attn": lr_ca,
                "niah_recall": niah["recall"],
                "mauve_500": mauve_score,
                "gates": gates,
                "run_config": run_config,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            with open(out_dir / "training_log.json", "w") as f:
                json.dump(history, f, indent=2)

            model.train()

            # Wall-clock budget
            if (time.time() - t0) / 3600 > args.max_wall_hours:
                print(f"*** Wall-clock budget exceeded, stopping ***")
                break

    # Final log dump
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Condition mix: {cond_counts}")
    print(f"Total steps: {step}. Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
