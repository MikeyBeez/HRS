"""RAFT v2: grounding-aux-loss variant.

Deviations from task 3's raft_train.py:
  1. Start from results/v18_raft/checkpoint_2000.pt (end of phase 1).
  2. No phase-2 backbone unfreeze: backbone stays frozen for all 5000 steps.
  3. Gate LR = 100x base (1e-3 when base = 1e-5), as a separate param group.
  4. Grounding auxiliary loss at weight 0.1, applied only on retrieve-active
     batch items, contrasts mean log-prob over source-doc (D+) content
     tokens vs random-distractor (D-) content tokens at all positions.
  5. Eval uses the expanded 25-needle benchmark (task 4) scored against
     cleaned_answer_tokens (task 5).

Loss composition per step:
    total = LM_CE + 0.1 * categorization_loss + 0.1 * grounding_loss
(Locality loss from original V18 training is NOT restored — consistent
with task 3.)

Outputs to results/v18_raft_grounded/.
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer
from engram_store import EngramStore
from data import load_wikitext, build_dataloaders

# Reuse task 3 helpers
from experiments.hrs_loop.raft_train import (
    load_v18_model as _load_v18_base,  # not used directly; we do our own load
    swap_buffer,
    restore_buffer,
    compute_query_engrams,
    get_gate_values,
    eval_mauve_500,
)
# Task 4/5 eval benchmark (drop-in-compatible)
from experiments.hrs_loop.niah_benchmark.eval_niah_v2 import (
    load_needles,
    build_distractor_pool,
    run_benchmark_config,
)

N_BUFFER_SLOTS = 32
EXTRACT_LAYER = 4
CROSS_ATTN_LAYERS = (1, 3, 5)
INV_SOFTPLUS_1 = math.log(math.e - 1.0)


# ------------------------------------------------------------
# Checkpoint loader (strict=False — works for V18 or RAFT-tuned)
# ------------------------------------------------------------
def load_ckpt(ckpt_path: Path, device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    patched = 0
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_scalar") and any(name == m for m in missing):
            with torch.no_grad():
                p.fill_(INV_SOFTPLUS_1)
            patched += 1
    print(f"Loaded {ckpt_path.name} (step {ckpt.get('step', '?')}). "
          f"missing={len(missing)} unexpected={len(unexpected)} gate_scalar_patched={patched}")
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    return model, cfg


# ------------------------------------------------------------
# Buffer construction — per task 3
# ------------------------------------------------------------
def build_batch_buffer(
    model, x, store_keys_gpu, baseline_buffer, extract_layer, device,
    p_retrieve=0.5, p_baseline=0.25, p_wrong=0.25,
    query_len=64, rng=None,
):
    """Return (buffer (B,32,D), conditions list, top_idxs dict for retrieve)."""
    B = x.shape[0]
    rng = rng or random.Random()
    conditions = []
    for _ in range(B):
        r = rng.random()
        if r < p_retrieve:
            conditions.append("retrieve")
        elif r < p_retrieve + p_baseline:
            conditions.append("baseline")
        else:
            conditions.append("wrong")

    top_idxs: dict[int, int] = {}
    wrong_idxs: dict[int, int] = {}
    if any(c == "retrieve" for c in conditions):
        q_engs = compute_query_engrams(model, x[:, :query_len], extract_layer)
        q_norm = F.normalize(q_engs.float(), dim=1)
        sims = q_norm @ store_keys_gpu.T
        best = sims.argmax(dim=1)
        for i, c in enumerate(conditions):
            if c == "retrieve":
                top_idxs[i] = int(best[i].item())

    n_store = store_keys_gpu.shape[0]
    slots = []
    for i, c in enumerate(conditions):
        if c == "baseline":
            slot = baseline_buffer.squeeze(0)
        elif c == "wrong":
            idx = rng.randint(0, n_store - 1)
            wrong_idxs[i] = idx
            v = store_keys_gpu[idx]
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
        else:  # retrieve
            v = store_keys_gpu[top_idxs[i]]
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
        slots.append(slot)
    buf = torch.stack(slots, dim=0).to(device).contiguous()
    return buf, conditions, top_idxs, wrong_idxs


# ------------------------------------------------------------
# Grounding auxiliary loss
# ------------------------------------------------------------
def compute_grounding_loss(
    logits: torch.Tensor,          # (B, T, V)
    x_ids: torch.Tensor,            # (B, T)
    conditions: list[str],
    top_idxs: dict[int, int],
    store_entries,                  # list[EngramEntry]
    tokenizer,
    rng: random.Random,
    content_cap: int = 32,
) -> tuple[torch.Tensor, dict]:
    """Contrastive grounding on retrieve-active items only.

    For each retrieve-active batch item i with top-retrieved source D+_i:
      - sample a random distractor D-_i (different index)
      - extract content tokens from D+ and D-: tokenize, drop those in x[i]
      - take up to content_cap unique token IDs per set
      - compute log-probs at every position of x[i] for those token sets
      - loss_i = softplus(-(mean_logp(D+) - mean_logp(D-))) averaged over T

    Returns (loss, stats). If no retrieve-active items have non-empty content
    sets, loss is a zero scalar with requires_grad.
    """
    device = logits.device
    retrieve_idxs = [i for i, c in enumerate(conditions) if c == "retrieve"]
    stats = {"n_retrieve_items": len(retrieve_idxs), "n_scored": 0,
             "mean_diff": 0.0}
    if not retrieve_idxs:
        return logits.sum() * 0.0, stats  # zero w/ grad plumbing

    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)
    losses = []
    diffs = []
    n_store = len(store_entries)
    for i in retrieve_idxs:
        d_plus_idx = top_idxs[i]
        d_minus_idx = rng.randint(0, n_store - 1)
        if n_store > 1:
            while d_minus_idx == d_plus_idx:
                d_minus_idx = rng.randint(0, n_store - 1)

        plus_text = store_entries[d_plus_idx].text
        minus_text = store_entries[d_minus_idx].text

        # Tokenize and filter against current sequence tokens
        x_set = set(x_ids[i].tolist())
        plus_ids = [t for t in tokenizer.encode(plus_text, add_special_tokens=False) if t not in x_set]
        minus_ids = [t for t in tokenizer.encode(minus_text, add_special_tokens=False) if t not in x_set]
        plus_ids = list(dict.fromkeys(plus_ids))[:content_cap]  # dedupe, cap
        minus_ids = list(dict.fromkeys(minus_ids))[:content_cap]
        if not plus_ids or not minus_ids:
            continue

        plus_t = torch.tensor(plus_ids, device=device, dtype=torch.long)
        minus_t = torch.tensor(minus_ids, device=device, dtype=torch.long)
        # log_probs[i, :, plus_t]  shape (T, |plus|) -> mean over tokens -> (T,)
        logp_plus = log_probs[i].index_select(dim=-1, index=plus_t).mean(dim=-1)
        logp_minus = log_probs[i].index_select(dim=-1, index=minus_t).mean(dim=-1)
        diff = logp_plus - logp_minus  # (T,)
        item_loss = F.softplus(-diff).mean()
        losses.append(item_loss)
        diffs.append(float(diff.mean().item()))

    if not losses:
        return logits.sum() * 0.0, stats
    loss = torch.stack(losses).mean()
    stats["n_scored"] = len(losses)
    stats["mean_diff"] = sum(diffs) / len(diffs)
    return loss, stats


# ------------------------------------------------------------
# Eval on expanded benchmark with cleaned tokens
# ------------------------------------------------------------
def eval_niah_expanded(model, tokenizer, device, distractors, needles_all,
                       token_set="cleaned_answer_tokens", seed=42):
    """Single-config eval (Config B: 25 needles × 40 distractors, cleaned)."""
    was_training = model.training
    model.eval()
    res = run_benchmark_config(
        model, tokenizer, device, needles_all, distractors,
        max_new_tokens=100, temperature=0.9, top_k=50, seed=seed,
        label="train_eval", token_set=token_set,
    )
    if was_training:
        model.train()
    return res


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
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=str, default="results/v18_raft/checkpoint_2000.pt")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr-ca", type=float, default=1e-5, help="base LR for cross-attn non-gate params")
    ap.add_argument("--lr-final", type=float, default=1e-6)
    ap.add_argument("--gate-lr-mult", type=float, default=100.0, help="gate LR = gate_lr_mult * lr-ca")
    ap.add_argument("--mauve-n", type=int, default=200)
    ap.add_argument("--p-retrieve", type=float, default=0.5)
    ap.add_argument("--p-baseline", type=float, default=0.25)
    ap.add_argument("--p-wrong", type=float, default=0.25)
    ap.add_argument("--grounding-weight", type=float, default=0.1)
    ap.add_argument("--cat-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="results/v18_raft_grounded")
    ap.add_argument("--max-wall-hours", type=float, default=24.0)
    args = ap.parse_args()
    assert abs(args.p_retrieve + args.p_baseline + args.p_wrong - 1.0) < 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    # ----- Model -----
    base_ckpt = Path(args.base_ckpt)
    if not base_ckpt.is_absolute():
        base_ckpt = REPO / base_ckpt
    model, cfg = load_ckpt(base_ckpt, device)
    baseline_buffer = model.engram_buffer.detach().clone().to(device)
    print(f"Baseline buffer slot-norm: {baseline_buffer[0, 0].norm().item():.3f}")

    # ----- Store -----
    store = EngramStore.load(str(REPO / "engram_store_data"))
    store_keys_gpu = store.keys.to(device)
    print(f"Store: {len(store)} entries")

    # ----- Data (cached) -----
    cache_dir = REPO / "experiments/hrs_loop/cache"
    cache_path = cache_dir / f"wt103_seqlen{args.seq_len}_ncat{cfg.cross_attn_engram.num_categories}.pt"
    if cache_path.exists():
        print(f"Loading cached WT-103 from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        splits = cached["splits"]
    else:
        print("Tokenizing WT-103 (no cache found; this is slow)...")
        splits, _ = load_wikitext(
            "wikitext-103", seq_len=args.seq_len,
            with_categories=True, n_categories=cfg.cross_attn_engram.num_categories,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"splits": splits}, cache_path)
    loaders = build_dataloaders(splits, batch_size=args.batch_size, num_workers=2)
    train_iter = iter(loaders["train"])
    test_tokens = splits["test"].tokens

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ----- Expanded benchmark distractors + needles (fixed for all evals) -----
    print("Building expanded-benchmark distractor pool (one-time)...")
    eval_distractors = build_distractor_pool(n_wt103_extra=20, seed=args.seed)
    eval_needles = load_needles()
    print(f"Eval: {len(eval_needles)} needles + {len(eval_distractors)} distractors")

    # ----- Param groups -----
    gate_params, ca_other_params, backbone_params = [], [], []
    ca_other_names = []
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_logit") or name.endswith(".cross_attn.gate_scalar"):
            gate_params.append(p)
        elif "cross_attn" in name or "categorization_head" in name:
            ca_other_params.append(p)
            ca_other_names.append(name)
        else:
            backbone_params.append(p)
    print(f"Gate params:         {sum(p.numel() for p in gate_params):,} ({len(gate_params)} tensors)")
    print(f"Cross-attn+cat rest: {sum(p.numel() for p in ca_other_params):,} ({len(ca_other_params)} tensors)")
    print(f"Backbone:            {sum(p.numel() for p in backbone_params):,} (FROZEN for all steps)")

    # Freeze backbone permanently (no phase 2)
    for p in backbone_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": gate_params, "lr": args.lr_ca * args.gate_lr_mult, "weight_decay": 0.0},
            {"params": ca_other_params, "lr": args.lr_ca, "weight_decay": 0.0},
            {"params": backbone_params, "lr": 0.0, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            **vars(args),
            "device": str(device),
            "base_ckpt_resolved": str(base_ckpt),
            "baseline_buffer_slot_norm": float(baseline_buffer[0, 0].norm().item()),
            "store_size": len(store),
            "n_eval_needles": len(eval_needles),
            "n_eval_distractors": len(eval_distractors),
        }, f, indent=2)

    # ----- Pre-training eval on expanded benchmark -----
    print("\n=== Pre-training eval (cleaned-token Config B) ===")
    pre = eval_niah_expanded(model, tokenizer, device, eval_distractors, eval_needles)
    pre_gates = get_gate_values(model)
    print(f"  NIAH: cleaned_recall={pre['recall_pooled']:.3f} "
          f"({pre['total_hits']}/{pre['total_possible']}) "
          f"acc@1={pre['acc_at_1']:.2f} mean_rank={pre['mean_rank']:.2f}")
    print(f"  gates: {pre_gates}")

    history = [{
        "step": 0, "phase": "pre",
        "niah_cleaned_recall": pre["recall_pooled"],
        "niah_hits": pre["total_hits"],
        "niah_total": pre["total_possible"],
        "niah_acc_at_1": pre["acc_at_1"],
        "niah_mean_rank": pre["mean_rank"],
        "mauve_500_n200": None,
        "gates": pre_gates,
        "lr_ca": 0.0,
        "lr_gate": 0.0,
        "ce_loss": None, "cat_loss": None, "grounding_loss": None,
    }]

    # ----- Training loop -----
    print(f"\n=== Starting grounded RAFT: {args.max_steps} steps, backbone FROZEN throughout, "
          f"gate LR = {args.gate_lr_mult}x base, grounding weight = {args.grounding_weight} ===\n")

    model.train()
    t0 = time.time()
    step = 0
    cond_counts = {"retrieve": 0, "baseline": 0, "wrong": 0}
    accum = {"ce": 0.0, "cat": 0.0, "ground": 0.0, "loss": 0.0, "diff": 0.0, "n": 0}

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

        # LR update
        lr_ca = get_lr(step, args.max_steps, args.warmup, args.lr_ca, args.lr_final)
        lr_gate = lr_ca * args.gate_lr_mult
        optimizer.param_groups[0]["lr"] = lr_gate
        optimizer.param_groups[1]["lr"] = lr_ca
        # backbone remains 0.0

        optimizer.zero_grad()

        # Build batch buffer
        buf, conditions, top_idxs, wrong_idxs = build_batch_buffer(
            model, x, store_keys_gpu, baseline_buffer, EXTRACT_LAYER, device,
            p_retrieve=args.p_retrieve, p_baseline=args.p_baseline,
            p_wrong=args.p_wrong, query_len=64, rng=rng,
        )
        for c in conditions:
            cond_counts[c] += 1

        orig_buf, orig_init = swap_buffer(model, buf)
        try:
            output = model(x, step=step)
            logits = output.logits  # (B, T, V)
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            total_loss = ce_loss
            cat_loss_val = 0.0
            if output.categorization_logits is not None and cat_ids is not None:
                cat_loss = F.cross_entropy(output.categorization_logits, cat_ids)
                total_loss = total_loss + args.cat_weight * cat_loss
                cat_loss_val = float(cat_loss.item())

            ground_loss, ground_stats = compute_grounding_loss(
                logits, x, conditions, top_idxs, store.entries, tokenizer, rng,
            )
            if ground_stats["n_scored"] > 0:
                total_loss = total_loss + args.grounding_weight * ground_loss
            ground_loss_val = float(ground_loss.item())
        finally:
            restore_buffer(model, orig_buf, orig_init)

        if not torch.isfinite(total_loss):
            print(f"!!! NaN/inf at step {step} — stopping")
            break

        total_loss.backward()

        if step == 100:
            print("=== Grad-flow snapshot at step 100 (backbone FROZEN) ===")
            for name, p in model.named_parameters():
                if p.grad is not None and ("cross_attn.gate" in name):
                    print(f"  {name}: grad_norm={p.grad.norm().item():.3e}")
            bb_gn2 = sum(p.grad.norm().item() ** 2 for p in backbone_params if p.grad is not None)
            ca_gn2 = sum(p.grad.norm().item() ** 2 for p in ca_other_params if p.grad is not None)
            gt_gn2 = sum(p.grad.norm().item() ** 2 for p in gate_params if p.grad is not None)
            print(f"  -- gate grad L2:       {math.sqrt(gt_gn2):.3e}")
            print(f"  -- cross-attn grad L2: {math.sqrt(ca_gn2):.3e}")
            print(f"  -- backbone grad L2:   {math.sqrt(bb_gn2):.3e} (should be 0)")

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        step += 1
        accum["ce"] += float(ce_loss.item())
        accum["cat"] += cat_loss_val
        accum["ground"] += ground_loss_val
        accum["loss"] += float(total_loss.item())
        accum["diff"] += ground_stats["mean_diff"]
        accum["n"] += 1

        if step % args.log_interval == 0:
            gates = get_gate_values(model)
            gate_str = " ".join(f"L{k}={v:.3f}" for k, v in gates.items())
            el = time.time() - t0
            rate = step / el if el > 0 else 0
            avg_ce = accum["ce"] / accum["n"]
            avg_cat = accum["cat"] / accum["n"]
            avg_gr = accum["ground"] / accum["n"]
            avg_diff = accum["diff"] / accum["n"]
            mix = " ".join(f"{k[:3]}={cond_counts[k]}" for k in ("retrieve", "baseline", "wrong"))
            print(f"[step {step:4d}/{args.max_steps}] "
                  f"ce={avg_ce:.3f} cat={avg_cat:.3f} "
                  f"ground={avg_gr:.4f} diff={avg_diff:+.3f} "
                  f"lr_ca={lr_ca:.2e} lr_g={lr_gate:.2e} "
                  f"gates: {gate_str} cond({mix}) {rate:.2f}it/s elapsed={el:.0f}s")
            accum = {"ce": 0.0, "cat": 0.0, "ground": 0.0, "loss": 0.0, "diff": 0.0, "n": 0}

        if step % args.eval_interval == 0 or step == args.max_steps:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
            # Ensure buffer is at baseline for eval
            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            niah = eval_niah_expanded(model, tokenizer, device, eval_distractors, eval_needles)
            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            mauve_score = eval_mauve_500(
                model, tokenizer, test_tokens, device,
                n_samples=args.mauve_n,
            )
            gates = get_gate_values(model)

            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            entry = {
                "step": step,
                "phase": "grounded",
                "niah_cleaned_recall": niah["recall_pooled"],
                "niah_hits": niah["total_hits"],
                "niah_total": niah["total_possible"],
                "niah_acc_at_1": niah["acc_at_1"],
                "niah_mean_rank": niah["mean_rank"],
                "mauve_500_n200": mauve_score,
                "gates": gates,
                "lr_ca": lr_ca,
                "lr_gate": lr_gate,
                "ce_loss": float(ce_loss.item()),
                "cat_loss": cat_loss_val,
                "grounding_loss": ground_loss_val,
                "elapsed_s": time.time() - t0,
            }
            history.append(entry)
            print(f"  NIAH cleaned: recall={niah['recall_pooled']:.3f} "
                  f"({niah['total_hits']}/{niah['total_possible']}) "
                  f"acc@1={niah['acc_at_1']:.2f} mean_rank={niah['mean_rank']:.2f}")
            print(f"  MAUVE-500 (n={args.mauve_n}): {mauve_score:.4f}")
            print(f"  Gates: {gates}")
            print(f"  elapsed={entry['elapsed_s']:.0f}s")

            if mauve_score < 0.90:
                print(f"  *** WARNING: MAUVE below 0.90 floor ({mauve_score:.4f}) ***")

            ckpt_path = out_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "lr_ca": lr_ca,
                "lr_gate": lr_gate,
                "niah_cleaned_recall": niah["recall_pooled"],
                "mauve_500": mauve_score,
                "gates": gates,
                "run_config": vars(args),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            with open(out_dir / "training_log.json", "w") as f:
                json.dump(history, f, indent=2)

            model.train()
            if (time.time() - t0) / 3600 > args.max_wall_hours:
                print(f"*** Wall-clock budget exceeded, stopping ***")
                break

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Condition mix: {cond_counts}")
    print(f"Total steps: {step}. Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
