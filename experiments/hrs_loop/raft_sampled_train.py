"""RAFT v3: sampled-token REINFORCE grounding (task 8).

Deviations from raft_grounded_train.py (tasks 6/7):
  1. Grounding loss replaced with REINFORCE on on-policy rollouts.
  2. For each retrieve-active batch item we sample K=8 tokens starting
     from a random prefix position P, reward a sampled token iff it is
     in the filtered D+ content-token set of that item's source engram
     (and not in the prefix).
  3. Running baseline (100-step mean, init 0.05) for variance reduction.
  4. total = LM_CE + 0.1 * cat_loss + 0.3 * grounding_loss.
  5. Everything else matches task 7 exactly.

Outputs to results/v18_raft_sampled/.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
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

from experiments.hrs_loop.raft_train import (
    swap_buffer,
    restore_buffer,
    compute_query_engrams,
    get_gate_values,
    eval_mauve_500,
)
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
def build_batch_buffer(
    model, x, store_keys_gpu, baseline_buffer, extract_layer, device,
    p_retrieve=0.5, p_baseline=0.25, p_wrong=0.25,
    query_len=64, rng=None,
):
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
            v = store_keys_gpu[idx]
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
        else:
            v = store_keys_gpu[top_idxs[i]]
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
        slots.append(slot)
    buf = torch.stack(slots, dim=0).to(device).contiguous()
    return buf, conditions, top_idxs


# ------------------------------------------------------------
# Distinctive content-token filter
# ------------------------------------------------------------
def filter_distinctive_ids(token_ids: list[int], tokenizer) -> set[int]:
    """Keep tokens whose decoded string (stripped of BPE leading space) is
    distinctive: len >= 4 OR has uppercase OR has digit. Drops short common
    function words like 'the', 'of', 'and', etc."""
    out = set()
    for tid in token_ids:
        s = tokenizer.decode([tid])
        s = s.strip()
        if not s:
            continue
        if len(s) >= 4:
            out.add(tid)
        elif any(c.isupper() for c in s):
            out.add(tid)
        elif any(c.isdigit() for c in s):
            out.add(tid)
    return out


# ------------------------------------------------------------
# REINFORCE rollout-based grounding loss
# ------------------------------------------------------------
class RunningBaseline:
    def __init__(self, init_value: float = 0.05, window: int = 100):
        self.init_value = init_value
        self.window = window
        self.buf = collections.deque(maxlen=window)

    def value(self) -> float:
        if not self.buf:
            return self.init_value
        return sum(self.buf) / len(self.buf)

    def update(self, reward: float):
        self.buf.append(float(reward))


def rollout_reinforce_backward(
    model, x, conditions, top_idxs, store_keys_gpu, store_entries, tokenizer,
    baseline: RunningBaseline, rng: random.Random, weight: float,
    K: int = 8, P_min: int = 32, top_k: int = 50, temperature: float = 1.0,
    content_cap: int = 512,
) -> dict:
    """For each retrieve-active item, run a K-step on-policy rollout and
    backward its per-step REINFORCE loss contribution immediately so
    activations from that forward pass are freed before the next. Gradients
    accumulate on the model's cross-attn / gate params.

    Returns stats dict. Does NOT zero_grad or step — caller handles both.
    Model parameters must have grad-enabled requires_grad where relevant.
    """
    device = x.device
    active = [i for i, c in enumerate(conditions) if c == "retrieve"]
    stats = {
        "n_retrieve_items": len(active),
        "n_rollouts": 0,
        "mean_reward": 0.0,
        "baseline": baseline.value(),
        "loss_val": 0.0,
    }
    L = x.shape[1]
    P_max = L - 16 - K
    if not active or P_max <= P_min:
        return stats

    P = rng.randint(P_min, P_max)
    baseline_b = baseline.value()

    loss_sum = 0.0
    reward_sum = 0.0
    n_rollouts = 0

    # Iterate one active item at a time; within each, step one token at a time
    # and backward per-step. Peak memory = one forward pass.
    for i in active:
        # Single-item (1, 32, D) buffer
        v = store_keys_gpu[top_idxs[i]]
        item_buf = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous().unsqueeze(0)

        # Build D+ content set for this item
        text = store_entries[top_idxs[i]].text
        ids = tokenizer.encode(text, add_special_tokens=False)[:content_cap]
        d_plus = filter_distinctive_ids(ids, tokenizer) - set(x[i, :P].tolist())

        gen = x[i:i+1, :P].clone()
        orig_buf, orig_init = swap_buffer(model, item_buf)
        try:
            for _k in range(K):
                output = model(gen)
                next_logits = output.logits[:, -1, :].float() / max(temperature, 1e-6)
                topk_vals, topk_idx = next_logits.topk(top_k, dim=-1)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(-1, topk_idx, topk_vals)
                probs = F.softmax(mask, dim=-1)
                logp_all = F.log_softmax(mask, dim=-1)
                sampled = torch.multinomial(probs, 1).detach()          # (1, 1)
                sampled_logp = logp_all.gather(-1, sampled).squeeze()    # scalar

                tok = int(sampled.item())
                r = 1.0 if tok in d_plus else 0.0
                adv = r - baseline_b

                # Per-step backward; scale by weight and 1/(N_active*K) so
                # the accumulated gradient matches -(mean A*logp) * weight.
                step_loss = -weight * adv * sampled_logp / (len(active) * K)
                step_loss.backward()

                loss_sum += float((-adv * sampled_logp).item())  # unweighted for logging
                reward_sum += r
                n_rollouts += 1

                # Append detached sampled token for next step's input
                gen = torch.cat([gen, sampled.detach()], dim=1)
        finally:
            restore_buffer(model, orig_buf, orig_init)

    if n_rollouts == 0:
        return stats

    mean_reward = reward_sum / n_rollouts
    baseline.update(mean_reward)
    stats.update({
        "n_rollouts": n_rollouts,
        "mean_reward": mean_reward,
        "baseline": baseline_b,
        "loss_val": loss_sum / n_rollouts,  # mean of (-A*logp), unweighted
    })
    return stats


# ------------------------------------------------------------
def eval_niah_expanded(model, tokenizer, device, distractors, needles_all,
                       token_set="cleaned_answer_tokens", seed=42):
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
def get_lr(step, max_steps=5000, warmup=500, base=1e-5, final=1e-6):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return final + (base - final) * cos


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
    ap.add_argument("--lr-ca", type=float, default=1e-5)
    ap.add_argument("--lr-final", type=float, default=1e-6)
    ap.add_argument("--gate-lr-mult", type=float, default=100.0)
    ap.add_argument("--mauve-n", type=int, default=200)
    ap.add_argument("--p-retrieve", type=float, default=0.5)
    ap.add_argument("--p-baseline", type=float, default=0.25)
    ap.add_argument("--p-wrong", type=float, default=0.25)
    ap.add_argument("--grounding-weight", type=float, default=0.3)
    ap.add_argument("--cat-weight", type=float, default=0.1)
    ap.add_argument("--rollout-k", type=int, default=8)
    ap.add_argument("--rollout-pmin", type=int, default=32)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--rollout-topk", type=int, default=50)
    ap.add_argument("--baseline-init", type=float, default=0.05)
    ap.add_argument("--baseline-window", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="results/v18_raft_sampled")
    ap.add_argument("--max-wall-hours", type=float, default=6.0)
    args = ap.parse_args()
    assert abs(args.p_retrieve + args.p_baseline + args.p_wrong - 1.0) < 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    base_ckpt = Path(args.base_ckpt)
    if not base_ckpt.is_absolute():
        base_ckpt = REPO / base_ckpt
    model, cfg = load_ckpt(base_ckpt, device)
    baseline_buffer = model.engram_buffer.detach().clone().to(device)
    print(f"Baseline buffer slot-norm: {baseline_buffer[0, 0].norm().item():.3f}")

    store = EngramStore.load(str(REPO / "engram_store_data"))
    store_keys_gpu = store.keys.to(device)
    print(f"Store: {len(store)} entries")

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

    print("Building expanded-benchmark distractor pool (one-time)...")
    eval_distractors = build_distractor_pool(n_wt103_extra=20, seed=args.seed)
    eval_needles = load_needles()
    print(f"Eval: {len(eval_needles)} needles + {len(eval_distractors)} distractors")

    gate_params, ca_other_params, backbone_params = [], [], []
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_logit") or name.endswith(".cross_attn.gate_scalar"):
            gate_params.append(p)
        elif "cross_attn" in name or "categorization_head" in name:
            ca_other_params.append(p)
        else:
            backbone_params.append(p)
    print(f"Gate params:         {sum(p.numel() for p in gate_params):,} ({len(gate_params)} tensors)")
    print(f"Cross-attn+cat rest: {sum(p.numel() for p in ca_other_params):,} ({len(ca_other_params)} tensors)")
    print(f"Backbone:            {sum(p.numel() for p in backbone_params):,} (FROZEN for all steps)")

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
        "lr_ca": 0.0, "lr_gate": 0.0,
        "ce_loss": None, "cat_loss": None, "grounding_loss": None,
        "mean_reward": None, "reward_ma100": None, "baseline": args.baseline_init,
    }]

    print(f"\n=== Starting RAFT sampled-REINFORCE: {args.max_steps} steps, backbone FROZEN, "
          f"gate LR = {args.gate_lr_mult}x base, grounding weight = {args.grounding_weight}, "
          f"K={args.rollout_k} ===\n")

    model.train()
    t0 = time.time()
    step = 0
    cond_counts = {"retrieve": 0, "baseline": 0, "wrong": 0}
    accum = {"ce": 0.0, "cat": 0.0, "ground": 0.0, "reward": 0.0, "n": 0,
             "n_rollouts": 0}
    reward_history = collections.deque(maxlen=100)
    variance_alarm_counter = 0
    baseline_tracker = RunningBaseline(args.baseline_init, args.baseline_window)

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

        lr_ca = get_lr(step, args.max_steps, args.warmup, args.lr_ca, args.lr_final)
        lr_gate = lr_ca * args.gate_lr_mult
        optimizer.param_groups[0]["lr"] = lr_gate
        optimizer.param_groups[1]["lr"] = lr_ca
        optimizer.zero_grad()

        buf, conditions, top_idxs = build_batch_buffer(
            model, x, store_keys_gpu, baseline_buffer, EXTRACT_LAYER, device,
            p_retrieve=args.p_retrieve, p_baseline=args.p_baseline,
            p_wrong=args.p_wrong, query_len=64, rng=rng,
        )
        for c in conditions:
            cond_counts[c] += 1

        # --- Main forward + backward (frees main-graph activations before rollout) ---
        orig_buf, orig_init = swap_buffer(model, buf)
        try:
            output = model(x, step=step)
            logits = output.logits
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            main_loss = ce_loss

            cat_loss_val = 0.0
            if output.categorization_logits is not None and cat_ids is not None:
                cat_loss = F.cross_entropy(output.categorization_logits, cat_ids)
                main_loss = main_loss + args.cat_weight * cat_loss
                cat_loss_val = float(cat_loss.item())
        finally:
            restore_buffer(model, orig_buf, orig_init)

        if not torch.isfinite(main_loss):
            print(f"!!! NaN/inf in main_loss at step {step} — stopping")
            break
        main_loss.backward()
        main_loss_val = float(main_loss.item())

        # --- Rollout REINFORCE: per-step backward to cap peak memory ---
        g_stats = rollout_reinforce_backward(
            model, x, conditions, top_idxs, store_keys_gpu, store.entries, tokenizer,
            baseline_tracker, rng, weight=args.grounding_weight,
            K=args.rollout_k, P_min=args.rollout_pmin,
            top_k=args.rollout_topk, temperature=args.rollout_temperature,
        )
        ground_loss_val = g_stats["loss_val"]

        total_loss_val = main_loss_val + args.grounding_weight * ground_loss_val

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

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        step += 1
        accum["ce"] += float(ce_loss.item())
        accum["cat"] += cat_loss_val
        accum["ground"] += ground_loss_val
        accum["n"] += 1
        if g_stats["n_rollouts"] > 0:
            accum["reward"] += g_stats["mean_reward"]
            accum["n_rollouts"] += 1
            reward_history.append(g_stats["mean_reward"])

        # Variance alarm (advisory; don't actually halt training)
        if len(reward_history) >= 50:
            mu = sum(reward_history) / len(reward_history)
            if mu > 1e-6:
                var = statistics.pvariance(reward_history)
                if var > 5.0 * mu:
                    variance_alarm_counter += 1
                else:
                    variance_alarm_counter = 0
                if variance_alarm_counter == 500:
                    print(f"  *** variance alarm: var={var:.3f} > 5*mu={5*mu:.3f} "
                          f"for 500 consecutive steps ***")

        if step % args.log_interval == 0:
            gates = get_gate_values(model)
            gate_str = " ".join(f"L{k}={v:.3f}" for k, v in gates.items())
            el = time.time() - t0
            rate = step / el if el > 0 else 0
            avg_ce = accum["ce"] / accum["n"]
            avg_cat = accum["cat"] / accum["n"]
            avg_gr = accum["ground"] / accum["n"]
            avg_rew = (accum["reward"] / accum["n_rollouts"]) if accum["n_rollouts"] else 0.0
            rew_ma = (sum(reward_history) / len(reward_history)) if reward_history else 0.0
            mix = " ".join(f"{k[:3]}={cond_counts[k]}" for k in ("retrieve", "baseline", "wrong"))
            print(f"[step {step:4d}/{args.max_steps}] "
                  f"ce={avg_ce:.3f} cat={avg_cat:.3f} "
                  f"ground={avg_gr:+.4f} rew={avg_rew:.3f} rew_ma100={rew_ma:.3f} "
                  f"b={baseline_tracker.value():.3f} "
                  f"lr_ca={lr_ca:.2e} lr_g={lr_gate:.2e} "
                  f"gates: {gate_str} cond({mix}) {rate:.2f}it/s elapsed={el:.0f}s")
            accum = {"ce": 0.0, "cat": 0.0, "ground": 0.0, "reward": 0.0,
                     "n": 0, "n_rollouts": 0}

        if step % args.eval_interval == 0 or step == args.max_steps:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
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

            rew_ma = (sum(reward_history) / len(reward_history)) if reward_history else 0.0
            entry = {
                "step": step, "phase": "sampled",
                "niah_cleaned_recall": niah["recall_pooled"],
                "niah_hits": niah["total_hits"],
                "niah_total": niah["total_possible"],
                "niah_acc_at_1": niah["acc_at_1"],
                "niah_mean_rank": niah["mean_rank"],
                "mauve_500_n200": mauve_score,
                "gates": gates,
                "lr_ca": lr_ca, "lr_gate": lr_gate,
                "ce_loss": float(ce_loss.item()),
                "cat_loss": cat_loss_val,
                "grounding_loss": ground_loss_val,
                "mean_reward": g_stats["mean_reward"],
                "reward_ma100": rew_ma,
                "baseline": baseline_tracker.value(),
                "elapsed_s": time.time() - t0,
            }
            history.append(entry)
            print(f"  NIAH cleaned: recall={niah['recall_pooled']:.3f} "
                  f"({niah['total_hits']}/{niah['total_possible']}) "
                  f"acc@1={niah['acc_at_1']:.2f} mean_rank={niah['mean_rank']:.2f}")
            print(f"  MAUVE-500 (n={args.mauve_n}): {mauve_score:.4f}")
            print(f"  Gates: {gates}")
            print(f"  reward_ma100={rew_ma:.3f} baseline={baseline_tracker.value():.3f}")
            print(f"  elapsed={entry['elapsed_s']:.0f}s")

            if mauve_score < 0.90:
                print(f"  *** WARNING: MAUVE below 0.90 floor ({mauve_score:.4f}) ***")

            ckpt_path = out_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "lr_ca": lr_ca, "lr_gate": lr_gate,
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
