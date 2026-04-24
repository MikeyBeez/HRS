"""Task 9 diagnostics on the step-5000 per-head checkpoint.

Three diagnostics:
  1. Seed variance: re-run eval_niah_perhead at 3 different seeds.
     Is 3/138 stable, or a lucky roll?
  2. Per-head DISABLED at inference: same trained checkpoint, but the
     per-head state is never set during eval — so the monkey-patched
     forward falls through to V18's shared-engram behavior. Does the
     recall gain disappear, or does it persist?
  3. Per-head attention entropy: during one eval, log the mean attention
     weight head h places on slot 16+h (diagonal), on other per-head
     slots (off-diagonal), and on shared slots. Specialization predicts
     diagonal > off-diagonal.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

# Importing raft_perhead_train triggers the monkey-patch and gives us helpers
from experiments.hrs_loop import raft_perhead_train as rpt
from experiments.hrs_loop.raft_train import swap_buffer, restore_buffer
from experiments.hrs_loop.niah_benchmark.eval_niah_v2 import (
    load_needles, build_distractor_pool,
)

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer
from engram import EngramCrossAttention

CROSS_ATTN_LAYERS = rpt.CROSS_ATTN_LAYERS
N_SHARED_SLOTS = rpt.N_SHARED_SLOTS
N_PERHEAD_SLOTS = rpt.N_PERHEAD_SLOTS


# ============================================================
# Diagnostic 3: attention-entropy-aware forward
# ============================================================
_ATTN_LOG_FORWARD = rpt._perhead_forward  # patched forward (per-head on)
_BASELINE_FORWARD = rpt._ORIGINAL_CA_FORWARD


def _forward_with_attn_log(self, h, engram):
    """Same as the patched forward, but computes attention with manual
    softmax so attention weights are exposed. Stashes mean attention
    weight vector (over T,B) on self._logged_attn_weights as (H, 32).
    Only active when self._log_attn = True AND per-head state is set.
    """
    src_h = getattr(self, "_perhead_src_h", None)
    log = getattr(self, "_log_attn", False)
    if src_h is None or not log:
        # fall through to patched (or baseline) forward
        return _ATTN_LOG_FORWARD(self, h, engram)

    B, T, D = h.shape
    H = self.n_heads
    hd = self.head_dim
    device = h.device
    dtype = h.dtype
    scale = 1.0 / math.sqrt(hd)

    if engram.shape[0] == 1 and B > 1:
        engram = engram.expand(B, -1, -1)

    q = self.q_proj(h).reshape(B, T, H, hd).transpose(1, 2)
    shared = engram[:, :N_SHARED_SLOTS]
    k_shared = self.k_proj(shared).reshape(B, N_SHARED_SLOTS, H, hd)
    v_shared = self.v_proj(shared).reshape(B, N_SHARED_SLOTS, H, hd)

    k_base_one = k_shared[:, 0:1, :, :]
    v_base_one = v_shared[:, 0:1, :, :]
    k_base = k_base_one.expand(B, N_PERHEAD_SLOTS, H, hd).contiguous()
    v_base = v_base_one.expand(B, N_PERHEAD_SLOTS, H, hd).contiguous()

    src_len = self._perhead_src_len
    active = self._perhead_active
    src_h_d = src_h.to(dtype)
    S = src_h_d.shape[1]
    K_src = self.k_proj(src_h_d).reshape(B, S, H, hd).transpose(1, 2)
    V_src = self.v_proj(src_h_d).reshape(B, S, H, hd).transpose(1, 2)
    q_mean = q.mean(dim=2)
    scores = torch.einsum("bhd,bhsd->bhs", q_mean, K_src) * scale
    arange_S = torch.arange(S, device=device)
    valid = arange_S.unsqueeze(0) < src_len.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    attn_w_pool = F.softmax(scores, dim=-1)
    E_k = torch.einsum("bhs,bhsd->bhd", attn_w_pool, K_src)
    E_v = torch.einsum("bhs,bhsd->bhd", attn_w_pool, V_src)

    ph_k = k_base.clone()
    ph_v = v_base.clone()
    diag = torch.arange(H, device=device)
    a = active.view(B, 1, 1).to(dtype)
    ph_k_diag_old = ph_k[:, diag, diag, :]
    ph_v_diag_old = ph_v[:, diag, diag, :]
    ph_k[:, diag, diag, :] = a * E_k + (1.0 - a) * ph_k_diag_old
    ph_v[:, diag, diag, :] = a * E_v + (1.0 - a) * ph_v_diag_old

    ph_k_t = ph_k.transpose(1, 2).contiguous()
    ph_v_t = ph_v.transpose(1, 2).contiguous()
    k_shared_t = k_shared.transpose(1, 2).contiguous()
    v_shared_t = v_shared.transpose(1, 2).contiguous()
    k = torch.cat([k_shared_t, ph_k_t], dim=2)
    v = torch.cat([v_shared_t, ph_v_t], dim=2)

    # Manual attention to expose weights
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, 32)
    attn_weights = F.softmax(attn_scores, dim=-1)
    # Mean over (B, T): per-head distribution over 32 slots
    mean_attn = attn_weights.mean(dim=(0, 2)).detach()  # (H, 32)
    self._logged_attn_weights = mean_attn

    out = torch.matmul(attn_weights, v)  # (B, H, T, hd)
    out = out.transpose(1, 2).reshape(B, T, D)
    out = self.resid_dropout(self.out_proj(out))
    gate = torch.sigmoid(self.gate_logit) * F.softplus(self.gate_scalar)
    return gate * out


# ============================================================
# Unified eval (per-head on or off)
# ============================================================
@torch.no_grad()
def eval_perhead_mode(model, tokenizer, device, distractors, needles,
                     use_perhead: bool, seed: int,
                     log_attn: bool = False, token_set="cleaned_answer_tokens",
                     max_new_tokens=100, temperature=0.9, top_k=50):
    """Run Config-B-style eval; if use_perhead=True, set per-head state per needle.
    Returns dict with total_hits, total_possible, per_needle list, and optionally
    attn_log (dict layer_idx -> (H, 32) mean-attention tensor averaged over needles).
    """
    was_training = model.training
    model.eval()

    baseline_buffer = model.engram_buffer.detach().clone()

    per_needle = []
    attn_accum = {l: torch.zeros(16, 32, device=device) for l in CROSS_ATTN_LAYERS} if log_attn else None
    n_attn_obs = 0

    # Install attn-log forward if requested
    if log_attn:
        EngramCrossAttention.forward = _forward_with_attn_log

    try:
        for nd_idx, nd in enumerate(needles):
            fact = nd["fact"]
            query = nd["query"]
            ans_tokens = nd[token_set]

            picks = random.Random(seed + nd_idx).sample(distractors, min(len(distractors), 20))
            insert_at = len(picks) // 2
            context = "\n\n".join(picks[:insert_at] + [fact] + picks[insert_at:])

            # Per-head context (only consumed if use_perhead=True)
            src_h = rpt.compute_src_h_from_text(model, tokenizer, context, device)
            S = src_h.shape[0]
            mean_vec = src_h.float().mean(dim=0).to(device)
            buf = mean_vec.unsqueeze(0).unsqueeze(0).expand(1, rpt.N_BUFFER_SLOTS, -1).contiguous()

            if use_perhead:
                rpt.set_perhead_state(
                    model,
                    src_h.unsqueeze(0).to(device),
                    torch.tensor([S], device=device, dtype=torch.long),
                    torch.tensor([True], device=device, dtype=torch.bool),
                )
                if log_attn:
                    for l in CROSS_ATTN_LAYERS:
                        model.blocks[l].cross_attn._log_attn = True
            else:
                rpt.clear_perhead_state(model)

            orig_buf, orig_init = swap_buffer(model, buf)
            try:
                q_ids = tokenizer.encode(query, add_special_tokens=False)
                gen_ids = torch.tensor([q_ids], device=device, dtype=torch.long)
                # One forward to populate attention logs (for log_attn mode)
                if log_attn:
                    _ = model(gen_ids)
                    for l in CROSS_ATTN_LAYERS:
                        aw = getattr(model.blocks[l].cross_attn, "_logged_attn_weights", None)
                        if aw is not None:
                            attn_accum[l] += aw
                    n_attn_obs += 1

                generated = []
                # Disable attn logging during generation loop (too much noise)
                if log_attn:
                    for l in CROSS_ATTN_LAYERS:
                        model.blocks[l].cross_attn._log_attn = False
                for _ in range(max_new_tokens):
                    output = model(gen_ids)
                    next_logits = output.logits[0, -1] / max(temperature, 1e-6)
                    vals, idx = next_logits.topk(top_k)
                    mask = torch.full_like(next_logits, float("-inf"))
                    mask[idx] = vals
                    probs = F.softmax(mask, dim=-1)
                    tok = torch.multinomial(probs, 1).item()
                    generated.append(tok)
                    gen_ids = torch.cat([gen_ids, torch.tensor([[tok]], device=device)], dim=1)
                cont_text = tokenizer.decode(generated)
                cont_lower = cont_text.lower()
                hits = sum(1 for t in ans_tokens if t.lower() in cont_lower)
                per_needle.append({
                    "id": nd["id"], "n_hits": hits, "n_answer_tokens": len(ans_tokens),
                })
            finally:
                restore_buffer(model, orig_buf, orig_init)
                rpt.clear_perhead_state(model)
                if log_attn:
                    for l in CROSS_ATTN_LAYERS:
                        model.blocks[l].cross_attn._log_attn = False

            if (nd_idx + 1) % 5 == 0 or nd_idx == len(needles) - 1:
                print(f"  [{'ph' if use_perhead else 'base'}/seed{seed}] {nd_idx+1}/{len(needles)}")
    finally:
        # Restore the normal patched forward (no attn logging) for subsequent calls
        EngramCrossAttention.forward = _ATTN_LOG_FORWARD

    total_hits = sum(r["n_hits"] for r in per_needle)
    total_possible = sum(r["n_answer_tokens"] for r in per_needle)
    result = {
        "total_hits": total_hits,
        "total_possible": total_possible,
        "recall": total_hits / max(total_possible, 1),
        "per_needle": per_needle,
    }
    if log_attn and n_attn_obs > 0:
        result["attn_log"] = {l: (attn_accum[l] / n_attn_obs).cpu().tolist()
                              for l in CROSS_ATTN_LAYERS}
        result["n_attn_obs"] = n_attn_obs

    if was_training:
        model.train()
    return result


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="results/v18_perhead/checkpoint_5000.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO / args.ckpt
    model, _cfg = rpt.load_ckpt(ckpt_path, device)
    model.eval()

    # Load per-head cache (for compute_src_h_from_text we use model's own forward,
    # so the stored cache is unused here; still imported indirectly via model only)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    eval_distractors = build_distractor_pool(n_wt103_extra=20, seed=42)
    eval_needles = load_needles()
    print(f"Eval: {len(eval_needles)} needles + {len(eval_distractors)} distractors\n")

    results = {
        "ckpt": str(ckpt_path),
        "seeds": args.seeds,
    }

    # ---------- DIAGNOSTIC 1: seed variance, per-head ON ----------
    print("=== Diagnostic 1: seed variance (per-head ENABLED) ===")
    d1 = []
    for s in args.seeds:
        r = eval_perhead_mode(model, tokenizer, device, eval_distractors, eval_needles,
                              use_perhead=True, seed=s)
        print(f"  seed={s}: {r['total_hits']}/{r['total_possible']} "
              f"(recall={r['recall']:.3f})")
        d1.append({"seed": s, "hits": r["total_hits"], "total": r["total_possible"],
                   "recall": r["recall"]})
    results["diagnostic_1_seed_variance"] = d1

    # ---------- DIAGNOSTIC 2: per-head DISABLED at inference ----------
    print("\n=== Diagnostic 2: per-head DISABLED at inference ===")
    d2 = []
    for s in args.seeds:
        r = eval_perhead_mode(model, tokenizer, device, eval_distractors, eval_needles,
                              use_perhead=False, seed=s)
        print(f"  seed={s}: {r['total_hits']}/{r['total_possible']} "
              f"(recall={r['recall']:.3f})  [per-head OFF]")
        d2.append({"seed": s, "hits": r["total_hits"], "total": r["total_possible"],
                   "recall": r["recall"]})
    results["diagnostic_2_perhead_disabled"] = d2

    # ---------- DIAGNOSTIC 3: attention entropy ----------
    print("\n=== Diagnostic 3: attention entropy (per-head ENABLED, attn logged) ===")
    r3 = eval_perhead_mode(model, tokenizer, device, eval_distractors, eval_needles,
                           use_perhead=True, seed=42, log_attn=True)
    # Analyze
    attn_stats_all = {}
    print(f"  (averaged over {r3.get('n_attn_obs', 0)} needle contexts)")
    print(f"  {'layer':>5s}  {'head':>4s}  {'diag':>7s}  {'off-diag':>8s}  {'shared':>7s}  "
          f"{'entropy':>7s}")
    for layer_idx in CROSS_ATTN_LAYERS:
        attn = torch.tensor(r3["attn_log"][layer_idx])  # (H=16, 32)
        diag_vals = []
        offdiag_vals = []
        shared_vals = []
        entropies = []
        for h in range(16):
            row = attn[h]  # (32,)
            diag = row[16 + h].item()                # slot meant for this head
            offdiag = (row[16:].sum().item() - diag) / 15.0  # mean over 15 other per-head slots
            shared = row[:16].mean().item()         # mean over shared slots
            ent = -(row * (row.clamp_min(1e-12)).log()).sum().item()
            diag_vals.append(diag); offdiag_vals.append(offdiag)
            shared_vals.append(shared); entropies.append(ent)
        mean_diag = sum(diag_vals) / 16
        mean_off = sum(offdiag_vals) / 16
        mean_shared = sum(shared_vals) / 16
        mean_ent = sum(entropies) / 16
        print(f"  L{layer_idx:>3d}  mean  {mean_diag:>7.4f}  {mean_off:>8.4f}  "
              f"{mean_shared:>7.4f}  {mean_ent:>7.3f}")
        attn_stats_all[layer_idx] = {
            "mean_diagonal": mean_diag,
            "mean_offdiagonal_perhead": mean_off,
            "mean_shared": mean_shared,
            "mean_entropy": mean_ent,
            "per_head": {h: {"diag": diag_vals[h], "offdiag": offdiag_vals[h],
                             "shared": shared_vals[h], "entropy": entropies[h]}
                         for h in range(16)},
        }
    results["diagnostic_3_attention_stats"] = attn_stats_all
    results["diagnostic_3_recall_seed42"] = {
        "hits": r3["total_hits"], "total": r3["total_possible"], "recall": r3["recall"]
    }

    # ---------- Summary ----------
    print("\n=== Summary ===")
    d1_hits = [d["hits"] for d in d1]
    d2_hits = [d["hits"] for d in d2]
    print(f"  Per-head ON  : {d1_hits} (seeds {args.seeds}) -> mean {sum(d1_hits)/len(d1_hits):.2f}")
    print(f"  Per-head OFF : {d2_hits} (seeds {args.seeds}) -> mean {sum(d2_hits)/len(d2_hits):.2f}")
    mean_on = sum(d1_hits) / len(d1_hits)
    mean_off = sum(d2_hits) / len(d2_hits)
    delta = mean_on - mean_off
    print(f"  ON − OFF     : {delta:+.2f} hits on 138-token floor "
          f"({delta/138*100:+.2f}% absolute recall)")

    out_path = REPO / "results/v18_perhead/diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
