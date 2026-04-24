"""RAFT v4: attention-pooled per-head engrams (task 9).

Architectural change: for each cross-attention block, slots 16..31 of the
buffer are replaced with per-head pooled content from the source document.
For each head h, E_h (and E_k_h) are computed by attention-pooling the
source doc's layer-4 hidden states with head h's query (mean-pooled over
the current sequence). Slot h of the per-head half has head-h's specialized
content; other heads see the shared engram's value at that slot.

Only `EngramCrossAttention.forward` is monkey-patched at module load;
other V18/RAFT code paths are unaffected.

Matches task 7 training recipe except:
- Base checkpoint: V18 (not RAFT-2000)
- No grounding aux loss
- Output: results/v18_perhead/
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
from engram import EngramCrossAttention
from data import load_wikitext, build_dataloaders

from experiments.hrs_loop.raft_train import (
    swap_buffer, restore_buffer, compute_query_engrams,
    get_gate_values, eval_mauve_500,
)
from experiments.hrs_loop.niah_benchmark.eval_niah_v2 import (
    load_needles, build_distractor_pool,
)
from experiments.hrs_loop.engram_content_ablation import run_variant_on_needle, VARIANTS

N_BUFFER_SLOTS = 32
N_SHARED_SLOTS = 16
N_PERHEAD_SLOTS = 16
EXTRACT_LAYER = 4
CROSS_ATTN_LAYERS = (1, 3, 5)
INV_SOFTPLUS_1 = math.log(math.e - 1.0)
PERHEAD_CACHE_PATH = REPO / "engram_store_data" / "engram_hiddens_v18_layer4.pt"


# ============================================================
# MONKEY-PATCH: per-head-aware forward for EngramCrossAttention
# ============================================================
_ORIGINAL_CA_FORWARD = EngramCrossAttention.forward


def _perhead_forward(self, h, engram):
    """Patched forward supporting attention-pooled per-head engrams.

    If `self._perhead_src_h` is set, slots 16..31 are constructed from
    source-doc layer-4 hidden states pooled through this block's W_k, W_v.
    Otherwise behaves identically to V18.

    Stashed state (cleared between steps):
      self._perhead_src_h   : (B, max_src_len, D) source hidden states
      self._perhead_src_len : (B,) long — per-item source length
      self._perhead_active  : (B,) bool — which items use per-head content
    """
    src_h = getattr(self, "_perhead_src_h", None)
    if src_h is None:
        return _ORIGINAL_CA_FORWARD(self, h, engram)

    B, T, D = h.shape
    H = self.n_heads
    hd = self.head_dim
    device = h.device
    dtype = h.dtype
    scale = 1.0 / math.sqrt(hd)

    # Broadcast engram if needed
    if engram.shape[0] == 1 and B > 1:
        engram = engram.expand(B, -1, -1)

    # Q projection (shared with main attention)
    q = self.q_proj(h).reshape(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)

    # --- Shared engram portion: first 16 slots through W_k, W_v ---
    shared = engram[:, :N_SHARED_SLOTS]                         # (B, 16, D)
    k_shared = self.k_proj(shared).reshape(B, N_SHARED_SLOTS, H, hd)  # (B, 16, H, hd)
    v_shared = self.v_proj(shared).reshape(B, N_SHARED_SLOTS, H, hd)

    # k_base: per-head baseline (what non-diagonal heads see at per-head slots).
    # Use the shared engram's projected K/V as the baseline, broadcast over the
    # 16 per-head slots.
    # We take the FIRST shared slot as canonical (they're identical).
    k_base_one = k_shared[:, 0:1, :, :]                   # (B, 1, H, hd)
    v_base_one = v_shared[:, 0:1, :, :]
    k_base = k_base_one.expand(B, N_PERHEAD_SLOTS, H, hd).contiguous()  # (B, 16, H, hd)
    v_base = v_base_one.expand(B, N_PERHEAD_SLOTS, H, hd).contiguous()

    # --- Per-head pooled engrams from source hidden states ---
    src_len = self._perhead_src_len                       # (B,)
    active = self._perhead_active                         # (B,)
    src_h_d = src_h.to(dtype)                             # (B, S, D)
    S = src_h_d.shape[1]

    K_src = self.k_proj(src_h_d).reshape(B, S, H, hd).transpose(1, 2)  # (B, H, S, hd)
    V_src = self.v_proj(src_h_d).reshape(B, S, H, hd).transpose(1, 2)

    # q_mean per head (mean over current sequence T)
    q_mean = q.mean(dim=2)                                # (B, H, hd)

    # Attention scores over source positions, per head
    scores = torch.einsum("bhd,bhsd->bhs", q_mean, K_src) * scale  # (B, H, S)

    # Mask padding
    arange_S = torch.arange(S, device=device)
    valid = arange_S.unsqueeze(0) < src_len.unsqueeze(1)  # (B, S)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    attn_w = F.softmax(scores, dim=-1)                    # (B, H, S)

    # Pooled K, V per head from source
    E_k = torch.einsum("bhs,bhsd->bhd", attn_w, K_src)    # (B, H, hd)
    E_v = torch.einsum("bhs,bhsd->bhd", attn_w, V_src)

    # --- Assemble per-head slots ---
    # Start from the shared baseline (so non-diagonal heads see the shared
    # engram's K/V at these slots), then override the diagonal.
    ph_k = k_base.clone()                                 # (B, 16, H, hd)
    ph_v = v_base.clone()
    diag = torch.arange(H, device=device)
    # scatter: for b, slot=h, head=h: E_k[b, h]
    ph_k_diag_new = E_k                                   # (B, H, hd)
    ph_v_diag_new = E_v

    # Blend by active mask so non-active items keep shared baseline on diag
    a = active.view(B, 1, 1).to(dtype)                    # (B, 1, 1)
    ph_k_diag_old = ph_k[:, diag, diag, :]                # (B, H, hd)
    ph_v_diag_old = ph_v[:, diag, diag, :]
    ph_k[:, diag, diag, :] = a * ph_k_diag_new + (1.0 - a) * ph_k_diag_old
    ph_v[:, diag, diag, :] = a * ph_v_diag_new + (1.0 - a) * ph_v_diag_old

    # Transpose to attention layout: (B, H, 16_slots, hd)
    ph_k_t = ph_k.transpose(1, 2).contiguous()
    ph_v_t = ph_v.transpose(1, 2).contiguous()
    k_shared_t = k_shared.transpose(1, 2).contiguous()    # (B, H, 16, hd)
    v_shared_t = v_shared.transpose(1, 2).contiguous()

    k = torch.cat([k_shared_t, ph_k_t], dim=2)            # (B, H, 32, hd)
    v = torch.cat([v_shared_t, ph_v_t], dim=2)

    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=False,
        dropout_p=self.attn_dropout.p if self.training else 0.0,
    )
    out = out.transpose(1, 2).reshape(B, T, D)
    out = self.resid_dropout(self.out_proj(out))
    gate = torch.sigmoid(self.gate_logit) * F.softplus(self.gate_scalar)
    return gate * out


EngramCrossAttention.forward = _perhead_forward


# ============================================================
# Utilities to set/clear per-head state on all CA blocks
# ============================================================
def set_perhead_state(model, src_h, src_len, active):
    for layer_idx in CROSS_ATTN_LAYERS:
        ca = model.blocks[layer_idx].cross_attn
        ca._perhead_src_h = src_h
        ca._perhead_src_len = src_len
        ca._perhead_active = active


def clear_perhead_state(model):
    for layer_idx in CROSS_ATTN_LAYERS:
        ca = model.blocks[layer_idx].cross_attn
        ca._perhead_src_h = None
        ca._perhead_src_len = None
        ca._perhead_active = None


# ============================================================
# Checkpoint loader
# ============================================================
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


# ============================================================
# Batch buffer + source-hidden-state assembly
# ============================================================
def build_batch_buffer_and_src(
    model, x, store_keys_gpu, perhead_hiddens, perhead_lengths,
    baseline_buffer, extract_layer, device,
    p_retrieve=0.5, p_baseline=0.25, p_wrong=0.25,
    query_len=64, rng=None,
):
    """Returns (buffer (B,32,D), conditions, top_idxs, src_h (B,S,D),
    src_len (B,), active (B,) bool).

    For retrieve items: src_h is source doc's layer-4 hidden states, active=True.
    For wrong items: src_h is a random store entry's hidden states (matches
        the wrong-engram slot identity), active=True.
    For baseline items: src_h is zero with src_len=1, active=False.
    """
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
    n_store = store_keys_gpu.shape[0]
    if any(c == "retrieve" for c in conditions):
        # Clear per-head state during query forward (V18 behavior for query extract)
        clear_perhead_state(model)
        q_engs = compute_query_engrams(model, x[:, :query_len], extract_layer)
        q_norm = F.normalize(q_engs.float(), dim=1)
        sims = q_norm @ store_keys_gpu.T
        best = sims.argmax(dim=1)
        for i, c in enumerate(conditions):
            if c == "retrieve":
                top_idxs[i] = int(best[i].item())

    # Pick wrong-engram indices for wrong condition
    for i, c in enumerate(conditions):
        if c == "wrong":
            wrong_idxs[i] = rng.randint(0, n_store - 1)

    slots = []
    src_h_batch: list[torch.Tensor] = []  # (S, D) per item
    src_len_batch: list[int] = []
    active_batch: list[bool] = []

    for i, c in enumerate(conditions):
        if c == "baseline":
            slot = baseline_buffer.squeeze(0)                     # (32, D)
            src_h_batch.append(torch.zeros(1, perhead_hiddens.shape[-1],
                                            device=device, dtype=perhead_hiddens.dtype))
            src_len_batch.append(1)
            active_batch.append(False)
        elif c == "wrong":
            idx = wrong_idxs[i]
            v = store_keys_gpu[idx]
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
            L = int(perhead_lengths[idx].item())
            src_h_batch.append(perhead_hiddens[idx, :L])
            src_len_batch.append(L)
            active_batch.append(True)
        else:  # retrieve
            idx = top_idxs[i]
            v = store_keys_gpu[idx]
            slot = v.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
            L = int(perhead_lengths[idx].item())
            src_h_batch.append(perhead_hiddens[idx, :L])
            src_len_batch.append(L)
            active_batch.append(True)
        slots.append(slot)

    buf = torch.stack(slots, dim=0).to(device).contiguous()

    # Pad src_h to max length in batch
    max_S = max(src_len_batch)
    D_h = perhead_hiddens.shape[-1]
    src_h = torch.zeros((B, max_S, D_h), device=device, dtype=perhead_hiddens.dtype)
    for i, h in enumerate(src_h_batch):
        src_h[i, :h.shape[0]] = h
    src_len = torch.tensor(src_len_batch, device=device, dtype=torch.long)
    active = torch.tensor(active_batch, device=device, dtype=torch.bool)

    return buf, conditions, top_idxs, src_h, src_len, active


# ============================================================
# Per-head-aware NIAH eval (computes src_h from needle+distractors text)
# ============================================================
@torch.no_grad()
def compute_src_h_from_text(model, tokenizer, text, device, max_len=512):
    """Forward the text through the model with per-head DISABLED, capture
    layer-4 hidden state output. Returns (L, D) fp16."""
    clear_perhead_state(model)
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    capture = {}

    def hook(m, _in, out):
        capture["h"] = out[0].detach()[0].half()  # (L, D)

    handle = model.blocks[EXTRACT_LAYER].register_forward_hook(hook)
    try:
        _ = model(x, step=0)
    finally:
        handle.remove()
    return capture["h"]


@torch.no_grad()
def eval_niah_perhead(model, tokenizer, device, distractors, needles_all,
                     token_set="cleaned_answer_tokens", seed=42,
                     max_new_tokens=100, temperature=0.9, top_k=50):
    """Task-4-style eval with per-head engrams live during generation.

    For each needle:
      1. Compose fact+distractors into a 'full text'; compute layer-4 hiddens.
      2. Build buffer from the fact's mean-pooled engram (matches v1_baseline).
      3. Set per-head state = full-text hiddens, active=True.
      4. Generate continuation on the query; score recall on answer tokens.
    """
    import random as _random
    was_training = model.training
    model.eval()

    rng = _random.Random(seed)
    per_needle = []

    baseline_buffer = model.engram_buffer.detach().clone()

    for nd_idx, nd in enumerate(needles_all):
        fact = nd["fact"]
        query = nd["query"]
        ans_tokens = nd[token_set]

        # Build "full text" context: shuffled distractors + fact + more distractors
        picks = _random.Random(seed + nd_idx).sample(distractors, min(len(distractors), 20))
        insert_at = (len(picks) // 2)
        pre = picks[:insert_at]
        post = picks[insert_at:]
        context = "\n\n".join(pre + [fact] + post)

        # Source hidden states
        src_h = compute_src_h_from_text(model, tokenizer, context, device)  # (L, D)
        S = src_h.shape[0]

        # Mean-pooled engram for the shared buffer half (v1_baseline)
        mean_vec = src_h.float().mean(dim=0)                # (D,)
        eng_slot = mean_vec.to(device)
        buf = eng_slot.unsqueeze(0).unsqueeze(0).expand(1, N_BUFFER_SLOTS, -1).contiguous()

        # Install per-head state
        set_perhead_state(
            model,
            src_h.unsqueeze(0).to(device),                  # (1, S, D) fp16
            torch.tensor([S], device=device, dtype=torch.long),
            torch.tensor([True], device=device, dtype=torch.bool),
        )

        orig_buf, orig_init = swap_buffer(model, buf)
        try:
            # Tokenize the query
            q_ids = tokenizer.encode(query, add_special_tokens=False)
            gen_ids = torch.tensor([q_ids], device=device, dtype=torch.long)
            generated = []
            for _ in range(max_new_tokens):
                output = model(gen_ids)
                next_logits = output.logits[0, -1] / max(temperature, 1e-6)
                if top_k:
                    vals, idx = next_logits.topk(top_k)
                    mask = torch.full_like(next_logits, float("-inf"))
                    mask[idx] = vals
                    next_logits = mask
                probs = F.softmax(next_logits, dim=-1)
                tok = torch.multinomial(probs, 1).item()
                generated.append(tok)
                gen_ids = torch.cat([gen_ids, torch.tensor([[tok]], device=device)], dim=1)
            cont_text = tokenizer.decode(generated)

            # Score recall (same substring-based logic as task 2/4)
            cont_lower = cont_text.lower()
            hits = sum(1 for t in ans_tokens if t.lower() in cont_lower)
            n_hits = hits
            n_total = len(ans_tokens)

            per_needle.append({
                "id": nd["id"],
                "version": nd.get("version", ""),
                "n_hits": n_hits,
                "n_answer_tokens": n_total,
                "rank": 1,  # retrieval rank placeholder — we don't re-rank here
            })
        finally:
            restore_buffer(model, orig_buf, orig_init)
            clear_perhead_state(model)

        if (nd_idx + 1) % 5 == 0 or nd_idx == len(needles_all) - 1:
            print(f"  [ph_eval] {nd_idx+1}/{len(needles_all)}")

    total_hits = sum(r["n_hits"] for r in per_needle)
    total_possible = sum(r["n_answer_tokens"] for r in per_needle)
    recall_pooled = total_hits / max(total_possible, 1)
    # acc@1 and mean_rank: placeholder constants (we don't re-rank here).
    # The main signal is cleaned_recall anyway.
    if was_training:
        model.train()
    return {
        "recall_pooled": recall_pooled,
        "total_hits": total_hits,
        "total_possible": total_possible,
        "acc_at_1": float("nan"),
        "mean_rank": float("nan"),
        "per_needle": per_needle,
    }


# ============================================================
# LR schedule
# ============================================================
def get_lr(step, max_steps=5000, warmup=500, base=1e-5, final=1e-6):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return final + (base - final) * cos


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=str, default="results/v18_cross_attn/best.pt")
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
    ap.add_argument("--cat-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="results/v18_perhead")
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

    # Load pre-computed source hidden states (layer 4) for all store entries
    print(f"Loading per-head cache from {PERHEAD_CACHE_PATH}")
    cache = torch.load(PERHEAD_CACHE_PATH, weights_only=False)
    perhead_hiddens = cache["hiddens"].to(device)          # (N, S, D) fp16
    perhead_lengths = cache["lengths"].to(device)          # (N,) long
    print(f"  hiddens={tuple(perhead_hiddens.shape)} {perhead_hiddens.dtype} "
          f"mean_len={perhead_lengths.float().mean().item():.0f}")

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
        print("Tokenizing WT-103...")
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

    print("Building expanded-benchmark distractor pool...")
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
    print(f"Backbone:            {sum(p.numel() for p in backbone_params):,} (FROZEN)")

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
            "architecture": "perhead_attention_pooled",
        }, f, indent=2)

    # ----- Pre-training eval (architecture disruption check) -----
    print("\n=== Pre-training eval (per-head architecture, cleaned-token Config B) ===")
    pre = eval_niah_perhead(model, tokenizer, device, eval_distractors, eval_needles)
    pre_gates = get_gate_values(model)
    print(f"  NIAH: cleaned_recall={pre['recall_pooled']:.3f} "
          f"({pre['total_hits']}/{pre['total_possible']})")
    print(f"  gates: {pre_gates}")

    # Pre-training MAUVE
    clear_perhead_state(model)
    pre_mauve = eval_mauve_500(model, tokenizer, test_tokens, device, n_samples=args.mauve_n)
    print(f"  MAUVE-500 (no perhead injection): {pre_mauve:.4f}")

    history = [{
        "step": 0, "phase": "pre",
        "niah_cleaned_recall": pre["recall_pooled"],
        "niah_hits": pre["total_hits"],
        "niah_total": pre["total_possible"],
        "mauve_500_n200": pre_mauve,
        "gates": pre_gates,
        "lr_ca": 0.0, "lr_gate": 0.0,
        "ce_loss": None, "cat_loss": None,
    }]

    if pre_mauve < 0.85:
        print(f"  *** WARNING: pre-training MAUVE {pre_mauve:.4f} < 0.85 — "
              f"architecture disruption may be too large ***")

    print(f"\n=== Starting perhead training: {args.max_steps} steps, backbone FROZEN, "
          f"gate LR = {args.gate_lr_mult}x base ===\n")

    model.train()
    t0 = time.time()
    step = 0
    cond_counts = {"retrieve": 0, "baseline": 0, "wrong": 0}
    accum = {"ce": 0.0, "cat": 0.0, "n": 0}

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

        buf, conditions, top_idxs, src_h, src_len, active = build_batch_buffer_and_src(
            model, x, store_keys_gpu, perhead_hiddens, perhead_lengths,
            baseline_buffer, EXTRACT_LAYER, device,
            p_retrieve=args.p_retrieve, p_baseline=args.p_baseline,
            p_wrong=args.p_wrong, query_len=64, rng=rng,
        )
        for c in conditions:
            cond_counts[c] += 1

        set_perhead_state(model, src_h, src_len, active)
        orig_buf, orig_init = swap_buffer(model, buf)
        try:
            output = model(x, step=step)
            logits = output.logits
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss = ce_loss
            cat_loss_val = 0.0
            if output.categorization_logits is not None and cat_ids is not None:
                cat_loss = F.cross_entropy(output.categorization_logits, cat_ids)
                total_loss = total_loss + args.cat_weight * cat_loss
                cat_loss_val = float(cat_loss.item())
        finally:
            restore_buffer(model, orig_buf, orig_init)
            clear_perhead_state(model)

        if not torch.isfinite(total_loss):
            print(f"!!! NaN/inf at step {step} — stopping")
            break
        total_loss.backward()

        if step == 100:
            print("=== Grad-flow snapshot at step 100 ===")
            for name, p in model.named_parameters():
                if p.grad is not None and "cross_attn.gate" in name:
                    print(f"  {name}: grad_norm={p.grad.norm().item():.3e}")
            ca_gn2 = sum(p.grad.norm().item() ** 2 for p in ca_other_params if p.grad is not None)
            gt_gn2 = sum(p.grad.norm().item() ** 2 for p in gate_params if p.grad is not None)
            print(f"  -- gate grad L2:       {math.sqrt(gt_gn2):.3e}")
            print(f"  -- cross-attn grad L2: {math.sqrt(ca_gn2):.3e}")

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        step += 1
        accum["ce"] += float(ce_loss.item())
        accum["cat"] += cat_loss_val
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
                  f"lr_ca={lr_ca:.2e} lr_g={lr_gate:.2e} "
                  f"gates: {gate_str} cond({mix}) {rate:.2f}it/s elapsed={el:.0f}s")
            accum = {"ce": 0.0, "cat": 0.0, "n": 0}

        if step % args.eval_interval == 0 or step == args.max_steps:
            print(f"\n--- Eval at step {step} ---")
            model.eval()
            clear_perhead_state(model)
            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            niah = eval_niah_perhead(model, tokenizer, device, eval_distractors, eval_needles)
            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True
            clear_perhead_state(model)

            mauve_score = eval_mauve_500(
                model, tokenizer, test_tokens, device,
                n_samples=args.mauve_n,
            )
            gates = get_gate_values(model)

            model._buffers["engram_buffer"] = baseline_buffer.clone().contiguous()
            model._engram_buffer_initialized = True

            entry = {
                "step": step, "phase": "perhead",
                "niah_cleaned_recall": niah["recall_pooled"],
                "niah_hits": niah["total_hits"],
                "niah_total": niah["total_possible"],
                "mauve_500_n200": mauve_score,
                "gates": gates,
                "lr_ca": lr_ca, "lr_gate": lr_gate,
                "ce_loss": float(ce_loss.item()),
                "cat_loss": cat_loss_val,
                "elapsed_s": time.time() - t0,
            }
            history.append(entry)
            print(f"  NIAH cleaned: recall={niah['recall_pooled']:.3f} "
                  f"({niah['total_hits']}/{niah['total_possible']})")
            print(f"  MAUVE-500 (n={args.mauve_n}): {mauve_score:.4f}")
            print(f"  Gates: {gates}")
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
