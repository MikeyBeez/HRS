"""Phase 72c — Base learns to use a frozen pre-trained adapter.

Conceptually different from Phase 72 / 72b. Those phases co-trained adapter
and base on OOD content; both learned in parallel and fought each other. This
phase has only one learning process: the base learns to use a fixed thing.
The frozen adapter never updates.

Two-stage protocol:

  Stage 1 (one-time setup; cached at models/phase72c_frozen_adapter.pt):
    - Train a rank-8 LoRA on a single passkey passage using the Phase 63
      softmax base. 150 steps, frozen base, optimizer only on LoRA.
      Rank 8 + 150 steps deliberately produces partial-retrieval performance
      (the adapter holds *some* signal about the passkey but can't reliably
      generate it). The base will be trained to extract more signal from this
      fixed adapter.

  Stage 2 (this phase, 200-step base training):
    - Load pristine softmax base + load frozen adapter weights.
    - Freeze all LoRA params (lora_A and lora_B set requires_grad=False).
    - Mixed-batch training: 1 OOD batch per 4 WikiText batches.
    - OOD batch joint objective:
        L_joint = L_attached + mean over passage positions of
                  relu(baseline_ce[i] - L_detached[i])
      One-sided hinge per position against the cached pristine baseline,
      same as the Phase 72b design but with the adapter held constant.
    - WikiText batch: standard CE, no adapter.
    - Base lr 1e-5 (matches Phase 72b's protective slow rate).

Three sub-claims at every checkpoint (every 10 steps):
  1. Adapter retrieval improves: greedy decode generates the passkey, and/or
     attached passkey CE drops below the pristine-adapter-attached baseline.
  2. Detached CE stays within 1 nat of the pristine-base baseline at every
     position (the base hasn't absorbed).
  3. WikiText PPL within 5% of pristine baseline.

Reuses Phase 63 base + WikiText-2 data. Frozen adapter cached for
reproducibility (committed to git as it's small, ~270K params).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase72c_frozen_adapter.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.phase63_softmax_baseline import StandardTransformer
from experiments.identity_ae.lora_wrapper import (
    LoRALayer, apply_lora, get_lora_state_dict, load_lora_state_dict,
)


SEED = 0
N_STEPS = 200
WIKITEXT_PER_OOD = 4
LR_BASE = 1e-5
# Pretrain config calibrated to give partial-retrieval baseline (the spec's intent).
# rank=8 + 150 steps overfits to perfect retrieval (pk_ce 0); we want headroom.
# rank=8 + 30 steps + lr=3e-4: pk_ce ~1.8, generation "4716161616..." (first digit
# correct then loops) — adapter holds *some* signal but doesn't reliably retrieve.
LR_LORA_PRETRAIN = 3e-4
PRETRAIN_STEPS = 30
RANK = 8
ALPHA = 16
MAX_GRAD_NORM = 1.0
WT_BATCH_SIZE = 4
WT_SEQ_LEN = 256
LOG_INTERVAL = 10
CKPT_INTERVAL = 10
GEN_TOKENS = 30
PHASE72C_DIR = Path("results/identity_ae/phase72c")
CKPT_DIR = PHASE72C_DIR / "checkpoints"
ADAPTER_PATH = Path("models/phase72c_frozen_adapter.pt")

OOD_PASSAGE = "The access code for the QZ9K7M facility is 47281639."
OOD_PROMPT = "The access code for the QZ9K7M facility is"
OOD_EXPECTED_PASSKEY = "47281639"

# LoRA target modules — same as Phase 72 (blocks 4-5 of softmax baseline)
LORA_TARGETS = [
    'blocks.4.attn.qkv', 'blocks.4.attn.out_proj',
    'blocks.4.mlp.fc1',  'blocks.4.mlp.fc2',
    'blocks.5.attn.qkv', 'blocks.5.attn.out_proj',
    'blocks.5.mlp.fc1',  'blocks.5.mlp.fc2',
]


# ============================================================
# LoRA active-flag patch (same as Phase 72)
# ============================================================

_orig_lora_forward = LoRALayer.forward


def _patched_lora_forward(self, x):
    base_out = self.base_layer(x)
    if not getattr(self, "active", True):
        return base_out
    lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
    return base_out + lora_out


def patch_lora_class():
    LoRALayer.forward = _patched_lora_forward


def set_lora_active(model, active: bool):
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.active = active


# ============================================================
# Helpers
# ============================================================

def load_pristine_base(device):
    ckpt = torch.load("results/identity_ae/phase63/best.pt",
                       map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StandardTransformer(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                                 cfg["n_layers"], cfg["d_ff"], cfg["max_seq_len"],
                                 cfg["dropout"], cfg["bias"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def per_token_ce(model, ids_t):
    out = model(ids_t[:, :-1])
    targets = ids_t[:, 1:]
    log_probs = F.log_softmax(out, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.squeeze(0).tolist()


def passkey_positions(tokenizer, full_text, passkey_text):
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    for start in range(len(full_ids)):
        for end in range(start + 1, len(full_ids) + 1):
            if tokenizer.decode(full_ids[start:end]).strip() == passkey_text:
                return [i for i in range(start - 1, end - 1)]
    return []


def greedy_generate(model, prompt, tokenizer, device, n_tokens=GEN_TOKENS):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(n_tokens):
        idx = input_ids[:, -512:]
        out = model(idx)
        next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
    return tokenizer.decode(input_ids[0, len(ids):], skip_special_tokens=True)


@torch.no_grad()
def measure_wikitext_ppl(model, batches, device):
    total_nll, total_tok = 0.0, 0
    for ids in batches:
        ids_t = ids.to(device)
        out = model(ids_t[:, :-1])
        nll = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                ids_t[:, 1:].reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_tok += ids_t[:, 1:].numel()
    return math.exp(total_nll / max(total_tok, 1))


def build_wikitext_loader(tokenizer, seq_len=WT_SEQ_LEN, batch_size=WT_BATCH_SIZE, seed=SEED):
    from datasets import load_dataset
    print(f"  loading WikiText-2")
    train_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_raw   = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize_split(split):
        ids = []
        for t in split["text"]:
            if not t.strip(): continue
            ids.extend(tokenizer.encode(t, add_special_tokens=False))
        return ids

    train_ids = tokenize_split(train_raw)
    val_ids   = tokenize_split(val_raw)

    def chunk(ids, n=seq_len):
        out = []
        for i in range(0, len(ids) - n, n):
            out.append(torch.tensor(ids[i:i + n], dtype=torch.long))
        return out

    train_chunks = chunk(train_ids)
    val_chunks   = chunk(val_ids)
    rng = random.Random(seed)
    rng.shuffle(train_chunks)

    train_batches = []
    for i in range(0, len(train_chunks) - batch_size, batch_size):
        train_batches.append(torch.stack(train_chunks[i:i + batch_size]))
    val_batches = []
    for i in range(0, min(len(val_chunks), 16 * batch_size) - batch_size, batch_size):
        val_batches.append(torch.stack(val_chunks[i:i + batch_size]))
    print(f"    train batches: {len(train_batches)}, val batches: {len(val_batches)}")
    return train_batches, val_batches


# ============================================================
# Stage 1: Pre-train the frozen adapter
# ============================================================

def pretrain_frozen_adapter(device, tokenizer):
    """Train a rank-8 LoRA on the passkey passage; save and return the LoRA
    state dict + baseline metrics. If cached, just load."""
    if ADAPTER_PATH.exists():
        print(f"[A1] loading cached frozen adapter from {ADAPTER_PATH}")
        blob = torch.load(ADAPTER_PATH, map_location=device, weights_only=False)
        return blob

    ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[A1] no cached adapter — pretraining rank-{RANK} LoRA, "
          f"{PRETRAIN_STEPS} steps")
    base, cfg = load_pristine_base(device)
    base.train()
    patch_lora_class()
    n_lora = apply_lora(base, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(base, True)

    # Freeze base, only train LoRA
    for n, p in base.named_parameters():
        p.requires_grad_("lora_" in n)

    lora_params = [p for n, p in base.named_parameters() if "lora_" in n]
    optim = torch.optim.AdamW(lora_params, lr=LR_LORA_PRETRAIN, weight_decay=0.0)

    ids = tokenizer.encode(OOD_PASSAGE, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for step in range(PRETRAIN_STEPS):
        out = base(ids_t[:, :-1])
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optim.zero_grad(); loss.backward(); optim.step()
        if (step + 1) % 50 == 0:
            print(f"    pretrain step {step+1}/{PRETRAIN_STEPS}  loss {loss.item():.3f}")

    base.eval()
    # Measure attached baseline retrieval
    set_lora_active(base, True)
    pk_pos = passkey_positions(tokenizer, OOD_PASSAGE, OOD_EXPECTED_PASSKEY)
    with torch.no_grad():
        ce = per_token_ce(base, ids_t)
    pristine_attached_passkey_ce = sum(ce[i] for i in pk_pos) / len(pk_pos)
    gen = greedy_generate(base, OOD_PROMPT, tokenizer, device)
    pristine_attached_retrieval = OOD_EXPECTED_PASSKEY in gen
    print(f"    pristine-adapter-attached passkey CE: {pristine_attached_passkey_ce:.3f}")
    print(f"    pristine-adapter-attached generation: {gen!r}")
    print(f"    pristine-adapter-attached retrieval pass: {pristine_attached_retrieval}")

    lora_state = get_lora_state_dict(base)
    blob = {
        "lora_state_dict": {k: v.cpu() for k, v in lora_state.items()},
        "rank": RANK, "alpha": ALPHA, "targets": LORA_TARGETS,
        "passage": OOD_PASSAGE, "prompt": OOD_PROMPT, "passkey": OOD_EXPECTED_PASSKEY,
        "pretrain_steps": PRETRAIN_STEPS, "pretrain_lr": LR_LORA_PRETRAIN,
        "baseline_attached_passkey_ce": pristine_attached_passkey_ce,
        "baseline_attached_generation": gen,
        "baseline_attached_retrieval_pass": pristine_attached_retrieval,
    }
    torch.save(blob, ADAPTER_PATH)
    print(f"    saved frozen adapter -> {ADAPTER_PATH}")
    return blob


# ============================================================
# Stage 2: Train the base to use the frozen adapter
# ============================================================

def save_checkpoint(step, model, optim, att_ce, det_mean_ce, ppl, retrieval_pass,
                     model_cfg):
    path = CKPT_DIR / f"step_{step:04d}.pt"
    base_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" not in k}
    torch.save({
        "step": step,
        "attached_passkey_ce": att_ce,
        "detached_mean_ce": det_mean_ce,
        "wikitext_ppl": ppl,
        "attached_retrieval_pass": retrieval_pass,
        "base_state_dict": base_state,
        "optim_state_dict": optim.state_dict() if optim is not None else None,
        "model_config": model_cfg,
    }, path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE72C_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Stage 1: Pre-train (or load) frozen adapter ----
    adapter_blob = pretrain_frozen_adapter(device, tokenizer)

    # ---- Stage 2: Load fresh pristine base, attach frozen adapter, freeze it ----
    print(f"\n[A2] loading pristine base + attaching frozen adapter")
    model, cfg = load_pristine_base(device)
    model.eval()
    patch_lora_class()
    n_lora = apply_lora(model, rank=adapter_blob["rank"], alpha=adapter_blob["alpha"],
                          target_modules=adapter_blob["targets"])
    load_lora_state_dict(model, {k: v.to(device) for k, v in adapter_blob["lora_state_dict"].items()})
    set_lora_active(model, True)
    # Freeze LoRA — only base trainable
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" not in n)
    base_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"    base trainable: {sum(p.numel() for p in base_params):,}  "
          f"lora frozen: {n_lora:,}")

    # ---- Tokenize OOD; locate passkey positions ----
    ood_ids = tokenizer.encode(OOD_PASSAGE, add_special_tokens=False)
    ood_t = torch.tensor(ood_ids, dtype=torch.long).unsqueeze(0).to(device)
    pk_pos = passkey_positions(tokenizer, OOD_PASSAGE, OOD_EXPECTED_PASSKEY)
    print(f"    OOD: {len(ood_ids)} tokens, passkey at CE positions {pk_pos}")

    # ---- Pristine baselines ----
    set_lora_active(model, False)
    with torch.no_grad():
        baseline_ce_per_pos = per_token_ce(model, ood_t)
    pristine_passkey_ce = sum(baseline_ce_per_pos[i] for i in pk_pos) / len(pk_pos)
    pristine_mean_ce = sum(baseline_ce_per_pos) / len(baseline_ce_per_pos)
    set_lora_active(model, True)
    print(f"    pristine BASE-only mean CE on passage: {pristine_mean_ce:.3f}")
    print(f"    pristine BASE-only passkey CE: {pristine_passkey_ce:.3f}")

    # Adapter-attached pristine retrieval (with the frozen adapter loaded but
    # base unchanged) — this is the headline starting point we want to lift
    set_lora_active(model, True)
    with torch.no_grad():
        ce_att_pristine = per_token_ce(model, ood_t)
    pristine_attached_passkey_ce = sum(ce_att_pristine[i] for i in pk_pos) / len(pk_pos)
    pristine_attached_gen = greedy_generate(model, OOD_PROMPT, tokenizer, device)
    pristine_attached_retrieval = OOD_EXPECTED_PASSKEY in pristine_attached_gen
    print(f"    pristine BASE+ADAPTER passkey CE: {pristine_attached_passkey_ce:.3f}")
    print(f"    pristine BASE+ADAPTER generation: {pristine_attached_gen!r}")
    print(f"    pristine BASE+ADAPTER retrieval: {'PASS' if pristine_attached_retrieval else 'FAIL'}")

    # ---- WikiText baseline ----
    train_batches, val_batches = build_wikitext_loader(tokenizer)
    set_lora_active(model, False)  # measure base-only WikiText (adapter is for OOD only)
    pristine_ppl = measure_wikitext_ppl(model, val_batches, device)
    set_lora_active(model, True)
    print(f"    pristine WikiText-2 val PPL: {pristine_ppl:.3f}")

    # Anchor for the regularizer
    baseline_ce_t = torch.tensor(baseline_ce_per_pos, device=device)

    # ---- Optimizer (only base parameters) ----
    optim = torch.optim.AdamW([{"params": base_params, "lr": LR_BASE}],
                                weight_decay=0.0)

    # ---- Step 0 checkpoint ----
    save_checkpoint(0, model, optim, pristine_attached_passkey_ce,
                     pristine_mean_ce, pristine_ppl,
                     pristine_attached_retrieval, model_cfg=cfg)
    checkpoint_records = [(0, pristine_attached_passkey_ce,
                            pristine_mean_ce, pristine_ppl,
                            pristine_attached_retrieval)]

    # ---- Training loop ----
    print(f"\n[B] training {N_STEPS} steps  (base only, adapter frozen)  "
          f"base lr {LR_BASE}  {WIKITEXT_PER_OOD}:1 wt:ood")
    trajectory = {
        "step": [], "is_ood_step": [],
        "L_attached": [], "L_detached": [], "L_reg": [], "L_joint": [],
        "attached_passkey_ce": [], "detached_mean_ce": [], "wikitext_ppl": [],
        "retrieval_pass": [],
    }
    composite_passes = []
    wt_iter_idx = 0
    t0 = time.time()
    n_ood = 0

    for step in range(N_STEPS):
        is_ood_step = (step % (WIKITEXT_PER_OOD + 1) == WIKITEXT_PER_OOD)
        model.train()
        optim.zero_grad()

        if is_ood_step:
            set_lora_active(model, True)
            out_att = model(ood_t[:, :-1])
            log_p_att = F.log_softmax(out_att, dim=-1)
            nll_att = -log_p_att.gather(-1, ood_t[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
            L_attached = nll_att.mean()

            set_lora_active(model, False)
            out_det = model(ood_t[:, :-1])
            log_p_det = F.log_softmax(out_det, dim=-1)
            nll_det = -log_p_det.gather(-1, ood_t[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
            set_lora_active(model, True)

            hinge = F.relu(baseline_ce_t - nll_det)
            L_reg = hinge.mean()
            L_joint = L_attached + L_reg

            L_joint.backward()
            torch.nn.utils.clip_grad_norm_(base_params, MAX_GRAD_NORM)
            optim.step()
            n_ood += 1
        else:
            set_lora_active(model, False)
            wt_batch = train_batches[wt_iter_idx % len(train_batches)].to(device)
            wt_iter_idx += 1
            out = model(wt_batch[:, :-1])
            L = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                  wt_batch[:, 1:].reshape(-1))
            set_lora_active(model, True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(base_params, MAX_GRAD_NORM)
            optim.step()

        # ---- Monitoring + checkpointing ----
        ckpt_step = (step + 1) if (step + 1) % CKPT_INTERVAL == 0 else None
        if ckpt_step is not None or step == 0:
            model.eval()
            with torch.no_grad():
                set_lora_active(model, True)
                ce_att = per_token_ce(model, ood_t)
                gen_att = greedy_generate(model, OOD_PROMPT, tokenizer, device)
                set_lora_active(model, False)
                ce_det = per_token_ce(model, ood_t)
                ppl = measure_wikitext_ppl(model, val_batches[:8], device)
                set_lora_active(model, True)
            att_pk = sum(ce_att[i] for i in pk_pos) / len(pk_pos)
            det_mean = sum(ce_det) / len(ce_det)
            retrieval_pass = OOD_EXPECTED_PASSKEY in gen_att

            trajectory["step"].append(step)
            trajectory["is_ood_step"].append(is_ood_step)
            trajectory["attached_passkey_ce"].append(att_pk)
            trajectory["detached_mean_ce"].append(det_mean)
            trajectory["wikitext_ppl"].append(ppl)
            trajectory["retrieval_pass"].append(retrieval_pass)
            if is_ood_step:
                trajectory["L_attached"].append(float(L_attached.item()))
                trajectory["L_detached"].append(float(nll_det.mean().item()))
                trajectory["L_reg"].append(float(L_reg.item()))
                trajectory["L_joint"].append(float(L_joint.item()))
            else:
                for k in ("L_attached", "L_detached", "L_reg", "L_joint"):
                    trajectory[k].append(None)

            # Composite check
            adapter_improved = (retrieval_pass and not pristine_attached_retrieval) \
                                or (att_pk < pristine_attached_passkey_ce - 0.5)
            det_within_1nat = abs(det_mean - pristine_mean_ce) <= 1.0
            ppl_within_5pct = abs(ppl - pristine_ppl) / pristine_ppl <= 0.05
            composite = adapter_improved and det_within_1nat and ppl_within_5pct

            if ckpt_step is not None:
                save_checkpoint(ckpt_step, model, optim, att_pk, det_mean, ppl,
                                  retrieval_pass, model_cfg=cfg)
                checkpoint_records.append((ckpt_step, att_pk, det_mean, ppl, retrieval_pass))
                if composite:
                    composite_passes.append(ckpt_step)

            ckpt_marker = " [ckpt]" if ckpt_step is not None else ""
            comp_marker = " ★COMPOSITE" if composite else ""
            ret_marker  = " ★RET"      if retrieval_pass and not pristine_attached_retrieval else ""
            dt = time.time() - t0
            eta = dt / (step + 1) * (N_STEPS - step - 1)
            print(f"    step {step+1:>3d}/{N_STEPS}  att_pk {att_pk:>6.2f}  "
                  f"det_mean {det_mean:>5.2f}  ppl {ppl:>6.2f}  "
                  f"ret={'P' if retrieval_pass else 'F'}  "
                  f"[ood {n_ood}, {dt:.0f}s, eta {eta:.0f}s]{ckpt_marker}{ret_marker}{comp_marker}")

    # ---- Final eval ----
    print(f"\n[C] final eval")
    model.eval()
    with torch.no_grad():
        set_lora_active(model, True)
        final_ce_att = per_token_ce(model, ood_t)
        final_gen_att = greedy_generate(model, OOD_PROMPT, tokenizer, device)
        set_lora_active(model, False)
        final_ce_det = per_token_ce(model, ood_t)
        final_gen_det = greedy_generate(model, OOD_PROMPT, tokenizer, device)
        final_ppl = measure_wikitext_ppl(model, val_batches, device)
        set_lora_active(model, True)
    final_att_pk = sum(final_ce_att[i] for i in pk_pos) / len(pk_pos)
    final_det_mean = sum(final_ce_det) / len(final_ce_det)
    final_retrieval = OOD_EXPECTED_PASSKEY in final_gen_att
    final_adapter_improved = (final_retrieval and not pristine_attached_retrieval) \
                              or (final_att_pk < pristine_attached_passkey_ce - 0.5)
    final_det_within = abs(final_det_mean - pristine_mean_ce) <= 1.0
    final_ppl_within = abs(final_ppl - pristine_ppl) / pristine_ppl <= 0.05
    final_composite = final_adapter_improved and final_det_within and final_ppl_within

    # Per-position detached check
    per_pos_within_1nat = all(abs(c - b) <= 1.0
                                for c, b in zip(final_ce_det, baseline_ce_per_pos))

    print(f"    pristine BASE+ADAPTER retrieval : {'PASS' if pristine_attached_retrieval else 'FAIL'} "
          f"(passkey CE {pristine_attached_passkey_ce:.3f})")
    print(f"    final    BASE+ADAPTER retrieval : {'PASS' if final_retrieval else 'FAIL'} "
          f"(passkey CE {final_att_pk:.3f})")
    print(f"    final BASE-only mean CE on OOD   : {final_det_mean:.3f}  (pristine {pristine_mean_ce:.3f}, within-1nat={'PASS' if final_det_within else 'FAIL'})")
    print(f"    per-position within 1 nat (strict): {'PASS' if per_pos_within_1nat else 'FAIL'}")
    print(f"    final WikiText PPL                : {final_ppl:.3f}  (pristine {pristine_ppl:.3f}, within-5pct={'PASS' if final_ppl_within else 'FAIL'})")
    print(f"    final composite : {'PASS' if final_composite else 'FAIL'}")
    if composite_passes:
        print(f"    composite PASSED at checkpoints: {composite_passes}")
        # Find longest consecutive run
        longest = 1
        cur = 1
        for i in range(1, len(composite_passes)):
            if composite_passes[i] == composite_passes[i-1] + CKPT_INTERVAL:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 1
        print(f"    longest consecutive composite run: {longest}")
    else:
        print(f"    composite never passed at any checkpoint")

    # ---- Save JSON ----
    out_path = PHASE72C_DIR / "frozen_adapter.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_steps": N_STEPS, "wikitext_per_ood": WIKITEXT_PER_OOD,
                "lr_base": LR_BASE, "lora_rank": RANK, "lora_alpha": ALPHA,
                "lora_targets": LORA_TARGETS,
                "wt_batch_size": WT_BATCH_SIZE, "wt_seq_len": WT_SEQ_LEN,
                "regularizer": "one_sided_hinge_per_position",
                "ood_passage": OOD_PASSAGE, "ood_prompt": OOD_PROMPT,
                "ood_expected_passkey": OOD_EXPECTED_PASSKEY,
                "passkey_positions": pk_pos,
                "seed": SEED,
            },
            "pristine_baselines": {
                "base_only_per_pos_ce": baseline_ce_per_pos,
                "base_only_passkey_ce": pristine_passkey_ce,
                "base_only_mean_ce": pristine_mean_ce,
                "base_plus_adapter_passkey_ce": pristine_attached_passkey_ce,
                "base_plus_adapter_generation": pristine_attached_gen,
                "base_plus_adapter_retrieval_pass": pristine_attached_retrieval,
                "wikitext_val_ppl": pristine_ppl,
            },
            "final": {
                "attached_passkey_ce": final_att_pk,
                "attached_generation": final_gen_att,
                "attached_retrieval_pass": final_retrieval,
                "detached_mean_ce": final_det_mean,
                "detached_generation": final_gen_det,
                "wikitext_val_ppl": final_ppl,
                "adapter_improved": final_adapter_improved,
                "detached_within_1nat": final_det_within,
                "per_position_within_1nat_strict": per_pos_within_1nat,
                "wikitext_within_5pct": final_ppl_within,
                "composite": final_composite,
            },
            "trajectory": trajectory,
            "composite_passes": composite_passes,
            "predictions": {
                "adapter_attached_improves":   {"P": 0.60, "outcome": final_adapter_improved},
                "detached_within_1nat":        {"P": 0.65, "outcome": final_det_within},
                "wikitext_within_5pct":        {"P": 0.60, "outcome": final_ppl_within},
                "composite_at_some_checkpoint":{"P": 0.30, "outcome": len(composite_passes) > 0},
            },
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    steps = trajectory["step"]
    axes[0].plot(steps, trajectory["attached_passkey_ce"], color="C2", marker="o", markersize=3, label="attached passkey CE")
    axes[0].axhline(pristine_attached_passkey_ce, color="C2", ls="--", lw=1,
                     label=f"pristine attached ({pristine_attached_passkey_ce:.2f})")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("CE on passkey tokens (nats)")
    axes[0].set_title("Adapter-attached passkey CE (lower = better retrieval)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, trajectory["detached_mean_ce"], color="C3", marker="s", markersize=3, label="detached mean CE")
    axes[1].axhline(pristine_mean_ce, color="C0", ls="--", lw=1, label=f"pristine ({pristine_mean_ce:.2f})")
    axes[1].axhline(pristine_mean_ce - 1.0, color="C0", ls=":", lw=0.6)
    axes[1].axhline(pristine_mean_ce + 1.0, color="C0", ls=":", lw=0.6, label="±1 nat band")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("mean CE on OOD passage (nats)")
    axes[1].set_title("Detached CE (base alone, no adapter)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].plot(steps, trajectory["wikitext_ppl"], color="C0", marker="o", markersize=3)
    axes[2].axhline(pristine_ppl, color="black", ls="--", lw=1, label=f"pristine ({pristine_ppl:.2f})")
    axes[2].axhline(pristine_ppl * 1.05, color="gray", ls=":", lw=0.6, label="±5%")
    axes[2].axhline(pristine_ppl * 0.95, color="gray", ls=":", lw=0.6)
    axes[2].set_xlabel("step"); axes[2].set_ylabel("WikiText PPL")
    axes[2].set_title("General competence")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = PHASE72C_DIR / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
