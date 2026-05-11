"""Phase 72 — Adapter-aware base training, single-adapter preliminary.

Tests whether the pristine softmax base from Phase 63 can be co-trained with a
fresh LoRA adapter on an OOD example such that:

  (1) the adapter learns the OOD content (attached retrieval succeeds),
  (2) the base does NOT absorb the OOD content (detached CE on passkey
      tokens stays elevated, within 1 nat of pristine baseline),
  (3) general competence is preserved (WikiText PPL within 5% of baseline).

Joint objective on OOD batches:
    L_joint = L_attached - tanh(L_detached / scale)

  L_attached / L_detached are next-token CE losses on the OOD passage with
  LoRA active / inactive respectively. The tanh saturates the
  anti-absorption pressure so the loss stays well-behaved.

Training mix: 1 OOD batch per 4 WikiText batches, 750 total steps.

OOD example: "The access code for the QZ9K7M facility is 47281639."
  Validated pre-training: passkey-token mean CE under pristine base is
  ~7.66 nats (well above the 3-nat OOD threshold).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase72_adapter_aware_base.py
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
    LoRALayer, apply_lora, reset_lora,
)


SEED = 0
GEN_TOKENS = 30
N_STEPS = 750
WIKITEXT_PER_OOD = 4               # 4 WT batches per 1 OOD batch
LR_BASE = 1e-4
LR_LORA = 1e-3
MAX_GRAD_NORM = 1.0
RANK = 128
ALPHA = 256
TANH_SCALE = 4.0
WT_BATCH_SIZE = 4
WT_SEQ_LEN = 256
LOG_INTERVAL = 25                  # also the checkpoint interval
CKPT_INTERVAL = 25                 # save model+adapter+optim state every N steps
PHASE72_DIR = Path("results/identity_ae/phase72")
CKPT_DIR = PHASE72_DIR / "checkpoints"

OOD_PASSAGE = "The access code for the QZ9K7M facility is 47281639."
OOD_PROMPT = "The access code for the QZ9K7M facility is"
OOD_EXPECTED_PASSKEY = "47281639"

L45_QKV_TARGETS = [
    'blocks.4.attn.qkv', 'blocks.4.attn.out_proj',
    'blocks.4.mlp.fc1',  'blocks.4.mlp.fc2',
    'blocks.5.attn.qkv', 'blocks.5.attn.out_proj',
    'blocks.5.mlp.fc1',  'blocks.5.mlp.fc2',
]


# ============================================================
# LoRA-active patching (lets us run "detached" forward passes)
# ============================================================

_original_lora_forward = LoRALayer.forward


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

def load_pristine_model(device):
    ckpt = torch.load("results/identity_ae/phase63/best.pt",
                       map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StandardTransformer(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                                 cfg["n_layers"], cfg["d_ff"], cfg["max_seq_len"],
                                 cfg["dropout"], cfg["bias"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def per_token_ce(model, ids_t):
    """Return (token_id_list, per_position_ce_list) for next-token prediction."""
    out = model(ids_t[:, :-1])
    targets = ids_t[:, 1:]
    log_probs = F.log_softmax(out, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return targets.squeeze(0).tolist(), nll.squeeze(0).tolist()


def passkey_positions(tokenizer, full_text, passkey_text):
    """Return the indices in the next-token-prediction loss (offset by 1)
    that correspond to the passkey tokens."""
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    # The passkey appears at some byte/token range; find it by decoding spans.
    # Simpler: take all-tokens, then locate passkey substring by stepping.
    for start in range(len(full_ids)):
        for end in range(start + 1, len(full_ids) + 1):
            if tokenizer.decode(full_ids[start:end]).strip() == passkey_text:
                # next-token loss target indices: positions [start..end-1] in
                # the target sequence are at indices [start-1..end-2] in the
                # cropped ids[1:] view. But our per_token_ce returns lengths T-1
                # corresponding to predictions at positions 0..T-2 with targets
                # at 1..T-1. So passkey targets at original positions [start..end-1]
                # are at indices [start-1..end-2] in the returned ce list.
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
    """Measure mean NLL across a list of pre-tokenized WikiText batches."""
    total_nll, total_tok = 0.0, 0
    for ids in batches:
        ids_t = ids.to(device)
        out = model(ids_t[:, :-1])
        nll = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                ids_t[:, 1:].reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_tok += ids_t[:, 1:].numel()
    return math.exp(total_nll / max(total_tok, 1))


def build_wikitext_loader(tokenizer, seq_len=WT_SEQ_LEN, batch_size=WT_BATCH_SIZE,
                          seed=SEED):
    """Return (train_batches, val_batches): pre-tokenized wikitext-2 tensors."""
    from datasets import load_dataset
    print(f"  loading WikiText-2 (light, distribution-matched subset of -103)")
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
# Main
# ============================================================

def save_checkpoint(step, model, optim, pk_att, pk_det, ppl, model_cfg, lora_targets):
    """Save base + LoRA + optim state at a given step. Filename is zero-padded
    so directory listings sort chronologically."""
    path = CKPT_DIR / f"step_{step:04d}.pt"
    base_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" not in k}
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({
        "step": step,
        "passkey_attached_ce_mean": pk_att,
        "passkey_detached_ce_mean": pk_det,
        "wikitext_val_ppl": ppl,
        "base_state_dict": base_state,
        "lora_state_dict": lora_state,
        "optim_state_dict": optim.state_dict() if optim is not None else None,
        "model_config": model_cfg,
        "lora_rank": RANK,
        "lora_alpha": ALPHA,
        "lora_targets": lora_targets,
    }, path)
    return path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE72_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load pristine base ----
    print(f"[A] loading pristine softmax base from Phase 63")
    model, cfg = load_pristine_model(device)
    model.eval()

    # ---- Tokenize OOD example and locate passkey positions ----
    ood_ids = tokenizer.encode(OOD_PASSAGE, add_special_tokens=False)
    ood_t = torch.tensor(ood_ids, dtype=torch.long).unsqueeze(0).to(device)
    passkey_pos = passkey_positions(tokenizer, OOD_PASSAGE, OOD_EXPECTED_PASSKEY)
    print(f"    OOD passage: {OOD_PASSAGE!r}")
    print(f"    OOD token count: {len(ood_ids)}, "
          f"passkey positions in CE list: {passkey_pos}")

    # ---- Pre-training baselines ----
    with torch.no_grad():
        targets, ce_list = per_token_ce(model, ood_t)
    pristine_ood_ce = ce_list
    pristine_passkey_ce_mean = sum(ce_list[i] for i in passkey_pos) / len(passkey_pos)
    print(f"    pristine passkey-token mean CE: {pristine_passkey_ce_mean:.3f} nats "
          f"(threshold for OOD validity: >= 3.0)")
    assert pristine_passkey_ce_mean >= 3.0, "OOD example not OOD enough"

    train_batches, val_batches = build_wikitext_loader(tokenizer)
    pristine_ppl = measure_wikitext_ppl(model, val_batches, device)
    print(f"    pristine WikiText-2 val PPL: {pristine_ppl:.3f}")

    # ---- Apply LoRA, patch active flag ----
    print(f"\n[B] applying LoRA (rank {RANK}) to blocks 4-5, patching active flag")
    patch_lora_class()
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_QKV_TARGETS)
    set_lora_active(model, True)   # default active, switched off in detached pass
    print(f"    LoRA params: {n_lora:,}")

    # Unfreeze base parameters (apply_lora froze everything except lora_*)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad_(True)

    base_params = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    print(f"    base trainable: {sum(p.numel() for p in base_params):,}  "
          f"lora trainable: {sum(p.numel() for p in lora_params):,}")

    optim = torch.optim.AdamW(
        [{"params": base_params, "lr": LR_BASE},
         {"params": lora_params, "lr": LR_LORA}],
        weight_decay=0.0,
    )

    # ---- Step 0 checkpoint: pristine state with zeroed LoRA (adapter
    # contribution is zero so attached == detached == pristine). Save
    # before any training happens. ----
    p0 = save_checkpoint(0, model, optim,
                          pk_att=pristine_passkey_ce_mean,
                          pk_det=pristine_passkey_ce_mean,
                          ppl=pristine_ppl, model_cfg=cfg,
                          lora_targets=L45_QKV_TARGETS)
    print(f"    saved pristine checkpoint -> {p0.name}")
    checkpoint_records = [(0, pristine_passkey_ce_mean,
                            pristine_passkey_ce_mean, pristine_ppl)]

    # ---- Training loop ----
    print(f"\n[C] training {N_STEPS} steps, {WIKITEXT_PER_OOD}:1 wikitext:ood ratio")
    wt_iter_idx = 0

    trajectory = {
        "step": [], "is_ood_step": [],
        "L_attached": [], "L_detached": [],
        "passkey_attached_ce": [], "passkey_detached_ce": [],
        "wikitext_ppl": [], "joint_loss": [],
    }

    t0 = time.time()
    n_ood_steps = 0
    for step in range(N_STEPS):
        is_ood_step = (step % (WIKITEXT_PER_OOD + 1) == WIKITEXT_PER_OOD)
        model.train()
        optim.zero_grad()

        if is_ood_step:
            # Attached pass
            set_lora_active(model, True)
            out_attached = model(ood_t[:, :-1])
            L_attached = F.cross_entropy(out_attached.reshape(-1, out_attached.shape[-1]),
                                           ood_t[:, 1:].reshape(-1))
            # Detached pass
            set_lora_active(model, False)
            out_detached = model(ood_t[:, :-1])
            L_detached = F.cross_entropy(out_detached.reshape(-1, out_detached.shape[-1]),
                                           ood_t[:, 1:].reshape(-1))
            # Restore active for downstream
            set_lora_active(model, True)

            L_joint = L_attached - torch.tanh(L_detached / TANH_SCALE)
            L_joint.backward()
            torch.nn.utils.clip_grad_norm_(base_params + lora_params, MAX_GRAD_NORM)
            optim.step()
            n_ood_steps += 1
        else:
            # WikiText batch — adapter inactive, standard LM loss
            set_lora_active(model, False)
            wt_batch = train_batches[wt_iter_idx % len(train_batches)].to(device)
            wt_iter_idx += 1
            out = model(wt_batch[:, :-1])
            L = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                  wt_batch[:, 1:].reshape(-1))
            set_lora_active(model, True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(base_params + lora_params, MAX_GRAD_NORM)
            optim.step()

        # ---- Monitoring + checkpointing ----
        ckpt_step = (step + 1) if (step + 1) % CKPT_INTERVAL == 0 else None
        if ckpt_step is not None or step == 0:
            model.eval()
            with torch.no_grad():
                set_lora_active(model, True)
                _, ce_att = per_token_ce(model, ood_t)
                set_lora_active(model, False)
                _, ce_det = per_token_ce(model, ood_t)
                set_lora_active(model, True)
                pk_att = sum(ce_att[i] for i in passkey_pos) / len(passkey_pos)
                pk_det = sum(ce_det[i] for i in passkey_pos) / len(passkey_pos)
                # Lightweight PPL (sample of 8 val batches)
                ppl = measure_wikitext_ppl(model, val_batches[:8], device)
            trajectory["step"].append(step)
            trajectory["is_ood_step"].append(is_ood_step)
            trajectory["passkey_attached_ce"].append(pk_att)
            trajectory["passkey_detached_ce"].append(pk_det)
            trajectory["wikitext_ppl"].append(ppl)
            trajectory["joint_loss"].append(float(L_joint.item()) if is_ood_step else None)
            trajectory["L_attached"].append(float(L_attached.item()) if is_ood_step else None)
            trajectory["L_detached"].append(float(L_detached.item()) if is_ood_step else None)
            dt = time.time() - t0
            eta = dt / (step + 1) * (N_STEPS - step - 1)
            ckpt_marker = ""
            if ckpt_step is not None:
                save_checkpoint(ckpt_step, model, optim, pk_att, pk_det, ppl,
                                  model_cfg=cfg, lora_targets=L45_QKV_TARGETS)
                checkpoint_records.append((ckpt_step, pk_att, pk_det, ppl))
                ckpt_marker = " [ckpt]"
            print(f"    step {step+1:>3d}/{N_STEPS}  pk_att {pk_att:>5.2f}  "
                  f"pk_det {pk_det:>5.2f}  ppl {ppl:>6.2f}  "
                  f"[ood_steps {n_ood_steps}, {dt:.0f}s, eta {eta:.0f}s]{ckpt_marker}")

    # ---- Final eval ----
    print(f"\n[D] final eval")
    model.eval()
    set_lora_active(model, True)
    final_gen_attached = greedy_generate(model, OOD_PROMPT, tokenizer, device)
    attached_retrieval_pass = OOD_EXPECTED_PASSKEY in final_gen_attached
    set_lora_active(model, False)
    final_gen_detached = greedy_generate(model, OOD_PROMPT, tokenizer, device)
    detached_retrieval_pass = OOD_EXPECTED_PASSKEY in final_gen_detached
    set_lora_active(model, True)

    with torch.no_grad():
        set_lora_active(model, True)
        _, final_ce_att = per_token_ce(model, ood_t)
        set_lora_active(model, False)
        _, final_ce_det = per_token_ce(model, ood_t)
        set_lora_active(model, True)
        final_pk_att = sum(final_ce_att[i] for i in passkey_pos) / len(passkey_pos)
        final_pk_det = sum(final_ce_det[i] for i in passkey_pos) / len(passkey_pos)
        final_ppl = measure_wikitext_ppl(model, val_batches, device)

    pk_det_within_1nat = abs(final_pk_det - pristine_passkey_ce_mean) <= 1.0
    ppl_within_5pct = abs(final_ppl - pristine_ppl) / pristine_ppl <= 0.05

    print(f"    attached generation: {final_gen_attached!r}")
    print(f"    detached generation: {final_gen_detached!r}")
    print(f"    attached passkey CE: {final_pk_att:.3f}")
    print(f"    detached passkey CE: {final_pk_det:.3f}  "
          f"(pristine {pristine_passkey_ce_mean:.3f}, "
          f"within-1nat={'PASS' if pk_det_within_1nat else 'FAIL'})")
    print(f"    final WikiText PPL : {final_ppl:.3f}  "
          f"(pristine {pristine_ppl:.3f}, "
          f"within-5pct={'PASS' if ppl_within_5pct else 'FAIL'})")
    print(f"    attached retrieval : {'PASS' if attached_retrieval_pass else 'FAIL'}  "
          f"(passkey {'found' if attached_retrieval_pass else 'not found'} in attached gen)")

    composite = attached_retrieval_pass and pk_det_within_1nat and ppl_within_5pct
    print(f"\n    COMPOSITE SUCCESS: {'PASS' if composite else 'FAIL'}")

    # ---- Save JSON ----
    out_path = PHASE72_DIR / "adapter_aware_base.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_steps": N_STEPS, "wikitext_per_ood": WIKITEXT_PER_OOD,
                "lr_base": LR_BASE, "lr_lora": LR_LORA, "rank": RANK,
                "alpha": ALPHA, "tanh_scale": TANH_SCALE,
                "wt_batch_size": WT_BATCH_SIZE, "wt_seq_len": WT_SEQ_LEN,
                "ood_passage": OOD_PASSAGE, "ood_prompt": OOD_PROMPT,
                "ood_expected_passkey": OOD_EXPECTED_PASSKEY,
                "passkey_positions_in_ce_list": passkey_pos,
                "seed": SEED,
            },
            "pristine_baselines": {
                "ood_per_position_ce": pristine_ood_ce,
                "passkey_mean_ce": pristine_passkey_ce_mean,
                "wikitext_ppl": pristine_ppl,
            },
            "final": {
                "attached_passkey_ce_mean": final_pk_att,
                "detached_passkey_ce_mean": final_pk_det,
                "wikitext_ppl": final_ppl,
                "attached_generation": final_gen_attached,
                "detached_generation": final_gen_detached,
                "attached_retrieval_pass": attached_retrieval_pass,
                "detached_passkey_within_1nat": pk_det_within_1nat,
                "wikitext_ppl_within_5pct": ppl_within_5pct,
                "composite_success": composite,
            },
            "trajectory": trajectory,
            "predictions": {
                "attached_retrieval": {"P": 0.60, "outcome": attached_retrieval_pass},
                "detached_within_1nat": {"P": 0.55, "outcome": pk_det_within_1nat},
                "general_within_5pct": {"P": 0.65, "outcome": ppl_within_5pct},
                "composite": {"P": 0.30, "outcome": composite},
            },
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    steps = trajectory["step"]
    axes[0].plot(steps, trajectory["passkey_attached_ce"], color="C2",
                  marker="o", markersize=3, label="attached")
    axes[0].plot(steps, trajectory["passkey_detached_ce"], color="C3",
                  marker="s", markersize=3, label="detached")
    axes[0].axhline(pristine_passkey_ce_mean, color="C0", ls="--", lw=1,
                     label=f"pristine ({pristine_passkey_ce_mean:.2f})")
    axes[0].axhline(pristine_passkey_ce_mean - 1.0, color="C0", ls=":", lw=0.6)
    axes[0].axhline(pristine_passkey_ce_mean + 1.0, color="C0", ls=":", lw=0.6,
                     label="+/- 1 nat band")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("passkey-token mean CE (nats)")
    axes[0].set_title("Passkey CE trajectory"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, trajectory["wikitext_ppl"], color="C0", marker="o", markersize=3)
    axes[1].axhline(pristine_ppl, color="black", ls="--", lw=1,
                     label=f"pristine ({pristine_ppl:.2f})")
    axes[1].axhline(pristine_ppl * 1.05, color="gray", ls=":", lw=0.6, label="+/- 5%")
    axes[1].axhline(pristine_ppl * 0.95, color="gray", ls=":", lw=0.6)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("WikiText-2 val PPL")
    axes[1].set_title("General competence trajectory"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    joint_steps = [s for s, j in zip(steps, trajectory["joint_loss"]) if j is not None]
    joint_vals  = [j for j in trajectory["joint_loss"] if j is not None]
    L_att_vals  = [v for v in trajectory["L_attached"] if v is not None]
    L_det_vals  = [v for v in trajectory["L_detached"] if v is not None]
    axes[2].plot(joint_steps, joint_vals, color="C1", marker="o", markersize=3, label="L_joint")
    axes[2].plot(joint_steps, L_att_vals,  color="C2", marker="s", markersize=3, label="L_attached")
    axes[2].plot(joint_steps, L_det_vals,  color="C3", marker="^", markersize=3, label="L_detached")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("loss value")
    axes[2].set_title("OOD-batch loss components"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = PHASE72_DIR / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Checkpoint records JSON (small, gets committed alongside README) ----
    ckpt_records_path = CKPT_DIR / "records.json"
    with open(ckpt_records_path, "w") as f:
        json.dump({
            "ckpt_interval": CKPT_INTERVAL,
            "n_checkpoints": len(checkpoint_records),
            "pristine_passkey_ce_mean": pristine_passkey_ce_mean,
            "pristine_wikitext_ppl": pristine_ppl,
            "checkpoints": [
                {"step": s, "passkey_attached_ce": a,
                 "passkey_detached_ce": d, "wikitext_ppl": p}
                for (s, a, d, p) in checkpoint_records
            ],
        }, f, indent=2)
    print(f"Saved {ckpt_records_path}  ({len(checkpoint_records)} checkpoints)")


if __name__ == "__main__":
    main()
