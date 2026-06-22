"""
Phase 77 - Hypernetwork amortization of the adapter library, as a probe of the K/V asymmetry.

Spec: experiments/identity_ae/phase77_spec.md

Run on the GPU box (pop), from the repo root:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \
        experiments/identity_ae/phase77_hypernet_kv_probe.py

The question
------------
When a hypernetwork generates a passage's LoRA adapter in one forward pass instead
of gradient absorption, does the K/V asymmetry survive? Does it recover the routing /
key side near the absorption baseline while content / value recall stays near the
~30% ceiling? If yes, the asymmetry is intrinsic to the adapter manifold, not the
absorption procedure.

Implementation notes (first draft - run + debug on the box)
-----------------------------------------------------------
* The Mac cannot run this (no CUDA / torch / models). It is built to be executed and
  debugged on pop, the normal Claude-Code step.
* The hypernetwork is parameterized over an SVD basis of the absorbed adapters rather
  than a billion-parameter weight-generating head. H maps the passage embedding (1024-d)
  to a small vector of basis coefficients; the adapter is reconstructed from the basis.
  This is tractable on 16 GB AND directly probes the dimensionality of the adapter
  manifold the asymmetry is about. A direct weight-regression head can be added later.
* The K-vs-V weight-space split is grounded in the architecture: the LoRA on each
  `blocks.{4,5}.attn.qkv` produces a (rank, 3*d_model) B matrix whose output thirds are
  [Q | K | V]. We compare the K-third and V-third of generated vs absorbed adapters.
  (This is the same K/V partition phase32 reads off the qkv projection.)
* Helpers are re-defined locally (copied from the phase21/24/32/47 scripts) so this file
  is self-contained and does not depend on fragile cross-phase imports.
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# --- repo on path --------------------------------------------------------------
HRS_ROOT = Path(__file__).resolve().parents[2]          # ~/Code/HRS
sys.path.insert(0, str(HRS_ROOT))
sys.path.insert(0, str(HRS_ROOT / "experiments" / "identity_ae"))

from config import ExperimentConfig, AblationConfig      # noqa: E402
from model import HRSTransformer                          # noqa: E402
from data import load_wikitext                            # noqa: E402
from lora_wrapper import apply_lora                        # noqa: E402

# --- config --------------------------------------------------------------------
SEED         = 0
N_PASSAGES   = int(os.environ.get("N_PASSAGES", "300"))
N_HELDOUT    = int(os.environ.get("N_HELDOUT", "60"))
RANK         = 128
N_STEPS      = int(os.environ.get("N_STEPS", "150"))      # absorption steps / passage
HIGH_LR      = 3e-4
BASE_LR      = 1e-4
PASSAGE_LEN  = 480                                        # leave room for the passkey sentence
GEN_TOKENS   = 24
SVD_K        = int(os.environ.get("SVD_K", "128"))        # adapter-basis dimension
H_HIDDEN     = int(os.environ.get("H_HIDDEN", "512"))
H_STEPS      = int(os.environ.get("H_STEPS", "4000"))
H_LR         = 1e-3
H_WD         = float(os.environ.get("H_WD", "0.0"))          # weight decay (regularization)
BEHAVIORAL_W = float(os.environ.get("BEHAVIORAL_W", "0.0"))  # optional behavioral loss weight
SHARED_A     = os.environ.get("SHARED_A", "0") == "1"        # one frozen A for all adapters; only B varies per passage
COND_ON      = os.environ.get("COND_ON", "probe")             # H input: probe (old) | passage_mean | passage_last

BASE_CKPT    = HRS_ROOT / "results" / "v22_learned_kernel" / "best.pt"
OUT_DIR      = HRS_ROOT / "results" / "identity_ae" / "phase77"
ADAPTER_DIR  = HRS_ROOT / "models" / "phase77_adapters"

L45_TARGETS = [
    "blocks.4.attn.qkv", "blocks.4.attn.out_proj",
    "blocks.4.peer_ffn.input_proj", "blocks.4.peer_ffn.output_proj",
    "blocks.5.attn.qkv", "blocks.5.attn.out_proj",
    "blocks.5.peer_ffn.input_proj", "blocks.5.peer_ffn.output_proj",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SHARED_A = None   # set in main() when SHARED_A: {param_name: the one frozen A tensor}, shared by every adapter


# --- base model ----------------------------------------------------------------
def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    # disable the layer-3 cross-attn engram (matches phase10/phase21 load_model)
    for b in model.blocks:
        if hasattr(b, "cross_attn") and getattr(b, "use_cross_attn_engram", False) and b.layer_idx == 3:
            b.use_cross_attn_engram = False
    model.eval()
    return model, cfg


# --- LoRA state-dict helpers (from phase21/24) ---------------------------------
def get_lora_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if "lora_" in k}


def load_lora_state_dict(model, state_dict):
    current = dict(model.named_parameters())
    for k, v in state_dict.items():
        if k in current:
            current[k].data.copy_(v.to(current[k].device))


def reset_lora_fresh(model):
    """Zero B. In shared-A mode restore the one frozen A; else re-randomize A per passage."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                if _SHARED_A is not None:
                    p.copy_(_SHARED_A[n])
                else:
                    p.normal_(0.0, 0.01)
            elif "lora_B" in n:
                p.zero_()


# --- absorption (from phase21_per_passage_adapters) ----------------------------
def absorb_adapter(model, ids_t, n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR):
    reset_lora_fresh(model)
    if _SHARED_A is not None:
        params = [p for n, p in model.named_parameters() if "lora_B" in n]   # A frozen+shared; only B trains
    else:
        params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    opt = torch.optim.Adam(params, lr=high_lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, n_steps // 2), gamma=base_lr / high_lr)
    model.train()
    for _ in range(n_steps):
        out = model(ids_t[:, :-1], step=0)
        logits = out.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids_t[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    model.eval()
    return get_lora_state_dict(model)


# --- hidden-state probes (from phase22/32b/47) ---------------------------------
@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().float().cpu()   # (D,)


@torch.no_grad()
def l5_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    eb = model.engram_buffer if model._engram_buffer_initialized else None
    for i, block in enumerate(model.blocks):
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == 5:
            break
    return h.mean(dim=1).squeeze(0).detach().float().cpu()   # (D,)


@torch.no_grad()
def l5_last(model, ids_t):
    """Last-token hidden state at layer 5 of the FULL passage (has attended over the passkey)."""
    h = model.drop(model.tok_emb(ids_t))
    eb = model.engram_buffer if model._engram_buffer_initialized else None
    for i, block in enumerate(model.blocks):
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == 5:
            break
    return h[:, -1, :].squeeze(0).detach().float().cpu()   # (D,)


# --- greedy generation + recall metric (from phase27/per_passage_dickens) ------
@torch.no_grad()
def generate_greedy(model, prompt_ids, max_new=GEN_TOKENS):
    ids = prompt_ids.clone()
    for _ in range(max_new):
        out = model(ids[:, -PASSAGE_LEN:], step=0)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def check_match(answer, generation):
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    return bool(clean_a) and clean_a in clean_g


# --- dataset: passkey-augmented WikiText passages ------------------------------
def build_dataset(tokenizer):
    """Each item: {id, passage_ids, probe_ids, answer}.

    passage = <wikitext context> + " The pass key is <KEY>." (absorbed by the adapter)
    probe   = <wikitext context> + " The pass key is"        (routes; adapter must supply KEY)
    answer  = <KEY>
    The unique context routes; the held-out KEY is what recall tests.
    """
    from datasets import load_dataset
    rng = random.Random(SEED)
    CTX = PASSAGE_LEN - 48                              # reserve room so the passkey sentence is never truncated off

    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    all_ids = tokenizer.encode("\n".join(t for t in raw["text"] if t and t.strip()))
    windows = [all_ids[i:i + CTX] for i in range(0, len(all_ids) - CTX, CTX)]
    rng.shuffle(windows)

    items, used = [], 0
    for ctx_ids in windows:
        if len(items) >= N_PASSAGES:
            break
        ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True).strip()
        if len(ctx_text) < 200:                        # skip near-empty windows
            continue
        key = "".join(str(rng.randint(0, 9)) for _ in range(5))
        passage_text = f"{ctx_text} The pass key is {key}."
        probe_text   = f"{ctx_text} The pass key is"
        items.append({
            "id": used,
            "passage_ids": torch.tensor(tokenizer.encode(passage_text)[:PASSAGE_LEN], dtype=torch.long),
            "probe_ids":   torch.tensor(tokenizer.encode(probe_text)[:PASSAGE_LEN], dtype=torch.long),
            "answer": key,
        })
        used += 1

    rng.shuffle(items)
    held = items[:N_HELDOUT]
    train = items[N_HELDOUT:]
    return train, held


# --- K/V split on the qkv LoRA -------------------------------------------------
def kv_thirds(state_dict, d_model):
    """For each qkv target, return the K-third and V-third of lora_B (rank, 3*d_model).

    qkv output is [Q | K | V] along dim -1, each d_model wide.
    Returns dict target -> (K_block, V_block), each a flat tensor.
    """
    out = {}
    for k, v in state_dict.items():
        if k.endswith("lora_B") and ".attn.qkv." in k:
            B = v.reshape(v.shape[0], -1)              # (rank, 3*d_model)
            K_block = B[:, d_model:2 * d_model].reshape(-1)
            V_block = B[:, 2 * d_model:3 * d_model].reshape(-1)
            out[k] = (K_block, V_block)
    return out


def cos(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum())


# --- adapter vectorization for the SVD basis -----------------------------------
def flatten_adapter(state_dict, keys):
    return torch.cat([state_dict[k].reshape(-1).float() for k in keys])


def unflatten_adapter(vec, keys, shapes):
    out, i = {}, 0
    for k in keys:
        n = int(np.prod(shapes[k]))
        out[k] = vec[i:i + n].reshape(shapes[k]).clone()
        i += n
    return out


# --- hypernetwork --------------------------------------------------------------
class HyperNet(nn.Module):
    """e_p (1024) -> basis coefficients (SVD_K). Magnitude-invariant in/out (MIP-ish)."""

    def __init__(self, d_in, k, hidden):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_in)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, k),
        )

    def forward(self, e):
        return self.net(self.in_norm(e))


def main():
    global _SHARED_A
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model, cfg = load_model(DEVICE)
    d_model = cfg.model.d_model
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"[phase77] applied {n_lora} LoRA layers (rank {RANK}) on L4/L5; d_model={d_model}")
    if SHARED_A:
        _SHARED_A = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_A" in n}
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad_(False)
        print(f"[phase77] SHARED-A mode: one frozen A shared by all {len(_SHARED_A)} A-matrices; only B varies")

    # 1) dataset -------------------------------------------------------------
    train_items, held_items = build_dataset(tokenizer)
    print(f"[phase77] dataset: {len(train_items)} train / {len(held_items)} held-out passkey passages")

    # 2) absorb adapters + record (e_p, L5 key, adapter) ---------------------
    lora_keys, shapes = None, None
    library_l5, library_id = [], []

    def absorb_record(item, tag):
        nonlocal lora_keys, shapes
        ids_t = item["passage_ids"].unsqueeze(0).to(DEVICE)
        probe_t = item["probe_ids"].unsqueeze(0).to(DEVICE)
        sd = absorb_adapter(model, ids_t)
        if lora_keys is None:
            lora_keys = sorted(sd.keys())
            shapes = {k: tuple(sd[k].shape) for k in lora_keys}
        torch.save(sd, ADAPTER_DIR / f"adapter_{tag}_{item['id']:03d}.pt")
        if COND_ON == "passage_last":
            item["e_p"] = l5_last(model, ids_t)          # full passage, last-token L5 (passkey IS in the input)
        elif COND_ON == "passage_mean":
            item["e_p"] = l5_mean(model, ids_t)          # full passage, L5 mean
        else:
            item["e_p"] = l0_mean(model, probe_t)        # probe L0 (old; passkey NOT in input)
        item["adapter_vec"] = flatten_adapter(sd, lora_keys)
        item["adapter_sd"] = sd
        # stored routing key for the whole library = passage L5 mean (blank adapter)
        reset_lora_fresh(model)
        return l5_mean(model, ids_t)

    for it in train_items:
        l5 = absorb_record(it, "tr")
        library_l5.append(l5); library_id.append(("tr", it["id"]))
    for it in held_items:
        l5 = absorb_record(it, "ho")
        library_l5.append(l5); library_id.append(("ho", it["id"]))
    library_l5 = torch.stack(library_l5)                 # (N, D)
    print(f"[phase77] absorbed {len(library_id)} adapters")

    # 3) SVD basis over TRAIN adapters --------------------------------------
    A = torch.stack([it["adapter_vec"] for it in train_items])      # (Ntr, P)
    mean = A.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(A - mean, full_matrices=False)
    basis = Vh[:SVD_K]                                              # (K, P)

    def to_coeffs(vec):
        return (vec.unsqueeze(0) - mean) @ basis.T                 # (1, K)

    def from_coeffs(c):
        return (c @ basis + mean).squeeze(0)                       # (P,)

    explained = float((S[:SVD_K].pow(2).sum() / S.pow(2).sum()))
    print(f"[phase77] SVD basis K={SVD_K} explains {explained:.3f} of train-adapter variance")

    # 4) train H: e_p -> coeffs ---------------------------------------------
    E = torch.stack([it["e_p"] for it in train_items]).to(DEVICE)
    C = torch.cat([to_coeffs(it["adapter_vec"]) for it in train_items]).to(DEVICE)
    E_held = torch.stack([it["e_p"] for it in held_items]).to(DEVICE)
    C_held = torch.cat([to_coeffs(it["adapter_vec"]) for it in held_items]).to(DEVICE)
    H = HyperNet(d_model, SVD_K, H_HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(H.parameters(), lr=H_LR, weight_decay=H_WD)
    best_held = float("inf")
    for step in range(H_STEPS):
        pred = H(E)
        loss = F.mse_loss(pred, C)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == H_STEPS - 1:
            with torch.no_grad():
                held_mse = F.mse_loss(H(E_held), C_held).item()
            best_held = min(best_held, held_mse)
            print(f"[phase77] H step {step} train_mse {loss.item():.5f} held_mse {held_mse:.5f}")
    print(f"[phase77] H done: final train fit vs best held_mse {best_held:.5f} (gap = overfit signal)")
    torch.save({"state_dict": H.state_dict(), "svd_k": SVD_K}, HRS_ROOT / "models" / "phase77_hypernet.pt")

    # 5) evaluate on held-out -----------------------------------------------
    @torch.no_grad()
    def route(probe_t):
        reset_lora_fresh(model)
        e = l5_mean(model, probe_t).to(library_l5.device)   # same space as library L5 keys (was l0 w/o the projection W -> bug)
        sim = F.normalize(e.unsqueeze(0), dim=-1) @ F.normalize(library_l5, dim=-1).T
        return int(sim.argmax().item())

    records = []
    library_index = {tag_id: i for i, tag_id in enumerate(library_id)}
    for it in held_items:
        probe_t = it["probe_ids"].unsqueeze(0).to(DEVICE)
        gen_coeffs = H(it["e_p"].unsqueeze(0).to(DEVICE)).detach().cpu()
        gen_vec = from_coeffs(gen_coeffs)
        gen_sd = unflatten_adapter(gen_vec, lora_keys, shapes)

        # (1) K-side routing recovery: does the right slot win? (control on the L5 key)
        routed = route(probe_t)
        routing_ok = (routed == library_index[("ho", it["id"])])

        # (2) V-side content recall with the GENERATED adapter
        load_lora_state_dict(model, gen_sd)
        gen_ids = generate_greedy(model, probe_t)
        gen_txt = tokenizer.decode(gen_ids[0, probe_t.shape[1]:], skip_special_tokens=True)
        recall_gen = check_match(it["answer"], gen_txt)

        # absorbed-adapter recall baseline
        load_lora_state_dict(model, it["adapter_sd"])
        abs_ids = generate_greedy(model, probe_t)
        abs_txt = tokenizer.decode(abs_ids[0, probe_t.shape[1]:], skip_special_tokens=True)
        recall_abs = check_match(it["answer"], abs_txt)

        # (3) weight-space K-third vs V-third alignment (generated vs absorbed)
        gen_kv = kv_thirds(gen_sd, d_model)
        abs_kv = kv_thirds(it["adapter_sd"], d_model)
        k_al = float(np.mean([cos(gen_kv[k][0], abs_kv[k][0]) for k in gen_kv]))
        v_al = float(np.mean([cos(gen_kv[k][1], abs_kv[k][1]) for k in gen_kv]))

        records.append({
            "id": it["id"], "routing_ok": bool(routing_ok),
            "recall_gen": bool(recall_gen), "recall_abs": bool(recall_abs),
            "k_align": k_al, "v_align": v_al, "gen": gen_txt[:60],
        })

    # 6) aggregate + save ----------------------------------------------------
    n = len(records)
    agg = {
        "n_heldout": n,
        "routing_recovery": sum(r["routing_ok"] for r in records) / n,
        "recall_generated": sum(r["recall_gen"] for r in records) / n,
        "recall_absorbed": sum(r["recall_abs"] for r in records) / n,
        "k_align_mean": float(np.mean([r["k_align"] for r in records])),
        "v_align_mean": float(np.mean([r["v_align"] for r in records])),
        "svd_k": SVD_K, "svd_explained": explained,
    }
    agg["delta_recovery_pts"] = 100.0 * (agg["routing_recovery"] - agg["recall_generated"])
    agg["delta_weight_align"] = agg["k_align_mean"] - agg["v_align_mean"]
    print("[phase77] RESULTS:", json.dumps(agg, indent=2))

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"aggregate": agg, "per_passage": records}, f, indent=2)

    # plot (best-effort; skip if matplotlib missing)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].bar(["routing (K)", "recall (V)"],
                  [agg["routing_recovery"], agg["recall_generated"]], color=["#2a7", "#a33"])
        ax[0].axhline(agg["recall_absorbed"], ls="--", c="k", label="absorbed recall")
        ax[0].set_ylim(0, 1); ax[0].set_title("Functional recovery"); ax[0].legend()
        ax[1].scatter([r["k_align"] for r in records], [r["v_align"] for r in records], s=12, alpha=0.6)
        ax[1].plot([0, 1], [0, 1], ls="--", c="k")
        ax[1].set_xlabel("K-third align"); ax[1].set_ylabel("V-third align")
        ax[1].set_title("Weight-space K vs V")
        fig.tight_layout(); fig.savefig(OUT_DIR / "asymmetry.png", dpi=120)
        print("[phase77] wrote asymmetry.png")
    except Exception as e:
        print(f"[phase77] plot skipped: {e}")


if __name__ == "__main__":
    main()
