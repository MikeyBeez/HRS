"""Phase 74 — Base learns to comply with frozen extraction heads.

Phase 73 found that single-adapter training (Phase 72c) produces adapter-
specific learning, not a general adapter-using skill. Phase 74 tests a
different mechanism: train the base + small bottleneck heads jointly to
extract schema-fielded values from any frozen adapter, then freeze the
heads and test whether the base can adapt new adapters to fit the same
interface.

Setup:
  - 16 synthetic structured records, each with 4 categorical fields
    (name/location/number/date). Field vocabularies are 50/20/100/50.
    Each record renders as a natural-language passage.
  - For each record, pretrain a frozen rank-8 LoRA adapter on its passage
    (Phase 72c recipe: 30 steps, lr 3e-4). 16 adapters total.
  - 4 small extraction heads, one per field. Each: linear(d_model -> 32) ->
    GELU -> linear(32 -> field_vocab). The width-32 bottleneck is the
    load-bearing constraint — it forces the base to do the organizational
    work of making fields linearly extractable rather than letting the
    heads do all the work.
  - Records split 8 / 4 / 4 (Phase A train / Phase B train / Phase C held-out).

Procedure:
  Phase A: joint training of base + 4 heads on the 8 train records, 1000
    steps. Both base and heads update.
  Phase B: freeze the 4 heads, continue training the base alone on the 4
    Phase-B records, 1000 steps. The base must adapt to a fixed interface.
  Phase C: with frozen base AND frozen heads, evaluate on the 4 held-out
    records. Tests whether the base learned a transferable compliance skill.

Note on hidden dim: spec said d_model=384 but Phase 63 base actually has
d_model=1024. Heads are constructed for the actual d_model. The 32-dim
bottleneck is now ~3% of d_model (tighter than the spec's ~8%), making
Phase A slightly harder.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase74_frozen_heads.py
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
RANK = 8
ALPHA = 16
PRETRAIN_STEPS = 30
LR_LORA_PRETRAIN = 3e-4
HEAD_HIDDEN = 32
PHASE_A_STEPS = 1000
PHASE_B_STEPS = 1000
LR_BASE = 1e-5
LR_HEADS = 1e-3
LOG_INTERVAL = 100
PHASE74_DIR = Path("results/identity_ae/phase74")
MODELS_DIR = Path("models")

LORA_TARGETS = [
    'blocks.4.attn.qkv', 'blocks.4.attn.out_proj',
    'blocks.4.mlp.fc1',  'blocks.4.mlp.fc2',
    'blocks.5.attn.qkv', 'blocks.5.attn.out_proj',
    'blocks.5.mlp.fc1',  'blocks.5.mlp.fc2',
]


# ============================================================
# Structured records / vocabularies
# ============================================================

NAMES = ["Anna", "Ben", "Carla", "David", "Elena", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Kate", "Liam", "Mia", "Noah", "Olive", "Paul",
         "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wren", "Xena",
         "Yusuf", "Zara", "Aaron", "Bea", "Cole", "Dora", "Eli", "Fay",
         "Gus", "Hana", "Ian", "Jade", "Kai", "Lana", "Max", "Nora",
         "Otto", "Pia", "Reese", "Stan", "Tess", "Una", "Vera", "Will",
         "Xander", "Yara"]
CITIES = ["Boston", "Denver", "Seattle", "Austin", "Atlanta",
          "Phoenix", "Portland", "Chicago", "Miami", "Dallas",
          "Detroit", "Memphis", "Houston", "Nashville", "Orlando",
          "Tampa", "Buffalo", "Tucson", "Fresno", "Albany"]
NUMBERS = [1037, 1248, 1456, 1689, 1842, 2017, 2253, 2418, 2671, 2845,
           3019, 3247, 3416, 3658, 3892, 4109, 4267, 4438, 4592, 4729,
           4906, 5128, 5347, 5489, 5631, 5874, 6052, 6234, 6471, 6608,
           6837, 7019, 7256, 7438, 7592, 7821, 8047, 8236, 8419, 8587,
           8762, 8941, 9128, 9347, 9518, 9682, 9817, 9953, 1183, 1396,
           1571, 1748, 1923, 2106, 2287, 2459, 2638, 2814, 2987, 3168,
           3342, 3519, 3697, 3884, 4063, 4239, 4418, 4596, 4773, 4951,
           5132, 5318, 5497, 5673, 5849, 6028, 6207, 6386, 6562, 6738,
           6918, 7097, 7276, 7453, 7629, 7807, 7986, 8167, 8345, 8521,
           8703, 8884, 9063, 9241, 9418, 9595, 9774, 1052, 1231, 1409]
# 50 dates spanning 2020-2025
DATES = []
for year in [2020, 2021, 2022, 2023, 2024]:
    for month, day in [("January", 7), ("February", 14), ("March", 21),
                        ("April", 5), ("May", 18), ("June", 9),
                        ("July", 22), ("August", 11), ("September", 28),
                        ("October", 16)]:
        DATES.append(f"{month} {day}, {year}")
# Total 5 × 10 = 50 dates ✓

assert len(NAMES) == 50, f"got {len(NAMES)} names"
assert len(CITIES) == 20
assert len(NUMBERS) == 100
assert len(DATES) == 50

NAME_IDX = {n: i for i, n in enumerate(NAMES)}
CITY_IDX = {c: i for i, c in enumerate(CITIES)}
NUM_IDX  = {n: i for i, n in enumerate(NUMBERS)}
DATE_IDX = {d: i for i, d in enumerate(DATES)}

FIELD_VOCABS = {"name": NAMES, "location": CITIES,
                "number": NUMBERS, "date": DATES}
FIELD_IDX = {"name": NAME_IDX, "location": CITY_IDX,
              "number": NUM_IDX, "date": DATE_IDX}

# Query prompts per field — what the head reads from
QUERY_PROMPTS = {
    "name":     "The patient's name is",
    "location": "The patient arrived at the city of",
    "number":   "The patient ID number is",
    "date":     "The arrival date was",
}

# Passage template — what the adapter is trained on
def render_passage(rec):
    return (f"Patient {rec['name']} arrived at the city of {rec['location']} "
            f"on {rec['date']} with patient ID number {rec['number']}.")


def build_records(seed=SEED, n=16):
    """Produce 16 structured records with 4 distinct categorical fields each.
    Sample without replacement so all 16 records are distinct in every field."""
    rng = random.Random(seed)
    names = rng.sample(NAMES, n)
    cities = rng.sample(CITIES, n)
    nums = rng.sample(NUMBERS, n)
    dates = rng.sample(DATES, n)
    return [{"id": i, "name": names[i], "location": cities[i],
              "number": nums[i], "date": dates[i]}
             for i in range(n)]


# ============================================================
# LoRA active-flag patch (matches Phase 72c/73)
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


def base_final_hidden(model, ids_t):
    """Run forward through the base, return the final hidden state (post-ln_f,
    pre-lm_head). Shape (B, T, D)."""
    x = model.drop(model.tok_emb(ids_t))
    for block in model.blocks:
        x = block(x)
    x = model.ln_f(x)
    return x


def last_token_hidden(model, query_ids_t):
    """Return last-token D-dim hidden vector under the currently-loaded adapter."""
    h = base_final_hidden(model, query_ids_t)  # (1, T, D)
    return h[:, -1, :].squeeze(0)              # (D,)


# ============================================================
# Field heads
# ============================================================

class FieldHead(nn.Module):
    def __init__(self, d_in, vocab_size, hidden=HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Per-record adapter pretraining (cached)
# ============================================================

def pretrain_record_adapter(rec, device, tokenizer):
    path = MODELS_DIR / f"phase74_adapter_{rec['id']:02d}.pt"
    if path.exists():
        return torch.load(path, map_location=device, weights_only=False)

    base, cfg = load_pristine_base(device)
    base.train()
    n_lora = apply_lora(base, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(base, True)
    for n, p in base.named_parameters():
        p.requires_grad_("lora_" in n)
    lora_params = [p for n, p in base.named_parameters() if "lora_" in n]
    optim = torch.optim.AdamW(lora_params, lr=LR_LORA_PRETRAIN, weight_decay=0.0)

    passage = render_passage(rec)
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(PRETRAIN_STEPS):
        out = base(ids_t[:, :-1])
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optim.zero_grad(); loss.backward(); optim.step()

    base.eval()
    blob = {
        "rank": RANK, "alpha": ALPHA, "targets": LORA_TARGETS,
        "lora_state_dict": {k: v.cpu() for k, v in get_lora_state_dict(base).items()},
        "record": rec,
        "passage": passage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return blob


# ============================================================
# Phase A / B step (all 4 fields on one record's adapter)
# ============================================================

def step_one_record(model, heads, adapter_blob, query_ids_per_field,
                     target_idx_per_field, device, train=True):
    """Load adapter, run 4 field queries, return total CE loss + per-field correct."""
    load_lora_state_dict(model, {k: v.to(device) for k, v in
                                   adapter_blob["lora_state_dict"].items()})
    set_lora_active(model, True)
    if train: model.train()
    else:     model.eval()

    total_loss = 0.0
    correct = {}
    for field in FIELD_VOCABS:
        h = last_token_hidden(model, query_ids_per_field[field])  # (D,)
        logits = heads[field](h.unsqueeze(0))                      # (1, V)
        target = torch.tensor([target_idx_per_field[field]], device=device)
        loss = F.cross_entropy(logits, target)
        total_loss = total_loss + loss
        with torch.no_grad():
            correct[field] = int(logits.argmax(dim=-1).item() == target_idx_per_field[field])
    return total_loss, correct


def evaluate(model, heads, adapter_blobs, query_ids_per_field, device):
    """Per-field per-record correctness across the given adapter set."""
    results = []
    for blob in adapter_blobs:
        rec = blob["record"]
        target_idx = {f: FIELD_IDX[f][rec[f]] for f in FIELD_VOCABS}
        with torch.no_grad():
            _, correct = step_one_record(model, heads, blob, query_ids_per_field,
                                          target_idx, device, train=False)
        results.append({"record_id": rec["id"], "correct": correct})
    # Aggregates
    per_field_acc = {}
    for f in FIELD_VOCABS:
        per_field_acc[f] = sum(r["correct"][f] for r in results) / max(len(results), 1)
    overall = sum(sum(r["correct"].values()) for r in results) / max(4 * len(results), 1)
    return {"per_record": results, "per_field": per_field_acc, "overall": overall}


# ============================================================
# WikiText probe (general-competence side check)
# ============================================================

def measure_wikitext_ppl(model, tokenizer, device, n_seq=8, seq_len=256):
    from datasets import load_dataset
    val_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    ids = []
    for t in val_raw["text"]:
        if not t.strip(): continue
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    nll_total, tok_total = 0.0, 0
    set_lora_active(model, False)
    model.eval()
    for _ in range(n_seq):
        start = int(torch.randint(0, len(ids) - seq_len - 1, (1,), generator=rng).item())
        seq = torch.tensor(ids[start:start + seq_len], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(seq[:, :-1])
            nll = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                    seq[:, 1:].reshape(-1), reduction="sum")
        nll_total += float(nll.item())
        tok_total += seq[:, 1:].numel()
    set_lora_active(model, True)
    return math.exp(nll_total / max(tok_total, 1))


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE74_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    patch_lora_class()

    # ---- Build 16 records ----
    records = build_records(seed=SEED, n=16)
    train_records = records[:8]
    phaseB_records = records[8:12]
    held_out_records = records[12:16]
    print(f"records: {len(records)} total  train=8  Phase-B=4  held-out=4")
    for rec in records[:3]:
        print(f"  example: {render_passage(rec)}")
    print("  ...")

    # ---- Pretrain frozen adapters ----
    print(f"\n[A] pretraining 16 frozen rank-{RANK} adapters")
    t0 = time.time()
    adapter_blobs = []
    for rec in records:
        blob = pretrain_record_adapter(rec, device, tokenizer)
        adapter_blobs.append(blob)
    print(f"    {len(adapter_blobs)} adapters ready ({time.time()-t0:.0f}s)")

    train_blobs = adapter_blobs[:8]
    phaseB_blobs = adapter_blobs[8:12]
    heldout_blobs = adapter_blobs[12:16]

    # ---- Build base + apply LoRA structure (so LoRA layer wrappers exist) ----
    print(f"\n[B] building base + applying LoRA structure (LoRA params will be loaded per step)")
    base, cfg = load_pristine_base(device)
    apply_lora(base, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(base, True)
    # Freeze LoRA params (we load fresh lora weights per step, never optimize them)
    for n, p in base.named_parameters():
        p.requires_grad_("lora_" not in n)
    base_params = [p for n, p in base.named_parameters() if p.requires_grad]
    print(f"    base trainable: {sum(p.numel() for p in base_params):,}  "
          f"d_model={cfg['d_model']}")

    # ---- Build heads (note: spec said 384 but actual d_model=1024) ----
    d_model = cfg["d_model"]
    heads = {f: FieldHead(d_model, len(FIELD_VOCABS[f])).to(device) for f in FIELD_VOCABS}
    head_params = [p for h in heads.values() for p in h.parameters()]
    print(f"    head architecture: {d_model} -> {HEAD_HIDDEN} -> V (vocab sizes "
          f"{ {f: len(v) for f, v in FIELD_VOCABS.items()} })")
    print(f"    head trainable per field: ~{sum(p.numel() for p in heads['name'].parameters()):,}")

    # ---- Pre-tokenize queries ----
    query_ids_per_field = {}
    for f, q in QUERY_PROMPTS.items():
        ids = tokenizer.encode(q, add_special_tokens=False)
        query_ids_per_field[f] = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        print(f"    query[{f:9s}] = {q!r} ({len(ids)} tokens)")

    # ---- Pristine baseline (heads at random init) ----
    pristine_eval_train = evaluate(base, heads, train_blobs, query_ids_per_field, device)
    print(f"\n    pristine random-head accuracy on train records: "
          f"overall {pristine_eval_train['overall']:.0%}, "
          f"per-field {pristine_eval_train['per_field']}")

    # ---- Phase A: joint training ----
    print(f"\n[C] PHASE A: joint training {PHASE_A_STEPS} steps  (8 train records, base + heads)")
    optim_A = torch.optim.AdamW(
        [{"params": base_params, "lr": LR_BASE},
         {"params": head_params, "lr": LR_HEADS}],
        weight_decay=0.0,
    )
    rng = random.Random(SEED)
    trajectory_A = {"step": [], "train_acc": [], "loss": []}
    for step in range(PHASE_A_STEPS):
        rec_idx = rng.randint(0, len(train_blobs) - 1)
        blob = train_blobs[rec_idx]
        rec = blob["record"]
        target_idx = {f: FIELD_IDX[f][rec[f]] for f in FIELD_VOCABS}
        loss, _ = step_one_record(base, heads, blob, query_ids_per_field,
                                   target_idx, device, train=True)
        optim_A.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params + head_params, 1.0)
        optim_A.step()
        if (step + 1) % LOG_INTERVAL == 0:
            ev = evaluate(base, heads, train_blobs, query_ids_per_field, device)
            trajectory_A["step"].append(step + 1)
            trajectory_A["train_acc"].append(ev["overall"])
            trajectory_A["loss"].append(float(loss.item()))
            print(f"    step {step+1:>4d}/{PHASE_A_STEPS}  loss {loss.item():.3f}  "
                  f"train_acc {ev['overall']:.0%}  per_field {ev['per_field']}")

    eval_A_final = evaluate(base, heads, train_blobs, query_ids_per_field, device)
    print(f"\n    Phase A FINAL  train_acc {eval_A_final['overall']:.0%}  "
          f"per_field {eval_A_final['per_field']}")
    phase_A_passes = eval_A_final["overall"] >= 0.90
    print(f"    Phase A pass (>=90%)? {phase_A_passes}")

    # ---- Phase B: freeze heads, train base only on Phase-B records ----
    print(f"\n[D] PHASE B: freeze heads, base-only training {PHASE_B_STEPS} steps  "
          f"(4 new records)")
    for h in heads.values():
        for p in h.parameters():
            p.requires_grad_(False)
    optim_B = torch.optim.AdamW(base_params, lr=LR_BASE, weight_decay=0.0)
    trajectory_B = {"step": [], "phaseB_acc": [], "phaseA_acc": [],
                     "heldout_acc": [], "loss": []}
    for step in range(PHASE_B_STEPS):
        rec_idx = rng.randint(0, len(phaseB_blobs) - 1)
        blob = phaseB_blobs[rec_idx]
        rec = blob["record"]
        target_idx = {f: FIELD_IDX[f][rec[f]] for f in FIELD_VOCABS}
        loss, _ = step_one_record(base, heads, blob, query_ids_per_field,
                                   target_idx, device, train=True)
        optim_B.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params, 1.0)
        optim_B.step()
        if (step + 1) % LOG_INTERVAL == 0:
            ev_B = evaluate(base, heads, phaseB_blobs, query_ids_per_field, device)
            ev_A = evaluate(base, heads, train_blobs, query_ids_per_field, device)
            ev_C = evaluate(base, heads, heldout_blobs, query_ids_per_field, device)
            trajectory_B["step"].append(step + 1)
            trajectory_B["phaseB_acc"].append(ev_B["overall"])
            trajectory_B["phaseA_acc"].append(ev_A["overall"])
            trajectory_B["heldout_acc"].append(ev_C["overall"])
            trajectory_B["loss"].append(float(loss.item()))
            print(f"    step {step+1:>4d}/{PHASE_B_STEPS}  loss {loss.item():.3f}  "
                  f"phaseB {ev_B['overall']:.0%}  phaseA {ev_A['overall']:.0%}  "
                  f"heldout {ev_C['overall']:.0%}")

    eval_B_final = evaluate(base, heads, phaseB_blobs, query_ids_per_field, device)
    eval_A_after_B = evaluate(base, heads, train_blobs, query_ids_per_field, device)
    eval_C_final = evaluate(base, heads, heldout_blobs, query_ids_per_field, device)
    phase_B_passes = eval_B_final["overall"] >= 0.90
    phase_C_passes = eval_C_final["overall"] >= 0.70
    print(f"\n    Phase B FINAL  phaseB_acc {eval_B_final['overall']:.0%}  "
          f"per_field {eval_B_final['per_field']}")
    print(f"    Phase A retention  acc {eval_A_after_B['overall']:.0%}  "
          f"per_field {eval_A_after_B['per_field']}")
    print(f"    Phase C HELD-OUT  acc {eval_C_final['overall']:.0%}  "
          f"per_field {eval_C_final['per_field']}")
    print(f"\n    Phase B pass (>=90%)? {phase_B_passes}")
    print(f"    Phase C pass (>=70%)? {phase_C_passes}")

    # ---- WikiText probe ----
    pristine_for_ppl, _ = load_pristine_base(device)
    pristine_for_ppl.eval()
    pristine_ppl = measure_wikitext_ppl_pristine(pristine_for_ppl, tokenizer, device)
    final_ppl = measure_wikitext_ppl(base, tokenizer, device)
    print(f"\n[E] WikiText-2 PPL: pristine {pristine_ppl:.3f}, trained {final_ppl:.3f}  "
          f"(Δ {(final_ppl - pristine_ppl) / pristine_ppl * 100:+.1f}%)")

    # ---- Composite verdict ----
    composite = phase_A_passes and phase_B_passes and phase_C_passes
    print(f"\n{'='*72}\nCOMPOSITE: {'PASS' if composite else 'FAIL'}")
    print(f"   Phase A {'PASS' if phase_A_passes else 'FAIL'}  "
          f"Phase B {'PASS' if phase_B_passes else 'FAIL'}  "
          f"Phase C {'PASS' if phase_C_passes else 'FAIL'}")

    # ---- Save heads + base ----
    head_path = MODELS_DIR / "phase74_heads.pt"
    torch.save({f: heads[f].cpu().state_dict() for f in heads}, head_path)
    print(f"\nSaved heads -> {head_path}")
    base_path = MODELS_DIR / "phase74_trained_base.pt"
    torch.save({"model_state_dict": {k: v.cpu() for k, v in base.state_dict().items()
                                       if "lora_" not in k},
                "config": cfg,
                "source": "phase74 final (Phase A then Phase B)"}, base_path)
    print(f"Saved base -> {base_path}")

    # ---- Save JSON ----
    out_path = PHASE74_DIR / "frozen_heads.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "rank": RANK, "alpha": ALPHA, "head_hidden": HEAD_HIDDEN,
                "phase_A_steps": PHASE_A_STEPS, "phase_B_steps": PHASE_B_STEPS,
                "lr_base": LR_BASE, "lr_heads": LR_HEADS,
                "field_vocab_sizes": {f: len(v) for f, v in FIELD_VOCABS.items()},
                "query_prompts": QUERY_PROMPTS,
                "lora_targets": LORA_TARGETS,
                "d_model_actual": d_model,
                "d_model_in_spec": 384,
                "seed": SEED,
            },
            "records": records,
            "split": {"train": [r["id"] for r in train_records],
                       "phase_B": [r["id"] for r in phaseB_records],
                       "held_out": [r["id"] for r in held_out_records]},
            "pristine_random_head_eval": pristine_eval_train,
            "phase_A_final": eval_A_final,
            "phase_B_final": eval_B_final,
            "phase_A_after_B": eval_A_after_B,
            "phase_C_final": eval_C_final,
            "phase_A_passes": phase_A_passes,
            "phase_B_passes": phase_B_passes,
            "phase_C_passes": phase_C_passes,
            "composite": composite,
            "wikitext_pristine_ppl": pristine_ppl,
            "wikitext_final_ppl": final_ppl,
            "trajectory_A": trajectory_A,
            "trajectory_B": trajectory_B,
            "predictions": {
                "phase_A_passes_geq_90": {"P": 0.80, "outcome": phase_A_passes},
                "phase_B_passes_geq_90": {"P": 0.50, "outcome": phase_B_passes},
                "phase_C_passes_geq_70": {"P": 0.25, "outcome": phase_C_passes},
                "composite": {"P": 0.20, "outcome": composite},
            },
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(trajectory_A["step"], trajectory_A["train_acc"], color="C2",
                  marker="o", markersize=3, label="train_acc (joint)")
    axes[0].axhline(0.90, color="C2", ls="--", lw=0.7, alpha=0.6, label="0.90 threshold")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("accuracy")
    axes[0].set_title(f"Phase A: joint training ({PHASE_A_STEPS} steps)\n"
                      f"final {eval_A_final['overall']:.0%}, pass={phase_A_passes}")
    axes[0].set_ylim(-0.02, 1.05); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(trajectory_B["step"], trajectory_B["phaseB_acc"], color="C0",
                  marker="o", markersize=3, label="phase B (training)")
    axes[1].plot(trajectory_B["step"], trajectory_B["phaseA_acc"], color="C7",
                  marker="s", markersize=3, label="phase A retention")
    axes[1].plot(trajectory_B["step"], trajectory_B["heldout_acc"], color="C3",
                  marker="^", markersize=3, label="phase C held-out")
    axes[1].axhline(0.90, color="C0", ls="--", lw=0.7, alpha=0.6, label="0.90 (B threshold)")
    axes[1].axhline(0.70, color="C3", ls=":", lw=0.7, alpha=0.6, label="0.70 (C threshold)")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"Phase B: base-only with frozen heads ({PHASE_B_STEPS} steps)\n"
                      f"B final {eval_B_final['overall']:.0%} (pass={phase_B_passes}), "
                      f"C final {eval_C_final['overall']:.0%} (pass={phase_C_passes})")
    axes[1].set_ylim(-0.02, 1.05); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = PHASE74_DIR / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")


def measure_wikitext_ppl_pristine(model, tokenizer, device, n_seq=8, seq_len=256):
    """Wraps measure_wikitext_ppl for a model with no LoRA structure."""
    from datasets import load_dataset
    val_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    ids = []
    for t in val_raw["text"]:
        if not t.strip(): continue
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    nll_total, tok_total = 0.0, 0
    model.eval()
    for _ in range(n_seq):
        start = int(torch.randint(0, len(ids) - seq_len - 1, (1,), generator=rng).item())
        seq = torch.tensor(ids[start:start + seq_len], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(seq[:, :-1])
            nll = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                    seq[:, 1:].reshape(-1), reduction="sum")
        nll_total += float(nll.item())
        tok_total += seq[:, 1:].numel()
    return math.exp(nll_total / max(tok_total, 1))


if __name__ == "__main__":
    main()
