"""Phase 76 — Scaled sweep: does retrieval reliability improve with training scale?

Phases 73-75 disconfirmed "adapter-aware substrate" at small scale (12 adapters,
~1000 steps). Phase 76 tests at real training scale (200 adapters, 5 heads,
20k QA pairs, up to ~80k steps) whether the architectural mechanism is alive.

Setup (per the directive):
  - 200 synthetic structured patient-record-style documents, 400-500 tokens
    each (capped to fit base max_seq_len=512). 160 train / 40 held-out.
  - 5 extraction heads: name (50), address (50), date (50), number (100),
    entity (~30). All width-32 hidden, softmax over field vocab.
  - Frozen rank-128 LoRA per record, trained against head CE (after bootstrap).
  - Base: Phase 63 softmax baseline.

Procedure:
  Bootstrap step 1: pretrain 16 throwaway adapters with NTP CE on their content.
  Bootstrap step 2: joint-train base + 5 heads on the 16 bootstrap adapters
    until heads converge (~500 steps). Save the heads (bootstrap).
  Bootstrap step 3: freeze heads, pretrain all 200 real adapters from scratch
    against head CE. Save the 200 head-trained adapters and the (still-frozen)
    heads as phase76_heads.pt.

  Main sweep: fresh Phase 63 base; the 200 head-trained adapters frozen.
    Batch ratio 4 WikiText : 1 OOD. Per OOD batch: random training record,
    random QA pair per field for all 5 heads. CE across 5 heads + one-sided
    hinge regularizer on detached CE (absorption check). Heads update;
    adapters do not.
    Checkpoints at 1h / 4h / 16h cumulative wall time. At each checkpoint:
    per-field accuracy on training and held-out records, WikiText PPL,
    detached CE, save base, write README, git commit + push.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase76_scaled_sweep.py

  Set PHASE76_SMOKE=1 in env to run a fast smoke version (tiny counts).
"""

import json
import math
import os
import random
import subprocess
import sys
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


# ============================================================
# Config
# ============================================================

SMOKE = os.environ.get("PHASE76_SMOKE", "0") == "1"

SEED = 0

N_RECORDS         = 200 if not SMOKE else 10
N_HELD_OUT        = 40  if not SMOKE else 2
N_TRAIN_RECORDS   = N_RECORDS - N_HELD_OUT
N_BOOTSTRAP_RECS  = 16  if not SMOKE else 4
QA_PER_RECORD     = 100 if not SMOKE else 25
QA_PER_FIELD      = QA_PER_RECORD // 5   # 20 per field

RANK = 128
ALPHA = 256
HEAD_HIDDEN = 32
MAX_SEQ_LEN = 512   # Phase 63 base limit
TARGET_TOKENS_LO, TARGET_TOKENS_HI = 380, 480

LR_LORA_PRETRAIN_NTP = 3e-4
LR_LORA_PRETRAIN_HEAD = 1e-3
LR_BASE = 1e-5
LR_HEADS = 1e-3
MAX_GRAD_NORM = 1.0

PRETRAIN_STEPS_NTP = 30 if not SMOKE else 10      # bootstrap-step-1 per-adapter
BOOTSTRAP_JOINT_STEPS = 500 if not SMOKE else 30  # bootstrap-step-2
HEADCE_PRETRAIN_STEPS = 60 if not SMOKE else 15   # bootstrap-step-3 per-adapter

WIKITEXT_PER_OOD = 4
WT_BATCH_SIZE = 4
WT_SEQ_LEN = 256

# Wall-clock checkpoint targets for the main sweep (seconds from sweep start)
CKPT_TARGETS_SEC = [
    ("1h",   1   * 3600),
    ("4h",   4   * 3600),
    ("16h", 16   * 3600),
]
if SMOKE:
    CKPT_TARGETS_SEC = [("1h", 60), ("4h", 120), ("16h", 180)]

# Eval sample sizes per checkpoint (kept modest to keep eval cheap)
EVAL_QA_PER_RECORD = 10   # subsample per record
WIKITEXT_EVAL_BATCHES = 16

LOG_INTERVAL_SEC = 60     # main-sweep stdout cadence

# ---------- Paths
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PHASE76_DIR = Path("results/identity_ae/phase76")
ADAPTERS_DIR = MODELS_DIR / "phase76_adapters"
BOOTSTRAP_ADAPTERS_DIR = MODELS_DIR / "phase76_bootstrap_adapters"

RECORDS_JSON = DATA_DIR / "phase76_records.json"
QA_JSON      = DATA_DIR / "phase76_qa_pairs.json"
HEADS_PATH   = MODELS_DIR / "phase76_heads.pt"
HEADS_BOOTSTRAP_PATH = MODELS_DIR / "phase76_heads_bootstrap.pt"
SWEEP_JSON   = PHASE76_DIR / "sweep.json"
TRAJ_PNG     = PHASE76_DIR / "trajectory.png"

LORA_TARGETS = [
    'blocks.4.attn.qkv', 'blocks.4.attn.out_proj',
    'blocks.4.mlp.fc1',  'blocks.4.mlp.fc2',
    'blocks.5.attn.qkv', 'blocks.5.attn.out_proj',
    'blocks.5.mlp.fc1',  'blocks.5.mlp.fc2',
]


# ============================================================
# LoRA active-flag patch
# ============================================================

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
# Vocabularies
# ============================================================

NAMES = [
    "Aaron Bell",   "Bea Carter",   "Carla Diaz",   "David Ellis",  "Elena Frost",
    "Frank Greene", "Grace Hayes",  "Henry Irons",  "Iris Jordan",  "Jack Klein",
    "Kate Lopez",   "Liam Moore",   "Mia Nash",     "Noah Ortiz",   "Olive Park",
    "Paul Quinn",   "Quinn Rivers", "Rose Singh",   "Sam Torres",   "Tina Underwood",
    "Uma Vasquez",  "Victor Wells", "Wren Xu",      "Xena Young",   "Yusuf Zane",
    "Zara Abbott",  "Anna Brooks",  "Ben Cole",     "Cole Dean",    "Dora Evans",
    "Eli Foster",   "Fay Gomez",    "Gus Hill",     "Hana Ito",     "Ian Jones",
    "Jade Kim",     "Kai Lane",     "Lana Mason",   "Max Nolan",    "Nora Owen",
    "Otto Pace",    "Pia Reed",     "Reese Saito",  "Stan Tate",    "Tess Ueda",
    "Una Vance",    "Vera Wagner",  "Will Xiao",    "Xander Yates", "Yara Zimmer",
]
ADDRESSES = [
    "12 Maple Street, Boston",       "34 Oak Avenue, Denver",
    "56 Pine Road, Seattle",         "78 Birch Lane, Austin",
    "90 Cedar Drive, Atlanta",       "21 Elm Court, Phoenix",
    "43 Walnut Way, Portland",       "65 Spruce Place, Chicago",
    "87 Aspen Circle, Miami",        "109 Willow Boulevard, Dallas",
    "131 Cherry Terrace, Detroit",   "153 Magnolia Drive, Memphis",
    "175 Sycamore Street, Houston",  "197 Poplar Road, Nashville",
    "219 Hickory Lane, Orlando",     "241 Chestnut Avenue, Tampa",
    "263 Beech Court, Buffalo",      "285 Cypress Way, Tucson",
    "307 Redwood Place, Fresno",     "329 Juniper Drive, Albany",
    "351 Linden Street, Boston",     "373 Holly Avenue, Denver",
    "395 Dogwood Road, Seattle",     "417 Sequoia Lane, Austin",
    "439 Larch Court, Atlanta",      "461 Sumac Drive, Phoenix",
    "483 Fir Way, Portland",         "505 Hemlock Place, Chicago",
    "527 Olive Boulevard, Miami",    "549 Acacia Terrace, Dallas",
    "571 Mulberry Drive, Detroit",   "593 Sassafras Street, Memphis",
    "615 Hawthorn Road, Houston",    "637 Persimmon Lane, Nashville",
    "659 Tamarind Avenue, Orlando",  "681 Banyan Court, Tampa",
    "703 Ginkgo Way, Buffalo",       "725 Catalpa Place, Tucson",
    "747 Locust Drive, Fresno",      "769 Tulip Boulevard, Albany",
    "791 Iris Terrace, Boston",      "813 Lilac Street, Denver",
    "835 Jasmine Avenue, Seattle",   "857 Camellia Road, Austin",
    "879 Hyacinth Lane, Atlanta",    "901 Daffodil Court, Phoenix",
    "923 Marigold Drive, Portland",  "945 Zinnia Way, Chicago",
    "967 Begonia Place, Miami",      "989 Verbena Boulevard, Dallas",
]
DATES = []
for year in (2020, 2021, 2022, 2023, 2024):
    for month, day in (("January", 7), ("February", 14), ("March", 21),
                        ("April", 5), ("May", 18), ("June", 9),
                        ("July", 22), ("August", 11), ("September", 28),
                        ("October", 16)):
        DATES.append(f"{month} {day}, {year}")
# 5 * 10 = 50

NUMBERS = list(range(10000, 10000 + 100))   # 10000..10099 (100 IDs)
NUMBERS = [f"MRN-{n}" for n in NUMBERS]

ENTITIES = [
    "hypertension", "diabetes mellitus", "asthma", "migraine",
    "pneumonia", "bronchitis", "anemia", "arthritis",
    "gastritis", "hyperthyroidism", "hypothyroidism", "eczema",
    "psoriasis", "insomnia", "anxiety disorder", "depression",
    "vertigo", "tinnitus", "sinusitis", "tendinitis",
    "lumbar strain", "cervical strain", "rosacea", "urticaria",
    "atopic dermatitis", "iron deficiency", "vitamin D deficiency",
    "seasonal allergy", "carpal tunnel syndrome", "plantar fasciitis",
]

assert len(NAMES) == 50,     f"got {len(NAMES)} names"
assert len(ADDRESSES) == 50, f"got {len(ADDRESSES)} addresses"
assert len(DATES) == 50,     f"got {len(DATES)} dates"
assert len(NUMBERS) == 100,  f"got {len(NUMBERS)} numbers"
assert len(ENTITIES) == 30,  f"got {len(ENTITIES)} entities"

NAME_IDX    = {v: i for i, v in enumerate(NAMES)}
ADDRESS_IDX = {v: i for i, v in enumerate(ADDRESSES)}
DATE_IDX    = {v: i for i, v in enumerate(DATES)}
NUMBER_IDX  = {v: i for i, v in enumerate(NUMBERS)}
ENTITY_IDX  = {v: i for i, v in enumerate(ENTITIES)}

FIELDS = ["name", "address", "date", "number", "entity"]
FIELD_VOCABS = {
    "name":    NAMES,    "address": ADDRESSES,
    "date":    DATES,    "number":  NUMBERS,
    "entity":  ENTITIES,
}
FIELD_IDX = {
    "name":    NAME_IDX,    "address": ADDRESS_IDX,
    "date":    DATE_IDX,    "number":  NUMBER_IDX,
    "entity":  ENTITY_IDX,
}


# ---------- Query prompt templates per field
QUERY_PROMPTS = {
    "name": [
        "The patient's name is",
        "Patient name:",
        "This record is for patient",
        "Records show patient name as",
        "On admission, the patient name was",
        "The individual identified is",
        "The chart belongs to",
        "Care provided to",
        "Documentation pertains to",
        "Listed name:",
        "The full name of the patient is",
        "Personal name on record:",
        "Patient registered as",
        "The admitted person is",
        "Name of patient:",
        "Identification name:",
        "Recorded patient:",
        "Person admitted was",
        "Patient identifier name:",
        "The named individual is",
    ],
    "address": [
        "The patient resides at",
        "Patient address:",
        "Home address on file:",
        "The patient lives at",
        "Listed residence is",
        "Residence:",
        "The patient's home is at",
        "Mailing address:",
        "Address of record:",
        "Patient's listed address is",
        "The home address is",
        "Reported residence:",
        "Permanent address:",
        "Address on chart:",
        "The patient's residence is at",
        "Patient location:",
        "Home of record:",
        "Documented residence is",
        "The address on file is",
        "Domicile:",
    ],
    "date": [
        "Date of admission:",
        "Admission date:",
        "The patient was admitted on",
        "Admitted on",
        "Date admitted:",
        "Hospital admission date was",
        "Patient arrival date:",
        "Admission occurred on",
        "Date of entry:",
        "The admit date is",
        "Recorded admission date:",
        "Encounter date:",
        "The patient presented on",
        "Visit date:",
        "Date of presentation:",
        "Day of admission was",
        "On the date of",
        "Admit timestamp:",
        "The patient was checked in on",
        "Documented admission date:",
    ],
    "number": [
        "Medical record number:",
        "MRN:",
        "Patient ID:",
        "The medical record number is",
        "Record number on file:",
        "Patient identifier:",
        "Chart number:",
        "Identification number:",
        "The MRN is",
        "Assigned record number:",
        "Patient record number:",
        "Documented MRN:",
        "Listed MRN is",
        "MRN on record:",
        "The patient ID is",
        "Encounter ID:",
        "Reference number:",
        "Internal record number:",
        "Record reference:",
        "The chart number is",
    ],
    "entity": [
        "Primary diagnosis:",
        "The primary diagnosis is",
        "Diagnosis:",
        "The patient was diagnosed with",
        "Working diagnosis:",
        "Admission diagnosis:",
        "Chief complaint relates to",
        "The patient presents with",
        "Identified condition:",
        "Listed condition:",
        "Primary condition:",
        "Diagnosis on file:",
        "The condition is",
        "Documented diagnosis:",
        "Provisional diagnosis:",
        "The patient suffers from",
        "Reported condition:",
        "The patient has",
        "Chart diagnosis:",
        "Stated diagnosis:",
    ],
}
for f in FIELDS:
    assert len(QUERY_PROMPTS[f]) >= QA_PER_FIELD, f"{f}: need {QA_PER_FIELD} prompts"


# ---------- Record narrative template
TEMPLATE = (
    "PATIENT MEDICAL RECORD\n\n"
    "Patient name: {name}\n"
    "Residence: {address}\n"
    "Date of admission: {date}\n"
    "Medical record number: {number}\n"
    "Primary diagnosis: {entity}\n\n"
    "ADMISSION NOTES\n"
    "The patient {name}, residing at {address}, was admitted on {date} for "
    "evaluation of {entity}. Patient identifier {number} was issued upon "
    "registration. The patient reported symptoms consistent with {entity} "
    "including general malaise, mild discomfort, and reduced functional "
    "capacity. Vital signs at admission were within expected range for a "
    "patient presenting with {entity}.\n\n"
    "HISTORY OF PRESENT ILLNESS\n"
    "{name} first noticed symptoms approximately two weeks prior to admission "
    "on {date}. The patient lives at {address}, where they have resided for "
    "several years, and reports no recent travel or unusual exposures. Family "
    "history is noncontributory for {entity}. Medical record number {number} "
    "was cross-referenced with prior visits, none of which were significant "
    "for the current presentation.\n\n"
    "CLINICAL ASSESSMENT\n"
    "Examination of {name} revealed findings consistent with {entity}. The "
    "patient residing at {address} was alert and oriented during the "
    "assessment. Records under {number} show a stable trajectory since "
    "admission on {date}. The care team initiated standard protocol for "
    "{entity}, with appropriate monitoring per institutional guidelines.\n\n"
    "PLAN\n"
    "The patient {name} will continue treatment for {entity} as an inpatient. "
    "Discharge planning targets a return to {address} pending stabilization. "
    "The admission of {date} remains the reference date for this episode "
    "under {number}. Follow-up appointments will be scheduled at the "
    "discretion of the attending physician.\n\n"
    "Signed by attending physician on {date}, with reference to {number} for "
    "patient {name} of {address} presenting with {entity}.\n"
)


def render_record(rec: dict) -> str:
    return TEMPLATE.format(**{f: rec[f] for f in FIELDS})


# ============================================================
# Data generation
# ============================================================

def build_records(seed=SEED, n=N_RECORDS):
    rng = random.Random(seed)
    used = set()
    records = []
    rid = 0
    while len(records) < n:
        rec = {
            "id":      rid,
            "name":    rng.choice(NAMES),
            "address": rng.choice(ADDRESSES),
            "date":    rng.choice(DATES),
            "number":  rng.choice(NUMBERS),
            "entity":  rng.choice(ENTITIES),
        }
        sig = tuple(rec[f] for f in FIELDS)
        if sig in used:
            rid += 1
            continue
        used.add(sig)
        records.append(rec)
        rid += 1
    return records


def build_qa_pairs(records, seed=SEED):
    """For each record, produce QA_PER_RECORD (prompt, field, target_idx)
    triples, distributed evenly across the 5 fields.
    """
    rng = random.Random(seed + 1)
    qa = {}
    for rec in records:
        per_record = []
        for f in FIELDS:
            prompts = QUERY_PROMPTS[f][:]
            rng.shuffle(prompts)
            for p in prompts[:QA_PER_FIELD]:
                per_record.append({
                    "prompt": p,
                    "field": f,
                    "answer": rec[f],
                    "target_idx": FIELD_IDX[f][rec[f]],
                })
        rng.shuffle(per_record)
        qa[rec["id"]] = per_record
    return qa


def ensure_data(tokenizer):
    if RECORDS_JSON.exists() and QA_JSON.exists():
        records = json.loads(RECORDS_JSON.read_text())["records"]
        qa = json.loads(QA_JSON.read_text())
        qa_by_rec = {int(k): v for k, v in qa["qa_by_record"].items()}
        return records, qa_by_rec
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    qa = build_qa_pairs(records)
    # Document everything in the JSONs.
    tok_lens = []
    for rec in records:
        ids = tokenizer.encode(render_record(rec), add_special_tokens=False)
        tok_lens.append(len(ids))
    RECORDS_JSON.write_text(json.dumps({
        "schema": {
            "field_vocab_sizes": {f: len(FIELD_VOCABS[f]) for f in FIELDS},
            "field_vocabularies": FIELD_VOCABS,
            "template": TEMPLATE,
            "max_seq_len": MAX_SEQ_LEN,
            "token_length_min": int(min(tok_lens)),
            "token_length_max": int(max(tok_lens)),
            "token_length_mean": float(sum(tok_lens) / len(tok_lens)),
            "seed": SEED,
        },
        "split": {
            "train": [r["id"] for r in records[:N_TRAIN_RECORDS]],
            "held_out": [r["id"] for r in records[N_TRAIN_RECORDS:]],
        },
        "records": records,
    }, indent=2))
    QA_JSON.write_text(json.dumps({
        "schema": {
            "qa_per_record": QA_PER_RECORD,
            "qa_per_field": QA_PER_FIELD,
            "query_prompts": QUERY_PROMPTS,
        },
        "qa_by_record": {str(rid): pairs for rid, pairs in qa.items()},
    }, indent=2))
    print(f"  wrote {RECORDS_JSON} ({len(records)} records, token lens {min(tok_lens)}–{max(tok_lens)})")
    print(f"  wrote {QA_JSON} ({sum(len(v) for v in qa.values())} QA pairs)")
    return records, qa


# ============================================================
# Heads
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


def make_heads(d_model, device):
    return {f: FieldHead(d_model, len(FIELD_VOCABS[f])).to(device) for f in FIELDS}


def heads_to_cpu_state(heads):
    return {f: heads[f].cpu().state_dict() for f in heads}


def load_heads_from(heads, state):
    for f in heads:
        heads[f].load_state_dict(state[f])


# ============================================================
# Base loader / forward helpers
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
    x = model.drop(model.tok_emb(ids_t))
    for block in model.blocks:
        x = block(x)
    x = model.ln_f(x)
    return x


def last_token_hidden(model, query_ids_t):
    h = base_final_hidden(model, query_ids_t)
    return h[:, -1, :].squeeze(0)


def per_token_ce(model, ids_t):
    out = model(ids_t[:, :-1])
    log_probs = F.log_softmax(out, dim=-1)
    targets = ids_t[:, 1:]
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(nll.mean().item())


# ============================================================
# WikiText loader / PPL
# ============================================================

def build_wikitext_loaders(tokenizer, seed=SEED):
    from datasets import load_dataset
    train_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_raw   = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    def tokenize(split):
        ids = []
        for t in split["text"]:
            if not t.strip(): continue
            ids.extend(tokenizer.encode(t, add_special_tokens=False))
        return ids
    train_ids = tokenize(train_raw)
    val_ids   = tokenize(val_raw)
    def chunk(ids):
        return [torch.tensor(ids[i:i + WT_SEQ_LEN], dtype=torch.long)
                for i in range(0, len(ids) - WT_SEQ_LEN, WT_SEQ_LEN)]
    train_chunks = chunk(train_ids)
    val_chunks = chunk(val_ids)
    rng = random.Random(seed)
    rng.shuffle(train_chunks)
    def batch(chunks, n_batches=None):
        out = []
        for i in range(0, len(chunks) - WT_BATCH_SIZE, WT_BATCH_SIZE):
            out.append(torch.stack(chunks[i:i + WT_BATCH_SIZE]))
            if n_batches is not None and len(out) >= n_batches:
                break
        return out
    return batch(train_chunks), batch(val_chunks, n_batches=WIKITEXT_EVAL_BATCHES)


def measure_wikitext_ppl(model, val_batches, device):
    model.eval()
    set_lora_active(model, False)
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for ids in val_batches:
            ids_t = ids.to(device)
            out = model(ids_t[:, :-1])
            nll = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                  ids_t[:, 1:].reshape(-1), reduction="sum")
            total_nll += float(nll.item())
            total_tok += ids_t[:, 1:].numel()
    set_lora_active(model, True)
    return math.exp(total_nll / max(total_tok, 1))


# ============================================================
# Adapter pretraining — NTP (bootstrap step 1)
# ============================================================

def pretrain_ntp_adapter(rec, device, tokenizer, save_dir, steps=PRETRAIN_STEPS_NTP):
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"adapter_{rec['id']:04d}.pt"
    if path.exists():
        return torch.load(path, map_location=device, weights_only=False)

    base, _ = load_pristine_base(device)
    base.train()
    apply_lora(base, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(base, True)
    for n, p in base.named_parameters():
        p.requires_grad_("lora_" in n)
    lora_params = [p for n, p in base.named_parameters() if "lora_" in n]
    optim = torch.optim.AdamW(lora_params, lr=LR_LORA_PRETRAIN_NTP, weight_decay=0.0)

    text = render_record(rec)
    ids = tokenizer.encode(text, add_special_tokens=False)[:MAX_SEQ_LEN]
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(steps):
        out = base(ids_t[:, :-1])
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optim.zero_grad(); loss.backward(); optim.step()

    base.eval()
    final_ce = per_token_ce(base, ids_t)
    blob = {
        "rank": RANK, "alpha": ALPHA, "targets": LORA_TARGETS,
        "lora_state_dict": {k: v.cpu() for k, v in get_lora_state_dict(base).items()},
        "record": rec,
        "pretrain_steps": steps,
        "pretrain_mode": "ntp",
        "final_ce": final_ce,
    }
    torch.save(blob, path)
    del base, optim, lora_params
    torch.cuda.empty_cache()
    return blob


# ============================================================
# Bootstrap step 2: joint base + 5 heads on 16 bootstrap adapters
# ============================================================

def bootstrap_step_2(records_subset, adapter_blobs, device, tokenizer, cfg, steps=BOOTSTRAP_JOINT_STEPS):
    """Joint train base + 5 heads on bootstrap adapters' QA pairs. Returns
    trained heads (CPU state) + base (we discard the base; only heads are
    used downstream)."""
    if HEADS_BOOTSTRAP_PATH.exists():
        print(f"  cached bootstrap heads -> loading {HEADS_BOOTSTRAP_PATH}")
        return torch.load(HEADS_BOOTSTRAP_PATH, map_location="cpu", weights_only=False)

    base, _ = load_pristine_base(device)
    apply_lora(base, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(base, True)
    for n, p in base.named_parameters():
        p.requires_grad_("lora_" not in n)
    base_params = [p for n, p in base.named_parameters() if p.requires_grad]
    heads = make_heads(cfg["d_model"], device)
    head_params = [p for h in heads.values() for p in h.parameters()]

    optim = torch.optim.AdamW(
        [{"params": base_params, "lr": LR_BASE},
         {"params": head_params, "lr": LR_HEADS}],
        weight_decay=0.0,
    )

    # Pre-encode queries (one per field per record will be sampled per step)
    qa_pairs = build_qa_pairs(records_subset, seed=SEED + 99)
    blob_by_id = {b["record"]["id"]: b for b in adapter_blobs}

    rng = random.Random(SEED + 100)
    t0 = time.time()
    for step in range(steps):
        # Pick a random record from the bootstrap set
        rec = rng.choice(records_subset)
        blob = blob_by_id[rec["id"]]
        load_lora_state_dict(base, {k: v.to(device) for k, v in blob["lora_state_dict"].items()})
        set_lora_active(base, True)
        base.train()
        loss = 0.0
        for f in FIELDS:
            # one random QA prompt per field
            prompts = [qa for qa in qa_pairs[rec["id"]] if qa["field"] == f]
            qa = rng.choice(prompts)
            q_ids = tokenizer.encode(qa["prompt"], add_special_tokens=False)
            q_t = torch.tensor(q_ids, dtype=torch.long).unsqueeze(0).to(device)
            h = last_token_hidden(base, q_t)
            logits = heads[f](h.unsqueeze(0))
            target = torch.tensor([qa["target_idx"]], device=device)
            loss = loss + F.cross_entropy(logits, target)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params + head_params, MAX_GRAD_NORM)
        optim.step()

        if (step + 1) % max(1, steps // 10) == 0:
            elapsed = time.time() - t0
            print(f"    step {step+1:>4d}/{steps}  loss {float(loss.item()):.3f}  "
                  f"({elapsed:.0f}s elapsed)")

    # Quick train-set accuracy probe
    correct = {f: 0 for f in FIELDS}
    total = 0
    base.eval()
    with torch.no_grad():
        for rec in records_subset:
            blob = blob_by_id[rec["id"]]
            load_lora_state_dict(base, {k: v.to(device) for k, v in blob["lora_state_dict"].items()})
            set_lora_active(base, True)
            for qa in qa_pairs[rec["id"]][:20]:
                f = qa["field"]
                q_ids = tokenizer.encode(qa["prompt"], add_special_tokens=False)
                q_t = torch.tensor(q_ids, dtype=torch.long).unsqueeze(0).to(device)
                h = last_token_hidden(base, q_t)
                logits = heads[f](h.unsqueeze(0))
                if int(logits.argmax(dim=-1).item()) == qa["target_idx"]:
                    correct[f] += 1
                total += 1
    per_field = {f: correct[f] / max(total // len(FIELDS), 1) for f in FIELDS}
    overall = sum(correct.values()) / max(total, 1)
    print(f"  bootstrap step 2 done: overall train_acc {overall:.0%}  per_field {per_field}")

    state = {
        "heads_state": heads_to_cpu_state(heads),
        "d_model": cfg["d_model"],
        "overall_train_acc": overall,
        "per_field": per_field,
        "n_steps": steps,
        "n_records": len(records_subset),
    }
    torch.save(state, HEADS_BOOTSTRAP_PATH)
    print(f"  saved bootstrap heads -> {HEADS_BOOTSTRAP_PATH}")

    del base, optim
    torch.cuda.empty_cache()
    return state


# ============================================================
# Bootstrap step 3: head-CE adapter pretraining
# ============================================================

def pretrain_headce_adapter(rec, qa_for_rec, heads, device, tokenizer, save_dir,
                              steps=HEADCE_PRETRAIN_STEPS):
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"adapter_{rec['id']:04d}.pt"
    if path.exists():
        return torch.load(path, map_location=device, weights_only=False)

    base, _ = load_pristine_base(device)
    base.train()
    apply_lora(base, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(base, True)
    for n, p in base.named_parameters():
        p.requires_grad_("lora_" in n)
    lora_params = [p for n, p in base.named_parameters() if "lora_" in n]
    optim = torch.optim.AdamW(lora_params, lr=LR_LORA_PRETRAIN_HEAD, weight_decay=0.0)

    rng = random.Random(SEED + 1000 + rec["id"])
    # Pre-tokenize all prompts for this record
    prompts_t = {}
    for qa in qa_for_rec:
        prompts_t.setdefault(qa["prompt"], None)
    for p in list(prompts_t.keys()):
        ids = tokenizer.encode(p, add_special_tokens=False)
        prompts_t[p] = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(steps):
        # One random QA per field this step
        loss = 0.0
        for f in FIELDS:
            qa = rng.choice([q for q in qa_for_rec if q["field"] == f])
            q_t = prompts_t[qa["prompt"]]
            h = last_token_hidden(base, q_t)
            logits = heads[f](h.unsqueeze(0))
            target = torch.tensor([qa["target_idx"]], device=device)
            loss = loss + F.cross_entropy(logits, target)
        optim.zero_grad(); loss.backward(); optim.step()

    # Eval per-field on this record's QA pairs
    base.eval()
    correct = {f: 0 for f in FIELDS}
    seen = {f: 0 for f in FIELDS}
    with torch.no_grad():
        for qa in qa_for_rec:
            f = qa["field"]
            q_t = prompts_t[qa["prompt"]]
            h = last_token_hidden(base, q_t)
            logits = heads[f](h.unsqueeze(0))
            if int(logits.argmax(dim=-1).item()) == qa["target_idx"]:
                correct[f] += 1
            seen[f] += 1
    per_field_acc = {f: correct[f] / max(seen[f], 1) for f in FIELDS}

    blob = {
        "rank": RANK, "alpha": ALPHA, "targets": LORA_TARGETS,
        "lora_state_dict": {k: v.cpu() for k, v in get_lora_state_dict(base).items()},
        "record": rec,
        "pretrain_steps": steps,
        "pretrain_mode": "head_ce",
        "self_eval_per_field": per_field_acc,
        "self_eval_overall": sum(correct.values()) / max(sum(seen.values()), 1),
    }
    torch.save(blob, path)
    del base, optim, lora_params
    torch.cuda.empty_cache()
    return blob


# ============================================================
# Evaluation (used during main sweep)
# ============================================================

def evaluate_records(base, heads, adapter_paths_by_rid, qa_by_rec, records,
                      device, tokenizer, max_qa_per_rec=EVAL_QA_PER_RECORD):
    """Per-field accuracy across `records` using their head-CE adapters."""
    base.eval()
    correct = {f: 0 for f in FIELDS}
    seen = {f: 0 for f in FIELDS}
    with torch.no_grad():
        for rec in records:
            rid = rec["id"]
            blob = torch.load(adapter_paths_by_rid[rid], map_location=device, weights_only=False)
            load_lora_state_dict(base, {k: v.to(device) for k, v in blob["lora_state_dict"].items()})
            set_lora_active(base, True)
            pairs = qa_by_rec[rid][:max_qa_per_rec]
            for qa in pairs:
                f = qa["field"]
                q_ids = tokenizer.encode(qa["prompt"], add_special_tokens=False)
                q_t = torch.tensor(q_ids, dtype=torch.long).unsqueeze(0).to(device)
                h = last_token_hidden(base, q_t)
                logits = heads[f](h.unsqueeze(0))
                if int(logits.argmax(dim=-1).item()) == qa["target_idx"]:
                    correct[f] += 1
                seen[f] += 1
    per_field = {f: correct[f] / max(seen[f], 1) for f in FIELDS}
    overall = sum(correct.values()) / max(sum(seen.values()), 1)
    return {"per_field": per_field, "overall": overall,
             "correct": correct, "seen": seen}


def detached_ce_on_records(base, records, device, tokenizer, n=8):
    """Mean detached (adapter-off) CE on a sample of records' content.
    Larger CE = base hasn't absorbed the content."""
    base.eval()
    set_lora_active(base, False)
    vals = []
    rng = random.Random(SEED + 7)
    chosen = rng.sample(records, min(n, len(records)))
    with torch.no_grad():
        for rec in chosen:
            ids = tokenizer.encode(render_record(rec), add_special_tokens=False)[:MAX_SEQ_LEN]
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            vals.append(per_token_ce(base, ids_t))
    set_lora_active(base, True)
    return float(sum(vals) / max(len(vals), 1))


# ============================================================
# Checkpoint writer + git
# ============================================================

def write_checkpoint_readme(tag, sec, step, train_eval, heldout_eval,
                              wikitext_ppl, detached_ce, pristine_ppl, run_meta):
    path = PHASE76_DIR / f"checkpoint_{tag}_README.md"
    pf_train = train_eval["per_field"]
    pf_heldout = heldout_eval["per_field"]
    body = []
    body.append(f"# Phase 76 — Checkpoint {tag}\n")
    body.append(f"Cumulative wall time (main sweep): **{sec/3600:.2f} h**  ")
    body.append(f"Sweep steps completed: **{step:,}**  ")
    body.append(f"Total adapters: {N_RECORDS} ({N_TRAIN_RECORDS} train / {N_HELD_OUT} held-out)\n")
    body.append("\n## Headline\n")
    body.append("| field   | train acc | held-out acc |")
    body.append("|---------|-----------|---------------|")
    for f in FIELDS:
        body.append(f"| {f:<7} | {pf_train[f]:.1%}    | {pf_heldout[f]:.1%}        |")
    body.append(f"| **overall** | **{train_eval['overall']:.1%}** | **{heldout_eval['overall']:.1%}** |\n")
    body.append(f"\nWikiText-2 val PPL: **{wikitext_ppl:.3f}**  (pristine baseline: {pristine_ppl:.3f}, "
                  f"Δ {(wikitext_ppl - pristine_ppl) / pristine_ppl * 100:+.1f}%)  ")
    body.append(f"Detached mean CE on training-record content: **{detached_ce:.3f}** (absorption check; "
                  f"higher = less absorbed).\n")
    body.append("\n## Setup recap\n")
    body.append(f"- 200 synthetic patient records, 5 field types each ({', '.join(FIELDS)})")
    body.append(f"- Records tokenize to {run_meta['rec_tok_min']}–{run_meta['rec_tok_max']} tokens "
                  f"(capped at base max_seq_len={MAX_SEQ_LEN})")
    body.append(f"- Rank-{RANK} LoRA adapters on blocks 4-5 (8 target modules)")
    body.append(f"- Heads: width-{HEAD_HIDDEN} bottleneck, softmax over field vocab "
                  f"({{'name':50,'address':50,'date':50,'number':100,'entity':30}})")
    body.append(f"- Main sweep: 4 WikiText : 1 OOD; one-sided hinge regularizer on detached CE")
    body.append(f"- Heads update during sweep; adapters frozen\n")
    body.append("\n## Files\n")
    body.append(f"- `models/phase76_base_{tag}.pt` — base checkpoint (gitignored)")
    body.append(f"- `models/phase76_heads.pt` — current heads (overwritten across checkpoints)")
    body.append(f"- `models/phase76_adapters/adapter_NNNN.pt` — 200 head-CE adapters")
    body.append(f"- `results/identity_ae/phase76/sweep.json` — full numerical record\n")
    path.write_text("\n".join(body))
    return path


def run_git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd="/mnt/data/Code/HRS", timeout=120)
        return out.returncode, out.stdout, out.stderr
    except Exception as e:
        return 1, "", str(e)


def commit_and_push(message, files):
    if SMOKE:
        print(f"  [SMOKE] skipping git commit: {message}")
        return
    if not files:
        return
    # Allow PNGs that gitignore would otherwise drop
    code, _, err = run_git("add", "-f", *[str(f) for f in files])
    if code != 0:
        print(f"  git add failed: {err}")
        return
    code, out, err = run_git("commit", "-m", message)
    if code != 0:
        # might be 'nothing to commit'
        print(f"  git commit: {out.strip()} {err.strip()}")
        return
    print(f"  git commit OK")
    code, _, err = run_git("push")
    if code != 0:
        print(f"  git push skipped/failed (no remote?): {err.strip()}")
    else:
        print(f"  git push OK")


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    random.seed(SEED)
    torch.manual_seed(SEED)
    patch_lora_class()

    PHASE76_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"PHASE 76 — Scaled sweep (SMOKE={SMOKE})")
    print("=" * 72)

    # ---------- Data
    print("\n[0/4] data")
    records, qa_by_rec = ensure_data(tokenizer)
    train_records = [r for r in records if r["id"] < N_TRAIN_RECORDS]
    heldout_records = [r for r in records if r["id"] >= N_TRAIN_RECORDS]
    rec_tok_lens = []
    for rec in records[:20]:
        ids = tokenizer.encode(render_record(rec), add_special_tokens=False)
        rec_tok_lens.append(len(ids))
    rec_tok_min, rec_tok_max = min(rec_tok_lens), max(rec_tok_lens)
    print(f"  records: {len(records)} total  train {len(train_records)}  held-out {len(heldout_records)}")
    print(f"  token length (sample of 20): {rec_tok_min}–{rec_tok_max}")
    print(f"  qa pairs total: {sum(len(v) for v in qa_by_rec.values())}")

    # ---------- Bootstrap step 1
    print("\n[1/4] bootstrap step 1: NTP adapters on 16 records")
    bootstrap_records = records[:N_BOOTSTRAP_RECS]
    t0 = time.time()
    n_new = 0
    for i, rec in enumerate(bootstrap_records):
        path = BOOTSTRAP_ADAPTERS_DIR / f"adapter_{rec['id']:04d}.pt"
        if not path.exists():
            n_new += 1
        pretrain_ntp_adapter(rec, device, tokenizer, BOOTSTRAP_ADAPTERS_DIR)
        if (i + 1) % max(1, len(bootstrap_records) // 4) == 0:
            print(f"    {i+1}/{len(bootstrap_records)} adapters  "
                  f"(elapsed {time.time()-t0:.0f}s, new={n_new})")
    bootstrap_blobs = [torch.load(BOOTSTRAP_ADAPTERS_DIR / f"adapter_{rec['id']:04d}.pt",
                                    map_location=device, weights_only=False)
                       for rec in bootstrap_records]
    print(f"  {len(bootstrap_blobs)} bootstrap adapters ready ({time.time()-t0:.0f}s)")

    # ---------- Bootstrap step 2
    print("\n[2/4] bootstrap step 2: joint train base + 5 heads on 16 adapters")
    _, cfg = load_pristine_base(device)
    bootstrap_heads_state = bootstrap_step_2(bootstrap_records, bootstrap_blobs,
                                              device, tokenizer, cfg,
                                              steps=BOOTSTRAP_JOINT_STEPS)

    # ---------- Bootstrap step 3
    print("\n[3/4] bootstrap step 3: head-CE adapter pretraining on 200 records "
            f"(frozen heads)")
    heads = make_heads(cfg["d_model"], device)
    load_heads_from(heads, bootstrap_heads_state["heads_state"])
    for h in heads.values():
        for p in h.parameters():
            p.requires_grad_(False)
    t0 = time.time()
    self_accs = []
    for i, rec in enumerate(records):
        path = ADAPTERS_DIR / f"adapter_{rec['id']:04d}.pt"
        is_new = not path.exists()
        blob = pretrain_headce_adapter(rec, qa_by_rec[rec["id"]], heads,
                                        device, tokenizer, ADAPTERS_DIR,
                                        steps=HEADCE_PRETRAIN_STEPS)
        self_accs.append(blob.get("self_eval_overall", 0.0))
        if (i + 1) % max(1, len(records) // 10) == 0:
            mean_self = sum(self_accs) / len(self_accs)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(records) - i - 1)
            print(f"    {i+1}/{len(records)}  self_acc(mean) {mean_self:.1%}  "
                  f"elapsed {elapsed:.0f}s  eta {eta:.0f}s")
    mean_self = sum(self_accs) / max(len(self_accs), 1)
    print(f"  200 head-CE adapters ready ({time.time()-t0:.0f}s)  "
          f"per-adapter self-eval mean {mean_self:.1%}")

    # Save heads (these are the same as bootstrap heads — they were frozen)
    torch.save({
        "heads_state": heads_to_cpu_state(heads),
        "d_model": cfg["d_model"],
        "source": "phase76 bootstrap (heads frozen since step 2)",
        "self_acc_mean": mean_self,
    }, HEADS_PATH)
    print(f"  saved heads -> {HEADS_PATH}")

    # ---------- Main sweep
    print("\n[4/4] main sweep — fresh base, frozen adapters, heads update")
    print(f"  checkpoints at: {[t for t, _ in CKPT_TARGETS_SEC]}")
    main_sweep(records, train_records, heldout_records, qa_by_rec,
                rec_tok_min, rec_tok_max, device, tokenizer)


def main_sweep(records, train_records, heldout_records, qa_by_rec,
                rec_tok_min, rec_tok_max, device, tokenizer):
    # ---- Load fresh base
    model, cfg = load_pristine_base(device)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=LORA_TARGETS)
    set_lora_active(model, True)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" not in n)
    base_params = [p for n, p in model.named_parameters() if p.requires_grad]

    # ---- Load heads (start from bootstrap; they update during sweep)
    heads = make_heads(cfg["d_model"], device)
    heads_state = torch.load(HEADS_PATH, map_location="cpu", weights_only=False)
    load_heads_from(heads, heads_state["heads_state"])
    for h in heads.values():
        h.to(device)
        for p in h.parameters():
            p.requires_grad_(True)
    head_params = [p for h in heads.values() for p in h.parameters()]

    optim = torch.optim.AdamW(
        [{"params": base_params, "lr": LR_BASE},
         {"params": head_params, "lr": LR_HEADS}],
        weight_decay=0.0,
    )

    # ---- WikiText
    train_batches, val_batches = build_wikitext_loaders(tokenizer)
    pristine_ppl = measure_wikitext_ppl(model, val_batches, device)
    print(f"  pristine WikiText-2 val PPL: {pristine_ppl:.3f}")

    # ---- Adapter path index
    adapter_paths = {rec["id"]: ADAPTERS_DIR / f"adapter_{rec['id']:04d}.pt"
                       for rec in records}

    # ---- Pristine eval (step 0)
    print("  pristine eval (step 0):")
    set_lora_active(model, True)
    t_eval = time.time()
    train_eval_0 = evaluate_records(model, heads, adapter_paths, qa_by_rec,
                                     train_records[:40], device, tokenizer)
    heldout_eval_0 = evaluate_records(model, heads, adapter_paths, qa_by_rec,
                                       heldout_records, device, tokenizer)
    detached_ce_0 = detached_ce_on_records(model, train_records, device, tokenizer)
    print(f"    pristine train overall {train_eval_0['overall']:.1%}  "
          f"held-out overall {heldout_eval_0['overall']:.1%}  "
          f"detached CE {detached_ce_0:.3f}  ({time.time()-t_eval:.0f}s eval)")

    sweep_meta = {
        "rec_tok_min": rec_tok_min, "rec_tok_max": rec_tok_max,
        "pristine_ppl": pristine_ppl, "pristine_detached_ce": detached_ce_0,
        "pristine_train": train_eval_0, "pristine_heldout": heldout_eval_0,
    }
    trajectory = {"step": [], "elapsed_sec": [],
                   "train_overall": [], "heldout_overall": [],
                   "train_per_field": [], "heldout_per_field": [],
                   "wikitext_ppl": [], "detached_ce": []}
    trajectory["step"].append(0)
    trajectory["elapsed_sec"].append(0.0)
    trajectory["train_overall"].append(train_eval_0["overall"])
    trajectory["heldout_overall"].append(heldout_eval_0["overall"])
    trajectory["train_per_field"].append(train_eval_0["per_field"])
    trajectory["heldout_per_field"].append(heldout_eval_0["per_field"])
    trajectory["wikitext_ppl"].append(pristine_ppl)
    trajectory["detached_ce"].append(detached_ce_0)

    # ---- Loop
    rng = random.Random(SEED + 200)
    wt_iter = 0
    t_start = time.time()
    last_log = t_start
    step = 0
    next_ckpt_idx = 0
    done = False

    # Precompute training record id list for fast sampling
    train_ids = [r["id"] for r in train_records]

    while not done:
        is_ood = (step % (WIKITEXT_PER_OOD + 1) == WIKITEXT_PER_OOD)
        model.train()
        optim.zero_grad()

        if is_ood:
            rid = rng.choice(train_ids)
            blob = torch.load(adapter_paths[rid], map_location=device, weights_only=False)
            load_lora_state_dict(model, {k: v.to(device) for k, v in blob["lora_state_dict"].items()})
            set_lora_active(model, True)

            # All 5 heads, one random QA per field
            qa_pairs = qa_by_rec[rid]
            L_attached = 0.0
            for f in FIELDS:
                cand = [q for q in qa_pairs if q["field"] == f]
                qa = rng.choice(cand)
                q_ids = tokenizer.encode(qa["prompt"], add_special_tokens=False)
                q_t = torch.tensor(q_ids, dtype=torch.long).unsqueeze(0).to(device)
                h = last_token_hidden(model, q_t)
                logits = heads[f](h.unsqueeze(0))
                target = torch.tensor([qa["target_idx"]], device=device)
                L_attached = L_attached + F.cross_entropy(logits, target)

            # Hinge regularizer on detached CE — base should NOT absorb content.
            ids = tokenizer.encode(render_record(next(r for r in train_records if r["id"] == rid)),
                                     add_special_tokens=False)[:MAX_SEQ_LEN]
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            set_lora_active(model, False)
            out_det = model(ids_t[:, :-1])
            log_p = F.log_softmax(out_det, dim=-1)
            nll_det = -log_p.gather(-1, ids_t[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
            set_lora_active(model, True)
            # Anchor: pristine base CE on a similar sequence is roughly
            # `sweep_meta["pristine_detached_ce"]`. Use a fixed anchor.
            anchor = sweep_meta["pristine_detached_ce"]
            hinge = F.relu(anchor - nll_det.mean())
            L_reg = hinge
            L = L_attached + L_reg
            L.backward()
            torch.nn.utils.clip_grad_norm_(base_params + head_params, MAX_GRAD_NORM)
            optim.step()
        else:
            set_lora_active(model, False)
            wt_batch = train_batches[wt_iter % len(train_batches)].to(device)
            wt_iter += 1
            out = model(wt_batch[:, :-1])
            L = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                wt_batch[:, 1:].reshape(-1))
            set_lora_active(model, True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(base_params + head_params, MAX_GRAD_NORM)
            optim.step()

        step += 1
        now = time.time()
        elapsed = now - t_start

        # Periodic stdout
        if now - last_log >= LOG_INTERVAL_SEC:
            print(f"    step {step:>6d}  elapsed {elapsed/60:.1f}m  "
                  f"loss {float(L.item()):.3f}  is_ood {is_ood}")
            last_log = now

        # Checkpoint trigger
        if next_ckpt_idx < len(CKPT_TARGETS_SEC):
            tag, target_sec = CKPT_TARGETS_SEC[next_ckpt_idx]
            if elapsed >= target_sec:
                print(f"\n  ===== Checkpoint {tag} @ {elapsed/3600:.2f}h, step {step} =====")
                t_eval = time.time()
                set_lora_active(model, True)
                train_eval = evaluate_records(model, heads, adapter_paths, qa_by_rec,
                                                train_records, device, tokenizer)
                heldout_eval = evaluate_records(model, heads, adapter_paths, qa_by_rec,
                                                  heldout_records, device, tokenizer)
                ppl = measure_wikitext_ppl(model, val_batches, device)
                det_ce = detached_ce_on_records(model, train_records, device, tokenizer)
                eval_sec = time.time() - t_eval
                print(f"  eval done in {eval_sec:.0f}s")
                print(f"    train   overall {train_eval['overall']:.1%}  per_field {train_eval['per_field']}")
                print(f"    heldout overall {heldout_eval['overall']:.1%}  per_field {heldout_eval['per_field']}")
                print(f"    wikitext_ppl {ppl:.3f}  detached_ce {det_ce:.3f}")

                trajectory["step"].append(step)
                trajectory["elapsed_sec"].append(elapsed)
                trajectory["train_overall"].append(train_eval["overall"])
                trajectory["heldout_overall"].append(heldout_eval["overall"])
                trajectory["train_per_field"].append(train_eval["per_field"])
                trajectory["heldout_per_field"].append(heldout_eval["per_field"])
                trajectory["wikitext_ppl"].append(ppl)
                trajectory["detached_ce"].append(det_ce)

                # Save base checkpoint
                base_ckpt_path = MODELS_DIR / f"phase76_base_{tag}.pt"
                base_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" not in k}
                torch.save({"model_state_dict": base_state, "config": cfg,
                            "step": step, "elapsed_sec": elapsed,
                            "tag": tag}, base_ckpt_path)
                # Overwrite heads
                torch.save({"heads_state": heads_to_cpu_state(heads),
                             "d_model": cfg["d_model"],
                             "source": f"phase76 main sweep @ {tag}"}, HEADS_PATH)
                # Restore heads to GPU
                for f, h in heads.items():
                    h.to(device)

                # Write README
                readme_path = write_checkpoint_readme(
                    tag, elapsed, step, train_eval, heldout_eval,
                    ppl, det_ce, sweep_meta["pristine_ppl"], sweep_meta)
                # Update sweep.json
                write_sweep_json(sweep_meta, trajectory)
                # Git commit
                commit_and_push(
                    f"Phase 76 checkpoint {tag}: step {step}, elapsed {elapsed/3600:.2f}h",
                    [PHASE76_DIR, RECORDS_JSON, QA_JSON],
                )
                next_ckpt_idx += 1
                if next_ckpt_idx >= len(CKPT_TARGETS_SEC):
                    done = True
                t_start_extra = time.time() - t_eval
                # Continue the same wall clock — checkpoint targets are absolute
                last_log = time.time()

    # ---- Final summary
    print("\n  main sweep complete. writing final README + trajectory plot")
    write_final_readme(trajectory, sweep_meta)
    plot_trajectory(trajectory, sweep_meta)
    commit_and_push("Phase 76 final: trajectory + final README",
                    [PHASE76_DIR / "final_README.md",
                     TRAJ_PNG, SWEEP_JSON])


def write_sweep_json(sweep_meta, trajectory):
    SWEEP_JSON.write_text(json.dumps({
        "config": {
            "n_records": N_RECORDS, "n_train": N_TRAIN_RECORDS,
            "n_held_out": N_HELD_OUT, "n_bootstrap": N_BOOTSTRAP_RECS,
            "qa_per_record": QA_PER_RECORD, "rank": RANK, "alpha": ALPHA,
            "head_hidden": HEAD_HIDDEN, "lr_base": LR_BASE, "lr_heads": LR_HEADS,
            "wikitext_per_ood": WIKITEXT_PER_OOD,
            "wt_batch_size": WT_BATCH_SIZE, "wt_seq_len": WT_SEQ_LEN,
            "lora_targets": LORA_TARGETS,
            "ckpt_targets_sec": CKPT_TARGETS_SEC,
            "smoke": SMOKE, "seed": SEED,
        },
        "sweep_meta": sweep_meta,
        "trajectory": trajectory,
    }, indent=2))


def write_final_readme(trajectory, sweep_meta):
    pristine_ppl = sweep_meta["pristine_ppl"]
    steps = trajectory["step"]
    heldout_overall = trajectory["heldout_overall"]
    train_overall = trajectory["train_overall"]
    monotonic = all(heldout_overall[i] <= heldout_overall[i + 1]
                     for i in range(1, len(heldout_overall) - 1))
    final_heldout = heldout_overall[-1]
    alive = monotonic and final_heldout > 0.70
    body = []
    body.append("# Phase 76 — Scaled sweep: full trajectory\n")
    body.append(f"**Alive verdict:** {'YES' if alive else 'NO'} (monotonic={monotonic}, "
                  f"final held-out={final_heldout:.1%}; needs monotonic improvement AND >70%)\n")
    body.append("\n## Trajectory across checkpoints\n")
    body.append("| checkpoint | elapsed (h) | step | train acc | held-out acc | WikiText PPL | detached CE |")
    body.append("|------------|--------------|------|------------|----------------|---------------|--------------|")
    tags = ["pristine"] + [t for t, _ in CKPT_TARGETS_SEC[:len(trajectory["step"]) - 1]]
    for i, tag in enumerate(tags):
        body.append(f"| {tag} | {trajectory['elapsed_sec'][i]/3600:.2f} | "
                      f"{trajectory['step'][i]:,} | "
                      f"{trajectory['train_overall'][i]:.1%} | "
                      f"{trajectory['heldout_overall'][i]:.1%} | "
                      f"{trajectory['wikitext_ppl'][i]:.2f} | "
                      f"{trajectory['detached_ce'][i]:.2f} |")
    body.append("\n## Per-field at final checkpoint\n")
    body.append("| field   | train | held-out |")
    body.append("|---------|-------|----------|")
    pf_train = trajectory["train_per_field"][-1]
    pf_held  = trajectory["heldout_per_field"][-1]
    for f in FIELDS:
        body.append(f"| {f:<7} | {pf_train[f]:.1%} | {pf_held[f]:.1%} |")
    body.append("\n## Reading the result\n")
    body.append("- **Held-out tracks training**: architectural mechanism transfers across adapters.")
    body.append("- **Held-out diverges from training**: per-adapter learning only; mechanism is dead.")
    body.append("- **Both flat / both degrade**: capacity exhausted, or content too large for rank 128.")
    body.append("- **Train high, held-out low**: heads + base over-fit the 160 training adapters.\n")
    body.append("\nWikiText preservation is secondary. The architectural metric is the held-out trajectory.\n")
    body.append(f"\n## Files\n- `sweep.json` — full numerical record")
    body.append("- `trajectory.png` — per-checkpoint plot")
    body.append("- `checkpoint_{1h,4h,16h}_README.md` — per-checkpoint READMEs")
    (PHASE76_DIR / "final_README.md").write_text("\n".join(body))


def plot_trajectory(trajectory, sweep_meta):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = [s / 3600 for s in trajectory["elapsed_sec"]]
    axes[0].plot(x, trajectory["train_overall"], "C2-o", label="train", markersize=5)
    axes[0].plot(x, trajectory["heldout_overall"], "C3-^", label="held-out", markersize=5)
    axes[0].axhline(0.70, color="gray", ls=":", lw=0.7, label="alive threshold")
    axes[0].set_xlabel("wall time (h)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Per-field accuracy (overall)")
    axes[0].set_ylim(-0.02, 1.05); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(x, trajectory["wikitext_ppl"], "C0-o", markersize=5)
    axes[1].axhline(sweep_meta["pristine_ppl"], color="gray", ls="--", lw=0.7,
                     label=f"pristine ({sweep_meta['pristine_ppl']:.2f})")
    axes[1].set_xlabel("wall time (h)")
    axes[1].set_ylabel("WikiText-2 val PPL")
    axes[1].set_title("General competence")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].plot(x, trajectory["detached_ce"], "C1-o", markersize=5)
    axes[2].axhline(sweep_meta["pristine_detached_ce"], color="gray", ls="--", lw=0.7,
                     label=f"pristine ({sweep_meta['pristine_detached_ce']:.2f})")
    axes[2].set_xlabel("wall time (h)")
    axes[2].set_ylabel("detached CE on training content")
    axes[2].set_title("Absorption check (higher = less absorbed)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAJ_PNG, dpi=120)
    print(f"  saved {TRAJ_PNG}")


if __name__ == "__main__":
    main()
