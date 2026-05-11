"""Phase 69 — Adapter verification stage.

Phase 65 established that template-overlapping near-neighbor OOD queries route
to the wrong adapter at ~92% FP and that no routing-layer fix recovers this.
Phase 69 tests whether a verification stage downstream of routing — using
confidence signals on the adapter's generation, plus closed-vocab entity
matching on the generation text — can catch the wrong-adapter cases at
deployment-grade FP rates.

Procedure per query:
  1. Route via C0b (L0 cosine, no projection) — Phase 65's recommended gate.
  2. Load the routed adapter.
  3. Generate 50 tokens greedy. Track per-step confidence signals.
  4. Compute query-side NLL under the loaded adapter; same under base (LoRA
     reset). Their delta is a derived signal.
  5. Extract the query's entity (closed vocab); extract any entity in the
     generation text. entity_match = (gen_entity == query_entity).

Signals computed (eight total):
  gen_nll                 mean -log p(generated tokens)
  gen_entropy_mean        mean per-token next-token entropy during generation
  gen_max_logit_mean      mean per-token max logit during generation
  gen_token_min_prob      min per-token probability over the generation
  query_nll_under_adapter mean -log p of query tokens under loaded adapter
  query_nll_under_base    mean -log p of query tokens under base (LoRA reset)
  delta_query_nll         adapter - base (negative = adapter helps)
  entity_match            binary, closed-vocab match

Verification target ("should this answer be trusted?"):
  TRUST          — in-library query AND routed to its true adapter
  DON'T-TRUST    — hard-OOD query, OR in-library query routed to wrong adapter

For each signal, report distribution by class, separation, and FP at 95%
in-library (TRUST) accept threshold.

Reuses Phase 65 cache; no LoRA retraining. Model in eval mode throughout.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase69_verification.py
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.phase10_passkey import (
    load_model, generate_passkeys,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import apply_lora, load_lora_state_dict


SEED = 0
GEN_TOKENS = 100  # was 50; longer gen gives the LoRA adapter more chance to regurgitate entity
PHASE65_CACHE = Path("results/identity_ae/phase65/cache")
PHASE69_DIR = Path("results/identity_ae/phase69")


# ---- Closed-vocab entity sets per category (matches generate_passkeys) ----
ENTITY_VOCAB = {
    "numeric":   ["northern", "southern", "eastern", "western", "central",
                  "orbital", "coastal", "highland", "basement", "rooftop"],
    "entity":    ["Dr. Elara Voss", "Professor Kian Nakamura", "Agent Sarah Thornhill",
                  "Commander Yuki Petrov", "Director Ravi Blackwood",
                  "Specialist Anya Morales", "Operative Chen Volkov",
                  "Researcher Fatima Okonkwo", "Engineer Dmitri Svensson",
                  "Analyst Priya Gutierrez"],
    "technical": ["reactor", "accelerator", "telescope", "spectrometer", "collider",
                  "centrifuge", "cryostat", "magnetron", "synchrotron", "interferometer"],
    "fact":      ["Thornfield", "Blackwater", "Meridian", "Vanguard", "Eclipse",
                  "Harbinger", "Sentinel", "Obsidian", "Crimson", "Phantom"],
}


def extract_entity(text: str, category: str):
    """Closed-vocab entity extraction — case-sensitive for proper nouns,
    case-insensitive for common nouns. Returns the matching entity string or None."""
    vocab = ENTITY_VOCAB[category]
    text_lower = text.lower()
    for ent in vocab:
        # Names and protocols: case-sensitive (proper nouns)
        if category in ("entity", "fact"):
            if ent in text:
                return ent
        else:
            if ent in text_lower:
                return ent
    return None


# ---- Routing helpers ----

@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


def cosine(a, b):
    return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))


def best_match_l0(q, keys_per_adapter):
    best_a, best_s = -1, -2.0
    for ai, keys in enumerate(keys_per_adapter):
        for k in keys:
            s = cosine(q, k)
            if s > best_s:
                best_s, best_a = s, ai
    return best_a, best_s


# ---- Generation with per-step signals ----

@torch.no_grad()
def generate_with_signals(model, prompt, tokenizer, device, n_tokens=GEN_TOKENS):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    nlls, entropies, max_logits, probs = [], [], [], []
    for _ in range(n_tokens):
        idx = input_ids[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :]                            # (1, V)
        log_probs = F.log_softmax(logits, dim=-1)
        ps = log_probs.exp()
        next_tok = logits.argmax(dim=-1, keepdim=True)            # (1, 1)
        tid = int(next_tok.item())
        nlls.append(float(-log_probs[0, tid].item()))
        entropies.append(float(-(ps * log_probs).sum(dim=-1).item()))
        max_logits.append(float(logits.max(dim=-1).values.item()))
        probs.append(float(ps[0, tid].item()))
        input_ids = torch.cat([input_ids, next_tok], dim=1)
    text = tokenizer.decode(input_ids[0, len(ids):], skip_special_tokens=True)
    return {
        "text": text,
        "gen_nll":             sum(nlls) / len(nlls),
        "gen_entropy_mean":    sum(entropies) / len(entropies),
        "gen_max_logit_mean":  sum(max_logits) / len(max_logits),
        "gen_token_min_prob":  min(probs),
    }


@torch.no_grad()
def query_nll(model, ids_t):
    """Mean NLL of next-token prediction over query tokens."""
    if ids_t.shape[1] < 2:
        return float("nan")
    out = model(ids_t[:, :-1], step=0)
    log_probs = F.log_softmax(out.logits, dim=-1)
    targets = ids_t[:, 1:]
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(nll.mean().item())


# ---- Hard-OOD construction ----

def build_hard_ood_tests():
    all_tests = generate_passkeys(50)
    by_type = defaultdict(list)
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][5:10] + by_type["entity"][5:10]
            + by_type["technical"][5:10] + by_type["fact"][5:10])


# ---- Per-signal calibration ----

def quantile(xs, q):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    pos = q * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def fp_at_accept_rate(trust_scores, reject_scores, accept_rate, higher_means_trust=True):
    """At the threshold where `accept_rate` of TRUST queries pass, what fraction of
    DON'T-TRUST queries also pass? higher_means_trust controls the direction."""
    if higher_means_trust:
        # Accept if score > tau. tau = (1-accept_rate)-th quantile of trust.
        tau = quantile(trust_scores, 1 - accept_rate)
        fp = sum(1 for s in reject_scores if s > tau) / max(len(reject_scores), 1)
    else:
        # Accept if score < tau. tau = accept_rate-th quantile of trust.
        tau = quantile(trust_scores, accept_rate)
        fp = sum(1 for s in reject_scores if s < tau) / max(len(reject_scores), 1)
    return tau, fp


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE69_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load Phase 65 cache ----
    print(f"[A] loading Phase 65 cache from {PHASE65_CACHE}")
    blob = torch.load(PHASE65_CACHE / "library.pt", map_location="cpu", weights_only=False)
    library  = blob["library"]
    keys_l0  = blob["keys_l0"]
    keys_l0_cpu = [[k.cpu() if k.is_cuda else k for k in ks] for ks in keys_l0]
    print(f"    {len(library)} adapters, {sum(len(k) for k in keys_l0)} L0 keys")

    # ---- Build query sets: 60 in-library + 20 hard-OOD = 80 queries ----
    in_records = []
    for i, entry in enumerate(library):
        for para in held_out_paraphrase(entry["test"]):
            in_records.append({
                "set": "in_library",
                "true_idx": i,
                "category": entry["test"]["type"],
                "query": para,
                "query_entity_truth": extract_entity(para, entry["test"]["type"]),
            })

    hard_ood = build_hard_ood_tests()
    ood_records = []
    for ood_test in hard_ood:
        ood_records.append({
            "set": "hard_ood",
            "true_idx": -1,
            "category": ood_test["type"],
            "query": ood_test["prompt"],
            "query_entity_truth": extract_entity(ood_test["prompt"], ood_test["type"]),
        })

    print(f"    in-library queries: {len(in_records)}, hard-OOD queries: {len(ood_records)}")
    queries = in_records + ood_records

    # ---- Build base model in eval mode ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    model.eval()
    reset_lora_to_zero(model)

    # ---- Process each query ----
    print(f"\n[B] processing {len(queries)} queries (route + generate + signals)")
    t0 = time.time()
    results = []
    for qi, q in enumerate(queries):
        # Step 1: Route via C0b
        reset_lora_to_zero(model)
        ids = tokenizer.encode(q["query"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q_l0 = l0_mean(model, ids_t)
        routed_idx, routing_score = best_match_l0(q_l0, keys_l0_cpu)

        # Step 5 (query NLL under base, before loading adapter)
        q_nll_base = query_nll(model, ids_t)

        # Step 2: Load routed adapter
        sd = library[routed_idx]["sd"]
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)

        # Step 3: Generate with signals
        gen = generate_with_signals(model, q["query"], tokenizer, device,
                                     n_tokens=GEN_TOKENS)

        # Step 4: Query NLL under adapter
        q_nll_adapter = query_nll(model, ids_t)
        delta_q_nll = q_nll_adapter - q_nll_base

        # Step 6: Entity extraction from generation
        gen_entity = extract_entity(gen["text"], q["category"])
        entity_match = int(gen_entity is not None and
                            gen_entity == q["query_entity_truth"])

        # Step 6b: Metadata entity check — extract the routed adapter's training
        # entity from its prompt and compare to the query entity. This is a free
        # routing-time signal that doesn't require generation.
        routed_prompt = library[routed_idx]["test"]["prompt"]
        routed_entity = extract_entity(routed_prompt, q["category"])
        metadata_entity_match = int(routed_entity is not None and
                                     routed_entity == q["query_entity_truth"])

        # Verification target label
        if q["set"] == "in_library":
            label = "TRUST" if routed_idx == q["true_idx"] else "DONT_TRUST_misroute"
        else:
            label = "DONT_TRUST_ood"

        results.append({
            "qi": qi,
            "set": q["set"], "category": q["category"],
            "query": q["query"], "query_entity": q["query_entity_truth"],
            "true_idx": q["true_idx"], "routed_idx": routed_idx,
            "routed_entity": routed_entity,
            "routing_score": routing_score,
            "gen_text": gen["text"],
            "gen_entity": gen_entity, "entity_match": entity_match,
            "metadata_entity_match": metadata_entity_match,
            "label": label,
            "signals": {
                "gen_nll":              gen["gen_nll"],
                "gen_entropy_mean":     gen["gen_entropy_mean"],
                "gen_max_logit_mean":   gen["gen_max_logit_mean"],
                "gen_token_min_prob":   gen["gen_token_min_prob"],
                "query_nll_adapter":    q_nll_adapter,
                "query_nll_base":       q_nll_base,
                "delta_query_nll":      delta_q_nll,
            },
        })

        if (qi + 1) % 10 == 0:
            dt = time.time() - t0
            eta = dt / (qi + 1) * (len(queries) - qi - 1)
            print(f"    [{qi+1:2d}/{len(queries)}]  elapsed {dt:.0f}s  eta {eta:.0f}s")

    # ---- Group by label ----
    trust = [r for r in results if r["label"] == "TRUST"]
    dont_misroute = [r for r in results if r["label"] == "DONT_TRUST_misroute"]
    dont_ood = [r for r in results if r["label"] == "DONT_TRUST_ood"]
    dont = dont_misroute + dont_ood
    print(f"\n[C] verification labels: TRUST={len(trust)}  "
          f"DONT_TRUST_misroute={len(dont_misroute)}  DONT_TRUST_ood={len(dont_ood)}")

    # ---- Per-signal analysis ----
    SIGNAL_DIRECTIONS = {
        # name: higher_means_trust (True/False)
        "gen_nll":              False,   # confident gen -> low NLL -> trust
        "gen_entropy_mean":     False,   # confident -> low entropy
        "gen_max_logit_mean":   True,    # confident -> high max logit
        "gen_token_min_prob":   True,    # confident -> high min prob
        "query_nll_adapter":    False,   # adapter-fits-query -> low NLL
        "query_nll_base":       False,   # (reference)
        "delta_query_nll":      False,   # negative delta = adapter helps
    }

    print(f"\n[D] per-signal verification analysis (FP at 95% TRUST accept):")
    print(f"  {'signal':<24s} {'TRUST mean':>10s} {'DONT mean':>10s} {'sep':>8s} "
          f"{'FP@95':>6s} {'tau':>10s}")
    signal_summaries = {}
    for sig, higher_means_trust in SIGNAL_DIRECTIONS.items():
        trust_scores = [r["signals"][sig] for r in trust]
        dont_scores  = [r["signals"][sig] for r in dont]
        if not trust_scores or not dont_scores:
            continue
        t_mean = sum(trust_scores) / len(trust_scores)
        d_mean = sum(dont_scores)  / len(dont_scores)
        sep = t_mean - d_mean if higher_means_trust else d_mean - t_mean
        tau, fp = fp_at_accept_rate(trust_scores, dont_scores, 0.95, higher_means_trust)
        print(f"  {sig:<24s} {t_mean:>+10.3f} {d_mean:>+10.3f} {sep:>+8.3f} "
              f"{fp:>6.0%} {tau:>+10.3f}")
        signal_summaries[sig] = {
            "higher_means_trust": higher_means_trust,
            "trust_mean": t_mean, "trust_std": (sum((s - t_mean)**2 for s in trust_scores) / len(trust_scores))**0.5,
            "dont_mean":  d_mean, "dont_std":  (sum((s - d_mean)**2 for s in dont_scores) / len(dont_scores))**0.5,
            "separation": sep,
            "tau_at_95_accept": tau,
            "fp_at_95_accept":  fp,
            "trust_scores": trust_scores,
            "dont_scores":  dont_scores,
        }

    # ---- Entity-match as a binary classifier ----
    em_trust = sum(r["entity_match"] for r in trust) / max(len(trust), 1)
    em_dont  = sum(r["entity_match"] for r in dont)  / max(len(dont), 1)
    em_tp = sum(r["entity_match"] for r in trust)
    em_fn = len(trust) - em_tp
    em_fp = sum(r["entity_match"] for r in dont)
    em_tn = len(dont) - em_fp
    em_precision = em_tp / max(em_tp + em_fp, 1)
    em_recall = em_tp / max(em_tp + em_fn, 1)
    em_specificity = em_tn / max(em_tn + em_fp, 1)
    em_balanced_acc = (em_recall + em_specificity) / 2

    print(f"\n[E] entity_match (gen-based) as a binary verification decision:")
    print(f"    TRUST queries: entity_match=1 in {em_tp}/{len(trust)} ({em_trust:.0%})")
    print(f"    DONT_TRUST queries: entity_match=1 in {em_fp}/{len(dont)} ({em_dont:.0%})")
    print(f"    confusion: TP={em_tp}  FN={em_fn}  FP={em_fp}  TN={em_tn}")
    print(f"    precision={em_precision:.0%}  recall={em_recall:.0%}  "
          f"specificity={em_specificity:.0%}  balanced_acc={em_balanced_acc:.0%}")

    # ---- metadata_entity_match (no generation required) ----
    mm_trust = sum(r["metadata_entity_match"] for r in trust) / max(len(trust), 1)
    mm_dont  = sum(r["metadata_entity_match"] for r in dont)  / max(len(dont), 1)
    mm_tp = sum(r["metadata_entity_match"] for r in trust)
    mm_fn = len(trust) - mm_tp
    mm_fp = sum(r["metadata_entity_match"] for r in dont)
    mm_tn = len(dont) - mm_fp
    mm_precision   = mm_tp / max(mm_tp + mm_fp, 1)
    mm_recall      = mm_tp / max(mm_tp + mm_fn, 1)
    mm_specificity = mm_tn / max(mm_tn + mm_fp, 1)
    mm_balanced    = (mm_recall + mm_specificity) / 2
    print(f"\n[E2] metadata_entity_match (free routing-time signal):")
    print(f"    TRUST queries: match=1 in {mm_tp}/{len(trust)} ({mm_trust:.0%})")
    print(f"    DONT_TRUST queries: match=1 in {mm_fp}/{len(dont)} ({mm_dont:.0%})")
    print(f"    confusion: TP={mm_tp}  FN={mm_fn}  FP={mm_fp}  TN={mm_tn}")
    print(f"    precision={mm_precision:.0%}  recall={mm_recall:.0%}  "
          f"specificity={mm_specificity:.0%}  balanced_acc={mm_balanced:.0%}")

    # ---- Generation samples for sanity ----
    print(f"\n[F] sample generations (3 per label):")
    for label, group in [("TRUST", trust), ("DONT_TRUST_misroute", dont_misroute), ("DONT_TRUST_ood", dont_ood)]:
        print(f"  --- {label} ---")
        for r in group[:3]:
            true_lib_prompt = library[r["routed_idx"]]["test"]["prompt"]
            print(f"    query: {r['query']!r}")
            print(f"    routed -> [{r['routed_idx']}] {true_lib_prompt!r}")
            print(f"    gen:   {r['gen_text']!r}")
            print(f"    query_ent={r['query_entity']!r}  gen_ent={r['gen_entity']!r}  "
                  f"match={r['entity_match']}  gen_nll={r['signals']['gen_nll']:.3f}  "
                  f"delta_qnll={r['signals']['delta_query_nll']:+.3f}")
            print()

    # ---- Save JSON ----
    out_path = PHASE69_DIR / "verification.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"gen_tokens": GEN_TOKENS, "seed": SEED,
                       "n_in_library": len(in_records), "n_hard_ood": len(ood_records)},
            "label_counts": {"trust": len(trust),
                             "dont_trust_misroute": len(dont_misroute),
                             "dont_trust_ood": len(dont_ood)},
            "signal_summaries": {k: {kk: vv for kk, vv in v.items()
                                       if kk not in ("trust_scores", "dont_scores")}
                                  for k, v in signal_summaries.items()},
            "entity_match_gen": {
                "trust_match_rate": em_trust, "dont_match_rate": em_dont,
                "tp": em_tp, "fn": em_fn, "fp": em_fp, "tn": em_tn,
                "precision": em_precision, "recall": em_recall,
                "specificity": em_specificity, "balanced_acc": em_balanced_acc,
            },
            "metadata_entity_match": {
                "trust_match_rate": mm_trust, "dont_match_rate": mm_dont,
                "tp": mm_tp, "fn": mm_fn, "fp": mm_fp, "tn": mm_tn,
                "precision": mm_precision, "recall": mm_recall,
                "specificity": mm_specificity, "balanced_acc": mm_balanced,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot per-signal histograms ----
    sigs_to_plot = ["gen_nll", "gen_entropy_mean", "gen_max_logit_mean",
                    "gen_token_min_prob", "query_nll_adapter", "delta_query_nll"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, sig in zip(axes.flat, sigs_to_plot):
        s = signal_summaries[sig]
        all_scores = s["trust_scores"] + s["dont_scores"]
        lo, hi = min(all_scores), max(all_scores)
        bins = [lo + i * (hi - lo) / 30 for i in range(31)]
        ax.hist(s["trust_scores"], bins=bins, alpha=0.6, color="C2",
                label=f"TRUST (n={len(s['trust_scores'])})")
        ax.hist(s["dont_scores"],  bins=bins, alpha=0.6, color="C3",
                label=f"DONT (n={len(s['dont_scores'])})")
        ax.axvline(s["tau_at_95_accept"], color="black", ls="--", lw=1)
        ax.set_title(f"{sig}\nFP@95 = {s['fp_at_95_accept']:.0%}, sep = {s['separation']:+.3f}",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plot_path = PHASE69_DIR / "signal_distributions.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict ----
    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    best_sig, best_fp = None, 1.01
    for sig, s in signal_summaries.items():
        if s["fp_at_95_accept"] < best_fp:
            best_fp = s["fp_at_95_accept"]
            best_sig = sig
    print(f"  best confidence signal: {best_sig} (FP@95 = {best_fp:.0%})")
    print(f"  entity_match (gen-based): balanced_acc = {em_balanced_acc:.0%}, "
          f"specificity = {em_specificity:.0%}, recall = {em_recall:.0%}")
    print(f"  metadata_entity_match:    balanced_acc = {mm_balanced:.0%}, "
          f"specificity = {mm_specificity:.0%}, recall = {mm_recall:.0%}")

    if mm_specificity >= 0.95 and mm_recall >= 0.95:
        print(f"\n  METADATA CHECK SUFFICES: routing-time string match against the "
              f"routed adapter's training entity gives {mm_balanced:.0%} balanced "
              f"accuracy. No generation needed; deployment-grade verification at "
              f"~zero added cost.")
    elif best_fp <= 0.10:
        print(f"\n  CHEAP CONFIDENCE FIX: {best_sig} alone gives FP {best_fp:.0%} at 95% accept.")
    elif em_specificity >= 0.90 and em_recall >= 0.90:
        print(f"\n  ENTITY-MATCH (gen) WORKS: balanced_acc {em_balanced_acc:.0%}. "
              f"Verification recipe: extract entity from generation, compare to query.")
    elif em_specificity == 1.00 and em_recall >= 0.50:
        print(f"\n  ENTITY-MATCH (gen) IS A ONE-SIDED FILTER: 100% specificity but "
              f"only {em_recall:.0%} recall. Use as 'soft accept' (em=1 -> trust); "
              f"em=0 cases need a second signal.")
    elif best_fp <= 0.30 or em_specificity >= 0.70:
        print(f"\n  PARTIAL: best signal FP {best_fp:.0%}, entity-match specificity "
              f"{em_specificity:.0%}. Combine signals (stretch goal) or move to Phase 70.")
    else:
        print(f"\n  CONFIDENCE-BASED VERIFICATION FAILS: best FP {best_fp:.0%}, "
              f"entity-match specificity {em_specificity:.0%}. Move to Phase 70 "
              f"(per-adapter classifier head).")


if __name__ == "__main__":
    main()
