"""Phase 48: KL-divergence regularized LoRA absorption.

Hypothesis: the frozen base model's output distribution is a compiled
expectation of what attention-plus-MLP should produce on in-distribution
content. Adding a KL term between the LoRA-augmented model's output and
the frozen base model's output should regularize absorption: the adapter
can still learn the new passkey (the NTP loss still wants it), but on
every other token the KL pulls the model back toward the base program.

In effect this is standard knowledge-distillation anchoring applied at
TTT time, with the teacher being the frozen pre-LoRA model. The spec
discusses a layer-local variant (compare layer-4/5 block outputs), but
since the LoRA only sits at layers 4 and 5, the LoRA's effect on any
output is already fully captured by the final logit distribution. The
output-head KL is the cleanest form of that same signal.

Loss:
    total = NTP(LoRA_model)  +  alpha * KL(LoRA_logits || base_logits)

Sweep alpha in {0.0, 0.01, 0.05, 0.1, 0.5, 1.0}. alpha=0.0 is the
Phase 14 / Phase 38b unregularized baseline.

For each alpha, measure:
  1. Same-prompt passkey retrieval over the 20-passage stratified benchmark.
  2. WikiText validation perplexity drift AFTER absorbing all 20 passages,
     against the base model baseline. (LoRA is reset between passages, so
     the drift measured is the residual per-adapter signature — Phase 14
     is already near zero; we just want to see whether alpha changes that.)
  3. Per-step KL trajectory on the first passage, so we can look for a
     separation between passage tokens and passkey tokens.

Secondary hypothesis (novelty detection): KL divergence between a LoRA-
modified attention output and the MLP's expectation is high precisely
on tokens the base model cannot predict. The recorded trajectory is
enough to check this for the single-passage case.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase48_kl_mlp_regularizer.py
"""

import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR, val_ppl_ungated
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict,
)


RANK = 128
ALPHA_LORA = 256
N_STEPS = 150
GEN_TOKENS = 50
ALPHAS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
KL_TEMPERATURE = 2.0


def train_with_kl(lora_model, base_model, passage, tokenizer, device,
                  alpha_kl, n_steps=N_STEPS,
                  high_lr=HIGH_LR, base_lr=BASE_LR,
                  record_trajectory=False):
    """Train LoRA on a passage with KL-to-base regularization.

    If record_trajectory is True, returns a list of per-step dicts:
        {"step": int, "ntp": float, "kl": float, "kl_per_token": [float]}
    The per-token KL list is recorded once at the final step so the
    passage-vs-passkey separation can be inspected.
    """
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    if ids_t.shape[1] < 2:
        return []

    params = [p for n, p in lora_model.named_parameters()
              if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    trajectory = []

    lora_model.train()
    base_model.eval()
    inputs = ids_t[:, :-1]
    targets = ids_t[:, 1:]
    B, T = targets.shape

    # Precompute base logits once (base model is frozen, its output
    # is identical every step — saves one forward pass per step).
    with torch.no_grad():
        base_out = base_model(inputs, step=0)
        base_logits_full = base_out.logits  # (1, T, V)

    for step in range(n_steps):
        out = lora_model(inputs, step=0)
        lora_logits = out.logits  # (1, T, V)
        V = lora_logits.shape[-1]

        ntp = F.cross_entropy(
            lora_logits.reshape(-1, V),
            targets.reshape(-1),
        )

        if alpha_kl > 0:
            lora_log = F.log_softmax(lora_logits / KL_TEMPERATURE, dim=-1)
            base_prob = F.softmax(base_logits_full / KL_TEMPERATURE, dim=-1)
            # KL(lora || base): "don't move the distribution away from base"
            # F.kl_div expects input=log Q, target=P, computes sum P*(log P - log Q)
            # batchmean divides by batch*seq, giving mean per-token KL.
            kl = F.kl_div(lora_log, base_prob, reduction='batchmean') \
                 * (KL_TEMPERATURE ** 2)
        else:
            kl = torch.tensor(0.0, device=device)

        loss = ntp + alpha_kl * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if record_trajectory:
            entry = {"step": step, "ntp": float(ntp.item()),
                     "kl": float(kl.item())}
            if step == n_steps - 1:
                with torch.no_grad():
                    lora_log_t = F.log_softmax(
                        lora_logits / KL_TEMPERATURE, dim=-1)
                    base_prob_t = F.softmax(
                        base_logits_full / KL_TEMPERATURE, dim=-1)
                    # per-token KL: sum over vocab
                    per_tok = (base_prob_t * (
                        base_prob_t.clamp_min(1e-12).log() - lora_log_t
                    )).sum(dim=-1).squeeze(0) * (KL_TEMPERATURE ** 2)
                    entry["kl_per_token"] = [float(v) for v in per_tok.tolist()]
            trajectory.append(entry)

    lora_model.eval()
    return trajectory


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase48")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()
    print(f"Stratified: {len(tests)} passages\n")

    # ============================================================
    # Load both models: trainable LoRA copy and frozen base reference
    # ============================================================
    print("Loading models...")
    lora_model, cfg = load_model(device)
    apply_lora(lora_model, rank=RANK, alpha=ALPHA_LORA,
               target_modules=L45_TARGETS)

    base_model, _ = load_model(device)  # no LoRA applied
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    # Validation loader for drift measurement
    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    # Baseline PPL (base model, no adapter loaded)
    reset_lora_to_zero(lora_model)
    baseline_ppl = val_ppl_ungated(lora_model, loaders["validation"], device)
    print(f"Baseline WikiText val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # Alpha sweep
    # ============================================================
    all_results = {}

    for alpha_kl in ALPHAS:
        print("=" * 64)
        print(f"ALPHA_KL = {alpha_kl}")
        print("=" * 64)

        n_correct = 0
        per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        per_passage = []
        trajectories = {}
        t0 = time.time()

        for ti, test in enumerate(tests):
            reset_lora_to_zero(lora_model)
            record = (ti == 0)  # trajectory on first passage only
            traj = train_with_kl(
                lora_model, base_model, test["passage"],
                tokenizer, device,
                alpha_kl=alpha_kl, n_steps=N_STEPS,
                high_lr=HIGH_LR, base_lr=BASE_LR,
                record_trajectory=record,
            )
            if record:
                trajectories[ti] = {
                    "test": dict(test),
                    "trajectory": traj,
                }

            gen = generate_greedy(lora_model, test["prompt"],
                                  tokenizer, device, GEN_TOKENS)
            hit = check_passkey(gen, test["passkey"])
            if hit:
                n_correct += 1
                per_type[test["type"]] += 1
            per_passage.append({
                "id": test["id"], "type": test["type"],
                "passkey": test["passkey"], "hit": hit,
                "gen": gen[:120],
            })

        elapsed = time.time() - t0

        # Measure drift with the LAST-trained adapter still loaded.
        # This tells us how far the model's distribution has moved with
        # a single adapter active; not cumulative forgetting.
        with_adapter_ppl = val_ppl_ungated(
            lora_model, loaders["validation"], device)
        drift_pct = (with_adapter_ppl - baseline_ppl) / baseline_ppl * 100

        # Reset and confirm the baseline comes back.
        reset_lora_to_zero(lora_model)
        reset_ppl = val_ppl_ungated(lora_model, loaders["validation"], device)
        reset_drift_pct = (reset_ppl - baseline_ppl) / baseline_ppl * 100

        print(f"  retrieval: {n_correct}/20  "
              f"(num={per_type['numeric']}/5 ent={per_type['entity']}/5 "
              f"tech={per_type['technical']}/5 fact={per_type['fact']}/5)")
        print(f"  WikiText PPL with last adapter loaded: {with_adapter_ppl:.3f} "
              f"({drift_pct:+.3f}%)")
        print(f"  WikiText PPL after reset:              {reset_ppl:.3f} "
              f"({reset_drift_pct:+.3f}%)")
        print(f"  total time: {elapsed:.0f}s\n")

        all_results[str(alpha_kl)] = {
            "alpha": alpha_kl,
            "n_correct": n_correct,
            "retrieval_rate": n_correct / 20,
            "per_type": dict(per_type),
            "baseline_ppl": baseline_ppl,
            "with_adapter_ppl": with_adapter_ppl,
            "drift_pct": drift_pct,
            "reset_ppl": reset_ppl,
            "reset_drift_pct": reset_drift_pct,
            "total_time_s": elapsed,
            "per_passage": per_passage,
            "trajectories": trajectories,
        }

    # ============================================================
    # Summary — no tables, one line per field
    # ============================================================
    print("=" * 72)
    print("PHASE 48 SUMMARY: KL-to-base regularized absorption")
    print("=" * 72)
    print(f"  rank {RANK}, alpha_lora {ALPHA_LORA}, {N_STEPS} steps/passage, "
          f"KL temperature {KL_TEMPERATURE}")
    print(f"  baseline WikiText val PPL: {baseline_ppl:.3f}")
    print()
    for alpha_kl in ALPHAS:
        r = all_results[str(alpha_kl)]
        print(f"  alpha_kl = {alpha_kl}")
        print(f"    retrieval                    : {r['n_correct']}/20 "
              f"({r['retrieval_rate']:.0%})")
        pt = r["per_type"]
        print(f"    per type                     : "
              f"num={pt['numeric']}/5  ent={pt['entity']}/5  "
              f"tech={pt['technical']}/5  fact={pt['fact']}/5")
        print(f"    PPL (adapter loaded)         : "
              f"{r['with_adapter_ppl']:.3f}  ({r['drift_pct']:+.3f}%)")
        print(f"    PPL (adapter reset)          : "
              f"{r['reset_ppl']:.3f}  ({r['reset_drift_pct']:+.3f}%)")
        print(f"    total time                   : {r['total_time_s']:.0f}s")
        print()

    # Trajectory inspection on first passage for each alpha.
    print("-" * 72)
    print("KL trajectory on first passage (passkey-token separation check)")
    print("-" * 72)
    first_passage = tests[0]
    print(f"  passage = {first_passage['passage'][:120]!r}")
    print(f"  passkey = {first_passage['passkey']!r}")
    print(f"  passage type = {first_passage['type']}")
    print()

    # Re-tokenize the passage for token display
    ids = tokenizer.encode(first_passage["passage"], add_special_tokens=False)
    display_tokens = [tokenizer.decode([t]) for t in ids[1:]]
    passkey_ids = set(tokenizer.encode(
        " " + first_passage["passkey"], add_special_tokens=False))

    for alpha_kl in ALPHAS:
        r = all_results[str(alpha_kl)]
        traj = r["trajectories"].get(0, {}).get("trajectory", [])
        if not traj:
            continue
        first = traj[0]
        last = traj[-1]
        print(f"  alpha_kl = {alpha_kl}")
        print(f"    step 0   : ntp={first['ntp']:.3f}  kl={first['kl']:.4f}")
        print(f"    step {len(traj)-1:3d} : ntp={last['ntp']:.3f}  "
              f"kl={last['kl']:.4f}")
        per_tok = last.get("kl_per_token", [])
        if per_tok and alpha_kl == 0.0:
            # For the unregularized baseline the per-token KL is the
            # cleanest view of where attention has been pushed away from
            # the base distribution by the NTP loss alone — i.e. where
            # the adapter is actually writing.
            sorted_tokens = sorted(
                enumerate(per_tok), key=lambda kv: kv[1], reverse=True)
            top_k = sorted_tokens[:10]
            print(f"    top-10 high-KL tokens in passage (NTP pressure points):")
            for idx, kl_val in top_k:
                tok = display_tokens[idx] if idx < len(display_tokens) else "?"
                is_passkey = ids[idx + 1] in passkey_ids if idx + 1 < len(ids) else False
                mark = " <-- passkey" if is_passkey else ""
                print(f"      pos {idx:3d}  kl={kl_val:.3f}  "
                      f"tok={tok!r}{mark}")
        print()

    with open(results_dir / "kl_sweep.json", "w") as f:
        json.dump({
            "rank": RANK,
            "alpha_lora": ALPHA_LORA,
            "n_steps": N_STEPS,
            "kl_temperature": KL_TEMPERATURE,
            "alphas": ALPHAS,
            "baseline_ppl": baseline_ppl,
            "results": all_results,
        }, f, indent=2)
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
