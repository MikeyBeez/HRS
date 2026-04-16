"""Phase 28: Staged absorption — fast foreground + background optimization.

Insight from the user: single-phrase training already achieves 100% retrieval
on verbatim queries (phase 24). The paraphrase robustness needs come from
multi-phrase training (phase 26). These can be staged temporally: do the
fast single-phrase absorption in the foreground (when the user submits
content), and run the multi-paraphrase optimization in the background
(while the user is typing the next prompt).

This phase measures the staged latency story:
  Stage 1 (foreground): train K=80 steps on the original (prompt + answer)
                        only. Test same-prompt retrieval.
  Stage 2 (background): continue training the same adapter for another 80
                        steps, sampling uniformly from passage + 4 prompt
                        forms. Test paraphrase retrieval after each
                        background step batch.

We expect:
  - Foreground stage achieves 100% same-prompt retrieval in ~1.5s/passage
  - Background stage progressively lifts paraphrase retrieval from low
    to ~100% as the optimization budget grows

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase28_staged.py
"""

import json
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
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, RANK, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase22_engram_key import (
    make_key, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


SOURCE = "L5_mean"
FG_STEPS = 80     # foreground budget
BG_STEPS = 80     # background budget (after foreground)


def train_steps(model, source_ids_list, n_steps, lr, optimizer=None, scheduler=None):
    """Run n_steps of training, sampling uniformly from source_ids_list.

    If optimizer/scheduler are None, create fresh ones with constant lr.
    Returns the optimizer/scheduler so caller can chain stages.
    """
    params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    if optimizer is None:
        optimizer = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(n_steps):
        ids_t = source_ids_list[random.randint(0, len(source_ids_list) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    model.eval()
    return optimizer, scheduler


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase28")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"Foreground budget: {FG_STEPS} steps  (single prompt, lr=high)")
    print(f"Background budget: {BG_STEPS} steps  (passage + 4 prompts, lr=low)")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # STAGE 1: FOREGROUND — single-prompt absorption
    # ============================================================
    print(f"{'='*60}")
    print(f"STAGE 1: FOREGROUND ({FG_STEPS} steps, single prompt+answer)")
    print(f"{'='*60}")

    library = []  # list of {"passage_ids", "prompt_ids_list", "sd_fg", "sd_bg", "keys", "meta"}
    fg_times = []

    for i, test in enumerate(tests):
        reset_lora_to_zero(model)

        # Foreground source: just the original prompt + answer
        original_pa = f"{test['prompt']} {test['passkey']}"
        fg_ids = tokenizer.encode(original_pa, add_special_tokens=False)
        fg_t = torch.tensor(fg_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

        t0 = time.time()
        train_steps(model, [fg_t], FG_STEPS, lr=HIGH_LR)
        fg_dt = time.time() - t0
        fg_times.append(fg_dt)

        sd_fg = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}

        # Stash the training sources for later background stage
        passage_ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        passage_t = torch.tensor(passage_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        all_prompts = [test["prompt"]] + paraphrase(test)
        prompt_t_list = []
        for p in all_prompts:
            pa = f"{p} {test['passkey']}"
            ids = tokenizer.encode(pa, add_special_tokens=False)
            prompt_t_list.append(torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device))
        bg_sources = [passage_t] + prompt_t_list

        # Compute key for the original prompt under base model
        reset_lora_to_zero(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        key_orig = make_key(model, ids_t, SOURCE)

        library.append({
            "test": dict(test),
            "sd_fg": sd_fg,
            "sd_bg": None,  # filled in stage 2
            "keys_fg": [key_orig],  # 1 key after foreground
            "keys_bg": None,  # filled in stage 2
            "bg_sources": bg_sources,
            "all_prompts": all_prompts,
        })

    print(f"  Foreground: {sum(fg_times):.1f}s for 20 = {sum(fg_times)/20*1000:.0f}ms/passage")

    # Test same-prompt retrieval after foreground
    def route_using(library_keys_attr):
        def routes(q):
            best_a, best_score = -1, -2.0
            q_n = q / (q.norm() + 1e-8)
            for ai, entry in enumerate(library):
                for k in entry[library_keys_attr]:
                    k_n = k / (k.norm() + 1e-8)
                    sim = float(torch.dot(q_n, k_n))
                    if sim > best_score:
                        best_score = sim
                        best_a = ai
            return best_a
        return routes

    def evaluate(library_keys_attr, sd_attr, query_prompts_per_test):
        """Evaluate retrieval against a set of query prompts.

        query_prompts_per_test: list of lists; query_prompts_per_test[i] is
        the list of paraphrases to test for test i.
        Returns (n_routed, n_retrieved, n_total).
        """
        router = route_using(library_keys_attr)
        n_routed = 0
        n_retrieved = 0
        n_total = 0
        for i, test in enumerate(tests):
            for para in query_prompts_per_test[i]:
                reset_lora_to_zero(model)
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                q = make_key(model, ids_t, SOURCE)
                best_a = router(q)
                if best_a == i:
                    n_routed += 1
                sd = library[best_a][sd_attr]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device, 50)
                if check_passkey(gen, test["passkey"]):
                    n_retrieved += 1
                n_total += 1
        return n_routed, n_retrieved, n_total

    # Same-prompt evaluation after foreground
    same_prompt_query = [[test["prompt"]] for test in tests]
    fg_routed, fg_retrieved, fg_total = evaluate("keys_fg", "sd_fg", same_prompt_query)
    print(f"  After foreground:")
    print(f"    Same-prompt retrieval: {fg_retrieved}/{fg_total} ({fg_retrieved/fg_total:.0%})")

    # Paraphrase evaluation after foreground (expected to be poor)
    para_query = [paraphrase(test) for test in tests]
    fg_para_routed, fg_para_retrieved, fg_para_total = evaluate("keys_fg", "sd_fg", para_query)
    print(f"    Paraphrase retrieval: {fg_para_retrieved}/{fg_para_total} ({fg_para_retrieved/fg_para_total:.0%})")

    # ============================================================
    # STAGE 2: BACKGROUND — multi-paraphrase optimization
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: BACKGROUND ({BG_STEPS} steps, passage + all paraphrases)")
    print(f"{'='*60}")

    bg_times = []
    for i, entry in enumerate(library):
        # Reload the adapter from the foreground state
        sd_gpu = {k: v.to(device) for k, v in entry["sd_fg"].items()}
        load_lora_state_dict(model, sd_gpu)

        t0 = time.time()
        train_steps(model, entry["bg_sources"], BG_STEPS, lr=BASE_LR)
        bg_dt = time.time() - t0
        bg_times.append(bg_dt)

        sd_bg = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        entry["sd_bg"] = sd_bg

        # Build multi-key set under base model
        reset_lora_to_zero(model)
        keys_bg = []
        for p in entry["all_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys_bg.append(make_key(model, ids_t, SOURCE))
        entry["keys_bg"] = keys_bg

    print(f"  Background: {sum(bg_times):.1f}s for 20 = {sum(bg_times)/20*1000:.0f}ms/passage")

    bg_routed, bg_retrieved, bg_total = evaluate("keys_bg", "sd_bg", same_prompt_query)
    print(f"  After background:")
    print(f"    Same-prompt retrieval: {bg_retrieved}/{bg_total} ({bg_retrieved/bg_total:.0%})")

    bg_para_routed, bg_para_retrieved, bg_para_total = evaluate("keys_bg", "sd_bg", para_query)
    print(f"    Paraphrase retrieval: {bg_para_retrieved}/{bg_para_total} ({bg_para_retrieved/bg_para_total:.0%})")

    # Forgetting check
    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print(f"PHASE 28 SUMMARY (staged absorption)")
    print(f"{'='*60}")
    print(f"  Stage         | Time/passage | Same-prompt | Paraphrase")
    print(f"  --------------|--------------|-------------|------------")
    print(f"  Foreground    | {sum(fg_times)/20*1000:>7.0f} ms | "
          f"{fg_retrieved:>3d}/{fg_total} ({fg_retrieved/fg_total:.0%}) | "
          f"{fg_para_retrieved:>3d}/{fg_para_total} ({fg_para_retrieved/fg_para_total:.0%})")
    print(f"  + Background  | {sum(bg_times)/20*1000:>7.0f} ms | "
          f"{bg_retrieved:>3d}/{bg_total} ({bg_retrieved/bg_total:.0%}) | "
          f"{bg_para_retrieved:>3d}/{bg_para_total} ({bg_para_retrieved/bg_para_total:.0%})")
    print(f"  Total budget  | {(sum(fg_times)+sum(bg_times))/20*1000:>7.0f} ms (= foreground + background)")
    print(f"\n  Val PPL drift: {drift:+.3f}%")

    summary = {
        "fg_steps": FG_STEPS, "bg_steps": BG_STEPS,
        "fg_time_per_passage_ms": sum(fg_times) / 20 * 1000,
        "bg_time_per_passage_ms": sum(bg_times) / 20 * 1000,
        "fg_same_prompt_retrieval": fg_retrieved / fg_total,
        "fg_paraphrase_retrieval": fg_para_retrieved / fg_para_total,
        "bg_same_prompt_retrieval": bg_retrieved / bg_total,
        "bg_paraphrase_retrieval": bg_para_retrieved / bg_para_total,
        "drift_pct": drift,
    }
    with open(results_dir / "staged.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
