"""Phase 49b: Train V22 from scratch with metadata-enriched inputs.

Phase 49 showed that bolt-on metadata enrichment breaks a pre-trained
model. This script tests the actual hypothesis: if the model learns WITH
structured metadata from step one, do the projection matrices (W_Q, W_K,
W_V) learn better transformations, producing a better-organized V-space?

Architecture: same V22 (V20_BONSIGNORE) config. The only change is that
90% of training batches get metadata-enriched embeddings (tok_emb + alpha
* MetadataEmbedder output), and 10% get raw embeddings (metadata dropout,
so the model doesn't become dependent on metadata at inference).

The tagger is frozen. The MetadataEmbedder parameters (~660K) are trained
alongside the main model (~510M). After training, we run the Phase 49
measurement battery to compare V-space quality.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase49b_train_enriched.py

Expected runtime: ~11-12 hours (same as V22 baseline).
"""

import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, build_dataloaders
from losses import CombinedHRSLoss
from metrics import run_all_metrics
from experiments.identity_ae.preprocessing.metadata_embedder import TaggerPipeline

# Import the LR/phase helpers from train.py
from train import (
    set_seed, get_lr, get_phase, get_phase_lr_multipliers,
    V2_PARAM_GROUP_INDEX, V1_PARAM_GROUP_INDEX,
)


META_PROB = 0.90   # fraction of batches that get metadata enrichment
TAGGER_PATH = "experiments/identity_ae/preprocessing/tagger_model.pt"


def main():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_v2 = cfg.is_v2()
    param_group_index = V2_PARAM_GROUP_INDEX if is_v2 else V1_PARAM_GROUP_INDEX

    print(f"Phase 49b: Train V22 from scratch with metadata enrichment")
    print(f"Device: {device}")
    print(f"Metadata probability: {META_PROB:.0%}")
    print(f"Ablation: {cfg.training.ablation.value}")

    # ---- Data ----
    print(f"Loading {cfg.training.dataset}...")
    splits, tokenizer = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # ---- Model ----
    model = HRSTransformer(cfg).to(device)
    param_counts = model.component_param_counts()
    print(f"Model parameters: {param_counts['total']:,}")

    # ---- Tagger pipeline (frozen tagger + trainable embedder) ----
    if not Path(TAGGER_PATH).exists():
        print(f"ERROR: tagger not found at {TAGGER_PATH}")
        return
    pipeline = TaggerPipeline(TAGGER_PATH, main_dim=cfg.model.d_model,
                               meta_dim=128, device="cpu")
    # Keep tagger on CPU (frozen, 16M params), only embedder on GPU (660K)
    pipeline.tagger.cpu()
    pipeline.embedder.to(device)
    n_meta_params = sum(p.numel() for p in pipeline.embedder.parameters())
    print(f"MetadataEmbedder parameters: {n_meta_params:,}")
    print(f"MetadataEmbedder alpha init: {pipeline.embedder.alpha.item():.3f}")

    # ---- Hook: use a forward hook on tok_emb to inject metadata ----
    # We use a post-forward hook so the pipeline params stay separate
    # from the model's param groups.
    _meta_active = [False]  # mutable flag, set per-batch
    _last_input_ids = [None]  # capture input ids for the pipeline

    def _capture_input_hook(module, args):
        """Pre-hook to capture input ids before tok_emb runs."""
        _last_input_ids[0] = args[0]

    def _enrich_hook(module, args, output):
        """Post-hook: add metadata embedding to tok_emb output."""
        if _meta_active[0] and _last_input_ids[0] is not None:
            meta = pipeline(_last_input_ids[0])
            return output + meta
        return output

    model.tok_emb.register_forward_pre_hook(_capture_input_hook)
    model.tok_emb.register_forward_hook(_enrich_hook)

    # ---- Loss ----
    loss_fn = CombinedHRSLoss(
        locality_cfg=cfg.locality if cfg.locality.enabled else None,
        label_smoothing=0.0,
    )

    # ---- Optimizer: main model groups + metadata embedder ----
    param_groups_dict = model.get_param_groups()
    optimizer_groups = []
    group_names = []
    for name, params in param_groups_dict.items():
        if params:
            optimizer_groups.append({
                "params": params,
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            })
            group_names.append(name)

    # Add metadata embedder as its own param group
    optimizer_groups.append({
        "params": list(pipeline.embedder.parameters()),
        "lr": cfg.training.learning_rate,
        "weight_decay": 0.0,
    })
    group_names.append("metadata_embedder")

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- Mixed precision ----
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # ---- Output directory ----
    run_dir = Path("results/v22_enriched")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "base_ablation": cfg.training.ablation.value,
            "meta_prob": META_PROB,
            "meta_params": n_meta_params,
            "total_params": param_counts["total"] + n_meta_params,
        }, f, indent=2)

    # ---- Training loop ----
    train_loader = loaders["train"]
    train_iter = iter(train_loader)
    best_val_ppl = float("inf")
    step = 0
    current_phase = get_phase(step, cfg)
    log_history = []

    accum_loss = 0.0
    accum_ce = 0.0
    accum_meta_frac = 0.0
    t0 = time.time()

    print(f"\nStarting training: {cfg.training.max_steps} steps")
    print(f"Effective batch size: {cfg.training.batch_size * cfg.training.grad_accum_steps}")
    print()

    while step < cfg.training.max_steps:
        optimizer.zero_grad()

        for micro_step in range(cfg.training.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x, y = batch[0].to(device), batch[1].to(device)

            # 90% metadata, 10% raw
            use_meta = random.random() < META_PROB
            _meta_active[0] = use_meta
            accum_meta_frac += float(use_meta)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                output = model(x, step=step,
                               collect_layer_reps=cfg.locality.enabled)

                loss_dict = loss_fn(
                    output.logits, y,
                    layer_representations=(output.layer_representations
                                           if cfg.locality.enabled else None),
                    routing_weights=(output.routing_weights
                                     if cfg.uses_router() else None),
                    routing_balance_loss_val=(output.routing_balance_loss
                                              if cfg.uses_router() else None),
                    balance_weight=(cfg.router.balance_loss_weight
                                    if cfg.uses_router() else 0.0),
                    routing_entropy_loss_val=(output.routing_entropy_loss
                                              if cfg.uses_router() else None),
                    entropy_weight=(cfg.router.entropy_loss_weight
                                    if cfg.uses_router() else 0.0),
                    routing_flops_loss_val=(output.routing_flops_loss
                                            if cfg.uses_router() else None),
                    flops_weight=(cfg.router.flops_loss_weight
                                   if cfg.uses_router() else 0.0),
                    engram_recon_loss=(output.engram_recon_loss
                                       if cfg.uses_engrams() else None),
                    recon_weight=(cfg.engram.recon_loss_weight
                                   if cfg.uses_engrams() else 0.0),
                )

                loss = loss_dict["loss"] / cfg.training.grad_accum_steps

            loss.backward()

        loss_val = loss_dict["loss"].item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"  WARNING: NaN/Inf at step {step}, skipping")
            optimizer.zero_grad()
            step += 1
            continue

        accum_loss += loss_val
        accum_ce += loss_dict["ce_loss"].item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(pipeline.embedder.parameters(),
                                        cfg.training.max_grad_norm)

        # Phase-based LR
        new_phase = get_phase(step, cfg)
        if new_phase != current_phase:
            print(f"\n*** Phase {current_phase} -> {new_phase} at step {step} ***\n")
            current_phase = new_phase

        base_lr = get_lr(step, cfg.training.warmup_steps,
                         cfg.training.max_steps, cfg.training.learning_rate)
        lr_mults = get_phase_lr_multipliers(current_phase, cfg)

        for i, (pg, name) in enumerate(zip(optimizer.param_groups, group_names)):
            if name == "metadata_embedder":
                pg["lr"] = base_lr  # same schedule as backbone
            elif name == "loss_scaler":
                pg["lr"] = base_lr * 0.1
            else:
                mult_idx = param_group_index.get(name, 0)
                pg["lr"] = base_lr * lr_mults[mult_idx]

        optimizer.step()
        step += 1

        # Logging
        if step % cfg.training.log_interval == 0:
            n = cfg.training.log_interval
            avg_loss = accum_loss / n
            avg_ce = accum_ce / n
            avg_meta = accum_meta_frac / (n * cfg.training.grad_accum_steps)
            ppl = math.exp(min(avg_ce, 20))
            elapsed = time.time() - t0
            alpha = pipeline.embedder.alpha.item()

            print(f"step {step:6d} | loss {avg_loss:.4f} | CE {avg_ce:.4f} | "
                  f"ppl {ppl:.1f} | lr {base_lr:.2e} | P{current_phase} | "
                  f"meta {avg_meta:.0%} | α {alpha:.3f} | {elapsed:.0f}s")

            log_history.append({
                "step": step, "loss": avg_loss, "ce": avg_ce, "ppl": ppl,
                "lr": base_lr, "phase": current_phase,
                "meta_frac": avg_meta, "alpha": alpha, "time": elapsed,
            })

            accum_loss = 0.0
            accum_ce = 0.0
            accum_meta_frac = 0.0

        # Evaluation
        if step % cfg.training.eval_interval == 0:
            print(f"\n--- Evaluation at step {step} ---")
            model.eval()
            _meta_active[0] = False  # eval without metadata
            metrics = run_all_metrics(model, loaders["validation"], device,
                                      amp_dtype, max_batches=10)
            model.train()

            val_ppl = metrics.get("val_ppl", float("inf"))
            print(f"  val_ppl (no meta):  {val_ppl:.1f}")
            print(f"  eff_rank_mean:      {metrics.get('effective_rank_mean', 'N/A')}")
            print(f"  cosine_sim_mean:    {metrics.get('cosine_sim_mean', 'N/A')}")

            # Also eval WITH metadata to see the difference
            _meta_active[0] = True
            metrics_meta = run_all_metrics(model, loaders["validation"], device,
                                           amp_dtype, max_batches=10)
            _meta_active[0] = False
            val_ppl_meta = metrics_meta.get("val_ppl", float("inf"))
            print(f"  val_ppl (with meta): {val_ppl_meta:.1f}")
            print(f"  alpha:               {pipeline.embedder.alpha.item():.4f}")

            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({
                    "step": step, "val_ppl": val_ppl,
                    "val_ppl_meta": val_ppl_meta,
                    "alpha": pipeline.embedder.alpha.item(),
                    **{k: v for k, v in metrics.items()
                       if isinstance(v, (int, float))},
                }) + "\n")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "pipeline_embedder_state": pipeline.embedder.state_dict(),
                    "val_ppl": val_ppl,
                    "val_ppl_meta": val_ppl_meta,
                    "alpha": pipeline.embedder.alpha.item(),
                }, run_dir / "best.pt")
                print(f"  ** New best val_ppl: {best_val_ppl:.2f}")
            print()

        # Checkpoint
        if step % cfg.training.save_interval == 0:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "pipeline_embedder_state": pipeline.embedder.state_dict(),
                "phase": current_phase,
                "alpha": pipeline.embedder.alpha.item(),
            }
            ckpt_path = run_dir / f"checkpoint_{step}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved {ckpt_path}")

            # Keep only last 2
            existing = sorted(run_dir.glob("checkpoint_*.pt"),
                              key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]:
                old.unlink()

    # Final save
    elapsed = time.time() - t0
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "pipeline_embedder_state": pipeline.embedder.state_dict(),
        "alpha": pipeline.embedder.alpha.item(),
    }, run_dir / "final.pt")

    with open(run_dir / "log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 49b COMPLETE: {step} steps in {elapsed/3600:.1f} hours")
    print(f"Best val PPL (no meta): {best_val_ppl:.2f}")
    print(f"Final alpha: {pipeline.embedder.alpha.item():.4f}")
    print(f"Results saved to {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
