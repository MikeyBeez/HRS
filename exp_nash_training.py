"""Nash Equilibrium Training: alternating optimization on Shakespeare.

Five conditions:
A. Full Nash alternating (6 players, foundation-up order)
B. Standard simultaneous (control)
C1-C3. Nash with different cycle orders
D. Partial Nash (proj-kernel only)
E. Nash with imbalanced cycles

Usage:
    python exp_nash_training.py [--device cuda] [--conditions A,B,C1,D,E]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from exp_kernel_attention import (
    load_shakespeare, CharDataset,
    extract_engrams, run_pair_analysis,
)
from scripts.reference_kernel import (
    BonsignoreTransformer, BonsignoreTransformerBlock,
)


# ============================================================
# Player definitions
# ============================================================

def get_player_groups(model):
    """Partition model parameters into 6 player groups."""
    players = {
        "projections": [],   # Q, K, V matrices
        "kernel": [],        # temperatures, MLPs, alphas
        "lm_head": [],       # output embeddings, final LN
        "cat_head": [],      # categorization (if present)
        "cross_attn": [],    # cross-attention + gates
        "scalars": [],       # output scalars, PEER routing
    }

    for name, param in model.named_parameters():
        if 'qkv' in name or ('out_proj' in name and 'cross' not in name):
            players["projections"].append(param)
        elif 'kernel' in name or 'log_tau' in name or 'head_alpha' in name or 'mlp' in name and 'attn' in name:
            players["kernel"].append(param)
        elif 'lm_head' in name or 'ln_f' in name:
            players["lm_head"].append(param)
        elif 'cross' in name or 'gate' in name:
            players["cross_attn"].append(param)
        else:
            # Default: assign to projections (backbone)
            players["projections"].append(param)

    return players


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_group(params):
    for p in params:
        p.requires_grad_(True)


# ============================================================
# Nash alternating training
# ============================================================

def train_nash(model, train_data, val_data, device, cycle_config, n_cycles=26,
               label="Nash", batch_size=16):
    """Train with alternating optimization cycles.

    Args:
        cycle_config: list of (player_name, n_steps, lr) tuples defining the cycle
        n_cycles: number of complete cycles
    """
    seq_len = model.max_seq_len
    train_ds = CharDataset(train_data, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    V = model.lm_head.weight.shape[0]

    players = get_player_groups(model)
    model.to(device)

    losses = []
    val_losses = []
    best_val = float('inf')
    best_state = None
    train_iter = iter(train_loader)
    t0 = time.time()
    total_steps = 0

    for cycle in range(n_cycles):
        for player_name, n_steps, lr in cycle_config:
            if player_name not in players or not players[player_name]:
                continue

            # Freeze all, unfreeze this player
            freeze_all(model)
            unfreeze_group(players[player_name])

            optimizer = torch.optim.AdamW(
                [p for p in players[player_name] if p.requires_grad],
                lr=lr, weight_decay=0.1 if player_name == "projections" else 0.01,
            )

            model.train()
            for step in range(n_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                B, T, V_ = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V_), y.reshape(B * T))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in players[player_name] if p.requires_grad], 1.0
                )
                optimizer.step()
                losses.append(loss.item())
                total_steps += 1

        # End of cycle: evaluate
        model.eval()
        freeze_all(model)  # all frozen for eval
        val_ds = CharDataset(val_data, seq_len)
        val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=True)
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vl, _ = model(vx)
                val_loss += F.cross_entropy(vl.reshape(-1, V), vy.reshape(-1)).item()
                n_val += 1
                if n_val >= 20:
                    break
        val_loss /= n_val
        val_losses.append((total_steps, val_loss))

        elapsed = time.time() - t0
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *best*"

        if (cycle + 1) % 5 == 0 or cycle == 0:
            avg_train = sum(losses[-200:]) / min(len(losses[-200:]), 200)
            print(f"  [{label}] cycle {cycle+1:3d} ({total_steps:5d} steps): "
                  f"train={avg_train:.4f} val={val_loss:.4f}{marker} ({elapsed:.0f}s)")

    # Restore best
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return losses, val_losses, best_val, total_steps


def train_simultaneous(model, train_data, val_data, device, n_steps=5000,
                       lr=3e-4, batch_size=16, label="Simultaneous"):
    """Standard simultaneous training (control)."""
    seq_len = model.max_seq_len
    train_ds = CharDataset(train_data, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    V = model.lm_head.weight.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.to(device)
    model.train()

    losses = []
    val_losses = []
    best_val = float('inf')
    best_state = None
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(n_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        B, T, V_ = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V_), y.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 250 == 0:
            model.eval()
            val_ds = CharDataset(val_data, seq_len)
            val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=True)
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vl, _ = model(vx)
                    val_loss += F.cross_entropy(vl.reshape(-1, V), vy.reshape(-1)).item()
                    n_val += 1
                    if n_val >= 20:
                        break
            val_loss /= n_val
            val_losses.append((step + 1, val_loss))

            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = " *best*"

            if (step + 1) % 1000 == 0:
                avg_train = sum(losses[-250:]) / 250
                elapsed = time.time() - t0
                print(f"  [{label}] step {step+1:5d}: train={avg_train:.4f} val={val_loss:.4f}{marker} ({elapsed:.0f}s)")

            model.train()

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return losses, val_losses, best_val


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conditions", type=str, default="A,B,D")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    conditions = [c.strip() for c in args.conditions.split(",")]

    text, stoi, itos, plays = load_shakespeare()
    vocab_size = len(stoi)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    # Engram analysis data
    lines = text.split('\n')
    p1_text = '\n'.join(lines[:15600])
    p2_text = '\n'.join(lines[15600:])
    def chunk(t, sz=256):
        return [t[i:i+sz] for i in range(0, len(t)-sz, sz)]
    p1_chunks = chunk(p1_text)[:50]
    p2_chunks = chunk(p2_text)[:50]

    all_results = {}

    # ============================================================
    # Condition A: Full Nash alternating
    # ============================================================
    if "A" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION A: Full Nash Alternating")
        print(f"{'#'*60}")

        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        cycle = [
            ("projections", 50, 3e-4),
            ("kernel", 50, 1e-3),
            ("lm_head", 30, 3e-4),
            ("cross_attn", 20, 1e-3),
            ("scalars", 20, 1e-3),
        ]

        losses, val_losses, best_val, total_steps = train_nash(
            model, train_data, val_data, device, cycle, n_cycles=26, label="NashA",
        )

        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "NashA play-level")

        all_results["A_nash_full"] = {
            "best_val": best_val, "total_steps": total_steps,
            "topic": topic,
        }
        del model; torch.cuda.empty_cache()

    # ============================================================
    # Condition B: Simultaneous (control)
    # ============================================================
    if "B" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION B: Simultaneous (Control)")
        print(f"{'#'*60}")

        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)

        losses, val_losses, best_val = train_simultaneous(
            model, train_data, val_data, device, n_steps=5000, label="Simult",
        )

        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "Simult play-level")

        all_results["B_simultaneous"] = {
            "best_val": best_val, "total_steps": 5000,
            "topic": topic,
        }
        del model; torch.cuda.empty_cache()

    # ============================================================
    # Condition C1: Nash heads-first order
    # ============================================================
    if "C1" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION C1: Nash Heads-First Order")
        print(f"{'#'*60}")

        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)

        cycle = [
            ("lm_head", 30, 3e-4),
            ("projections", 50, 3e-4),
            ("kernel", 50, 1e-3),
            ("cross_attn", 20, 1e-3),
            ("scalars", 20, 1e-3),
        ]

        losses, val_losses, best_val, total_steps = train_nash(
            model, train_data, val_data, device, cycle, n_cycles=26, label="NashC1",
        )

        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "NashC1 play-level")

        all_results["C1_nash_heads_first"] = {
            "best_val": best_val, "total_steps": total_steps,
            "topic": topic,
        }
        del model; torch.cuda.empty_cache()

    # ============================================================
    # Condition D: Partial Nash (proj-kernel only)
    # ============================================================
    if "D" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION D: Partial Nash (Proj-Kernel Only)")
        print(f"{'#'*60}")

        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)

        # Alternate proj and kernel; everything else trains with projections
        cycle = [
            ("projections", 50, 3e-4),
            ("kernel", 50, 1e-3),
        ]
        # But we need all params to get gradients somehow — the "projections" group
        # includes backbone params. Let's handle this by making "projections" actually
        # include everything except kernel.

        # Override: for partial Nash, put all non-kernel params together
        players = get_player_groups(model)
        non_kernel = []
        for name, params in players.items():
            if name != "kernel":
                non_kernel.extend(params)
        players_partial = {
            "all_except_kernel": non_kernel,
            "kernel": players["kernel"],
        }

        # Custom training for partial Nash
        seq_len = model.max_seq_len
        train_ds = CharDataset(train_data, seq_len)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        V = vocab_size

        model.to(device)
        train_iter = iter(train_loader)
        losses = []
        val_losses_d = []
        best_val = float('inf')
        best_state = None
        t0 = time.time()
        total_steps = 0

        for cycle_num in range(50):  # 50 cycles × 100 steps = 5000
            for player_name, n_steps, lr in [("all_except_kernel", 50, 3e-4), ("kernel", 50, 1e-3)]:
                freeze_all(model)
                group = players_partial[player_name]
                unfreeze_group(group)
                opt = torch.optim.AdamW([p for p in group if p.requires_grad], lr=lr, weight_decay=0.1)
                model.train()

                for _ in range(n_steps):
                    try: x, y = next(train_iter)
                    except StopIteration: train_iter = iter(train_loader); x, y = next(train_iter)
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in group if p.requires_grad], 1.0)
                    opt.step(); losses.append(loss.item()); total_steps += 1

            # Eval
            model.eval(); freeze_all(model)
            vds = CharDataset(val_data, seq_len)
            vdl = DataLoader(vds, batch_size=16, drop_last=True)
            vl = 0; nv = 0
            with torch.no_grad():
                for vx, vy in vdl:
                    vx, vy = vx.to(device), vy.to(device)
                    vo, _ = model(vx)
                    vl += F.cross_entropy(vo.reshape(-1, V), vy.reshape(-1)).item()
                    nv += 1
                    if nv >= 20: break
            vl /= nv; val_losses_d.append((total_steps, vl))
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (cycle_num + 1) % 10 == 0:
                avg = sum(losses[-200:]) / min(200, len(losses[-200:]))
                print(f"  [NashD] cycle {cycle_num+1} ({total_steps} steps): train={avg:.4f} val={vl:.4f} ({time.time()-t0:.0f}s)")

        if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "NashD play-level")

        all_results["D_partial_nash"] = {
            "best_val": best_val, "total_steps": total_steps,
            "topic": topic,
        }
        del model; torch.cuda.empty_cache()

    # ============================================================
    # Condition E: Nash with imbalanced cycles
    # ============================================================
    if "E" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION E: Nash Imbalanced Cycles")
        print(f"{'#'*60}")

        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)

        cycle = [
            ("projections", 100, 3e-4),
            ("kernel", 50, 1e-3),
            ("lm_head", 20, 3e-4),
            ("cross_attn", 20, 1e-3),
            ("scalars", 20, 1e-3),
        ]

        losses, val_losses, best_val, total_steps = train_nash(
            model, train_data, val_data, device, cycle, n_cycles=22, label="NashE",
        )

        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "NashE play-level")

        all_results["E_imbalanced"] = {
            "best_val": best_val, "total_steps": total_steps,
            "topic": topic,
        }
        del model; torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<30} {'Val Loss':>10} {'Steps':>7} {'Play Acc':>10}")
    print(f"  {'-'*60}")
    for name, r in all_results.items():
        play_acc = f"{r['topic']['accuracy']:.1%}" if r.get('topic') else "N/A"
        print(f"  {name:<30} {r['best_val']:>10.4f} {r['total_steps']:>7} {play_acc:>10}")

    # Save
    out_dir = Path("results/nash_training")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
