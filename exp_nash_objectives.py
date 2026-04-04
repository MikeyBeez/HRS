"""Nash Bargaining on Objective Functions.

Train all parameters simultaneously (what works) but dynamically
balance objective weights (what we haven't tried).

A. Fixed weighting (control): total = lm + 0.1 * cat
B. Nash bargaining on loss rates: struggling objective gets more weight
C. Nash bargaining on gradient norms: dominating objective gets dampened

Usage:
    python exp_nash_objectives.py [--device cuda] [--conditions A,B]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exp_kernel_attention import (
    load_shakespeare, CharDataset,
    extract_engrams, run_pair_analysis,
)
from scripts.reference_kernel import BonsignoreTransformer


# ============================================================
# Simple categorization head for Shakespeare
# ============================================================

class ShakespeareCatHead(nn.Module):
    """Binary categorization: first half vs second half of corpus."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 2)

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return self.proj(pooled)


# ============================================================
# Training functions
# ============================================================

def make_cat_labels(indices, boundary_idx, seq_len, total_seqs):
    """Assign binary category based on position in corpus."""
    boundary_seq = boundary_idx // seq_len
    return (indices >= boundary_seq).long()


def train_fixed(model, cat_head, train_data, val_data, device, n_steps=5000,
                lr=3e-4, cat_weight=0.1, batch_size=16, label="Fixed"):
    """Condition A: fixed weighting."""
    seq_len = model.max_seq_len
    train_ds = CharDataset(train_data, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    V = model.lm_head.weight.shape[0]
    boundary = len(train_data) // 2

    all_params = list(model.parameters()) + list(cat_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.1)
    model.to(device); cat_head.to(device)
    model.train(); cat_head.train()

    losses_lm, losses_cat, val_losses = [], [], []
    best_val = float('inf')
    best_state_m, best_state_c = None, None
    train_iter = iter(train_loader)
    batch_idx = 0
    t0 = time.time()

    for step in range(n_steps):
        try:
            x, y = next(train_iter)
            batch_idx += 1
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            batch_idx = 0

        x, y = x.to(device), y.to(device)
        logits, hidden = model(x)
        B, T, V_ = logits.shape
        lm_loss = F.cross_entropy(logits.reshape(B * T, V_), y.reshape(B * T))

        # Cat loss
        cat_logits = cat_head(hidden.detach())
        cat_labels = make_cat_labels(
            torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size),
            boundary // seq_len, seq_len, len(train_ds)
        ).to(device)[:B]
        cat_loss = F.cross_entropy(cat_logits, cat_labels)

        total = lm_loss + cat_weight * cat_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        losses_lm.append(lm_loss.item())
        losses_cat.append(cat_loss.item())

        if (step + 1) % 250 == 0:
            model.eval()
            vds = CharDataset(val_data, seq_len)
            vdl = DataLoader(vds, batch_size=batch_size, drop_last=True)
            vl = 0; nv = 0
            with torch.no_grad():
                for vx, vy in vdl:
                    vx, vy = vx.to(device), vy.to(device)
                    vo, _ = model(vx)
                    vl += F.cross_entropy(vo.reshape(-1, V), vy.reshape(-1)).item()
                    nv += 1
                    if nv >= 20: break
            vl /= nv
            val_losses.append((step + 1, vl))
            if vl < best_val:
                best_val = vl
                best_state_m = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (step + 1) % 1000 == 0:
                avg_lm = sum(losses_lm[-250:]) / 250
                avg_cat = sum(losses_cat[-250:]) / 250
                elapsed = time.time() - t0
                print(f"  [{label}] step {step+1}: lm={avg_lm:.4f} cat={avg_cat:.4f} "
                      f"val={vl:.4f} w=[{1.0:.2f}/{cat_weight:.2f}] ({elapsed:.0f}s)")
            model.train()

    if best_state_m:
        model.load_state_dict({k: v.to(device) for k, v in best_state_m.items()})

    return {
        "best_val": best_val,
        "final_cat": sum(losses_cat[-100:]) / 100,
        "weight_history": [(1.0, cat_weight)] * (n_steps // 100),
    }


def train_nash_rates(model, cat_head, train_data, val_data, device, n_steps=5000,
                     lr=3e-4, temperature=1.0, batch_size=16, label="NashRate"):
    """Condition B: Nash bargaining on loss improvement rates."""
    seq_len = model.max_seq_len
    train_ds = CharDataset(train_data, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    V = model.lm_head.weight.shape[0]
    boundary = len(train_data) // 2

    all_params = list(model.parameters()) + list(cat_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.1)
    model.to(device); cat_head.to(device)
    model.train(); cat_head.train()

    # EMA tracking
    ema_lm = 5.0  # initial estimate
    ema_cat = 3.0
    prev_ema_lm = ema_lm
    prev_ema_cat = ema_cat
    ema_decay = 0.99

    losses_lm, losses_cat, val_losses = [], [], []
    weight_history = []
    best_val = float('inf')
    best_state_m = None
    train_iter = iter(train_loader)
    batch_idx = 0
    t0 = time.time()

    for step in range(n_steps):
        try:
            x, y = next(train_iter)
            batch_idx += 1
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            batch_idx = 0

        x, y = x.to(device), y.to(device)
        logits, hidden = model(x)
        B, T, V_ = logits.shape
        lm_loss = F.cross_entropy(logits.reshape(B * T, V_), y.reshape(B * T))

        cat_logits = cat_head(hidden.detach())
        cat_labels = make_cat_labels(
            torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size),
            boundary // seq_len, seq_len, len(train_ds)
        ).to(device)[:B]
        cat_loss = F.cross_entropy(cat_logits, cat_labels)

        # Update EMAs
        prev_ema_lm = ema_lm
        prev_ema_cat = ema_cat
        ema_lm = ema_decay * ema_lm + (1 - ema_decay) * lm_loss.item()
        ema_cat = ema_decay * ema_cat + (1 - ema_decay) * cat_loss.item()

        # Improvement rates (positive = improving)
        rate_lm = (prev_ema_lm - ema_lm) / (prev_ema_lm + 1e-8)
        rate_cat = (prev_ema_cat - ema_cat) / (prev_ema_cat + 1e-8)

        # Struggling objective gets more weight
        rates = torch.tensor([-rate_lm / temperature, -rate_cat / temperature])
        weights = F.softmax(rates, dim=0)
        w_lm = weights[0].item()
        w_cat = weights[1].item()

        total = w_lm * lm_loss + w_cat * cat_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        losses_lm.append(lm_loss.item())
        losses_cat.append(cat_loss.item())

        if (step + 1) % 100 == 0:
            weight_history.append((w_lm, w_cat))

        if (step + 1) % 250 == 0:
            model.eval()
            vds = CharDataset(val_data, seq_len)
            vdl = DataLoader(vds, batch_size=batch_size, drop_last=True)
            vl = 0; nv = 0
            with torch.no_grad():
                for vx, vy in vdl:
                    vx, vy = vx.to(device), vy.to(device)
                    vo, _ = model(vx)
                    vl += F.cross_entropy(vo.reshape(-1, V), vy.reshape(-1)).item()
                    nv += 1
                    if nv >= 20: break
            vl /= nv
            val_losses.append((step + 1, vl))
            if vl < best_val:
                best_val = vl
                best_state_m = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (step + 1) % 1000 == 0:
                avg_lm = sum(losses_lm[-250:]) / 250
                avg_cat = sum(losses_cat[-250:]) / 250
                elapsed = time.time() - t0
                print(f"  [{label}] step {step+1}: lm={avg_lm:.4f} cat={avg_cat:.4f} "
                      f"val={vl:.4f} w=[{w_lm:.3f}/{w_cat:.3f}] ({elapsed:.0f}s)")
            model.train()

    if best_state_m:
        model.load_state_dict({k: v.to(device) for k, v in best_state_m.items()})

    return {
        "best_val": best_val,
        "final_cat": sum(losses_cat[-100:]) / 100,
        "weight_history": weight_history,
        "final_weights": (w_lm, w_cat),
    }


def train_nash_grads(model, cat_head, train_data, val_data, device, n_steps=5000,
                     lr=3e-4, batch_size=16, label="NashGrad"):
    """Condition C: Nash bargaining on gradient norms."""
    seq_len = model.max_seq_len
    train_ds = CharDataset(train_data, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    V = model.lm_head.weight.shape[0]
    boundary = len(train_data) // 2

    all_params = list(model.parameters()) + list(cat_head.parameters())
    # Shared params: first few layers' parameters (representation layers)
    shared_params = [p for p in model.tok_emb.parameters()] + \
                    [p for b in model.blocks[:3] for p in b.parameters()]
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.1)
    model.to(device); cat_head.to(device)
    model.train(); cat_head.train()

    losses_lm, losses_cat, val_losses = [], [], []
    weight_history = []
    best_val = float('inf')
    best_state_m = None
    train_iter = iter(train_loader)
    batch_idx = 0
    t0 = time.time()

    for step in range(n_steps):
        try:
            x, y = next(train_iter)
            batch_idx += 1
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            batch_idx = 0

        x, y = x.to(device), y.to(device)
        logits, hidden = model(x)
        B, T, V_ = logits.shape
        lm_loss = F.cross_entropy(logits.reshape(B * T, V_), y.reshape(B * T))

        cat_logits = cat_head(hidden)  # Don't detach — need gradients through shared params
        cat_labels = make_cat_labels(
            torch.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size),
            boundary // seq_len, seq_len, len(train_ds)
        ).to(device)[:B]
        cat_loss = F.cross_entropy(cat_logits, cat_labels)

        # Compute gradient norms on shared params (every 10 steps for speed)
        if (step + 1) % 10 == 0:
            grads_lm = torch.autograd.grad(lm_loss, shared_params, retain_graph=True,
                                            allow_unused=True)
            grads_cat = torch.autograd.grad(cat_loss, shared_params, retain_graph=True,
                                            allow_unused=True)
            norm_lm = sum(g.norm().item() for g in grads_lm if g is not None)
            norm_cat = sum(g.norm().item() for g in grads_cat if g is not None)

            # Balance: dominating objective gets less weight
            w_lm = norm_cat / (norm_lm + norm_cat + 1e-8)
            w_cat = norm_lm / (norm_lm + norm_cat + 1e-8)
        # else use previous weights (initialized to equal)
        elif step == 0:
            w_lm, w_cat = 0.5, 0.5

        total = w_lm * lm_loss + w_cat * cat_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        losses_lm.append(lm_loss.item())
        losses_cat.append(cat_loss.item())

        if (step + 1) % 100 == 0:
            weight_history.append((w_lm, w_cat))

        if (step + 1) % 250 == 0:
            model.eval()
            vds = CharDataset(val_data, seq_len)
            vdl = DataLoader(vds, batch_size=batch_size, drop_last=True)
            vl = 0; nv = 0
            with torch.no_grad():
                for vx, vy in vdl:
                    vx, vy = vx.to(device), vy.to(device)
                    vo, _ = model(vx)
                    vl += F.cross_entropy(vo.reshape(-1, V), vy.reshape(-1)).item()
                    nv += 1
                    if nv >= 20: break
            vl /= nv
            val_losses.append((step + 1, vl))
            if vl < best_val:
                best_val = vl
                best_state_m = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (step + 1) % 1000 == 0:
                avg_lm = sum(losses_lm[-250:]) / 250
                avg_cat = sum(losses_cat[-250:]) / 250
                elapsed = time.time() - t0
                print(f"  [{label}] step {step+1}: lm={avg_lm:.4f} cat={avg_cat:.4f} "
                      f"val={vl:.4f} w=[{w_lm:.3f}/{w_cat:.3f}] ({elapsed:.0f}s)")
            model.train()

    if best_state_m:
        model.load_state_dict({k: v.to(device) for k, v in best_state_m.items()})

    return {
        "best_val": best_val,
        "final_cat": sum(losses_cat[-100:]) / 100,
        "weight_history": weight_history,
        "final_weights": (w_lm, w_cat),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conditions", type=str, default="A,B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    conditions = [c.strip() for c in args.conditions.split(",")]

    text, stoi, itos, plays = load_shakespeare()
    vocab_size = len(stoi)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    lines = text.split('\n')
    p1_text = '\n'.join(lines[:15600])
    p2_text = '\n'.join(lines[15600:])
    def chunk(t, sz=256):
        return [t[i:i+sz] for i in range(0, len(t)-sz, sz)]
    p1_chunks = chunk(p1_text)[:50]
    p2_chunks = chunk(p2_text)[:50]

    all_results = {}

    if "A" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION A: Fixed Weighting (Control)")
        print(f"{'#'*60}")
        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)
        cat_head = ShakespeareCatHead(model.d_model)
        result = train_fixed(model, cat_head, train_data, val_data, device, label="Fixed")
        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "Fixed play-level")
        result["topic"] = topic
        all_results["A_fixed"] = result
        del model, cat_head; torch.cuda.empty_cache()

    if "B" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION B: Nash Bargaining (Loss Rates)")
        print(f"{'#'*60}")
        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)
        cat_head = ShakespeareCatHead(model.d_model)
        result = train_nash_rates(model, cat_head, train_data, val_data, device, label="NashRate")
        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "NashRate play-level")
        result["topic"] = topic
        all_results["B_nash_rates"] = result
        del model, cat_head; torch.cuda.empty_cache()

    if "C" in conditions:
        print(f"\n{'#'*60}")
        print("CONDITION C: Nash Bargaining (Gradient Norms)")
        print(f"{'#'*60}")
        torch.manual_seed(args.seed)
        model = BonsignoreTransformer(vocab_size)
        cat_head = ShakespeareCatHead(model.d_model)
        result = train_nash_grads(model, cat_head, train_data, val_data, device, label="NashGrad")
        eng1 = extract_engrams(model, p1_chunks, stoi, device)
        eng2 = extract_engrams(model, p2_chunks, stoi, device)
        topic = run_pair_analysis(eng1, eng2, "NashGrad play-level")
        result["topic"] = topic
        all_results["C_nash_grads"] = result
        del model, cat_head; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<25} {'Val Loss':>10} {'Cat Loss':>10} {'Play Acc':>10} {'Weights':>15}")
    print(f"  {'-'*72}")
    for name, r in all_results.items():
        play_acc = f"{r['topic']['accuracy']:.1%}" if r.get('topic') else "N/A"
        if 'final_weights' in r:
            w = f"{r['final_weights'][0]:.3f}/{r['final_weights'][1]:.3f}"
        else:
            w = "1.000/0.100"
        print(f"  {name:<25} {r['best_val']:>10.4f} {r['final_cat']:>10.4f} {play_acc:>10} {w:>15}")

    # Save
    out_dir = Path("results/nash_objectives")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
