#!/bin/bash
# v7: Full HRS + Memory MLP + V7Router (from-scratch training)
# v4 base (PEER + 3-tier routing + engrams) with Memory MLP trained independently via SGD
# and V7Router blending base + memory logits, trained end-to-end via Adam.
#
# Memory MLP starts training once theta is auto-calibrated (~step 100).
# V7Router is frozen in P1-P2 (steps 0-16K), wakes at P3 (16K), full at P4 (26K).
# P5 skipped (always regresses in prior experiments).
#
# Expected: ~7-8 hours on RTX 5070 Ti, 38K steps total.

set -e

mkdir -p results/v7_full

python3 train.py \
    --ablation v7_full \
    --batch-size 8 \
    --output-dir results \
    2>&1 | tee results/v7_full/train.log
