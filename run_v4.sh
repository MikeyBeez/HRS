#!/bin/bash
# HRS Ablation v4: All fixes applied
# - Softmax + Gumbel-softmax routing (no Sinkhorn)
# - Causal ConvTier (left-only padding)
# - Entropy regularization (0.01)
# - Tier output gate (init 0.1)
# - Stronger balance loss (0.1), slower anneal (40K, tau_min=0.3)
#
# Runs configs 3-7: dual_head_router through full_hrs_refined

set -e
cd /Data/Code/HRS

CONFIGS=(
    "dual_head_router"
    "dual_head_router_sink"
    "full_core"
    "full_hrs"
    "full_hrs_refined"
)

for cfg in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Starting: $cfg"
    echo "Time: $(date)"
    echo "========================================"
    python3 train.py --ablation "$cfg" 2>&1
    echo ""
    echo "$cfg completed at $(date)"
    echo ""
done

echo "All ablation runs complete at $(date)"
