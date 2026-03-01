#!/bin/bash
# HRS v3: PEER + Routing + Engrams
#
# v1 routing framework with PEER as the expert tier.
# Same 5-phase schedule as full_hrs_refined (proven best: 9.19 PPL).
# Target: beat v1's 9.19 by combining PEER's sparse experts with routing geometry.

set -e
cd /Data/Code/HRS

echo "========================================"
echo "Starting: v3_full (PEER + routing + engrams)"
echo "Time: $(date)"
echo "========================================"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py --ablation v3_full --batch-size 8 2>&1
echo ""
echo "v3_full completed at $(date)"
echo ""

echo "All v3 runs complete at $(date)"
