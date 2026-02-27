#!/bin/bash
# HRS v5: Engram refinement fix + TRC test
#
# 1. full_hrs_refined: now actually freezes engram encoder + reinits injector at P5
# 2. full_hrs_trc: full_hrs with TRC low-pass filter (window=8)

set -e
cd /Data/Code/HRS

echo "========================================"
echo "Starting: full_hrs_refined (with actual refinement)"
echo "Time: $(date)"
echo "========================================"
python3 train.py --ablation full_hrs_refined 2>&1
echo ""
echo "full_hrs_refined completed at $(date)"
echo ""

echo "========================================"
echo "Starting: full_hrs + TRC (window=8)"
echo "Time: $(date)"
echo "========================================"
python3 train.py --ablation full_hrs --trc-window 8 --run-name full_hrs_trc 2>&1
echo ""
echo "full_hrs_trc completed at $(date)"
echo ""

echo "All v5 runs complete at $(date)"
