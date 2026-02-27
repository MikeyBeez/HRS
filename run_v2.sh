#!/bin/bash
# HRS v2: Attention->Conv backbone + PEER FFN + Engrams
#
# Ablation study (6 configs):
# 1. dense_baseline      — Standard transformer, CE only (reuse v1 result: 23.83)
# 2. dual_head           — + locality head (reuse v1 result: 24.08)
# 3. v2_attn_conv        — Attention->Conv backbone, standard MLP, no dual-head
# 4. v2_attn_conv_dual   — + dual-head
# 5. v2_attn_conv_peer   — + PEER FFN (replaces standard MLP)
# 6. v2_full             — + Engrams + phased training
#
# Configs 1-2 reuse v1 results. Configs 3-6 are new.

set -e
cd /Data/Code/HRS

echo "========================================"
echo "HRS v2 Ablation Study"
echo "Started: $(date)"
echo "========================================"
echo ""

# Config 3: Attention->Conv backbone, standard MLP, no dual-head
echo "========================================"
echo "Config 3: v2_attn_conv"
echo "Time: $(date)"
echo "========================================"
python3 train.py --ablation v2_attn_conv 2>&1
echo ""
echo "v2_attn_conv completed at $(date)"
echo ""

# Config 4: + dual-head (locality)
echo "========================================"
echo "Config 4: v2_attn_conv_dual"
echo "Time: $(date)"
echo "========================================"
python3 train.py --ablation v2_attn_conv_dual 2>&1
echo ""
echo "v2_attn_conv_dual completed at $(date)"
echo ""

# Config 5: + PEER FFN
echo "========================================"
echo "Config 5: v2_attn_conv_peer"
echo "Time: $(date)"
echo "========================================"
python3 train.py --ablation v2_attn_conv_peer 2>&1
echo ""
echo "v2_attn_conv_peer completed at $(date)"
echo ""

# Config 6: Full v2 (+ engrams + phased training)
echo "========================================"
echo "Config 6: v2_full"
echo "Time: $(date)"
echo "========================================"
python3 train.py --ablation v2_full 2>&1
echo ""
echo "v2_full completed at $(date)"
echo ""

echo "========================================"
echo "All v2 runs complete at $(date)"
echo "========================================"
