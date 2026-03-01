#!/bin/bash
# HRS v4: PEER as Universal FFN + 3-Tier Routing
#
# v4 architecture:
#   router -> [conv | attn | sink] -> pathway_out    (3-tier, attention pathway only)
#   PEER(x) -> peer_out                              (unconditional, every token)
#   output = pathway_out + peer_out                   (both always applied)
#
# PEER runs unconditionally as FFN (replaces MLP), router only decides attention pathway.
# Same 5-phase schedule as v1/v3 (proven best: v1=9.19, v3=9.46).
# Target: beat v1's 9.19 by letting PEER and attention serve complementary roles.

set -e
cd /Data/Code/HRS

echo "========================================"
echo "Starting: v4_full (PEER universal FFN + 3-tier routing)"
echo "Time: $(date)"
echo "========================================"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py --ablation v4_full --batch-size 8 2>&1
echo ""
echo "v4_full completed at $(date)"
echo ""

echo "All v4 runs complete at $(date)"
