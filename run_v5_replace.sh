#!/bin/bash
# HRS v5: Learnable engram-based context replacement
# v4 base (PEER + 3-tier routing) with in-place engram blending
# 38K steps: P1(8K) + P2(8K) + P3(10K) + P4(12K), P5 skipped

set -e

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
    --ablation v5_replace \
    --batch-size 8
