#!/bin/bash
# HRS v6: Learned Remember Gate + seq_len=1024
# Config 16: v4_1024 (v4 baseline at longer sequences)
# Config 17: v6_gate (v4 + learned remember gate at seq_len=1024)

set -e
cd /Data/Code/HRS

echo "=== Config 16: v4_1024 (baseline at seq_len=1024) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
    --ablation v4_1024 --batch-size 4

echo "=== Config 17: v6_gate (remember gate + seq_len=1024) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
    --ablation v6_gate --batch-size 4
