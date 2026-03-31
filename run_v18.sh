#!/usr/bin/env bash
# V18: PEER + Cross-Attention Engram + Categorization Head
# Run from /mnt/data/Code/HRS with .venv activated

set -euo pipefail

python train.py \
    --ablation v18_cross_attn \
    --output-dir results \
    --run-name v18_cross_attn \
    "$@"
