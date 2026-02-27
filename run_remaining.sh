#!/bin/bash
# Run configs 3-7 sequentially
set -e
cd /Data/Code/HRS

for config in dual_head_router dual_head_router_sink full_core full_hrs full_hrs_refined; do
    echo "=========================================="
    echo "  Starting: $config"
    echo "  Time: $(date)"
    echo "=========================================="
    python3 train.py --ablation "$config" --output-dir results
    echo "  Completed: $config at $(date)"
    echo ""
done

echo "=== All remaining ablation runs complete ==="
