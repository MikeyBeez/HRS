#!/bin/bash
# Run all 7 HRS ablation experiments sequentially.
# Each run trains for ~50K steps on WikiText-103.
# Total estimated time: ~30-40 hours on RTX 5070 Ti.
#
# Usage:
#   ./run_ablation.sh                  # Run all 7
#   ./run_ablation.sh 1                # Run only config 1
#   ./run_ablation.sh 3 5              # Run configs 3 through 5

set -e
cd "$(dirname "$0")"

CONFIGS=(
    "dense_baseline"
    "dual_head"
    "dual_head_router"
    "dual_head_router_sink"
    "full_core"
    "full_hrs"
    "full_hrs_refined"
)

START=${1:-1}
END=${2:-7}

echo "=== HRS Ablation Study ==="
echo "Running configs $START through $END"
echo "Output directory: results/"
echo ""

for i in $(seq $START $END); do
    idx=$((i - 1))
    config="${CONFIGS[$idx]}"
    echo "=========================================="
    echo "  Run $i/7: $config"
    echo "  Started: $(date)"
    echo "=========================================="

    python3 train.py \
        --ablation "$config" \
        --output-dir results \
        2>&1 | tee "results/${config}.log"

    echo ""
    echo "  Completed: $(date)"
    echo ""
done

echo "=== All ablation runs complete ==="
echo "Results in results/"
