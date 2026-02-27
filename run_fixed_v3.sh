#!/bin/bash
set -e
cd /Data/Code/HRS
LOG="results/ablation_3_7_v3.log"
echo "=== Ablation configs 3-7 (fixed router + causal conv) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

for config in dual_head_router dual_head_router_sink full_core full_hrs full_hrs_refined; do
    echo "" | tee -a "$LOG"
    echo "==========================================" | tee -a "$LOG"
    echo "  Starting: $config" | tee -a "$LOG"
    echo "  Time: $(date)" | tee -a "$LOG"
    echo "==========================================" | tee -a "$LOG"
    stdbuf -oL python3 train.py --ablation "$config" --output-dir results 2>&1 | tee -a "$LOG" | tee "results/${config}.log"
    echo "  Completed: $config at $(date)" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== All ablation runs complete at $(date) ===" | tee -a "$LOG"
