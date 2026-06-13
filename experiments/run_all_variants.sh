#!/bin/bash
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
# Run all 16 prompt variants and collect results

VARIANTS="v01_baseline v02_answer_only v03_cot_explicit v04_expert_role v05_decompose v06_few_shot v07_verify v08_confidence v09_concise v10_structured v11_reframe v12_negative v13_positive v14_meta v15_units v16_final_first"

mkdir -p /workspace/data/variant_results_fresh

echo "Starting 16 variant benchmarks at $(date)"
echo "=============================================="

for v in $VARIANTS; do
    echo "[$v] Starting..."
    python3 /workspace/experiments/run_variant_benchmark.py \
        --variant "$v" \
        --math-limit 100 \
        --logic-limit 50 \
        --output "/workspace/data/variant_results_fresh/${v}_results.json" \
        > "/workspace/data/variant_results_fresh/${v}.log" 2>&1 &
done

echo "All 16 variants launched in parallel"
echo "Waiting for completion..."
wait

echo "=============================================="
echo "All variants complete at $(date)"
echo ""
echo "Results summary:"
for v in $VARIANTS; do
    acc=$(cat "/workspace/data/variant_results_fresh/${v}_results.json" 2>/dev/null | grep -o '"combined_accuracy": [0-9.]*' | cut -d: -f2)
    echo "  $v: ${acc:-FAILED}%"
done
