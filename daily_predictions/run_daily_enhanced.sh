#!/bin/bash
#
# Enhanced Automated Daily Stock Predictions
#
# Features:
# 1. Detects ALL missing dates (not just today)
# 2. Fetches data for all missing dates
# 3. Applies EXACT SAME normalization as training
# 4. Appends all missing data in one batch
# 5. Runs predictions
#
# Usage: ./run_daily_enhanced.sh [number_of_stocks]
#

set -e

TOP_K=${1:-10}
DATASET="data/all_complete_dataset.h5"
PRICES="data/actual_prices.h5"
MODEL="checkpoints/best_model_100m_1.18.pt"
BIN_EDGES="data/adaptive_bin_edges.pt"

echo "========================================"
echo "Enhanced Automated Stock Predictions"
echo "========================================"
echo "Date: $(date)"
echo "Top-K: ${TOP_K} stocks"
echo ""

# Run enhanced pipeline
python daily_predictions/auto_daily_predictions_enhanced.py \
    --dataset "${DATASET}" \
    --prices "${PRICES}" \
    --model "${MODEL}" \
    --bin-edges "${BIN_EDGES}" \
    --top-k ${TOP_K} \
    --horizon-idx 1 \
    --confidence-percentile 0.005 \
    --batch-size 256 \
    --device cuda

echo ""
echo "========================================"
echo "✅ Complete!"
echo "========================================"
echo ""
echo "Key features:"
echo "  ✅ Auto-detected missing dates"
echo "  ✅ Fetched all missing data"
echo "  ✅ Applied training-matched normalization"
echo "  ✅ Updated dataset incrementally"
echo "  ✅ Generated predictions"
