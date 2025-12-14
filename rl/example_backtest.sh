#!/bin/bash
# Example backtesting script for trained RL agent
#
# This script evaluates a trained RL agent and compares it against baselines.
#
# Before running:
# 1. Update paths below to match your data files
# 2. Ensure you have a trained RL checkpoint
# 3. Make script executable: chmod +x example_backtest.sh

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Trained RL checkpoint (from training)
CHECKPOINT_PATH="./rl_checkpoints_phase1/best_episode_500.pt"

# Predictor checkpoint (same as used in training)
PREDICTOR_CHECKPOINT="checkpoints/best_model.pt"

# Data files (same as used in training)
DATASET_PATH="all_complete_dataset_temporal_split_d2c2e63d.h5"
PRICES_PATH="actual_prices.h5"
NUM_TEST_STOCKS=1000

# Output directory for backtest results
OUTPUT_DIR="./backtest_results"

# =============================================================================
# BACKTEST PARAMETERS
# =============================================================================

# Number of test episodes to run
NUM_EPISODES=20

# Episode length (should match training)
EPISODE_LENGTH=40

# Compare against baselines?
COMPARE_BASELINES="--compare-baselines"
NUM_BASELINE_EPISODES=10

# Generate outputs
SAVE_RESULTS="--save-results"
PLOT_RESULTS="--plot-results"

# Environment (should match training)
INITIAL_CAPITAL=100000
MAX_POSITIONS=10

# Device
DEVICE="cuda"

# =============================================================================
# RUN BACKTEST
# =============================================================================

echo "========================================================================"
echo "RL TRADING AGENT - BACKTESTING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Predictor: $PREDICTOR_CHECKPOINT"
echo "  Episodes: $NUM_EPISODES"
echo "  Compare baselines: $([ -z "$COMPARE_BASELINES" ] && echo "No" || echo "Yes")"
echo ""
echo "Data:"
echo "  Dataset: $DATASET_PATH"
echo "  Prices: $PRICES_PATH"
echo "  Test stocks: $NUM_TEST_STOCKS"
echo ""
echo "Output:"
echo "  Directory: $OUTPUT_DIR"
echo ""
echo "Starting backtest in 3 seconds..."
sleep 3

python -m rl.rl_backtest \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --prices-path "$PRICES_PATH" \
    --num-test-stocks "$NUM_TEST_STOCKS" \
    --num-episodes "$NUM_EPISODES" \
    --episode-length "$EPISODE_LENGTH" \
    $COMPARE_BASELINES \
    --num-baseline-episodes "$NUM_BASELINE_EPISODES" \
    --initial-capital "$INITIAL_CAPITAL" \
    --max-positions "$MAX_POSITIONS" \
    --output-dir "$OUTPUT_DIR" \
    $SAVE_RESULTS \
    $PLOT_RESULTS \
    --device "$DEVICE"

echo ""
echo "========================================================================"
echo "BACKTEST COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Files generated:"
echo "  - backtest_results.json: Detailed results in JSON format"
echo "  - returns_distribution.png: Return distribution histogram"
echo "  - metrics_summary.png: Performance metrics summary"
echo ""
echo "Next steps:"
echo "  1. Review results in $OUTPUT_DIR/"
echo "  2. Compare RL agent vs baselines"
echo "  3. Analyze trade-by-trade performance"
echo "  4. If Phase 1 successful, proceed to Phase 2 (full action space)"
echo ""
