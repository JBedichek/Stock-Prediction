#!/bin/bash
# Example training script for Phase 2 RL agent
#
# Phase 2: Full 5-action space with dynamic exit timing
# Goal: Learn position sizing and when to exit positions
#
# Before running:
# 1. Update paths below to match your data files
# 2. Ensure you have a trained predictor checkpoint
# 3. Optionally: Use Phase 1 checkpoint as initialization (advanced)
# 4. Make script executable: chmod +x example_train_phase2.sh

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Data files
DATASET_PATH="all_complete_dataset_temporal_split_d2c2e63d.h5"  # Main dataset with features
PRICES_PATH="actual_prices.h5"  # Optional: actual prices if features are normalized
NUM_TEST_STOCKS=1000  # Number of test stocks to use (Phase 2: use more stocks)

# Predictor checkpoint (trained price predictor)
PREDICTOR_CHECKPOINT="checkpoints/best_model.pt"

# Output directory for RL checkpoints
CHECKPOINT_DIR="./rl_checkpoints_phase2"

# Optional: Load Phase 1 checkpoint for warm start (advanced)
# PHASE1_CHECKPOINT="./rl_checkpoints_phase1/best_episode_500.pt"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Phase 2 specific
NUM_STOCKS=1000          # Use 1,000 stocks (10x more than Phase 1)
FREEZE_PREDICTOR=true    # Set to false for joint training (Phase 3)

# Training
NUM_EPISODES=1000        # More episodes for harder task
EPISODE_LENGTH=60        # Longer episodes (60 days vs 40)
BATCH_SIZE=512           # Larger batch size
BUFFER_CAPACITY=200000   # Larger replay buffer

# RL hyperparameters (tuned for Phase 2)
LR_Q_NETWORK=0.00005     # Lower learning rate (5e-5)
LR_PREDICTOR=0.000001    # For joint training only (1e-6)
GAMMA=0.99               # Discount factor
TAU=0.005                # Soft target update rate
EPSILON_START=0.5        # Lower initial exploration (Phase 2: start at 50%)
EPSILON_END=0.01         # Lower final exploration (1%)
EPSILON_DECAY=100000     # Longer decay period

# Environment
INITIAL_CAPITAL=100000   # Starting capital ($100k)
MAX_POSITIONS=20         # More positions allowed (20 vs 10 in Phase 1)
TRANSACTION_COST=0.001   # Transaction cost (0.1%)

# Architecture
STATE_DIM=1920           # State dimension (predictor features + portfolio context)
HIDDEN_DIM=1024          # Hidden layer dimension for Q-network

# Logging
LOG_FREQUENCY=10         # Log every 10 episodes
EVAL_FREQUENCY=50        # Evaluate every 50 episodes
EVAL_EPISODES=5          # Number of evaluation episodes
CHECKPOINT_FREQUENCY=100 # Save checkpoint every 100 episodes

# W&B (optional - comment out --use-wandb if not using)
WANDB_PROJECT="rl-trading-phase2"
RUN_NAME="phase2_$(date +%Y%m%d_%H%M%S)"

# Device
DEVICE="cuda"            # Use "cpu" if no GPU available

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "========================================================================"
echo "RL TRADING AGENT - PHASE 2 TRAINING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Stocks: $NUM_STOCKS"
echo "  Episodes: $NUM_EPISODES"
echo "  Episode length: $EPISODE_LENGTH days"
echo "  Actions: 5 (HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL)"
echo "  Exit timing: DYNAMIC (agent learns when to sell)"
echo "  Max positions: $MAX_POSITIONS"
echo "  Predictor: $([ "$FREEZE_PREDICTOR" = true ] && echo "FROZEN" || echo "JOINT TRAINING")"
echo ""
echo "Data:"
echo "  Dataset: $DATASET_PATH"
echo "  Prices: $PRICES_PATH"
echo "  Test stocks: $NUM_TEST_STOCKS"
echo "  Predictor: $PREDICTOR_CHECKPOINT"
echo ""
echo "Output:"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  W&B project: $WANDB_PROJECT"
echo ""
echo "Key improvements over Phase 1:"
echo "  ✓ Full action space (5 actions vs 2)"
echo "  ✓ Dynamic exit timing (no fixed holding period)"
echo "  ✓ 10x more stocks (1,000 vs 100)"
echo "  ✓ 2x more positions (20 vs 10)"
echo "  ✓ Longer episodes (60 days vs 40)"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

python -m rl.train_rl_phase2 \
    --dataset-path "$DATASET_PATH" \
    --prices-path "$PRICES_PATH" \
    --num-test-stocks "$NUM_TEST_STOCKS" \
    --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
    --num-stocks "$NUM_STOCKS" \
    --freeze-predictor "$FREEZE_PREDICTOR" \
    --num-episodes "$NUM_EPISODES" \
    --episode-length "$EPISODE_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --buffer-capacity "$BUFFER_CAPACITY" \
    --lr-q-network "$LR_Q_NETWORK" \
    --lr-predictor "$LR_PREDICTOR" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --epsilon-start "$EPSILON_START" \
    --epsilon-end "$EPSILON_END" \
    --epsilon-decay-steps "$EPSILON_DECAY" \
    --initial-capital "$INITIAL_CAPITAL" \
    --max-positions "$MAX_POSITIONS" \
    --transaction-cost "$TRANSACTION_COST" \
    --state-dim "$STATE_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --log-frequency "$LOG_FREQUENCY" \
    --eval-frequency "$EVAL_FREQUENCY" \
    --eval-episodes "$EVAL_EPISODES" \
    --checkpoint-frequency "$CHECKPOINT_FREQUENCY" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME" \
    --device "$DEVICE" \
    --seed 42

echo ""
echo "========================================================================"
echo "TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Best model saved to: $CHECKPOINT_DIR/best_episode_*.pt"
echo ""
echo "Next steps:"
echo "  1. Review W&B dashboard: https://wandb.ai/<your-username>/$WANDB_PROJECT"
echo "  2. Run backtest: ./rl/example_backtest.sh (update checkpoint path)"
echo "  3. Compare Phase 2 vs Phase 1 performance"
echo "  4. If successful (>3% improvement over Phase 1), consider Phase 3 (joint training)"
echo ""
echo "Success criteria for Phase 2:"
echo "  - Mean return: >15% per 60-day episode"
echo "  - Sharpe ratio: >1.5"
echo "  - Win rate: >55%"
echo "  - Improvement over Phase 1: >3%"
echo ""
