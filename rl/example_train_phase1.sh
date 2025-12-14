#!/bin/bash
# Example training script for Phase 1 RL agent
#
# Phase 1: Simplified system with 2 actions (HOLD, BUY_FULL)
# Goal: Prove RL can learn stock selection better than random
#
# Before running:
# 1. Update paths below to match your data files
# 2. Ensure you have a trained predictor checkpoint
# 3. Make script executable: chmod +x example_train_phase1.sh

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Data files
DATASET_PATH="all_complete_dataset_temporal_split_d2c2e63d.h5"  # Main dataset with features
PRICES_PATH="actual_prices.h5"  # Optional: actual prices if features are normalized
NUM_TEST_STOCKS=1000  # Number of test stocks to use

# Predictor checkpoint (trained price predictor)
PREDICTOR_CHECKPOINT="checkpoints/best_model.pt"

# Output directory for RL checkpoints
CHECKPOINT_DIR="./rl_checkpoints_phase1"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Phase 1 specific
NUM_STOCKS=100           # Subsample 100 stocks (Phase 1 limitation)
HOLDING_PERIOD=5         # Auto-sell after 5 days (Phase 1 constraint)

# Training
NUM_EPISODES=500         # Number of training episodes
EPISODE_LENGTH=40        # Each episode = 40 trading days
BATCH_SIZE=256           # Training batch size
BUFFER_CAPACITY=100000   # Replay buffer capacity

# RL hyperparameters
LR_Q_NETWORK=0.0001      # Learning rate for Q-network
GAMMA=0.99               # Discount factor
TAU=0.005                # Soft target update rate
EPSILON_START=1.0        # Initial exploration rate (100% random)
EPSILON_END=0.05         # Final exploration rate (5% random)
EPSILON_DECAY=50000      # Steps to decay epsilon

# Environment
INITIAL_CAPITAL=100000   # Starting capital ($100k)
MAX_POSITIONS=10         # Max simultaneous positions
TOP_K_BUYS=10            # Top-K stocks to buy per step

# Architecture
STATE_DIM=1920           # State dimension (predictor features + portfolio context)
HIDDEN_DIM=1024          # Hidden layer dimension for Q-network

# Logging
LOG_FREQUENCY=10         # Log every 10 episodes
EVAL_FREQUENCY=50        # Evaluate every 50 episodes
EVAL_EPISODES=5          # Number of evaluation episodes
CHECKPOINT_FREQUENCY=100 # Save checkpoint every 100 episodes

# W&B (optional - comment out --use-wandb if not using)
WANDB_PROJECT="rl-trading-phase1"
RUN_NAME="phase1_$(date +%Y%m%d_%H%M%S)"

# Device
DEVICE="cuda"            # Use "cpu" if no GPU available

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "========================================================================"
echo "RL TRADING AGENT - PHASE 1 TRAINING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Stocks: $NUM_STOCKS"
echo "  Episodes: $NUM_EPISODES"
echo "  Actions: 2 (HOLD, BUY_FULL)"
echo "  Holding period: $HOLDING_PERIOD days (auto-sell)"
echo "  Predictor: FROZEN"
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
echo "Starting training in 3 seconds..."
sleep 3

python -m rl.train_rl \
    --dataset-path "$DATASET_PATH" \
    --prices-path "$PRICES_PATH" \
    --num-test-stocks "$NUM_TEST_STOCKS" \
    --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
    --num-stocks "$NUM_STOCKS" \
    --holding-period "$HOLDING_PERIOD" \
    --num-episodes "$NUM_EPISODES" \
    --episode-length "$EPISODE_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --buffer-capacity "$BUFFER_CAPACITY" \
    --lr-q-network "$LR_Q_NETWORK" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --epsilon-start "$EPSILON_START" \
    --epsilon-end "$EPSILON_END" \
    --epsilon-decay-steps "$EPSILON_DECAY" \
    --initial-capital "$INITIAL_CAPITAL" \
    --max-positions "$MAX_POSITIONS" \
    --top-k-buys "$TOP_K_BUYS" \
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
echo "  2. Evaluate model performance vs. random baseline"
echo "  3. If successful, proceed to Phase 2 (full 5-action space)"
echo ""
