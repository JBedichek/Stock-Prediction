#!/bin/bash
# Example training script for Phase 1 RL agent with Multi-GPU
#
# Phase 1: Simplified system with 2 actions (HOLD, BUY_FULL)
# Multi-GPU: Asynchronous episode collection on 2 GPUs
#
# Before running:
# 1. Update paths below to match your data files
# 2. Ensure you have a trained predictor checkpoint
# 3. Ensure you have 2 GPUs available
# 4. Make script executable: chmod +x example_train_phase1_multigpu.sh

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Data files
DATASET_PATH="all_complete_dataset_temporal_split_d2c2e63d.h5"  # Main dataset with features
PRICES_PATH="actual_prices.h5"  # Optional: actual prices if features are normalized
NUM_TEST_STOCKS=1000  # Number of test stocks to use

# Feature cache (optional - speeds up worker initialization)
FEATURES_CACHE="./data/preloaded_features.h5"

# Predictor checkpoint (trained price predictor)
PREDICTOR_CHECKPOINT="checkpoints/best_model.pt"

# Output directory for RL checkpoints
CHECKPOINT_DIR="./rl_checkpoints_phase1_multigpu"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Phase 1 specific
NUM_STOCKS=100           # Subsample 100 stocks (Phase 1 limitation)
HOLDING_PERIOD=5         # Auto-sell after 5 days (Phase 1 constraint)

# Training
NUM_EPISODES=500         # Number of training episodes (collected by workers)
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
STATE_DIM=1469           # State dimension (1444 predictor + 25 portfolio context)
HIDDEN_DIM=1024          # Hidden layer dimension for Q-network

# Logging
LOG_FREQUENCY=10         # Log every 10 training steps
CHECKPOINT_FREQUENCY=5000 # Save checkpoint every 5000 training steps

# Multi-GPU settings
NUM_WORKERS=2            # Number of worker processes (1 per GPU)
WORKER_GPUS="0,1"        # GPUs to use (comma-separated)
WEIGHT_SYNC_FREQ=500     # Sync weights every N steps

# W&B (optional - comment out --use-wandb if not using)
WANDB_PROJECT="rl-trading-phase1-multigpu"
RUN_NAME="phase1_multigpu_$(date +%Y%m%d_%H%M%S)"

# Device (main process uses GPU 0 for training)
DEVICE="cuda"

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "========================================================================"
echo "RL TRADING AGENT - PHASE 1 MULTI-GPU TRAINING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Stocks: $NUM_STOCKS"
echo "  Episodes: $NUM_EPISODES"
echo "  Actions: 2 (HOLD, BUY_FULL)"
echo "  Holding period: $HOLDING_PERIOD days (auto-sell)"
echo "  Predictor: FROZEN"
echo ""
echo "Multi-GPU:"
echo "  Workers: $NUM_WORKERS"
echo "  GPUs: $WORKER_GPUS"
echo "  Weight sync frequency: $WEIGHT_SYNC_FREQ steps"
echo "  Expected speedup: ~2.25x"
echo ""
echo "Data:"
echo "  Dataset: $DATASET_PATH"
echo "  Prices: $PRICES_PATH"
echo "  Features cache: $FEATURES_CACHE"
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
    --checkpoint-frequency "$CHECKPOINT_FREQUENCY" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME" \
    --device "$DEVICE" \
    --seed 42 \
    --multi-gpu \
    --num-workers "$NUM_WORKERS" \
    --worker-gpus "$WORKER_GPUS" \
    --weight-sync-frequency "$WEIGHT_SYNC_FREQ"

echo ""
echo "========================================================================"
echo "TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review W&B dashboard: https://wandb.ai/<your-username>/$WANDB_PROJECT"
echo "  2. Compare multi-GPU speedup vs single-GPU"
echo "  3. Evaluate model performance vs. random baseline"
echo "  4. If successful, proceed to Phase 2 multi-GPU training"
echo ""
