#!/bin/bash
# Example training script for Phase 2 RL agent with Multi-GPU
#
# Phase 2: Full 5-action space with dynamic exit timing
# Multi-GPU: Asynchronous episode collection on 2 GPUs
#
# Before running:
# 1. Update paths below to match your data files
# 2. Ensure you have a trained predictor checkpoint
# 3. Ensure you have 2 GPUs available
# 4. Make script executable: chmod +x example_train_phase2_multigpu.sh

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Data files
DATASET_PATH="all_complete_dataset_temporal_split_d2c2e63d.h5"  # Main dataset with features
PRICES_PATH="actual_prices.h5"  # Optional: actual prices if features are normalized
NUM_TEST_STOCKS=1000  # Number of test stocks to use (Phase 2: use more stocks)

# Feature cache (optional - speeds up worker initialization)
FEATURES_CACHE="./data/preloaded_features.h5"

# Predictor checkpoint (trained price predictor)
PREDICTOR_CHECKPOINT="checkpoints/best_model.pt"

# Output directory for RL checkpoints
CHECKPOINT_DIR="./rl_checkpoints_phase2_multigpu"

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
WANDB_PROJECT="rl-trading-phase2-multigpu"
RUN_NAME="phase2_multigpu_$(date +%Y%m%d_%H%M%S)"

# Device (main process uses GPU 0 for training)
DEVICE="cuda"

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "========================================================================"
echo "RL TRADING AGENT - PHASE 2 MULTI-GPU TRAINING"
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
echo "Key improvements over Phase 1:"
echo "  ✓ Full action space (5 actions vs 2)"
echo "  ✓ Dynamic exit timing (no fixed holding period)"
echo "  ✓ 10x more stocks (1,000 vs 100)"
echo "  ✓ 2x more positions (20 vs 10)"
echo "  ✓ Longer episodes (60 days vs 40)"
echo "  ✓ Multi-GPU training (~2.25x faster)"
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
echo "  3. Run backtest with trained model"
echo "  4. Compare Phase 2 vs Phase 1 performance"
echo "  5. If successful, consider Phase 3 (joint training with predictor)"
echo ""
echo "Success criteria for Phase 2:"
echo "  - Mean return: >15% per 60-day episode"
echo "  - Sharpe ratio: >1.5"
echo "  - Win rate: >55%"
echo "  - Improvement over Phase 1: >3%"
echo "  - Training time: <50 minutes (vs ~100 min single-GPU)"
echo ""
