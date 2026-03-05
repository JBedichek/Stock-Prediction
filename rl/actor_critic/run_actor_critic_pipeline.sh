#!/bin/bash
# Complete Actor-Critic Training Pipeline
#
# Runs all three phases:
# 1. Generate synthetic data using programmatic strategy
# 2. Pre-train critic on synthetic data
# 3. Train actor-critic online
#
# Usage: ./run_actor_critic_pipeline.sh

set -e  # Exit on error

echo "========================================================================"
echo "ACTOR-CRITIC TRAINING PIPELINE"
echo "========================================================================"
echo ""

# Phase 1: Generate synthetic data
echo "Phase 1: Generating synthetic training data..."
echo "------------------------------------------------------------------------"
python rl/generate_critic_data.py
echo ""
echo "✅ Phase 1 complete!"
echo ""

# Phase 2: Pre-train critic
echo "Phase 2: Pre-training critic on synthetic data..."
echo "------------------------------------------------------------------------"
python rl/pretrain_critic.py
echo ""
echo "✅ Phase 2 complete!"
echo ""

# Phase 3: Online actor-critic training
echo "Phase 3: Online actor-critic training..."
echo "------------------------------------------------------------------------"
python rl/train_actor_critic.py
echo ""
echo "✅ Phase 3 complete!"
echo ""

echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Checkpoints saved to:"
echo "  - ./checkpoints/pretrained_critic.pt (Phase 2)"
echo "  - ./checkpoints/actor_critic_final.pt (Phase 3)"
echo ""
echo "Training history saved to:"
echo "  - ./checkpoints/actor_critic_training_history.csv"
echo ""
