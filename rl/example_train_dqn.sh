#!/bin/bash
# Example script to train the simplified DQN agent
#
# This is MUCH simpler than actor-critic:
# - One network (Q-network)
# - One loss function (TD error)
# - Direct Q-value prediction
# - 9 actions: HOLD + 4 LONG + 4 SHORT

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "========================================="
echo "  Simplified DQN Training"
echo "========================================="
echo "Working directory: $(pwd)"
echo ""

# Check for required files
if [ ! -f "data/prices.h5" ]; then
    echo "ERROR: data/prices.h5 not found!"
    exit 1
fi

if [ ! -f "data/rl_stock_selections_4yr.h5" ]; then
    echo "ERROR: data/rl_stock_selections_4yr.h5 not found!"
    echo "Run generate_stock_selections.py first"
    exit 1
fi

# Training parameters
NUM_EPISODES=10000
NUM_PARALLEL=32
BATCH_SIZE=1024
LEARNING_RATE=0.0001
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=5000
GAMMA=0.99

echo ""
echo "Configuration:"
echo "  Episodes: $NUM_EPISODES"
echo "  Parallel envs: $NUM_PARALLEL"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Exploration: ε=$EPSILON_START → $EPSILON_END over $EPSILON_DECAY episodes"
echo "  Action space: 9 (HOLD + 4 LONG + 4 SHORT)"
echo ""

python -c "
import torch
import h5py
from rl.train_dqn_simple import SimpleDQNTrainer
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.rl_environment import StockTradingEnv
from rl.gpu_stock_cache import GPUStockSelectionCache

# Config
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'state_dim': 1469,
    'hidden_dim': 1024,
    'learning_rate': ${LEARNING_RATE},
    'weight_decay': 1e-5,
    'buffer_capacity': 100000,
    'batch_size': ${BATCH_SIZE},
    'gamma': ${GAMMA},
    'n_step': 1,
    'use_per': False,
    'epsilon_start': ${EPSILON_START},
    'epsilon_end': ${EPSILON_END},
    'epsilon_decay_episodes': ${EPSILON_DECAY},
    'target_update_freq': 1000,
    'updates_per_step': 1,
    'min_buffer_size': 1000,
    'num_episodes': ${NUM_EPISODES},
    'num_parallel_envs': ${NUM_PARALLEL},
    'initial_capital': 100000,
    'reward_scale': 1.0,
    'use_wandb': False,
    'save_freq': 100,
    'top_k_per_horizon': 50
}

print('Initializing DQN trainer...')
trainer = SimpleDQNTrainer(config)

print('Creating reference environment...')
ref_env = StockTradingEnv(
    prices_file='data/prices.h5',
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=config['initial_capital'],
    transaction_cost=0.001,
    allow_short=True,
    device=config['device']
)

print('Creating vectorized environment...')
vec_env = GPUVectorizedTradingEnv(
    ref_env=ref_env,
    num_envs=${NUM_PARALLEL},
    device=config['device']
)

print('Loading GPU stock selection cache...')
with h5py.File('data/prices.h5', 'r') as f:
    prices_file_keys = list(f.keys())

stock_cache = GPUStockSelectionCache(
    h5_path='data/rl_stock_selections_4yr.h5',
    prices_file_keys=prices_file_keys,
    device=config['device']
)

print('Starting training...')
trainer.train(
    vec_env=vec_env,
    stock_selections_cache=stock_cache,
    use_precomputed_selections=True
)

print('Training complete!')
"

echo ""
echo "========================================="
echo "  Training Complete!"
echo "========================================="
