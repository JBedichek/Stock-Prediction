"""
Script to run comprehensive evaluation of trained DQN agent.

Usage:
    python -m rl.evaluation.run_evaluation \
        --checkpoint checkpoints/dqn_best.pt \
        --test-start 2024-01-01 \
        --test-end 2024-12-31 \
        --num-episodes 100

"""

import os
import sys
import argparse
import torch
import h5py
from datetime import datetime
from typing import Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.evaluation.evaluator import DQNEvaluator
from rl.train_dqn_simple import SimpleDQNTrainer
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.rl_environment import TradingEnvironment
from rl.rl_components import TradingAgent
from inference.backtest_simulation import DatasetLoader


def load_trained_dqn(checkpoint_path: str, config: Dict) -> SimpleDQNTrainer:
    """
    Load trained DQN agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: DQN configuration

    Returns:
        Loaded DQN trainer
    """
    print(f"\nLoading DQN checkpoint from: {checkpoint_path}")

    trainer = SimpleDQNTrainer(config)

    checkpoint = torch.load(checkpoint_path, map_location=config['device'])

    trainer.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    trainer.target_network.load_state_dict(checkpoint['target_network_state_dict'])

    print(f"  ✅ Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/dqn_best.pt",
                       help='Path to DQN checkpoint file')

    # Test period
    parser.add_argument('--test-start', type=str, default='2024-01-01',
                       help='Test period start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, default='2024-12-31',
                       help='Test period end date (YYYY-MM-DD)')

    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--episode-length', type=int, default=30,
                       help='Episode length in trading days')

    # Environment setup
    parser.add_argument('--dataset', type=str, default='data/all_complete_dataset.h5',
                       help='Path to dataset')
    parser.add_argument('--prices', type=str, default='data/actual_prices_clean.h5',
                       help='Path to prices file (use _clean.h5 for validated data)')
    parser.add_argument('--stock-selections', type=str, default='data/rl_stock_selections_4yr.h5',
                       help='Path to stock selections cache')
    parser.add_argument('--state-cache', type=str, default='data/rl_state_cache_4yr.h5',
                       help='Path to state cache')
    parser.add_argument('--predictor-checkpoint', type=str,
                       default='./checkpoints/best_model_100m_1.18.pt',
                       help='Path to predictor checkpoint')

    # Resources
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel environments (use 1 for evaluation)')

    # Baselines
    parser.add_argument('--baselines', type=str, nargs='+',
                       default=['random', 'hold', 'long', 'random_universe'],
                       help='Baselines to compare against')

    # Output
    parser.add_argument('--output-dir', type=str, default='rl/evaluation/results',
                       help='Output directory for results')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("DQN AGENT EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test period: {args.test_start} to {args.test_end}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Episode length: {args.episode_length} days")
    print(f"Baselines: {', '.join(args.baselines)}")
    print("="*80 + "\n")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # DQN config (must match training config)
    allow_short = False  # Match training config (prevents negative portfolio values)

    # State dimension depends on whether short selling is enabled:
    # - With shorts: 4 longs × 1469 + 4 shorts × 1469 + 9 position = 11761
    # - Without shorts: 4 longs × 1469 + 5 position = 5881
    state_dim = 11761 if allow_short else 5881

    config = {
        'device': device,
        'state_dim': state_dim,
        'hidden_dim': 1024,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'buffer_capacity': 100000,
        'batch_size': 1024,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_episodes': 5000,
        'target_update_freq': 1000,
        'initial_capital': 100000,
        'top_k_per_horizon': 50,
        'allow_short': allow_short
    }

    # Load dataset
    print("Loading dataset...")
    data_loader = DatasetLoader(
        dataset_path=args.dataset,
        prices_path=args.prices,
        num_test_stocks=100
    )

    # Count stocks
    all_tickers = list(data_loader.h5_file.keys()) if data_loader.is_hdf5 else list(data_loader.data.keys())
    stocks_with_prices = sum(1 for ticker in all_tickers if ticker in data_loader.prices_file)
    print(f"  ✅ Loaded {len(all_tickers)} total stocks")
    print(f"  ✅ {stocks_with_prices} stocks have price data")

    # Create dummy agent for feature extraction
    print("\nCreating dummy agent for feature extraction...")
    dummy_agent = TradingAgent(
        predictor_checkpoint_path=args.predictor_checkpoint,
        state_dim=1469,
        hidden_dim=512,
        action_dim=5
    ).to(device)
    dummy_agent.train()  # Keep in train mode for gradients
    print("  ✅ Dummy agent created")

    # Load stock selection cache
    print("\nLoading stock selection cache...")
    with h5py.File(args.prices, 'r') as f:
        prices_file_keys = list(f.keys())

    stock_cache = GPUStockSelectionCache(
        h5_path=args.stock_selections,
        prices_file_keys=prices_file_keys,
        device=device
    )
    print(f"  ✅ Stock cache loaded ({len(stock_cache)} dates)")

    # Create temporary environment to load state cache
    print("\nCreating temporary environment for state cache...")
    temp_env = TradingEnvironment(
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=args.episode_length,
        device=device,
        transaction_cost=0.0
    )

    # Load state cache
    if os.path.exists(args.state_cache):
        print(f"Loading state cache from {args.state_cache}...")
        temp_env.load_state_cache(args.state_cache)
        print(f"  ✅ State cache loaded")
    else:
        print(f"WARNING: State cache not found at {args.state_cache}")
        print("Proceeding without state cache (will be slower)")

    # Filter to test period trading days
    all_trading_days = sorted(stock_cache.selections.keys())
    test_trading_days = [
        d for d in all_trading_days
        if args.test_start <= d <= args.test_end
    ]

    print(f"\nTest period: {len(test_trading_days)} trading days")

    # Create vectorized environment
    print(f"\nCreating GPU-vectorized environment...")
    vec_env = GPUVectorizedTradingEnv(
        num_envs=args.parallel,
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=args.episode_length,
        transaction_cost=0.0,
        device=device,
        trading_days_filter=test_trading_days,
        top_k_per_horizon=config['top_k_per_horizon']
    )

    # Share caches
    vec_env.ref_env.state_cache = temp_env.state_cache
    vec_env.ref_env.price_cache = temp_env.price_cache

    print("  ✅ Vectorized environment created")

    # Load trained DQN
    print("\nLoading trained DQN agent...")
    trainer = load_trained_dqn(args.checkpoint, config)

    # Get all tickers for random_universe baseline
    all_tickers = [ticker for ticker in all_tickers if ticker in data_loader.prices_file]
    print(f"\n  ✅ {len(all_tickers)} tickers with price data (for random_universe baseline)")

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = DQNEvaluator(
        agent=trainer,
        vec_env=vec_env,
        stock_selections_cache=stock_cache,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        initial_capital=config['initial_capital'],
        device=device,
        all_tickers=all_tickers
    )

    # Run full evaluation
    print("\nStarting evaluation...")
    report_path = evaluator.run_full_evaluation(baseline_names=args.baselines)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
