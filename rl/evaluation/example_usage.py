"""
Example usage of the DQN evaluation framework.

This shows how to use the evaluation framework programmatically
instead of using the CLI script.
"""

import torch
import h5py
from rl.evaluation import DQNEvaluator
from rl.train_dqn_simple import SimpleDQNTrainer
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.rl_environment import TradingEnvironment
from rl.rl_components import TradingAgent
from dataset_creation.dataset_loader import DatasetLoader


def main():
    """Example evaluation workflow."""

    # Configuration
    checkpoint_path = 'checkpoints/dqn_best.pt'
    test_start = '2024-01-01'
    test_end = '2024-12-31'
    num_episodes = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("DQN EVALUATION EXAMPLE")
    print("="*80)

    # Load dataset
    print("\n1. Loading dataset...")
    data_loader = DatasetLoader('data/train_data_enhanced.h5')
    print(f"   Loaded {len(data_loader.tickers)} tickers")

    # Create dummy agent for feature extraction
    print("\n2. Creating feature extractor...")
    dummy_agent = TradingAgent(
        predictor_checkpoint_path='checkpoints/contrastive_best.pt',
        state_dim=1469,
        hidden_dim=512,
        action_dim=5
    ).to(device)
    dummy_agent.train()

    # Load stock selection cache
    print("\n3. Loading stock selection cache...")
    with h5py.File('data/prices.h5', 'r') as f:
        prices_file_keys = list(f.keys())

    stock_cache = GPUStockSelectionCache(
        h5_path='data/rl_stock_selections_4yr.h5',
        prices_file_keys=prices_file_keys,
        device=device
    )

    # Create environment
    print("\n4. Creating environment...")
    temp_env = TradingEnvironment(
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=100000,
        max_positions=1,
        episode_length=30,
        device=device,
        transaction_cost=0.0
    )

    # Load state cache
    print("\n5. Loading state cache...")
    temp_env.load_state_cache('data/state_cache.h5')

    # Get test period trading days
    all_dates = sorted(stock_cache.selections.keys())
    test_dates = [d for d in all_dates if test_start <= d <= test_end]

    # Create vectorized environment
    print("\n6. Creating vectorized environment...")
    vec_env = GPUVectorizedTradingEnv(
        num_envs=1,
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=100000,
        max_positions=1,
        episode_length=30,
        transaction_cost=0.0,
        device=device,
        trading_days_filter=test_dates,
        top_k_per_horizon=50
    )

    # Share caches
    vec_env.ref_env.state_cache = temp_env.state_cache
    vec_env.ref_env.price_cache = temp_env.price_cache

    # Load trained DQN
    print("\n7. Loading trained DQN...")
    config = {
        'device': device,
        'state_dim': 11761,
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
        'top_k_per_horizon': 50
    }

    trainer = SimpleDQNTrainer(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trainer.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    trainer.target_network.load_state_dict(checkpoint['target_network_state_dict'])

    # Create evaluator
    print("\n8. Creating evaluator...")
    evaluator = DQNEvaluator(
        agent=trainer,
        vec_env=vec_env,
        stock_selections_cache=stock_cache,
        test_start_date=test_start,
        test_end_date=test_end,
        num_episodes=num_episodes,
        episode_length=30,
        initial_capital=100000,
        device=device
    )

    # Run evaluation
    print("\n9. Running evaluation...")
    report_path = evaluator.run_full_evaluation(
        baseline_names=['random', 'hold', 'long']
    )

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Report: {report_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
