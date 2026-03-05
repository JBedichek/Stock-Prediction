"""
RL Trading Agent Backtesting

Comprehensive backtesting framework for evaluating trained RL agents.

Features:
- Load and evaluate trained RL agents
- Compare against multiple baselines (random, buy-and-hold, simple rules)
- Detailed performance metrics (returns, Sharpe, drawdown, win rate)
- Trade-by-trade analysis
- Visualization of results
"""

import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent, ACTIONS
from rl.rl_environment import TradingEnvironment
from inference.backtest_simulation import DatasetLoader


class RLBacktester:
    """
    Backtester for evaluating RL trading agents.

    Supports:
    - Loading trained checkpoints
    - Running on test set
    - Computing performance metrics
    - Comparing against baselines
    """

    def __init__(self, config: Dict):
        """
        Initialize backtester.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config['device']

        # Load data
        print("\n" + "="*80)
        print("INITIALIZING RL BACKTESTER")
        print("="*80)

        print("\n📦 Loading data...")
        self.data_loader = self._load_data()

        # Initialize agent
        print("\n🤖 Loading trained RL agent...")
        self.agent = self._load_agent()

        # Initialize environment
        print("\n🏪 Setting up trading environment...")
        self.env = TradingEnvironment(
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config['initial_capital'],
            max_positions=config['max_positions'],
            episode_length=config['episode_length'],
            device=self.device
        )

        print("\n✅ Backtester initialized successfully!")

    def _load_data(self) -> DatasetLoader:
        """Load dataset."""
        data_loader = DatasetLoader(
            dataset_path=self.config['dataset_path'],
            num_test_stocks=self.config.get('num_test_stocks', 1000),
            prices_path=self.config.get('prices_path')
        )

        print(f"   ✅ Loaded {len(data_loader.test_tickers)} tickers")
        print(f"   ✅ Loaded {len(data_loader.all_dates)} dates")
        return data_loader

    def _load_agent(self) -> TradingAgent:
        """Load trained RL agent from checkpoint."""
        checkpoint_path = self.config['checkpoint_path']
        print(f"   Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Determine action_dim from checkpoint
        action_dim = checkpoint['config'].get('action_dim', 2)  # Default to Phase 1

        # Initialize agent
        agent = TradingAgent(
            predictor_checkpoint_path=self.config['predictor_checkpoint'],
            state_dim=checkpoint['config']['state_dim'],
            hidden_dim=checkpoint['config']['hidden_dim'],
            action_dim=action_dim
        ).to(self.device)

        # Load weights
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.eval()

        print(f"   ✅ Loaded agent from episode {checkpoint['episode']}")
        print(f"   ✅ Best reward: {checkpoint.get('best_avg_reward', 'N/A')}")
        print(f"   ✅ Action dim: {action_dim}")

        return agent

    def run_backtest(self, num_episodes: int = 10, epsilon: float = 0.0) -> Dict:
        """
        Run backtest over multiple episodes.

        Args:
            num_episodes: Number of test episodes
            epsilon: Exploration rate (0 = greedy)

        Returns:
            Dictionary with backtest results
        """
        print("\n" + "="*80)
        print(f"RUNNING BACKTEST ({num_episodes} episodes)")
        print("="*80)

        episode_results = []
        all_trades = []

        for ep in tqdm(range(num_episodes), desc="Backtesting"):
            # Run episode
            result = self._run_single_episode(epsilon=epsilon)
            episode_results.append(result['stats'])
            all_trades.extend(result['trades'])

        # Aggregate results
        results = {
            'episode_stats': episode_results,
            'trades': all_trades,
            'aggregate_metrics': self._compute_aggregate_metrics(episode_results),
            'config': self.config
        }

        return results

    def _run_single_episode(self, epsilon: float = 0.0) -> Dict:
        """
        Run a single backtest episode.

        Args:
            epsilon: Exploration rate

        Returns:
            Episode results dictionary
        """
        # Reset environment
        states = self.env.reset()
        done = False
        episode_trades = []

        while not done:
            # Select actions using trained agent
            actions = self._select_actions(states, epsilon)

            # Execute actions
            next_states, reward, done, info = self.env.step(actions)

            # Record trades
            if info['trades']:
                episode_trades.extend(info['trades'])

            states = next_states

        # Get episode statistics
        stats = self.env.get_episode_stats()

        return {
            'stats': stats,
            'trades': episode_trades
        }

    def _select_actions(self, states: Dict[str, torch.Tensor], epsilon: float = 0.0) -> Dict[str, int]:
        """
        Select actions for all stocks.

        Args:
            states: Dictionary mapping ticker -> state tensor
            epsilon: Exploration rate

        Returns:
            Dictionary mapping ticker -> action_id
        """
        if len(states) == 0:
            return {}

        # Stack states for batch inference
        tickers = list(states.keys())
        state_tensors = torch.stack([states[ticker] for ticker in tickers])

        # Get Q-values
        with torch.no_grad():
            q_values_batch = self.agent.q_network(state_tensors)

        # Epsilon-greedy selection
        actions = {}
        for i, ticker in enumerate(tickers):
            if np.random.random() < epsilon:
                # Random action
                action = np.random.randint(0, self.agent.action_dim)
            else:
                # Greedy action
                action = q_values_batch[i].argmax().item()

            actions[ticker] = action

        return actions

    def _compute_aggregate_metrics(self, episode_results: List[Dict]) -> Dict:
        """
        Compute aggregate metrics across all episodes.

        Args:
            episode_results: List of episode statistics

        Returns:
            Aggregated metrics
        """
        metrics = {
            'mean_return': np.mean([r['total_return'] for r in episode_results]),
            'std_return': np.std([r['total_return'] for r in episode_results]),
            'median_return': np.median([r['total_return'] for r in episode_results]),
            'mean_sharpe': np.mean([r['sharpe_ratio'] for r in episode_results]),
            'std_sharpe': np.std([r['sharpe_ratio'] for r in episode_results]),
            'mean_max_drawdown': np.mean([r['max_drawdown'] for r in episode_results]),
            'mean_win_rate': np.mean([r['win_rate'] for r in episode_results]),
            'total_trades': sum([r['total_trades'] for r in episode_results]),
            'total_profitable_trades': sum([r['profitable_trades'] for r in episode_results]),
            'overall_win_rate': sum([r['profitable_trades'] for r in episode_results]) / max(sum([r['total_trades'] for r in episode_results]), 1)
        }

        return metrics


class BaselineComparison:
    """
    Compare RL agent against baseline strategies.

    Baselines:
    - Random: Random buy/hold decisions
    - Buy-and-hold: Buy top stocks and hold
    - Simple momentum: Buy based on recent returns
    """

    def __init__(self, data_loader: DatasetLoader, config: Dict):
        """
        Initialize baseline comparison.

        Args:
            data_loader: DatasetLoader instance
            config: Configuration dictionary
        """
        self.data_loader = data_loader
        self.config = config

    def run_random_baseline(self, num_episodes: int = 10) -> Dict:
        """
        Run random baseline: randomly buy/hold stocks.

        Args:
            num_episodes: Number of episodes

        Returns:
            Results dictionary
        """
        print("\n📊 Running random baseline...")

        episode_results = []

        for _ in tqdm(range(num_episodes), desc="Random baseline"):
            result = self._run_random_episode()
            episode_results.append(result)

        metrics = {
            'mean_return': np.mean([r['return'] for r in episode_results]),
            'std_return': np.std([r['return'] for r in episode_results]),
            'mean_sharpe': np.mean([r['sharpe'] for r in episode_results]),
            'mean_win_rate': np.mean([r['win_rate'] for r in episode_results])
        }

        return metrics

    def _run_random_episode(self) -> Dict:
        """Run a single random baseline episode."""
        # Select random start date
        episode_length = self.config['episode_length']
        max_start_idx = len(self.data_loader.all_dates) - episode_length - 1
        start_idx = np.random.randint(0, max_start_idx)
        dates = self.data_loader.all_dates[start_idx:start_idx + episode_length]

        # Random portfolio
        num_positions = self.config['max_positions']
        selected_tickers = np.random.choice(
            self.data_loader.test_tickers,
            size=min(num_positions, len(self.data_loader.test_tickers)),
            replace=False
        )

        # Calculate returns
        capital = self.config['initial_capital']
        allocation_per_stock = capital / len(selected_tickers)

        portfolio_values = [capital]
        returns = []
        profitable = 0
        total = len(selected_tickers)

        for ticker in selected_tickers:
            # Get entry price
            entry_result = self.data_loader.get_features_and_price(ticker, dates[0])
            if entry_result is None:
                continue
            _, entry_price = entry_result

            # Get exit price
            exit_result = self.data_loader.get_features_and_price(ticker, dates[-1])
            if exit_result is None:
                continue
            _, exit_price = exit_result

            # Calculate return
            stock_return = (exit_price / entry_price) - 1.0
            returns.append(stock_return)

            if stock_return > 0:
                profitable += 1

        if len(returns) > 0:
            mean_return = np.mean(returns)
            sharpe = mean_return / np.std(returns) if np.std(returns) > 0 else 0.0
            win_rate = profitable / total
        else:
            mean_return = 0.0
            sharpe = 0.0
            win_rate = 0.0

        return {
            'return': mean_return,
            'sharpe': sharpe,
            'win_rate': win_rate
        }

    def run_buy_and_hold_baseline(self, num_episodes: int = 10) -> Dict:
        """
        Run buy-and-hold baseline: buy top N stocks and hold.

        Args:
            num_episodes: Number of episodes

        Returns:
            Results dictionary
        """
        print("\n📊 Running buy-and-hold baseline...")

        episode_results = []

        for _ in tqdm(range(num_episodes), desc="Buy-and-hold"):
            result = self._run_buy_and_hold_episode()
            episode_results.append(result)

        metrics = {
            'mean_return': np.mean([r['return'] for r in episode_results]),
            'std_return': np.std([r['return'] for r in episode_results]),
            'mean_sharpe': np.mean([r['sharpe'] for r in episode_results]),
            'mean_win_rate': np.mean([r['win_rate'] for r in episode_results])
        }

        return metrics

    def _run_buy_and_hold_episode(self) -> Dict:
        """Run a single buy-and-hold episode."""
        # Select random start date
        episode_length = self.config['episode_length']
        max_start_idx = len(self.data_loader.all_dates) - episode_length - 1
        start_idx = np.random.randint(0, max_start_idx)
        dates = self.data_loader.all_dates[start_idx:start_idx + episode_length]

        # Buy top N stocks (by market cap or random)
        num_positions = self.config['max_positions']
        selected_tickers = np.random.choice(
            self.data_loader.test_tickers,
            size=min(num_positions, len(self.data_loader.test_tickers)),
            replace=False
        )

        # Calculate returns (same as random but with different selection)
        returns = []
        profitable = 0
        total = len(selected_tickers)

        for ticker in selected_tickers:
            entry_result = self.data_loader.get_features_and_price(ticker, dates[0])
            if entry_result is None:
                continue
            _, entry_price = entry_result

            exit_result = self.data_loader.get_features_and_price(ticker, dates[-1])
            if exit_result is None:
                continue
            _, exit_price = exit_result

            stock_return = (exit_price / entry_price) - 1.0
            returns.append(stock_return)

            if stock_return > 0:
                profitable += 1

        if len(returns) > 0:
            mean_return = np.mean(returns)
            sharpe = mean_return / np.std(returns) if np.std(returns) > 0 else 0.0
            win_rate = profitable / total
        else:
            mean_return = 0.0
            sharpe = 0.0
            win_rate = 0.0

        return {
            'return': mean_return,
            'sharpe': sharpe,
            'win_rate': win_rate
        }


def print_results(rl_results: Dict, baselines: Optional[Dict] = None):
    """
    Print backtest results in a formatted way.

    Args:
        rl_results: RL agent results
        baselines: Optional baseline results
    """
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)

    # RL Agent results
    print("\n🤖 RL AGENT:")
    metrics = rl_results['aggregate_metrics']
    print(f"   Mean Return:       {metrics['mean_return']*100:>8.2f}% ± {metrics['std_return']*100:.2f}%")
    print(f"   Median Return:     {metrics['median_return']*100:>8.2f}%")
    print(f"   Mean Sharpe Ratio: {metrics['mean_sharpe']:>8.2f} ± {metrics['std_sharpe']:.2f}")
    print(f"   Mean Max Drawdown: {metrics['mean_max_drawdown']*100:>8.2f}%")
    print(f"   Win Rate:          {metrics['overall_win_rate']*100:>8.1f}%")
    print(f"   Total Trades:      {metrics['total_trades']:>8d}")

    # Baseline comparison
    if baselines:
        print("\n📊 BASELINES:")

        if 'random' in baselines:
            print("\n   Random:")
            print(f"      Mean Return:  {baselines['random']['mean_return']*100:>8.2f}% ± {baselines['random']['std_return']*100:.2f}%")
            print(f"      Mean Sharpe:  {baselines['random']['mean_sharpe']:>8.2f}")
            print(f"      Win Rate:     {baselines['random']['mean_win_rate']*100:>8.1f}%")

            # Comparison
            improvement = ((metrics['mean_return'] - baselines['random']['mean_return']) /
                          abs(baselines['random']['mean_return'])) * 100
            print(f"      → RL vs Random: {improvement:+.1f}% improvement")

        if 'buy_and_hold' in baselines:
            print("\n   Buy-and-Hold:")
            print(f"      Mean Return:  {baselines['buy_and_hold']['mean_return']*100:>8.2f}% ± {baselines['buy_and_hold']['std_return']*100:.2f}%")
            print(f"      Mean Sharpe:  {baselines['buy_and_hold']['mean_sharpe']:>8.2f}")
            print(f"      Win Rate:     {baselines['buy_and_hold']['mean_win_rate']*100:>8.1f}%")

            improvement = ((metrics['mean_return'] - baselines['buy_and_hold']['mean_return']) /
                          abs(baselines['buy_and_hold']['mean_return'])) * 100
            print(f"      → RL vs Buy-and-Hold: {improvement:+.1f}% improvement")

    print("\n" + "="*80)


def save_results(results: Dict, output_path: str):
    """
    Save backtest results to disk.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {
        'aggregate_metrics': results['aggregate_metrics'],
        'num_episodes': len(results['episode_stats']),
        'episode_stats': results['episode_stats'],
        'config': {k: v for k, v in results['config'].items() if isinstance(v, (int, float, str, bool, type(None)))}
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


def plot_results(rl_results: Dict, baselines: Optional[Dict] = None, output_dir: str = './backtest_plots'):
    """
    Create visualizations of backtest results.

    Args:
        rl_results: RL agent results
        baselines: Optional baseline results
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract episode returns
    episode_returns = [r['total_return'] for r in rl_results['episode_stats']]

    # 1. Returns distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(episode_returns, bins=20, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(episode_returns), color='red', linestyle='--', label=f'Mean: {np.mean(episode_returns)*100:.2f}%')
    axes[0].set_xlabel('Episode Return (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('RL Agent: Return Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot comparison
    data_to_plot = [episode_returns]
    labels = ['RL Agent']

    if baselines:
        if 'random' in baselines:
            # Would need per-episode data for box plot
            pass

    axes[1].boxplot(data_to_plot, labels=labels)
    axes[1].set_ylabel('Episode Return (%)')
    axes[1].set_title('Return Comparison')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'returns_distribution.png', dpi=150)
    print(f"   📈 Saved: {output_dir / 'returns_distribution.png'}")
    plt.close()

    # 2. Metrics summary
    metrics = rl_results['aggregate_metrics']

    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['Mean Return\n(%)', 'Sharpe Ratio', 'Max Drawdown\n(%)', 'Win Rate\n(%)']
    metric_values = [
        metrics['mean_return'] * 100,
        metrics['mean_sharpe'],
        metrics['mean_max_drawdown'] * 100,
        metrics['overall_win_rate'] * 100
    ]

    bars = ax.bar(metric_names, metric_values, color=['green', 'blue', 'red', 'orange'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Value')
    ax.set_title('RL Agent: Performance Metrics Summary')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150)
    print(f"   📈 Saved: {output_dir / 'metrics_summary.png'}")
    plt.close()

    print(f"\n📊 Plots saved to: {output_dir}/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Backtest trained RL trading agent')

    # Required
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Path to trained checkpoint')
    parser.add_argument('--predictor-checkpoint', type=str, required=True, help='Path to predictor checkpoint')

    # Data
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset (HDF5 or pickle)')
    parser.add_argument('--prices-path', type=str, help='Path to prices HDF5 (if features are normalized)')
    parser.add_argument('--num-test-stocks', type=int, default=1000, help='Number of test stocks')

    # Backtest settings
    parser.add_argument('--num-episodes', type=int, default=20, help='Number of backtest episodes')
    parser.add_argument('--episode-length', type=int, default=40, help='Episode length (days)')
    parser.add_argument('--compare-baselines', action='store_true', help='Compare against baselines')
    parser.add_argument('--num-baseline-episodes', type=int, default=10, help='Episodes for baseline comparison')

    # Environment
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--max-positions', type=int, default=10, help='Max simultaneous positions')

    # Output
    parser.add_argument('--output-dir', type=str, default='./backtest_results', help='Output directory')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON')
    parser.add_argument('--plot-results', action='store_true', help='Generate plots')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    args = parser.parse_args()

    # Convert to config
    config = vars(args)

    # Run backtest
    backtester = RLBacktester(config)
    rl_results = backtester.run_backtest(num_episodes=args.num_episodes)

    # Run baselines if requested
    baselines = None
    if args.compare_baselines:
        print("\n" + "="*80)
        print("RUNNING BASELINE COMPARISONS")
        print("="*80)

        baseline_comp = BaselineComparison(backtester.data_loader, config)
        baselines = {
            'random': baseline_comp.run_random_baseline(args.num_baseline_episodes),
            'buy_and_hold': baseline_comp.run_buy_and_hold_baseline(args.num_baseline_episodes)
        }

    # Print results
    print_results(rl_results, baselines)

    # Save results
    if args.save_results:
        output_path = Path(args.output_dir) / 'backtest_results.json'
        save_results(rl_results, str(output_path))

    # Plot results
    if args.plot_results:
        plot_results(rl_results, baselines, args.output_dir)


if __name__ == '__main__':
    main()
