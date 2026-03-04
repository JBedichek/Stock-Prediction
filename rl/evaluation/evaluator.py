"""
Comprehensive evaluation framework for DQN trading agent.

Runs out-of-sample testing with multiple episodes, computes metrics,
compares against baselines, and generates statistical significance tests.
"""

import os
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from rl.evaluation.metrics import compute_all_metrics, aggregate_metrics, confidence_interval
from rl.evaluation.baselines import get_baseline_strategy, evaluate_baseline
from rl.reduced_action_space import create_global_state
from rl.state_creation_optimized import create_next_states_batch_optimized


class DQNEvaluator:
    """
    Comprehensive evaluator for trained DQN agent.

    Runs multiple evaluation episodes with:
    - Random stock selections
    - Random episode start dates
    - Multiple baselines for comparison
    - Statistical significance testing
    """

    def __init__(
        self,
        agent,
        vec_env,
        stock_selections_cache,
        test_start_date: str,
        test_end_date: str,
        num_episodes: int = 100,
        episode_length: int = 30,
        initial_capital: float = 100000,
        device: str = 'cuda',
        all_tickers: List[str] = None
    ):
        """
        Initialize evaluator.

        Args:
            agent: Trained DQN agent (SimpleDQNTrainer)
            vec_env: Vectorized trading environment
            stock_selections_cache: GPU stock selection cache
            test_start_date: Start date for test period (e.g., '2024-01-01')
            test_end_date: End date for test period (e.g., '2024-12-31')
            num_episodes: Number of evaluation episodes
            episode_length: Length of each episode in trading days
            initial_capital: Initial capital per episode
            device: Device to run on
        """
        self.agent = agent
        self.vec_env = vec_env
        self.stock_cache = stock_selections_cache
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.initial_capital = initial_capital
        self.device = device
        self.all_tickers = all_tickers or []

        # Get allow_short from agent
        self.allow_short = getattr(agent, 'allow_short', True)

        # Get available trading days in test period
        self.test_trading_days = self._get_test_trading_days()

        print(f"\n{'='*80}")
        print(f"DQN EVALUATOR INITIALIZED")
        print(f"{'='*80}")
        print(f"Test period: {test_start_date} to {test_end_date}")
        print(f"Trading days in test period: {len(self.test_trading_days)}")
        print(f"Episodes to run: {num_episodes}")
        print(f"Episode length: {episode_length} days")
        print(f"Initial capital: ${initial_capital:,.0f}")
        print(f"{'='*80}\n")

    def _get_test_trading_days(self) -> List[str]:
        """Get list of trading days in test period."""
        # Get all trading days from stock cache
        all_dates = sorted(self.stock_cache.selections.keys())

        # Filter to test period
        test_dates = [
            d for d in all_dates
            if self.test_start_date <= d <= self.test_end_date
        ]

        return test_dates

    def run_dqn_episode(self, start_date: str) -> Tuple[List[float], List[Dict]]:
        """
        Run one DQN evaluation episode.

        Args:
            start_date: Episode start date

        Returns:
            Tuple of (portfolio_values, trades)
        """
        # Get episode dates
        start_idx = self.test_trading_days.index(start_date)
        end_idx = min(start_idx + self.episode_length, len(self.test_trading_days))
        episode_dates = self.test_trading_days[start_idx:end_idx]

        if len(episode_dates) < 2:
            return [self.initial_capital], []

        # Manually reset environment for this specific episode
        # (Simpler than trying to use vec_env which randomizes dates)
        portfolio_values = [self.initial_capital]
        trades = []
        current_position = None
        is_short = False

        # Get initial stock selections
        current_date = episode_dates[0]
        num_samples = self.stock_cache.get_num_samples(current_date)
        sample_idx = random.randint(0, num_samples - 1)
        top_4_stocks, bottom_4_stocks = self.stock_cache.get_sample(current_date, sample_idx)

        # Get initial states from ref_env
        self.vec_env.ref_env.current_date = current_date
        self.vec_env.ref_env.step_idx = 0
        self.vec_env.ref_env.portfolio = {}
        self.vec_env.ref_env.cash = self.initial_capital

        states_dict = self.vec_env.ref_env._get_states()

        # Create global state
        if len(states_dict) > 0:
            global_state = create_global_state(
                top_4_stocks=top_4_stocks,
                bottom_4_stocks=bottom_4_stocks,
                states=states_dict,
                current_position=current_position,
                is_short=is_short,
                device=self.device,
                allow_short=self.allow_short
            )
        else:
            global_state = None

        # Simulate episode
        for step_idx, date in enumerate(episode_dates[:-1]):  # -1 because we need next date
            if global_state is None:
                break

            # Agent selects action (greedy, no exploration)
            with torch.no_grad():
                action = self.agent.select_action(global_state, epsilon=0.0)

            # Convert action to trade
            trade = self.agent.action_to_trade(action, top_4_stocks, bottom_4_stocks)

            # Execute trade manually using ref_env
            if trade['action'] != 'HOLD':
                ticker = trade['ticker']
                cached_prices = self.vec_env.ref_env.price_cache.get(date, {})
                current_price = cached_prices.get(ticker)

                if current_price is not None:
                    # Execute single action on ref_env
                    if trade['action'] == 'BUY':
                        action_id = 1
                    elif trade['action'] == 'SHORT':
                        action_id = 3
                    else:
                        action_id = 0

                    trade_result = self.vec_env.ref_env._execute_single_action(
                        ticker, action_id, current_price
                    )

                    if trade_result is not None:
                        trades.append(trade_result)
                        current_position = ticker
                        is_short = (trade['position_type'] == 'SHORT')

            # Advance to next day
            next_date = episode_dates[step_idx + 1]
            self.vec_env.ref_env.current_date = next_date
            self.vec_env.ref_env.step_idx = step_idx + 1

            # Update portfolio value
            next_cached_prices = self.vec_env.ref_env.price_cache.get(next_date, {})
            portfolio_value = self.vec_env.ref_env._portfolio_value_cached(next_cached_prices)
            portfolio_values.append(portfolio_value)

            # Get next states
            next_states_dict = self.vec_env.ref_env._get_states()
            num_samples = self.stock_cache.get_num_samples(next_date)
            sample_idx = random.randint(0, num_samples - 1)
            top_4_stocks, bottom_4_stocks = self.stock_cache.get_sample(next_date, sample_idx)

            if len(next_states_dict) > 0:
                global_state = create_global_state(
                    top_4_stocks=top_4_stocks,
                    bottom_4_stocks=bottom_4_stocks,
                    states=next_states_dict,
                    current_position=current_position,
                    is_short=is_short,
                    device=self.device,
                    allow_short=self.allow_short
                )
            else:
                global_state = None

        # Close all positions at end
        final_date = episode_dates[-1]
        final_cached_prices = self.vec_env.ref_env.price_cache.get(final_date, {})
        self.vec_env.ref_env._close_all_positions(final_cached_prices)

        # Final portfolio value
        final_pv = self.vec_env.ref_env._portfolio_value_cached(final_cached_prices)
        if len(portfolio_values) == len(episode_dates):
            portfolio_values[-1] = final_pv
        else:
            portfolio_values.append(final_pv)

        return portfolio_values, trades

    def evaluate_dqn(self) -> Dict:
        """
        Evaluate DQN agent over multiple episodes.

        Returns:
            Dictionary with results for all episodes
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING DQN AGENT")
        print(f"{'='*80}\n")

        results = []

        # Generate random start dates for episodes
        max_start_idx = len(self.test_trading_days) - self.episode_length - 1
        if max_start_idx < 0:
            print("ERROR: Not enough trading days in test period!")
            return {'episodes': [], 'aggregated': {}}

        start_indices = [random.randint(0, max_start_idx) for _ in range(self.num_episodes)]

        for ep_idx in tqdm(range(self.num_episodes), desc="Running DQN episodes"):
            start_date = self.test_trading_days[start_indices[ep_idx]]

            # Run episode
            portfolio_values, trades = self.run_dqn_episode(start_date)

            # Compute metrics
            metrics = compute_all_metrics(portfolio_values, trades)
            metrics['start_date'] = start_date
            metrics['episode_idx'] = ep_idx

            results.append(metrics)

        # Aggregate results
        aggregated = aggregate_metrics(results)

        return {
            'episodes': results,
            'aggregated': aggregated
        }

    def evaluate_baselines(self, baseline_names: List[str] = ['random', 'hold', 'long']) -> Dict:
        """
        Evaluate baseline strategies.

        Args:
            baseline_names: List of baseline names to evaluate

        Returns:
            Dictionary with results for each baseline
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING BASELINE STRATEGIES")
        print(f"{'='*80}\n")

        baseline_results = {}

        for baseline_name in baseline_names:
            print(f"Evaluating {baseline_name} baseline...")

            strategy = get_baseline_strategy(
                baseline_name,
                self.initial_capital,
                all_tickers=self.all_tickers
            )

            # For simplicity, we'll run baselines using vectorized env
            # This is a simplified evaluation - ideally would use same episode dates as DQN
            results = evaluate_baseline(
                strategy,
                self.vec_env,
                self.stock_cache,
                num_episodes=self.num_episodes,
                episode_length=self.episode_length
            )

            aggregated = aggregate_metrics(results)

            baseline_results[baseline_name] = {
                'episodes': results,
                'aggregated': aggregated
            }

        return baseline_results

    def compare_to_baselines(self, dqn_results: Dict, baseline_results: Dict) -> Dict:
        """
        Statistical comparison of DQN vs baselines.

        Args:
            dqn_results: DQN evaluation results
            baseline_results: Baseline evaluation results

        Returns:
            Dictionary with comparison statistics
        """
        from scipy import stats

        print(f"\n{'='*80}")
        print(f"STATISTICAL COMPARISON")
        print(f"{'='*80}\n")

        comparisons = {}

        # Get DQN returns
        dqn_returns = [ep['total_return'] for ep in dqn_results['episodes']]

        for baseline_name, baseline_data in baseline_results.items():
            baseline_returns = [ep['total_return'] for ep in baseline_data['episodes']]

            # T-test
            t_stat, p_value = stats.ttest_ind(dqn_returns, baseline_returns)

            # Effect size (Cohen's d)
            mean_diff = np.mean(dqn_returns) - np.mean(baseline_returns)
            pooled_std = np.sqrt(
                (np.var(dqn_returns, ddof=1) + np.var(baseline_returns, ddof=1)) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            comparisons[baseline_name] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'dqn_mean': float(np.mean(dqn_returns)),
                'baseline_mean': float(np.mean(baseline_returns)),
                'significant': p_value < 0.05
            }

            print(f"{baseline_name.upper()}:")
            print(f"  DQN mean return: {comparisons[baseline_name]['dqn_mean']:.4f}")
            print(f"  Baseline mean return: {comparisons[baseline_name]['baseline_mean']:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {comparisons[baseline_name]['significant']}")
            print(f"  Cohen's d: {cohens_d:.4f}\n")

        return comparisons

    def generate_histograms(self, dqn_results: Dict, baseline_results: Dict,
                           output_dir: str = 'rl/evaluation/results', timestamp: str = None):
        """
        Generate histogram plots for returns of DQN and baselines.

        Args:
            dqn_results: DQN evaluation results
            baseline_results: Baseline results
            output_dir: Directory to save plots
            timestamp: Timestamp for filename
        """
        os.makedirs(output_dir, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract returns
        dqn_returns = np.array([ep['total_return'] for ep in dqn_results['episodes']]) * 100  # Convert to percentage

        baseline_returns = {}
        for baseline_name, baseline_data in baseline_results.items():
            baseline_returns[baseline_name] = np.array(
                [ep['total_return'] for ep in baseline_data['episodes']]
            ) * 100

        # Create figure with subplots
        num_baselines = len(baseline_results)
        fig, axes = plt.subplots(1, num_baselines + 1, figsize=(5 * (num_baselines + 1), 4))

        if num_baselines == 0:
            axes = [axes]

        # Plot DQN histogram
        axes[0].hist(dqn_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(dqn_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {dqn_returns.mean():.2f}%')
        axes[0].set_xlabel('Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('DQN Agent Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot baseline histograms
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for idx, (baseline_name, returns) in enumerate(baseline_returns.items()):
            ax = axes[idx + 1]
            color = colors[idx % len(colors)]
            ax.hist(returns, bins=30, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{baseline_name.title()} Strategy Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        histogram_path = os.path.join(output_dir, f'returns_histogram_{timestamp}.png')
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Histogram saved: {histogram_path}")

        # Create combined comparison histogram
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all strategies on same axis
        ax.hist(dqn_returns, bins=30, alpha=0.5, color='blue', label='DQN', edgecolor='black')

        for idx, (baseline_name, returns) in enumerate(baseline_returns.items()):
            color = colors[idx % len(colors)]
            ax.hist(returns, bins=30, alpha=0.5, color=color, label=baseline_name.title(), edgecolor='black')

        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution: DQN vs Baselines')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save comparison figure
        comparison_path = os.path.join(output_dir, f'returns_comparison_{timestamp}.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Comparison histogram saved: {comparison_path}")

        return histogram_path, comparison_path

    def generate_report(self, dqn_results: Dict, baseline_results: Dict,
                       comparisons: Dict, output_dir: str = 'rl/evaluation/results'):
        """
        Generate comprehensive evaluation report.

        Args:
            dqn_results: DQN evaluation results
            baseline_results: Baseline results
            comparisons: Statistical comparisons
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate histograms
        print(f"\n{'='*80}")
        print(f"GENERATING HISTOGRAMS")
        print(f"{'='*80}\n")
        histogram_path, comparison_path = self.generate_histograms(
            dqn_results, baseline_results, output_dir, timestamp
        )

        # Save JSON results
        report = {
            'metadata': {
                'test_start_date': self.test_start_date,
                'test_end_date': self.test_end_date,
                'num_episodes': self.num_episodes,
                'episode_length': self.episode_length,
                'initial_capital': self.initial_capital,
                'timestamp': timestamp,
                'histogram_path': histogram_path,
                'comparison_path': comparison_path
            },
            'dqn': dqn_results,
            'baselines': baseline_results,
            'comparisons': comparisons
        }

        json_path = os.path.join(output_dir, f'evaluation_report_{timestamp}.json')

        # Custom JSON encoder to handle numpy types and complex numbers
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, complex):
                    return float(obj.real)  # Take real part, discard imaginary
                elif isinstance(obj, np.complexfloating):
                    return float(obj.real)
                return super().default(obj)

        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        print(f"\n{'='*80}")
        print(f"REPORT SAVED")
        print(f"{'='*80}")
        print(f"JSON:        {json_path}")
        print(f"Histograms:  {histogram_path}")
        print(f"Comparison:  {comparison_path}")
        print(f"{'='*80}\n")

        # Print summary to console
        self._print_summary(dqn_results, baseline_results, comparisons)

        return json_path

    def _print_summary(self, dqn_results: Dict, baseline_results: Dict, comparisons: Dict):
        """Print summary table to console."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}\n")

        # DQN results
        print("DQN AGENT:")
        dqn_agg = dqn_results['aggregated']
        print(f"  Total Return:       {dqn_agg['total_return']['mean']:.4f} ± {dqn_agg['total_return']['std']:.4f}")
        print(f"  Final Portfolio:    ${dqn_agg['final_portfolio_value']['mean']:,.2f} ± ${dqn_agg['final_portfolio_value']['std']:,.2f}")
        print(f"  Return %:           {dqn_agg['total_return']['mean']*100:+.2f}% ± {dqn_agg['total_return']['std']*100:.2f}%")
        print(f"  Sharpe Ratio:       {dqn_agg['sharpe_ratio']['mean']:.4f} ± {dqn_agg['sharpe_ratio']['std']:.4f}")
        print(f"  Max Drawdown:       {dqn_agg['max_drawdown']['mean']:.4f} ± {dqn_agg['max_drawdown']['std']:.4f}")
        print(f"  Win Rate:           {dqn_agg['win_rate']['mean']:.4f} ± {dqn_agg['win_rate']['std']:.4f}")
        print(f"  Num Trades:         {dqn_agg['num_trades']['mean']:.1f} ± {dqn_agg['num_trades']['std']:.1f}")
        print()

        # Baseline comparisons
        print("BASELINE COMPARISONS:")
        for baseline_name, baseline_data in baseline_results.items():
            baseline_agg = baseline_data['aggregated']
            comp = comparisons.get(baseline_name, {})

            print(f"\n  {baseline_name.upper()}:")
            print(f"    Total Return:       {baseline_agg['total_return']['mean']:.4f} ± {baseline_agg['total_return']['std']:.4f}")
            print(f"    Final Portfolio:    ${baseline_agg['final_portfolio_value']['mean']:,.2f} ± ${baseline_agg['final_portfolio_value']['std']:,.2f}")
            print(f"    Return %:           {baseline_agg['total_return']['mean']*100:+.2f}% ± {baseline_agg['total_return']['std']*100:.2f}%")

            if comp:
                print(f"    vs DQN difference:  {comp['dqn_mean'] - comp['baseline_mean']:+.4f}")
                print(f"    p-value:            {comp['p_value']:.4f}")
                print(f"    Significant:        {'YES' if comp['significant'] else 'NO'}")

        print(f"\n{'='*80}\n")

    def run_full_evaluation(self, baseline_names: List[str] = ['random', 'hold', 'long']) -> str:
        """
        Run complete evaluation pipeline.

        Args:
            baseline_names: List of baselines to compare against

        Returns:
            Path to saved report
        """
        # Set agent to eval mode
        self.agent.q_network.eval()

        # Evaluate DQN
        dqn_results = self.evaluate_dqn()

        # Evaluate baselines
        baseline_results = self.evaluate_baselines(baseline_names)

        # Compare
        comparisons = self.compare_to_baselines(dqn_results, baseline_results)

        # Generate report
        report_path = self.generate_report(dqn_results, baseline_results, comparisons)

        return report_path
