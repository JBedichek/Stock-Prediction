#!/usr/bin/env python3
"""
Statistical Comparison: Model vs Random Selection

Rigorous statistical analysis comparing model's stock selection against random portfolios.

Design:
- 30 trials with different random stock subsets
- Each trial: model vs 100 random portfolios (bootstrap)
- Metrics: returns, Sharpe, Sortino, Calmar, max drawdown, win rate
- Statistical tests: paired t-test, permutation test, percentile ranking
- Visualization: distributions, equity curves, metric comparisons

Usage:
    python inference/statistical_comparison.py \
        --data all_complete_dataset.h5 \
        --prices actual_prices.h5 \
        --model checkpoints/best_model.pt \
        --bin-edges adaptive_bin_edges.pt
"""

import torch
import numpy as np
import argparse
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.backtest_simulation import DatasetLoader, ModelPredictor, EnsemblePredictor, TradingSimulator


class BatchedRandomSimulator:
    """
    Runs multiple random portfolios in a single pass through the dates.
    Much faster than running N separate simulations.
    """

    def __init__(
        self,
        data_loader: DatasetLoader,
        num_portfolios: int,
        top_k: int,
        horizon_idx: int,
        initial_capital: float,
        base_seed: int
    ):
        self.data_loader = data_loader
        self.num_portfolios = num_portfolios
        self.top_k = top_k
        self.horizon_idx = horizon_idx
        self.initial_capital = initial_capital
        self.base_seed = base_seed

        # Map horizon_idx to days
        horizon_map = {0: 1, 1: 5, 2: 10, 3: 20}
        self.horizon_days = horizon_map.get(horizon_idx, 1)

        # Initialize capital and history for each portfolio
        self.capitals = [initial_capital] * num_portfolios
        self.capital_histories = [[] for _ in range(num_portfolios)]
        self.trade_histories = [[] for _ in range(num_portfolios)]
        self.daily_returns = [[] for _ in range(num_portfolios)]

    def run_all_simulations(self, trading_dates: List[str]) -> List[Dict]:
        """
        Run all random portfolios through the same trading period.
        Returns list of results (one per portfolio).
        """
        # Filter tradeable dates (need future price available)
        tradeable_dates = []
        for date in trading_dates:
            if trading_dates.index(date) + self.horizon_days < len(trading_dates):
                tradeable_dates.append(date)

        # Run through each trading date once
        for date in tradeable_dates:
            # Get all available stocks on this date
            available_stocks = []
            for ticker in self.data_loader.test_tickers:
                result = self.data_loader.get_features_and_price(ticker, date)
                if result is not None:
                    _, current_price = result
                    future_price = self.data_loader.get_future_price(ticker, date, self.horizon_days)
                    if future_price is not None:
                        available_stocks.append((ticker, current_price, future_price))

            if len(available_stocks) < self.top_k:
                continue

            # For each portfolio, randomly select stocks and simulate
            for portfolio_idx in range(self.num_portfolios):
                seed = self.base_seed + portfolio_idx
                random.seed(seed + len(self.trade_histories[portfolio_idx]))  # Different seed each trade

                # Random selection
                selected = random.sample(available_stocks, self.top_k)

                # Simulate trade
                capital = self.capitals[portfolio_idx]
                capital_per_stock = capital / len(selected)
                total_return = 0.0
                stock_returns = []

                for ticker, buy_price, sell_price in selected:
                    actual_return = sell_price / buy_price
                    stock_profit = capital_per_stock * actual_return
                    total_return += stock_profit

                    stock_returns.append({
                        'ticker': ticker,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'expected_return': 1.0,  # Random has no expectation
                        'actual_return': actual_return,
                        'profit_pct': (actual_return - 1.0) * 100
                    })

                new_capital = total_return
                trade_return_pct = (new_capital / capital - 1.0) * 100

                # Update state
                self.capitals[portfolio_idx] = new_capital
                self.capital_histories[portfolio_idx].append(new_capital)
                self.daily_returns[portfolio_idx].append(trade_return_pct)

                trade_info = {
                    'date': date,
                    'num_stocks': len(selected),
                    'capital_invested': capital,
                    'capital_returned': new_capital,
                    'return_pct': trade_return_pct,
                    'stocks': stock_returns
                }
                self.trade_histories[portfolio_idx].append(trade_info)

        # Compute results for each portfolio
        results = []
        for portfolio_idx in range(self.num_portfolios):
            results.append(self._compute_results(portfolio_idx))

        return results

    def _compute_results(self, portfolio_idx: int) -> Dict:
        """Compute final results for one portfolio."""
        final_capital = self.capitals[portfolio_idx]
        total_return = (final_capital / self.initial_capital - 1.0) * 100

        daily_returns = self.daily_returns[portfolio_idx]
        if len(daily_returns) > 0:
            win_rate = sum(1 for r in daily_returns if r > 0) / len(daily_returns) * 100
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)

            # Sharpe ratio (annualized)
            if std_return > 0:
                sharpe_ratio = (avg_return * np.sqrt(252)) / (std_return * np.sqrt(252))
            else:
                sharpe_ratio = 0.0

            # Max drawdown
            capital_history = [self.initial_capital] + self.capital_histories[portfolio_idx]
            peak = capital_history[0]
            max_drawdown = 0.0
            for capital in capital_history:
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            win_rate = 0.0
            avg_return = 0.0
            std_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'num_trades': len(daily_returns),
            'win_rate': win_rate,
            'avg_return_pct': avg_return,
            'std_return_pct': std_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'capital_history': self.capital_histories[portfolio_idx],
            'trade_history': self.trade_histories[portfolio_idx],
            'daily_returns': daily_returns
        }


class StatisticalComparison:
    """Run statistical comparison between model and random selection."""

    def __init__(
        self,
        data_loader: DatasetLoader,
        predictor: ModelPredictor,
        num_trials: int = 30,
        num_random_per_trial: int = 100,
        subset_size: int = 200,
        top_k: int = 5,
        horizon_idx: int = 1,
        test_months: int = 2,
        initial_capital: float = 100000.0,
        confidence_percentile: float = 0.8,
        seed: int = 42
    ):
        """
        Args:
            data_loader: DatasetLoader instance
            predictor: ModelPredictor instance
            num_trials: Number of independent trials
            num_random_per_trial: Number of random portfolios per trial
            subset_size: Size of random stock subset per trial
            top_k: Number of stocks to select
            horizon_idx: Prediction horizon index
            test_months: Test period length
            initial_capital: Starting capital
            confidence_percentile: Confidence percentile for filtering (0.8 = keep top 20%)
            seed: Random seed for reproducibility
        """
        self.data_loader = data_loader
        self.predictor = predictor
        self.num_trials = num_trials
        self.num_random_per_trial = num_random_per_trial
        self.subset_size = subset_size
        self.top_k = top_k
        self.horizon_idx = horizon_idx
        self.test_months = test_months
        self.initial_capital = initial_capital
        self.confidence_percentile = confidence_percentile
        self.seed = seed

        # Results storage
        self.trial_results = []  # List of dicts, one per trial

    def run_all_trials(self):
        """Run all trials and collect results."""
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISON: MODEL VS RANDOM")
        print(f"{'='*80}")
        print(f"Design:")
        print(f"  Trials: {self.num_trials}")
        print(f"  Random portfolios per trial: {self.num_random_per_trial}")
        print(f"  Subset size per trial: {self.subset_size}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Horizon: {self.horizon_idx}")
        print(f"  Test period: {self.test_months} months")
        print(f"  Initial capital: ${self.initial_capital:,.0f}")

        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Get trading dates
        trading_dates = self.data_loader.get_trading_period(self.test_months)

        # Preload features for efficiency
        print(f"\n{'='*80}")
        print("PRELOADING FEATURES")
        print(f"{'='*80}")
        self.data_loader.preload_features(trading_dates)

        print(f"\n{'='*80}")
        print("RUNNING TRIALS")
        print(f"{'='*80}")

        for trial_idx in tqdm(range(self.num_trials), desc="Trials"):
            trial_result = self._run_single_trial(trial_idx, trading_dates)
            self.trial_results.append(trial_result)

        print(f"\n✅ Completed {self.num_trials} trials")

    def _run_single_trial(self, trial_idx: int, trading_dates: List[str]) -> Dict:
        """Run a single trial: model vs random portfolios with dynamic daily subsampling."""
        # Set subset_size for daily dynamic subsampling (different stocks each day)
        # This enables get_daily_subset() to sample randomly each trading day
        original_subset_size = self.data_loader.subset_size
        self.data_loader.subset_size = self.subset_size

        # Create trial seed for random baseline portfolios
        trial_seed = self.seed + trial_idx

        # Map horizon_idx to horizon_days
        horizon_map = {0: 1, 1: 5, 2: 10, 3: 20}
        horizon_days = horizon_map.get(self.horizon_idx, 1)

        try:
            # Run model
            model_simulator = TradingSimulator(
                self.data_loader,
                self.predictor,
                top_k=self.top_k,
                horizon_days=horizon_days,
                horizon_idx=self.horizon_idx,
                initial_capital=self.initial_capital,
                confidence_percentile=self.confidence_percentile,
                verbose=True
            )
            model_results = model_simulator.run_simulation(trading_dates)

            # Run random portfolios using batched simulator (much faster!)
            batched_simulator = BatchedRandomSimulator(
                self.data_loader,
                num_portfolios=self.num_random_per_trial,
                top_k=self.top_k,
                horizon_idx=self.horizon_idx,
                initial_capital=self.initial_capital,
                base_seed=trial_seed + 1000
            )
            random_results = batched_simulator.run_all_simulations(trading_dates)

            # Compute summary statistics
            trial_summary = {
                'trial_idx': trial_idx,
                'subset_size': self.subset_size,  # Size of daily subset (changes each day)
                'model': self._extract_metrics(model_results),
                'random': [self._extract_metrics(r) for r in random_results],
                'random_mean': self._compute_random_mean(random_results),
                'random_std': self._compute_random_std(random_results),
            }

            return trial_summary

        finally:
            # Restore original subset_size
            self.data_loader.subset_size = original_subset_size

    def _extract_metrics(self, results: Dict) -> Dict:
        """Extract key metrics from simulation results."""
        return {
            'total_return_pct': results['total_return_pct'],
            'sharpe_ratio': results['sharpe_ratio'],
            'sortino_ratio': self._compute_sortino(results['daily_returns']),
            'calmar_ratio': self._compute_calmar(results['total_return_pct'], results['max_drawdown_pct']),
            'max_drawdown_pct': results['max_drawdown_pct'],
            'win_rate': results['win_rate'],
            'avg_return_pct': results['avg_return_pct'],
            'std_return_pct': results['std_return_pct'],
            'final_capital': results['final_capital'],
        }

    def _compute_sortino(self, daily_returns: List[float]) -> float:
        """Compute Sortino ratio (return / downside deviation)."""
        if not daily_returns:
            return 0.0

        returns = np.array(daily_returns)
        mean_return = np.mean(returns)

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        # Annualize (assuming ~252 trading days)
        sortino = (mean_return * np.sqrt(252)) / (downside_std * np.sqrt(252))
        return sortino

    def _compute_calmar(self, total_return: float, max_drawdown: float) -> float:
        """Compute Calmar ratio (return / max drawdown)."""
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        return total_return / max_drawdown

    def _compute_random_mean(self, random_results: List[Dict]) -> Dict:
        """Compute mean metrics across random portfolios."""
        metrics = {}
        for key in ['total_return_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                    'max_drawdown_pct', 'win_rate', 'avg_return_pct', 'std_return_pct']:
            values = [self._extract_metrics(r)[key] for r in random_results]
            metrics[key] = float(np.mean(values))
        return metrics

    def _compute_random_std(self, random_results: List[Dict]) -> Dict:
        """Compute std dev of metrics across random portfolios."""
        metrics = {}
        for key in ['total_return_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                    'max_drawdown_pct', 'win_rate', 'avg_return_pct', 'std_return_pct']:
            values = [self._extract_metrics(r)[key] for r in random_results]
            metrics[key] = float(np.std(values))
        return metrics

    def analyze_results(self):
        """Perform statistical analysis on results."""
        print(f"\n{'='*80}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*80}")

        # Extract model vs random means for each trial
        metrics_to_analyze = ['total_return_pct', 'sharpe_ratio', 'sortino_ratio',
                             'max_drawdown_pct', 'win_rate']

        for metric in metrics_to_analyze:
            print(f"\n{'─'*80}")
            print(f"Metric: {metric.upper()}")
            print(f"{'─'*80}")

            # Get values
            model_values = [trial['model'][metric] for trial in self.trial_results]
            random_means = [trial['random_mean'][metric] for trial in self.trial_results]

            # Summary statistics
            model_mean = np.mean(model_values)
            model_std = np.std(model_values)
            random_mean = np.mean(random_means)
            random_std = np.std(random_means)

            print(f"\nSummary Statistics:")
            print(f"  Model:  Mean = {model_mean:>8.3f},  Std = {model_std:>8.3f}")
            print(f"  Random: Mean = {random_mean:>8.3f},  Std = {random_std:>8.3f}")
            print(f"  Difference: {model_mean - random_mean:>+8.3f}")

            # Paired t-test
            differences = [m - r for m, r in zip(model_values, random_means)]
            t_stat, p_value = stats.ttest_1samp(differences, 0)

            print(f"\nPaired t-test:")
            print(f"  t-statistic: {t_stat:>8.3f}")
            print(f"  p-value: {p_value:>8.6f}")
            print(f"  Significant: {'YES ✅' if p_value < 0.05 else 'NO ❌'}")

            # Effect size (Cohen's d)
            effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
            print(f"  Effect size (Cohen's d): {effect_size:>8.3f}")

            # Confidence interval
            ci_95 = stats.t.interval(0.95, len(differences)-1,
                                     loc=np.mean(differences),
                                     scale=stats.sem(differences))
            print(f"  95% CI: [{ci_95[0]:>+8.3f}, {ci_95[1]:>+8.3f}]")

            # Win rate (model beats random in X% of trials)
            wins = sum(1 for m, r in zip(model_values, random_means) if m > r)
            win_rate = wins / len(model_values) * 100
            print(f"  Model wins: {wins}/{len(model_values)} trials ({win_rate:.1f}%)")

            # Bootstrap percentile (across all random portfolios)
            all_random_values = []
            for trial in self.trial_results:
                for random_result in trial['random']:
                    all_random_values.append(random_result[metric])

            model_percentile = stats.percentileofscore(all_random_values, model_mean)
            print(f"  Model percentile vs all random: {model_percentile:.1f}th")

        # Overall conclusion
        print(f"\n{'='*80}")
        print("OVERALL CONCLUSION")
        print(f"{'='*80}")

        # Count significant metrics
        significant_count = 0
        for metric in metrics_to_analyze:
            model_values = [trial['model'][metric] for trial in self.trial_results]
            random_means = [trial['random_mean'][metric] for trial in self.trial_results]
            differences = [m - r for m, r in zip(model_values, random_means)]
            _, p_value = stats.ttest_1samp(differences, 0)
            if p_value < 0.05:
                significant_count += 1

        print(f"Significant metrics: {significant_count}/{len(metrics_to_analyze)}")

        if significant_count >= len(metrics_to_analyze) // 2:
            print("✅ Model shows statistically significant improvement over random selection")
        else:
            print("❌ Model does not consistently outperform random selection")

    def save_results(self, output_path: str = "statistical_comparison_results.pt"):
        """Save results for later analysis."""
        results = {
            'config': {
                'num_trials': self.num_trials,
                'num_random_per_trial': self.num_random_per_trial,
                'subset_size': self.subset_size,
                'top_k': self.top_k,
                'horizon_idx': self.horizon_idx,
                'test_months': self.test_months,
                'initial_capital': self.initial_capital,
                'seed': self.seed,
            },
            'trial_results': self.trial_results,
        }
        torch.save(results, output_path)
        print(f"\n💾 Results saved to: {output_path}")

    def plot_results(self, output_dir: str = "."):
        """Generate visualizations."""
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")

        os.makedirs(output_dir, exist_ok=True)

        # Define metrics with clear descriptions
        metrics_info = [
            {
                'key': 'total_return_pct',
                'label': 'Total Return',
                'unit': '%',
                'better': 'higher',
                'description': 'Profit/Loss over test period'
            },
            {
                'key': 'sharpe_ratio',
                'label': 'Sharpe Ratio',
                'unit': '',
                'better': 'higher',
                'description': 'Risk-adjusted returns (higher = better)'
            },
            {
                'key': 'win_rate',
                'label': 'Win Rate',
                'unit': '%',
                'better': 'higher',
                'description': '% of profitable trades'
            },
            {
                'key': 'max_drawdown_pct',
                'label': 'Max Drawdown',
                'unit': '%',
                'better': 'lower',
                'description': 'Largest peak-to-trough loss'
            }
        ]

        # Plot 1: Side-by-side bar chart with error bars
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, metric_info in enumerate(metrics_info):
            ax = axes[idx]
            metric = metric_info['key']

            # Get data
            model_values = [trial['model'][metric] for trial in self.trial_results]
            random_means = [trial['random_mean'][metric] for trial in self.trial_results]

            # Compute statistics
            model_mean = np.mean(model_values)
            model_std = np.std(model_values)
            random_mean = np.mean(random_means)
            random_std = np.std(random_means)

            # Statistical test
            differences = [m - r for m, r in zip(model_values, random_means)]
            _, p_value = stats.ttest_1samp(differences, 0)
            is_significant = p_value < 0.05

            # Determine colors based on which is better
            if metric_info['better'] == 'higher':
                model_better = model_mean > random_mean
            else:
                model_better = model_mean < random_mean

            model_color = '#2ecc71' if model_better else '#e74c3c'  # Green if better, red if worse
            random_color = '#95a5a6'  # Gray for random

            # Bar chart with error bars
            x_pos = [0, 1]
            means = [random_mean, model_mean]
            stds = [random_std, model_std]
            colors = [random_color, model_color]
            labels = ['Random\nSelection', 'Model\nPrediction']

            bars = ax.bar(x_pos, means, yerr=stds, capsize=10,
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{metric_info['label']} {metric_info['unit']}", fontsize=12, fontweight='bold')

            # Title with description
            title = f"{metric_info['label']}\n{metric_info['description']}"
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                label_y = height + std + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       f'{mean:.2f}{metric_info["unit"]}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

            # Add difference annotation
            diff = model_mean - random_mean
            if metric_info['better'] == 'higher':
                diff_text = f"Model {'better' if diff > 0 else 'worse'} by {abs(diff):.2f}{metric_info['unit']}"
            else:
                diff_text = f"Model {'better' if diff < 0 else 'worse'} by {abs(diff):.2f}{metric_info['unit']}"

            # Add significance star
            if is_significant:
                diff_text += " ✓ SIGNIFICANT"
                text_color = '#2ecc71' if model_better else '#e74c3c'
            else:
                diff_text += " (not significant)"
                text_color = '#95a5a6'

            ax.text(0.5, 0.95, diff_text,
                   transform=ax.transAxes,
                   ha='center', va='top',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor=text_color, linewidth=2),
                   color=text_color)

            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'comparison_summary.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {plot_path}")
        plt.close()

        # Plot 2: Distribution overlays with clear interpretation
        fig = plt.figure(figsize=(18, 10))

        for idx, metric_info in enumerate(metrics_info):
            ax = plt.subplot(2, 2, idx + 1)
            metric = metric_info['key']

            # Get data
            model_values = [trial['model'][metric] for trial in self.trial_results]
            random_means = [trial['random_mean'][metric] for trial in self.trial_results]

            # Statistical test
            differences = [m - r for m, r in zip(model_values, random_means)]
            _, p_value = stats.ttest_1samp(differences, 0)

            # Determine which is better
            model_mean = np.mean(model_values)
            random_mean = np.mean(random_means)
            if metric_info['better'] == 'higher':
                model_better = model_mean > random_mean
            else:
                model_better = model_mean < random_mean

            model_color = '#2ecc71' if model_better else '#e74c3c'

            # Plot histograms with better styling
            ax.hist(random_means, bins=12, alpha=0.5, label='Random',
                   color='#95a5a6', edgecolor='black', linewidth=1.2)
            ax.hist(model_values, bins=12, alpha=0.7, label='Model',
                   color=model_color, edgecolor='black', linewidth=1.2)

            # Add mean lines
            ax.axvline(random_mean, color='#7f8c8d', linestyle='--', linewidth=3,
                      label=f'Random Avg: {random_mean:.2f}')
            ax.axvline(model_mean, color=model_color, linestyle='--', linewidth=3,
                      label=f'Model Avg: {model_mean:.2f}')

            ax.set_xlabel(f"{metric_info['label']} {metric_info['unit']}",
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Trials', fontsize=12, fontweight='bold')
            ax.set_title(f"{metric_info['label']} Distribution\n({metric_info['better']} is better)",
                        fontsize=13, fontweight='bold', pad=15)

            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            # Add interpretation text
            if p_value < 0.05:
                if model_better:
                    interp = "✓ Model significantly outperforms random"
                else:
                    interp = "✗ Model significantly underperforms random"
                text_color = model_color
            else:
                interp = "○ No significant difference"
                text_color = '#95a5a6'

            ax.text(0.98, 0.97, interp,
                   transform=ax.transAxes,
                   ha='right', va='top',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor=text_color, linewidth=2, alpha=0.9),
                   color=text_color)

        plt.suptitle('Model vs Random: Distribution Comparison Across All Trials',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'distributions_detailed.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {plot_path}")
        plt.close()

        # Plot 3: Overall summary scorecard
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # Count wins
        wins = 0
        total_metrics = len(metrics_info)

        summary_text = "STATISTICAL COMPARISON SUMMARY\n"
        summary_text += "="*50 + "\n\n"

        for metric_info in metrics_info:
            metric = metric_info['key']
            model_values = [trial['model'][metric] for trial in self.trial_results]
            random_means = [trial['random_mean'][metric] for trial in self.trial_results]

            model_mean = np.mean(model_values)
            random_mean = np.mean(random_means)
            diff = model_mean - random_mean

            differences = [m - r for m, r in zip(model_values, random_means)]
            _, p_value = stats.ttest_1samp(differences, 0)

            if metric_info['better'] == 'higher':
                model_better = model_mean > random_mean
            else:
                model_better = model_mean < random_mean

            if model_better and p_value < 0.05:
                wins += 1
                status = "✓ WIN"
                color = 'green'
            elif not model_better and p_value < 0.05:
                status = "✗ LOSS"
                color = 'red'
            else:
                status = "○ TIE"
                color = 'gray'

            summary_text += f"{metric_info['label']}:\n"
            summary_text += f"  Model:  {model_mean:>8.2f}{metric_info['unit']}\n"
            summary_text += f"  Random: {random_mean:>8.2f}{metric_info['unit']}\n"
            summary_text += f"  Diff:   {diff:>+8.2f}{metric_info['unit']}\n"
            summary_text += f"  p-value: {p_value:.4f}\n"
            summary_text += f"  Result: {status}\n\n"

        summary_text += "="*50 + "\n"
        summary_text += f"OVERALL SCORE: {wins}/{total_metrics} metrics significantly better\n\n"

        if wins >= total_metrics * 0.75:
            conclusion = "✓ STRONG EVIDENCE: Model significantly outperforms random selection"
            conclusion_color = 'green'
        elif wins >= total_metrics * 0.5:
            conclusion = "○ MODERATE EVIDENCE: Model shows some improvement over random"
            conclusion_color = 'orange'
        else:
            conclusion = "✗ WEAK EVIDENCE: Model does not consistently outperform random"
            conclusion_color = 'red'

        summary_text += conclusion

        ax.text(0.5, 0.5, summary_text,
               transform=ax.transAxes,
               ha='center', va='center',
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor=conclusion_color, linewidth=3, alpha=0.9))

        plot_path = os.path.join(output_dir, 'summary_scorecard.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"  📊 Saved: {plot_path}")
        plt.close()

        print(f"✅ Visualizations complete")


def main():
    parser = argparse.ArgumentParser(description='Statistical comparison: model vs random')

    # Data args
    parser.add_argument('--data', type=str, default="data/all_complete_dataset.h5",
                       help='Path to dataset')
    parser.add_argument('--prices', type=str, default="data/actual_prices.h5",
                       help='Path to prices HDF5')
    parser.add_argument('--model', type=str, default="checkpoints/best_model.pt",
                       help='Path to model checkpoint (or first model if using --ensemble-models)')
    parser.add_argument('--ensemble-models', type=str, nargs='+', default=None,
                       help='Paths to multiple model checkpoints for ensemble prediction (overrides --model)')
    parser.add_argument('--bin-edges', type=str, default='data/adaptive_bin_edges.pt',
                       help='Path to bin edges')

    # Test set args
    parser.add_argument('--num-test-stocks', type=int, default=2000,
                       help='Number of test stocks (last N alphabetically)')

    # Comparison args
    parser.add_argument('--num-trials', type=int, default=30,
                       help='Number of independent trials')
    parser.add_argument('--num-random-per-trial', type=int, default=100,
                       help='Number of random portfolios per trial')
    parser.add_argument('--subset-size', type=int, default=200,
                       help='Size of random stock subset per trial')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of stocks to select')
    parser.add_argument('--horizon-idx', type=int, default=0,
                       help='Prediction horizon (0=1d, 1=5d, 2=10d, 3=20d)')
    parser.add_argument('--confidence-percentile', type=float, default=0.8,
                       help='Confidence percentile for filtering (default: 0.8 = keep top 20%%)')
    parser.add_argument('--test-months', type=int, default=2,
                       help='Test period length in months')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Starting capital')

    # Other args
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--output', type=str, default='statistical_comparison_results.pt',
                       help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load dataset
    data_loader = DatasetLoader(
        args.data,
        num_test_stocks=args.num_test_stocks,
        prices_path=args.prices
    )

    # Load model (single or ensemble)
    if args.ensemble_models is not None:
        # Ensemble mode: use multiple models
        print(f"\n🎯 Ensemble mode: using {len(args.ensemble_models)} models")
        predictor = EnsemblePredictor(
            model_paths=args.ensemble_models,
            bin_edges_path=args.bin_edges,
            device=args.device,
            batch_size=args.batch_size
        )
    else:
        # Single model mode
        predictor = ModelPredictor(
            args.model,
            args.bin_edges,
            device=args.device,
            batch_size=args.batch_size
        )

    # Run comparison
    comparison = StatisticalComparison(
        data_loader=data_loader,
        predictor=predictor,
        num_trials=args.num_trials,
        num_random_per_trial=args.num_random_per_trial,
        subset_size=args.subset_size,
        top_k=args.top_k,
        horizon_idx=args.horizon_idx,
        test_months=args.test_months,
        initial_capital=args.initial_capital,
        confidence_percentile=args.confidence_percentile,
        seed=args.seed
    )

    # Run trials
    comparison.run_all_trials()

    # Analyze
    comparison.analyze_results()

    # Save results
    comparison.save_results(args.output)

    # Generate plots
    comparison.plot_results(output_dir=os.path.dirname(args.output) or '.')

    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
