#!/usr/bin/env python3
"""
Interactive Transaction Cost Analysis Tool

A simpler tool that works with saved portfolio weights and returns,
without needing to re-run model inference.

Can load from:
1. Walk-forward results (JSON/PT files)
2. Backtest results
3. Custom CSV/NPY files with weights and returns

Usage:
    # From walk-forward results
    python inference/interactive_tc_analysis.py \
        --results walk_forward_portfolio_results.json

    # From numpy arrays
    python inference/interactive_tc_analysis.py \
        --weights weights.npy \
        --returns returns.npy \
        --dates dates.npy
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from typing import Dict, List, Tuple, Optional
import os


class TransactionCostAnalyzer:
    """Analyze portfolio performance with varying transaction costs."""

    def __init__(
        self,
        dates: np.ndarray,
        returns: np.ndarray,
        weights: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ):
        """
        Initialize analyzer.

        Args:
            dates: Array of date strings
            returns: (num_days, num_stocks) array of stock returns
            weights: (num_days, num_stocks) array of portfolio weights
            benchmark_returns: Optional (num_days,) array of benchmark returns
        """
        self.dates = dates
        self.returns = returns
        self.weights = weights

        # Compute portfolio returns (before transaction costs)
        self.portfolio_returns = (weights * returns).sum(axis=1)

        # Compute daily turnover
        self.turnover = self._compute_turnover(weights)

        # Benchmark
        if benchmark_returns is not None:
            self.benchmark_returns = benchmark_returns
        else:
            # Use equal-weight as benchmark
            valid_mask = ~np.isnan(returns) & (returns != 0)
            valid_counts = valid_mask.sum(axis=1)
            self.benchmark_returns = np.nansum(returns, axis=1) / np.maximum(valid_counts, 1)

        print(f"Loaded {len(dates)} trading days")
        print(f"Average daily turnover: {self.turnover.mean():.2%}")
        print(f"Average portfolio return: {self.portfolio_returns.mean():.4%}")

    def _compute_turnover(self, weights: np.ndarray) -> np.ndarray:
        """Compute daily turnover."""
        turnover = np.zeros(len(weights))
        for i in range(1, len(weights)):
            # Normalize weights
            w_prev = weights[i-1] / (weights[i-1].sum() + 1e-8)
            w_curr = weights[i] / (weights[i].sum() + 1e-8)
            turnover[i] = np.abs(w_curr - w_prev).sum()
        return turnover

    def compute_portfolio_value(
        self,
        transaction_cost: float = 0.001,
        initial_value: float = 100.0
    ) -> np.ndarray:
        """Compute cumulative portfolio value with transaction costs."""
        values = np.zeros(len(self.dates))
        values[0] = initial_value

        for i in range(1, len(self.dates)):
            # Net return after transaction costs
            net_return = self.portfolio_returns[i] - transaction_cost * self.turnover[i]
            values[i] = values[i-1] * (1 + net_return)

        return values

    def compute_benchmark_value(self, initial_value: float = 100.0) -> np.ndarray:
        """Compute cumulative benchmark value."""
        values = np.zeros(len(self.dates))
        values[0] = initial_value

        for i in range(1, len(self.dates)):
            values[i] = values[i-1] * (1 + self.benchmark_returns[i])

        return values

    def compute_metrics(
        self,
        values: np.ndarray,
        trading_days_per_year: int = 252
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return {k: 0.0 for k in ['total_return', 'annual_return', 'volatility', 'sharpe', 'max_drawdown']}

        total_return = (values[-1] / values[0] - 1) * 100
        annual_factor = trading_days_per_year / len(values)
        annual_return = ((values[-1] / values[0]) ** annual_factor - 1) * 100
        volatility = returns.std() * np.sqrt(trading_days_per_year) * 100
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(trading_days_per_year)

        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min() * 100

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }

    def compute_breakeven_tc(self, benchmark_values: np.ndarray) -> float:
        """Find the transaction cost at which portfolio matches benchmark."""
        # Binary search for breakeven point
        low, high = 0.0, 0.05  # 0 to 5%

        benchmark_final = benchmark_values[-1]

        for _ in range(50):  # 50 iterations of binary search
            mid = (low + high) / 2
            portfolio_values = self.compute_portfolio_value(mid)

            if portfolio_values[-1] > benchmark_final:
                low = mid
            else:
                high = mid

        return mid

    def create_interactive_plot(self):
        """Create interactive matplotlib figure."""

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Transaction Cost Sensitivity Analysis', fontsize=14, fontweight='bold')

        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        ax_main = fig.add_subplot(gs[0:2, 0:3])
        ax_metrics = fig.add_subplot(gs[0:2, 3])
        ax_dd = fig.add_subplot(gs[2, 0:2])
        ax_sensitivity = fig.add_subplot(gs[2, 2:4])

        ax_metrics.axis('off')

        # Slider axes
        ax_slider1 = plt.axes([0.15, 0.02, 0.3, 0.02])
        ax_slider2 = plt.axes([0.55, 0.02, 0.3, 0.02])

        # Create sliders
        slider_tc = Slider(ax_slider1, 'Transaction Cost (bps)', 0, 100, valinit=10, valstep=1)
        slider_tc2 = Slider(ax_slider2, 'Compare TC (bps)', 0, 100, valinit=0, valstep=1)

        # Data for plotting
        date_nums = np.arange(len(self.dates))
        n_labels = min(10, len(self.dates))
        label_indices = np.linspace(0, len(self.dates)-1, n_labels, dtype=int)
        date_labels = [self.dates[i] if i < len(self.dates) else '' for i in label_indices]

        # Initial values
        tc = 0.001
        portfolio_values = self.compute_portfolio_value(tc)
        benchmark_values = self.compute_benchmark_value()

        # Main plot
        ax_main.set_title('Portfolio Value Over Time')
        line_portfolio, = ax_main.plot(date_nums, portfolio_values, 'b-',
                                        label='Portfolio', linewidth=2)
        line_benchmark, = ax_main.plot(date_nums, benchmark_values, 'gray',
                                        label='Benchmark', linewidth=1.5, linestyle='--', alpha=0.7)
        line_compare, = ax_main.plot([], [], 'r-', label='Compare', linewidth=1.5, alpha=0.8)

        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Portfolio Value ($)')
        ax_main.set_xticks(label_indices)
        ax_main.set_xticklabels(date_labels, rotation=45, ha='right')
        ax_main.legend(loc='upper left')
        ax_main.grid(True, alpha=0.3)

        # Drawdown plot
        ax_dd.set_title('Drawdown')
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        line_dd, = ax_dd.plot(date_nums, drawdown, 'r-', linewidth=1)
        ax_dd.fill_between(date_nums, drawdown, 0, alpha=0.3, color='red')
        ax_dd.set_xlabel('Date')
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.set_xticks(label_indices)
        ax_dd.set_xticklabels(date_labels, rotation=45, ha='right')
        ax_dd.grid(True, alpha=0.3)

        # Sensitivity plot - pre-compute
        tc_range = np.arange(0, 101, 5) / 10000  # 0 to 100 bps
        final_values = [self.compute_portfolio_value(tc)[-1] for tc in tc_range]
        sharpe_values = [self.compute_metrics(self.compute_portfolio_value(tc))['sharpe']
                        for tc in tc_range]

        ax_sensitivity.set_title('TC Sensitivity')
        line_sens, = ax_sensitivity.plot(tc_range * 10000, final_values, 'b-', linewidth=2)
        ax_sensitivity.axhline(y=benchmark_values[-1], color='gray', linestyle='--',
                               label='Benchmark', alpha=0.7)
        vline = ax_sensitivity.axvline(x=10, color='red', linestyle='-', alpha=0.5, label='Current TC')
        ax_sensitivity.set_xlabel('Transaction Cost (bps)')
        ax_sensitivity.set_ylabel('Final Value ($)')
        ax_sensitivity.legend(loc='upper right')
        ax_sensitivity.grid(True, alpha=0.3)

        # Find breakeven TC
        breakeven_tc = self.compute_breakeven_tc(benchmark_values)

        # Metrics text
        def format_metrics(metrics: Dict[str, float], tc_bps: float, label: str = "") -> str:
            prefix = f"{label}\n" if label else ""
            return (
                f"{prefix}"
                f"TC: {tc_bps:.0f} bps ({tc_bps/100:.2f}%)\n"
                f"{'─' * 24}\n"
                f"Total Return:  {metrics['total_return']:>7.1f}%\n"
                f"Annual Return: {metrics['annual_return']:>7.1f}%\n"
                f"Volatility:    {metrics['volatility']:>7.1f}%\n"
                f"Sharpe Ratio:  {metrics['sharpe']:>7.2f}\n"
                f"Max Drawdown:  {metrics['max_drawdown']:>7.1f}%\n"
            )

        metrics_text = ax_metrics.text(
            0.05, 0.95, '', transform=ax_metrics.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=9
        )

        # Breakeven text
        breakeven_text = ax_metrics.text(
            0.05, 0.25, f"Breakeven TC: {breakeven_tc*10000:.0f} bps",
            transform=ax_metrics.transAxes,
            fontfamily='monospace', fontsize=10, fontweight='bold',
            color='green' if breakeven_tc > 0.001 else 'red'
        )

        def update(val=None):
            tc = slider_tc.val / 10000
            tc2 = slider_tc2.val / 10000

            # Update portfolio
            portfolio_values = self.compute_portfolio_value(tc)
            line_portfolio.set_ydata(portfolio_values)

            # Update drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak * 100
            line_dd.set_ydata(drawdown)

            # Update fill
            ax_dd.collections.clear()
            ax_dd.fill_between(date_nums, drawdown, 0, alpha=0.3, color='red')

            # Update comparison
            if tc2 > 0:
                compare_values = self.compute_portfolio_value(tc2)
                line_compare.set_data(date_nums, compare_values)
                line_compare.set_label(f'TC={slider_tc2.val:.0f}bps')
            else:
                line_compare.set_data([], [])

            # Update sensitivity marker
            vline.set_xdata([slider_tc.val])

            # Update metrics
            metrics = self.compute_metrics(portfolio_values)
            metrics_str = format_metrics(metrics, slider_tc.val, "PORTFOLIO")

            bench_metrics = self.compute_metrics(benchmark_values)
            metrics_str += f"\n{'─' * 24}\n"
            metrics_str += format_metrics(bench_metrics, 0, "BENCHMARK")

            if tc2 > 0:
                compare_values = self.compute_portfolio_value(tc2)
                compare_metrics = self.compute_metrics(compare_values)
                metrics_str += f"\n{'─' * 24}\n"
                metrics_str += format_metrics(compare_metrics, slider_tc2.val, "COMPARE")

            metrics_text.set_text(metrics_str)

            # Rescale
            all_values = [portfolio_values, benchmark_values]
            if tc2 > 0:
                all_values.append(compare_values)
            all_values = np.concatenate(all_values)
            ax_main.set_ylim(all_values.min() * 0.95, all_values.max() * 1.05)

            ax_dd.set_ylim(min(drawdown.min() * 1.1, -5), 2)

            ax_main.legend(loc='upper left')
            fig.canvas.draw_idle()

        slider_tc.on_changed(update)
        slider_tc2.on_changed(update)

        update()
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()


def load_walk_forward_results(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load results from walk-forward training output."""

    if path.endswith('.json'):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = torch.load(path)
        if isinstance(data, dict) and 'fold_results' in data:
            data = {'fold_results': data['fold_results']}

    # Extract from fold results
    all_dates = []
    all_returns = []
    all_weights = []

    for fold in data.get('fold_results', []):
        if 'test_dates' in fold and 'test_returns' in fold:
            all_dates.extend(fold['test_dates'])
            all_returns.extend(fold['test_returns'])
            # If weights aren't saved, use uniform based on top-k
            if 'test_weights' in fold:
                all_weights.extend(fold['test_weights'])

    if not all_dates:
        raise ValueError("No test data found in results file")

    dates = np.array(all_dates)
    returns = np.array(all_returns)

    # Create simple weights if not available (assume top-k equal weight)
    if all_weights:
        weights = np.array(all_weights)
    else:
        # Assume equal weight on stocks with non-zero returns
        weights = (returns != 0).astype(float)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

    return dates, returns, weights


def load_numpy_data(
    weights_path: str,
    returns_path: str,
    dates_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load from numpy files."""
    weights = np.load(weights_path)
    returns = np.load(returns_path)

    if dates_path:
        dates = np.load(dates_path, allow_pickle=True)
    else:
        dates = np.array([f"Day_{i}" for i in range(len(weights))])

    return dates, returns, weights


def main():
    parser = argparse.ArgumentParser(description='Interactive Transaction Cost Analysis')

    parser.add_argument('--results', type=str, default=None,
                       help='Path to walk-forward results (JSON or PT)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to weights numpy file')
    parser.add_argument('--returns', type=str, default=None,
                       help='Path to returns numpy file')
    parser.add_argument('--dates', type=str, default=None,
                       help='Path to dates numpy file')

    args = parser.parse_args()

    if args.results:
        print(f"Loading from {args.results}...")
        dates, returns, weights = load_walk_forward_results(args.results)
    elif args.weights and args.returns:
        print("Loading from numpy files...")
        dates, returns, weights = load_numpy_data(args.weights, args.returns, args.dates)
    else:
        parser.error("Must provide either --results or both --weights and --returns")

    analyzer = TransactionCostAnalyzer(dates, returns, weights)
    analyzer.create_interactive_plot()


if __name__ == '__main__':
    main()
