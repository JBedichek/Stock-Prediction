#!/usr/bin/env python3
"""
Interactive Portfolio Analysis Tool

Visualize portfolio performance with adjustable:
- Transaction costs
- Time ranges (start/end dates)
- Comparison scenarios

Usage:
    # Interactive mode (requires display)
    python inference/interactive_portfolio_analysis.py \
        --checkpoint checkpoints/walk_forward_portfolio/fold_0_best.pt \
        --data data/features.h5 \
        --prices data/prices.h5

    # Non-interactive mode (headless, saves to file)
    python inference/interactive_portfolio_analysis.py \
        --checkpoint checkpoints/walk_forward_portfolio/fold_0_best.pt \
        --data data/features.h5 \
        --prices data/prices.h5 \
        --save results/analysis.png
"""

import argparse
import sys
import os

# Check for headless mode BEFORE importing matplotlib.pyplot
# Parse --save argument early to determine if we need Agg backend
_save_mode = '--save' in sys.argv
_no_display = not os.environ.get('DISPLAY')

if _save_mode or _no_display:
    import matplotlib
    matplotlib.use('Agg')

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider, CheckButtons
from matplotlib.dates import DateFormatter, AutoDateLocator
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_portfolio_differentiable import PortfolioModel
from training.model import SimpleTransformerPredictor


class PortfolioAnalyzer:
    """Interactive portfolio analysis with adjustable parameters."""

    def __init__(
        self,
        checkpoint_path: str,
        data_path: str,
        prices_path: str,
        device: str = 'cuda',
        max_stocks: int = 500
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.prices_path = prices_path
        self.max_stocks = max_stocks

        # Load model
        print("Loading checkpoint...")
        self.model, self.config = self._load_model(checkpoint_path)

        # Load all available data
        print("Loading data...")
        self._load_all_data()

        print(f"Loaded {len(self.all_dates)} trading days")
        print(f"Date range: {self.all_dates[0]} to {self.all_dates[-1]}")

    def _load_model(self, checkpoint_path: str) -> Tuple[PortfolioModel, dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'args' in checkpoint:
            config = vars(checkpoint['args']) if hasattr(checkpoint['args'], '__dict__') else checkpoint['args']
        else:
            # Try to infer from model state dict
            config = {
                'input_dim': 128,
                'hidden_dim': 512,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': 0.1,
                'top_k': 10,
                'selection_method': 'gumbel',
                'min_temperature': 0.2,
                'horizon_idx': 0,
                'seq_len': 500
            }
            print("Warning: No config found in checkpoint, using defaults")

        # Store seq_len for inference
        self.seq_len = config.get('seq_len', 500)

        # Get input_dim from data if not in config
        with h5py.File(self.data_path, 'r') as f:
            sample_ticker = list(f.keys())[0]
            input_dim = f[sample_ticker]['features'].shape[1]
            config['input_dim'] = input_dim

        # Create encoder
        encoder = SimpleTransformerPredictor(
            input_dim=config.get('input_dim', 128),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            num_pred_days=4,
            pred_mode='regression'
        )

        # Create portfolio model
        model = PortfolioModel(
            encoder=encoder,
            k=config.get('top_k', 10),
            selection_method=config.get('selection_method', config.get('selection', 'gumbel')),
            initial_temperature=config.get('min_temp', config.get('min_temperature', 0.2)),
            min_temperature=config.get('min_temp', config.get('min_temperature', 0.2)),
            horizon_idx=config.get('horizon_idx', 0),
        )

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # Try loading directly
            try:
                model.load_state_dict(checkpoint)
            except:
                print("Warning: Could not load model weights")

        model = model.to(self.device)
        model.eval()

        return model, config

    def _load_all_data(self):
        """Load and preprocess all available data."""

        horizon_idx = self.config.get('horizon_idx', 0)
        horizon_days = [1, 5, 10, 20][horizon_idx]

        with h5py.File(self.data_path, 'r') as f_data, \
             h5py.File(self.prices_path, 'r') as f_prices:

            tickers = list(f_data.keys())

            # Get all unique dates from features
            sample_ticker = tickers[0]
            all_feature_dates = set()

            for ticker in tickers[:100]:  # Sample first 100 for speed
                if ticker in f_data:
                    dates = [d.decode() if isinstance(d, bytes) else d
                            for d in f_data[ticker]['dates'][:]]
                    all_feature_dates.update(dates)

            all_feature_dates = sorted(all_feature_dates)

            # Filter to dates we can actually use (have enough history and future)
            valid_dates = [d for d in all_feature_dates[self.seq_len:-horizon_days]]

            self.all_dates = np.array(valid_dates)
            self.tickers = tickers

            # Pre-cache ticker data for faster access
            print("Caching ticker data...")
            self.ticker_data = {}

            for ticker in tickers:
                if ticker not in f_data or ticker not in f_prices:
                    continue

                try:
                    feat_dates = [d.decode() if isinstance(d, bytes) else d
                                 for d in f_data[ticker]['dates'][:]]
                    price_dates = [d.decode() if isinstance(d, bytes) else d
                                  for d in f_prices[ticker]['dates'][:]]

                    # Try both 'prices' and 'close' keys for compatibility
                    if 'prices' in f_prices[ticker]:
                        prices = f_prices[ticker]['prices'][:]
                    elif 'close' in f_prices[ticker]:
                        prices = f_prices[ticker]['close'][:]
                    else:
                        continue

                    self.ticker_data[ticker] = {
                        'features': f_data[ticker]['features'][:],
                        'feat_dates': feat_dates,
                        'feat_date_to_idx': {d: i for i, d in enumerate(feat_dates)},
                        'prices': prices,
                        'price_dates': price_dates,
                        'price_date_to_idx': {d: i for i, d in enumerate(price_dates)},
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {ticker}: {e}")
                    continue

            print(f"Cached {len(self.ticker_data)} tickers")

    def run_inference_for_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback=None
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        """Run model inference for a specific date range."""

        top_k = self.config.get('top_k', 10)
        horizon_idx = self.config.get('horizon_idx', 0)
        horizon_days = [1, 5, 10, 20][horizon_idx]

        # Filter dates
        dates_in_range = [d for d in self.all_dates if start_date <= d <= end_date]

        all_dates = []
        all_returns = []
        all_weights = []
        all_scores = []

        total = len(dates_in_range)

        # Get expected feature dimension from config
        expected_feat_dim = self.config.get('input_dim', None)
        dim_mismatch_count = 0

        # Use tqdm if no callback provided (static mode), otherwise use callback (interactive)
        date_iterator = tqdm(dates_in_range, desc="Running inference", disable=progress_callback is not None)

        for idx, date in enumerate(date_iterator):
            if progress_callback and idx % 10 == 0:
                progress_callback(idx / total)

            # Gather data for all valid tickers on this date
            features_list = []
            returns_list = []
            valid_tickers = []

            for ticker, data in self.ticker_data.items():
                if date not in data['feat_date_to_idx']:
                    continue
                if date not in data['price_date_to_idx']:
                    continue

                t_idx = data['feat_date_to_idx'][date]
                p_idx = data['price_date_to_idx'][date]

                if t_idx < self.seq_len:
                    continue
                if p_idx + horizon_days >= len(data['prices']):
                    continue

                # Get features
                feat_seq = data['features'][t_idx - self.seq_len:t_idx]
                if len(feat_seq) != self.seq_len:
                    continue

                # Check feature dimension consistency
                if expected_feat_dim is not None:
                    if feat_seq.shape[1] != expected_feat_dim:
                        dim_mismatch_count += 1
                        continue
                else:
                    # Set expected dimension from first valid feature
                    expected_feat_dim = feat_seq.shape[1]
                    print(f"Using feature dimension: {expected_feat_dim}")

                # Get future return
                future_return = (data['prices'][p_idx + horizon_days] /
                               data['prices'][p_idx]) - 1

                if np.isnan(future_return) or np.isinf(future_return):
                    continue

                features_list.append(feat_seq)
                returns_list.append(future_return)
                valid_tickers.append(ticker)

                if len(features_list) >= self.max_stocks:
                    break

            if len(features_list) < top_k:
                continue

            # Run inference
            features = torch.tensor(np.stack(features_list), dtype=torch.float32).unsqueeze(0)
            returns = np.array(returns_list)
            masks = torch.ones(1, len(features_list))

            features = features.to(self.device)
            masks = masks.to(self.device)

            with torch.no_grad():
                scores, weights, _ = self.model(features, masks, hard=True)

            weights = weights.cpu().numpy()[0]
            scores = scores.cpu().numpy()[0]

            all_dates.append(date)
            all_returns.append(returns)
            all_weights.append(weights)
            all_scores.append(scores)

        if dim_mismatch_count > 0:
            print(f"Skipped {dim_mismatch_count} ticker-date pairs due to feature dimension mismatch")

        if not all_dates:
            print("WARNING: No valid dates found for inference!")
            return [], np.array([]), np.array([]), np.array([])

        # Pad to same size
        max_stocks = max(len(r) for r in all_returns)

        padded_returns = np.zeros((len(all_dates), max_stocks))
        padded_weights = np.zeros((len(all_dates), max_stocks))
        padded_scores = np.zeros((len(all_dates), max_stocks))

        for i, (r, w, s) in enumerate(zip(all_returns, all_weights, all_scores)):
            padded_returns[i, :len(r)] = r
            padded_weights[i, :len(w)] = w
            padded_scores[i, :len(s)] = s

        return all_dates, padded_returns, padded_weights, padded_scores

    def compute_portfolio_metrics(
        self,
        dates: List[str],
        returns: np.ndarray,
        weights: np.ndarray,
        transaction_cost: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[str]]:
        """Compute portfolio value and metrics using non-overlapping holding periods.

        Returns:
            values: Portfolio values at each rebalance point
            benchmark_values: Benchmark values at each rebalance point
            metrics: Performance metrics dictionary
            rebalance_dates: Dates at each rebalance point
        """

        if len(dates) == 0:
            return np.array([100.0]), np.array([0.0]), {}, []

        # Get horizon from config
        horizon_idx = self.config.get('horizon_idx', 0)
        horizon_days = [1, 5, 10, 20][horizon_idx]

        # Use non-overlapping holding periods
        rebalance_indices = list(range(0, len(dates), horizon_days))
        n_periods = len(rebalance_indices)

        # Extract data at rebalance points
        rebalance_weights = weights[rebalance_indices]
        rebalance_returns = returns[rebalance_indices]

        # Normalize weights
        weight_sums = rebalance_weights.sum(axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1)
        normalized_weights = rebalance_weights / weight_sums

        # Portfolio returns for each holding period
        portfolio_returns = (normalized_weights * rebalance_returns).sum(axis=1)

        # Turnover at each rebalance
        turnover = np.zeros(n_periods)
        for i in range(1, n_periods):
            turnover[i] = np.abs(normalized_weights[i] - normalized_weights[i-1]).sum()

        # Portfolio values
        values = np.zeros(n_periods)
        values[0] = 100.0

        for i in range(1, n_periods):
            net_return = portfolio_returns[i] - transaction_cost * turnover[i]
            values[i] = values[i-1] * (1 + net_return)

        # Benchmark (equal weight)
        valid_mask = rebalance_returns != 0
        valid_counts = valid_mask.sum(axis=1)
        benchmark_returns = np.where(valid_counts > 0,
                                     rebalance_returns.sum(axis=1) / valid_counts,
                                     0)

        benchmark_values = np.zeros(n_periods)
        benchmark_values[0] = 100.0
        for i in range(1, n_periods):
            benchmark_values[i] = benchmark_values[i-1] * (1 + benchmark_returns[i])

        # Metrics with correct annualization for horizon periods
        periods_per_year = 252 / horizon_days

        if n_periods > 1:
            period_returns = np.diff(values) / values[:-1]
            total_return = (values[-1] / values[0] - 1) * 100
            annual_factor = periods_per_year / len(period_returns)
            annual_return = ((values[-1] / values[0]) ** annual_factor - 1) * 100
            volatility = period_returns.std() * np.sqrt(periods_per_year) * 100
            sharpe = (period_returns.mean() / (period_returns.std() + 1e-8)) * np.sqrt(periods_per_year)

            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            max_drawdown = drawdown.min() * 100

            avg_turnover = turnover.mean() * 100
        else:
            total_return = annual_return = volatility = sharpe = max_drawdown = avg_turnover = 0

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'avg_turnover': avg_turnover,
            'num_periods': n_periods,
            'horizon_days': horizon_days
        }

        # Get dates at rebalance points
        rebalance_dates = [dates[i] for i in rebalance_indices]

        return values, benchmark_values, metrics, rebalance_dates

    def create_interactive_plot(self):
        """Create interactive matplotlib figure with sliders."""

        # Create figure
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle('Interactive Portfolio Analysis', fontsize=14, fontweight='bold')

        # Subplots
        ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=2)
        ax_metrics = plt.subplot2grid((4, 4), (0, 3), rowspan=2)
        ax_dd = plt.subplot2grid((4, 4), (2, 0), colspan=3)
        ax_monthly = plt.subplot2grid((4, 4), (3, 0), colspan=3)
        ax_info = plt.subplot2grid((4, 4), (2, 3), rowspan=2)

        ax_metrics.axis('off')
        ax_info.axis('off')

        # Convert dates to indices for range slider
        date_to_idx = {d: i for i, d in enumerate(self.all_dates)}
        num_dates = len(self.all_dates)

        # Initial range (last 2 years or all data)
        initial_end = num_dates - 1
        initial_start = max(0, num_dates - 504)  # ~2 years

        # Slider axes
        plt.subplots_adjust(bottom=0.15)
        ax_range = plt.axes([0.15, 0.08, 0.55, 0.03])
        ax_tc = plt.axes([0.15, 0.04, 0.25, 0.02])
        ax_tc2 = plt.axes([0.45, 0.04, 0.25, 0.02])

        # Range slider for dates
        range_slider = RangeSlider(
            ax_range, 'Date Range',
            0, num_dates - 1,
            valinit=(initial_start, initial_end),
            valstep=1
        )

        # TC sliders
        slider_tc = Slider(ax_tc, 'TC (bps)', 0, 100, valinit=10, valstep=1)
        slider_tc2 = Slider(ax_tc2, 'Compare TC', 0, 100, valinit=0, valstep=1)

        # Button axes
        ax_btn_update = plt.axes([0.75, 0.04, 0.08, 0.03])
        ax_btn_ytd = plt.axes([0.75, 0.08, 0.05, 0.02])
        ax_btn_1y = plt.axes([0.81, 0.08, 0.05, 0.02])
        ax_btn_3y = plt.axes([0.87, 0.08, 0.05, 0.02])
        ax_btn_all = plt.axes([0.93, 0.08, 0.05, 0.02])

        btn_update = Button(ax_btn_update, 'Update', color='lightblue')
        btn_ytd = Button(ax_btn_ytd, 'YTD', color='lightyellow')
        btn_1y = Button(ax_btn_1y, '1Y', color='lightyellow')
        btn_3y = Button(ax_btn_3y, '3Y', color='lightyellow')
        btn_all = Button(ax_btn_all, 'All', color='lightyellow')

        # State
        state = {
            'dates': [],
            'returns': None,
            'weights': None,
            'values': None,
            'benchmark': None,
            'needs_update': True
        }

        # Plot elements
        line_portfolio, = ax_main.plot([], [], 'b-', label='Portfolio', linewidth=2)
        line_benchmark, = ax_main.plot([], [], 'gray', label='Benchmark',
                                        linewidth=1.5, linestyle='--', alpha=0.7)
        line_compare, = ax_main.plot([], [], 'r-', label='Compare', linewidth=1.5, alpha=0.8)

        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Portfolio Value ($)')
        ax_main.set_title('Portfolio Value Over Time')
        ax_main.legend(loc='upper left')
        ax_main.grid(True, alpha=0.3)

        line_dd, = ax_dd.plot([], [], 'r-', linewidth=1)
        ax_dd.set_xlabel('Date')
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.set_title('Drawdown')
        ax_dd.grid(True, alpha=0.3)

        ax_monthly.set_xlabel('Month')
        ax_monthly.set_ylabel('Return (%)')
        ax_monthly.set_title('Monthly Returns')
        ax_monthly.grid(True, alpha=0.3)

        metrics_text = ax_metrics.text(
            0.05, 0.95, 'Click "Update" to compute',
            transform=ax_metrics.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=9
        )

        info_text = ax_info.text(
            0.05, 0.95, '',
            transform=ax_info.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=9
        )

        # Progress indicator
        progress_text = ax_main.text(
            0.5, 0.5, '', transform=ax_main.transAxes,
            ha='center', va='center', fontsize=14, fontweight='bold'
        )

        def format_metrics(metrics: Dict[str, float], tc_bps: float, label: str = "") -> str:
            if not metrics:
                return f"{label}\nNo data"
            horizon = metrics.get('horizon_days', 1)
            return (
                f"{label}\n"
                f"TC: {tc_bps:.0f} bps\n"
                f"{'─' * 22}\n"
                f"Total Return: {metrics['total_return']:>7.1f}%\n"
                f"Annual Return:{metrics['annual_return']:>7.1f}%\n"
                f"Volatility:   {metrics['volatility']:>7.1f}%\n"
                f"Sharpe Ratio: {metrics['sharpe']:>7.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:>7.1f}%\n"
                f"Avg Turnover: {metrics['avg_turnover']:>7.1f}%\n"
                f"Periods:      {metrics.get('num_periods', 0):>7d}\n"
                f"Horizon:      {horizon:>5d} days\n"
            )

        def update_plot():
            """Update plot with current data and TC."""
            if state['values'] is None or len(state['values']) == 0:
                return

            tc = slider_tc.val / 10000
            tc2 = slider_tc2.val / 10000

            original_dates = state['dates']

            # Recompute with current TC - returns values at rebalance points
            values, benchmark, metrics, dates = self.compute_portfolio_metrics(
                original_dates, state['returns'], state['weights'], tc
            )

            date_nums = np.arange(len(dates))

            state['values'] = values
            state['benchmark'] = benchmark

            # Update main plot
            line_portfolio.set_data(date_nums, values)
            line_benchmark.set_data(date_nums, benchmark)

            # Comparison
            if tc2 > 0:
                values2, _, metrics2, _ = self.compute_portfolio_metrics(
                    original_dates, state['returns'], state['weights'], tc2
                )
                line_compare.set_data(date_nums, values2)
                line_compare.set_label(f'TC={slider_tc2.val:.0f}bps')
            else:
                line_compare.set_data([], [])
                metrics2 = None

            # Drawdown
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak * 100
            line_dd.set_data(date_nums, drawdown)

            # Monthly returns
            ax_monthly.clear()
            if len(dates) > 20:
                monthly_returns = []
                monthly_labels = []
                current_month = dates[0][:7]
                month_start_val = values[0]

                for i, d in enumerate(dates):
                    if d[:7] != current_month:
                        month_return = (values[i-1] / month_start_val - 1) * 100
                        monthly_returns.append(month_return)
                        monthly_labels.append(current_month)
                        current_month = d[:7]
                        month_start_val = values[i-1]

                # Last month
                month_return = (values[-1] / month_start_val - 1) * 100
                monthly_returns.append(month_return)
                monthly_labels.append(current_month)

                colors = ['green' if r > 0 else 'red' for r in monthly_returns]
                ax_monthly.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
                ax_monthly.set_xticks(range(0, len(monthly_labels), max(1, len(monthly_labels)//12)))
                ax_monthly.set_xticklabels(
                    [monthly_labels[i] for i in range(0, len(monthly_labels), max(1, len(monthly_labels)//12))],
                    rotation=45, ha='right'
                )
                ax_monthly.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax_monthly.set_ylabel('Return (%)')
                ax_monthly.set_title('Monthly Returns')
                ax_monthly.grid(True, alpha=0.3, axis='y')

            # Set axis limits
            n_labels = min(8, len(dates))
            label_indices = np.linspace(0, len(dates)-1, n_labels, dtype=int)
            date_labels = [dates[i][:10] for i in label_indices]

            ax_main.set_xticks(label_indices)
            ax_main.set_xticklabels(date_labels, rotation=45, ha='right')

            all_values = np.concatenate([values, benchmark])
            if tc2 > 0:
                all_values = np.concatenate([all_values, values2])
            ax_main.set_xlim(-1, len(dates))
            ax_main.set_ylim(all_values.min() * 0.95, all_values.max() * 1.05)
            ax_main.legend(loc='upper left')

            ax_dd.set_xticks(label_indices)
            ax_dd.set_xticklabels(date_labels, rotation=45, ha='right')
            ax_dd.set_xlim(-1, len(dates))
            ax_dd.set_ylim(min(drawdown.min() * 1.1, -5), 2)

            # Update metrics
            metrics_str = format_metrics(metrics, slider_tc.val, "PORTFOLIO")

            # Benchmark metrics (use same horizon as portfolio)
            horizon_days = metrics.get('horizon_days', 1)
            periods_per_year = 252 / horizon_days
            bench_returns = np.diff(benchmark) / benchmark[:-1]
            bench_metrics = {
                'total_return': (benchmark[-1] / benchmark[0] - 1) * 100,
                'annual_return': ((benchmark[-1] / benchmark[0]) ** (periods_per_year / len(benchmark)) - 1) * 100,
                'volatility': bench_returns.std() * np.sqrt(periods_per_year) * 100,
                'sharpe': (bench_returns.mean() / (bench_returns.std() + 1e-8)) * np.sqrt(periods_per_year),
                'max_drawdown': ((benchmark - np.maximum.accumulate(benchmark)) /
                                np.maximum.accumulate(benchmark)).min() * 100,
                'avg_turnover': 0,
                'num_periods': len(benchmark),
                'horizon_days': horizon_days
            }
            metrics_str += f"\n{'─' * 22}\n"
            metrics_str += format_metrics(bench_metrics, 0, "BENCHMARK")

            if metrics2:
                metrics_str += f"\n{'─' * 22}\n"
                metrics_str += format_metrics(metrics2, slider_tc2.val, "COMPARE")

            metrics_text.set_text(metrics_str)

            # Info text
            info_str = (
                f"Date Range:\n"
                f"  {dates[0]} to\n"
                f"  {dates[-1]}\n\n"
                f"Model Config:\n"
                f"  top_k: {self.config.get('top_k', 'N/A')}\n"
                f"  selection: {self.config.get('selection_method', self.config.get('selection', 'N/A'))}\n"
                f"  hidden_dim: {self.config.get('hidden_dim', 'N/A')}\n"
                f"  num_layers: {self.config.get('num_layers', 'N/A')}\n"
            )
            info_text.set_text(info_str)

            fig.canvas.draw_idle()

        def run_inference(event=None):
            """Run inference for selected date range."""
            start_idx, end_idx = int(range_slider.val[0]), int(range_slider.val[1])
            start_date = self.all_dates[start_idx]
            end_date = self.all_dates[end_idx]

            progress_text.set_text('Computing...')
            fig.canvas.draw()
            fig.canvas.flush_events()

            def progress_cb(p):
                progress_text.set_text(f'Computing... {p*100:.0f}%')
                fig.canvas.draw()
                fig.canvas.flush_events()

            dates, returns, weights, scores = self.run_inference_for_range(
                start_date, end_date, progress_cb
            )

            progress_text.set_text('')

            if len(dates) == 0:
                metrics_text.set_text("No valid data in range")
                return

            state['dates'] = dates
            state['returns'] = returns
            state['weights'] = weights

            update_plot()

        def on_tc_change(val):
            """Handle TC slider change."""
            if state['returns'] is not None:
                update_plot()

        def set_range_ytd(event):
            """Set range to YTD."""
            current_year = self.all_dates[-1][:4]
            start_date = f"{current_year}-01-01"
            start_idx = 0
            for i, d in enumerate(self.all_dates):
                if d >= start_date:
                    start_idx = i
                    break
            range_slider.set_val((start_idx, len(self.all_dates) - 1))

        def set_range_1y(event):
            """Set range to last 1 year."""
            range_slider.set_val((max(0, len(self.all_dates) - 252), len(self.all_dates) - 1))

        def set_range_3y(event):
            """Set range to last 3 years."""
            range_slider.set_val((max(0, len(self.all_dates) - 756), len(self.all_dates) - 1))

        def set_range_all(event):
            """Set range to all data."""
            range_slider.set_val((0, len(self.all_dates) - 1))

        def on_range_change(val):
            """Update date display when range changes."""
            start_idx, end_idx = int(val[0]), int(val[1])
            start_date = self.all_dates[start_idx]
            end_date = self.all_dates[end_idx]
            range_slider.label.set_text(f'Range: {start_date} to {end_date}')

        # Connect callbacks
        btn_update.on_clicked(run_inference)
        btn_ytd.on_clicked(set_range_ytd)
        btn_1y.on_clicked(set_range_1y)
        btn_3y.on_clicked(set_range_3y)
        btn_all.on_clicked(set_range_all)
        slider_tc.on_changed(on_tc_change)
        slider_tc2.on_changed(on_tc_change)
        range_slider.on_changed(on_range_change)

        # Initial range display
        on_range_change(range_slider.val)

        # Instructions
        ax_info.text(
            0.05, 0.3,
            "Instructions:\n"
            "1. Adjust date range\n"
            "2. Click 'Update' to\n"
            "   run inference\n"
            "3. Adjust TC sliders\n"
            "   for instant update",
            transform=ax_info.transAxes,
            fontsize=8, style='italic'
        )

        # Adjust layout - avoid tight_layout with interactive widgets
        fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15, hspace=0.35, wspace=0.3)
        plt.show()

    def _compute_turnover(self, weights: np.ndarray) -> np.ndarray:
        """Compute daily turnover from weight changes."""
        turnover = np.zeros(len(weights))
        for i in range(1, len(weights)):
            # Normalize weights
            w_prev = weights[i-1] / (weights[i-1].sum() + 1e-8)
            w_curr = weights[i] / (weights[i].sum() + 1e-8)
            turnover[i] = np.abs(w_curr - w_prev).sum()
        return turnover

    def run_static_analysis(
        self,
        start_date: str,
        end_date: str,
        transaction_cost: float,
        save_path: str
    ):
        """Run non-interactive analysis and save plot to file."""
        print(f"\nRunning analysis from {start_date} to {end_date}...")
        print(f"Transaction cost: {transaction_cost*100:.2f}%")

        # Get horizon days from config
        horizon_idx = self.config.get('horizon_idx', 0)
        horizon_days = [1, 5, 10, 20][horizon_idx]
        print(f"Horizon: {horizon_days} days (rebalancing every {horizon_days} trading days)")

        # Run inference - returns (dates, returns, weights, scores)
        dates, stock_returns, weights, scores = self.run_inference_for_range(
            start_date, end_date
        )

        if len(dates) == 0:
            print("ERROR: No valid dates in range!")
            return

        print(f"Processed {len(dates)} trading days")

        # Use non-overlapping holding periods
        # Only use every horizon_days-th date to avoid overlapping returns
        rebalance_indices = list(range(0, len(dates), horizon_days))
        n_periods = len(rebalance_indices)

        print(f"Using {n_periods} non-overlapping {horizon_days}-day holding periods")

        # Compute portfolio value with transaction costs over non-overlapping periods
        values = np.zeros(n_periods)
        values[0] = 100.0

        rebalance_dates = [dates[i] for i in rebalance_indices]
        rebalance_weights = weights[rebalance_indices]
        rebalance_returns = stock_returns[rebalance_indices]

        # Normalize weights to sum to 1
        weight_sums = rebalance_weights.sum(axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1)
        normalized_weights = rebalance_weights / weight_sums

        # Debug: check weight statistics
        avg_weight_sum = weight_sums.mean()
        avg_nonzero = (rebalance_weights > 0).sum(axis=1).mean()
        print(f"Avg weight sum before norm: {avg_weight_sum:.2f}, Avg stocks selected: {avg_nonzero:.1f}")

        # Debug: check return statistics
        flat_returns = rebalance_returns[rebalance_returns != 0]
        if len(flat_returns) > 0:
            print(f"Return stats: mean={flat_returns.mean()*100:.2f}%, std={flat_returns.std()*100:.2f}%, "
                  f"min={flat_returns.min()*100:.2f}%, max={flat_returns.max()*100:.2f}%")

        turnover = self._compute_turnover(normalized_weights)

        for i in range(1, n_periods):
            # Portfolio return for this holding period (using normalized weights)
            port_return = (normalized_weights[i] * rebalance_returns[i]).sum()
            net_return = port_return - transaction_cost * turnover[i]
            values[i] = values[i-1] * (1 + net_return)

        # Compute benchmark (equal weight on valid stocks) over same periods
        valid_mask = rebalance_returns != 0
        valid_counts = valid_mask.sum(axis=1)
        benchmark_returns = np.where(valid_counts > 0,
                                     rebalance_returns.sum(axis=1) / valid_counts,
                                     0)
        bench_values = np.zeros(n_periods)
        bench_values[0] = 100.0
        for i in range(1, n_periods):
            bench_values[i] = bench_values[i-1] * (1 + benchmark_returns[i])

        # Use rebalance_dates for plotting
        dates = rebalance_dates

        # Compute metrics
        # These are horizon-period returns, not daily returns
        period_returns = np.diff(values) / values[:-1]
        total_return = (values[-1] / values[0] - 1) * 100

        # Annualization: periods_per_year = 252 / horizon_days
        periods_per_year = 252 / horizon_days
        annual_factor = periods_per_year / len(period_returns) if len(period_returns) > 0 else 1
        annual_return = ((values[-1] / values[0]) ** annual_factor - 1) * 100
        volatility = period_returns.std() * np.sqrt(periods_per_year) * 100
        sharpe = (period_returns.mean() / (period_returns.std() + 1e-8)) * np.sqrt(periods_per_year)

        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        max_drawdown = drawdown.min()

        bench_total = (bench_values[-1] / bench_values[0] - 1) * 100

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Portfolio value
        ax = axes[0, 0]
        ax.plot(range(len(values)), values, 'b-', linewidth=2, label='Portfolio')
        ax.plot(range(len(bench_values)), bench_values, 'gray', linewidth=1.5,
                linestyle='--', alpha=0.7, label='Benchmark')
        ax.axhline(100, color='black', linestyle=':', alpha=0.3)
        ax.set_xlabel(f'Holding Period ({horizon_days}-day)')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title(f'Portfolio Value ({start_date} to {end_date})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Drawdown
        ax = axes[0, 1]
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
        ax.set_xlabel(f'Holding Period ({horizon_days}-day)')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Over Time')
        ax.grid(True, alpha=0.3)

        # Period returns distribution
        ax = axes[1, 0]
        ax.hist(period_returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(period_returns.mean() * 100, color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {period_returns.mean()*100:.3f}%')
        ax.set_xlabel(f'{horizon_days}-Day Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{horizon_days}-Day Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = (
            f"PORTFOLIO METRICS\n"
            f"{'─' * 30}\n"
            f"Transaction Cost:   {transaction_cost*100:.2f}%\n"
            f"Horizon:            {horizon_days} days\n"
            f"Holding Periods:    {n_periods}\n"
            f"Avg Turnover:       {turnover.mean()*100:.1f}%\n"
            f"\n"
            f"RETURNS\n"
            f"{'─' * 30}\n"
            f"Total Return:       {total_return:+.2f}%\n"
            f"Annual Return:      {annual_return:+.2f}%\n"
            f"Benchmark Return:   {bench_total:+.2f}%\n"
            f"Excess Return:      {total_return - bench_total:+.2f}%\n"
            f"\n"
            f"RISK\n"
            f"{'─' * 30}\n"
            f"Volatility (ann):   {volatility:.2f}%\n"
            f"Sharpe Ratio:       {sharpe:.2f}\n"
            f"Max Drawdown:       {max_drawdown:.2f}%\n"
            f"Win Rate:           {(period_returns > 0).mean()*100:.1f}%\n"
        )
        ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Portfolio Analysis: {start_date} to {end_date}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nResults saved to: {save_path}")
        print(f"\nSummary:")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Interactive Portfolio Analysis')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to features HDF5 file')
    parser.add_argument('--prices', type=str, required=True,
                       help='Path to prices HDF5 file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max-stocks', type=int, default=500,
                       help='Maximum stocks to consider per day')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to file instead of showing (e.g., results/analysis.png)')
    parser.add_argument('--start-date', type=str, default="2025-01-01",
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default="2025-03-31",
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--tc', type=float, default=0.001,
                       help='Transaction cost (default: 0.1%%)')

    args = parser.parse_args()

    # Check if we're in a headless environment
    import matplotlib
    if args.save or not os.environ.get('DISPLAY'):
        matplotlib.use('Agg')
        print("Running in non-interactive mode (saving to file)")

    analyzer = PortfolioAnalyzer(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        prices_path=args.prices,
        device=args.device,
        max_stocks=args.max_stocks
    )

    # Determine if we need to use static mode
    headless = not os.environ.get('DISPLAY')

    if args.save:
        # Explicit save mode
        start = args.start_date or analyzer.all_dates[0]
        end = args.end_date or analyzer.all_dates[-1]
        analyzer.run_static_analysis(start, end, args.tc, args.save)
    elif headless:
        # No display available - auto-save to default location
        print("\nNo display available. Running in static mode.")
        save_path = 'results/portfolio_analysis_static.png'
        start = args.start_date or analyzer.all_dates[0]
        end = args.end_date or analyzer.all_dates[-1]
        analyzer.run_static_analysis(start, end, args.tc, save_path)
    else:
        analyzer.create_interactive_plot()


if __name__ == '__main__':
    main()
