#!/usr/bin/env python3
"""
Walk-Forward Validation for Stock Prediction Model

Implements temporal cross-validation strategies that respect the time-series nature of financial data:

1. **Expanding Window (Walk-Forward)**:
   - Train on chunks 1..k, test on chunk k+1
   - Training set grows with each fold
   - Example: Train[1], Test[2] → Train[1,2], Test[3] → Train[1,2,3], Test[4]

2. **Sliding Window**:
   - Train on fixed-size recent window, test on next chunk
   - Training set maintains constant size
   - Example: Train[1,2], Test[3] → Train[2,3], Test[4] → Train[3,4], Test[5]

This addresses the temporal validation weakness identified in eval.md:
- Prevents lookahead bias in feature engineering
- Captures performance across different market regimes
- Provides more realistic out-of-sample estimates

Usage:
    python -m inference.walk_forward_validation \
        --data data/all_complete_dataset.h5 \
        --prices data/actual_prices.h5 \
        --num-folds 5 \
        --mode expanding
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import h5py

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.backtest_simulation import (
    DatasetLoader, ModelPredictor, TradingSimulator, PerformanceReporter
)
from utils.utils import pic_load, save_pickle


@dataclass
class FoldResult:
    """Results from a single fold evaluation."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int

    # Performance metrics
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_return_pct: float
    std_return_pct: float
    num_trades: int
    final_capital: float

    # Market regime info (optional)
    market_regime: Optional[str] = None
    market_return_pct: Optional[float] = None


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation."""
    mode: str  # 'expanding' or 'sliding'
    num_folds: int
    total_test_days: int

    # Aggregated metrics
    mean_return: float
    std_return: float
    mean_sharpe: float
    std_sharpe: float
    mean_sortino: float
    mean_max_drawdown: float
    mean_win_rate: float

    # Per-fold results
    fold_results: List[FoldResult]

    # Statistical tests
    t_stat_vs_zero: float
    p_value_vs_zero: float
    ci_95_lower: float
    ci_95_upper: float

    # Config
    config: Dict


class TemporalDataSplitter:
    """
    Splits dataset into temporal chunks for walk-forward validation.

    Ensures strict temporal ordering: training data always precedes test data.
    """

    def __init__(self, dataset_path: str, prices_path: Optional[str] = None):
        """
        Initialize splitter with dataset.

        Args:
            dataset_path: Path to features dataset (HDF5 or pickle)
            prices_path: Path to prices HDF5 (optional)
        """
        self.dataset_path = dataset_path
        self.prices_path = prices_path

        # Load dates from dataset
        print(f"Loading temporal information from: {dataset_path}")

        if dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5'):
            with h5py.File(dataset_path, 'r') as f:
                sample_ticker = list(f.keys())[0]
                dates_bytes = f[sample_ticker]['dates'][:]
                self.all_dates = sorted([d.decode('utf-8') for d in dates_bytes])
        else:
            data = pic_load(dataset_path)
            sample_ticker = list(data.keys())[0]
            self.all_dates = sorted(list(data[sample_ticker].keys()))

        print(f"  Date range: {self.all_dates[0]} to {self.all_dates[-1]}")
        print(f"  Total dates: {len(self.all_dates)}")

        # If prices file provided, use its trading days
        if prices_path and os.path.exists(prices_path):
            with h5py.File(prices_path, 'r') as f:
                sample_ticker = list(f.keys())[0]
                prices_dates_bytes = f[sample_ticker]['dates'][:]
                self.trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
            print(f"  Trading days from prices: {len(self.trading_days)}")
        else:
            self.trading_days = self.all_dates

    def get_expanding_window_folds(
        self,
        num_folds: int = 5,
        min_train_months: int = 12,
        test_months: int = 3,
        gap_days: int = 0
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Generate expanding window folds.

        Each fold has a growing training set and fixed test set size.

        Args:
            num_folds: Number of test folds to generate
            min_train_months: Minimum training period in months
            test_months: Test period length in months
            gap_days: Gap between train and test (to avoid lookahead)

        Returns:
            List of (train_dates, test_dates) tuples
        """
        # Estimate days per month
        days_per_month = 21  # ~21 trading days per month

        min_train_days = min_train_months * days_per_month
        test_days = test_months * days_per_month

        total_days = len(self.trading_days)

        # Calculate total test days needed
        total_test_days = num_folds * test_days

        # First test fold starts after min_train_days
        first_test_start = min_train_days + gap_days

        # Check we have enough data
        needed_days = first_test_start + total_test_days
        if needed_days > total_days:
            raise ValueError(
                f"Not enough data for {num_folds} folds. "
                f"Need {needed_days} days, have {total_days}"
            )

        folds = []
        for fold_idx in range(num_folds):
            # Test period for this fold
            test_start_idx = first_test_start + fold_idx * test_days
            test_end_idx = min(test_start_idx + test_days, total_days)

            # Training: all data before test (minus gap)
            train_end_idx = test_start_idx - gap_days
            train_start_idx = 0  # Expanding window: always start from beginning

            train_dates = self.trading_days[train_start_idx:train_end_idx]
            test_dates = self.trading_days[test_start_idx:test_end_idx]

            if len(test_dates) > 0:
                folds.append((train_dates, test_dates))

        return folds

    def get_sliding_window_folds(
        self,
        num_folds: int = 5,
        train_months: int = 12,
        test_months: int = 3,
        gap_days: int = 0
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Generate sliding window folds.

        Each fold has a fixed-size training window that slides forward.

        Args:
            num_folds: Number of test folds to generate
            train_months: Training period length in months (fixed)
            test_months: Test period length in months
            gap_days: Gap between train and test

        Returns:
            List of (train_dates, test_dates) tuples
        """
        days_per_month = 21

        train_days = train_months * days_per_month
        test_days = test_months * days_per_month

        total_days = len(self.trading_days)

        # Calculate stride (how much to move between folds)
        stride = test_days  # Non-overlapping test sets

        folds = []
        for fold_idx in range(num_folds):
            # Test period
            test_start_idx = train_days + gap_days + fold_idx * stride
            test_end_idx = min(test_start_idx + test_days, total_days)

            if test_end_idx > total_days:
                break

            # Training: fixed window before test
            train_end_idx = test_start_idx - gap_days
            train_start_idx = max(0, train_end_idx - train_days)

            train_dates = self.trading_days[train_start_idx:train_end_idx]
            test_dates = self.trading_days[test_start_idx:test_end_idx]

            if len(test_dates) > 0 and len(train_dates) > 0:
                folds.append((train_dates, test_dates))

        return folds

    def get_regime_based_folds(
        self,
        prices_path: str,
        regime_threshold: float = 0.1
    ) -> List[Tuple[List[str], List[str], str]]:
        """
        Generate folds based on market regimes (bull/bear/sideways).

        Identifies market regimes and creates test folds for each regime.

        Args:
            prices_path: Path to market index prices
            regime_threshold: Threshold for classifying bull/bear (e.g., +/-10%)

        Returns:
            List of (train_dates, test_dates, regime) tuples
        """
        # This is a more advanced feature - placeholder for now
        # Would analyze S&P 500 returns to classify regimes
        raise NotImplementedError("Regime-based folds not yet implemented")


class WalkForwardValidator:
    """
    Orchestrates walk-forward validation for stock prediction models.

    For each fold:
    1. (Optional) Retrain model on training data
    2. Evaluate on test data using backtesting simulation
    3. Collect metrics
    """

    def __init__(
        self,
        dataset_path: str,
        prices_path: str,
        model_path: str,
        bin_edges_path: str,
        num_folds: int = 5,
        mode: str = 'expanding',
        min_train_months: int = 12,
        test_months: int = 3,
        gap_days: int = 0,
        top_k: int = 5,
        horizon_idx: int = 0,
        confidence_percentile: float = 0.6,
        subset_size: int = 512,
        num_test_stocks: int = 1000,
        initial_capital: float = 100000.0,
        device: str = 'cuda',
        batch_size: int = 64,
        retrain: bool = False,
        seed: int = 42
    ):
        """
        Initialize walk-forward validator.

        Args:
            dataset_path: Path to features dataset
            prices_path: Path to actual prices HDF5
            model_path: Path to model checkpoint (used if not retraining)
            bin_edges_path: Path to bin edges
            num_folds: Number of test folds
            mode: 'expanding' or 'sliding' window
            min_train_months: Minimum training period
            test_months: Test period per fold
            gap_days: Gap between train/test to avoid lookahead
            top_k: Number of stocks to select
            horizon_idx: Prediction horizon (0=1d, 1=5d, etc.)
            confidence_percentile: Confidence threshold for filtering
            subset_size: Daily stock subset size
            num_test_stocks: Total stocks to consider
            initial_capital: Starting capital
            device: Compute device
            batch_size: Inference batch size
            retrain: Whether to retrain model on each fold
            seed: Random seed
        """
        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.model_path = model_path
        self.bin_edges_path = bin_edges_path
        self.num_folds = num_folds
        self.mode = mode
        self.min_train_months = min_train_months
        self.test_months = test_months
        self.gap_days = gap_days
        self.top_k = top_k
        self.horizon_idx = horizon_idx
        self.confidence_percentile = confidence_percentile
        self.subset_size = subset_size
        self.num_test_stocks = num_test_stocks
        self.initial_capital = initial_capital
        self.device = device
        self.batch_size = batch_size
        self.retrain = retrain
        self.seed = seed

        # Map horizon_idx to days
        horizon_map = {0: 1, 1: 5, 2: 10, 3: 20}
        self.horizon_days = horizon_map.get(horizon_idx, 1)

        # Initialize splitter
        self.splitter = TemporalDataSplitter(dataset_path, prices_path)

        # Results storage
        self.fold_results: List[FoldResult] = []

    def run_validation(self) -> WalkForwardResults:
        """
        Run the full walk-forward validation.

        Returns:
            WalkForwardResults with aggregated metrics
        """
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD VALIDATION ({self.mode.upper()} WINDOW)")
        print(f"{'='*80}")
        print(f"\nConfiguration:")
        print(f"  Mode: {self.mode}")
        print(f"  Folds: {self.num_folds}")
        print(f"  Min train months: {self.min_train_months}")
        print(f"  Test months per fold: {self.test_months}")
        print(f"  Gap days: {self.gap_days}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Horizon: {self.horizon_days} days")
        print(f"  Retrain on each fold: {self.retrain}")

        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Get folds based on mode
        if self.mode == 'expanding':
            folds = self.splitter.get_expanding_window_folds(
                num_folds=self.num_folds,
                min_train_months=self.min_train_months,
                test_months=self.test_months,
                gap_days=self.gap_days
            )
        elif self.mode == 'sliding':
            folds = self.splitter.get_sliding_window_folds(
                num_folds=self.num_folds,
                train_months=self.min_train_months,
                test_months=self.test_months,
                gap_days=self.gap_days
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        print(f"\nGenerated {len(folds)} folds")

        # Print fold summary
        print(f"\n{'─'*80}")
        print("FOLD SUMMARY")
        print(f"{'─'*80}")
        for i, (train_dates, test_dates) in enumerate(folds):
            print(f"  Fold {i+1}: Train [{train_dates[0]} to {train_dates[-1]}] ({len(train_dates)} days)")
            print(f"          Test  [{test_dates[0]} to {test_dates[-1]}] ({len(test_dates)} days)")

        # Run each fold
        self.fold_results = []

        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx + 1}/{len(folds)}")
            print(f"{'='*80}")

            fold_result = self._run_single_fold(
                fold_idx=fold_idx,
                train_dates=train_dates,
                test_dates=test_dates
            )
            self.fold_results.append(fold_result)

            # Print fold result
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Total Return: {fold_result.total_return_pct:+.2f}%")
            print(f"  Sharpe Ratio: {fold_result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {fold_result.max_drawdown_pct:.2f}%")
            print(f"  Win Rate: {fold_result.win_rate:.1f}%")

        # Aggregate results
        results = self._aggregate_results()

        return results

    def _run_single_fold(
        self,
        fold_idx: int,
        train_dates: List[str],
        test_dates: List[str]
    ) -> FoldResult:
        """
        Run evaluation on a single fold.

        Args:
            fold_idx: Fold index
            train_dates: Training period dates
            test_dates: Test period dates

        Returns:
            FoldResult with metrics
        """
        # If retraining, train model on train_dates (not implemented yet)
        if self.retrain:
            print(f"  ⚠️ Retraining not implemented - using pre-trained model")
            # TODO: Implement retraining
            # model = self._train_model(train_dates)

        # Load data for test period
        data_loader = DatasetLoader(
            dataset_path=self.dataset_path,
            num_test_stocks=self.num_test_stocks,
            subset_size=self.subset_size,
            prices_path=self.prices_path
        )

        # Filter to only test dates that exist in dataset
        available_test_dates = [d for d in test_dates if d in data_loader.all_dates]

        if len(available_test_dates) == 0:
            raise ValueError(f"No test dates available for fold {fold_idx}")

        # Preload features for test period
        data_loader.preload_features(available_test_dates)

        # Load model
        predictor = ModelPredictor(
            model_path=self.model_path,
            bin_edges_path=self.bin_edges_path,
            device=self.device,
            batch_size=self.batch_size
        )

        # Run simulation
        simulator = TradingSimulator(
            data_loader=data_loader,
            predictor=predictor,
            top_k=self.top_k,
            horizon_days=self.horizon_days,
            horizon_idx=self.horizon_idx,
            initial_capital=self.initial_capital,
            confidence_percentile=self.confidence_percentile,
            verbose=False
        )

        results = simulator.run_simulation(available_test_dates)

        # Compute Sortino ratio
        sortino = self._compute_sortino(results['daily_returns'])

        # Create fold result
        fold_result = FoldResult(
            fold_idx=fold_idx,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            train_days=len(train_dates),
            test_days=len(available_test_dates),
            total_return_pct=results['total_return_pct'],
            sharpe_ratio=results['sharpe_ratio'],
            sortino_ratio=sortino,
            max_drawdown_pct=results['max_drawdown_pct'],
            win_rate=results['win_rate'],
            avg_return_pct=results['avg_return_pct'],
            std_return_pct=results['std_return_pct'],
            num_trades=results['num_trades'],
            final_capital=results['final_capital']
        )

        return fold_result

    def _compute_sortino(self, daily_returns: List[float]) -> float:
        """Compute Sortino ratio from daily returns."""
        if not daily_returns:
            return 0.0

        returns = np.array(daily_returns)
        mean_return = np.mean(returns)

        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        # Annualize
        sortino = (mean_return * np.sqrt(252)) / (downside_std * np.sqrt(252))
        return sortino

    def _aggregate_results(self) -> WalkForwardResults:
        """Aggregate results across all folds."""

        returns = [f.total_return_pct for f in self.fold_results]
        sharpes = [f.sharpe_ratio for f in self.fold_results]
        sortinos = [f.sortino_ratio for f in self.fold_results]
        drawdowns = [f.max_drawdown_pct for f in self.fold_results]
        win_rates = [f.win_rate for f in self.fold_results]

        # Statistical test: is mean return significantly different from zero?
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        # 95% confidence interval
        ci_95 = stats.t.interval(
            0.95,
            len(returns) - 1,
            loc=np.mean(returns),
            scale=stats.sem(returns)
        )

        # Total test days
        total_test_days = sum(f.test_days for f in self.fold_results)

        results = WalkForwardResults(
            mode=self.mode,
            num_folds=len(self.fold_results),
            total_test_days=total_test_days,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_sharpe=float(np.mean(sharpes)),
            std_sharpe=float(np.std(sharpes)),
            mean_sortino=float(np.mean(sortinos)),
            mean_max_drawdown=float(np.mean(drawdowns)),
            mean_win_rate=float(np.mean(win_rates)),
            fold_results=self.fold_results,
            t_stat_vs_zero=float(t_stat),
            p_value_vs_zero=float(p_value),
            ci_95_lower=float(ci_95[0]),
            ci_95_upper=float(ci_95[1]),
            config={
                'mode': self.mode,
                'num_folds': self.num_folds,
                'min_train_months': self.min_train_months,
                'test_months': self.test_months,
                'gap_days': self.gap_days,
                'top_k': self.top_k,
                'horizon_idx': self.horizon_idx,
                'horizon_days': self.horizon_days,
                'confidence_percentile': self.confidence_percentile,
                'subset_size': self.subset_size,
                'num_test_stocks': self.num_test_stocks,
                'retrain': self.retrain,
                'seed': self.seed
            }
        )

        return results


def print_results_summary(results: WalkForwardResults):
    """Print a formatted summary of walk-forward validation results."""

    print(f"\n{'='*80}")
    print("WALK-FORWARD VALIDATION RESULTS")
    print(f"{'='*80}")

    print(f"\nConfiguration:")
    print(f"  Mode: {results.mode}")
    print(f"  Folds: {results.num_folds}")
    print(f"  Total test days: {results.total_test_days}")

    print(f"\n{'─'*80}")
    print("PERFORMANCE METRICS (across all folds)")
    print(f"{'─'*80}")

    print(f"\nTotal Return:")
    print(f"  Mean:   {results.mean_return:+.2f}%")
    print(f"  Std:    {results.std_return:.2f}%")
    print(f"  95% CI: [{results.ci_95_lower:+.2f}%, {results.ci_95_upper:+.2f}%]")

    print(f"\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio:  {results.mean_sharpe:.2f} (±{results.std_sharpe:.2f})")
    print(f"  Sortino Ratio: {results.mean_sortino:.2f}")
    print(f"  Max Drawdown:  {results.mean_max_drawdown:.2f}%")
    print(f"  Win Rate:      {results.mean_win_rate:.1f}%")

    print(f"\n{'─'*80}")
    print("STATISTICAL SIGNIFICANCE")
    print(f"{'─'*80}")

    print(f"\nTest: H0 = mean return equals zero")
    print(f"  t-statistic: {results.t_stat_vs_zero:.3f}")
    print(f"  p-value:     {results.p_value_vs_zero:.6f}")

    if results.p_value_vs_zero < 0.05:
        if results.mean_return > 0:
            print(f"  Result: ✅ SIGNIFICANT positive returns (p < 0.05)")
        else:
            print(f"  Result: ❌ SIGNIFICANT negative returns (p < 0.05)")
    else:
        print(f"  Result: ○ Returns not significantly different from zero")

    print(f"\n{'─'*80}")
    print("PER-FOLD BREAKDOWN")
    print(f"{'─'*80}")

    print(f"\n{'Fold':<6} {'Test Period':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print(f"{'-'*6} {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for fold in results.fold_results:
        test_period = f"{fold.test_start} to {fold.test_end}"
        print(f"{fold.fold_idx+1:<6} {test_period:<25} {fold.total_return_pct:>+9.2f}% {fold.sharpe_ratio:>8.2f} {fold.max_drawdown_pct:>7.2f}% {fold.win_rate:>7.1f}%")

    print(f"\n{'='*80}")


def plot_walk_forward_results(results: WalkForwardResults, output_dir: str = "."):
    """Generate visualizations for walk-forward validation results."""

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    # Plot 1: Returns across folds (bar chart with trend)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    folds = results.fold_results
    fold_indices = [f.fold_idx + 1 for f in folds]

    # Returns bar chart
    ax = axes[0, 0]
    returns = [f.total_return_pct for f in folds]
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
    bars = ax.bar(fold_indices, returns, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(results.mean_return, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {results.mean_return:+.2f}%')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Returns by Fold ({results.mode.title()} Window)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(fold_indices)

    # Sharpe ratio across folds
    ax = axes[0, 1]
    sharpes = [f.sharpe_ratio for f in folds]
    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpes]
    ax.bar(fold_indices, sharpes, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(results.mean_sharpe, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {results.mean_sharpe:.2f}')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Sharpe Ratio by Fold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(fold_indices)

    # Max Drawdown across folds
    ax = axes[1, 0]
    drawdowns = [f.max_drawdown_pct for f in folds]
    ax.bar(fold_indices, drawdowns, color='#e74c3c', edgecolor='black', linewidth=1.2)
    ax.axhline(results.mean_max_drawdown, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {results.mean_max_drawdown:.2f}%')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Max Drawdown by Fold (lower is better)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(fold_indices)

    # Win Rate across folds
    ax = axes[1, 1]
    win_rates = [f.win_rate for f in folds]
    colors = ['#2ecc71' if w > 50 else '#e74c3c' for w in win_rates]
    ax.bar(fold_indices, win_rates, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(50, color='black', linestyle=':', linewidth=1, label='50% baseline')
    ax.axhline(results.mean_win_rate, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {results.mean_win_rate:.1f}%')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate by Fold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(fold_indices)
    ax.set_ylim(0, 100)

    plt.suptitle(f'Walk-Forward Validation Results ({results.mode.title()} Window, {results.num_folds} Folds)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'walk_forward_{results.mode}_metrics.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    # Plot 2: Timeline visualization
    fig, ax = plt.subplots(figsize=(16, 8))

    colors_train = '#3498db'
    colors_test_pos = '#2ecc71'
    colors_test_neg = '#e74c3c'

    y_positions = list(range(len(folds), 0, -1))

    for fold, y_pos in zip(folds, y_positions):
        # Parse dates
        train_start = datetime.strptime(fold.train_start, '%Y-%m-%d')
        train_end = datetime.strptime(fold.train_end, '%Y-%m-%d')
        test_start = datetime.strptime(fold.test_start, '%Y-%m-%d')
        test_end = datetime.strptime(fold.test_end, '%Y-%m-%d')

        # Training bar
        ax.barh(y_pos, (train_end - train_start).days,
                left=train_start, height=0.4, color=colors_train, alpha=0.7,
                label='Training' if fold.fold_idx == 0 else None)

        # Test bar (colored by return)
        test_color = colors_test_pos if fold.total_return_pct > 0 else colors_test_neg
        ax.barh(y_pos, (test_end - test_start).days,
                left=test_start, height=0.4, color=test_color, alpha=0.8,
                label=f'Test (+)' if fold.fold_idx == 0 and fold.total_return_pct > 0 else
                      (f'Test (-)' if fold.fold_idx == 0 and fold.total_return_pct <= 0 else None))

        # Annotate with return
        ax.text(test_end + timedelta(days=10), y_pos, f'{fold.total_return_pct:+.1f}%',
               va='center', ha='left', fontsize=10, fontweight='bold',
               color=test_color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'Fold {f.fold_idx + 1}' for f in folds])
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title(f'Walk-Forward Timeline ({results.mode.title()} Window)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'walk_forward_{results.mode}_timeline.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    # Plot 3: Summary scorecard
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # Build summary text
    summary = f"""
WALK-FORWARD VALIDATION SUMMARY
{'='*50}

Mode: {results.mode.title()} Window
Folds: {results.num_folds}
Total Test Days: {results.total_test_days}

PERFORMANCE METRICS
{'-'*50}

Return:
  Mean:   {results.mean_return:+.2f}%
  Std:    {results.std_return:.2f}%
  95% CI: [{results.ci_95_lower:+.2f}%, {results.ci_95_upper:+.2f}%]

Risk-Adjusted:
  Sharpe Ratio:  {results.mean_sharpe:.2f} (±{results.std_sharpe:.2f})
  Sortino Ratio: {results.mean_sortino:.2f}
  Max Drawdown:  {results.mean_max_drawdown:.2f}%
  Win Rate:      {results.mean_win_rate:.1f}%

STATISTICAL TEST
{'-'*50}

H0: Mean return = 0
t-statistic: {results.t_stat_vs_zero:.3f}
p-value:     {results.p_value_vs_zero:.6f}

"""

    # Determine conclusion
    if results.p_value_vs_zero < 0.05 and results.mean_return > 0:
        conclusion = "SIGNIFICANT POSITIVE RETURNS"
        conclusion_color = 'green'
    elif results.p_value_vs_zero < 0.05 and results.mean_return < 0:
        conclusion = "SIGNIFICANT NEGATIVE RETURNS"
        conclusion_color = 'red'
    else:
        conclusion = "NO SIGNIFICANT RETURNS"
        conclusion_color = 'gray'

    summary += f"Result: {conclusion}"

    ax.text(0.5, 0.5, summary,
           transform=ax.transAxes,
           ha='center', va='center',
           fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor=conclusion_color, linewidth=3, alpha=0.9))

    plot_path = os.path.join(output_dir, f'walk_forward_{results.mode}_summary.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    print(f"\n Visualizations complete")


def save_results(results: WalkForwardResults, output_path: str):
    """Save results to file."""

    # Convert to serializable format
    results_dict = {
        'mode': results.mode,
        'num_folds': results.num_folds,
        'total_test_days': results.total_test_days,
        'mean_return': results.mean_return,
        'std_return': results.std_return,
        'mean_sharpe': results.mean_sharpe,
        'std_sharpe': results.std_sharpe,
        'mean_sortino': results.mean_sortino,
        'mean_max_drawdown': results.mean_max_drawdown,
        'mean_win_rate': results.mean_win_rate,
        't_stat_vs_zero': results.t_stat_vs_zero,
        'p_value_vs_zero': results.p_value_vs_zero,
        'ci_95_lower': results.ci_95_lower,
        'ci_95_upper': results.ci_95_upper,
        'config': results.config,
        'fold_results': [asdict(f) for f in results.fold_results]
    }

    torch.save(results_dict, output_path)
    print(f"\n Results saved to: {output_path}")

    # Also save as JSON for easier inspection
    json_path = output_path.replace('.pt', '.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f" JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation for Stock Prediction')

    # Data paths
    parser.add_argument('--data', type=str, default='data/all_complete_dataset.h5',
                       help='Path to features dataset')
    parser.add_argument('--prices', type=str, default='data/actual_prices_clean.h5',
                       help='Path to actual prices HDF5 (use _clean.h5 for validated data)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--bin-edges', type=str, default='data/adaptive_bin_edges.pt',
                       help='Path to bin edges')

    # Validation config
    parser.add_argument('--num-folds', type=int, default=5,
                       help='Number of test folds')
    parser.add_argument('--mode', type=str, default='expanding',
                       choices=['expanding', 'sliding'],
                       help='Window mode: expanding or sliding')
    parser.add_argument('--min-train-months', type=int, default=12,
                       help='Minimum training period in months')
    parser.add_argument('--test-months', type=int, default=3,
                       help='Test period per fold in months')
    parser.add_argument('--gap-days', type=int, default=0,
                       help='Gap days between train and test (to avoid lookahead)')

    # Trading strategy
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of stocks to select')
    parser.add_argument('--horizon-idx', type=int, default=0,
                       help='Prediction horizon (0=1d, 1=5d, 2=10d, 3=20d)')
    parser.add_argument('--confidence-percentile', type=float, default=0.6,
                       help='Confidence percentile for filtering')
    parser.add_argument('--subset-size', type=int, default=512,
                       help='Daily stock subset size')
    parser.add_argument('--num-test-stocks', type=int, default=1000,
                       help='Number of test stocks')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Starting capital')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Compute device')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Inference batch size')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain model on each fold (not implemented)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='walk_forward_results.pt',
                       help='Output file path')

    args = parser.parse_args()

    # Run validation
    validator = WalkForwardValidator(
        dataset_path=args.data,
        prices_path=args.prices,
        model_path=args.model,
        bin_edges_path=args.bin_edges,
        num_folds=args.num_folds,
        mode=args.mode,
        min_train_months=args.min_train_months,
        test_months=args.test_months,
        gap_days=args.gap_days,
        top_k=args.top_k,
        horizon_idx=args.horizon_idx,
        confidence_percentile=args.confidence_percentile,
        subset_size=args.subset_size,
        num_test_stocks=args.num_test_stocks,
        initial_capital=args.initial_capital,
        device=args.device,
        batch_size=args.batch_size,
        retrain=args.retrain,
        seed=args.seed
    )

    results = validator.run_validation()

    # Print summary
    print_results_summary(results)

    # Save results
    save_results(results, args.output)

    # Generate visualizations
    output_dir = os.path.dirname(args.output) or '.'
    plot_walk_forward_results(results, output_dir)

    print(f"\n{'='*80}")
    print(" WALK-FORWARD VALIDATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
