#!/usr/bin/env python3
"""
Principled Evaluation of Walk-Forward Checkpoints

Evaluates model checkpoints using statistically rigorous metrics:
1. Information Coefficient (IC) - correlation between predictions and actual returns
2. Information Ratio (IR) - IC consistency (mean/std)
3. Quantile Analysis - do top predictions actually outperform?
4. Baseline Comparisons - momentum, random selection
5. Statistical Significance - t-tests, confidence intervals
6. Trading Simulation - as ONE metric among many (not primary)

Usage:
    python -m inference.principled_evaluation \
        --checkpoint-dir checkpoints/walk_forward_seed42 \
        --data all_complete_dataset.h5 \
        --prices actual_prices.h5
"""

import os
import sys
import glob
import argparse
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import SimpleTransformerPredictor


# ============================================================================
# ENHANCED STATISTICAL METHODS
# ============================================================================

def block_bootstrap(
    daily_returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    seed: int = None
) -> Dict:
    """
    Block bootstrap that preserves time-series autocorrelation.

    Unlike standard bootstrap which assumes IID, block bootstrap samples
    contiguous blocks of observations, preserving serial correlation.
    This gives more realistic confidence intervals for time series.

    Args:
        daily_returns: Array of daily returns
        n_bootstrap: Number of bootstrap samples
        block_size: Size of each block (default 5 days)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with bootstrap statistics
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(daily_returns)
    if n < block_size * 2:
        # Fall back to standard bootstrap for short series
        block_size = max(1, n // 5)

    # Number of blocks needed
    n_blocks = int(np.ceil(n / block_size))

    bootstrap_total_returns = []

    for _ in range(n_bootstrap):
        # Randomly select starting positions for blocks
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)

        # Concatenate blocks
        resampled = []
        for start in block_starts:
            resampled.extend(daily_returns[start:start + block_size])
        resampled = np.array(resampled[:n])  # Trim to original length

        # Compound returns
        capital = 100000.0
        for ret in resampled:
            capital *= (1 + ret)
        bootstrap_total_returns.append((capital / 100000.0 - 1) * 100)

    bootstrap_total_returns = np.array(bootstrap_total_returns)

    return {
        'mean': float(np.mean(bootstrap_total_returns)),
        'std': float(np.std(bootstrap_total_returns)),
        'ci_lower': float(np.percentile(bootstrap_total_returns, 5)),
        'ci_upper': float(np.percentile(bootstrap_total_returns, 95)),
        'ci_99_lower': float(np.percentile(bootstrap_total_returns, 0.5)),
        'ci_99_upper': float(np.percentile(bootstrap_total_returns, 99.5)),
        'prob_positive': float(np.mean(bootstrap_total_returns > 0)),
        'var_5pct': float(np.percentile(bootstrap_total_returns, 5)),
        'cvar_5pct': float(np.mean(bootstrap_total_returns[bootstrap_total_returns <= np.percentile(bootstrap_total_returns, 5)])) if len(bootstrap_total_returns) > 0 else 0,
        'all_returns': bootstrap_total_returns.tolist(),
    }


def permutation_test(
    model_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: int = None
) -> Dict:
    """
    Permutation test for comparing model vs baseline returns.

    More powerful than t-tests for small samples and non-normal distributions.
    Tests the null hypothesis that model and baseline come from same distribution.

    Args:
        model_returns: Array of model returns per trade
        baseline_returns: Array of baseline returns per trade
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        Dictionary with permutation test results
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(model_returns)
    if n != len(baseline_returns):
        raise ValueError("Model and baseline must have same length")

    # Observed difference in means
    observed_diff = np.mean(model_returns) - np.mean(baseline_returns)

    # Pool all returns
    combined = np.concatenate([model_returns, baseline_returns])

    # Generate null distribution by permuting labels
    null_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_model = combined[:n]
        perm_baseline = combined[n:]
        null_diffs.append(np.mean(perm_model) - np.mean(perm_baseline))

    null_diffs = np.array(null_diffs)

    # One-sided p-value: fraction of permutations with diff >= observed
    p_value = np.mean(null_diffs >= observed_diff)

    # Two-sided p-value
    p_value_two_sided = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    return {
        'observed_diff': float(observed_diff),
        'p_value_one_sided': float(p_value),
        'p_value_two_sided': float(p_value_two_sided),
        'null_mean': float(np.mean(null_diffs)),
        'null_std': float(np.std(null_diffs)),
        'significant_5pct': p_value < 0.05,
        'significant_1pct': p_value < 0.01,
    }


def compute_effect_size(
    model_returns: np.ndarray,
    baseline_returns: np.ndarray
) -> Dict:
    """
    Compute effect size metrics (Cohen's d and Hedges' g).

    Effect size tells you the practical significance of a difference,
    independent of sample size. P-values can be significant with tiny
    effects given enough data.

    Cohen's d interpretation:
    - 0.2: Small effect
    - 0.5: Medium effect
    - 0.8: Large effect

    Args:
        model_returns: Array of model returns
        baseline_returns: Array of baseline returns

    Returns:
        Dictionary with effect size metrics
    """
    n1, n2 = len(model_returns), len(baseline_returns)
    mean1, mean2 = np.mean(model_returns), np.mean(baseline_returns)
    var1, var2 = np.var(model_returns, ddof=1), np.var(baseline_returns, ddof=1)

    # Cohen's d (pooled standard deviation)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std > 1e-10:
        cohens_d = (mean1 - mean2) / pooled_std
    else:
        cohens_d = 0.0

    # Hedges' g (bias-corrected for small samples)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    hedges_g = cohens_d * correction

    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        'cohens_d': float(cohens_d),
        'hedges_g': float(hedges_g),
        'interpretation': interpretation,
        'mean_diff': float(mean1 - mean2),
        'pooled_std': float(pooled_std),
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction for multiple hypothesis testing.

    When testing multiple hypotheses (e.g., vs random AND vs momentum),
    the chance of at least one false positive increases. Bonferroni
    divides alpha by number of tests to control family-wise error rate.

    Args:
        p_values: List of p-values from individual tests
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with corrected significance thresholds
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    # Holm-Bonferroni (step-down, more powerful)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    holm_significant = []
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        threshold = alpha / (n_tests - i)
        holm_significant.append((int(idx), p < threshold))

    # Benjamini-Hochberg FDR control (even more powerful)
    bh_significant = []
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        threshold = (i + 1) * alpha / n_tests
        bh_significant.append((int(idx), p < threshold))

    return {
        'bonferroni_alpha': float(corrected_alpha),
        'bonferroni_significant': [p < corrected_alpha for p in p_values],
        'holm_significant': [sig for _, sig in sorted(holm_significant)],
        'bh_fdr_significant': [sig for _, sig in sorted(bh_significant)],
        'n_tests': n_tests,
    }


def pooled_significance_test(
    all_fold_model_returns: List[np.ndarray],
    all_fold_baseline_returns: List[np.ndarray],
    test_type: str = 'paired'
) -> Dict:
    """
    Compute aggregate significance by pooling daily returns across folds.

    More powerful than averaging per-fold statistics because it uses
    all available data points for a single test.

    Args:
        all_fold_model_returns: List of arrays, one per fold
        all_fold_baseline_returns: List of arrays, one per fold
        test_type: 'paired' or 'independent'

    Returns:
        Dictionary with pooled test results
    """
    # Concatenate all folds
    pooled_model = np.concatenate(all_fold_model_returns)
    pooled_baseline = np.concatenate(all_fold_baseline_returns)

    n_total = len(pooled_model)

    # Summary statistics
    model_mean = np.mean(pooled_model)
    baseline_mean = np.mean(pooled_baseline)
    diff_mean = model_mean - baseline_mean

    # Paired t-test on pooled data
    if test_type == 'paired':
        t_stat, p_value = stats.ttest_rel(pooled_model, pooled_baseline)
    else:
        t_stat, p_value = stats.ttest_ind(pooled_model, pooled_baseline)

    p_value_one_sided = p_value / 2  # One-sided

    # Wilcoxon signed-rank test (non-parametric)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            pooled_model - pooled_baseline,
            alternative='greater'
        )
    except ValueError:
        # Wilcoxon fails if all differences are zero
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    # Effect size on pooled data
    effect_size = compute_effect_size(pooled_model, pooled_baseline)

    return {
        'n_total_observations': n_total,
        'n_folds': len(all_fold_model_returns),
        'pooled_model_mean': float(model_mean * 100),  # As percentage
        'pooled_baseline_mean': float(baseline_mean * 100),
        'pooled_diff_mean': float(diff_mean * 100),
        't_stat': float(t_stat),
        'p_value_one_sided': float(p_value_one_sided),
        'p_value_two_sided': float(p_value),
        'wilcoxon_p_one_sided': float(wilcoxon_p),
        'cohens_d': effect_size['cohens_d'],
        'effect_interpretation': effect_size['interpretation'],
    }


def regime_stratified_analysis(
    daily_model_returns: np.ndarray,
    daily_baseline_returns: np.ndarray,
    daily_volatility: np.ndarray = None,
    n_regimes: int = 3
) -> Dict:
    """
    Stratify analysis by market volatility regime.

    Models may perform differently in high vs low volatility periods.
    This analysis splits data into regimes and computes metrics separately.

    Args:
        daily_model_returns: Model returns per day
        daily_baseline_returns: Baseline returns per day
        daily_volatility: Volatility measure per day (computed if None)
        n_regimes: Number of volatility regimes (default 3: low/med/high)

    Returns:
        Dictionary with per-regime statistics
    """
    n = len(daily_model_returns)

    # Compute volatility if not provided (rolling 20-day realized vol)
    if daily_volatility is None:
        # Use absolute return as volatility proxy
        daily_volatility = np.abs(daily_baseline_returns)

    # Classify into regimes by volatility quantile
    regime_thresholds = np.percentile(daily_volatility, [100/n_regimes * i for i in range(1, n_regimes)])

    regimes = np.digitize(daily_volatility, regime_thresholds)
    regime_names = ['low_vol', 'medium_vol', 'high_vol'] if n_regimes == 3 else [f'regime_{i}' for i in range(n_regimes)]

    results = {}

    for regime_idx in range(n_regimes):
        mask = regimes == regime_idx
        n_days = np.sum(mask)

        if n_days < 5:
            continue

        model_regime = daily_model_returns[mask]
        baseline_regime = daily_baseline_returns[mask]

        # Compute metrics for this regime
        excess = np.mean(model_regime) - np.mean(baseline_regime)

        # T-test within regime
        if len(model_regime) > 2:
            t_stat, p_val = stats.ttest_rel(model_regime, baseline_regime)
            p_val_one_sided = p_val / 2
        else:
            t_stat, p_val_one_sided = 0, 1

        results[regime_names[regime_idx]] = {
            'n_days': int(n_days),
            'pct_of_total': float(n_days / n * 100),
            'model_mean_return': float(np.mean(model_regime) * 100),
            'baseline_mean_return': float(np.mean(baseline_regime) * 100),
            'excess_return': float(excess * 100),
            't_stat': float(t_stat),
            'p_value': float(p_val_one_sided),
            'significant': p_val_one_sided < 0.05,
            'avg_volatility': float(np.mean(daily_volatility[mask])),
        }

    # Check if skill is concentrated in one regime
    excesses = [results[r]['excess_return'] for r in results if r in results]
    regime_consistency = "consistent" if all(e > 0 for e in excesses) or all(e < 0 for e in excesses) else "regime_dependent"

    results['summary'] = {
        'regime_consistency': regime_consistency,
        'best_regime': max(results.keys() - {'summary'}, key=lambda r: results[r]['excess_return']) if results else None,
        'worst_regime': min(results.keys() - {'summary'}, key=lambda r: results[r]['excess_return']) if results else None,
    }

    return results


@dataclass
class FoldEvaluation:
    """Comprehensive evaluation results for a single fold."""
    fold_idx: int
    test_start: str
    test_end: str
    num_test_days: int

    # Information Coefficient metrics
    mean_ic: float  # Mean daily IC
    std_ic: float   # Std of daily IC
    ir: float       # Information Ratio = mean_ic / std_ic
    ic_t_stat: float  # T-statistic for IC > 0
    ic_p_value: float  # P-value for IC > 0
    pct_positive_ic: float  # % of days with positive IC

    # Rank correlation
    mean_rank_ic: float  # Spearman rank correlation

    # Quantile analysis
    top_decile_return: float  # Mean return of top 10% predictions
    bottom_decile_return: float  # Mean return of bottom 10%
    long_short_spread: float  # Top - Bottom (the edge)
    hit_rate_top_decile: float  # % of top decile that beat median

    # Baseline comparisons
    model_mean_return: float  # Mean return of top-k selection
    momentum_mean_return: float  # Momentum baseline
    random_mean_return: float  # Random selection baseline
    excess_vs_momentum: float  # Model - Momentum
    excess_vs_random: float  # Model - Random

    # Statistical significance vs baselines
    vs_random_t_stat: float
    vs_random_p_value: float
    vs_momentum_t_stat: float
    vs_momentum_p_value: float

    # Trading simulation (secondary metric)
    simulation_total_return: float
    simulation_sharpe: float
    simulation_max_drawdown: float

    # Analytical expected return (Fundamental Law of Active Management)
    analytical_expected_annual_return: float  # IC * sigma * sqrt(breadth)
    cross_sectional_volatility: float  # σ of returns across stocks

    # Bootstrap confidence intervals (from resampling daily returns)
    bootstrap_mean_return: float
    bootstrap_std_return: float
    bootstrap_ci_lower: float  # 5th percentile
    bootstrap_ci_upper: float  # 95th percentile
    bootstrap_prob_positive: float  # P(return > 0)
    bootstrap_var_5pct: float  # 5% Value at Risk
    bootstrap_cvar_5pct: float  # Conditional VaR (Expected Shortfall)

    # === ENHANCED STATISTICS (optional, may be None) ===

    # Block bootstrap (preserves autocorrelation)
    block_bootstrap_mean: float = None
    block_bootstrap_ci_lower: float = None
    block_bootstrap_ci_upper: float = None
    block_bootstrap_prob_positive: float = None

    # Permutation test (non-parametric)
    perm_vs_random_p_value: float = None
    perm_vs_momentum_p_value: float = None

    # Effect size (practical significance)
    cohens_d_vs_random: float = None
    cohens_d_vs_momentum: float = None
    effect_size_interpretation: str = None

    # Regime analysis (per-regime performance)
    low_vol_excess: float = None
    medium_vol_excess: float = None
    high_vol_excess: float = None
    regime_consistency: str = None


class PrincipledEvaluator:
    """Evaluate model predictions using principled financial metrics."""

    def __init__(
        self,
        dataset_path: str,
        prices_path: str,
        seq_len: int = 60,
        horizon_days: int = 5,
        horizon_idx: int = 1,
        top_k: int = 10,
        device: str = 'cuda',
        num_quantiles: int = 10,
    ):
        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.seq_len = seq_len
        self.horizon_days = horizon_days
        self.horizon_idx = horizon_idx
        self.top_k = top_k
        self.device = device
        self.num_quantiles = num_quantiles

        # Load dataset info
        self._load_dataset_info()

    def _load_dataset_info(self):
        """Load dataset metadata."""
        print(f"Loading dataset info from {self.dataset_path}...")

        with h5py.File(self.dataset_path, 'r') as f:
            self.tickers = sorted(list(f.keys()))
            sample_ticker = self.tickers[0]
            dates_bytes = f[sample_ticker]['dates'][:]
            self.all_dates = sorted([d.decode('utf-8') for d in dates_bytes])

            # Get input dimension
            features = f[sample_ticker]['features']
            self.input_dim = features.shape[1]

        print(f"  {len(self.tickers)} tickers, {len(self.all_dates)} dates")
        print(f"  Input dim: {self.input_dim}")

        # Load prices
        if self.prices_path and os.path.exists(self.prices_path):
            self.prices_file = h5py.File(self.prices_path, 'r')
            print(f"  Prices loaded from {self.prices_path}")
        else:
            self.prices_file = None
            print(f"  No prices file, using features[0]")

    def _get_price(self, ticker: str, date: str) -> Optional[float]:
        """Get price for ticker on date."""
        if self.prices_file is not None and ticker in self.prices_file:
            try:
                dates_bytes = self.prices_file[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                idx = dates.index(date)
                return float(self.prices_file[ticker]['prices'][idx])
            except (ValueError, IndexError):
                return None
        return None

    def _get_future_return(self, ticker: str, date: str) -> Optional[float]:
        """Get horizon-day return for ticker starting from date."""
        return self._get_future_return_n_days(ticker, date, self.horizon_days)

    def _get_future_return_n_days(self, ticker: str, date: str, n_days: int) -> Optional[float]:
        """Get n-day return for ticker starting from date."""
        if self.prices_file is not None and ticker in self.prices_file:
            try:
                dates_bytes = self.prices_file[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                idx = dates.index(date)
                future_idx = idx + n_days
                if future_idx >= len(dates):
                    return None
                current_price = float(self.prices_file[ticker]['prices'][idx])
                future_price = float(self.prices_file[ticker]['prices'][future_idx])
                if current_price > 0:
                    return (future_price / current_price) - 1.0
            except (ValueError, IndexError):
                return None
        return None

    def _get_momentum(self, ticker: str, date: str, lookback: int = 20) -> Optional[float]:
        """Get past return as momentum signal."""
        if self.prices_file is not None and ticker in self.prices_file:
            try:
                dates_bytes = self.prices_file[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                idx = dates.index(date)
                past_idx = idx - lookback
                if past_idx < 0:
                    return None
                current_price = float(self.prices_file[ticker]['prices'][idx])
                past_price = float(self.prices_file[ticker]['prices'][past_idx])
                if past_price > 0:
                    return (current_price / past_price) - 1.0
            except (ValueError, IndexError):
                return None
        return None

    def load_model(self, checkpoint_path: str) -> Tuple[SimpleTransformerPredictor, dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        config = checkpoint['config']

        model = SimpleTransformerPredictor(
            input_dim=self.input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            num_pred_days=4,
            pred_mode=config.get('pred_mode', 'classification')
        )

        # Handle compiled model state dict
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        # Verify model is on correct device
        param_device = next(model.parameters()).device
        print(f"    Model loaded to device: {param_device}")

        # Load bin edges if classification
        bin_edges = checkpoint.get('bin_edges', None)
        if bin_edges is not None:
            bin_edges = bin_edges.to(self.device)
            print(f"    Bin edges on device: {bin_edges.device}")

        return model, checkpoint

    @torch.no_grad()
    def predict_batch(
        self,
        model: SimpleTransformerPredictor,
        features_list: List[torch.Tensor],
        bin_edges: Optional[torch.Tensor] = None
    ) -> List[float]:
        """Get expected return predictions for a batch of features."""
        all_horizon_preds = self.predict_batch_all_horizons(model, features_list, bin_edges)
        if all_horizon_preds is None:
            return [0.0] * len(features_list)
        return all_horizon_preds[:, self.horizon_idx].tolist()

    @torch.no_grad()
    def predict_batch_all_horizons(
        self,
        model: SimpleTransformerPredictor,
        features_list: List[torch.Tensor],
        bin_edges: Optional[torch.Tensor] = None,
        batch_size: int = 512,
    ) -> Optional[np.ndarray]:
        """
        Get expected return predictions for ALL horizons at once.

        Efficiently batches data to GPU, runs inference, and returns results.

        Args:
            model: The model to use for prediction
            features_list: List of feature tensors (on CPU)
            bin_edges: Bin edges for classification mode
            batch_size: GPU batch size for inference

        Returns:
            numpy array of shape (num_stocks, 4) with predictions for each horizon,
            or None if no valid features.
        """
        if len(features_list) == 0:
            return None

        # Filter to only features matching expected shape (seq_len, input_dim)
        valid_features = []
        valid_indices = []
        for i, f in enumerate(features_list):
            if len(f.shape) == 2 and f.shape[0] == self.seq_len and f.shape[1] == self.input_dim:
                valid_features.append(f)
                valid_indices.append(i)

        if len(valid_features) == 0:
            return None

        n_samples = len(valid_features)

        # Pre-allocate output on GPU, process in batches
        all_expected_returns_list = []

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_features = valid_features[batch_start:batch_end]

            # Stack and move to GPU in one operation
            # Each feature is (seq_len, input_dim), stack to get (batch, seq_len, input_dim)
            features_batch = torch.stack(batch_features).to(self.device, non_blocking=True)

            # Forward pass on GPU
            pred, confidence = model(features_batch)

            pred_mode = 'classification' if pred.shape[1] == 100 else 'regression'

            # Compute expected returns for ALL horizons (stay on GPU)
            if pred_mode == 'classification' and bin_edges is not None:
                # pred shape: (batch, num_bins, 4)
                probs = F.softmax(pred, dim=1)  # (batch, num_bins, 4)
                bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # (num_bins,)
                # Expand for broadcasting: (1, num_bins, 1)
                bin_midpoints = bin_midpoints.view(1, -1, 1)
                # Expected value: sum over bins
                batch_expected = (probs * bin_midpoints).sum(dim=1)  # (batch, 4)
            else:
                # Regression mode: pred is already (batch, 4)
                batch_expected = pred

            # Move batch result to CPU
            all_expected_returns_list.append(batch_expected.cpu().numpy())

        # Concatenate all batches
        all_expected_returns = np.concatenate(all_expected_returns_list, axis=0)  # (n_samples, 4)

        # Map back to original indices if needed
        if len(valid_indices) < len(features_list):
            full_returns = np.zeros((len(features_list), 4), dtype=np.float32)
            for i, idx in enumerate(valid_indices):
                full_returns[idx] = all_expected_returns[i]
            return full_returns

        return all_expected_returns

    def evaluate_fold(
        self,
        model: SimpleTransformerPredictor,
        checkpoint: dict,
        test_dates: List[str],
        bin_edges: Optional[torch.Tensor] = None,
        batch_size: int = 256,
        max_stocks_per_day: int = 500,
    ) -> FoldEvaluation:
        """Evaluate model on a single fold's test period."""

        fold_idx = checkpoint['fold_idx']

        # Storage for daily metrics (with dates for time-series analysis)
        daily_ics = []
        daily_rank_ics = []
        daily_top_decile_returns = []
        daily_bottom_decile_returns = []
        daily_model_returns = []  # Top-k selection
        daily_momentum_returns = []  # Momentum baseline
        daily_random_returns = []  # Random baseline
        daily_dates = []  # Track dates for time-series plotting

        # For simulation
        capital = 100000.0
        capital_history = [capital]

        # ========================================
        # PRE-LOAD ALL DATA INTO RAM (like sweep mode)
        # ========================================
        print(f"  Pre-loading features into RAM...")
        features_cache = {}  # ticker -> (dates_list, features_2d)
        with h5py.File(self.dataset_path, 'r') as h5f:
            for ticker in tqdm(self.tickers[:max_stocks_per_day], desc="    Loading", leave=False):
                if ticker not in h5f:
                    continue
                try:
                    dates_bytes = h5f[ticker]['dates'][:]
                    dates = [d.decode('utf-8') for d in dates_bytes]
                    features = h5f[ticker]['features'][:].astype(np.float32)
                    if features.shape[1] == self.input_dim:
                        features_cache[ticker] = (dates, features)
                except Exception:
                    continue

        # Pre-load prices into RAM
        prices_cache = {}  # ticker -> (dates_list, prices_array)
        if self.prices_file is not None:
            for ticker in features_cache.keys():
                if ticker not in self.prices_file:
                    continue
                try:
                    dates_bytes = self.prices_file[ticker]['dates'][:]
                    dates = [d.decode('utf-8') for d in dates_bytes]
                    prices = self.prices_file[ticker]['prices'][:].astype(np.float32)
                    # Skip stocks with unreasonable prices
                    if prices.max() <= 10000:
                        prices_cache[ticker] = (dates, prices)
                except Exception:
                    continue

        print(f"  Loaded {len(features_cache)} tickers, evaluating {len(test_dates)} test days...")

        # Build date index for fast lookup
        date_to_idx = {d: i for i, d in enumerate(self.all_dates)}

        # Use non-overlapping dates for evaluation
        eval_dates = test_dates[::self.horizon_days]

        for date in tqdm(eval_dates, desc=f"    Fold {fold_idx}", leave=False):
            # Collect features, actual returns, and momentum for all stocks
            features_list = []
            actual_returns = []
            momentum_signals = []
            valid_tickers = []

            if date not in date_to_idx:
                continue
            date_idx = date_to_idx[date]
            if date_idx < self.seq_len:
                continue

            for ticker in features_cache.keys():
                feat_dates, feat_array = features_cache[ticker]

                # Find date index in this ticker's data
                try:
                    ticker_date_idx = feat_dates.index(date)
                except ValueError:
                    continue

                if ticker_date_idx >= feat_array.shape[0]:
                    continue

                # Need seq_len days of history ending at ticker_date_idx
                start_idx = ticker_date_idx - self.seq_len + 1
                if start_idx < 0:
                    continue

                # Get full sequence of features
                seq_features = feat_array[start_idx:ticker_date_idx + 1]  # Shape: (seq_len, features)
                features = torch.from_numpy(seq_features).float()

                # Get actual future return from cache
                if ticker in prices_cache:
                    price_dates, price_array = prices_cache[ticker]
                    try:
                        price_idx = price_dates.index(date)
                        future_idx = price_idx + self.horizon_days
                        if future_idx < len(price_array) and price_array[price_idx] > 0:
                            actual_ret = (price_array[future_idx] / price_array[price_idx]) - 1.0
                        else:
                            continue
                    except (ValueError, IndexError):
                        continue

                    # Get momentum from cache
                    past_idx = price_idx - 20
                    if past_idx >= 0 and price_array[past_idx] > 0:
                        momentum = (price_array[price_idx] / price_array[past_idx]) - 1.0
                    else:
                        momentum = 0.0
                else:
                    continue

                features_list.append(features)
                actual_returns.append(actual_ret)
                momentum_signals.append(momentum)
                valid_tickers.append(ticker)

            # Check if we have enough stocks for this date (outside ticker loop)
            if len(features_list) < self.top_k * 2:
                continue

            # Get model predictions
            predictions = []
            for i in range(0, len(features_list), batch_size):
                batch = features_list[i:i+batch_size]
                preds = self.predict_batch(model, batch, bin_edges)
                predictions.extend(preds)

            predictions = np.array(predictions)
            actual_returns = np.array(actual_returns)
            momentum_signals = np.array(momentum_signals)

            # === Information Coefficient ===
            # Pearson correlation between predictions and actual returns
            if len(predictions) > 10 and np.std(predictions) > 1e-8 and np.std(actual_returns) > 1e-8:
                ic, _ = pearsonr(predictions, actual_returns)
                rank_ic, _ = spearmanr(predictions, actual_returns)
                if not np.isnan(ic):
                    daily_ics.append(ic)
                    daily_dates.append(date)  # Track date for this IC
                if not np.isnan(rank_ic):
                    daily_rank_ics.append(rank_ic)

            # === Quantile Analysis ===
            # Sort by prediction, compute returns by decile
            sorted_indices = np.argsort(predictions)[::-1]  # High to low
            n_per_decile = len(sorted_indices) // self.num_quantiles

            if n_per_decile > 0:
                top_decile_idx = sorted_indices[:n_per_decile]
                bottom_decile_idx = sorted_indices[-n_per_decile:]

                top_decile_return = actual_returns[top_decile_idx].mean()
                bottom_decile_return = actual_returns[bottom_decile_idx].mean()

                daily_top_decile_returns.append(top_decile_return)
                daily_bottom_decile_returns.append(bottom_decile_return)

            # === Model Selection (Top-K) ===
            top_k_idx = sorted_indices[:self.top_k]
            model_return = actual_returns[top_k_idx].mean()
            daily_model_returns.append(model_return)

            # === Momentum Baseline ===
            momentum_sorted = np.argsort(momentum_signals)[::-1]
            momentum_top_k = momentum_sorted[:self.top_k]
            momentum_return = actual_returns[momentum_top_k].mean()
            daily_momentum_returns.append(momentum_return)

            # === Random Baseline ===
            random_idx = np.random.choice(len(actual_returns), self.top_k, replace=False)
            random_return = actual_returns[random_idx].mean()
            daily_random_returns.append(random_return)

            # === Simulation Update ===
            capital *= (1 + model_return)
            capital_history.append(capital)

        # Compute aggregate metrics
        daily_ics = np.array(daily_ics)
        daily_rank_ics = np.array(daily_rank_ics)
        daily_model_returns = np.array(daily_model_returns)
        daily_momentum_returns = np.array(daily_momentum_returns)
        daily_random_returns = np.array(daily_random_returns)
        daily_top_decile_returns = np.array(daily_top_decile_returns)
        daily_bottom_decile_returns = np.array(daily_bottom_decile_returns)

        # IC statistics
        mean_ic = np.mean(daily_ics) if len(daily_ics) > 0 else 0
        std_ic = np.std(daily_ics) if len(daily_ics) > 0 else 1
        ir = mean_ic / std_ic if std_ic > 1e-8 else 0

        # T-test for IC > 0
        if len(daily_ics) > 1:
            ic_t_stat, ic_p_value = stats.ttest_1samp(daily_ics, 0)
            ic_p_value = ic_p_value / 2  # One-sided test
        else:
            ic_t_stat, ic_p_value = 0, 1

        pct_positive_ic = np.mean(daily_ics > 0) * 100 if len(daily_ics) > 0 else 0

        # Rank IC
        mean_rank_ic = np.mean(daily_rank_ics) if len(daily_rank_ics) > 0 else 0

        # Quantile analysis
        top_decile_return = np.mean(daily_top_decile_returns) * 100 if len(daily_top_decile_returns) > 0 else 0
        bottom_decile_return = np.mean(daily_bottom_decile_returns) * 100 if len(daily_bottom_decile_returns) > 0 else 0
        long_short_spread = top_decile_return - bottom_decile_return

        # Hit rate: % of top decile stocks that beat median
        # (approximated by checking if top decile beats average)
        hit_rate = np.mean(daily_top_decile_returns > np.mean(daily_model_returns)) * 100 if len(daily_top_decile_returns) > 0 else 0

        # Mean returns
        model_mean_return = np.mean(daily_model_returns) * 100 if len(daily_model_returns) > 0 else 0
        momentum_mean_return = np.mean(daily_momentum_returns) * 100 if len(daily_momentum_returns) > 0 else 0
        random_mean_return = np.mean(daily_random_returns) * 100 if len(daily_random_returns) > 0 else 0

        # Excess returns
        excess_vs_momentum = model_mean_return - momentum_mean_return
        excess_vs_random = model_mean_return - random_mean_return

        # Statistical significance vs baselines (paired t-test)
        if len(daily_model_returns) > 1:
            vs_random_t_stat, vs_random_p_value = stats.ttest_rel(daily_model_returns, daily_random_returns)
            vs_random_p_value = vs_random_p_value / 2  # One-sided
            vs_momentum_t_stat, vs_momentum_p_value = stats.ttest_rel(daily_model_returns, daily_momentum_returns)
            vs_momentum_p_value = vs_momentum_p_value / 2
        else:
            vs_random_t_stat, vs_random_p_value = 0, 1
            vs_momentum_t_stat, vs_momentum_p_value = 0, 1

        # Simulation metrics
        capital_history = np.array(capital_history)
        simulation_total_return = (capital_history[-1] / capital_history[0] - 1) * 100

        returns = np.diff(capital_history) / capital_history[:-1]
        trades_per_year = 252 / self.horizon_days
        if len(returns) > 1 and np.std(returns) > 1e-8:
            simulation_sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
        else:
            simulation_sharpe = 0

        peak = np.maximum.accumulate(capital_history)
        drawdown = (capital_history - peak) / peak
        simulation_max_drawdown = np.min(drawdown) * 100

        # === ANALYTICAL EXPECTED RETURN (Fundamental Law of Active Management) ===
        # E[excess_return] ≈ IC × σ_cross × √(breadth) × transfer_coefficient
        # For top-k selection, transfer coefficient ≈ sqrt(2/π) for one-sided selection
        cross_sectional_vol = np.std(daily_model_returns) if len(daily_model_returns) > 0 else 0
        breadth = trades_per_year  # Number of independent bets per year
        transfer_coef = np.sqrt(2 / np.pi)  # For long-only top-k selection

        # Annualized expected return from IC
        if mean_ic > 0 and cross_sectional_vol > 0:
            analytical_expected_annual_return = (
                mean_ic * cross_sectional_vol * np.sqrt(breadth) * transfer_coef * 100
            )
        else:
            analytical_expected_annual_return = 0

        # === BOOTSTRAP CONFIDENCE INTERVALS ===
        # Resample daily returns to build distribution of outcomes
        n_bootstrap = 1000
        bootstrap_total_returns = []

        if len(daily_model_returns) > 5:
            for _ in range(n_bootstrap):
                # Resample with replacement
                resampled = np.random.choice(daily_model_returns, size=len(daily_model_returns), replace=True)
                # Compound returns
                bootstrap_capital = 100000.0
                for ret in resampled:
                    bootstrap_capital *= (1 + ret)
                bootstrap_total_returns.append((bootstrap_capital / 100000.0 - 1) * 100)

            bootstrap_total_returns = np.array(bootstrap_total_returns)
            bootstrap_mean_return = np.mean(bootstrap_total_returns)
            bootstrap_std_return = np.std(bootstrap_total_returns)
            bootstrap_ci_lower = np.percentile(bootstrap_total_returns, 5)
            bootstrap_ci_upper = np.percentile(bootstrap_total_returns, 95)
            bootstrap_prob_positive = np.mean(bootstrap_total_returns > 0)

            # Value at Risk and Conditional VaR (Expected Shortfall)
            bootstrap_var_5pct = np.percentile(bootstrap_total_returns, 5)
            worst_5pct = bootstrap_total_returns[bootstrap_total_returns <= bootstrap_var_5pct]
            bootstrap_cvar_5pct = np.mean(worst_5pct) if len(worst_5pct) > 0 else bootstrap_var_5pct
        else:
            bootstrap_mean_return = simulation_total_return
            bootstrap_std_return = 0
            bootstrap_ci_lower = simulation_total_return
            bootstrap_ci_upper = simulation_total_return
            bootstrap_prob_positive = 1.0 if simulation_total_return > 0 else 0.0
            bootstrap_var_5pct = simulation_total_return
            bootstrap_cvar_5pct = simulation_total_return

        # === ENHANCED STATISTICS ===
        # These provide more robust significance assessment

        # 1. Block bootstrap (preserves time-series autocorrelation)
        block_bootstrap_results = None
        if len(daily_model_returns) > 10:
            block_size = min(5, len(daily_model_returns) // 4)
            block_bootstrap_results = block_bootstrap(
                daily_model_returns,
                n_bootstrap=2000,  # More samples for better tail estimates
                block_size=block_size
            )

        # 2. Permutation tests (non-parametric alternative to t-tests)
        perm_vs_random = None
        perm_vs_momentum = None
        if len(daily_model_returns) > 10:
            perm_vs_random = permutation_test(
                daily_model_returns, daily_random_returns,
                n_permutations=5000
            )
            perm_vs_momentum = permutation_test(
                daily_model_returns, daily_momentum_returns,
                n_permutations=5000
            )

        # 3. Effect size (practical significance)
        effect_vs_random = compute_effect_size(daily_model_returns, daily_random_returns)
        effect_vs_momentum = compute_effect_size(daily_model_returns, daily_momentum_returns)

        # 4. Regime stratification (does model work in all market conditions?)
        regime_results = None
        if len(daily_model_returns) > 20:
            regime_results = regime_stratified_analysis(
                daily_model_returns, daily_random_returns
            )

        fold_eval = FoldEvaluation(
            fold_idx=fold_idx,
            test_start=test_dates[0],
            test_end=test_dates[-1],
            num_test_days=len(test_dates),
            mean_ic=mean_ic,
            std_ic=std_ic,
            ir=ir,
            ic_t_stat=ic_t_stat,
            ic_p_value=ic_p_value,
            pct_positive_ic=pct_positive_ic,
            mean_rank_ic=mean_rank_ic,
            top_decile_return=top_decile_return,
            bottom_decile_return=bottom_decile_return,
            long_short_spread=long_short_spread,
            hit_rate_top_decile=hit_rate,
            model_mean_return=model_mean_return,
            momentum_mean_return=momentum_mean_return,
            random_mean_return=random_mean_return,
            excess_vs_momentum=excess_vs_momentum,
            excess_vs_random=excess_vs_random,
            vs_random_t_stat=vs_random_t_stat,
            vs_random_p_value=vs_random_p_value,
            vs_momentum_t_stat=vs_momentum_t_stat,
            vs_momentum_p_value=vs_momentum_p_value,
            simulation_total_return=simulation_total_return,
            simulation_sharpe=simulation_sharpe,
            simulation_max_drawdown=simulation_max_drawdown,
            # Analytical estimate
            analytical_expected_annual_return=analytical_expected_annual_return,
            cross_sectional_volatility=cross_sectional_vol * 100,
            # Bootstrap results
            bootstrap_mean_return=bootstrap_mean_return,
            bootstrap_std_return=bootstrap_std_return,
            bootstrap_ci_lower=bootstrap_ci_lower,
            bootstrap_ci_upper=bootstrap_ci_upper,
            bootstrap_prob_positive=bootstrap_prob_positive * 100,
            bootstrap_var_5pct=bootstrap_var_5pct,
            bootstrap_cvar_5pct=bootstrap_cvar_5pct,
            # Enhanced statistics
            block_bootstrap_mean=block_bootstrap_results['mean'] if block_bootstrap_results else None,
            block_bootstrap_ci_lower=block_bootstrap_results['ci_lower'] if block_bootstrap_results else None,
            block_bootstrap_ci_upper=block_bootstrap_results['ci_upper'] if block_bootstrap_results else None,
            block_bootstrap_prob_positive=block_bootstrap_results['prob_positive'] * 100 if block_bootstrap_results else None,
            perm_vs_random_p_value=perm_vs_random['p_value_one_sided'] if perm_vs_random else None,
            perm_vs_momentum_p_value=perm_vs_momentum['p_value_one_sided'] if perm_vs_momentum else None,
            cohens_d_vs_random=effect_vs_random['cohens_d'],
            cohens_d_vs_momentum=effect_vs_momentum['cohens_d'],
            effect_size_interpretation=effect_vs_random['interpretation'],
            low_vol_excess=regime_results.get('low_vol', {}).get('excess_return') if regime_results else None,
            medium_vol_excess=regime_results.get('medium_vol', {}).get('excess_return') if regime_results else None,
            high_vol_excess=regime_results.get('high_vol', {}).get('excess_return') if regime_results else None,
            regime_consistency=regime_results.get('summary', {}).get('regime_consistency') if regime_results else None,
        )

        # Return time-series data for plotting
        time_series_data = {
            'dates': daily_dates,
            'daily_ics': daily_ics.tolist() if isinstance(daily_ics, np.ndarray) else daily_ics,
            'daily_model_returns': daily_model_returns.tolist() if isinstance(daily_model_returns, np.ndarray) else daily_model_returns,
            'daily_random_returns': daily_random_returns.tolist() if isinstance(daily_random_returns, np.ndarray) else daily_random_returns,
            'daily_momentum_returns': daily_momentum_returns.tolist() if isinstance(daily_momentum_returns, np.ndarray) else daily_momentum_returns,
        }

        return fold_eval, time_series_data


def plot_ic_time_series(all_time_series: List[dict], output_path: str = None):
    """Plot IC and returns over time across all folds."""
    # Combine all folds
    all_dates = []
    all_ics = []
    all_model_returns = []
    all_random_returns = []

    for ts_data in all_time_series:
        all_dates.extend(ts_data['dates'])
        all_ics.extend(ts_data['daily_ics'])
        all_model_returns.extend(ts_data['daily_model_returns'])
        all_random_returns.extend(ts_data['daily_random_returns'])

    if len(all_dates) == 0:
        print("No data to plot")
        return

    # Sort by date
    sorted_idx = np.argsort(all_dates)
    dates = [all_dates[i] for i in sorted_idx]
    ics = np.array([all_ics[i] for i in sorted_idx])
    model_rets = np.array([all_model_returns[i] for i in sorted_idx])
    random_rets = np.array([all_random_returns[i] for i in sorted_idx])

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Daily IC with rolling mean
    ax1 = axes[0]
    ax1.bar(range(len(ics)), ics, alpha=0.4, color=['green' if ic > 0 else 'red' for ic in ics], width=1.0)
    # Rolling 20-period IC
    if len(ics) >= 20:
        rolling_ic = np.convolve(ics, np.ones(20)/20, mode='valid')
        ax1.plot(range(10, 10+len(rolling_ic)), rolling_ic, 'b-', linewidth=2, label='20-period rolling IC')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=np.mean(ics), color='blue', linestyle='--', linewidth=1, label=f'Mean IC: {np.mean(ics):.4f}')
    ax1.set_ylabel('Information Coefficient')
    ax1.set_title(f'Daily IC Over Time (Mean: {np.mean(ics):.4f}, Positive: {100*np.mean(ics > 0):.1f}%)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative IC (sum of daily ICs)
    ax2 = axes[1]
    cumulative_ic = np.cumsum(ics)
    ax2.plot(range(len(cumulative_ic)), cumulative_ic, 'b-', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(range(len(cumulative_ic)), cumulative_ic, 0,
                     where=cumulative_ic >= 0, alpha=0.3, color='green')
    ax2.fill_between(range(len(cumulative_ic)), cumulative_ic, 0,
                     where=cumulative_ic < 0, alpha=0.3, color='red')
    ax2.set_ylabel('Cumulative IC')
    ax2.set_title('Cumulative IC (Rising = Consistent Predictive Power)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative excess returns (model vs random)
    ax3 = axes[2]
    excess_returns = model_rets - random_rets
    cumulative_excess = np.cumsum(excess_returns) * 100  # Convert to percentage
    ax3.plot(range(len(cumulative_excess)), cumulative_excess, 'purple', linewidth=2, label='Model - Random')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.fill_between(range(len(cumulative_excess)), cumulative_excess, 0,
                     where=cumulative_excess >= 0, alpha=0.3, color='green')
    ax3.fill_between(range(len(cumulative_excess)), cumulative_excess, 0,
                     where=cumulative_excess < 0, alpha=0.3, color='red')
    ax3.set_ylabel('Cumulative Excess Return (%)')
    ax3.set_title(f'Cumulative Excess Return vs Random (Total: {cumulative_excess[-1]:.2f}%)')
    ax3.grid(True, alpha=0.3)

    # X-axis labels (show subset of dates)
    n_labels = min(10, len(dates))
    label_indices = np.linspace(0, len(dates)-1, n_labels, dtype=int)
    ax3.set_xticks(label_indices)
    ax3.set_xticklabels([dates[i] for i in label_indices], rotation=45, ha='right')
    ax3.set_xlabel('Date')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved IC time-series plot to {output_path}")
    else:
        plt.show()

    plt.close()

    # Print summary statistics by time period
    print(f"\n📈 IC BY TIME PERIOD:")
    if len(dates) >= 4:
        quarter_size = len(ics) // 4
        for i, period in enumerate(['Q1 (oldest)', 'Q2', 'Q3', 'Q4 (newest)']):
            start = i * quarter_size
            end = (i + 1) * quarter_size if i < 3 else len(ics)
            period_ics = ics[start:end]
            print(f"  {period}: Mean IC = {np.mean(period_ics):+.4f}, Positive = {100*np.mean(period_ics > 0):.1f}%")


def print_fold_results(result: FoldEvaluation):
    """Print detailed results for a single fold."""
    print(f"\n{'='*70}")
    print(f"FOLD {result.fold_idx}: {result.test_start} to {result.test_end}")
    print(f"{'='*70}")

    print(f"\n📊 INFORMATION COEFFICIENT (Primary Metric)")
    print(f"  Mean IC:           {result.mean_ic:>8.4f}")
    print(f"  Std IC:            {result.std_ic:>8.4f}")
    print(f"  Information Ratio: {result.ir:>8.4f}")
    print(f"  T-statistic:       {result.ic_t_stat:>8.2f}")
    print(f"  P-value (IC > 0):  {result.ic_p_value:>8.4f} {'***' if result.ic_p_value < 0.01 else '**' if result.ic_p_value < 0.05 else '*' if result.ic_p_value < 0.1 else ''}")
    print(f"  % Positive IC:     {result.pct_positive_ic:>7.1f}%")
    print(f"  Mean Rank IC:      {result.mean_rank_ic:>8.4f}")

    print(f"\n📈 QUANTILE ANALYSIS")
    print(f"  Top Decile Return:    {result.top_decile_return:>+7.3f}%")
    print(f"  Bottom Decile Return: {result.bottom_decile_return:>+7.3f}%")
    print(f"  Long-Short Spread:    {result.long_short_spread:>+7.3f}% {'✓' if result.long_short_spread > 0 else '✗'}")
    print(f"  Hit Rate (Top Dec):   {result.hit_rate_top_decile:>7.1f}%")

    print(f"\n🎯 BASELINE COMPARISON (per-trade returns)")
    print(f"  Model (Top-K):     {result.model_mean_return:>+7.3f}%")
    print(f"  Momentum Baseline: {result.momentum_mean_return:>+7.3f}%")
    print(f"  Random Baseline:   {result.random_mean_return:>+7.3f}%")
    print(f"  Excess vs Momentum:{result.excess_vs_momentum:>+7.3f}% (p={result.vs_momentum_p_value:.3f})")
    print(f"  Excess vs Random:  {result.excess_vs_random:>+7.3f}% (p={result.vs_random_p_value:.3f})")

    print(f"\n💰 TRADING SIMULATION (Secondary)")
    print(f"  Total Return:  {result.simulation_total_return:>+8.2f}%")
    print(f"  Sharpe Ratio:  {result.simulation_sharpe:>8.2f}")
    print(f"  Max Drawdown:  {result.simulation_max_drawdown:>8.2f}%")

    print(f"\n📐 ANALYTICAL ESTIMATE (Fundamental Law)")
    print(f"  Cross-sectional σ:      {result.cross_sectional_volatility:>7.2f}%")
    print(f"  Expected Annual Return: {result.analytical_expected_annual_return:>+7.2f}%")

    print(f"\n🎲 BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)")
    print(f"  Mean Return:       {result.bootstrap_mean_return:>+8.2f}%")
    print(f"  Std Return:        {result.bootstrap_std_return:>8.2f}%")
    print(f"  90% CI:            [{result.bootstrap_ci_lower:>+7.2f}%, {result.bootstrap_ci_upper:>+7.2f}%]")
    print(f"  P(Return > 0):     {result.bootstrap_prob_positive:>7.1f}%")
    print(f"  5% VaR:            {result.bootstrap_var_5pct:>+8.2f}%")
    print(f"  5% CVaR (ES):      {result.bootstrap_cvar_5pct:>+8.2f}%")

    # === ENHANCED STATISTICS ===
    if result.block_bootstrap_mean is not None:
        print(f"\n🔗 BLOCK BOOTSTRAP (autocorrelation-aware, 2000 resamples)")
        print(f"  Mean Return:       {result.block_bootstrap_mean:>+8.2f}%")
        print(f"  90% CI:            [{result.block_bootstrap_ci_lower:>+7.2f}%, {result.block_bootstrap_ci_upper:>+7.2f}%]")
        print(f"  P(Return > 0):     {result.block_bootstrap_prob_positive:>7.1f}%")

    if result.perm_vs_random_p_value is not None:
        print(f"\n🔀 PERMUTATION TESTS (5000 permutations)")
        sig_rand = "***" if result.perm_vs_random_p_value < 0.01 else "**" if result.perm_vs_random_p_value < 0.05 else "*" if result.perm_vs_random_p_value < 0.1 else ""
        sig_mom = "***" if result.perm_vs_momentum_p_value < 0.01 else "**" if result.perm_vs_momentum_p_value < 0.05 else "*" if result.perm_vs_momentum_p_value < 0.1 else ""
        print(f"  vs Random:   p = {result.perm_vs_random_p_value:>7.4f} {sig_rand}")
        print(f"  vs Momentum: p = {result.perm_vs_momentum_p_value:>7.4f} {sig_mom}")

    if result.cohens_d_vs_random is not None:
        print(f"\n📏 EFFECT SIZE (practical significance)")
        print(f"  Cohen's d vs Random:   {result.cohens_d_vs_random:>+7.3f} ({result.effect_size_interpretation})")
        print(f"  Cohen's d vs Momentum: {result.cohens_d_vs_momentum:>+7.3f}")

    if result.regime_consistency is not None:
        print(f"\n🌡️ REGIME ANALYSIS (by volatility)")
        print(f"  Low-vol excess:    {result.low_vol_excess:>+7.3f}%" if result.low_vol_excess is not None else "  Low-vol excess:    N/A")
        print(f"  Medium-vol excess: {result.medium_vol_excess:>+7.3f}%" if result.medium_vol_excess is not None else "  Medium-vol excess: N/A")
        print(f"  High-vol excess:   {result.high_vol_excess:>+7.3f}%" if result.high_vol_excess is not None else "  High-vol excess:   N/A")
        print(f"  Consistency:       {result.regime_consistency}")


def print_pooled_analysis(all_time_series: List[dict]):
    """
    Print pooled statistical analysis combining all folds.

    This provides more power than per-fold tests by using all observations.
    """
    # Combine all daily returns across folds
    all_model_returns = []
    all_random_returns = []
    all_momentum_returns = []

    for ts in all_time_series:
        all_model_returns.extend(ts['daily_model_returns'])
        all_random_returns.extend(ts['daily_random_returns'])
        all_momentum_returns.extend(ts['daily_momentum_returns'])

    if len(all_model_returns) < 10:
        return

    all_model_returns = np.array(all_model_returns)
    all_random_returns = np.array(all_random_returns)
    all_momentum_returns = np.array(all_momentum_returns)

    print(f"\n{'='*70}")
    print(f"POOLED ANALYSIS ({len(all_model_returns)} observations across all folds)")
    print(f"{'='*70}")

    # Pooled significance vs random
    pooled_vs_random = pooled_significance_test(
        [all_model_returns],
        [all_random_returns]
    )
    pooled_vs_momentum = pooled_significance_test(
        [all_model_returns],
        [all_momentum_returns]
    )

    print(f"\n📊 POOLED T-TEST (paired, all observations)")
    sig_rand = "***" if pooled_vs_random['p_value_one_sided'] < 0.01 else "**" if pooled_vs_random['p_value_one_sided'] < 0.05 else "*" if pooled_vs_random['p_value_one_sided'] < 0.1 else ""
    sig_mom = "***" if pooled_vs_momentum['p_value_one_sided'] < 0.01 else "**" if pooled_vs_momentum['p_value_one_sided'] < 0.05 else "*" if pooled_vs_momentum['p_value_one_sided'] < 0.1 else ""

    print(f"  vs Random:")
    print(f"    Mean excess:     {pooled_vs_random['pooled_diff_mean']:>+7.4f}%")
    print(f"    t-stat:          {pooled_vs_random['t_stat']:>8.2f}")
    print(f"    p-value:         {pooled_vs_random['p_value_one_sided']:>8.4f} {sig_rand}")
    print(f"    Wilcoxon p:      {pooled_vs_random['wilcoxon_p_one_sided']:>8.4f}")
    print(f"    Cohen's d:       {pooled_vs_random['cohens_d']:>+8.3f} ({pooled_vs_random['effect_interpretation']})")

    print(f"  vs Momentum:")
    print(f"    Mean excess:     {pooled_vs_momentum['pooled_diff_mean']:>+7.4f}%")
    print(f"    t-stat:          {pooled_vs_momentum['t_stat']:>8.2f}")
    print(f"    p-value:         {pooled_vs_momentum['p_value_one_sided']:>8.4f} {sig_mom}")
    print(f"    Wilcoxon p:      {pooled_vs_momentum['wilcoxon_p_one_sided']:>8.4f}")
    print(f"    Cohen's d:       {pooled_vs_momentum['cohens_d']:>+8.3f} ({pooled_vs_momentum['effect_interpretation']})")

    # Pooled permutation test
    print(f"\n🔀 POOLED PERMUTATION TEST (10000 permutations)")
    perm_vs_random = permutation_test(all_model_returns, all_random_returns, n_permutations=10000)
    perm_vs_momentum = permutation_test(all_model_returns, all_momentum_returns, n_permutations=10000)

    sig_perm_rand = "***" if perm_vs_random['p_value_one_sided'] < 0.01 else "**" if perm_vs_random['p_value_one_sided'] < 0.05 else "*" if perm_vs_random['p_value_one_sided'] < 0.1 else ""
    sig_perm_mom = "***" if perm_vs_momentum['p_value_one_sided'] < 0.01 else "**" if perm_vs_momentum['p_value_one_sided'] < 0.05 else "*" if perm_vs_momentum['p_value_one_sided'] < 0.1 else ""

    print(f"  vs Random:   p = {perm_vs_random['p_value_one_sided']:.4f} {sig_perm_rand}")
    print(f"  vs Momentum: p = {perm_vs_momentum['p_value_one_sided']:.4f} {sig_perm_mom}")

    # Block bootstrap on pooled data
    print(f"\n🔗 POOLED BLOCK BOOTSTRAP (2000 resamples, block_size=10)")
    pooled_block_boot = block_bootstrap(all_model_returns - all_random_returns, n_bootstrap=2000, block_size=10)
    print(f"  Mean excess return:  {pooled_block_boot['mean']:>+7.3f}%")
    print(f"  90% CI:              [{pooled_block_boot['ci_lower']:>+7.3f}%, {pooled_block_boot['ci_upper']:>+7.3f}%]")
    print(f"  P(Excess > 0):       {pooled_block_boot['prob_positive']*100:>7.1f}%")

    # 99% CI to be more conservative
    print(f"  99% CI:              [{pooled_block_boot['ci_99_lower']:>+7.3f}%, {pooled_block_boot['ci_99_upper']:>+7.3f}%]")

    ci_excludes_zero = (pooled_block_boot['ci_lower'] > 0) or (pooled_block_boot['ci_upper'] < 0)
    print(f"  90% CI excludes 0:   {'Yes ✓' if ci_excludes_zero else 'No'}")


def print_aggregate_results(results: List[FoldEvaluation]):
    """Print aggregate results across all folds."""
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS ({len(results)} folds)")
    print(f"{'='*70}")

    # IC metrics
    mean_ics = [r.mean_ic for r in results]
    irs = [r.ir for r in results]

    print(f"\n📊 INFORMATION COEFFICIENT")
    print(f"  Mean IC:  {np.mean(mean_ics):>8.4f} (±{np.std(mean_ics):.4f})")
    print(f"  Mean IR:  {np.mean(irs):>8.4f} (±{np.std(irs):.4f})")

    # Test if aggregate IC is significant
    if len(mean_ics) > 1:
        t_stat, p_value = stats.ttest_1samp(mean_ics, 0)
        p_value = p_value / 2
        print(f"  Aggregate T-stat: {t_stat:>8.2f}")
        print(f"  Aggregate P-value: {p_value:>8.4f} {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''}")

    # Quantile metrics
    spreads = [r.long_short_spread for r in results]
    print(f"\n📈 LONG-SHORT SPREAD")
    print(f"  Mean Spread: {np.mean(spreads):>+7.3f}% (±{np.std(spreads):.3f}%)")
    print(f"  % Positive:  {np.mean([s > 0 for s in spreads])*100:>7.1f}%")

    # Baseline comparison
    excess_momentum = [r.excess_vs_momentum for r in results]
    excess_random = [r.excess_vs_random for r in results]

    print(f"\n🎯 EXCESS RETURNS")
    print(f"  vs Momentum: {np.mean(excess_momentum):>+7.3f}% (±{np.std(excess_momentum):.3f}%)")
    print(f"  vs Random:   {np.mean(excess_random):>+7.3f}% (±{np.std(excess_random):.3f}%)")

    # Count significant folds
    sig_vs_random = sum(1 for r in results if r.vs_random_p_value < 0.05)
    sig_vs_momentum = sum(1 for r in results if r.vs_momentum_p_value < 0.05)
    print(f"  Folds sig. vs Random (p<0.05):   {sig_vs_random}/{len(results)}")
    print(f"  Folds sig. vs Momentum (p<0.05): {sig_vs_momentum}/{len(results)}")

    # Simulation results
    sim_returns = [r.simulation_total_return for r in results]
    sim_sharpes = [r.simulation_sharpe for r in results]

    print(f"\n💰 TRADING SIMULATION")
    print(f"  Mean Return: {np.mean(sim_returns):>+8.2f}% (±{np.std(sim_returns):.2f}%)")
    print(f"  Mean Sharpe: {np.mean(sim_sharpes):>8.2f} (±{np.std(sim_sharpes):.2f})")

    # Analytical vs Bootstrap comparison
    analytical_returns = [r.analytical_expected_annual_return for r in results]
    bootstrap_means = [r.bootstrap_mean_return for r in results]
    bootstrap_probs = [r.bootstrap_prob_positive for r in results]

    print(f"\n📐 EXPECTED RETURN ANALYSIS")
    print(f"  Analytical (from IC):  {np.mean(analytical_returns):>+8.2f}% annual")
    print(f"  Bootstrap Mean:        {np.mean(bootstrap_means):>+8.2f}% per fold")
    print(f"  Avg P(Return > 0):     {np.mean(bootstrap_probs):>7.1f}%")

    # Aggregate bootstrap across all folds
    print(f"\n  Combined across folds:")
    print(f"    Mean: {np.mean(bootstrap_means):>+8.2f}%")
    print(f"    Std:  {np.std(bootstrap_means):>8.2f}%")
    print(f"    Range: [{np.min(bootstrap_means):>+7.2f}%, {np.max(bootstrap_means):>+7.2f}%]")

    # === ENHANCED AGGREGATE STATISTICS ===

    # Multiple testing correction for per-fold p-values
    print(f"\n🔬 MULTIPLE TESTING CORRECTION")
    all_p_values = [r.vs_random_p_value for r in results] + [r.vs_momentum_p_value for r in results]
    correction = bonferroni_correction(all_p_values, alpha=0.05)
    print(f"  Tests performed:     {correction['n_tests']}")
    print(f"  Bonferroni α:        {correction['bonferroni_alpha']:.4f}")
    n_bonf_sig = sum(correction['bonferroni_significant'])
    n_holm_sig = sum(correction['holm_significant'])
    n_bh_sig = sum(correction['bh_fdr_significant'])
    print(f"  Bonferroni sig:      {n_bonf_sig}/{correction['n_tests']}")
    print(f"  Holm-Bonferroni sig: {n_holm_sig}/{correction['n_tests']}")
    print(f"  BH-FDR sig:          {n_bh_sig}/{correction['n_tests']}")

    # Block bootstrap summary if available
    block_probs = [r.block_bootstrap_prob_positive for r in results if r.block_bootstrap_prob_positive is not None]
    if len(block_probs) > 0:
        print(f"\n🔗 BLOCK BOOTSTRAP (aggregated)")
        print(f"  Avg P(Return > 0):   {np.mean(block_probs):>7.1f}%")
        print(f"  All folds > 50%:     {'Yes' if all(p > 50 for p in block_probs) else 'No'}")

    # Permutation test summary if available
    perm_p_random = [r.perm_vs_random_p_value for r in results if r.perm_vs_random_p_value is not None]
    perm_p_momentum = [r.perm_vs_momentum_p_value for r in results if r.perm_vs_momentum_p_value is not None]
    if len(perm_p_random) > 0:
        print(f"\n🔀 PERMUTATION TESTS (aggregated)")
        print(f"  vs Random sig (p<0.05):   {sum(1 for p in perm_p_random if p < 0.05)}/{len(perm_p_random)}")
        print(f"  vs Momentum sig (p<0.05): {sum(1 for p in perm_p_momentum if p < 0.05)}/{len(perm_p_momentum)}")

    # Effect size summary
    cohens_ds = [r.cohens_d_vs_random for r in results if r.cohens_d_vs_random is not None]
    if len(cohens_ds) > 0:
        avg_d = np.mean(cohens_ds)
        print(f"\n📏 EFFECT SIZE (aggregated)")
        print(f"  Mean Cohen's d vs Random: {avg_d:>+7.3f}")
        if abs(avg_d) < 0.2:
            print(f"  Interpretation:           negligible")
        elif abs(avg_d) < 0.5:
            print(f"  Interpretation:           small")
        elif abs(avg_d) < 0.8:
            print(f"  Interpretation:           medium")
        else:
            print(f"  Interpretation:           large")

    # Regime consistency
    regime_cons = [r.regime_consistency for r in results if r.regime_consistency is not None]
    if len(regime_cons) > 0:
        n_consistent = sum(1 for r in regime_cons if r == 'consistent')
        print(f"\n🌡️ REGIME ANALYSIS (aggregated)")
        print(f"  Folds with consistent regime performance: {n_consistent}/{len(regime_cons)}")
        if n_consistent == len(regime_cons):
            print(f"  Conclusion: Model works across all volatility regimes ✓")
        else:
            print(f"  Conclusion: Model performance varies by volatility regime")

    # Overall assessment
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")

    avg_ic = np.mean(mean_ics)
    avg_ir = np.mean(irs)
    avg_spread = np.mean(spreads)

    if avg_ic > 0.03 and avg_ir > 0.5 and avg_spread > 0.1:
        print("✓ STRONG: Consistent positive IC, good IR, positive spread")
    elif avg_ic > 0.01 and avg_spread > 0:
        print("○ MODERATE: Weak but positive signal detected")
    elif avg_ic > 0:
        print("△ WEAK: Marginal positive IC, may not be tradeable")
    else:
        print("✗ NO SIGNAL: Model does not outperform baselines")


def plot_results(results: List[FoldEvaluation], output_dir: str):
    """Generate visualization of evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    fold_nums = [r.fold_idx for r in results]

    # IC across folds
    ax = axes[0, 0]
    ics = [r.mean_ic for r in results]
    colors = ['green' if ic > 0 else 'red' for ic in ics]
    ax.bar(fold_nums, ics, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(np.mean(ics), color='blue', linestyle='--', label=f'Mean: {np.mean(ics):.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Information Coefficient')
    ax.set_title('Daily IC by Fold')
    ax.legend()

    # Long-short spread
    ax = axes[0, 1]
    spreads = [r.long_short_spread for r in results]
    colors = ['green' if s > 0 else 'red' for s in spreads]
    ax.bar(fold_nums, spreads, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(np.mean(spreads), color='blue', linestyle='--', label=f'Mean: {np.mean(spreads):.3f}%')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Spread (%)')
    ax.set_title('Long-Short Spread by Fold')
    ax.legend()

    # Model vs baselines
    ax = axes[0, 2]
    x = np.arange(len(results))
    width = 0.25
    ax.bar(x - width, [r.model_mean_return for r in results], width, label='Model', color='blue', alpha=0.7)
    ax.bar(x, [r.momentum_mean_return for r in results], width, label='Momentum', color='orange', alpha=0.7)
    ax.bar(x + width, [r.random_mean_return for r in results], width, label='Random', color='gray', alpha=0.7)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Mean Return (%)')
    ax.set_title('Model vs Baselines')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(fold_nums)

    # Simulation returns
    ax = axes[1, 0]
    sim_returns = [r.simulation_total_return for r in results]
    colors = ['green' if r > 0 else 'red' for r in sim_returns]
    ax.bar(fold_nums, sim_returns, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Simulation Returns by Fold')

    # P-values
    ax = axes[1, 1]
    p_random = [r.vs_random_p_value for r in results]
    p_momentum = [r.vs_momentum_p_value for r in results]
    ax.plot(fold_nums, p_random, 'o-', label='vs Random', color='blue')
    ax.plot(fold_nums, p_momentum, 's-', label='vs Momentum', color='orange')
    ax.axhline(0.05, color='red', linestyle='--', label='p=0.05')
    ax.set_xlabel('Fold')
    ax.set_ylabel('P-value')
    ax.set_title('Statistical Significance')
    ax.legend()
    ax.set_ylim(0, 1)

    # IR distribution
    ax = axes[1, 2]
    irs = [r.ir for r in results]
    ax.hist(irs, bins=10, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(irs), color='red', linestyle='--', label=f'Mean: {np.mean(irs):.3f}')
    ax.axvline(0.5, color='green', linestyle=':', label='Good IR (0.5)')
    ax.set_xlabel('Information Ratio')
    ax.set_ylabel('Count')
    ax.set_title('IR Distribution')
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'principled_evaluation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


def run_multi_trial_simulation(
    evaluator: PrincipledEvaluator,
    model: SimpleTransformerPredictor,
    checkpoint: dict,
    test_dates: List[str],
    bin_edges: Optional[torch.Tensor],
    n_trials: int = 50,
    subsample_ratio: float = 0.7,
    batch_size: int = 256,
    max_stocks: int = 500,
) -> Dict:
    """
    Run multiple simulations with different random stock subsets.

    This validates whether the model's performance is robust to different
    stock selections, or if it's driven by a few lucky picks.

    Args:
        n_trials: Number of simulation trials
        subsample_ratio: Fraction of stocks to use each trial

    Returns:
        Dictionary with distribution of outcomes
    """
    print(f"\n{'='*70}")
    print(f"MULTI-TRIAL SIMULATION ({n_trials} trials, {subsample_ratio:.0%} subsample)")
    print(f"{'='*70}")

    trial_returns = []
    trial_sharpes = []

    # Use non-overlapping dates
    eval_dates = test_dates[::evaluator.horizon_days]

    with h5py.File(evaluator.dataset_path, 'r') as h5f:
        for trial in tqdm(range(n_trials), desc="  Trials"):
            # Random subset of tickers for this trial
            n_subsample = int(len(evaluator.tickers) * subsample_ratio)
            trial_tickers = list(np.random.choice(
                evaluator.tickers[:max_stocks],
                size=min(n_subsample, max_stocks),
                replace=False
            ))

            capital = 100000.0
            trial_daily_returns = []

            for date in eval_dates:
                date_idx = evaluator.all_dates.index(date) if date in evaluator.all_dates else -1
                if date_idx < evaluator.seq_len:
                    continue

                features_list = []
                actual_returns = []

                for ticker in trial_tickers:
                    if ticker not in h5f:
                        continue
                    try:
                        features_2d = h5f[ticker]['features'][:]
                        if date_idx >= features_2d.shape[0]:
                            continue
                        features = torch.from_numpy(features_2d[date_idx]).float()

                        # Skip if feature dimension doesn't match
                        if features.shape[0] != evaluator.input_dim:
                            continue

                        actual_ret = evaluator._get_future_return(ticker, date)
                        if actual_ret is None:
                            continue
                        features_list.append(features)
                        actual_returns.append(actual_ret)
                    except:
                        continue

                if len(features_list) < evaluator.top_k:
                    continue

                # Get predictions
                predictions = []
                for i in range(0, len(features_list), batch_size):
                    batch = features_list[i:i+batch_size]
                    preds = evaluator.predict_batch(model, batch, bin_edges)
                    predictions.extend(preds)

                predictions = np.array(predictions)
                actual_returns = np.array(actual_returns)

                # Top-k selection
                top_k_idx = np.argsort(predictions)[::-1][:evaluator.top_k]
                model_return = actual_returns[top_k_idx].mean()
                trial_daily_returns.append(model_return)

                capital *= (1 + model_return)

            if len(trial_daily_returns) > 0:
                total_return = (capital / 100000.0 - 1) * 100
                trial_returns.append(total_return)

                returns = np.array(trial_daily_returns)
                trades_per_year = 252 / evaluator.horizon_days
                if np.std(returns) > 1e-8:
                    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
                else:
                    sharpe = 0
                trial_sharpes.append(sharpe)

    trial_returns = np.array(trial_returns)
    trial_sharpes = np.array(trial_sharpes)

    results = {
        'n_trials': n_trials,
        'subsample_ratio': subsample_ratio,
        'mean_return': float(np.mean(trial_returns)),
        'std_return': float(np.std(trial_returns)),
        'median_return': float(np.median(trial_returns)),
        'ci_5': float(np.percentile(trial_returns, 5)),
        'ci_95': float(np.percentile(trial_returns, 95)),
        'prob_positive': float(np.mean(trial_returns > 0)),
        'var_5pct': float(np.percentile(trial_returns, 5)),
        'cvar_5pct': float(np.mean(trial_returns[trial_returns <= np.percentile(trial_returns, 5)])) if len(trial_returns) > 0 else 0,
        'mean_sharpe': float(np.mean(trial_sharpes)),
        'std_sharpe': float(np.std(trial_sharpes)),
        'all_returns': trial_returns.tolist(),
        'all_sharpes': trial_sharpes.tolist(),
    }

    print(f"\n  Results:")
    print(f"    Mean Return:    {results['mean_return']:>+8.2f}%")
    print(f"    Std Return:     {results['std_return']:>8.2f}%")
    print(f"    Median Return:  {results['median_return']:>+8.2f}%")
    print(f"    90% CI:         [{results['ci_5']:>+7.2f}%, {results['ci_95']:>+7.2f}%]")
    print(f"    P(Return > 0):  {results['prob_positive']*100:>7.1f}%")
    print(f"    5% VaR:         {results['var_5pct']:>+8.2f}%")
    print(f"    5% CVaR:        {results['cvar_5pct']:>+8.2f}%")
    print(f"    Mean Sharpe:    {results['mean_sharpe']:>8.2f} (±{results['std_sharpe']:.2f})")

    return results


def run_monte_carlo_validation(
    evaluator: PrincipledEvaluator,
    model: SimpleTransformerPredictor,
    checkpoint: dict,
    test_dates: List[str],
    bin_edges: Optional[torch.Tensor],
    n_trials: int = 100,
    stocks_per_trial: int = 150,
    top_k_values: List[int] = None,
    batch_size: int = 256,
    max_stocks: int = 500,
    seed: int = None,
) -> Dict:
    """
    Comprehensive Monte Carlo validation of trading performance.

    EFFICIENT IMPLEMENTATION: Scores all stocks ONCE, then resamples for trials.

    Runs many trials with different random stock subsets to:
    1. Get diverse model behavior across stock universes
    2. Compare model vs random in each trial (paired)
    3. Validate that simulation returns match analytical IC predictions

    The Fundamental Law of Active Management predicts:
        E[excess_return] ≈ IC × σ_cross × √(breadth) × TC

    We test whether simulated excess returns match this prediction.

    Args:
        n_trials: Number of Monte Carlo trials
        stocks_per_trial: Number of stocks to sample per trial
        top_k_values: List of top-k values to test (default: [5, 10, 15, 20])
        seed: Random seed for reproducibility

    Returns:
        Dictionary with comprehensive validation results
    """
    if seed is not None:
        np.random.seed(seed)

    if top_k_values is None:
        top_k_values = [5, 10, 15, 20]

    print(f"\n{'='*70}")
    print(f"MONTE CARLO VALIDATION")
    print(f"{'='*70}")
    print(f"  Trials:           {n_trials}")
    print(f"  Stocks/trial:     {stocks_per_trial}")
    print(f"  Top-k values:     {top_k_values}")
    print(f"  Horizon:          {evaluator.horizon_days} days")

    # Use non-overlapping dates
    eval_dates = test_dates[::evaluator.horizon_days]
    trades_per_year = 252 / evaluator.horizon_days

    # =========================================================================
    # PHASE 1: Pre-load data and score ALL stocks ONCE
    # =========================================================================
    print(f"\n  Phase 1: Loading data and scoring all stocks (single forward pass)...")

    features_cache = {}
    prices_cache = {}

    with h5py.File(evaluator.dataset_path, 'r') as h5f:
        available_tickers = [t for t in evaluator.tickers[:max_stocks] if t in h5f]
        for ticker in tqdm(available_tickers, desc="  Loading features", leave=False):
            try:
                dates_bytes = h5f[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                features = h5f[ticker]['features'][:].astype(np.float32)
                if features.shape[1] == evaluator.input_dim:
                    features_cache[ticker] = (dates, features)
            except Exception:
                continue

    if evaluator.prices_file is not None:
        for ticker in features_cache.keys():
            if ticker not in evaluator.prices_file:
                continue
            try:
                dates_bytes = evaluator.prices_file[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                prices = evaluator.prices_file[ticker]['prices'][:].astype(np.float32)
                if prices.max() <= 10000:
                    prices_cache[ticker] = (dates, prices)
            except Exception:
                continue

    valid_tickers = list(set(features_cache.keys()) & set(prices_cache.keys()))
    print(f"  Loaded {len(valid_tickers)} valid tickers")

    if len(valid_tickers) < stocks_per_trial:
        stocks_per_trial = len(valid_tickers)
        print(f"  ⚠️  Reduced stocks_per_trial to {stocks_per_trial}")

    # Score ALL stocks on ALL dates - SINGLE FORWARD PASS
    # Structure: date_data[date] = {'tickers': [...], 'predictions': [...], 'returns': [...]}
    date_data = {}

    for date in tqdm(eval_dates, desc="  Scoring all stocks", leave=False):
        features_list = []
        ticker_list = []
        returns_list = []

        for ticker in valid_tickers:
            feat_dates, feat_array = features_cache[ticker]
            price_dates, price_array = prices_cache[ticker]

            try:
                feat_idx = feat_dates.index(date)
                price_idx = price_dates.index(date)
                future_idx = price_idx + evaluator.horizon_days
            except ValueError:
                continue

            if feat_idx >= feat_array.shape[0] or future_idx >= len(price_array):
                continue
            if price_array[price_idx] <= 0:
                continue

            # Need seq_len days of history ending at feat_idx
            start_idx = feat_idx - evaluator.seq_len + 1
            if start_idx < 0:
                continue

            actual_ret = (price_array[future_idx] / price_array[price_idx]) - 1.0

            # Filter extreme returns (likely data errors)
            if abs(actual_ret) > 0.5:
                continue

            # Get full sequence of features
            seq_features = feat_array[start_idx:feat_idx + 1]  # Shape: (seq_len, features)
            features = torch.from_numpy(seq_features).float()
            features_list.append(features)
            ticker_list.append(ticker)
            returns_list.append(actual_ret)

        if len(features_list) < max(top_k_values) * 2:
            continue

        # Get model predictions for ALL stocks on this date (SINGLE BATCH)
        predictions = evaluator.predict_batch(model, features_list, bin_edges)
        predictions = np.array(predictions)
        returns_array = np.array(returns_list)

        date_data[date] = {
            'tickers': ticker_list,
            'predictions': predictions,
            'returns': returns_array,
        }

    print(f"  Scored {len(date_data)} dates with sufficient stocks")

    if len(date_data) < 5:
        print("  ⚠️  Insufficient dates for Monte Carlo validation")
        return {'error': 'insufficient_dates', 'n_dates': len(date_data)}

    # =========================================================================
    # DIAGNOSTIC: Compute market benchmark and check for survivorship bias
    # =========================================================================
    all_returns_flat = []
    for date, data in date_data.items():
        all_returns_flat.extend(data['returns'])
    all_returns_flat = np.array(all_returns_flat)

    # Market benchmark: equal-weight all stocks on each date
    market_capital = 100000.0
    market_daily_returns = []
    for date in sorted(date_data.keys()):
        data = date_data[date]
        market_ret = np.mean(data['returns'])
        market_daily_returns.append(market_ret)
        market_capital *= (1 + market_ret)
    market_total = (market_capital / 100000.0 - 1) * 100

    n_trades = len(date_data)
    avg_per_trade = np.mean(all_returns_flat) * 100
    std_per_trade = np.std(all_returns_flat) * 100
    dates_list = sorted(date_data.keys())

    # Annualized market return
    years_elapsed = len(eval_dates) * evaluator.horizon_days / 252
    market_annualized = ((market_capital / 100000.0) ** (1 / max(years_elapsed, 0.01)) - 1) * 100 if years_elapsed > 0 else 0

    print(f"\n  📊 UNIVERSE DIAGNOSTICS:")
    print(f"     Date range:     {dates_list[0]} to {dates_list[-1]}")
    print(f"     Trading dates:  {n_trades}")
    print(f"     Horizon days:   {evaluator.horizon_days}")
    print(f"     Years elapsed:  {years_elapsed:.2f}")
    print(f"     Avg stocks/day: {np.mean([len(d['returns']) for d in date_data.values()]):.0f}")
    print(f"     Avg per-trade:  {avg_per_trade:+.3f}% (std: {std_per_trade:.3f}%)")
    print(f"     ")
    print(f"  📈 MARKET BENCHMARK (Equal-Weight All Stocks):")
    print(f"     Total Return:   {market_total:+.2f}%")
    print(f"     Annualized:     {market_annualized:+.2f}%")
    print(f"     ")
    print(f"  ⚠️  SURVIVORSHIP BIAS NOTE:")
    print(f"     Only stocks with data on BOTH dates included.")
    print(f"     Delisted/bankrupt stocks excluded from universe.")
    print(f"     Random selection should match market benchmark if no bias.")
    print(f"     Compare random returns below to market: {market_total:+.2f}%")

    # Per-trade return diagnostics
    n_positive = np.sum(all_returns_flat > 0)
    n_negative = np.sum(all_returns_flat < 0)
    n_total_trades = len(all_returns_flat)
    pct_positive = n_positive / n_total_trades * 100
    median_ret = np.median(all_returns_flat) * 100
    skewness = stats.skew(all_returns_flat) if len(all_returns_flat) > 10 else 0
    p10 = np.percentile(all_returns_flat, 10) * 100
    p90 = np.percentile(all_returns_flat, 90) * 100

    print(f"     ")
    print(f"  📉 PER-TRADE RETURN DISTRIBUTION:")
    print(f"     Total stock-days: {n_total_trades:,}")
    print(f"     Positive trades:  {n_positive:,} ({pct_positive:.1f}%)")
    print(f"     Negative trades:  {n_negative:,} ({100-pct_positive:.1f}%)")
    print(f"     Median return:    {median_ret:+.3f}%")
    print(f"     Mean return:      {avg_per_trade:+.3f}%")
    print(f"     Skewness:         {skewness:+.3f} ({'right-tailed' if skewness > 0.5 else 'left-tailed' if skewness < -0.5 else 'symmetric'})")
    print(f"     10th-90th pctl:   [{p10:+.2f}%, {p90:+.2f}%]")

    # Check for issues
    if pct_positive > 55:
        print(f"     ⚠️  {pct_positive:.0f}% positive trades may indicate bias or bull market")
    if abs(skewness) > 1:
        print(f"     ⚠️  High skewness ({skewness:+.2f}) - asymmetric return distribution")

    # =========================================================================
    # PHASE 2: Monte Carlo resampling (CPU only, no forward passes)
    # =========================================================================
    print(f"\n  Phase 2: Running {n_trials} Monte Carlo trials (CPU resampling)...")

    # Results storage by top_k
    results_by_topk = {k: {
        'model_returns': [],
        'random_returns': [],
        'excess_returns': [],
        'daily_ics': [],
        'trial_ics': [],
        'trial_sharpes': [],
        # Diagnostics to verify random selection correctness
        'expected_random_returns': [],  # What random SHOULD get (mean of all available)
        'actual_random_returns': [],    # What random actually got
    } for k in top_k_values}

    for trial in tqdm(range(n_trials), desc="  Trials"):
        # Sample random subset of stocks for this trial
        trial_tickers = set(np.random.choice(valid_tickers, size=stocks_per_trial, replace=False))

        # Collect data for this trial by filtering cached predictions
        all_predictions = []
        all_actual_returns = []
        all_dates_filtered = []  # Per-date filtered data

        for date, data in date_data.items():
            # Filter to only stocks in this trial
            mask = [t in trial_tickers for t in data['tickers']]
            if sum(mask) < max(top_k_values) * 2:
                continue

            mask = np.array(mask)
            filtered_preds = data['predictions'][mask]
            filtered_rets = data['returns'][mask]

            all_predictions.extend(filtered_preds)
            all_actual_returns.extend(filtered_rets)
            all_dates_filtered.append({
                'predictions': filtered_preds,
                'returns': filtered_rets,
            })

        if len(all_dates_filtered) < 5:
            continue

        # Compute trial-level IC (correlation across all predictions in this trial)
        all_preds_array = np.array(all_predictions)
        all_rets_array = np.array(all_actual_returns)

        if np.std(all_preds_array) > 1e-8 and np.std(all_rets_array) > 1e-8:
            trial_ic, _ = pearsonr(all_preds_array, all_rets_array)
        else:
            trial_ic = 0.0

        # Now compute returns for each top_k
        for top_k in top_k_values:
            model_capital = 100000.0
            random_capital = 100000.0
            expected_random_capital = 100000.0  # Diagnostic: what random SHOULD get
            daily_model_returns = []
            daily_random_returns = []
            daily_expected_random = []
            daily_ics = []

            for data in all_dates_filtered:
                preds = data['predictions']
                actual = data['returns']

                if len(preds) < top_k:
                    continue

                # Model selection (use cached predictions)
                sorted_idx = np.argsort(preds)[::-1]
                model_topk = sorted_idx[:top_k]
                model_ret = actual[model_topk].mean()

                # Random selection
                random_idx = np.random.choice(len(actual), top_k, replace=False)
                random_ret = actual[random_idx].mean()

                # Expected random = equal-weight all available stocks (diagnostic)
                expected_random = actual.mean()

                daily_model_returns.append(model_ret)
                daily_random_returns.append(random_ret)
                daily_expected_random.append(expected_random)
                expected_random_capital *= (1 + expected_random)

                # Daily IC
                if np.std(preds) > 1e-8 and np.std(actual) > 1e-8:
                    ic, _ = pearsonr(preds, actual)
                    if not np.isnan(ic):
                        daily_ics.append(ic)

                model_capital *= (1 + model_ret)
                random_capital *= (1 + random_ret)

            if len(daily_model_returns) == 0:
                continue

            # Compound returns
            model_total = (model_capital / 100000.0 - 1) * 100
            random_total = (random_capital / 100000.0 - 1) * 100
            expected_random_total = (expected_random_capital / 100000.0 - 1) * 100
            excess = model_total - random_total

            # Sharpe
            daily_model = np.array(daily_model_returns)
            if np.std(daily_model) > 1e-8:
                sharpe = (np.mean(daily_model) / np.std(daily_model)) * np.sqrt(trades_per_year)
            else:
                sharpe = 0.0

            # Store results
            results_by_topk[top_k]['model_returns'].append(model_total)
            results_by_topk[top_k]['random_returns'].append(random_total)
            results_by_topk[top_k]['excess_returns'].append(excess)
            results_by_topk[top_k]['daily_ics'].extend(daily_ics)
            results_by_topk[top_k]['trial_ics'].append(trial_ic if not np.isnan(trial_ic) else 0.0)
            results_by_topk[top_k]['trial_sharpes'].append(sharpe)
            # Diagnostics
            results_by_topk[top_k]['expected_random_returns'].append(expected_random_total)
            results_by_topk[top_k]['actual_random_returns'].append(random_total)

    # Aggregate results and validate analytical model
    print(f"\n{'='*70}")
    print("MONTE CARLO RESULTS")
    print(f"{'='*70}")

    summary = {}

    for top_k in top_k_values:
        data = results_by_topk[top_k]

        if len(data['model_returns']) < 10:
            print(f"\n  Top-{top_k}: Insufficient trials ({len(data['model_returns'])})")
            continue

        model_rets = np.array(data['model_returns'])
        random_rets = np.array(data['random_returns'])
        excess_rets = np.array(data['excess_returns'])
        trial_ics = np.array(data['trial_ics'])
        daily_ics = np.array(data['daily_ics'])
        trial_sharpes = np.array(data['trial_sharpes'])
        expected_random_rets = np.array(data['expected_random_returns'])

        # === Diagnostic: Check if random selection is unbiased ===
        # Random selection should, in expectation, match the equal-weight portfolio
        expected_random_mean = np.mean(expected_random_rets)
        actual_random_mean = np.mean(random_rets)
        random_bias = actual_random_mean - expected_random_mean

        # === Statistical tests: Model vs Random ===
        # Paired t-test on total returns
        t_stat, p_value = stats.ttest_rel(model_rets, random_rets)
        p_value_one_sided = p_value / 2

        # Wilcoxon signed-rank (non-parametric)
        try:
            _, wilcoxon_p = stats.wilcoxon(model_rets - random_rets, alternative='greater')
        except ValueError:
            wilcoxon_p = 1.0

        # Effect size
        effect = compute_effect_size(model_rets, random_rets)

        # Permutation test
        perm_result = permutation_test(model_rets, random_rets, n_permutations=5000)

        # === Validate Analytical IC Model ===
        # Fundamental Law: E[return] ≈ IC × σ_cross × √(breadth) × TC
        mean_ic = np.mean(daily_ics) if len(daily_ics) > 0 else 0
        cross_sectional_std = np.std(daily_ics) if len(daily_ics) > 1 else 0.05
        breadth = trades_per_year
        transfer_coef = np.sqrt(2 / np.pi)  # For long-only top-k

        # Analytical expected excess return (annualized, in %)
        # Note: cross_sectional_std here approximates the cross-sectional vol of returns
        # A better estimate would be the average daily cross-sectional std of returns
        analytical_excess = mean_ic * 0.02 * np.sqrt(breadth) * transfer_coef * 100  # 2% daily cross-sec vol

        # Actual mean excess return per trade (annualized)
        actual_excess_per_trade = np.mean(excess_rets) / len(eval_dates) * trades_per_year

        # Test if analytical and simulation agree
        # Use bootstrap to get CI on simulation excess
        boot_excesses = []
        for _ in range(1000):
            resampled = np.random.choice(excess_rets, size=len(excess_rets), replace=True)
            boot_excesses.append(np.mean(resampled))
        boot_excesses = np.array(boot_excesses)
        sim_ci_lower = np.percentile(boot_excesses, 5)
        sim_ci_upper = np.percentile(boot_excesses, 95)

        # Does analytical fall within simulation CI?
        analytical_in_ci = sim_ci_lower <= analytical_excess <= sim_ci_upper

        # Store summary
        summary[top_k] = {
            'n_trials': len(model_rets),
            'model_mean': float(np.mean(model_rets)),
            'model_std': float(np.std(model_rets)),
            'random_mean': float(np.mean(random_rets)),
            'random_std': float(np.std(random_rets)),
            'excess_mean': float(np.mean(excess_rets)),
            'excess_std': float(np.std(excess_rets)),
            'excess_ci_lower': float(np.percentile(excess_rets, 5)),
            'excess_ci_upper': float(np.percentile(excess_rets, 95)),
            'prob_model_beats_random': float(np.mean(model_rets > random_rets)),
            't_stat': float(t_stat),
            'p_value': float(p_value_one_sided),
            'wilcoxon_p': float(wilcoxon_p),
            'permutation_p': float(perm_result['p_value_one_sided']),
            'cohens_d': float(effect['cohens_d']),
            'effect_interpretation': effect['interpretation'],
            'mean_ic': float(mean_ic),
            'mean_sharpe': float(np.mean(trial_sharpes)),
            'analytical_excess': float(analytical_excess),
            'simulation_excess': float(np.mean(excess_rets)),
            'analytical_in_sim_ci': analytical_in_ci,
            'expected_random_mean': float(expected_random_mean),
            'random_selection_bias': float(random_bias),
        }

        # Print results for this top_k
        print(f"\n  Top-{top_k} ({len(model_rets)} trials):")
        print(f"    Model Mean:     {np.mean(model_rets):>+8.2f}% (±{np.std(model_rets):.2f}%)")
        print(f"    Random Mean:    {np.mean(random_rets):>+8.2f}% (±{np.std(random_rets):.2f}%)")
        print(f"    Excess Mean:    {np.mean(excess_rets):>+8.2f}% (±{np.std(excess_rets):.2f}%)")
        print(f"    90% CI Excess:  [{np.percentile(excess_rets, 5):>+7.2f}%, {np.percentile(excess_rets, 95):>+7.2f}%]")
        print(f"    P(Model > Rand):{np.mean(model_rets > random_rets)*100:>7.1f}%")
        print(f"    ")
        sig = "***" if p_value_one_sided < 0.01 else "**" if p_value_one_sided < 0.05 else "*" if p_value_one_sided < 0.1 else ""
        print(f"    Statistical Tests:")
        print(f"      t-test p:     {p_value_one_sided:>8.4f} {sig}")
        print(f"      Wilcoxon p:   {wilcoxon_p:>8.4f}")
        print(f"      Permutation p:{perm_result['p_value_one_sided']:>8.4f}")
        print(f"      Cohen's d:    {effect['cohens_d']:>+8.3f} ({effect['interpretation']})")
        print(f"    ")
        print(f"    IC Metrics:")
        print(f"      Mean IC:      {mean_ic:>+8.4f}")
        print(f"      Mean Sharpe:  {np.mean(trial_sharpes):>8.2f}")
        print(f"    ")
        print(f"    Analytical vs Simulation:")
        print(f"      Analytical:   {analytical_excess:>+8.2f}%")
        print(f"      Simulation:   {np.mean(excess_rets):>+8.2f}%")
        print(f"      Sim 90% CI:   [{sim_ci_lower:>+7.2f}%, {sim_ci_upper:>+7.2f}%]")
        print(f"      Agreement:    {'✓ Analytical within simulation CI' if analytical_in_ci else '✗ Analytical outside simulation CI'}")
        print(f"    ")
        print(f"    Random Selection Validation:")
        print(f"      Expected (EW): {expected_random_mean:>+8.2f}% (equal-weight all stocks)")
        print(f"      Actual random: {actual_random_mean:>+8.2f}%")
        print(f"      Bias:          {random_bias:>+8.2f}%")
        if abs(random_bias) < 1.0:
            print(f"      Status:        ✓ Random selection is unbiased")
        else:
            print(f"      Status:        ⚠️  Random selection bias detected!")

    # Overall summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_prob_beats = [summary[k]['prob_model_beats_random'] for k in summary]
    all_p_values = [summary[k]['p_value'] for k in summary]
    all_agreements = [summary[k]['analytical_in_sim_ci'] for k in summary]
    all_random_means = [summary[k]['random_mean'] for k in summary]

    print(f"\n  Market Context:")
    print(f"    Market (EW) Return:        {market_total:+.2f}% ({market_annualized:+.2f}% ann.)")
    print(f"    Avg Random Return:         {np.mean(all_random_means):+.2f}%")
    diff_from_market = np.mean(all_random_means) - market_total
    if abs(diff_from_market) < 5:
        print(f"    Random ≈ Market:           ✓ Confirms no selection bias ({diff_from_market:+.1f}% diff)")
    else:
        print(f"    Random vs Market gap:      ⚠️  {diff_from_market:+.1f}% (potential bias)")

    print(f"\n  Across all top-k configurations:")
    print(f"    Avg P(Model > Random):     {np.mean(all_prob_beats)*100:.1f}%")
    print(f"    Configs with p < 0.05:     {sum(1 for p in all_p_values if p < 0.05)}/{len(all_p_values)}")
    print(f"    Configs with IC agreement: {sum(all_agreements)}/{len(all_agreements)}")

    if all(p < 0.05 for p in all_p_values) and np.mean(all_prob_beats) > 0.6:
        print(f"\n  ✓ ROBUST: Model consistently beats random across configurations")
    elif np.mean(all_prob_beats) > 0.55:
        print(f"\n  ○ MARGINAL: Model shows some edge but not consistently significant")
    else:
        print(f"\n  ✗ NO EDGE: Model does not reliably beat random")

    if all(all_agreements):
        print(f"  ✓ IC MODEL VALIDATED: Analytical predictions match simulation")
    elif sum(all_agreements) > len(all_agreements) / 2:
        print(f"  ○ IC MODEL PARTIAL: Some agreement between analytical and simulation")
    else:
        print(f"  ✗ IC MODEL MISMATCH: Analytical predictions don't match simulation")

    # Check random selection bias
    all_biases = [summary[k].get('random_selection_bias', 0) for k in summary]
    avg_bias = np.mean(all_biases)
    if abs(avg_bias) < 1.0:
        print(f"  ✓ RANDOM SELECTION UNBIASED: Random matches expected ({avg_bias:+.2f}% avg bias)")
    else:
        print(f"  ⚠️  RANDOM SELECTION BIAS: {avg_bias:+.2f}% (investigate simulation code)")

    # Store raw data for plotting
    raw_data = {k: {
        'model_returns': data['model_returns'],
        'random_returns': data['random_returns'],
        'excess_returns': data['excess_returns'],
    } for k, data in results_by_topk.items() if len(data['model_returns']) > 0}

    return {
        'n_trials': n_trials,
        'stocks_per_trial': stocks_per_trial,
        'top_k_values': top_k_values,
        'by_topk': summary,
        'raw_data': raw_data,
        'market_benchmark': {
            'total_return': float(market_total),
            'annualized_return': float(market_annualized),
            'date_range': [dates_list[0], dates_list[-1]],
            'n_trading_dates': n_trades,
            'years_elapsed': float(years_elapsed),
        },
        'overall': {
            'avg_prob_beats_random': float(np.mean(all_prob_beats)),
            'avg_random_return': float(np.mean(all_random_means)),
            'random_vs_market_diff': float(np.mean(all_random_means) - market_total),
            'configs_significant': int(sum(1 for p in all_p_values if p < 0.05)),
            'configs_ic_agreement': int(sum(all_agreements)),
            'total_configs': len(all_p_values),
        }
    }


def plot_monte_carlo_results(mc_results: Dict, output_dir: str, fold_idx: int = None):
    """
    Generate distribution plots for Monte Carlo validation results.

    Creates one chart per trading configuration (top-k), showing:
    - Distribution of model returns vs random returns
    - Excess return distribution
    - Statistical test results

    Args:
        mc_results: Results from run_monte_carlo_validation
        output_dir: Directory to save plots
        fold_idx: Optional fold index for filename
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_data = mc_results.get('raw_data', {})
    if not raw_data:
        print("  No Monte Carlo data to plot")
        return

    top_k_values = sorted(raw_data.keys())
    n_configs = len(top_k_values)

    if n_configs == 0:
        return

    # Create subplot grid - one row per top-k
    fig, axes = plt.subplots(n_configs, 3, figsize=(15, 4 * n_configs))

    if n_configs == 1:
        axes = axes.reshape(1, -1)

    for i, top_k in enumerate(top_k_values):
        data = raw_data[top_k]
        model_rets = np.array(data['model_returns'])
        random_rets = np.array(data['random_returns'])
        excess_rets = np.array(data['excess_returns'])
        summary = mc_results['by_topk'].get(top_k, {})

        # Column 1: Model vs Random distributions
        ax = axes[i, 0]
        bins = np.linspace(
            min(model_rets.min(), random_rets.min()),
            max(model_rets.max(), random_rets.max()),
            30
        )
        ax.hist(model_rets, bins=bins, alpha=0.6, label='Model', color='blue', edgecolor='black')
        ax.hist(random_rets, bins=bins, alpha=0.6, label='Random', color='gray', edgecolor='black')
        ax.axvline(np.mean(model_rets), color='blue', linestyle='--', linewidth=2,
                   label=f'Model μ: {np.mean(model_rets):+.1f}%')
        ax.axvline(np.mean(random_rets), color='gray', linestyle='--', linewidth=2,
                   label=f'Random μ: {np.mean(random_rets):+.1f}%')
        ax.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Total Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top-{top_k}: Return Distributions')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Column 2: Excess return distribution
        ax = axes[i, 1]
        ax.hist(excess_rets, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(excess_rets), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(excess_rets):+.2f}%')
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ci_lower = np.percentile(excess_rets, 5)
        ci_upper = np.percentile(excess_rets, 95)
        ax.axvline(ci_lower, color='orange', linestyle=':', linewidth=1.5, label=f'90% CI')
        ax.axvline(ci_upper, color='orange', linestyle=':', linewidth=1.5)
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
                         ci_lower, ci_upper, alpha=0.2, color='orange')
        ax.set_xlabel('Excess Return (Model - Random) (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top-{top_k}: Excess Return Distribution')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Column 3: Summary statistics box
        ax = axes[i, 2]
        ax.axis('off')

        # Build summary text
        p_val = summary.get('p_value', 1.0)
        sig_stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""

        text = (
            f"Top-{top_k} Configuration\n"
            f"{'=' * 30}\n\n"
            f"Trials: {len(model_rets)}\n\n"
            f"Model Return:\n"
            f"  Mean: {np.mean(model_rets):+.2f}%\n"
            f"  Std:  {np.std(model_rets):.2f}%\n\n"
            f"Random Return:\n"
            f"  Mean: {np.mean(random_rets):+.2f}%\n"
            f"  Std:  {np.std(random_rets):.2f}%\n\n"
            f"Excess Return:\n"
            f"  Mean: {np.mean(excess_rets):+.2f}%\n"
            f"  90% CI: [{ci_lower:+.1f}%, {ci_upper:+.1f}%]\n\n"
            f"P(Model > Random): {summary.get('prob_model_beats_random', 0)*100:.1f}%\n\n"
            f"Statistical Tests:\n"
            f"  t-test p-value: {p_val:.4f} {sig_stars}\n"
            f"  Cohen's d: {summary.get('cohens_d', 0):+.3f}\n"
            f"  Effect: {summary.get('effect_interpretation', 'N/A')}\n\n"
            f"IC Analysis:\n"
            f"  Mean IC: {summary.get('mean_ic', 0):+.4f}\n"
            f"  Analytical: {summary.get('analytical_excess', 0):+.2f}%\n"
            f"  Simulation: {summary.get('simulation_excess', 0):+.2f}%\n"
            f"  Agreement: {'✓' if summary.get('analytical_in_sim_ci', False) else '✗'}"
        )

        ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(
        f'Monte Carlo Validation: {mc_results["n_trials"]} trials, '
        f'{mc_results["stocks_per_trial"]} stocks/trial',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    # Save plot
    suffix = f'_fold{fold_idx}' if fold_idx is not None else ''
    plot_path = os.path.join(output_dir, f'monte_carlo_distributions{suffix}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Monte Carlo plot saved to: {plot_path}")

    # Also save a combined comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Box plot comparison across top-k
    ax = axes[0]
    positions = np.arange(len(top_k_values))
    width = 0.35

    model_data = [raw_data[k]['model_returns'] for k in top_k_values]
    random_data = [raw_data[k]['random_returns'] for k in top_k_values]

    bp1 = ax.boxplot(model_data, positions=positions - width/2, widths=width,
                     patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.6))
    bp2 = ax.boxplot(random_data, positions=positions + width/2, widths=width,
                     patch_artist=True, boxprops=dict(facecolor='gray', alpha=0.6))

    ax.set_xticks(positions)
    ax.set_xticklabels([f'k={k}' for k in top_k_values])
    ax.set_xlabel('Top-K Configuration')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Model vs Random: Distribution by Configuration')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Model', 'Random'], loc='upper right')
    ax.grid(alpha=0.3)

    # Right: Excess return by configuration
    ax = axes[1]
    excess_data = [raw_data[k]['excess_returns'] for k in top_k_values]
    bp = ax.boxplot(excess_data, positions=positions, widths=0.6,
                    patch_artist=True, boxprops=dict(facecolor='purple', alpha=0.6))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'k={k}' for k in top_k_values])
    ax.set_xlabel('Top-K Configuration')
    ax.set_ylabel('Excess Return (%)')
    ax.set_title('Model Excess Return by Configuration')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.grid(alpha=0.3)

    # Add significance markers
    for i, k in enumerate(top_k_values):
        p_val = mc_results['by_topk'].get(k, {}).get('p_value', 1.0)
        if p_val < 0.01:
            ax.text(i, ax.get_ylim()[1] * 0.95, '***', ha='center', fontsize=12)
        elif p_val < 0.05:
            ax.text(i, ax.get_ylim()[1] * 0.95, '**', ha='center', fontsize=12)
        elif p_val < 0.1:
            ax.text(i, ax.get_ylim()[1] * 0.95, '*', ha='center', fontsize=12)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f'monte_carlo_comparison{suffix}.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Monte Carlo comparison saved to: {comparison_path}")


def run_sweep(
    checkpoint_files: List[str],
    evaluator_base: PrincipledEvaluator,
    args,
    wf_config: dict,
) -> Dict:
    """
    Run evaluation sweep across multiple horizons and top-k values.

    EFFICIENT: Pre-loads all data into RAM, runs model forward pass ONCE per
    fold/date, then does CPU-only operations for different horizon/top-k combinations.

    Returns a summary of all configurations tested.
    """
    horizon_map = {0: 1, 1: 5, 2: 10, 3: 20}
    max_horizon = max(horizon_map[h] for h in args.sweep_horizons)

    print(f"\n{'='*70}")
    print("CONFIGURATION SWEEP (Efficient Mode)")
    print(f"{'='*70}")
    print(f"  Horizons: {[f'{horizon_map[h]}d (idx={h})' for h in args.sweep_horizons]}")
    print(f"  Top-K values: {args.sweep_top_ks}")
    print(f"  Total configurations: {len(args.sweep_horizons) * len(args.sweep_top_ks)}")
    print(f"  Strategy: Pre-load data to RAM, single forward pass per fold, CPU-only config eval")

    # Verify and report device
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠️  CUDA requested but not available, falling back to CPU")
            args.device = 'cpu'
            evaluator_base.device = 'cpu'
    else:
        print(f"  Using device: {args.device}")

    # ========================================
    # PRE-LOAD ALL DATA INTO RAM
    # ========================================
    print(f"\n  Pre-loading features and prices into RAM...")

    # Load all features into RAM (dict: ticker -> features array)
    features_cache = {}  # ticker -> (dates_list, features_2d)
    with h5py.File(evaluator_base.dataset_path, 'r') as h5f:
        for ticker in tqdm(evaluator_base.tickers[:args.max_stocks], desc="  Loading features"):
            if ticker not in h5f:
                continue
            try:
                dates_bytes = h5f[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                features = h5f[ticker]['features'][:].astype(np.float32)  # Load into RAM
                if features.shape[1] == evaluator_base.input_dim:
                    features_cache[ticker] = (dates, features)
            except Exception:
                continue

    # Load all prices into RAM (with data quality filtering)
    prices_cache = {}  # ticker -> (dates_list, prices_array)
    skipped_tickers = []
    if evaluator_base.prices_file is not None:
        for ticker in tqdm(list(features_cache.keys()), desc="  Loading prices"):
            if ticker not in evaluator_base.prices_file:
                continue
            try:
                dates_bytes = evaluator_base.prices_file[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                prices = evaluator_base.prices_file[ticker]['prices'][:].astype(np.float32)

                # Data quality filter: skip stocks with unreasonable prices
                # (likely unadjusted for stock splits)
                max_price = prices.max()
                if max_price > 10000:  # $10,000 max reasonable price
                    skipped_tickers.append((ticker, max_price))
                    continue

                prices_cache[ticker] = (dates, prices)
            except Exception:
                continue

    print(f"  Loaded {len(features_cache)} tickers into RAM")
    if skipped_tickers:
        print(f"  Skipped {len(skipped_tickers)} tickers with suspicious prices (>$10,000)")

    # Build date index for fast lookup
    date_to_idx = {d: i for i, d in enumerate(evaluator_base.all_dates)}

    # Storage for all sweep results
    # Key: (horizon_idx, top_k), Value: list of per-fold metrics
    config_fold_metrics = defaultdict(list)

    # Time-series tracking for the primary config (first horizon, middle top_k)
    # This will be used for IC time-series plotting
    primary_horizon = args.sweep_horizons[0] if args.sweep_horizons else 1
    primary_top_k = args.sweep_top_ks[len(args.sweep_top_ks)//2] if args.sweep_top_ks else 10
    all_time_series = []  # Collect across folds

    # Process each fold - single forward pass per fold
    for checkpoint_path in checkpoint_files:
        fold_name = os.path.basename(checkpoint_path)
        print(f"\n{'='*70}")
        print(f"Processing {fold_name}")
        print(f"{'='*70}")

        model, checkpoint = evaluator_base.load_model(checkpoint_path)
        fold_idx = checkpoint.get('fold_idx', 0)

        # Get test dates
        test_dates = checkpoint.get('test_dates_list')
        if test_dates is None:
            test_range = checkpoint.get('test_dates')
            if test_range is not None:
                test_start, test_end = test_range
                test_dates = [d for d in evaluator_base.all_dates if test_start <= d <= test_end]
            else:
                print(f"  Skipping fold - no test dates found")
                continue

        bin_edges = checkpoint.get('bin_edges')
        if bin_edges is not None:
            bin_edges = bin_edges.to(args.device)

        # ========================================
        # GPU WARMUP - ensure CUDA is fully initialized
        # ========================================
        if args.device == 'cuda' and torch.cuda.is_available():
            # Warmup forward pass with dummy data
            dummy_features = [torch.randn(evaluator_base.input_dim) for _ in range(32)]
            _ = evaluator_base.predict_batch_all_horizons(model, dummy_features, bin_edges)
            torch.cuda.synchronize()

        # ========================================
        # STEP 1: Batch inference using pre-loaded RAM data
        # ========================================
        print(f"  Running inference for {len(test_dates)} test dates...")

        all_date_data = []

        for date in tqdm(test_dates, desc=f"  Inference", leave=False):
            if date not in date_to_idx:
                continue
            date_idx = date_to_idx[date]
            if date_idx < evaluator_base.seq_len:
                continue

            # Collect features and returns from pre-loaded cache
            features_list = []
            actual_returns_by_horizon = {h: [] for h in args.sweep_horizons}
            momentum_signals = []

            for ticker in features_cache.keys():
                feat_dates, feat_array = features_cache[ticker]

                # Find date index in this ticker's data
                try:
                    ticker_date_idx = feat_dates.index(date)
                except ValueError:
                    continue

                if ticker_date_idx >= feat_array.shape[0]:
                    continue

                # Need seq_len days of history ending at ticker_date_idx
                start_idx = ticker_date_idx - evaluator_base.seq_len + 1
                if start_idx < 0:
                    continue

                # Get full sequence of features (already in RAM as numpy)
                features = feat_array[start_idx:ticker_date_idx + 1]  # Shape: (seq_len, features)

                # Get returns for each horizon from prices cache
                if ticker not in prices_cache:
                    continue

                price_dates, price_array = prices_cache[ticker]
                try:
                    price_idx = price_dates.index(date)
                except ValueError:
                    continue

                # Check all horizons have valid returns
                returns_valid = True
                ticker_returns = {}
                for horizon_idx in args.sweep_horizons:
                    horizon_days = horizon_map[horizon_idx]
                    future_idx = price_idx + horizon_days
                    if future_idx >= len(price_array):
                        returns_valid = False
                        break
                    current_price = price_array[price_idx]
                    future_price = price_array[future_idx]
                    if current_price > 0:
                        ret = (future_price / current_price) - 1.0
                        # Filter extreme returns (>100% gain or >50% loss)
                        # These are likely data errors or extreme events that skew results
                        if ret > 1.0 or ret < -0.5:
                            returns_valid = False
                            break
                        ticker_returns[horizon_idx] = ret
                    else:
                        returns_valid = False
                        break

                if not returns_valid:
                    continue

                # Momentum (20-day lookback)
                lookback = 20
                past_idx = price_idx - lookback
                if past_idx >= 0 and price_array[past_idx] > 0:
                    momentum = (price_array[price_idx] / price_array[past_idx]) - 1.0
                else:
                    momentum = 0.0

                # Convert to tensor (will be batched and moved to GPU)
                features_list.append(torch.from_numpy(features))
                for h in args.sweep_horizons:
                    actual_returns_by_horizon[h].append(ticker_returns[h])
                momentum_signals.append(momentum)

            if len(features_list) < max(args.sweep_top_ks):
                continue

            # Get ALL-horizon predictions in ONE forward pass (batched on GPU)
            all_horizon_preds = evaluator_base.predict_batch_all_horizons(
                model, features_list, bin_edges
            )  # Shape: (num_stocks, 4)

            if all_horizon_preds is None:
                continue

            # Store data for this date (predictions are already numpy on CPU)
            all_date_data.append({
                'date': date,
                'predictions': all_horizon_preds,
                'actual_returns': {h: np.array(actual_returns_by_horizon[h], dtype=np.float32) for h in args.sweep_horizons},
                'momentum': np.array(momentum_signals, dtype=np.float32),
            })

        print(f"  Collected data for {len(all_date_data)} valid dates")

        # ========================================
        # STEP 2: CPU-only evaluation for each config (horizon_idx, top_k)
        # ========================================
        print(f"  Evaluating {len(args.sweep_horizons) * len(args.sweep_top_ks)} configurations (CPU-only)...")

        for horizon_idx in args.sweep_horizons:
            horizon_days = horizon_map[horizon_idx]

            # Use non-overlapping dates for this horizon
            # Select every horizon_days-th date
            eval_indices = list(range(0, len(all_date_data), horizon_days))
            eval_data = [all_date_data[i] for i in eval_indices]

            if len(eval_data) < 5:
                print(f"    Skipping horizon={horizon_days}d: only {len(eval_data)} non-overlapping dates")
                continue

            for top_k in args.sweep_top_ks:
                # CPU-only computation of metrics for this config
                daily_ics = []
                daily_rank_ics = []
                daily_model_returns = []
                daily_momentum_returns = []
                daily_random_returns = []
                daily_top_decile_returns = []
                daily_bottom_decile_returns = []
                daily_dates = []  # Track dates for time-series plotting

                capital = 100000.0

                # Track if this is the primary config for time-series plotting
                is_primary_config = (horizon_idx == primary_horizon and top_k == primary_top_k)

                for data in eval_data:
                    preds = data['predictions'][:, horizon_idx]  # Select horizon column
                    actual = data['actual_returns'][horizon_idx]
                    momentum = data['momentum']

                    if len(preds) < top_k * 2:
                        continue

                    # IC (Pearson)
                    if np.std(preds) > 1e-8 and np.std(actual) > 1e-8:
                        ic, _ = pearsonr(preds, actual)
                        rank_ic, _ = spearmanr(preds, actual)
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                            daily_dates.append(data['date'])  # Track date for this IC
                        if not np.isnan(rank_ic):
                            daily_rank_ics.append(rank_ic)

                    # Quantile analysis
                    sorted_indices = np.argsort(preds)[::-1]
                    n_per_decile = len(sorted_indices) // 10

                    if n_per_decile > 0:
                        top_decile_idx = sorted_indices[:n_per_decile]
                        bottom_decile_idx = sorted_indices[-n_per_decile:]
                        daily_top_decile_returns.append(actual[top_decile_idx].mean())
                        daily_bottom_decile_returns.append(actual[bottom_decile_idx].mean())

                    # Top-k selection (model)
                    top_k_idx = sorted_indices[:top_k]
                    model_return = actual[top_k_idx].mean()
                    daily_model_returns.append(model_return)

                    # Momentum baseline
                    momentum_sorted = np.argsort(momentum)[::-1][:top_k]
                    momentum_return = actual[momentum_sorted].mean()
                    daily_momentum_returns.append(momentum_return)

                    # Random baseline
                    random_idx = np.random.choice(len(actual), top_k, replace=False)
                    random_return = actual[random_idx].mean()
                    daily_random_returns.append(random_return)

                    capital *= (1 + model_return)

                if len(daily_model_returns) < 3:
                    continue

                # Compute aggregate metrics (all CPU ops)
                daily_ics = np.array(daily_ics)
                daily_model_returns = np.array(daily_model_returns)
                daily_momentum_returns = np.array(daily_momentum_returns)
                daily_random_returns = np.array(daily_random_returns)
                daily_top_decile_returns = np.array(daily_top_decile_returns)
                daily_bottom_decile_returns = np.array(daily_bottom_decile_returns)

                mean_ic = np.mean(daily_ics) if len(daily_ics) > 0 else 0
                std_ic = np.std(daily_ics) if len(daily_ics) > 0 else 1
                ir = mean_ic / std_ic if std_ic > 1e-8 else 0

                top_decile_ret = np.mean(daily_top_decile_returns) * 100 if len(daily_top_decile_returns) > 0 else 0
                bottom_decile_ret = np.mean(daily_bottom_decile_returns) * 100 if len(daily_bottom_decile_returns) > 0 else 0
                spread = top_decile_ret - bottom_decile_ret

                model_mean = np.mean(daily_model_returns) * 100
                momentum_mean = np.mean(daily_momentum_returns) * 100
                random_mean = np.mean(daily_random_returns) * 100
                excess_vs_random = model_mean - random_mean
                excess_vs_momentum = model_mean - momentum_mean

                # Simulation metrics
                sim_total_return = (capital / 100000.0 - 1) * 100
                trades_per_year = 252 / horizon_days
                if np.std(daily_model_returns) > 1e-8:
                    sharpe = (np.mean(daily_model_returns) / np.std(daily_model_returns)) * np.sqrt(trades_per_year)
                else:
                    sharpe = 0

                # Store metrics for this fold/config
                config_fold_metrics[(horizon_idx, top_k)].append({
                    'fold_idx': fold_idx,
                    'mean_ic': mean_ic,
                    'ir': ir,
                    'spread': spread,
                    'model_return': model_mean,
                    'excess_vs_random': excess_vs_random,
                    'excess_vs_momentum': excess_vs_momentum,
                    'sim_return': sim_total_return,
                    'sharpe': sharpe,
                    'num_trades': len(daily_model_returns),
                })

                # Store time-series data for the primary config (for plotting)
                if is_primary_config:
                    all_time_series.append({
                        'dates': daily_dates,
                        'daily_ics': daily_ics.tolist() if isinstance(daily_ics, np.ndarray) else list(daily_ics),
                        'daily_model_returns': daily_model_returns.tolist() if isinstance(daily_model_returns, np.ndarray) else list(daily_model_returns),
                        'daily_random_returns': daily_random_returns.tolist() if isinstance(daily_random_returns, np.ndarray) else list(daily_random_returns),
                        'daily_momentum_returns': daily_momentum_returns.tolist() if isinstance(daily_momentum_returns, np.ndarray) else list(daily_momentum_returns),
                    })

    # ========================================
    # STEP 3: Aggregate results across folds for each config
    # ========================================
    all_sweep_results = []

    for (horizon_idx, top_k), fold_metrics in config_fold_metrics.items():
        horizon_days = horizon_map[horizon_idx]

        config_summary = {
            'horizon_days': horizon_days,
            'horizon_idx': horizon_idx,
            'top_k': top_k,
            'num_folds': len(fold_metrics),
            'mean_ic': float(np.mean([m['mean_ic'] for m in fold_metrics])),
            'mean_ir': float(np.mean([m['ir'] for m in fold_metrics])),
            'mean_spread': float(np.mean([m['spread'] for m in fold_metrics])),
            'mean_model_return': float(np.mean([m['model_return'] for m in fold_metrics])),
            'mean_excess_vs_random': float(np.mean([m['excess_vs_random'] for m in fold_metrics])),
            'mean_excess_vs_momentum': float(np.mean([m['excess_vs_momentum'] for m in fold_metrics])),
            'mean_sim_return': float(np.mean([m['sim_return'] for m in fold_metrics])),
            'mean_sharpe': float(np.mean([m['sharpe'] for m in fold_metrics])),
            'total_trades': int(sum([m['num_trades'] for m in fold_metrics])),
        }
        all_sweep_results.append(config_summary)

    # Sort by horizon, then top_k
    all_sweep_results.sort(key=lambda x: (x['horizon_idx'], x['top_k']))

    # Print sweep summary table
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Horizon':<10} {'Top-K':<8} {'IC':<10} {'IR':<10} {'Spread':<10} {'Excess':<10} {'Sharpe':<10}")
    print("-" * 70)

    for r in all_sweep_results:
        print(f"{r['horizon_days']}d{'':<7} {r['top_k']:<8} {r['mean_ic']:>+.4f}   {r['mean_ir']:>+.4f}   "
              f"{r['mean_spread']:>+.3f}%   {r['mean_excess_vs_random']:>+.3f}%   {r['mean_sharpe']:>+.2f}")

    # Find best configs
    print(f"\n{'='*70}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*70}")

    if len(all_sweep_results) > 0:
        best_ic = max(all_sweep_results, key=lambda x: x['mean_ic'])
        best_sharpe = max(all_sweep_results, key=lambda x: x['mean_sharpe'])
        best_spread = max(all_sweep_results, key=lambda x: x['mean_spread'])

        print(f"\n  Best IC:     {best_ic['horizon_days']}d, top-{best_ic['top_k']} (IC={best_ic['mean_ic']:.4f})")
        print(f"  Best Sharpe: {best_sharpe['horizon_days']}d, top-{best_sharpe['top_k']} (Sharpe={best_sharpe['mean_sharpe']:.2f})")
        print(f"  Best Spread: {best_spread['horizon_days']}d, top-{best_spread['top_k']} (Spread={best_spread['mean_spread']:.3f}%)")

    return {'sweep_results': all_sweep_results, 'time_series': all_time_series}


def plot_sweep_results(sweep_results: List[Dict], output_dir: str):
    """Generate heatmaps for sweep results."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique horizons and top-ks
    horizons = sorted(set(r['horizon_days'] for r in sweep_results))
    top_ks = sorted(set(r['top_k'] for r in sweep_results))

    # Create matrices for each metric
    metrics = ['mean_ic', 'mean_sharpe', 'mean_spread', 'mean_excess_vs_random']
    titles = ['Information Coefficient', 'Sharpe Ratio', 'Long-Short Spread (%)', 'Excess vs Random (%)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # Build matrix
        matrix = np.zeros((len(horizons), len(top_ks)))
        for r in sweep_results:
            i = horizons.index(r['horizon_days'])
            j = top_ks.index(r['top_k'])
            matrix[i, j] = r[metric]

        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(top_ks)))
        ax.set_xticklabels([f'k={k}' for k in top_ks])
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels([f'{h}d' for h in horizons])
        ax.set_xlabel('Top-K')
        ax.set_ylabel('Horizon')
        ax.set_title(title)

        # Add text annotations
        for i in range(len(horizons)):
            for j in range(len(top_ks)):
                val = matrix[i, j]
                color = 'white' if abs(val) > (matrix.max() - matrix.min()) / 2 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)

        plt.colorbar(im, ax=ax)

    plt.suptitle('Configuration Sweep Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'sweep_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSweep plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Principled Evaluation of Walk-Forward Checkpoints')

    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing fold checkpoints')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset')
    parser.add_argument('--prices', type=str, default=None,
                       help='Path to prices HDF5 file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for inference')
    parser.add_argument('--max-stocks', type=int, default=500,
                       help='Max stocks to evaluate per day')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--multi-trial', type=int, default=0,
                       help='Run N multi-trial simulations per fold (0=skip)')
    parser.add_argument('--subsample-ratio', type=float, default=0.7,
                       help='Fraction of stocks to use per trial in multi-trial mode')

    # Manual overrides for old checkpoint format (without saved config)
    parser.add_argument('--seq-len', type=int, default=None,
                       help='Override sequence length (for old checkpoints without config)')
    parser.add_argument('--horizon-days', type=int, default=None,
                       help='Override horizon days (for old checkpoints without config)')
    parser.add_argument('--horizon-idx', type=int, default=None,
                       help='Override horizon index (for old checkpoints without config)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Override top-k (for old checkpoints without config)')

    # Sweep mode: evaluate across multiple configurations
    parser.add_argument('--sweep', action='store_true',
                       help='Run sweep across all horizons and multiple top-k values')
    parser.add_argument('--sweep-top-ks', type=int, nargs='+', default=[5, 10, 15, 20, 25],
                       help='Top-k values to sweep (default: 5 10 15 20 25)')
    parser.add_argument('--sweep-horizons', type=int, nargs='+', default=[0, 1, 2, 3],
                       help='Horizon indices to sweep (0=1d, 1=5d, 2=10d, 3=20d)')

    # Monte Carlo validation mode
    parser.add_argument('--monte-carlo', action='store_true',
                       help='Run Monte Carlo validation with many random stock subsets')
    parser.add_argument('--mc-trials', type=int, default=100,
                       help='Number of Monte Carlo trials (default: 100)')
    parser.add_argument('--mc-stocks', type=int, default=150,
                       help='Number of stocks to sample per trial (default: 150)')
    parser.add_argument('--mc-top-ks', type=int, nargs='+', default=[5, 10, 15, 20],
                       help='Top-k values to test in Monte Carlo (default: 5 10 15 20)')
    parser.add_argument('--mc-seed', type=int, default=None,
                       help='Random seed for Monte Carlo reproducibility')

    args = parser.parse_args()

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(args.checkpoint_dir, 'fold_*_best.pt')
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))

    if len(checkpoint_files) == 0:
        print(f"No checkpoints found matching {checkpoint_pattern}")
        return

    print(f"Found {len(checkpoint_files)} checkpoints in {args.checkpoint_dir}")

    # Load first checkpoint to get config
    first_checkpoint = torch.load(checkpoint_files[0], map_location='cpu', weights_only=False)
    wf_config = first_checkpoint.get('walk_forward_config', {})

    # Check if old format (no walk_forward_config)
    is_old_format = len(wf_config) == 0
    if is_old_format:
        print("\n⚠️  Old checkpoint format detected (no walk_forward_config)")
        print("   Using command-line arguments or defaults for evaluation config")

    # Use command-line overrides if provided, else checkpoint config, else defaults
    seq_len = args.seq_len or wf_config.get('seq_len', 60)
    horizon_days = args.horizon_days or wf_config.get('horizon_days', 5)
    horizon_idx = args.horizon_idx or wf_config.get('horizon_idx', 1)
    top_k = args.top_k or wf_config.get('top_k', 10)

    # Initialize evaluator
    evaluator = PrincipledEvaluator(
        dataset_path=args.data,  # Always use CLI arg for data paths
        prices_path=args.prices,
        seq_len=seq_len,
        horizon_days=horizon_days,
        horizon_idx=horizon_idx,
        top_k=top_k,
        device=args.device,
    )

    print(f"\nEvaluation config:")
    print(f"  Horizon: {evaluator.horizon_days} days (idx={evaluator.horizon_idx})")
    print(f"  Top-K: {evaluator.top_k}")
    print(f"  Seq len: {evaluator.seq_len}")
    if is_old_format:
        print(f"  ⚠️  Test dates will be inferred from checkpoint date range")

    # ========================================
    # SWEEP MODE: Evaluate all horizon/top-k combinations efficiently
    # ========================================
    if args.sweep:
        sweep_results = run_sweep(
            checkpoint_files=checkpoint_files,
            evaluator_base=evaluator,
            args=args,
            wf_config=wf_config,
        )

        # Save sweep results
        output_dir = args.output or args.checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)

        # Plot sweep results
        if len(sweep_results.get('sweep_results', [])) > 0:
            plot_sweep_results(sweep_results['sweep_results'], output_dir)

            # Plot IC time-series
            time_series = sweep_results.get('time_series', [])
            if len(time_series) > 0:
                ic_plot_path = os.path.join(output_dir, 'ic_time_series.png')
                plot_ic_time_series(time_series, ic_plot_path)

            # Save JSON (exclude time_series from JSON to keep it small)
            sweep_json_path = os.path.join(output_dir, 'sweep_results.json')
            json_data = {'sweep_results': sweep_results['sweep_results']}
            with open(sweep_json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"\nSweep results saved to: {sweep_json_path}")

        return

    # ========================================
    # STANDARD MODE: Evaluate with single config
    # ========================================

    # Verify and report device
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"\n⚠️  CUDA requested but not available, falling back to CPU")
            args.device = 'cpu'
            evaluator.device = 'cpu'
    else:
        print(f"\nUsing device: {args.device}")

    # Evaluate each fold
    results = []
    all_time_series = []  # Collect time-series data for plotting

    for checkpoint_path in checkpoint_files:
        print(f"\nLoading {os.path.basename(checkpoint_path)}...")

        model, checkpoint = evaluator.load_model(checkpoint_path)

        # Get test dates from checkpoint
        test_dates = checkpoint.get('test_dates_list')
        if test_dates is None:
            # Fall back to date range (old format)
            test_range = checkpoint.get('test_dates')
            if test_range is not None:
                test_start, test_end = test_range
                test_dates = [d for d in evaluator.all_dates if test_start <= d <= test_end]
                print(f"    (Inferred {len(test_dates)} test dates from range {test_start} to {test_end})")
            else:
                print(f"    ⚠️  No test_dates in checkpoint, skipping fold {checkpoint.get('fold_idx', '?')}")
                continue

        bin_edges = checkpoint.get('bin_edges')
        if bin_edges is not None:
            bin_edges = bin_edges.to(args.device)

        # GPU warmup - ensure CUDA is fully initialized before timing
        if args.device == 'cuda' and torch.cuda.is_available():
            dummy_features = [torch.randn(evaluator.input_dim) for _ in range(32)]
            _ = evaluator.predict_batch_all_horizons(model, dummy_features, bin_edges)
            torch.cuda.synchronize()

        result, time_series = evaluator.evaluate_fold(
            model=model,
            checkpoint=checkpoint,
            test_dates=test_dates,
            bin_edges=bin_edges,
            batch_size=args.batch_size,
            max_stocks_per_day=args.max_stocks,
        )

        results.append(result)
        all_time_series.append(time_series)
        print_fold_results(result)

        # Run multi-trial simulation if requested
        if args.multi_trial > 0:
            multi_trial_result = run_multi_trial_simulation(
                evaluator=evaluator,
                model=model,
                checkpoint=checkpoint,
                test_dates=test_dates,
                bin_edges=bin_edges,
                n_trials=args.multi_trial,
                subsample_ratio=args.subsample_ratio,
                batch_size=args.batch_size,
                max_stocks=args.max_stocks,
            )
            # Store with result for later
            result.multi_trial = multi_trial_result

        # Run Monte Carlo validation if requested
        if args.monte_carlo:
            mc_result = run_monte_carlo_validation(
                evaluator=evaluator,
                model=model,
                checkpoint=checkpoint,
                test_dates=test_dates,
                bin_edges=bin_edges,
                n_trials=args.mc_trials,
                stocks_per_trial=args.mc_stocks,
                top_k_values=args.mc_top_ks,
                batch_size=args.batch_size,
                max_stocks=args.max_stocks,
                seed=args.mc_seed,
            )
            # Store with result for later
            result.monte_carlo = mc_result

            # Save Monte Carlo plots to special directory
            mc_output_dir = os.path.join(args.output or args.checkpoint_dir, 'monte_carlo')
            fold_idx = checkpoint.get('fold_idx', len(results) - 1)
            plot_monte_carlo_results(mc_result, mc_output_dir, fold_idx=fold_idx)

    # Aggregate results
    print_aggregate_results(results)

    # Pooled analysis (combines all daily returns for more power)
    if len(all_time_series) > 0:
        print_pooled_analysis(all_time_series)

    # Save and plot
    output_dir = args.output or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_results(results, output_dir)

    # Plot IC time-series
    if len(all_time_series) > 0:
        ic_plot_path = os.path.join(output_dir, 'ic_time_series.png')
        plot_ic_time_series(all_time_series, ic_plot_path)

    # Save JSON results
    fold_results_dicts = []
    for r in results:
        rd = asdict(r)
        # Add multi-trial results if present
        if hasattr(r, 'multi_trial'):
            rd['multi_trial'] = r.multi_trial
        # Add Monte Carlo results if present
        if hasattr(r, 'monte_carlo'):
            rd['monte_carlo'] = r.monte_carlo
        fold_results_dicts.append(rd)

    results_dict = {
        'num_folds': len(results),
        'config': wf_config,
        'fold_results': fold_results_dicts,
        'aggregate': {
            'mean_ic': float(np.mean([r.mean_ic for r in results])),
            'mean_ir': float(np.mean([r.ir for r in results])),
            'mean_spread': float(np.mean([r.long_short_spread for r in results])),
            'mean_excess_vs_random': float(np.mean([r.excess_vs_random for r in results])),
            'mean_sim_return': float(np.mean([r.simulation_total_return for r in results])),
            'mean_analytical_return': float(np.mean([r.analytical_expected_annual_return for r in results])),
            'mean_bootstrap_prob_positive': float(np.mean([r.bootstrap_prob_positive for r in results])),
        }
    }

    json_path = os.path.join(output_dir, 'principled_evaluation.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to: {json_path}")


if __name__ == '__main__':
    main()
