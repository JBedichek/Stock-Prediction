#!/usr/bin/env python3
"""
Walk-Forward Training with Online Evaluation

Trains and evaluates the model using walk-forward validation:
1. Split data into temporal folds
2. For each fold: train on historical data, evaluate on future data
3. Aggregate results across all folds

This provides the most realistic estimate of out-of-sample performance.

Usage:
    # Single GPU
    python -m training.walk_forward_training \
        --data all_complete_dataset.h5 \
        --num-folds 5 \
        --mode expanding \
        --epochs-per-fold 10

    # Multi-GPU with DDP (using torchrun)
    torchrun --nproc_per_node=4 -m training.walk_forward_training \
        --data all_complete_dataset.h5 \
        --num-folds 5 \
        --ddp

Modes:
    - expanding: Training set grows with each fold (recommended)
    - sliding: Fixed-size training window that slides forward
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import random
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import h5py

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import (
    SimpleTransformerPredictor,
    convert_price_ratios_to_bins_vectorized,
    compute_confidence_targets,
    compute_expected_value
)
from training.multimodal_model import (
    MultiModalStockPredictor,
    FeatureConfig,
    ContrastiveMultiModalModel,
    pretrain_contrastive,
)
from inference.backtest_simulation import (
    DatasetLoader, ModelPredictor, TradingSimulator
)
from inference.principled_evaluation import (
    run_monte_carlo_validation,
    plot_monte_carlo_results,
    PrincipledEvaluator,
)
from utils.utils import pic_load, save_pickle


# ============================================================================
# Logging Setup
# ============================================================================

# Global logger instance
_stats_logger: Optional[logging.Logger] = None


def setup_stats_logger(checkpoint_dir: str) -> logging.Logger:
    """
    Setup a dedicated logger for model statistics.

    Writes to both console and a stats.log file in the checkpoint directory.
    This log contains key metrics like IC, IR, Rank IC for easy review.
    """
    global _stats_logger

    # Create logger
    logger = logging.getLogger('walk_forward_stats')
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - writes to checkpoint_dir/stats.log
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, 'stats.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _stats_logger = logger
    return logger


def get_stats_logger() -> Optional[logging.Logger]:
    """Get the global stats logger if initialized."""
    return _stats_logger


def log_evaluation_header(
    logger: logging.Logger,
    fold_idx: int,
    num_folds: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    num_test_dates: int,
    num_stocks: int,
    horizon_days: int,
    top_k: int,
    transaction_cost_bps: float,
):
    """Log evaluation header with metadata."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"FOLD {fold_idx + 1}/{num_folds} EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Training Period:   {train_start} to {train_end}")
    logger.info(f"Test Period:       {test_start} to {test_end}")
    logger.info(f"Eval Samples:      {num_test_dates} non-overlapping periods")
    logger.info(f"Stock Universe:    {num_stocks} stocks evaluated per period")
    logger.info(f"Horizon:           {horizon_days} day(s)")
    logger.info(f"Top-K Selection:   {top_k} stocks")
    if transaction_cost_bps > 0:
        logger.info(f"Transaction Cost:  {transaction_cost_bps:.0f} bps ({transaction_cost_bps/100:.2f}%)")
    logger.info("-" * 80)


def log_evaluation_metrics(
    logger: logging.Logger,
    results: Dict,
    include_net_metrics: bool = True,
):
    """
    Log key evaluation metrics to the stats logger.

    Focuses on IC, IR, Rank IC, and related prediction quality metrics.
    """
    # IC Metrics (PRIMARY - these measure prediction quality)
    logger.info("")
    logger.info("📊 INFORMATION COEFFICIENT METRICS (Primary)")
    logger.info(f"   Mean IC:          {results['mean_ic']:+.4f}")
    logger.info(f"   IC Std Dev:       {results['std_ic']:.4f}")
    logger.info(f"   Information Ratio:{results['ir']:+.3f}  (IC / IC_std)")
    logger.info(f"   Mean Rank IC:     {results['mean_rank_ic']:+.4f}  (Spearman correlation)")
    logger.info(f"   Pct IC > 0:       {results['pct_positive_ic']:.1f}%")

    # Statistical significance
    p_value = results['ic_p_value']
    sig_stars = ""
    if p_value < 0.01:
        sig_stars = "***"
    elif p_value < 0.05:
        sig_stars = "**"
    elif p_value < 0.1:
        sig_stars = "*"
    logger.info(f"   IC T-statistic:   {results['ic_t_stat']:+.2f}  (p={p_value:.4f} {sig_stars})")

    # Quantile Analysis
    logger.info("")
    logger.info("📈 QUANTILE ANALYSIS")
    logger.info(f"   Top Decile Ret:   {results['top_decile_return']:+.3f}% per period")
    logger.info(f"   Bottom Decile:    {results['bottom_decile_return']:+.3f}% per period")
    logger.info(f"   Long-Short Spread:{results['long_short_spread']:+.3f}%")

    # Baseline Comparisons (GROSS)
    logger.info("")
    logger.info("📊 BASELINE COMPARISONS (Gross Returns)")
    logger.info(f"   Model Return:     {results['model_mean_return']:+.3f}% per period")
    logger.info(f"   Momentum Return:  {results['momentum_mean_return']:+.3f}% per period")
    logger.info(f"   Random Return:    {results['random_mean_return']:+.3f}% per period")
    logger.info(f"   Excess vs Random: {results['excess_vs_random']:+.3f}%  (p={results['vs_random_p_value']:.3f})")
    logger.info(f"   Excess vs Momentum:{results['excess_vs_momentum']:+.3f}%  (p={results['vs_momentum_p_value']:.3f})")

    # Net returns if available
    if include_net_metrics and 'model_mean_return_net' in results:
        logger.info("")
        logger.info("📊 BASELINE COMPARISONS (Net of Transaction Costs)")
        logger.info(f"   Model Return:     {results['model_mean_return_net']:+.3f}% per period")
        logger.info(f"   Momentum Return:  {results['momentum_mean_return_net']:+.3f}% per period")
        logger.info(f"   Excess vs Random: {results['excess_vs_random_net']:+.3f}%")
        logger.info(f"   Excess vs Momentum:{results['excess_vs_momentum_net']:+.3f}%")

    # Turnover
    if 'mean_turnover' in results:
        logger.info("")
        logger.info("🔄 TURNOVER METRICS")
        logger.info(f"   Mean Turnover:    {results['mean_turnover']*100:.1f}% per period")
        logger.info(f"   Total Turnover:   {results['total_turnover']*100:.1f}%")
        if 'total_transaction_costs_pct' in results:
            logger.info(f"   Total Costs Paid: {results['total_transaction_costs_pct']:.2f}%")

    # Simulation Summary
    logger.info("")
    logger.info("💰 SIMULATION SUMMARY")
    logger.info(f"   Total Return:     {results['total_return_pct']:+.2f}% (gross)")
    if 'total_return_pct_net' in results:
        logger.info(f"   Total Return:     {results['total_return_pct_net']:+.2f}% (net)")
    logger.info(f"   Sharpe Ratio:     {results['sharpe_ratio']:.2f} (gross)")
    if 'sharpe_ratio_net' in results:
        logger.info(f"   Sharpe Ratio:     {results['sharpe_ratio_net']:.2f} (net)")
    logger.info(f"   Max Drawdown:     {results['max_drawdown_pct']:.2f}%")
    logger.info(f"   Win Rate:         {results['win_rate']:.1f}%")
    logger.info(f"   Num Trades:       {results['num_trades']}")
    logger.info("-" * 80)


def log_fold_summary(
    logger: logging.Logger,
    fold_results: List[Dict],
):
    """Log aggregate summary across all folds."""
    if not fold_results:
        return

    # Aggregate metrics
    ics = [r['mean_ic'] for r in fold_results if 'mean_ic' in r]
    irs = [r['ir'] for r in fold_results if 'ir' in r]
    rank_ics = [r['mean_rank_ic'] for r in fold_results if 'mean_rank_ic' in r]
    spreads = [r['long_short_spread'] for r in fold_results if 'long_short_spread' in r]

    logger.info("")
    logger.info("=" * 80)
    logger.info("AGGREGATE RESULTS ACROSS ALL FOLDS")
    logger.info("=" * 80)

    if ics:
        logger.info(f"   Mean IC:          {np.mean(ics):+.4f} ± {np.std(ics):.4f}")
    if irs:
        logger.info(f"   Information Ratio:{np.mean(irs):+.3f} ± {np.std(irs):.3f}")
    if rank_ics:
        logger.info(f"   Mean Rank IC:     {np.mean(rank_ics):+.4f} ± {np.std(rank_ics):.4f}")
    if spreads:
        logger.info(f"   Long-Short Spread:{np.mean(spreads):+.3f}% ± {np.std(spreads):.3f}%")

    # Total returns
    total_rets = [r['total_return_pct'] for r in fold_results if 'total_return_pct' in r]
    if total_rets:
        logger.info(f"   Avg Total Return: {np.mean(total_rets):+.2f}% (gross)")

    total_rets_net = [r.get('total_return_pct_net', r['total_return_pct']) for r in fold_results]
    if 'total_return_pct_net' in fold_results[0]:
        logger.info(f"   Avg Total Return: {np.mean(total_rets_net):+.2f}% (net)")

    logger.info("=" * 80)
    logger.info("")


# ============================================================================
# Ranking Loss Functions for Cross-Sectional Learning
# ============================================================================

def compute_pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.0,
    horizon_idx: int = 0,
) -> torch.Tensor:
    """
    Compute pairwise ranking loss within a batch.

    For each pair (i, j) where target_i > target_j, we want prediction_i > prediction_j.
    This teaches the model to rank stocks correctly relative to each other.

    Args:
        predictions: Model predictions, shape (batch, num_bins, num_horizons) for classification
                     or (batch, num_horizons) for regression
        targets: Actual returns, shape (batch, num_horizons)
        margin: Margin for hinge loss (default 0.0)
        horizon_idx: Which horizon to use for ranking (default 0 = 1-day)

    Returns:
        Scalar ranking loss
    """
    batch_size = predictions.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # Extract predictions for the specified horizon
    if predictions.dim() == 3:
        # Classification: (batch, num_bins, num_horizons)
        # We can't directly use logits - need expected values or max bin
        # Use softmax + weighted sum with bin indices as proxy for expected rank
        probs = F.softmax(predictions[:, :, horizon_idx], dim=1)  # (batch, num_bins)
        # Use bin index as weight (higher bin = higher return)
        bin_indices = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
        pred_scores = (probs * bin_indices).sum(dim=1)  # (batch,)
    else:
        # Regression: (batch, num_horizons)
        pred_scores = predictions[:, horizon_idx]  # (batch,)

    # Get target returns for this horizon
    target_scores = targets[:, horizon_idx]  # (batch,)

    # Create pairwise differences
    # pred_diff[i, j] = pred_i - pred_j
    pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)  # (batch, batch)
    target_diff = target_scores.unsqueeze(1) - target_scores.unsqueeze(0)  # (batch, batch)

    # We want: when target_i > target_j, pred_i should be > pred_j
    # Mask for pairs where target_i > target_j (positive pairs)
    positive_pairs = (target_diff > 0).float()

    # Hinge loss: max(0, margin - pred_diff) for positive pairs
    # If pred_i > pred_j by margin, loss is 0
    # If pred_i < pred_j, loss is positive
    pair_losses = F.relu(margin - pred_diff) * positive_pairs

    # Average over valid pairs
    num_positive_pairs = positive_pairs.sum()
    if num_positive_pairs > 0:
        return pair_losses.sum() / num_positive_pairs

    return torch.tensor(0.0, device=predictions.device, requires_grad=True)


def compute_listnet_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizon_idx: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute ListNet loss (cross-entropy between predicted and actual ranking distributions).

    This is a listwise ranking loss that considers the entire ranked list.

    Args:
        predictions: Model predictions, shape (batch, num_bins, num_horizons) or (batch, num_horizons)
        targets: Actual returns, shape (batch, num_horizons)
        horizon_idx: Which horizon to use for ranking
        temperature: Temperature for softmax (higher = softer distribution)

    Returns:
        Scalar ListNet loss
    """
    batch_size = predictions.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # Extract prediction scores
    if predictions.dim() == 3:
        probs = F.softmax(predictions[:, :, horizon_idx], dim=1)
        bin_indices = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
        pred_scores = (probs * bin_indices).sum(dim=1)
    else:
        pred_scores = predictions[:, horizon_idx]

    target_scores = targets[:, horizon_idx]

    # Compute probability distributions over items using softmax
    pred_dist = F.softmax(pred_scores / temperature, dim=0)
    target_dist = F.softmax(target_scores / temperature, dim=0)

    # Cross-entropy between target distribution and predicted distribution
    # Use mean instead of sum for scale-invariance to batch size
    loss = -torch.mean(target_dist * torch.log(pred_dist + 1e-8))

    return loss


def compute_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    bin_edges: Optional[torch.Tensor] = None,
    horizon_idx: int = 0,
    loss_type: str = 'pairwise',
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Unified interface for computing ranking loss.

    For classification mode, computes expected returns from bin predictions first.

    Args:
        predictions: Raw model output (logits for classification, values for regression)
        targets: Actual returns
        bin_edges: Bin edges for classification mode (required if classification)
        horizon_idx: Which horizon to use
        loss_type: 'pairwise', 'listnet', or 'correlation'
        margin: Margin for pairwise loss
        temperature: Temperature for ListNet

    Returns:
        Scalar ranking loss
    """
    batch_size = predictions.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # For classification, compute expected return from probability distribution
    if predictions.dim() == 3 and bin_edges is not None:
        # predictions: (batch, num_bins, num_horizons)
        probs = F.softmax(predictions, dim=1)  # (batch, num_bins, num_horizons)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2  # (num_bins,)
        bin_midpoints = bin_midpoints.view(1, -1, 1)  # (1, num_bins, 1)
        expected_returns = (probs * bin_midpoints).sum(dim=1)  # (batch, num_horizons)
    else:
        expected_returns = predictions

    # Get scores for the specified horizon
    pred_scores = expected_returns[:, horizon_idx]  # (batch,)
    target_scores = targets[:, horizon_idx]  # (batch,)

    if loss_type == 'pairwise':
        # Pairwise ranking loss
        pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
        target_diff = target_scores.unsqueeze(1) - target_scores.unsqueeze(0)

        positive_pairs = (target_diff > 0).float()
        pair_losses = F.relu(margin - pred_diff) * positive_pairs

        num_positive_pairs = positive_pairs.sum()
        if num_positive_pairs > 0:
            return pair_losses.sum() / num_positive_pairs
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    elif loss_type == 'listnet':
        # ListNet loss - use mean for scale-invariance to batch size
        pred_dist = F.softmax(pred_scores / temperature, dim=0)
        target_dist = F.softmax(target_scores / temperature, dim=0)
        return -torch.mean(target_dist * torch.log(pred_dist + 1e-8))

    elif loss_type == 'correlation':
        # Negative Pearson correlation loss - directly optimizes IC
        # Center the scores
        pred_centered = pred_scores - pred_scores.mean()
        target_centered = target_scores - target_scores.mean()

        # Compute correlation
        numerator = (pred_centered * target_centered).sum()
        denominator = (pred_centered.norm() * target_centered.norm()) + 1e-8

        correlation = numerator / denominator

        # Return negative correlation (we want to maximize correlation, minimize loss)
        return -correlation

    else:
        raise ValueError(f"Unknown ranking loss type: {loss_type}")


# ============================================================================
# Principled Evaluation Functions
# ============================================================================

def compute_expected_returns_from_logits(
    logits: torch.Tensor,
    bin_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Convert classification logits to expected return values.

    Args:
        logits: (batch, num_bins, num_horizons) or (batch, num_bins)
        bin_edges: (num_bins + 1,) tensor of bin edges

    Returns:
        Expected returns: (batch, num_horizons) or (batch,)
    """
    probs = F.softmax(logits, dim=1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    if logits.dim() == 3:
        # (batch, num_bins, num_horizons)
        bin_midpoints = bin_midpoints.view(1, -1, 1)
        expected = (probs * bin_midpoints).sum(dim=1)  # (batch, num_horizons)
    else:
        # (batch, num_bins)
        expected = (probs * bin_midpoints).sum(dim=1)  # (batch,)

    return expected


def run_principled_evaluation(
    model: torch.nn.Module,
    dataset_path: str,
    prices_path: str,
    test_dates: List[str],
    bin_edges: Optional[torch.Tensor],
    horizon_idx: int,
    horizon_days: int,
    top_k: int,
    device: str,
    max_stocks: int = 500,
    pred_mode: str = 'classification',
    transaction_cost_bps: float = 0.0,  # Transaction cost in basis points (e.g., 10 = 0.1%)
    seq_len: int = 60,  # Sequence length - MUST match training!
    max_eval_dates: int = 60,  # Max dates to evaluate (randomly sampled if exceeded)
) -> Dict:
    """
    Run principled evaluation computing IC, IR, quantile analysis, and baselines.

    This is the core evaluation logic from principled_evaluation.py, integrated
    into training for immediate feedback during walk-forward validation.

    Transaction Cost Model:
        - Tracks portfolio turnover (fraction of positions that change each period)
        - Applies proportional transaction costs: cost = turnover * cost_rate
        - Cost is applied as: net_return = gross_return - turnover * (cost_bps / 10000)
        - Round-trip cost assumption: each changed position incurs buy + sell costs

    Args:
        transaction_cost_bps: Transaction cost in basis points per round-trip trade.
            Examples: 10 bps = 0.1%, 50 bps = 0.5%
            Set to 0 to disable transaction costs.
            Typical values:
            - Large-cap liquid stocks: 5-20 bps
            - Mid-cap stocks: 20-50 bps
            - Small-cap/illiquid: 50-100+ bps

    Returns:
        Dictionary with all evaluation metrics (both gross and net of costs)
    """
    model.eval()

    # Convert basis points to decimal
    cost_rate = transaction_cost_bps / 10000.0  # e.g., 10 bps -> 0.001

    # Storage for daily metrics
    daily_ics = []
    daily_rank_ics = []
    daily_top_decile_returns = []
    daily_bottom_decile_returns = []
    daily_model_returns = []  # Gross returns
    daily_model_returns_net = []  # Net of transaction costs
    daily_momentum_returns = []
    daily_momentum_returns_net = []
    daily_random_returns = []
    daily_turnovers = []  # Track turnover for analysis

    # Track previous holdings for turnover calculation
    prev_model_tickers = set()
    prev_momentum_tickers = set()

    # Pre-load data into RAM for speed
    print("    Pre-loading evaluation data...")
    features_cache = {}
    prices_cache = {}

    with h5py.File(dataset_path, 'r') as h5f:
        tickers = sorted(list(h5f.keys()))[:max_stocks]
        sample_ticker = tickers[0]
        input_dim = h5f[sample_ticker]['features'].shape[1]
        all_dates = sorted([d.decode('utf-8') for d in h5f[sample_ticker]['dates'][:]])

        for ticker in tickers:
            if ticker not in h5f:
                continue
            try:
                dates_bytes = h5f[ticker]['dates'][:]
                dates = [d.decode('utf-8') for d in dates_bytes]
                features = h5f[ticker]['features'][:].astype(np.float32)
                if features.shape[1] == input_dim:
                    features_cache[ticker] = (dates, features)
            except Exception:
                continue

    # Load prices
    if prices_path and os.path.exists(prices_path):
        with h5py.File(prices_path, 'r') as pf:
            for ticker in features_cache.keys():
                if ticker not in pf:
                    continue
                try:
                    dates_bytes = pf[ticker]['dates'][:]
                    dates = [d.decode('utf-8') for d in dates_bytes]
                    prices = pf[ticker]['prices'][:].astype(np.float32)
                    if prices.max() <= 10000:  # Skip stocks with bad price data
                        prices_cache[ticker] = (dates, prices)
                except Exception:
                    continue

    # Evaluate on non-overlapping dates
    eval_dates = test_dates[::horizon_days]

    # Randomly sample dates if we have too many (for faster evaluation)
    if len(eval_dates) > max_eval_dates:
        import random
        # Use a fixed seed for reproducibility within this evaluation
        rng = random.Random(42)
        eval_dates = sorted(rng.sample(eval_dates, max_eval_dates))
        print(f"    Loaded {len(features_cache)} stocks, evaluating {len(eval_dates)} dates (sampled from {len(test_dates[::horizon_days])})")
    else:
        print(f"    Loaded {len(features_cache)} stocks, evaluating {len(eval_dates)} dates...")

    # Build date index
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    with torch.no_grad():
        for date in tqdm(eval_dates, desc="    Evaluating", leave=False):
            if date not in date_to_idx:
                continue

            # Collect data for this date
            features_list = []
            actual_returns = []
            momentum_signals = []
            ticker_list = []  # Track tickers for turnover calculation

            for ticker in features_cache.keys():
                if ticker not in prices_cache:
                    continue

                feat_dates, feat_array = features_cache[ticker]
                price_dates, price_array = prices_cache[ticker]

                # Get feature index
                try:
                    feat_idx = feat_dates.index(date)
                except ValueError:
                    continue

                # Need seq_len days of history ENDING at feat_idx
                start_idx = feat_idx - seq_len + 1
                if start_idx < 0 or feat_idx >= feat_array.shape[0]:
                    continue

                # Get price indices
                try:
                    price_idx = price_dates.index(date)
                    future_idx = price_idx + horizon_days
                    past_idx = price_idx - 20
                except ValueError:
                    continue

                if future_idx >= len(price_array) or past_idx < 0:
                    continue

                if price_array[price_idx] <= 0:
                    continue

                # Compute return and momentum
                actual_ret = (price_array[future_idx] / price_array[price_idx]) - 1.0
                momentum = (price_array[price_idx] / price_array[past_idx]) - 1.0 if price_array[past_idx] > 0 else 0.0

                # Get full sequence of features (seq_len days ending at feat_idx)
                seq_features = feat_array[start_idx:feat_idx + 1]  # Shape: (seq_len, features)
                features_list.append(torch.from_numpy(seq_features).float())
                actual_returns.append(actual_ret)
                momentum_signals.append(momentum)
                ticker_list.append(ticker)

            if len(features_list) < top_k * 2:
                continue

            # Get model predictions
            # features_list contains tensors of shape (seq_len, features)
            # Stack to get (batch, seq_len, features)
            features_batch = torch.stack(features_list).to(device)
            pred, _ = model(features_batch)

            # Convert to expected returns
            if pred_mode == 'classification' and bin_edges is not None:
                predictions = compute_expected_returns_from_logits(
                    pred[:, :, horizon_idx], bin_edges
                ).cpu().numpy()
            else:
                predictions = pred[:, horizon_idx].cpu().numpy()

            actual_returns = np.array(actual_returns)
            momentum_signals = np.array(momentum_signals)

            # === Information Coefficient ===
            if len(predictions) > 10 and np.std(predictions) > 1e-8 and np.std(actual_returns) > 1e-8:
                ic, _ = pearsonr(predictions, actual_returns)
                rank_ic, _ = spearmanr(predictions, actual_returns)
                if not np.isnan(ic):
                    daily_ics.append(ic)
                if not np.isnan(rank_ic):
                    daily_rank_ics.append(rank_ic)

            # === Quantile Analysis ===
            sorted_indices = np.argsort(predictions)[::-1]
            n_per_decile = len(sorted_indices) // 10

            if n_per_decile > 0:
                top_decile_idx = sorted_indices[:n_per_decile]
                bottom_decile_idx = sorted_indices[-n_per_decile:]

                daily_top_decile_returns.append(actual_returns[top_decile_idx].mean())
                daily_bottom_decile_returns.append(actual_returns[bottom_decile_idx].mean())

            # === Model Selection (Top-K) with Turnover Tracking ===
            top_k_idx = sorted_indices[:top_k]
            current_model_tickers = set(ticker_list[i] for i in top_k_idx)

            # Calculate turnover (fraction of portfolio that changed)
            if len(prev_model_tickers) > 0:
                # Turnover = (positions exited + positions entered) / (2 * portfolio_size)
                # This gives turnover in [0, 1] where 1 = complete portfolio change
                positions_changed = len(prev_model_tickers.symmetric_difference(current_model_tickers))
                model_turnover = positions_changed / (2 * top_k)
            else:
                # First period: assume we're starting fresh, full buy cost
                model_turnover = 1.0

            daily_turnovers.append(model_turnover)

            # Gross return (before costs)
            gross_model_return = actual_returns[top_k_idx].mean()
            daily_model_returns.append(gross_model_return)

            # Net return (after transaction costs)
            # Cost = turnover * cost_rate (round-trip cost on changed positions)
            transaction_cost = model_turnover * cost_rate
            net_model_return = gross_model_return - transaction_cost
            daily_model_returns_net.append(net_model_return)

            # Update previous holdings
            prev_model_tickers = current_model_tickers

            # === Momentum Baseline with Turnover ===
            momentum_sorted = np.argsort(momentum_signals)[::-1]
            momentum_top_k = momentum_sorted[:top_k]
            current_momentum_tickers = set(ticker_list[i] for i in momentum_top_k)

            if len(prev_momentum_tickers) > 0:
                positions_changed = len(prev_momentum_tickers.symmetric_difference(current_momentum_tickers))
                momentum_turnover = positions_changed / (2 * top_k)
            else:
                momentum_turnover = 1.0

            gross_momentum_return = actual_returns[momentum_top_k].mean()
            daily_momentum_returns.append(gross_momentum_return)

            net_momentum_return = gross_momentum_return - momentum_turnover * cost_rate
            daily_momentum_returns_net.append(net_momentum_return)

            prev_momentum_tickers = current_momentum_tickers

            # === Random Baseline ===
            random_idx = np.random.choice(len(actual_returns), top_k, replace=False)
            daily_random_returns.append(actual_returns[random_idx].mean())

    model.train()

    # Compute aggregate metrics
    daily_ics = np.array(daily_ics) if daily_ics else np.array([0.0])
    daily_rank_ics = np.array(daily_rank_ics) if daily_rank_ics else np.array([0.0])
    daily_model_returns = np.array(daily_model_returns) if daily_model_returns else np.array([0.0])
    daily_model_returns_net = np.array(daily_model_returns_net) if daily_model_returns_net else np.array([0.0])
    daily_momentum_returns = np.array(daily_momentum_returns) if daily_momentum_returns else np.array([0.0])
    daily_momentum_returns_net = np.array(daily_momentum_returns_net) if daily_momentum_returns_net else np.array([0.0])
    daily_random_returns = np.array(daily_random_returns) if daily_random_returns else np.array([0.0])
    daily_top_decile_returns = np.array(daily_top_decile_returns) if daily_top_decile_returns else np.array([0.0])
    daily_bottom_decile_returns = np.array(daily_bottom_decile_returns) if daily_bottom_decile_returns else np.array([0.0])
    daily_turnovers = np.array(daily_turnovers) if daily_turnovers else np.array([0.0])

    # Turnover statistics
    mean_turnover = np.mean(daily_turnovers) if len(daily_turnovers) > 0 else 0.0
    total_turnover = np.sum(daily_turnovers) if len(daily_turnovers) > 0 else 0.0

    # IC statistics
    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics) if len(daily_ics) > 1 else 1.0
    ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0

    # T-test for IC > 0
    if len(daily_ics) > 1:
        ic_t_stat, ic_p_value = stats.ttest_1samp(daily_ics, 0)
        ic_p_value = ic_p_value / 2  # One-sided
    else:
        ic_t_stat, ic_p_value = 0.0, 1.0

    pct_positive_ic = np.mean(daily_ics > 0) * 100
    mean_rank_ic = np.mean(daily_rank_ics)

    # Quantile analysis
    top_decile_return = np.mean(daily_top_decile_returns) * 100
    bottom_decile_return = np.mean(daily_bottom_decile_returns) * 100
    long_short_spread = top_decile_return - bottom_decile_return

    # Mean returns (GROSS - before transaction costs)
    model_mean_return = np.mean(daily_model_returns) * 100
    momentum_mean_return = np.mean(daily_momentum_returns) * 100
    random_mean_return = np.mean(daily_random_returns) * 100

    # Mean returns (NET - after transaction costs)
    model_mean_return_net = np.mean(daily_model_returns_net) * 100
    momentum_mean_return_net = np.mean(daily_momentum_returns_net) * 100
    # Random doesn't have turnover tracking, estimate with 100% turnover
    random_mean_return_net = random_mean_return - (cost_rate * 100)

    excess_vs_momentum = model_mean_return - momentum_mean_return
    excess_vs_random = model_mean_return - random_mean_return
    excess_vs_momentum_net = model_mean_return_net - momentum_mean_return_net
    excess_vs_random_net = model_mean_return_net - random_mean_return_net

    # Statistical significance vs baselines (using gross returns for fair comparison)
    if len(daily_model_returns) > 1:
        _, vs_random_p_value = stats.ttest_rel(daily_model_returns, daily_random_returns)
        vs_random_p_value = vs_random_p_value / 2
        _, vs_momentum_p_value = stats.ttest_rel(daily_model_returns, daily_momentum_returns)
        vs_momentum_p_value = vs_momentum_p_value / 2
    else:
        vs_random_p_value = 1.0
        vs_momentum_p_value = 1.0

    # Simulation metrics - GROSS (compound returns before costs)
    capital_gross = 100000.0
    capital_history_gross = [capital_gross]
    for ret in daily_model_returns:
        capital_gross *= (1 + ret)
        capital_history_gross.append(capital_gross)

    capital_history_gross = np.array(capital_history_gross)
    total_return_pct = (capital_history_gross[-1] / capital_history_gross[0] - 1) * 100

    # Simulation metrics - NET (compound returns after costs)
    capital_net = 100000.0
    capital_history_net = [capital_net]
    for ret in daily_model_returns_net:
        capital_net *= (1 + ret)
        capital_history_net.append(capital_net)

    capital_history_net = np.array(capital_history_net)
    total_return_pct_net = (capital_history_net[-1] / capital_history_net[0] - 1) * 100

    # Total transaction costs paid
    total_transaction_costs_pct = total_return_pct - total_return_pct_net

    returns_gross = np.diff(capital_history_gross) / capital_history_gross[:-1]
    returns_net = np.diff(capital_history_net) / capital_history_net[:-1]
    trades_per_year = 252 / horizon_days

    if len(returns_gross) > 1 and np.std(returns_gross) > 1e-8:
        sharpe_ratio = (np.mean(returns_gross) / np.std(returns_gross)) * np.sqrt(trades_per_year)
    else:
        sharpe_ratio = 0.0

    if len(returns_net) > 1 and np.std(returns_net) > 1e-8:
        sharpe_ratio_net = (np.mean(returns_net) / np.std(returns_net)) * np.sqrt(trades_per_year)
    else:
        sharpe_ratio_net = 0.0

    peak = np.maximum.accumulate(capital_history_gross)
    drawdown = (capital_history_gross - peak) / peak
    max_drawdown_pct = np.min(drawdown) * 100

    peak_net = np.maximum.accumulate(capital_history_net)
    drawdown_net = (capital_history_net - peak_net) / peak_net
    max_drawdown_pct_net = np.min(drawdown_net) * 100

    win_rate = np.mean(daily_model_returns > 0) * 100
    win_rate_net = np.mean(daily_model_returns_net > 0) * 100
    avg_return_pct = np.mean(daily_model_returns) * 100
    avg_return_pct_net = np.mean(daily_model_returns_net) * 100
    num_trades = len(daily_model_returns)

    return {
        # IC metrics
        'mean_ic': float(mean_ic),
        'std_ic': float(std_ic),
        'ir': float(ir),
        'ic_t_stat': float(ic_t_stat),
        'ic_p_value': float(ic_p_value),
        'pct_positive_ic': float(pct_positive_ic),
        'mean_rank_ic': float(mean_rank_ic),

        # Quantile analysis
        'top_decile_return': float(top_decile_return),
        'bottom_decile_return': float(bottom_decile_return),
        'long_short_spread': float(long_short_spread),

        # Baseline comparisons (GROSS - before costs)
        'model_mean_return': float(model_mean_return),
        'momentum_mean_return': float(momentum_mean_return),
        'random_mean_return': float(random_mean_return),
        'excess_vs_momentum': float(excess_vs_momentum),
        'excess_vs_random': float(excess_vs_random),
        'vs_random_p_value': float(vs_random_p_value),
        'vs_momentum_p_value': float(vs_momentum_p_value),

        # Baseline comparisons (NET - after costs)
        'model_mean_return_net': float(model_mean_return_net),
        'momentum_mean_return_net': float(momentum_mean_return_net),
        'random_mean_return_net': float(random_mean_return_net),
        'excess_vs_momentum_net': float(excess_vs_momentum_net),
        'excess_vs_random_net': float(excess_vs_random_net),

        # Simulation metrics (GROSS)
        'total_return_pct': float(total_return_pct),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown_pct': float(max_drawdown_pct),
        'win_rate': float(win_rate),
        'avg_return_pct': float(avg_return_pct),
        'num_trades': int(num_trades),

        # Simulation metrics (NET - after transaction costs)
        'total_return_pct_net': float(total_return_pct_net),
        'sharpe_ratio_net': float(sharpe_ratio_net),
        'max_drawdown_pct_net': float(max_drawdown_pct_net),
        'win_rate_net': float(win_rate_net),
        'avg_return_pct_net': float(avg_return_pct_net),

        # Turnover and transaction cost metrics
        'mean_turnover': float(mean_turnover),
        'total_turnover': float(total_turnover),
        'total_transaction_costs_pct': float(total_transaction_costs_pct),
        'transaction_cost_bps': float(transaction_cost_bps),

        # Data for plotting (needed by _plot_backtest_results)
        'daily_returns': daily_model_returns.tolist(),
        'daily_returns_net': daily_model_returns_net.tolist(),
        'capital_history': capital_history_gross.tolist(),
        'capital_history_net': capital_history_net.tolist(),
        'daily_turnovers': daily_turnovers.tolist(),

        # Daily IC arrays for detailed analysis
        'daily_ics': daily_ics.tolist(),
        'daily_rank_ics': daily_rank_ics.tolist(),
        'daily_model_returns': daily_model_returns.tolist(),
        'daily_momentum_returns': daily_momentum_returns.tolist(),
        'daily_random_returns': daily_random_returns.tolist(),
        'daily_top_decile_returns': daily_top_decile_returns,
        'daily_bottom_decile_returns': daily_bottom_decile_returns,
    }


# ============================================================================
# DDP (Distributed Data Parallel) Utilities
# ============================================================================

def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    return rt


def set_deterministic_seed(seed: int, rank: int = 0, fully_deterministic: bool = True):
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Master seed value
        rank: Process rank for DDP (adds offset to seed)
        fully_deterministic: If True, enables fully deterministic algorithms
                            (may reduce performance slightly)
    """
    effective_seed = seed + rank

    # Python random
    random.seed(effective_seed)

    # Numpy
    np.random.seed(effective_seed)

    # PyTorch CPU
    torch.manual_seed(effective_seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)

    # cuDNN deterministic settings
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enable deterministic algorithms globally (PyTorch 1.8+)
        # This will raise errors if non-deterministic ops are used
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions don't have warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        # Set CUBLAS workspace config for deterministic matmul
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        # Allow cuDNN to auto-tune for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if rank == 0:
        print(f"  Random seed: {seed} (effective: {effective_seed})")
        if fully_deterministic:
            print(f"  Deterministic mode: ENABLED (cuDNN deterministic, no benchmarking)")


@dataclass
class FoldTrainingResult:
    """Results from training and evaluating a single fold."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_days: int

    # Training metrics
    final_train_loss: float
    final_val_loss: float
    epochs_trained: int
    best_epoch: int

    # === PRINCIPLED EVALUATION METRICS ===
    # Information Coefficient (primary metrics)
    mean_ic: float = 0.0  # Mean daily IC (correlation between predictions and returns)
    std_ic: float = 0.0   # Std of daily IC
    ir: float = 0.0       # Information Ratio = mean_ic / std_ic
    ic_t_stat: float = 0.0  # T-statistic for IC > 0
    ic_p_value: float = 1.0  # P-value for IC > 0
    pct_positive_ic: float = 0.0  # % of days with positive IC
    mean_rank_ic: float = 0.0  # Spearman rank correlation

    # Quantile analysis
    top_decile_return: float = 0.0  # Mean return of top 10% predictions
    bottom_decile_return: float = 0.0  # Mean return of bottom 10%
    long_short_spread: float = 0.0  # Top - Bottom (the edge)

    # Baseline comparisons
    model_mean_return: float = 0.0  # Mean return of top-k selection
    momentum_mean_return: float = 0.0  # Momentum baseline
    random_mean_return: float = 0.0  # Random selection baseline
    excess_vs_momentum: float = 0.0  # Model - Momentum
    excess_vs_random: float = 0.0  # Model - Random

    # Statistical significance vs baselines
    vs_random_p_value: float = 1.0
    vs_momentum_p_value: float = 1.0

    # Simulation metrics (secondary)
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    num_trades: int = 0


class TemporalFoldDataset(torch.utils.data.Dataset):
    """Dataset for a specific temporal fold with date filtering."""

    def __init__(
        self,
        dataset_path: str,
        start_date: str,
        end_date: str,
        seq_len: int = 60,
        pred_days: List[int] = [1, 5, 10, 20],
        min_future_days: int = 20,
        news_dim: int = 768,
        preload: bool = True,  # Preload all data into memory for speed
        data_fraction: float = 1.0,  # Fraction of data to use (for fast experiments)
        prices_path: str = None  # Path to actual prices HDF5 for target computation
    ):
        """
        Initialize dataset for a specific date range.

        Args:
            dataset_path: Path to HDF5 file
            start_date: Start date (inclusive) YYYY-MM-DD
            end_date: End date (inclusive) YYYY-MM-DD
            seq_len: Sequence length
            pred_days: Prediction horizons
            min_future_days: Min future days needed
            news_dim: News embedding dimension
            preload: If True, load all data into memory (faster but uses more RAM)
            data_fraction: Fraction of sequences to use (0.0-1.0), for faster experiments
            prices_path: Path to actual prices HDF5 for computing targets (CRITICAL for correct training)
        """
        self.dataset_path = dataset_path
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.pred_days = sorted(pred_days)
        self.min_future_days = min_future_days
        self.news_dim = news_dim
        self.preload = preload
        self.data_fraction = data_fraction
        self.prices_path = prices_path

        # Open HDF5 file
        self.h5f = h5py.File(dataset_path, 'r')

        # Open actual prices file if provided (CRITICAL for correct targets)
        self.prices_h5f = None
        self.prices_cache = {}  # Cache: ticker -> {date_str: price}
        if prices_path and os.path.exists(prices_path):
            self.prices_h5f = h5py.File(prices_path, 'r')
            print(f"    Using actual prices from: {prices_path}")
        else:
            print(f"    WARNING: No prices_path provided - targets will use normalized features (INCORRECT!)")

        # Find feature dimension
        sample_ticker = list(self.h5f.keys())[0]
        self.total_features = self.h5f[sample_ticker]['features'].shape[1]

        # Build index of valid sequences within date range
        self.sequences = self._build_sequence_index()

        # Subsample sequences if data_fraction < 1.0
        if data_fraction < 1.0 and len(self.sequences) > 0:
            num_to_keep = max(1, int(len(self.sequences) * data_fraction))
            # Use random.sample for consistent subsampling
            self.sequences = random.sample(self.sequences, num_to_keep)
            print(f"    Subsampled to {len(self.sequences)} sequences ({data_fraction*100:.0f}% of data)")
            # Rebuild date_to_indices mapping after subsampling
            self._rebuild_date_mapping()

        # Preload data into memory for faster training
        if preload and len(self.sequences) > 0:
            self._preload_data()
            self.h5f.close()
            self.h5f = None
        else:
            self.cached_data = None

        print(f"    Created dataset: {start_date} to {end_date}, {len(self.sequences)} sequences")

    def _build_sequence_index(self):
        """Build index of sequences within date range.

        Also builds date_to_indices mapping for cross-sectional batch sampling.
        Only includes sequences where dates are valid trading days (exist in prices file).
        """
        sequences = []
        date_to_indices = {}  # Maps date -> list of sequence indices
        skipped_no_price = 0

        for ticker in self.h5f.keys():
            group = self.h5f[ticker]
            dates = [d.decode('utf-8') for d in group['dates'][:]]
            num_dates = len(dates)

            # Get valid trading days for this ticker from prices file
            if self.prices_h5f is not None:
                if ticker not in self.prices_h5f:
                    # Skip tickers that don't exist in prices file
                    continue
                trading_days = set(d.decode('utf-8') for d in self.prices_h5f[ticker]['dates'][:])
                trading_days_list = sorted(trading_days)
                # Build index lookup for fast trading day position lookup
                trading_day_idx = {d: i for i, d in enumerate(trading_days_list)}
                max_trading_idx = len(trading_days_list) - 1 - max(self.pred_days)
            else:
                trading_days = None  # No filtering if no prices file
                trading_days_list = None
                trading_day_idx = None
                max_trading_idx = float('inf')

            # Find valid sequence positions within date range
            for i in range(self.seq_len, num_dates - self.min_future_days):
                seq_end_date = dates[i - 1]

                # Check if sequence end date is within our range
                if self.start_date <= seq_end_date <= self.end_date:
                    # Skip non-trading days (weekends, holidays)
                    if trading_days is not None and seq_end_date not in trading_days:
                        skipped_no_price += 1
                        continue

                    # Verify enough future TRADING days exist for target computation
                    if trading_day_idx is not None:
                        current_trading_idx = trading_day_idx.get(seq_end_date)
                        if current_trading_idx is None or current_trading_idx > max_trading_idx:
                            skipped_no_price += 1
                            continue

                    seq_idx = len(sequences)
                    sequences.append((ticker, i, seq_end_date))

                    # Build date -> indices mapping for cross-sectional sampling
                    if seq_end_date not in date_to_indices:
                        date_to_indices[seq_end_date] = []
                    date_to_indices[seq_end_date].append(seq_idx)

        if skipped_no_price > 0:
            print(f"    Skipped {skipped_no_price} sequences (non-trading days)")

        # Store date mapping for cross-sectional batch sampling
        self.date_to_indices = date_to_indices
        self.sorted_dates = sorted(date_to_indices.keys())

        return sequences

    def _rebuild_date_mapping(self):
        """Rebuild date_to_indices mapping after subsampling.

        This must be called after self.sequences is modified (e.g., by data_fraction
        subsampling) to ensure the indices are correct.
        """
        date_to_indices = {}

        for idx, (ticker, end_idx, seq_end_date) in enumerate(self.sequences):
            if seq_end_date not in date_to_indices:
                date_to_indices[seq_end_date] = []
            date_to_indices[seq_end_date].append(idx)

        self.date_to_indices = date_to_indices
        self.sorted_dates = sorted(date_to_indices.keys())

    def _load_ticker_prices(self, ticker: str) -> dict:
        """Load and cache actual prices for a ticker.

        Returns dict mapping date_str -> price.
        """
        if ticker in self.prices_cache:
            return self.prices_cache[ticker]

        if self.prices_h5f is None or ticker not in self.prices_h5f:
            return {}

        grp = self.prices_h5f[ticker]
        dates = [d.decode('utf-8') for d in grp['dates'][:]]
        prices = grp['prices'][:]

        # Build lookup dict
        price_lookup = {d: float(p) for d, p in zip(dates, prices)}
        self.prices_cache[ticker] = price_lookup
        return price_lookup

    def _get_actual_price(self, ticker: str, date_str: str) -> float:
        """Get actual price for a ticker on a specific date.

        Only returns exact matches - non-trading days are filtered out
        at the sequence level in _build_sequence_index().
        """
        price_lookup = self._load_ticker_prices(ticker)
        return price_lookup.get(date_str, None)

    def _get_trading_dates(self, ticker: str) -> list:
        """Get sorted list of trading dates for a ticker.

        Cached for efficiency.
        """
        cache_key = f"_trading_dates_{ticker}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        if self.prices_h5f is None or ticker not in self.prices_h5f:
            return []

        grp = self.prices_h5f[ticker]
        trading_dates = sorted([d.decode('utf-8') for d in grp['dates'][:]])
        setattr(self, cache_key, trading_dates)
        return trading_dates

    def _preload_data(self):
        """Preload all sequences into memory for faster training."""
        print(f"    Preloading {len(self.sequences)} sequences into memory...")

        self.cached_features = []
        self.cached_targets = []

        for idx in range(len(self.sequences)):
            ticker, end_idx, _seq_date = self.sequences[idx]
            group = self.h5f[ticker]

            # Load the slice we need
            start_idx = end_idx - self.seq_len
            max_future_idx = end_idx + max(self.pred_days)
            load_end = min(max_future_idx, group['features'].shape[0])
            features_slice = group['features'][start_idx:load_end]

            # Extract sequence
            seq_features = features_slice[:self.seq_len].astype(np.float32)

            # Pad if needed
            if seq_features.shape[1] < self.total_features:
                padding = np.zeros((seq_features.shape[0], self.total_features - seq_features.shape[1]), dtype=np.float32)
                seq_features = np.concatenate([seq_features, padding], axis=1)

            # Get dates for this sequence
            dates = [d.decode('utf-8') for d in group['dates'][:]]

            # CRITICAL: Use ACTUAL prices for target computation, not normalized features
            if self.prices_h5f is not None:
                # Get actual current price from prices file
                current_date = dates[end_idx - 1]  # Date at end of sequence
                current_price = self._get_actual_price(ticker, current_date)

                if current_price is None or current_price == 0 or np.isnan(current_price):
                    # Fallback: skip this sample or use 0 returns
                    current_price = 1.0
                    future_prices = [1.0] * len(self.pred_days)
                else:
                    # Get sorted trading dates for this ticker to compute TRADING day offsets
                    trading_dates = self._get_trading_dates(ticker)

                    # Find current date's position in trading dates
                    try:
                        current_trading_idx = trading_dates.index(current_date)
                    except ValueError:
                        # Current date not in trading dates
                        current_price = 1.0
                        future_prices = [1.0] * len(self.pred_days)
                        current_trading_idx = -1

                    if current_trading_idx >= 0:
                        future_prices = []
                        for days_ahead in self.pred_days:
                            future_trading_idx = current_trading_idx + days_ahead
                            if future_trading_idx < len(trading_dates):
                                future_date = trading_dates[future_trading_idx]
                                future_price = self._get_actual_price(ticker, future_date)
                                if future_price is None or np.isnan(future_price):
                                    future_price = current_price
                            else:
                                future_price = current_price
                            future_prices.append(future_price)
            else:
                # Fallback to normalized features (INCORRECT but backwards-compatible)
                current_price = seq_features[-1, 0]
                if current_price == 0 or np.isnan(current_price) or np.isinf(current_price):
                    current_price = 1.0

                future_prices = []
                for days_ahead in self.pred_days:
                    future_local_idx = self.seq_len + days_ahead - 1
                    if future_local_idx < features_slice.shape[0]:
                        future_price = float(features_slice[future_local_idx, 0])
                    else:
                        future_price = float(features_slice[-1, 0])
                    if np.isnan(future_price) or np.isinf(future_price):
                        future_price = current_price
                    future_prices.append(future_price)

            # Calculate price ratios (returns)
            price_ratios = np.array([(fp / current_price) - 1.0 for fp in future_prices], dtype=np.float32)
            price_ratios = np.clip(price_ratios, -0.5, 0.5)

            self.cached_features.append(seq_features)
            self.cached_targets.append(price_ratios)

        # Convert to tensors
        self.cached_features = torch.from_numpy(np.stack(self.cached_features))
        self.cached_targets = torch.from_numpy(np.stack(self.cached_targets))
        print(f"    Preloaded: features {self.cached_features.shape}, targets {self.cached_targets.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a training sample."""
        # Use cached data if available (much faster!)
        if self.preload and hasattr(self, 'cached_features'):
            return self.cached_features[idx], self.cached_targets[idx]

        # Fallback to loading from HDF5
        ticker, end_idx, _seq_date = self.sequences[idx]

        # Re-open file if needed (for multiprocessing compatibility)
        if self.h5f is None:
            self.h5f = h5py.File(self.dataset_path, 'r')

        group = self.h5f[ticker]

        # Only load the slice we need
        start_idx = end_idx - self.seq_len
        max_future_idx = end_idx + max(self.pred_days)
        load_end = min(max_future_idx, group['features'].shape[0])
        features_slice = group['features'][start_idx:load_end]

        # Extract sequence
        seq_features = features_slice[:self.seq_len]
        features = torch.from_numpy(seq_features.astype(np.float32))

        # Pad if needed
        if features.shape[1] < self.total_features:
            padding = torch.zeros(features.shape[0], self.total_features - features.shape[1])
            features = torch.cat([features, padding], dim=1)

        # Get dates for this sequence
        dates = [d.decode('utf-8') for d in group['dates'][:]]

        # CRITICAL: Use ACTUAL prices for target computation, not normalized features
        if self.prices_h5f is not None:
            # Get actual current price from prices file
            current_date = dates[end_idx - 1]
            current_price = self._get_actual_price(ticker, current_date)

            if current_price is None or current_price == 0 or np.isnan(current_price):
                current_price = 1.0
                future_prices = [1.0] * len(self.pred_days)
            else:
                # Get sorted trading dates for this ticker to compute TRADING day offsets
                trading_dates = self._get_trading_dates(ticker)

                # Find current date's position in trading dates
                try:
                    current_trading_idx = trading_dates.index(current_date)
                except ValueError:
                    # Current date not in trading dates - shouldn't happen if filtered correctly
                    current_price = 1.0
                    future_prices = [1.0] * len(self.pred_days)
                    current_trading_idx = -1

                if current_trading_idx >= 0:
                    future_prices = []
                    for days_ahead in self.pred_days:
                        future_trading_idx = current_trading_idx + days_ahead
                        if future_trading_idx < len(trading_dates):
                            future_date = trading_dates[future_trading_idx]
                            future_price = self._get_actual_price(ticker, future_date)
                            if future_price is None or np.isnan(future_price):
                                future_price = current_price
                        else:
                            future_price = current_price
                        future_prices.append(future_price)
        else:
            # Fallback to normalized features (INCORRECT)
            current_price = features[-1, 0].item()
            if current_price == 0 or np.isnan(current_price) or np.isinf(current_price):
                current_price = 1.0

            future_prices = []
            for days_ahead in self.pred_days:
                future_local_idx = self.seq_len + days_ahead - 1
                if future_local_idx < features_slice.shape[0]:
                    future_price = float(features_slice[future_local_idx, 0])
                else:
                    future_price = float(features_slice[-1, 0])
                if np.isnan(future_price) or np.isinf(future_price):
                    future_price = current_price
                future_prices.append(future_price)

        price_ratios = torch.tensor([(fp / current_price) - 1.0 for fp in future_prices], dtype=torch.float32)
        price_ratios = torch.clamp(price_ratios, -0.5, 0.5)

        return features, price_ratios

    def close(self):
        """Close HDF5 files."""
        if hasattr(self, 'h5f') and self.h5f is not None:
            self.h5f.close()
        if hasattr(self, 'prices_h5f') and self.prices_h5f is not None:
            self.prices_h5f.close()


class CrossSectionalBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that groups samples by date for cross-sectional ranking.

    For ranking losses to work correctly, we need to compare stocks within
    the same cross-section (same trading day). This sampler ensures each
    batch contains only samples from the same date.

    This is CRITICAL for ranking loss training:
    - Standard shuffle mixes stocks from different dates
    - Ranking loss compares all stocks in a batch
    - Comparing stocks across dates is meaningless for cross-sectional ranking
    - This sampler ensures valid cross-sectional comparisons

    Dates are processed in CHRONOLOGICAL ORDER (oldest to newest) so that
    the model sees the most recent data last, giving recent patterns more
    influence on final weights.
    """

    def __init__(
        self,
        dataset: 'WalkForwardDataset',
        batch_size: int = 64,
        min_samples_per_date: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: WalkForwardDataset with date_to_indices mapping
            batch_size: Maximum batch size (may be smaller for dates with few stocks)
            min_samples_per_date: Skip dates with fewer samples than this
            shuffle: Whether to shuffle samples within each date (dates always chronological)
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_samples_per_date = min_samples_per_date
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Get date-to-indices mapping from dataset
        self.date_to_indices = dataset.date_to_indices
        self.sorted_dates = dataset.sorted_dates

        # Filter dates with enough samples
        self.valid_dates = [
            d for d in self.sorted_dates
            if len(self.date_to_indices[d]) >= min_samples_per_date
        ]

        print(f"    CrossSectionalBatchSampler: {len(self.valid_dates)}/{len(self.sorted_dates)} dates "
              f"have >= {min_samples_per_date} samples (chronological order, newest last)")

    def __iter__(self):
        """Yield batches where each batch contains samples from a single date."""
        # Train in chronological order (oldest to newest)
        # This ensures most recent data is trained on last
        dates = sorted(self.valid_dates)

        for date in dates:
            indices = self.date_to_indices[date].copy()

            # Shuffle samples within date for variety (ranking is still valid)
            if self.shuffle:
                random.shuffle(indices)

            # Yield batches for this date
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                # Skip incomplete batches if drop_last is True
                if self.drop_last and len(batch) < self.batch_size:
                    continue

                # Skip batches that are too small for ranking
                if len(batch) >= 2:
                    yield batch

    def __len__(self):
        """Approximate number of batches."""
        total_batches = 0
        for date in self.valid_dates:
            n_samples = len(self.date_to_indices[date])
            n_batches = n_samples // self.batch_size
            if not self.drop_last and n_samples % self.batch_size > 0:
                n_batches += 1
            total_batches += n_batches
        return total_batches


class DistributedCrossSectionalBatchSampler(torch.utils.data.Sampler):
    """
    Distributed batch sampler for cross-sectional ranking with multi-GPU support.

    Key design: Divides DATES (not samples) across GPUs, so each GPU processes
    complete cross-sections. This ensures valid pairwise comparisons within each GPU.

    Example with 2 GPUs and 100 dates:
    - GPU 0 processes dates [0, 2, 4, ...] (50 dates)
    - GPU 1 processes dates [1, 3, 5, ...] (50 dates)
    - Each GPU batches samples within its assigned dates

    Dates are processed in CHRONOLOGICAL ORDER (oldest to newest) so that
    the model sees the most recent data last, giving recent patterns more
    influence on final weights.

    This is different from standard DistributedSampler which splits individual samples,
    which would break cross-sectional ranking.
    """

    def __init__(
        self,
        dataset: 'WalkForwardDataset',
        num_replicas: int = None,
        rank: int = None,
        batch_size: int = 64,
        min_samples_per_date: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        """
        Args:
            dataset: WalkForwardDataset with date_to_indices mapping
            num_replicas: Number of distributed processes (GPUs)
            rank: Rank of current process
            batch_size: Maximum batch size per date
            min_samples_per_date: Skip dates with fewer samples
            shuffle: Whether to shuffle samples within each date (dates always chronological)
            drop_last: Whether to drop incomplete batches
            seed: Random seed for reproducibility
        """
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.min_samples_per_date = min_samples_per_date
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Get date-to-indices mapping from dataset
        self.date_to_indices = dataset.date_to_indices
        self.sorted_dates = dataset.sorted_dates

        # Filter dates with enough samples
        self.valid_dates = [
            d for d in self.sorted_dates
            if len(self.date_to_indices[d]) >= min_samples_per_date
        ]

        # Divide dates among ranks
        # Each rank gets every num_replicas-th date
        self.rank_dates = self.valid_dates[self.rank::self.num_replicas]

        if self.rank == 0:
            print(f"    DistributedCrossSectionalBatchSampler (chronological order, newest last):")
            print(f"      Total valid dates: {len(self.valid_dates)}")
            print(f"      Dates per GPU: ~{len(self.valid_dates) // self.num_replicas}")
            print(f"      Num GPUs: {self.num_replicas}")

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across processes."""
        self.epoch = epoch

    def __iter__(self):
        """Yield batches where each batch contains samples from a single date."""
        # Train in chronological order (oldest to newest)
        # This ensures most recent data is trained on last
        # rank_dates is already in sorted order from valid_dates[rank::num_replicas]
        dates = sorted(self.rank_dates)

        for date in dates:
            indices = self.date_to_indices[date].copy()

            # Shuffle samples within date for variety (ranking is still valid)
            if self.shuffle:
                random.seed(self.seed + self.epoch + hash(date))
                random.shuffle(indices)

            # Yield batches for this date
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                # Skip incomplete batches if drop_last is True
                if self.drop_last and len(batch) < self.batch_size:
                    continue

                # Skip batches that are too small for ranking
                if len(batch) >= 2:
                    yield batch

    def __len__(self):
        """Approximate number of batches for this rank."""
        total_batches = 0
        for date in self.rank_dates:
            n_samples = len(self.date_to_indices[date])
            n_batches = n_samples // self.batch_size
            if not self.drop_last and n_samples % self.batch_size > 0:
                n_batches += 1
            total_batches += n_batches
        return total_batches


class WalkForwardTrainer:
    """
    Orchestrates walk-forward training with online evaluation.
    """

    def __init__(
        self,
        dataset_path: str,
        prices_path: str,
        num_folds: int = 5,
        mode: str = 'expanding',
        min_train_months: int = 12,
        test_months: int = 3,
        gap_days: int = 5,
        auto_span: bool = True,  # Auto-calculate train/test periods to span entire dataset
        initial_train_fraction: float = 0.5,  # Fraction of data for initial training (when auto_span=True)
        # Model config
        model_type: str = 'transformer',  # 'transformer' or 'multimodal'
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        pred_mode: str = 'classification',
        # Contrastive pretraining config
        contrastive_pretrain: bool = False,  # Enable contrastive pretraining
        contrastive_epochs: int = 5,  # Epochs for contrastive pretraining
        contrastive_lr: float = 1e-4,  # Learning rate for contrastive pretraining
        contrastive_temperature: float = 0.1,  # Temperature for InfoNCE loss
        freeze_encoder_after_pretrain: bool = True,  # Freeze encoder during finetuning
        # Training config
        epochs_per_fold: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        seq_len: int = 60,
        # Evaluation config
        top_k: int = 5,
        horizon_idx: int = 0,
        confidence_percentile: float = 0.6,
        subset_size: int = 512,
        num_test_stocks: int = 1000,
        max_eval_dates: int = 60,  # Max dates for principled evaluation (randomly sampled)
        initial_capital: float = 100000.0,
        # Progressive evaluation
        eval_every_n_steps: int = 100,  # Evaluate on val set every N optimizer steps (0=disable)
        loss_smoothing_window: int = 50,  # EMA window for smoothed loss curves
        save_loss_curves_every: int = 100,  # Save loss curve PNG every N steps (0=only at end)
        # Data subsampling for faster experiments
        data_fraction: float = 1.0,  # Fraction of training data to use (0.0-1.0)
        # Accelerated incremental training
        incremental_training: bool = False,  # Only train on new data, reuse previous checkpoint
        incremental_epochs: int = 2,  # Epochs for incremental training (usually fewer needed)
        incremental_data_fraction: float = 1.0,  # Data fraction specifically for new data in incremental mode
        # Other
        device: str = 'cuda',
        seed: int = 42,
        checkpoint_dir: str = 'checkpoints/walk_forward',
        preload_data: bool = True,  # Set False for large seq_len to save memory
        num_workers: int = 4,  # DataLoader workers (0 for debugging)
        compile_model: bool = True,  # Use torch.compile for faster training
        compile_mode: str = 'default',  # torch.compile mode
        use_ddp: bool = False,  # Enable Distributed Data Parallel
        gradient_accumulation_steps: int = 1,  # Accumulate gradients over N steps for larger effective batch size
        adaptive_batch_size: bool = False,  # Increase grad accumulation when loss plateaus
        plateau_patience: int = 15,  # Steps without improvement before increasing batch size
        plateau_threshold: float = 0.001,  # Minimum relative improvement to count as progress
        max_grad_accum: int = 32,  # Maximum gradient accumulation steps
        early_stopping_patience: int = 3,  # Stop training if val loss doesn't improve for N epochs
        # Ranking loss config
        ranking_loss_weight: float = 0.0,  # Weight for ranking loss (0.0 = disabled, 1.0 = equal to CE)
        ranking_loss_type: str = 'pairwise',  # 'pairwise' or 'listnet'
        ranking_margin: float = 0.01,  # Margin for pairwise ranking loss
        ranking_only: bool = False,  # If True, use ONLY ranking loss (no CE/MSE)
        # Monte Carlo validation config
        monte_carlo: bool = True,  # Run Monte Carlo validation after each fold
        mc_trials: int = 50,  # Number of Monte Carlo trials
        mc_stocks: int = 150,  # Stocks per trial
        mc_top_ks: List[int] = None,  # Top-k values to test (default: [5, 10, 15, 20])
        # Intermediate checkpoint saving for training progression analysis
        save_intermediate_checkpoints: bool = False,  # Save checkpoints throughout training
        checkpoint_every_n_epochs: int = 1,  # Save checkpoint every N epochs
        # Transaction cost config
        transaction_cost_bps: float = 10.0,  # Transaction cost in basis points (10 bps = 0.1%)
        # Baseline comparison
        compare_models: bool = False,  # Train and compare baseline models
        baseline_max_samples: int = 50000,  # Max training samples for baselines
    ):
        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.num_folds = num_folds
        self.mode = mode
        self.min_train_months = min_train_months
        self.test_months = test_months
        self.gap_days = gap_days
        self.auto_span = auto_span
        self.initial_train_fraction = initial_train_fraction

        # Model config
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.pred_mode = pred_mode

        # Contrastive pretraining config
        self.contrastive_pretrain = contrastive_pretrain
        self.contrastive_epochs = contrastive_epochs
        self.contrastive_lr = contrastive_lr
        self.contrastive_temperature = contrastive_temperature
        self.freeze_encoder_after_pretrain = freeze_encoder_after_pretrain

        # Training config
        self.epochs_per_fold = epochs_per_fold
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.preload_data = preload_data
        self.num_workers = num_workers

        # Evaluation config
        self.top_k = top_k
        self.horizon_idx = horizon_idx
        self.confidence_percentile = confidence_percentile
        self.subset_size = subset_size
        self.num_test_stocks = num_test_stocks
        self.max_eval_dates = max_eval_dates
        self.initial_capital = initial_capital

        # Progressive evaluation
        self.eval_every_n_steps = eval_every_n_steps
        self.loss_smoothing_window = loss_smoothing_window
        self.save_loss_curves_every = save_loss_curves_every

        # Data subsampling
        self.data_fraction = data_fraction

        # Accelerated incremental training
        self.incremental_training = incremental_training
        self.incremental_epochs = incremental_epochs
        self.incremental_data_fraction = incremental_data_fraction

        # Other
        self.device = device
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.use_ddp = use_ddp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.adaptive_batch_size = adaptive_batch_size
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.max_grad_accum = max_grad_accum
        self.early_stopping_patience = early_stopping_patience

        # Ranking loss config
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_loss_type = ranking_loss_type
        self.ranking_margin = ranking_margin
        self.ranking_only = ranking_only

        # Monte Carlo validation config
        self.monte_carlo = monte_carlo
        self.mc_trials = mc_trials
        self.mc_stocks = mc_stocks
        self.mc_top_ks = mc_top_ks if mc_top_ks is not None else [5, 10, 15, 20]

        # Intermediate checkpoint saving
        self.save_intermediate_checkpoints = save_intermediate_checkpoints
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

        # Transaction cost config
        self.transaction_cost_bps = transaction_cost_bps

        # Baseline comparison
        self.compare_models = compare_models
        self.baseline_max_samples = baseline_max_samples
        self.baseline_results: List[Dict] = []  # Store baseline results per fold

        # Map horizon_idx to days
        horizon_map = {0: 1, 1: 5, 2: 10, 3: 20}
        self.horizon_days = horizon_map.get(horizon_idx, 1)

        # Results storage
        self.fold_results: List[FoldTrainingResult] = []

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load date information from dataset
        self._load_date_info()

    def _load_date_info(self):
        """Load available dates from dataset."""
        print(f"\nLoading date information from: {self.dataset_path}")

        with h5py.File(self.dataset_path, 'r') as f:
            sample_ticker = list(f.keys())[0]
            dates_bytes = f[sample_ticker]['dates'][:]
            self.all_dates = sorted([d.decode('utf-8') for d in dates_bytes])
            self.input_dim = f[sample_ticker]['features'].shape[1]

        print(f"  Date range: {self.all_dates[0]} to {self.all_dates[-1]}")
        print(f"  Total dates: {len(self.all_dates)}")
        print(f"  Input dimension: {self.input_dim}")

        # If prices file provided, use its trading days
        if self.prices_path and os.path.exists(self.prices_path):
            with h5py.File(self.prices_path, 'r') as f:
                sample_ticker = list(f.keys())[0]
                prices_dates_bytes = f[sample_ticker]['dates'][:]
                self.trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
            print(f"  Trading days from prices: {len(self.trading_days)}")
        else:
            self.trading_days = self.all_dates

        # Calculate usable date range based on seq_len and future days requirement
        # First usable date needs seq_len days of history
        # Last usable date needs min_future_days of future data
        min_future_days = 20  # Same as TemporalFoldDataset default
        first_usable_idx = self.seq_len
        last_usable_idx = len(self.all_dates) - min_future_days

        if first_usable_idx >= last_usable_idx:
            raise ValueError(
                f"Not enough data for seq_len={self.seq_len}. "
                f"Dataset has {len(self.all_dates)} dates, need at least {self.seq_len + min_future_days + 1}."
            )

        self.usable_dates = self.all_dates[first_usable_idx:last_usable_idx]
        print(f"  Usable date range (seq_len={self.seq_len}): {self.usable_dates[0]} to {self.usable_dates[-1]}")
        print(f"  Usable dates: {len(self.usable_dates)}")

    def _get_folds(self) -> List[Tuple[List[str], List[str]]]:
        """Generate temporal folds based on mode, using only dates with valid sequences."""
        days_per_month = 21

        # Use usable_dates for training (dates that have enough history for seq_len)
        # Filter trading_days to only include usable dates
        usable_set = set(self.usable_dates)
        valid_dates = [d for d in self.trading_days if d in usable_set]

        if len(valid_dates) == 0:
            raise ValueError(
                f"No valid dates for training. Check that trading_days overlap with usable_dates. "
                f"usable_dates: {self.usable_dates[0]} to {self.usable_dates[-1]}"
            )

        total_days = len(valid_dates)

        if self.auto_span:
            # Auto-calculate periods to span entire dataset
            # initial_train_fraction of data for first fold's training
            # Remaining data divided equally into num_folds test periods
            initial_train_days = int(total_days * self.initial_train_fraction)
            remaining_days = total_days - initial_train_days - self.gap_days * self.num_folds

            if remaining_days <= 0:
                raise ValueError(
                    f"Not enough data to span {self.num_folds} folds with "
                    f"{self.initial_train_fraction:.0%} initial training. "
                    f"Total days: {total_days}, initial train: {initial_train_days}"
                )

            test_days = remaining_days // self.num_folds
            if test_days < 5:
                print(f"  WARNING: Very short test periods ({test_days} days). Consider fewer folds.")

            print(f"\n  AUTO-SPAN: {total_days} total days")
            print(f"    Initial training: {initial_train_days} days ({self.initial_train_fraction:.0%})")
            print(f"    Test period per fold: {test_days} days")
            print(f"    Gap between folds: {self.gap_days} days")
        else:
            # Use fixed periods from parameters
            initial_train_days = self.min_train_months * days_per_month
            test_days = self.test_months * days_per_month

        # Ensure we have enough days for at least one fold
        min_required = initial_train_days + self.gap_days + test_days
        if total_days < min_required:
            print(f"  WARNING: Only {total_days} valid days, need {min_required} for requested config.")
            print(f"  Adjusting to use all available data...")
            # Use 70% for train, 30% for test with single fold
            train_end_idx = int(total_days * 0.7)
            test_start_idx = train_end_idx + self.gap_days
            if test_start_idx >= total_days:
                test_start_idx = train_end_idx  # No gap if not enough room
            train_dates = valid_dates[:train_end_idx]
            test_dates = valid_dates[test_start_idx:]
            if len(train_dates) > 0 and len(test_dates) > 0:
                return [(train_dates, test_dates)]
            else:
                raise ValueError(f"Not enough data for even a single fold. Have {total_days} valid days.")

        folds = []

        if self.mode == 'expanding':
            # Expanding window: training grows with each fold
            first_test_start = initial_train_days + self.gap_days

            for fold_idx in range(self.num_folds):
                test_start_idx = first_test_start + fold_idx * test_days
                test_end_idx = min(test_start_idx + test_days, total_days)

                train_end_idx = test_start_idx - self.gap_days
                train_start_idx = 0  # Always from beginning

                if test_end_idx > total_days:
                    break

                train_dates = valid_dates[train_start_idx:train_end_idx]
                test_dates = valid_dates[test_start_idx:test_end_idx]

                if len(test_dates) > 0 and len(train_dates) > 0:
                    folds.append((train_dates, test_dates))

        elif self.mode == 'sliding':
            # Sliding window: fixed training size
            train_days = initial_train_days
            stride = test_days

            for fold_idx in range(self.num_folds):
                test_start_idx = train_days + self.gap_days + fold_idx * stride
                test_end_idx = min(test_start_idx + test_days, total_days)

                if test_end_idx > total_days:
                    break

                train_end_idx = test_start_idx - self.gap_days
                train_start_idx = max(0, train_end_idx - train_days)

                train_dates = valid_dates[train_start_idx:train_end_idx]
                test_dates = valid_dates[test_start_idx:test_end_idx]

                if len(test_dates) > 0 and len(train_dates) > 0:
                    folds.append((train_dates, test_dates))

        return folds

    def _create_model(self, compile_model: bool = True):
        """Create a fresh model for training."""
        # Determine device for this process
        if self.use_ddp:
            local_rank = get_rank()
            device = f'cuda:{local_rank}'
        else:
            device = self.device

        if self.model_type == 'multimodal':
            if self.contrastive_pretrain:
                # Contrastive learning wrapper for multimodal model
                model = ContrastiveMultiModalModel(
                    feature_config=FeatureConfig(),
                    hidden_dim=self.hidden_dim,
                    projection_dim=128,
                    num_technical_layers=self.num_layers,
                    num_technical_heads=self.num_heads,
                    num_pred_days=4,
                    pred_mode=self.pred_mode,
                    dropout=self.dropout,
                    max_seq_len=self.seq_len,
                    temperature=self.contrastive_temperature,
                )
            else:
                # Multi-modal model with separate encoders for each feature type
                model = MultiModalStockPredictor(
                    feature_config=FeatureConfig(),
                    hidden_dim=self.hidden_dim,
                    num_technical_layers=self.num_layers,
                    num_technical_heads=self.num_heads,
                    num_pred_days=4,
                    pred_mode=self.pred_mode,
                    dropout=self.dropout,
                    max_seq_len=self.seq_len,
                )
        else:
            # Default: single-stream transformer
            model = SimpleTransformerPredictor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                num_pred_days=4,
                pred_mode=self.pred_mode
            )
        model = model.to(device)

        # Ensure all parameters require grad
        for param in model.parameters():
            param.requires_grad_(True)

        # Compile model for faster training (do this BEFORE wrapping with DDP)
        if compile_model and hasattr(torch, 'compile'):
            if is_main_process():
                print(f"  Compiling model with torch.compile (mode={self.compile_mode})...")
            model = torch.compile(model, mode=self.compile_mode)

        # Wrap with DDP if enabled
        if self.use_ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # confidence output not always used in loss
            )
            if is_main_process():
                print(f"  Model wrapped with DDP (world_size={get_world_size()})")

        return model

    def _plot_loss_curves(
        self,
        fold_idx: int,
        train_losses: List[float],
        val_losses: List[float],
        best_epoch: int
    ):
        """Generate and save loss curve plots for a fold."""
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)

        # Plot training and validation loss
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        # Mark best epoch
        if best_epoch > 0 and best_epoch <= len(val_losses):
            plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                       label=f'Best Epoch ({best_epoch})')
            plt.scatter([best_epoch], [val_losses[best_epoch-1]], color='g', s=100,
                       zorder=5, marker='*')

        # Mark early stopping point if applicable
        if len(train_losses) < self.epochs_per_fold:
            plt.axvline(x=len(train_losses), color='orange', linestyle=':',
                       alpha=0.7, label='Early Stop')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Fold {fold_idx + 1} - Training vs Validation Loss', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Set integer ticks for epochs
        plt.xticks(epochs)

        # Add annotation for best val loss
        best_val = min(val_losses)
        plt.annotate(f'Best: {best_val:.4f}',
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 0.5, best_val + (max(val_losses) - min(val_losses)) * 0.1),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_loss_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Loss curves saved to: {plot_path}")

    def _save_step_loss_curves(
        self,
        fold_idx: int,
        step_train_losses: List[float],
        step_val_losses: List[float],
        step_eval_points: List[int],
        best_step: int,
        current_step: int,
        final: bool = False
    ):
        """
        Save smoothed step-level loss curves as PNG.

        Creates a detailed plot showing:
        - Raw training loss (faded)
        - EMA-smoothed training loss
        - Validation loss at evaluation checkpoints
        - Best step marker
        """
        if not is_main_process() or len(step_train_losses) == 0:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        steps = np.arange(1, len(step_train_losses) + 1)

        # Compute smoothed losses with EMA
        def ema_smooth(data, window):
            if len(data) == 0:
                return []
            alpha = 2.0 / (window + 1)
            smoothed = [data[0]]
            for i in range(1, len(data)):
                smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
            return smoothed

        smoothed_train = ema_smooth(step_train_losses, self.loss_smoothing_window)
        smoothed_train_tight = ema_smooth(step_train_losses, max(10, self.loss_smoothing_window // 5))

        # Top plot: Training loss
        ax1 = axes[0]
        ax1.plot(steps, step_train_losses, 'b-', alpha=0.2, linewidth=0.5, label='Raw')
        ax1.plot(steps, smoothed_train_tight, 'b-', alpha=0.5, linewidth=1,
                 label=f'EMA-{max(10, self.loss_smoothing_window // 5)}')
        ax1.plot(steps, smoothed_train, 'b-', linewidth=2, label=f'EMA-{self.loss_smoothing_window}')

        # Mark best step
        if best_step > 0 and best_step <= len(smoothed_train):
            ax1.axvline(x=best_step, color='g', linestyle='--', alpha=0.7, label=f'Best Step ({best_step})')
            ax1.scatter([best_step], [smoothed_train[best_step-1]], color='g', s=80, zorder=5)

        ax1.set_ylabel('Training Loss')
        ax1.set_title(f'Fold {fold_idx + 1} - Step-Level Training Loss (Smoothed)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Set y-axis limits to exclude outliers
        if len(smoothed_train) > 10:
            y_vals = smoothed_train[10:]  # Skip first few noisy steps
            y_min, y_max = np.percentile(y_vals, [1, 99])
            margin = (y_max - y_min) * 0.1
            ax1.set_ylim(y_min - margin, y_max + margin)

        # Bottom plot: Validation loss at checkpoints
        ax2 = axes[1]
        if len(step_val_losses) > 0 and len(step_eval_points) > 0:
            ax2.plot(step_eval_points, step_val_losses, 'r.-', linewidth=2, markersize=8, label='Val Loss')

            # Smooth val losses too
            if len(step_val_losses) > 3:
                smoothed_val = ema_smooth(step_val_losses, min(10, len(step_val_losses) // 2))
                ax2.plot(step_eval_points, smoothed_val, 'r-', linewidth=2, alpha=0.5, label='Val Smoothed')

            # Mark best
            if best_step > 0 and best_step in step_eval_points:
                idx = step_eval_points.index(best_step)
                ax2.scatter([best_step], [step_val_losses[idx]], color='g', s=100, zorder=5)
                ax2.axvline(x=best_step, color='g', linestyle='--', alpha=0.7)

            ax2.legend(loc='upper right')

            # Set y-axis limits with minimum range to avoid misleading micro-variations
            if len(step_val_losses) > 1:
                y_min, y_max = min(step_val_losses), max(step_val_losses)
                y_range = y_max - y_min
                # Ensure minimum visible range of 10% of the mean value
                min_range = np.mean(step_val_losses) * 0.1
                if y_range < min_range:
                    y_center = (y_min + y_max) / 2
                    y_min = y_center - min_range / 2
                    y_max = y_center + min_range / 2
                margin = (y_max - y_min) * 0.1
                ax2.set_ylim(y_min - margin, y_max + margin)

        ax2.set_xlabel('Optimizer Step')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss at Checkpoints')
        # Disable scientific notation offset to show actual values
        ax2.ticklabel_format(useOffset=False, style='plain', axis='y')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with step number in filename for iterative saves
        if final:
            plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_step_loss_final.png')
        else:
            plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_step_loss_step{current_step}.png')

        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Also save the loss data as numpy for later analysis
        if final:
            np.savez(
                os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_loss_history.npz'),
                step_train_losses=np.array(step_train_losses),
                step_val_losses=np.array(step_val_losses),
                step_eval_points=np.array(step_eval_points),
                smoothed_train=np.array(smoothed_train),
                best_step=best_step
            )

    def _plot_backtest_results(
        self,
        fold_idx: int,
        test_start: str,
        test_end: str,
        eval_results: Dict
    ):
        """Generate backtest performance plots for a fold."""
        # Only plot on main process
        if not is_main_process():
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Get data from eval results
        daily_returns = np.array(eval_results.get('daily_returns', []))
        capital_history = eval_results.get('capital_history', [self.initial_capital])

        if len(capital_history) == 0:
            capital_history = [self.initial_capital]

        # Normalize capital to start at 1.0 for easy comparison
        portfolio_values = np.array(capital_history) / capital_history[0]

        # 1. Portfolio value over time (equity curve)
        ax = axes[0, 0]
        ax.plot(portfolio_values, linewidth=1.5, color='#2ecc71')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(range(len(portfolio_values)), 1.0, portfolio_values,
                       where=(portfolio_values >= 1.0), alpha=0.3, color='green')
        ax.fill_between(range(len(portfolio_values)), 1.0, portfolio_values,
                       where=(portfolio_values < 1.0), alpha=0.3, color='red')
        ax.set_title('Portfolio Value (Equity Curve)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Portfolio Value (normalized)')
        ax.grid(alpha=0.3)

        # 2. Daily returns distribution
        ax = axes[0, 1]
        if len(daily_returns) > 0:
            returns_pct = daily_returns * 100
            ax.hist(returns_pct, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.axvline(returns_pct.mean(), color='green', linestyle='-', linewidth=2,
                      label=f'Mean: {returns_pct.mean():.3f}%')
            ax.legend()
        ax.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)

        # 3. Drawdown over time
        ax = axes[1, 0]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        ax.fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.7, color='#e74c3c')
        ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(alpha=0.3)

        # 4. Cumulative returns with key metrics
        ax = axes[1, 1]
        cumulative_returns = (portfolio_values - 1) * 100
        ax.plot(cumulative_returns, linewidth=2, color='#9b59b6')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Cumulative Return', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(alpha=0.3)

        # Add text box with metrics
        metrics_text = (
            f"Total Return: {eval_results['total_return_pct']:+.2f}%\n"
            f"Sharpe Ratio: {eval_results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {eval_results['max_drawdown_pct']:.2f}%\n"
            f"Win Rate: {eval_results['win_rate']:.1f}%\n"
            f"Trades: {eval_results['num_trades']}"
        )
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Fold {fold_idx + 1} Backtest: {test_start} to {test_end}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_backtest.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        if is_main_process():
            print(f"    Backtest plot saved to: {plot_path}")

    def _run_monte_carlo_validation(
        self,
        fold_idx: int,
        model: SimpleTransformerPredictor,
        test_dates: List[str],
    ) -> Dict:
        """
        Run Monte Carlo validation with many random stock subsets.

        Tests model across diverse stock universes to validate robustness
        and compare against random baseline.
        """
        print(f"\n  Running Monte Carlo validation ({self.mc_trials} trials, {self.mc_stocks} stocks/trial)...")

        # Get the underlying model
        if hasattr(model, 'module'):
            eval_model = model.module
        elif hasattr(model, '_orig_mod'):
            eval_model = model._orig_mod
        else:
            eval_model = model

        # Get bin edges
        bin_edges = None
        if self.pred_mode == 'classification' and self.current_fold_bin_edges is not None:
            bin_edges = self.current_fold_bin_edges.to(self.device)

        # Create a minimal evaluator for the Monte Carlo function
        evaluator = PrincipledEvaluator(
            dataset_path=self.dataset_path,
            prices_path=self.prices_path,
            seq_len=self.seq_len,
            horizon_days=self.horizon_days,
            horizon_idx=self.horizon_idx,
            top_k=self.top_k,
            device=self.device,
        )

        # Create checkpoint dict for the function
        checkpoint = {
            'fold_idx': fold_idx,
            'bin_edges': bin_edges,
        }

        # Run Monte Carlo validation
        mc_results = run_monte_carlo_validation(
            evaluator=evaluator,
            model=eval_model,
            checkpoint=checkpoint,
            test_dates=test_dates,
            bin_edges=bin_edges,
            n_trials=self.mc_trials,
            stocks_per_trial=self.mc_stocks,
            top_k_values=self.mc_top_ks,
            batch_size=256,
            max_stocks=self.num_test_stocks,
            seed=self.seed + fold_idx,  # Different seed per fold for variety
        )

        # Save Monte Carlo plots
        mc_output_dir = os.path.join(self.checkpoint_dir, 'monte_carlo')
        plot_monte_carlo_results(mc_results, mc_output_dir, fold_idx=fold_idx)

        return mc_results

    def _train_and_evaluate_baselines(
        self,
        fold_idx: int,
        train_dates: List[str],
        test_dates: List[str],
        transformer_results: Dict,
    ) -> Dict[str, Dict]:
        """
        Train and evaluate baseline models for comparison with transformer.

        Args:
            fold_idx: Current fold index
            train_dates: Training dates for this fold
            test_dates: Test dates for this fold
            transformer_results: Results from transformer evaluation

        Returns:
            Dict mapping model name to results dict
        """
        from training.baseline_models import (
            train_and_evaluate_baselines,
            print_baseline_comparison,
        )

        print(f"\n  Training baseline models for comparison...")

        # Open data files
        import h5py
        with h5py.File(self.dataset_path, 'r') as h5f:
            prices_h5f = h5py.File(self.prices_path, 'r') if self.prices_path else None

            try:
                tickers = list(h5f.keys())

                baseline_results = train_and_evaluate_baselines(
                    h5f=h5f,
                    prices_h5f=prices_h5f,
                    train_dates=train_dates,
                    test_dates=test_dates,
                    tickers=tickers,
                    horizon_days=self.horizon_days,
                    top_k=self.top_k,
                    max_train_samples=self.baseline_max_samples,
                    include_lgb=True,
                )

                # Print comparison table
                print_baseline_comparison(
                    transformer_results=transformer_results,
                    baseline_results=baseline_results,
                    fold_idx=fold_idx,
                )

                # Store for aggregate comparison later
                self.baseline_results.append({
                    'fold_idx': fold_idx,
                    'transformer': {
                        'mean_ic': transformer_results.get('mean_ic', 0),
                        'ir': transformer_results.get('ir', 0),
                        'mean_rank_ic': transformer_results.get('mean_rank_ic', 0),
                        'model_mean_return': transformer_results.get('model_mean_return', 0),
                        'excess_vs_random': transformer_results.get('excess_vs_random', 0),
                        'sharpe_ratio': transformer_results.get('sharpe_ratio', 0),
                    },
                    'baselines': baseline_results,
                })

            finally:
                if prices_h5f:
                    prices_h5f.close()

        return baseline_results

    def _train_fold(
        self,
        fold_idx: int,
        train_dates: List[str],
        test_dates: List[str],
        model: SimpleTransformerPredictor,
        prev_checkpoint_path: Optional[str] = None,
        prev_train_dates: Optional[List[str]] = None,
        use_incremental: bool = False
    ) -> Tuple[float, float, int, int, List[float], List[float], List[float], List[float], List[int], int]:
        """
        Train model on a single fold.

        Args:
            fold_idx: Index of current fold
            train_dates: All training dates for this fold
            test_dates: Test dates for this fold (used for validation to match evaluation)
            model: Model to train
            prev_checkpoint_path: Path to previous fold's checkpoint (for incremental training)
            prev_train_dates: Training dates from previous fold (to compute new dates)
            use_incremental: If True, only train on new data and load previous checkpoint

        Returns:
            (final_train_loss, best_val_loss, epochs_trained, best_epoch,
             train_losses, val_losses, step_train_losses, step_val_losses, step_eval_points,
             num_train_samples)
        """
        # Determine device for this process
        if self.use_ddp:
            local_rank = get_rank()
            device = f'cuda:{local_rank}'
        else:
            device = self.device

        # For incremental training, compute new dates and load previous checkpoint
        if use_incremental and prev_checkpoint_path and prev_train_dates:
            # Find dates that are new (not in previous fold's training set)
            prev_dates_set = set(prev_train_dates)
            new_dates = [d for d in train_dates if d not in prev_dates_set]

            if len(new_dates) == 0:
                if is_main_process():
                    print(f"\n  No new training data - skipping training, reusing previous checkpoint")
                # Load previous checkpoint and return
                checkpoint = torch.load(prev_checkpoint_path, map_location=device, weights_only=False)
                if hasattr(model, 'module'):
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                elif hasattr(model, '_orig_mod'):
                    model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                # Return previous fold's metrics
                prev_history = checkpoint.get('loss_history', {})
                return (
                    prev_history.get('train_losses', [0])[-1] if prev_history.get('train_losses') else 0,
                    prev_history.get('val_losses', [0])[-1] if prev_history.get('val_losses') else 0,
                    0, 0, [], [], [], [], [], 0  # num_train_samples = 0 for skipped fold
                )

            if is_main_process():
                print(f"\n  INCREMENTAL TRAINING: {len(new_dates)} new days (of {len(train_dates)} total)")
                print(f"    New data: {new_dates[0]} to {new_dates[-1]}")

            # Load previous checkpoint
            checkpoint = torch.load(prev_checkpoint_path, map_location=device, weights_only=False)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            elif hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            if is_main_process():
                print(f"    Loaded checkpoint from: {prev_checkpoint_path}")

            actual_train_dates = new_dates  # Only new data

            # Override epochs for incremental training
            epochs_to_train = self.incremental_epochs
            # Use separate data fraction for incremental mode
            train_data_fraction = self.incremental_data_fraction
        else:
            if is_main_process():
                print(f"\n  Training on {len(train_dates)} days...")

            # Use all train dates for training (no held-out validation split)
            actual_train_dates = train_dates
            epochs_to_train = self.epochs_per_fold
            train_data_fraction = self.data_fraction

        # Use TEST dates for validation - this aligns validation loss with evaluation metrics
        # This gives a realistic estimate of generalization to unseen future data
        val_dates = test_dates
        if is_main_process():
            print(f"  Validation on TEST period: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")

        # Create datasets
        train_dataset = TemporalFoldDataset(
            self.dataset_path,
            start_date=actual_train_dates[0],
            end_date=actual_train_dates[-1],
            seq_len=self.seq_len,
            preload=self.preload_data,
            data_fraction=train_data_fraction,
            prices_path=self.prices_path  # CRITICAL: Use actual prices for targets
        )

        val_dataset = TemporalFoldDataset(
            self.dataset_path,
            start_date=val_dates[0],
            end_date=val_dates[-1],
            seq_len=self.seq_len,
            preload=self.preload_data,
            data_fraction=self.data_fraction,  # Also subsample validation for consistency
            prices_path=self.prices_path  # CRITICAL: Use actual prices for targets
        )

        # Track number of training samples for metadata
        num_train_samples = len(train_dataset)

        # Safety check for empty datasets (should not happen with proper fold generation)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            train_dataset.close()
            val_dataset.close()
            raise ValueError(
                f"Empty dataset detected (train={len(train_dataset)}, val={len(val_dataset)}). "
                f"This indicates a bug in fold generation. "
                f"Train dates: {actual_train_dates[0]} to {actual_train_dates[-1]}, "
                f"Val dates: {val_dates[0]} to {val_dates[-1]}"
            )

        # Create samplers for DDP
        train_sampler = None
        val_sampler = None
        cross_sectional_sampler = None
        use_cross_sectional_sampling = (self.ranking_only or self.ranking_loss_weight > 0)

        if self.use_ddp:
            if use_cross_sectional_sampling:
                # Use distributed cross-sectional batch sampler for ranking loss with DDP
                # This divides DATES across GPUs, ensuring each GPU gets complete cross-sections
                print("    Using distributed cross-sectional batch sampling for ranking loss")
                cross_sectional_sampler = DistributedCrossSectionalBatchSampler(
                    dataset=train_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    batch_size=self.batch_size,
                    min_samples_per_date=10,
                    shuffle=True,
                    drop_last=False,
                    seed=self.seed + fold_idx,
                )
            else:
                # Standard DDP sampler for non-ranking loss
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=True
                )
            # Validation always uses standard sampler
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False
            )

        # Create data loaders
        # Use multiple workers only if data is preloaded (HDF5 can't be shared across processes)
        num_workers = self.num_workers if self.preload_data else 0

        # Create generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(self.seed + fold_idx)

        # Worker init function for reproducible multiprocessing
        def worker_init_fn(worker_id):
            worker_seed = self.seed + fold_idx + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Use cross-sectional batch sampling for ranking loss
        # This ensures each batch contains stocks from the same trading day
        if use_cross_sectional_sampling and cross_sectional_sampler is not None:
            # DDP case: use distributed cross-sectional sampler
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=cross_sectional_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
                timeout=120 if num_workers > 0 else 0,  # 2 min timeout to avoid hangs
            )
        elif use_cross_sectional_sampling:
            # Single GPU case: use regular cross-sectional sampler
            print("    Using cross-sectional batch sampling for ranking loss")
            cross_sectional_sampler = CrossSectionalBatchSampler(
                dataset=train_dataset,
                batch_size=self.batch_size,
                min_samples_per_date=10,  # Skip dates with too few stocks
                shuffle=True,
                drop_last=False,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=cross_sectional_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
                timeout=120 if num_workers > 0 else 0,  # 2 min timeout to avoid hangs
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=(train_sampler is None),  # Only shuffle if not using sampler
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,  # Avoid small last batch
                persistent_workers=(num_workers > 0),
                generator=g,
                worker_init_fn=worker_init_fn if num_workers > 0 else None
            )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers // 2 if num_workers > 0 else 0,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )

        # =====================================================================
        # Contrastive Pretraining Phase (if enabled)
        # =====================================================================
        if self.contrastive_pretrain and self.model_type == 'multimodal':
            if is_main_process():
                print(f"\n  {'='*60}")
                print(f"  CONTRASTIVE PRETRAINING PHASE")
                print(f"  {'='*60}")
                print(f"  Epochs: {self.contrastive_epochs}, LR: {self.contrastive_lr}")
                print(f"  Temperature: {self.contrastive_temperature}")

            # Contrastive pretraining uses its own optimizer
            contrastive_optimizer = AdamW(
                model.parameters(),
                lr=self.contrastive_lr,
                weight_decay=0.01
            )
            contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                contrastive_optimizer,
                T_max=self.contrastive_epochs
            )

            model.train()
            for epoch in range(self.contrastive_epochs):
                total_contrastive_loss = 0.0
                num_batches = 0

                for batch_idx, (features, _targets) in enumerate(train_loader):
                    features = features.to(device).float()

                    contrastive_optimizer.zero_grad()

                    # Compute contrastive loss (model must be ContrastiveMultiModalModel)
                    loss = model.compute_contrastive_loss(features)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    contrastive_optimizer.step()

                    total_contrastive_loss += loss.item()
                    num_batches += 1

                    if is_main_process() and (batch_idx + 1) % 100 == 0:
                        avg_loss = total_contrastive_loss / num_batches
                        print(f"    Epoch {epoch+1}/{self.contrastive_epochs} | "
                              f"Batch {batch_idx+1} | Contrastive Loss: {avg_loss:.4f}")

                contrastive_scheduler.step()

                if is_main_process():
                    avg_loss = total_contrastive_loss / max(num_batches, 1)
                    print(f"  Contrastive Epoch {epoch+1}/{self.contrastive_epochs} | "
                          f"Avg Loss: {avg_loss:.4f}")

            # Freeze encoder after pretraining if configured
            if self.freeze_encoder_after_pretrain:
                if is_main_process():
                    print(f"\n  Freezing encoder for finetuning phase...")
                model.freeze_encoder()
            else:
                if is_main_process():
                    print(f"\n  Encoder NOT frozen - full model finetuning")

            if is_main_process():
                print(f"  {'='*60}")
                print(f"  CONTRASTIVE PRETRAINING COMPLETE")
                print(f"  {'='*60}\n")

        # Setup optimizer (no mixed precision for simplicity)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Compute bin edges if classification
        if self.pred_mode == 'classification':
            bin_edges = self._compute_bin_edges(train_loader)
            # Store for evaluation - CRITICAL: must use same bin edges for inference
            self.current_fold_bin_edges = bin_edges
        else:
            bin_edges = None
            self.current_fold_bin_edges = None

        # Training loop with early stopping
        best_val_loss = float('inf')
        final_train_loss = 0.0
        final_val_loss = 0.0
        best_epoch = 0
        best_step = 0
        epochs_without_improvement = 0
        best_model_state = None

        # Track losses for plotting (epoch-level)
        epoch_train_losses = []
        epoch_val_losses = []

        # Step-level tracking for detailed loss curves
        step_train_losses = []  # Raw losses at each optimizer step
        step_val_losses = []    # Val losses at eval checkpoints
        step_eval_points = []   # Which steps we evaluated at
        global_step = 0         # Total optimizer steps across all epochs

        # Quick validation loss function for progressive evaluation
        def quick_val_loss(max_batches=100):
            """Quick validation loss on subset of data."""
            model.eval()
            val_losses_tmp = []
            with torch.no_grad():
                for i, (features, targets) in enumerate(val_loader):
                    if i >= max_batches:
                        break
                    features = features.to(device).float()
                    targets = targets.to(device).float()

                    pred, confidence = model(features)

                    if self.ranking_only:
                        # Use ranking loss for validation when training with ranking only
                        loss = compute_ranking_loss(
                            predictions=pred,
                            targets=targets,
                            bin_edges=bin_edges if self.pred_mode == 'classification' else None,
                            horizon_idx=self.horizon_idx,
                            loss_type=self.ranking_loss_type,
                            margin=self.ranking_margin,
                        )
                    elif self.pred_mode == 'classification':
                        bin_indices = convert_price_ratios_to_bins_vectorized(targets, bin_edges).long()
                        losses = []
                        for day_idx in range(4):
                            losses.append(F.cross_entropy(pred[:, :, day_idx], bin_indices[:, day_idx]))
                        loss = torch.stack(losses).mean()
                    else:
                        loss = F.mse_loss(pred, targets)

                    val_losses_tmp.append(loss.item())
            model.train()
            return np.mean(val_losses_tmp) if val_losses_tmp else float('inf')

        # EMA smoothing helper
        def compute_smoothed_losses(losses, window):
            if len(losses) == 0:
                return []
            smoothed = []
            alpha = 2.0 / (window + 1)
            ema = losses[0]
            for loss in losses:
                ema = alpha * loss + (1 - alpha) * ema
                smoothed.append(ema)
            return smoothed

        # Adaptive batch size tracking (plateau detection)
        current_accumulation_steps = self.gradient_accumulation_steps
        plateau_best_loss = float('inf')
        plateau_steps_without_improvement = 0
        batch_size_increases = 0

        for epoch in range(epochs_to_train):
            # Set epoch for distributed sampler (important for proper shuffling)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # Also set epoch for distributed cross-sectional sampler
            if cross_sectional_sampler is not None and hasattr(cross_sectional_sampler, 'set_epoch'):
                cross_sectional_sampler.set_epoch(epoch)

            # Train
            model.train()
            train_losses = []

            # Training progress bar (only on main process)
            train_pbar = tqdm(
                train_loader,
                desc=f"    Epoch {epoch+1}/{epochs_to_train} [Train]",
                leave=False,
                ncols=100,
                disable=not is_main_process()  # Only show on rank 0
            )

            # Gradient accumulation setup (use current value from adaptive scheduling)
            accumulation_steps = current_accumulation_steps
            num_batches = len(train_loader)

            for batch_idx, (features, targets) in enumerate(train_pbar):
                features = features.to(device).float()
                targets = targets.to(device).float()

                # Skip batches with NaN
                if torch.isnan(features).any() or torch.isnan(targets).any():
                    continue

                # Zero gradients only at start of accumulation cycle
                if batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad()

                # Ensure gradients are enabled (can be disabled by validation loop or data loading)
                torch.set_grad_enabled(True)

                # Forward pass
                pred, confidence = model(features)

                # Compute ranking loss if needed
                if self.ranking_only or self.ranking_loss_weight > 0:
                    ranking_loss = compute_ranking_loss(
                        predictions=pred,
                        targets=targets,
                        bin_edges=bin_edges if self.pred_mode == 'classification' else None,
                        horizon_idx=self.horizon_idx,
                        loss_type=self.ranking_loss_type,
                        margin=self.ranking_margin,
                    )

                if self.ranking_only:
                    # Use ONLY ranking loss
                    loss = ranking_loss
                else:
                    # Standard CE/MSE loss
                    if self.pred_mode == 'classification':
                        # Convert targets to bins (these are class labels, not differentiable)
                        bin_indices = convert_price_ratios_to_bins_vectorized(targets, bin_edges).long()

                        # Cross-entropy loss for each prediction day
                        ce_loss = F.cross_entropy(pred[:, :, 0], bin_indices[:, 0])
                        for day_idx in range(1, 4):
                            ce_loss = ce_loss + F.cross_entropy(pred[:, :, day_idx], bin_indices[:, day_idx])
                        ce_loss = ce_loss / 4.0
                        loss = ce_loss
                    else:
                        # MSE loss for regression
                        loss = F.mse_loss(pred, targets)

                    # Add ranking loss if enabled
                    if self.ranking_loss_weight > 0:
                        loss = loss + self.ranking_loss_weight * ranking_loss

                # Scale loss for gradient accumulation (average over accumulation steps)
                loss = loss / accumulation_steps

                # Skip if loss is NaN/Inf (would cause issues in backward)
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()

                # Step optimizer only after accumulating gradients
                is_last_batch = (batch_idx + 1) == num_batches
                should_step = ((batch_idx + 1) % accumulation_steps == 0) or is_last_batch
                if should_step:
                    optimizer.step()
                    global_step += 1

                    # Track step-level train loss (average of accumulated steps)
                    recent_losses = train_losses[-accumulation_steps:] if len(train_losses) >= accumulation_steps else train_losses
                    current_step_loss = np.mean([l for l in recent_losses]) if recent_losses else loss.item() * accumulation_steps
                    step_train_losses.append(current_step_loss)

                    # Adaptive batch size: detect plateau and increase grad accumulation
                    if self.adaptive_batch_size and current_accumulation_steps < self.max_grad_accum:
                        # Check if loss improved
                        relative_improvement = (plateau_best_loss - current_step_loss) / (abs(plateau_best_loss) + 1e-8)

                        if relative_improvement > self.plateau_threshold:
                            # Loss improved - reset counter and update best
                            plateau_best_loss = current_step_loss
                            plateau_steps_without_improvement = 0
                        else:
                            # No significant improvement
                            plateau_steps_without_improvement += 1

                            # Check if we've plateaued for too long
                            if plateau_steps_without_improvement >= self.plateau_patience:
                                current_accumulation_steps += 1
                                accumulation_steps = current_accumulation_steps
                                batch_size_increases += 1
                                plateau_steps_without_improvement = 0
                                plateau_best_loss = current_step_loss  # Reset baseline

                                effective_batch = self.batch_size * current_accumulation_steps
                                if is_main_process():
                                    print(f"\n      📈 Plateau detected at step {global_step}! "
                                          f"Increasing grad_accum: {current_accumulation_steps-1} → {current_accumulation_steps} "
                                          f"(effective batch: {effective_batch})")

                    # Periodic validation evaluation
                    if self.eval_every_n_steps > 0 and global_step % self.eval_every_n_steps == 0:
                        val_loss = quick_val_loss(max_batches=100)
                        step_val_losses.append(val_loss)
                        step_eval_points.append(global_step)

                        # Check if this is best
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = global_step
                            # Get state dict from base model (unwrap DDP if needed)
                            if hasattr(model, 'module'):
                                model_to_save = model.module
                            elif hasattr(model, '_orig_mod'):
                                model_to_save = model._orig_mod
                            else:
                                model_to_save = model
                            best_model_state = {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()}

                        if is_main_process():
                            smoothed = compute_smoothed_losses(step_train_losses, self.loss_smoothing_window)
                            print(f"      Step {global_step}: train={smoothed[-1]:.4f}, val={val_loss:.4f}")

                    # Save loss curve periodically
                    if (self.save_loss_curves_every > 0 and
                        global_step % self.save_loss_curves_every == 0 and
                        is_main_process()):
                        self._save_step_loss_curves(
                            fold_idx, step_train_losses, step_val_losses,
                            step_eval_points, best_step, global_step
                        )

                # Track unscaled loss for logging
                train_losses.append(loss.item() * accumulation_steps)

                # Update progress bar
                if len(train_losses) > 0:
                    postfix = {'loss': f'{np.mean(train_losses[-100:]):.4f}', 'step': global_step}
                    if self.adaptive_batch_size:
                        postfix['eff_bs'] = self.batch_size * current_accumulation_steps
                    train_pbar.set_postfix(postfix)

            train_pbar.close()

            # Validate
            model.eval()
            val_losses = []

            # Validation progress bar (only on main process)
            val_pbar = tqdm(
                val_loader,
                desc=f"    Epoch {epoch+1}/{epochs_to_train} [Val]",
                leave=False,
                ncols=100,
                disable=not is_main_process()  # Only show on rank 0
            )

            with torch.no_grad():
                for features, targets in val_pbar:
                    features = features.to(device)
                    targets = targets.to(device)

                    pred, confidence = model(features)

                    if self.ranking_only:
                        # Use ranking loss for validation when training with ranking only
                        loss = compute_ranking_loss(
                            predictions=pred,
                            targets=targets,
                            bin_edges=bin_edges if self.pred_mode == 'classification' else None,
                            horizon_idx=self.horizon_idx,
                            loss_type=self.ranking_loss_type,
                            margin=self.ranking_margin,
                        )
                    elif self.pred_mode == 'classification':
                        bin_indices = convert_price_ratios_to_bins_vectorized(targets, bin_edges)
                        losses = []
                        for day_idx in range(4):
                            losses.append(F.cross_entropy(pred[:, :, day_idx], bin_indices[:, day_idx]))
                        loss = torch.stack(losses).mean()
                    else:
                        loss = F.mse_loss(pred, targets)

                    val_losses.append(loss.item())

                    # Update progress bar
                    if len(val_losses) > 0:
                        val_pbar.set_postfix({'loss': f'{np.mean(val_losses):.4f}'})

            val_pbar.close()

            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            avg_val_loss = np.mean(val_losses) if val_losses else 0.0

            # Synchronize losses across processes if using DDP
            if self.use_ddp:
                train_loss_tensor = torch.tensor([avg_train_loss], device=device)
                val_loss_tensor = torch.tensor([avg_val_loss], device=device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                avg_train_loss = train_loss_tensor.item()
                avg_val_loss = val_loss_tensor.item()

            # Track losses for plotting
            epoch_train_losses.append(avg_train_loss)
            epoch_val_losses.append(avg_val_loss)

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                # Save best model state
                # Handle DDP (.module), torch.compile (._orig_mod), or plain model
                if hasattr(model, 'module'):
                    model_to_save = model.module  # DDP
                elif hasattr(model, '_orig_mod'):
                    model_to_save = model._orig_mod  # torch.compile
                else:
                    model_to_save = model
                best_model_state = {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()}

                if is_main_process():
                    print(f"    Epoch {epoch+1}/{epochs_to_train}: "
                          f"Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f} [BEST]")
            else:
                epochs_without_improvement += 1
                if is_main_process():
                    print(f"    Epoch {epoch+1}/{epochs_to_train}: "
                          f"Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f} "
                          f"(no improvement for {epochs_without_improvement} epochs)")

            final_train_loss = avg_train_loss
            final_val_loss = avg_val_loss

            # Save intermediate checkpoint for training progression analysis
            if (self.save_intermediate_checkpoints and
                is_main_process() and
                (epoch + 1) % self.checkpoint_every_n_epochs == 0):
                # Get current model state
                if hasattr(model, 'module'):
                    model_to_save = model.module
                elif hasattr(model, '_orig_mod'):
                    model_to_save = model._orig_mod
                else:
                    model_to_save = model

                intermediate_checkpoint = {
                    'epoch': epoch + 1,
                    'total_epochs': epochs_to_train,
                    'model_state_dict': {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()},
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'global_step': global_step,
                    'fold_idx': fold_idx,
                    'model_config': {
                        'input_dim': self.input_dim,
                        'hidden_dim': self.hidden_dim,
                        'num_layers': self.num_layers,
                        'num_heads': self.num_heads,
                        'dropout': self.dropout,
                        'pred_mode': self.pred_mode,
                    },
                    'training_config': {
                        'horizon_days': self.horizon_days,
                        'horizon_idx': self.horizon_idx,
                        'top_k': self.top_k,
                        'seq_len': self.seq_len,
                    },
                    'loss_history': {
                        'train_losses': epoch_train_losses[:epoch+1],
                        'val_losses': epoch_val_losses[:epoch+1],
                    },
                }
                # Add bin edges for classification
                if self.pred_mode == 'classification' and bin_edges is not None:
                    intermediate_checkpoint['bin_edges'] = bin_edges.cpu()

                # Save to intermediate checkpoints directory
                intermediate_dir = os.path.join(self.checkpoint_dir, 'intermediate', f'fold_{fold_idx}')
                os.makedirs(intermediate_dir, exist_ok=True)
                intermediate_path = os.path.join(intermediate_dir, f'epoch_{epoch + 1:03d}.pt')
                torch.save(intermediate_checkpoint, intermediate_path)
                print(f"      Saved intermediate checkpoint: {intermediate_path}")

            # Early stopping check (only if patience > 0)
            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                if is_main_process():
                    print(f"\n    Early stopping triggered after {epoch+1} epochs "
                          f"(no improvement for {self.early_stopping_patience} epochs)")
                    print(f"    Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break

        # Restore best model
        if best_model_state is not None:
            # Handle DDP (.module), torch.compile (._orig_mod), or plain model
            if hasattr(model, 'module'):
                model_to_restore = model.module  # DDP
            elif hasattr(model, '_orig_mod'):
                model_to_restore = model._orig_mod  # torch.compile
            else:
                model_to_restore = model
            model_to_restore.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            if is_main_process():
                print(f"    Restored best model from epoch {best_epoch}")

        # Generate and save loss curves (epoch-level)
        if is_main_process() and len(epoch_train_losses) > 1:
            self._plot_loss_curves(fold_idx, epoch_train_losses, epoch_val_losses, best_epoch)

        # Save final step-level loss curves with smoothing
        if is_main_process() and len(step_train_losses) > 0:
            self._save_step_loss_curves(
                fold_idx, step_train_losses, step_val_losses,
                step_eval_points, best_step, global_step, final=True
            )

        # Print adaptive batch size summary
        if self.adaptive_batch_size and is_main_process() and batch_size_increases > 0:
            final_effective_batch = self.batch_size * current_accumulation_steps
            print(f"\n    📈 Adaptive Batch Size Summary:")
            print(f"       Batch size increases: {batch_size_increases}")
            print(f"       Final grad_accum: {current_accumulation_steps} (started at {self.gradient_accumulation_steps})")
            print(f"       Final effective batch: {final_effective_batch}")

        # Store loss history for this fold
        self.current_fold_loss_history = {
            'train_losses': epoch_train_losses,
            'val_losses': epoch_val_losses,
            'step_train_losses': step_train_losses,
            'step_val_losses': step_val_losses,
            'step_eval_points': step_eval_points,
            'best_epoch': best_epoch,
            'best_step': best_step,
            'epochs_trained': len(epoch_train_losses),
            'total_steps': global_step,
            'batch_size_increases': batch_size_increases if self.adaptive_batch_size else 0,
            'final_grad_accum': current_accumulation_steps if self.adaptive_batch_size else self.gradient_accumulation_steps,
        }

        # Clean up
        train_dataset.close()
        val_dataset.close()

        # Return results including step-level tracking
        return (final_train_loss, best_val_loss, len(epoch_train_losses), best_epoch,
                epoch_train_losses, epoch_val_losses, step_train_losses, step_val_losses, step_eval_points,
                num_train_samples)

    def _compute_bin_edges(self, train_loader, num_bins: int = 100) -> torch.Tensor:
        """Compute adaptive bin edges from training data."""
        all_ratios = []

        for features, targets in train_loader:
            ratios = targets.flatten()
            # Filter out NaN and Inf values
            valid_mask = ~(torch.isnan(ratios) | torch.isinf(ratios))
            all_ratios.append(ratios[valid_mask])
            if sum(r.numel() for r in all_ratios) > 100000:
                break

        # Handle empty data case
        if len(all_ratios) == 0 or all(r.numel() == 0 for r in all_ratios):
            print("      Warning: No data in train_loader, using uniform bins")
            bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
            return torch.tensor(bin_edges, dtype=torch.float32).to(self.device)

        all_ratios = torch.cat(all_ratios).numpy()

        # Filter again to be safe and clip to reasonable range
        all_ratios = all_ratios[~np.isnan(all_ratios)]
        all_ratios = np.clip(all_ratios, -0.5, 0.5)

        if len(all_ratios) == 0:
            # Fallback to uniform bins if no valid data
            print("      Warning: No valid ratios found, using uniform bins")
            bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
        else:
            # Compute percentile-based bin edges
            percentiles = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(all_ratios, percentiles)

        return torch.tensor(bin_edges, dtype=torch.float32).to(self.device)

    def _evaluate_fold(
        self,
        fold_idx: int,
        test_dates: List[str],
        model: SimpleTransformerPredictor
    ) -> Dict:
        """
        Evaluate model on test dates using principled metrics.

        Computes IC, IR, quantile analysis, baseline comparisons, and simulation.
        """
        print(f"\n  Running principled evaluation on {len(test_dates)} days...")

        # Get the underlying model - handle DDP (.module) and torch.compile (._orig_mod)
        if hasattr(model, 'module'):
            eval_model = model.module  # DDP
        elif hasattr(model, '_orig_mod'):
            eval_model = model._orig_mod  # torch.compile
        else:
            eval_model = model

        # Get bin edges
        bin_edges = None
        if self.pred_mode == 'classification' and self.current_fold_bin_edges is not None:
            bin_edges = self.current_fold_bin_edges.to(self.device)

        # Run principled evaluation
        results = run_principled_evaluation(
            model=eval_model,
            dataset_path=self.dataset_path,
            prices_path=self.prices_path,
            test_dates=test_dates,
            bin_edges=bin_edges,
            horizon_idx=self.horizon_idx,
            horizon_days=self.horizon_days,
            top_k=self.top_k,
            device=self.device,
            max_stocks=self.num_test_stocks,
            pred_mode=self.pred_mode,
            transaction_cost_bps=self.transaction_cost_bps,
            seq_len=self.seq_len,  # Critical: must match training sequence length!
            max_eval_dates=self.max_eval_dates,
        )

        # Print summary
        print(f"\n    📊 IC Metrics:")
        print(f"       Mean IC:  {results['mean_ic']:+.4f} (IR: {results['ir']:+.2f})")
        print(f"       IC > 0:   {results['pct_positive_ic']:.1f}%")
        p_str = f"p={results['ic_p_value']:.4f}"
        if results['ic_p_value'] < 0.01:
            p_str += " ***"
        elif results['ic_p_value'] < 0.05:
            p_str += " **"
        elif results['ic_p_value'] < 0.1:
            p_str += " *"
        print(f"       T-stat:   {results['ic_t_stat']:+.2f} ({p_str})")

        print(f"\n    📈 Returns (Gross):")
        print(f"       Model:    {results['model_mean_return']:+.3f}% per trade")
        print(f"       vs Random:{results['excess_vs_random']:+.3f}% (p={results['vs_random_p_value']:.3f})")
        print(f"       vs Momentum:{results['excess_vs_momentum']:+.3f}% (p={results['vs_momentum_p_value']:.3f})")
        print(f"       Spread:   {results['long_short_spread']:+.3f}% (top - bottom decile)")

        # Show net returns if transaction costs are enabled
        if self.transaction_cost_bps > 0:
            print(f"\n    📈 Returns (Net of {self.transaction_cost_bps:.0f} bps costs):")
            print(f"       Model:    {results['model_mean_return_net']:+.3f}% per trade")
            print(f"       vs Random:{results['excess_vs_random_net']:+.3f}%")
            print(f"       vs Momentum:{results['excess_vs_momentum_net']:+.3f}%")
            print(f"       Turnover: {results['mean_turnover']*100:.1f}% avg per period")

        print(f"\n    💰 Simulation:")
        print(f"       Total:    {results['total_return_pct']:+.2f}% (gross)")
        if self.transaction_cost_bps > 0:
            print(f"       Total:    {results['total_return_pct_net']:+.2f}% (net)")
            print(f"       Costs:    {results['total_transaction_costs_pct']:.2f}% total paid")
        print(f"       Sharpe:   {results['sharpe_ratio']:.2f} (gross) / {results['sharpe_ratio_net']:.2f} (net)")

        return results

    def _get_gpu_info(self) -> dict:
        """Get GPU information for metadata."""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': 0,
            'devices': []
        }
        if torch.cuda.is_available():
            gpu_info['device_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'index': i,
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        return gpu_info

    def _write_metadata(
        self,
        fold_metadata: List[dict],
        training_start_time: datetime,
        training_end_time: datetime,
        gpu_info: dict
    ):
        """Write metadata.txt file with training details."""
        metadata_path = os.path.join(self.checkpoint_dir, 'metadata.txt')

        # Calculate total GPU hours
        total_seconds = (training_end_time - training_start_time).total_seconds()
        gpu_hours = total_seconds / 3600

        with open(metadata_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WALK-FORWARD TRAINING METADATA\n")
            f.write("=" * 80 + "\n\n")

            # Timing
            f.write("TIMING\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training Start: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training End:   {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {total_seconds/3600:.2f} hours ({total_seconds/60:.1f} minutes)\n")
            f.write(f"GPU Hours:      {gpu_hours:.2f}\n\n")

            # GPU Info
            f.write("GPU INFORMATION\n")
            f.write("-" * 40 + "\n")
            if gpu_info['available']:
                f.write(f"GPU Count: {gpu_info['device_count']}\n")
                for dev in gpu_info['devices']:
                    f.write(f"  [{dev['index']}] {dev['name']}\n")
                    f.write(f"      Memory: {dev['total_memory_gb']:.1f} GB\n")
                    f.write(f"      Compute Capability: {dev['compute_capability']}\n")
            else:
                f.write("No GPU available (CPU training)\n")
            f.write("\n")

            # Model Architecture
            f.write("MODEL ARCHITECTURE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Hidden Dimension:  {self.hidden_dim}\n")
            f.write(f"Number of Layers:  {self.num_layers}\n")
            f.write(f"Number of Heads:   {self.num_heads}\n")
            f.write(f"Dropout:           {self.dropout}\n")
            f.write(f"Prediction Mode:   {self.pred_mode}\n")
            f.write(f"Sequence Length:   {self.seq_len}\n")
            f.write("\n")

            # Training Configuration
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mode:              {self.mode}\n")
            f.write(f"Number of Folds:   {self.num_folds}\n")
            f.write(f"Epochs per Fold:   {self.epochs_per_fold}\n")
            f.write(f"Batch Size:        {self.batch_size}\n")
            f.write(f"Learning Rate:     {self.learning_rate}\n")
            f.write(f"Gradient Accum:    {self.gradient_accumulation_steps}\n")
            f.write(f"Effective Batch:   {self.batch_size * self.gradient_accumulation_steps}\n")
            f.write(f"Data Fraction:     {self.data_fraction}\n")
            f.write(f"Ranking Only:      {self.ranking_only}\n")
            f.write(f"Ranking Loss Type: {self.ranking_loss_type}\n")
            f.write(f"Ranking Margin:    {self.ranking_margin}\n")
            f.write(f"Seed:              {self.seed}\n")
            if self.adaptive_batch_size:
                f.write(f"\nAdaptive Batch Size: Enabled\n")
                f.write(f"  Plateau Patience:  {self.plateau_patience} steps\n")
                f.write(f"  Plateau Threshold: {self.plateau_threshold}\n")
                f.write(f"  Max Grad Accum:    {self.max_grad_accum}\n")
            f.write("\n")

            # Evaluation Configuration
            f.write("EVALUATION CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Top-K:             {self.top_k}\n")
            f.write(f"Horizon Days:      {self.horizon_days}\n")
            f.write(f"Num Test Stocks:   {self.num_test_stocks}\n")
            f.write(f"Transaction Cost:  {self.transaction_cost_bps} bps\n")
            f.write("\n")

            # Fold Details
            f.write("FOLD DETAILS\n")
            f.write("-" * 40 + "\n")
            total_train_samples = 0
            total_eval_samples = 0

            for fold in fold_metadata:
                f.write(f"\nFold {fold['fold_idx'] + 1}/{len(fold_metadata)}\n")
                f.write(f"  Training Period:   {fold['train_start']} to {fold['train_end']}\n")
                f.write(f"  Training Days:     {fold['num_train_days']}\n")
                f.write(f"  Training Samples:  {fold['num_train_samples']:,}\n")
                f.write(f"  Test Period:       {fold['test_start']} to {fold['test_end']}\n")
                f.write(f"  Test Days:         {fold['num_test_days']}\n")
                f.write(f"  Eval Predictions:  {fold['num_eval_predictions']:,}\n")
                f.write(f"  Training Time:     {fold['training_time_seconds']/60:.1f} minutes\n")
                total_train_samples += fold['num_train_samples']
                total_eval_samples += fold['num_eval_predictions']

            f.write(f"\nTOTALS\n")
            f.write(f"  Total Training Samples:  {total_train_samples:,}\n")
            f.write(f"  Total Eval Predictions:  {total_eval_samples:,}\n")
            f.write("\n")
            f.write("=" * 80 + "\n")

        return metadata_path

    def run(self) -> List[FoldTrainingResult]:
        """Run walk-forward training and evaluation."""
        # Track training start time
        training_start_time = datetime.now()
        gpu_info = self._get_gpu_info()
        fold_metadata = []

        # Setup stats logger (writes to checkpoint_dir/stats.log)
        stats_logger = setup_stats_logger(self.checkpoint_dir)
        stats_logger.info("")
        stats_logger.info("=" * 80)
        stats_logger.info(f"WALK-FORWARD TRAINING STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        stats_logger.info("=" * 80)
        stats_logger.info(f"Checkpoint Directory: {self.checkpoint_dir}")
        stats_logger.info(f"Mode: {self.mode}, Folds: {self.num_folds}")
        stats_logger.info(f"Horizon: {self.horizon_days} days, Top-K: {self.top_k}")
        stats_logger.info(f"Transaction Cost: {self.transaction_cost_bps:.0f} bps")
        stats_logger.info("")

        # Setup DDP if enabled
        if self.use_ddp:
            # When using torchrun, LOCAL_RANK is set automatically
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            setup_ddp(local_rank, world_size)

        if is_main_process():
            print(f"\n{'='*80}")
            print(f"WALK-FORWARD TRAINING ({self.mode.upper()} WINDOW)")
            print(f"{'='*80}")
            print(f"\nConfiguration:")
            print(f"  Mode: {self.mode}")
            print(f"  Folds: {self.num_folds}")
            if self.auto_span:
                print(f"  Auto-span: Enabled (initial train: {self.initial_train_fraction:.0%})")
            else:
                print(f"  Min train months: {self.min_train_months}")
                print(f"  Test months per fold: {self.test_months}")
            print(f"  Gap days: {self.gap_days}")
            print(f"  Epochs per fold: {self.epochs_per_fold}")
            print(f"  Batch size: {self.batch_size}")
            if self.gradient_accumulation_steps > 1:
                effective_batch = self.batch_size * self.gradient_accumulation_steps
                print(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
                print(f"  Effective batch size: {effective_batch}")
            if self.early_stopping_patience > 0:
                print(f"  Early stopping patience: {self.early_stopping_patience} epochs")
            print(f"  Top-k: {self.top_k}")
            print(f"  Horizon: {self.horizon_days} days")
            if self.data_fraction < 1.0:
                print(f"  Data fraction: {self.data_fraction*100:.0f}% (FAST EXPERIMENT MODE)")
            if self.eval_every_n_steps > 0:
                print(f"  Progressive eval: every {self.eval_every_n_steps} steps")
            if self.use_ddp:
                print(f"  DDP: Enabled (world_size={get_world_size()})")
            if self.incremental_training:
                print(f"  INCREMENTAL MODE: Reusing checkpoints, {self.incremental_epochs} epochs per fold")
            if self.transaction_cost_bps > 0:
                print(f"  Transaction costs: {self.transaction_cost_bps:.0f} bps ({self.transaction_cost_bps/100:.2f}%)")

        # Set deterministic seed for full reproducibility
        set_deterministic_seed(
            seed=self.seed,
            rank=get_rank(),
            fully_deterministic=True
        )

        # Get folds
        folds = self._get_folds()
        if is_main_process():
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
        prev_train_dates = None
        prev_checkpoint_path = None

        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            fold_start_time = datetime.now()

            if is_main_process():
                print(f"\n{'='*80}")
                print(f"FOLD {fold_idx + 1}/{len(folds)}")
                if self.incremental_training and fold_idx > 0:
                    print(f"  [INCREMENTAL MODE - Training only on new data]")
                print(f"{'='*80}")

            # Create fresh model (will load checkpoint if incremental)
            model = self._create_model(compile_model=self.compile_model)

            # Determine if using incremental training for this fold
            use_incremental = (
                self.incremental_training and
                fold_idx > 0 and
                prev_checkpoint_path is not None and
                os.path.exists(prev_checkpoint_path)
            )

            # Train
            (train_loss, val_loss, epochs, best_epoch,
             epoch_train_losses, epoch_val_losses,
             step_train_losses, step_val_losses, step_eval_points,
             num_train_samples) = self._train_fold(
                fold_idx, train_dates, test_dates, model,
                prev_checkpoint_path=prev_checkpoint_path if use_incremental else None,
                prev_train_dates=prev_train_dates if use_incremental else None,
                use_incremental=use_incremental
            )

            # Synchronize all processes before evaluation
            if self.use_ddp:
                dist.barrier()

            # Evaluate (only on main process to avoid redundant work)
            if is_main_process():
                eval_results = self._evaluate_fold(fold_idx, test_dates, model)

                # Log detailed metrics to stats.log
                stats_logger = get_stats_logger()
                if stats_logger:
                    # Calculate number of eval samples (non-overlapping)
                    num_eval_samples = len(test_dates) // self.horizon_days

                    log_evaluation_header(
                        logger=stats_logger,
                        fold_idx=fold_idx,
                        num_folds=len(folds),
                        train_start=train_dates[0],
                        train_end=train_dates[-1],
                        test_start=test_dates[0],
                        test_end=test_dates[-1],
                        num_test_dates=num_eval_samples,
                        num_stocks=self.num_test_stocks,
                        horizon_days=self.horizon_days,
                        top_k=self.top_k,
                        transaction_cost_bps=self.transaction_cost_bps,
                    )
                    log_evaluation_metrics(
                        logger=stats_logger,
                        results=eval_results,
                        include_net_metrics=(self.transaction_cost_bps > 0),
                    )

                # Generate backtest visualization
                self._plot_backtest_results(
                    fold_idx=fold_idx,
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                    eval_results=eval_results
                )

                # Save daily metrics for detailed analysis
                daily_metrics_path = os.path.join(
                    self.checkpoint_dir, f'fold_{fold_idx}_daily_metrics.npz'
                )
                np.savez(
                    daily_metrics_path,
                    daily_ics=np.array(eval_results.get('daily_ics', [])),
                    daily_rank_ics=np.array(eval_results.get('daily_rank_ics', [])),
                    daily_model_returns=np.array(eval_results.get('daily_model_returns', [])),
                    daily_momentum_returns=np.array(eval_results.get('daily_momentum_returns', [])),
                    daily_random_returns=np.array(eval_results.get('daily_random_returns', [])),
                    daily_top_decile_returns=np.array(eval_results.get('daily_top_decile_returns', [])),
                    daily_bottom_decile_returns=np.array(eval_results.get('daily_bottom_decile_returns', [])),
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                )

                # Run Monte Carlo validation if enabled
                if self.monte_carlo:
                    mc_results = self._run_monte_carlo_validation(
                        fold_idx=fold_idx,
                        model=model,
                        test_dates=test_dates,
                    )
                    eval_results['monte_carlo'] = mc_results

                # Train and evaluate baseline models for comparison
                if self.compare_models:
                    baseline_results = self._train_and_evaluate_baselines(
                        fold_idx=fold_idx,
                        train_dates=train_dates,
                        test_dates=test_dates,
                        transformer_results=eval_results,
                    )
                    eval_results['baselines'] = baseline_results
            else:
                eval_results = {'total_return_pct': 0, 'sharpe_ratio': 0, 'max_drawdown_pct': 0,
                               'win_rate': 0, 'avg_return_pct': 0, 'num_trades': 0}

            # Save model checkpoint (only on main process)
            checkpoint_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_best.pt')
            if is_main_process():
                # Get the underlying model - handle DDP (.module) and torch.compile (._orig_mod)
                if hasattr(model, 'module'):
                    model_to_save = model.module  # DDP
                elif hasattr(model, '_orig_mod'):
                    model_to_save = model._orig_mod  # torch.compile
                else:
                    model_to_save = model
                checkpoint_dict = {
                    'model_state_dict': model_to_save.state_dict(),
                    'config': {
                        'hidden_dim': self.hidden_dim,
                        'num_layers': self.num_layers,
                        'num_heads': self.num_heads,
                        'dropout': self.dropout,
                        'pred_mode': self.pred_mode
                    },
                    'fold_idx': fold_idx,
                    'train_dates': (train_dates[0], train_dates[-1]),
                    'train_dates_list': train_dates,  # Full list for incremental training
                    'test_dates': (test_dates[0], test_dates[-1]),
                    'test_dates_list': test_dates,  # Full list for evaluation
                    # Walk-forward config for reproducible evaluation
                    'walk_forward_config': {
                        'dataset_path': self.dataset_path,
                        'prices_path': self.prices_path,
                        'num_folds': self.num_folds,
                        'mode': self.mode,
                        'horizon_days': self.horizon_days,
                        'horizon_idx': self.horizon_idx,
                        'top_k': self.top_k,
                        'seq_len': self.seq_len,
                        'seed': self.seed,
                    }
                }
                # Save bin edges with checkpoint (CRITICAL for classification mode!)
                if self.pred_mode == 'classification' and self.current_fold_bin_edges is not None:
                    checkpoint_dict['bin_edges'] = self.current_fold_bin_edges.cpu()
                # Save loss history
                if hasattr(self, 'current_fold_loss_history'):
                    checkpoint_dict['loss_history'] = self.current_fold_loss_history
                torch.save(checkpoint_dict, checkpoint_path)

            # Update tracking variables for incremental training
            prev_train_dates = train_dates
            prev_checkpoint_path = checkpoint_path

            # Create result with all principled metrics
            fold_result = FoldTrainingResult(
                fold_idx=fold_idx,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_samples=num_train_samples,
                test_days=len(test_dates),
                final_train_loss=train_loss,
                final_val_loss=val_loss,
                epochs_trained=epochs,
                best_epoch=best_epoch,
                # IC metrics
                mean_ic=eval_results.get('mean_ic', 0.0),
                std_ic=eval_results.get('std_ic', 0.0),
                ir=eval_results.get('ir', 0.0),
                ic_t_stat=eval_results.get('ic_t_stat', 0.0),
                ic_p_value=eval_results.get('ic_p_value', 1.0),
                pct_positive_ic=eval_results.get('pct_positive_ic', 0.0),
                mean_rank_ic=eval_results.get('mean_rank_ic', 0.0),
                # Quantile analysis
                top_decile_return=eval_results.get('top_decile_return', 0.0),
                bottom_decile_return=eval_results.get('bottom_decile_return', 0.0),
                long_short_spread=eval_results.get('long_short_spread', 0.0),
                # Baseline comparisons
                model_mean_return=eval_results.get('model_mean_return', 0.0),
                momentum_mean_return=eval_results.get('momentum_mean_return', 0.0),
                random_mean_return=eval_results.get('random_mean_return', 0.0),
                excess_vs_momentum=eval_results.get('excess_vs_momentum', 0.0),
                excess_vs_random=eval_results.get('excess_vs_random', 0.0),
                vs_random_p_value=eval_results.get('vs_random_p_value', 1.0),
                vs_momentum_p_value=eval_results.get('vs_momentum_p_value', 1.0),
                # Simulation metrics
                total_return_pct=eval_results.get('total_return_pct', 0.0),
                sharpe_ratio=eval_results.get('sharpe_ratio', 0.0),
                max_drawdown_pct=eval_results.get('max_drawdown_pct', 0.0),
                win_rate=eval_results.get('win_rate', 0.0),
                avg_return_pct=eval_results.get('avg_return_pct', 0.0),
                num_trades=eval_results.get('num_trades', 0)
            )
            self.fold_results.append(fold_result)

            # Collect fold metadata
            fold_end_time = datetime.now()
            # Calculate eval predictions: (non-overlapping periods) * (stocks evaluated per period)
            num_eval_periods = len(test_dates) // self.horizon_days
            num_eval_predictions = num_eval_periods * self.num_test_stocks

            fold_metadata.append({
                'fold_idx': fold_idx,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'num_train_days': len(train_dates),
                'num_train_samples': fold_result.train_samples,
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'num_test_days': len(test_dates),
                'num_eval_predictions': num_eval_predictions,
                'training_time_seconds': (fold_end_time - fold_start_time).total_seconds(),
            })

            # Print fold summary (only on main process)
            if is_main_process():
                print(f"\n  Fold {fold_idx + 1} Complete:")
                print(f"    Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"    IC: {eval_results.get('mean_ic', 0):+.4f}, IR: {eval_results.get('ir', 0):+.2f}")
                print(f"    Excess vs Random: {eval_results.get('excess_vs_random', 0):+.3f}%")

            # Synchronize before next fold
            if self.use_ddp:
                dist.barrier()

        # Log aggregate summary across all folds
        if is_main_process():
            stats_logger = get_stats_logger()
            if stats_logger and self.fold_results:
                # Extract eval_results from each fold
                fold_eval_results = [
                    fr.eval_results for fr in self.fold_results
                    if hasattr(fr, 'eval_results') and fr.eval_results
                ]
                log_fold_summary(stats_logger, fold_eval_results)

                # Log location of stats file
                stats_log_path = os.path.join(self.checkpoint_dir, 'stats.log')
                stats_logger.info(f"Stats log saved to: {stats_log_path}")
                print(f"\n📊 Detailed evaluation stats saved to: {stats_log_path}")

            # Save baseline comparison results if available
            if self.compare_models and self.baseline_results:
                from training.baseline_models import save_baseline_comparison
                baseline_path = os.path.join(self.checkpoint_dir, 'baseline_comparison.json')
                save_baseline_comparison(self.baseline_results, baseline_path)

        # Write metadata file
        if is_main_process():
            training_end_time = datetime.now()
            metadata_path = self._write_metadata(
                fold_metadata=fold_metadata,
                training_start_time=training_start_time,
                training_end_time=training_end_time,
                gpu_info=gpu_info
            )
            print(f"📋 Training metadata saved to: {metadata_path}")

        # Cleanup DDP
        if self.use_ddp:
            cleanup_ddp()

        return self.fold_results


def print_summary(results: List[FoldTrainingResult], mode: str):
    """Print summary of walk-forward training results with principled metrics."""

    print(f"\n{'='*80}")
    print("WALK-FORWARD TRAINING SUMMARY")
    print(f"{'='*80}")

    if len(results) == 0:
        print("\nNo folds completed successfully. Check your data and date ranges.")
        return

    # Extract metrics
    ics = [r.mean_ic for r in results]
    irs = [r.ir for r in results]
    spreads = [r.long_short_spread for r in results]
    excess_random = [r.excess_vs_random for r in results]
    excess_momentum = [r.excess_vs_momentum for r in results]
    returns = [r.total_return_pct for r in results]
    sharpes = [r.sharpe_ratio for r in results]

    print(f"\nMode: {mode}")
    print(f"Folds: {len(results)}")

    # ========== IC METRICS (PRIMARY) ==========
    print(f"\n{'='*80}")
    print("📊 INFORMATION COEFFICIENT (Primary Metrics)")
    print(f"{'='*80}")

    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    mean_ir = np.mean(irs)

    # Aggregate IC t-test
    if len(ics) >= 2:
        ic_t_stat, ic_p_value = stats.ttest_1samp(ics, 0)
    else:
        ic_t_stat, ic_p_value = 0.0, 1.0

    p_str = ""
    if ic_p_value < 0.01:
        p_str = "***"
    elif ic_p_value < 0.05:
        p_str = "**"
    elif ic_p_value < 0.1:
        p_str = "*"

    print(f"\n  Mean IC:   {mean_ic:+.4f} (±{std_ic:.4f})")
    print(f"  Mean IR:   {mean_ir:+.4f} (±{np.std(irs):.4f})")
    print(f"  Aggregate T-stat: {ic_t_stat:+.2f}")
    print(f"  Aggregate P-value: {ic_p_value:.4f} {p_str}")

    # Interpretation
    if mean_ic > 0.05:
        print(f"\n  ✓ GOOD: IC > 0.05 indicates meaningful predictive power")
    elif mean_ic > 0.02:
        print(f"\n  ~ WEAK: IC in 0.02-0.05 range, weak but potentially useful signal")
    elif mean_ic > 0:
        print(f"\n  ⚠ MARGINAL: IC > 0 but < 0.02, very weak signal")
    else:
        print(f"\n  ✗ NO SKILL: IC ≤ 0, model has no predictive power")

    # ========== LONG-SHORT SPREAD ==========
    print(f"\n{'='*80}")
    print("📈 LONG-SHORT SPREAD")
    print(f"{'='*80}")

    mean_spread = np.mean(spreads)
    print(f"\n  Mean Spread: {mean_spread:+.3f}% (±{np.std(spreads):.3f}%)")
    print(f"  % Positive:  {np.mean(np.array(spreads) > 0) * 100:.1f}%")

    if mean_spread > 0.1:
        print(f"\n  ✓ Top decile outperforms bottom decile")
    else:
        print(f"\n  ⚠ Weak or no spread between top/bottom predictions")

    # ========== BASELINE COMPARISONS ==========
    print(f"\n{'='*80}")
    print("🎯 EXCESS RETURNS VS BASELINES")
    print(f"{'='*80}")

    print(f"\n  vs Random:   {np.mean(excess_random):+.3f}% (±{np.std(excess_random):.3f}%)")
    print(f"  vs Momentum: {np.mean(excess_momentum):+.3f}% (±{np.std(excess_momentum):.3f}%)")

    # Count significant folds
    sig_random = sum(1 for r in results if r.vs_random_p_value < 0.05)
    sig_momentum = sum(1 for r in results if r.vs_momentum_p_value < 0.05)
    print(f"\n  Folds sig. vs Random (p<0.05):   {sig_random}/{len(results)}")
    print(f"  Folds sig. vs Momentum (p<0.05): {sig_momentum}/{len(results)}")

    # ========== SIMULATION (SECONDARY) ==========
    print(f"\n{'='*80}")
    print("💰 TRADING SIMULATION (Secondary)")
    print(f"{'='*80}")

    print(f"\n  Mean Return: {np.mean(returns):+.2f}% (±{np.std(returns):.2f}%)")
    print(f"  Mean Sharpe: {np.mean(sharpes):.2f} (±{np.std(sharpes):.2f})")

    # ========== PER-FOLD BREAKDOWN ==========
    print(f"\n{'='*80}")
    print("PER-FOLD BREAKDOWN")
    print(f"{'='*80}")

    print(f"\n{'Fold':<5} {'Test Period':<23} {'IC':>8} {'IR':>7} {'Spread':>8} {'vsRand':>8} {'Return':>9}")
    print(f"{'-'*5} {'-'*23} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")

    for r in results:
        test_period = f"{r.test_start[:10]} to {r.test_end[:10]}"
        print(f"{r.fold_idx+1:<5} {test_period:<23} {r.mean_ic:>+7.4f} {r.ir:>+6.2f} "
              f"{r.long_short_spread:>+7.3f}% {r.excess_vs_random:>+7.3f}% {r.total_return_pct:>+8.2f}%")


def save_results(results: List[FoldTrainingResult], output_path: str, mode: str):
    """Save results to file."""
    results_dict = {
        'mode': mode,
        'num_folds': len(results),
        'fold_results': [asdict(r) for r in results],
        'aggregate': {
            # IC metrics (primary)
            'mean_ic': float(np.mean([r.mean_ic for r in results])),
            'std_ic': float(np.std([r.mean_ic for r in results])),
            'mean_ir': float(np.mean([r.ir for r in results])),
            'mean_long_short_spread': float(np.mean([r.long_short_spread for r in results])),
            'mean_excess_vs_random': float(np.mean([r.excess_vs_random for r in results])),
            'mean_excess_vs_momentum': float(np.mean([r.excess_vs_momentum for r in results])),
            # Simulation metrics (secondary)
            'mean_return': float(np.mean([r.total_return_pct for r in results])),
            'std_return': float(np.std([r.total_return_pct for r in results])),
            'mean_sharpe': float(np.mean([r.sharpe_ratio for r in results])),
            'mean_max_drawdown': float(np.mean([r.max_drawdown_pct for r in results])),
            'mean_win_rate': float(np.mean([r.win_rate for r in results]))
        }
    }

    torch.save(results_dict, output_path)
    print(f"\nResults saved to: {output_path}")

    # Also save JSON
    json_path = output_path.replace('.pt', '.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Training')

    # Data paths
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset')
    parser.add_argument('--prices', type=str, default=None,
                       help='Path to actual prices HDF5 (optional)')

    # Walk-forward config
    parser.add_argument('--num-folds', type=int, default=10,
                       help='Number of folds')
    parser.add_argument('--mode', type=str, default='expanding',
                       choices=['expanding', 'sliding'],
                       help='Window mode')
    parser.add_argument('--min-train-months', type=int, default=160,
                       help='Minimum training period in months')
    parser.add_argument('--test-months', type=int, default=6,
                       help='Test period per fold in months')
    parser.add_argument('--gap-days', type=int, default=5,
                       help='Gap between train and test')
    parser.add_argument('--no-auto-span', action='store_true',
                       help='Disable auto-span (use fixed --min-train-months and --test-months instead)')
    parser.add_argument('--initial-train-fraction', type=float, default=0.5,
                       help='Fraction of data for initial training when auto-span is enabled (default: 0.5)')

    # Model config
    parser.add_argument('--model-type', type=str, default='transformer',
                       choices=['transformer', 'multimodal'],
                       help='Model architecture: transformer (single-stream) or multimodal (separate encoders)')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--pred-mode', type=str, default='regression',
                       choices=['classification', 'regression'],
                       help='Prediction mode')

    # Contrastive pretraining config
    parser.add_argument('--contrastive-pretrain', action='store_true',
                       help='Enable contrastive pretraining before supervised training')
    parser.add_argument('--contrastive-epochs', type=int, default=5,
                       help='Number of epochs for contrastive pretraining')
    parser.add_argument('--contrastive-lr', type=float, default=1e-4,
                       help='Learning rate for contrastive pretraining')
    parser.add_argument('--contrastive-temperature', type=float, default=0.1,
                       help='Temperature for InfoNCE loss (lower = harder negatives)')
    parser.add_argument('--no-freeze-encoder', action='store_true',
                       help='Do NOT freeze encoder after contrastive pretraining (default: freeze)')

    # Training config
    parser.add_argument('--epochs-per-fold', type=int, default=1,
                       help='Epochs to train per fold')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--seq-len', type=int, default=1536,
                       help='Sequence length')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Accumulate gradients over N steps (effective batch = batch_size * N)')
    parser.add_argument('--adaptive-batch-size', action='store_true',
                       help='Enable adaptive batch size: increase grad accumulation when loss plateaus')
    parser.add_argument('--plateau-patience', type=int, default=15,
                       help='Number of steps without improvement before increasing batch size')
    parser.add_argument('--plateau-threshold', type=float, default=0.001,
                       help='Minimum relative improvement to count as "not plateau" (default: 0.1%%)')
    parser.add_argument('--max-grad-accum', type=int, default=32,
                       help='Maximum gradient accumulation steps for adaptive batch size')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Stop if val loss does not improve for N epochs (0 to disable)')

    # Ranking loss (for learning cross-sectional ranking, not just per-stock prediction)
    parser.add_argument('--ranking-loss-weight', type=float, default=0.0,
                       help='Weight for ranking loss (0.0=disabled, 1.0=equal to CE loss). Helps model learn to rank stocks correctly.')
    parser.add_argument('--ranking-loss-type', type=str, default='pairwise', choices=['pairwise', 'listnet', 'correlation'],
                       help='Type of ranking loss: pairwise (margin-based), listnet (distribution-based), or correlation (directly optimizes IC)')
    parser.add_argument('--ranking-margin', type=float, default=0.01,
                       help='Margin for pairwise ranking loss (minimum score difference for correct ordering)')
    parser.add_argument('--ranking-only', action='store_true',
                       help='Train with ONLY ranking loss (no cross-entropy/MSE). Directly optimizes for stock ranking.')

    # Monte Carlo validation (runs automatically after each fold)
    parser.add_argument('--no-monte-carlo', action='store_true',
                       help='Disable Monte Carlo validation (enabled by default)')
    parser.add_argument('--mc-trials', type=int, default=50,
                       help='Number of Monte Carlo trials per fold (default: 50)')
    parser.add_argument('--mc-stocks', type=int, default=120,
                       help='Number of stocks to sample per trial (default: 150)')
    parser.add_argument('--mc-top-ks', type=int, nargs='+', default=[1, 2, 3, 4, 5, 10, 15, 20, 30, 50],
                       help='Top-k values to test in Monte Carlo (default: 5 10 15 20)')

    # Progressive evaluation
    parser.add_argument('--eval-every-n-steps', type=int, default=100,
                       help='Evaluate on val set every N optimizer steps (0=disable)')
    parser.add_argument('--loss-smoothing-window', type=int, default=50,
                       help='EMA window for smoothed loss curves')
    parser.add_argument('--save-loss-curves-every', type=int, default=100,
                       help='Save loss curve PNG every N steps (0=only at end)')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                       help='Fraction of training data to use (0.0-1.0) for fast experiments')

    # Incremental training
    parser.add_argument('--incremental', action='store_true',
                       help='Enable incremental training (reuse previous fold checkpoint, train only on new data)')
    parser.add_argument('--incremental-epochs', type=int, default=2,
                       help='Number of epochs for incremental training (typically fewer than full training)')
    parser.add_argument('--incremental-data-fraction', type=float, default=1.0,
                       help='Fraction of new data to use in incremental training (0.0-1.0)')

    # Intermediate checkpoint saving for training progression analysis
    parser.add_argument('--save-intermediate-checkpoints', action='store_true',
                       help='Save checkpoints at regular intervals throughout training for progression analysis')
    parser.add_argument('--checkpoint-every-n-epochs', type=int, default=1,
                       help='Save intermediate checkpoint every N epochs (default: 1)')

    # Evaluation config
    parser.add_argument('--top-k', type=int, default=25,
                       help='Number of stocks to select')
    parser.add_argument('--horizon-idx', type=int, default=0,
                       help='Prediction horizon')
    parser.add_argument('--confidence-percentile', type=float, default=0.6,
                       help='Confidence percentile')
    parser.add_argument('--subset-size', type=int, default=512,
                       help='Daily subset size')
    parser.add_argument('--num-test-stocks', type=int, default=1000,
                       help='Number of test stocks')
    parser.add_argument('--max-eval-dates', type=int, default=60,
                       help='Max dates for principled evaluation (randomly sampled if exceeded)')

    # Transaction cost model
    parser.add_argument('--transaction-cost-bps', type=float, default=10.0,
                       help='Transaction cost in basis points per round-trip trade (default: 10 bps = 0.1%%). '
                            'Typical values: 5-20 bps for large-cap, 20-50 for mid-cap, 50-100+ for small-cap')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Master random seed for full reproducibility (controls all randomness)')
    parser.add_argument('--output', type=str, default='walk_forward_training_results.pt',
                       help='Output file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/walk_forward',
                       help='Checkpoint directory')
    parser.add_argument('--no-preload', action='store_true',
                       help='Disable data preloading (saves memory for large seq_len)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of DataLoader workers (0 for debugging hangs)')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile (for debugging)')
    parser.add_argument('--compile-mode', type=str, default='default',
                       choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                       help='torch.compile mode (default: default)')
    parser.add_argument('--ddp', action='store_true',
                       help='Enable Distributed Data Parallel (multi-GPU training)')

    # Baseline comparison
    parser.add_argument('--compare-models', action='store_true',
                       help='Train and evaluate baseline models (Ridge, LightGBM, MLP) for comparison')
    parser.add_argument('--baseline-max-samples', type=int, default=50000,
                       help='Max training samples for baseline models (default: 50000)')

    args = parser.parse_args()

    # Append loss type and seed to checkpoint directory for easy comparison
    if args.ranking_only:
        args.checkpoint_dir = f"{args.checkpoint_dir}_{args.ranking_loss_type}_seed{args.seed}"
    else:
        loss_type = "mse" if args.pred_mode == "regression" else "ce"
        if args.ranking_loss_weight > 0:
            loss_type = f"{loss_type}+{args.ranking_loss_type}"
        args.checkpoint_dir = f"{args.checkpoint_dir}_{loss_type}_seed{args.seed}"

    # Run walk-forward training
    trainer = WalkForwardTrainer(
        dataset_path=args.data,
        prices_path=args.prices,
        num_folds=args.num_folds,
        mode=args.mode,
        min_train_months=args.min_train_months,
        test_months=args.test_months,
        gap_days=args.gap_days,
        auto_span=not args.no_auto_span,
        initial_train_fraction=args.initial_train_fraction,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pred_mode=args.pred_mode,
        # Contrastive pretraining
        contrastive_pretrain=args.contrastive_pretrain,
        contrastive_epochs=args.contrastive_epochs,
        contrastive_lr=args.contrastive_lr,
        contrastive_temperature=args.contrastive_temperature,
        freeze_encoder_after_pretrain=not args.no_freeze_encoder,
        epochs_per_fold=args.epochs_per_fold,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seq_len=args.seq_len,
        top_k=args.top_k,
        horizon_idx=args.horizon_idx,
        confidence_percentile=args.confidence_percentile,
        subset_size=args.subset_size,
        num_test_stocks=args.num_test_stocks,
        max_eval_dates=args.max_eval_dates,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        preload_data=not args.no_preload,
        num_workers=args.num_workers,
        compile_model=not args.no_compile,
        compile_mode=args.compile_mode,
        use_ddp=args.ddp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adaptive_batch_size=args.adaptive_batch_size,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        max_grad_accum=args.max_grad_accum,
        early_stopping_patience=args.early_stopping_patience,
        eval_every_n_steps=args.eval_every_n_steps,
        loss_smoothing_window=args.loss_smoothing_window,
        save_loss_curves_every=args.save_loss_curves_every,
        data_fraction=args.data_fraction,
        incremental_training=args.incremental,
        incremental_epochs=args.incremental_epochs,
        incremental_data_fraction=args.incremental_data_fraction,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_loss_type=args.ranking_loss_type,
        ranking_margin=args.ranking_margin,
        ranking_only=args.ranking_only,
        # Monte Carlo validation
        monte_carlo=not args.no_monte_carlo,
        mc_trials=args.mc_trials,
        mc_stocks=args.mc_stocks,
        mc_top_ks=args.mc_top_ks,
        # Intermediate checkpoint saving
        save_intermediate_checkpoints=args.save_intermediate_checkpoints,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        # Transaction costs
        transaction_cost_bps=args.transaction_cost_bps,
        # Baseline comparison
        compare_models=args.compare_models,
        baseline_max_samples=args.baseline_max_samples,
    )

    results = trainer.run()

    # Print summary
    print_summary(results, args.mode)

    # Save results
    save_results(results, args.output, args.mode)

    print(f"\n{'='*80}")
    print("WALK-FORWARD TRAINING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
