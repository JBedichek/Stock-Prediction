#!/usr/bin/env python3
"""
Evaluate Checkpoint Progression

Evaluates intermediate checkpoints saved throughout training to validate
that model performance improves as training progresses.

EFFICIENT IMPLEMENTATION:
- Loads features and returns data ONCE per fold
- For each checkpoint, only runs model inference ONCE
- Monte Carlo trials use cached predictions (CPU-only resampling)
- No redundant data loading or model computation

Usage:
    python -m inference.evaluate_checkpoint_progression \
        --checkpoint-dir checkpoints/walk_forward_seed42 \
        --data data/all_complete_dataset.h5 \
        --prices data/actual_prices.h5 \
        --fold 0

    # Evaluate all folds
    python -m inference.evaluate_checkpoint_progression \
        --checkpoint-dir checkpoints/walk_forward_seed42 \
        --data data/all_complete_dataset.h5 \
        --prices data/actual_prices.h5 \
        --all-folds
"""

import os
import sys
import glob
import argparse
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Set up logging to both console and file.

    Returns a logger that writes to both:
    - Console (INFO level)
    - File: {output_dir}/checkpoint_progression.log
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('checkpoint_progression')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # File handler
    log_file = os.path.join(output_dir, 'checkpoint_progression.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Global logger (initialized in main)
logger: Optional[logging.Logger] = None


def log(message: str):
    """Log a message to both console and file."""
    global logger
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def log_checkpoint_metrics(metrics: 'CheckpointMetrics', fold_idx: int):
    """Log detailed metrics for a single checkpoint evaluation."""
    log(f"\n  Epoch {metrics.epoch} Evaluation Results:")
    log(f"  {'-'*60}")
    log(f"  Loss Metrics:")
    log(f"    Train Loss:         {metrics.train_loss:.6f}")
    log(f"    Val Loss:           {metrics.val_loss:.6f}")
    log(f"  IC Metrics:")
    log(f"    Mean IC:            {metrics.mean_ic:+.6f}")
    log(f"    Std IC:             {metrics.std_ic:.6f}")
    log(f"    Information Ratio:  {metrics.ir:+.4f}")
    log(f"    % Positive IC:      {metrics.pct_positive_ic:.2f}%")
    log(f"    Mean Rank IC:       {metrics.mean_rank_ic:+.6f}")
    log(f"  Quantile Metrics:")
    log(f"    Top Decile Return:  {metrics.top_decile_return:+.4f}%")
    log(f"    Bottom Decile Ret:  {metrics.bottom_decile_return:+.4f}%")
    log(f"    Long-Short Spread:  {metrics.long_short_spread:+.4f}%")
    log(f"  Baseline Comparisons:")
    log(f"    Model Mean Return:  {metrics.model_mean_return:+.4f}%")
    log(f"    Random Mean Return: {metrics.random_mean_return:+.4f}%")
    log(f"    Excess vs Random:   {metrics.excess_vs_random:+.4f}%")
    log(f"  Simulation Metrics:")
    log(f"    Total Return:       {metrics.total_return_pct:+.4f}%")
    log(f"    Sharpe Ratio:       {metrics.sharpe_ratio:+.4f}")
    if metrics.mc_prob_beats_random > 0:
        log(f"  Monte Carlo Metrics:")
        log(f"    P(Model > Random):  {metrics.mc_prob_beats_random:.2f}%")
        log(f"    MC Excess Mean:     {metrics.mc_excess_mean:+.4f}%")
        log(f"    MC Excess Std:      {metrics.mc_excess_std:.4f}%")


def log_fold_summary(metrics_list: List['CheckpointMetrics'], fold_idx: int):
    """Log comprehensive summary for a fold's evaluation."""
    if len(metrics_list) < 2:
        log(f"\n  Not enough checkpoints for summary (need >= 2, got {len(metrics_list)})")
        return

    first = metrics_list[0]
    last = metrics_list[-1]
    best_ic_idx = np.argmax([m.mean_ic for m in metrics_list])
    best = metrics_list[best_ic_idx]

    # Compute statistical significance
    ics = np.array([m.mean_ic for m in metrics_list])
    losses = np.array([m.val_loss for m in metrics_list])

    # T-test for IC > 0 (using the final checkpoint's daily ICs if we had them)
    # Since we only have aggregate metrics, we'll compute a rough t-stat
    if last.std_ic > 1e-8:
        # Approximate t-stat: IC / (std_IC / sqrt(n_days))
        # We don't have n_days, so we'll estimate based on the IR
        approx_t_stat = last.ir * np.sqrt(50)  # Assume ~50 evaluation dates
    else:
        approx_t_stat = 0.0

    # Loss-IC correlation
    if len(losses) > 2 and np.std(losses) > 1e-8 and np.std(ics) > 1e-8:
        loss_ic_corr, loss_ic_pval = pearsonr(losses, ics)
    else:
        loss_ic_corr, loss_ic_pval = 0.0, 1.0

    # Compute improvement statistics
    ic_improved = last.mean_ic > first.mean_ic
    ic_change = last.mean_ic - first.mean_ic
    ic_pct_change = (ic_change / abs(first.mean_ic) * 100) if abs(first.mean_ic) > 1e-8 else 0

    log(f"\n{'='*70}")
    log(f"FOLD {fold_idx} EVALUATION SUMMARY")
    log(f"{'='*70}")
    log(f"\nCHECKPOINT PROGRESSION:")
    log(f"  Checkpoints evaluated: {len(metrics_list)}")
    log(f"  Epoch range: {first.epoch} -> {last.epoch}")
    log(f"  Best IC at epoch: {best.epoch}")

    log(f"\nMETRIC CHANGES (Epoch {first.epoch} -> {last.epoch}):")
    log(f"  {'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15} {'Best':<15}")
    log(f"  {'-'*85}")
    log(f"  {'Val Loss':<25} {first.val_loss:<15.6f} {last.val_loss:<15.6f} {last.val_loss - first.val_loss:<+15.6f} {best.val_loss:<15.6f}")
    log(f"  {'Mean IC':<25} {first.mean_ic:<+15.6f} {last.mean_ic:<+15.6f} {ic_change:<+15.6f} {best.mean_ic:<+15.6f}")
    log(f"  {'Information Ratio':<25} {first.ir:<+15.4f} {last.ir:<+15.4f} {last.ir - first.ir:<+15.4f} {best.ir:<+15.4f}")
    log(f"  {'% Positive IC':<25} {first.pct_positive_ic:<15.2f}% {last.pct_positive_ic:<15.2f}% {last.pct_positive_ic - first.pct_positive_ic:<+15.2f}% {best.pct_positive_ic:<15.2f}%")
    log(f"  {'Rank IC':<25} {first.mean_rank_ic:<+15.6f} {last.mean_rank_ic:<+15.6f} {last.mean_rank_ic - first.mean_rank_ic:<+15.6f} {best.mean_rank_ic:<+15.6f}")
    log(f"  {'Long-Short Spread':<25} {first.long_short_spread:<+15.4f}% {last.long_short_spread:<+15.4f}% {last.long_short_spread - first.long_short_spread:<+15.4f}% {best.long_short_spread:<+15.4f}%")
    log(f"  {'Excess vs Random':<25} {first.excess_vs_random:<+15.4f}% {last.excess_vs_random:<+15.4f}% {last.excess_vs_random - first.excess_vs_random:<+15.4f}% {best.excess_vs_random:<+15.4f}%")
    log(f"  {'Sharpe Ratio':<25} {first.sharpe_ratio:<+15.4f} {last.sharpe_ratio:<+15.4f} {last.sharpe_ratio - first.sharpe_ratio:<+15.4f} {best.sharpe_ratio:<+15.4f}")
    log(f"  {'MC P(>Random)':<25} {first.mc_prob_beats_random:<15.2f}% {last.mc_prob_beats_random:<15.2f}% {last.mc_prob_beats_random - first.mc_prob_beats_random:<+15.2f}% {best.mc_prob_beats_random:<15.2f}%")

    log(f"\nSTATISTICAL ANALYSIS:")
    log(f"  Approximate t-stat (IC > 0): {approx_t_stat:+.4f}")
    log(f"  Loss-IC Correlation: {loss_ic_corr:+.4f} (p={loss_ic_pval:.4f})")
    if loss_ic_corr < -0.3:
        log(f"    -> Good: Lower loss correlates with higher IC")
    elif abs(loss_ic_corr) < 0.3:
        log(f"    -> Neutral: Weak correlation between loss and IC")
    else:
        log(f"    -> Warning: Higher loss correlates with higher IC (unusual)")

    log(f"\nVALIDATION:")
    if ic_improved:
        log(f"  PASS: IC improved during training")
        log(f"    IC change: {ic_change:+.6f} ({ic_pct_change:+.2f}%)")
    else:
        log(f"  WARNING: IC did not improve during training")
        log(f"    IC change: {ic_change:+.6f} ({ic_pct_change:+.2f}%)")

    if last.mean_ic > 0.02:
        log(f"  Final IC ({last.mean_ic:+.4f}) > 0.02: Weak but useful signal")
    if last.mean_ic > 0.05:
        log(f"  Final IC ({last.mean_ic:+.4f}) > 0.05: Good predictive power")
    if last.ir > 0.5:
        log(f"  Final IR ({last.ir:+.2f}) > 0.5: Consistent signal")
    if last.pct_positive_ic > 55:
        log(f"  Final %Positive IC ({last.pct_positive_ic:.1f}%) > 55%: Model right more often")
    if last.mc_prob_beats_random > 70:
        log(f"  MC P(>Random) ({last.mc_prob_beats_random:.1f}%) > 70%: Strong edge")

    log(f"{'='*70}\n")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import SimpleTransformerPredictor, compute_expected_value


@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint evaluation."""
    epoch: int
    train_loss: float
    val_loss: float
    # IC metrics
    mean_ic: float
    std_ic: float
    ir: float
    pct_positive_ic: float
    mean_rank_ic: float
    # Quantile metrics
    top_decile_return: float
    bottom_decile_return: float
    long_short_spread: float
    # Baseline comparisons
    model_mean_return: float
    random_mean_return: float
    excess_vs_random: float
    # Simulation metrics
    total_return_pct: float
    sharpe_ratio: float
    # Monte Carlo (if enabled)
    mc_prob_beats_random: float = 0.0
    mc_excess_mean: float = 0.0
    mc_excess_std: float = 0.0


def find_intermediate_checkpoints(checkpoint_dir: str, fold_idx: int) -> List[str]:
    """Find all intermediate checkpoints for a fold, sorted by epoch."""
    intermediate_dir = os.path.join(checkpoint_dir, 'intermediate', f'fold_{fold_idx}')
    if not os.path.exists(intermediate_dir):
        return []

    pattern = os.path.join(intermediate_dir, 'epoch_*.pt')
    checkpoints = sorted(glob.glob(pattern))
    return checkpoints


def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[SimpleTransformerPredictor, dict]:
    """Load a checkpoint and return model and metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config
    model_config = checkpoint.get('model_config', {})

    # Create model
    model = SimpleTransformerPredictor(
        input_dim=model_config.get('input_dim', 203),
        hidden_dim=model_config.get('hidden_dim', 512),
        num_layers=model_config.get('num_layers', 6),
        num_heads=model_config.get('num_heads', 8),
        num_bins=100,
        num_output_days=4,
        dropout=model_config.get('dropout', 0.1),
        use_pos_encoding=True,
        mode=model_config.get('pred_mode', 'classification'),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


class FoldDataCache:
    """
    Caches all features and returns for a fold to avoid redundant data loading.

    This allows evaluating multiple checkpoints efficiently - we load data ONCE,
    then just re-run model inference for each checkpoint.
    """

    def __init__(
        self,
        dataset_path: str,
        prices_path: str,
        test_dates: List[str],
        horizon_days: int = 5,
        max_stocks: int = 500,
        input_dim: int = 203,
    ):
        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.test_dates = test_dates
        self.horizon_days = horizon_days
        self.max_stocks = max_stocks
        self.input_dim = input_dim

        # Cached data: date -> {'features': tensor, 'returns': array, 'tickers': list}
        self.date_data = {}

        # Load data once
        self._load_data()

    def _load_data(self):
        """Load all features and returns into memory."""
        log("  Loading data into cache (one-time operation)...")

        # Use non-overlapping dates
        eval_dates = self.test_dates[::self.horizon_days]

        # Load features
        features_cache = {}
        with h5py.File(self.dataset_path, 'r') as h5f:
            tickers = list(h5f.keys())[:self.max_stocks]
            for ticker in tqdm(tickers, desc="    Loading features", leave=False):
                try:
                    dates_bytes = h5f[ticker]['dates'][:]
                    dates = [d.decode('utf-8') for d in dates_bytes]
                    features = h5f[ticker]['features'][:].astype(np.float32)
                    if features.shape[1] == self.input_dim:
                        features_cache[ticker] = (dates, features)
                except Exception:
                    continue

        # Load prices
        prices_cache = {}
        with h5py.File(self.prices_path, 'r') as h5f:
            for ticker in features_cache.keys():
                if ticker not in h5f:
                    continue
                try:
                    dates_bytes = h5f[ticker]['dates'][:]
                    dates = [d.decode('utf-8') for d in dates_bytes]
                    prices = h5f[ticker]['prices'][:].astype(np.float32)
                    prices_cache[ticker] = (dates, prices)
                except Exception:
                    continue

        valid_tickers = list(set(features_cache.keys()) & set(prices_cache.keys()))
        log(f"    Loaded {len(valid_tickers)} valid tickers")

        # Build per-date data cache
        for date in tqdm(eval_dates, desc="    Building date cache", leave=False):
            features_list = []
            returns_list = []
            tickers_list = []

            for ticker in valid_tickers:
                feat_dates, feat_array = features_cache[ticker]
                price_dates, price_array = prices_cache[ticker]

                try:
                    feat_idx = feat_dates.index(date)
                    price_idx = price_dates.index(date)
                    future_idx = price_idx + self.horizon_days
                except ValueError:
                    continue

                if feat_idx >= feat_array.shape[0] or future_idx >= len(price_array):
                    continue
                if price_array[price_idx] <= 0:
                    continue

                actual_ret = (price_array[future_idx] / price_array[price_idx]) - 1.0

                # Filter extreme returns
                if abs(actual_ret) > 0.5:
                    continue

                features_list.append(feat_array[feat_idx])
                returns_list.append(actual_ret)
                tickers_list.append(ticker)

            if len(features_list) >= 20:
                self.date_data[date] = {
                    'features': np.stack(features_list),  # Shape: (n_stocks, input_dim)
                    'returns': np.array(returns_list),
                    'tickers': tickers_list,
                }

        log(f"    Cached {len(self.date_data)} evaluation dates")

    def get_dates(self) -> List[str]:
        """Get list of cached dates."""
        return sorted(self.date_data.keys())

    def get_date_features(self, date: str) -> Optional[np.ndarray]:
        """Get features for a date. Returns None if date not cached."""
        if date not in self.date_data:
            return None
        return self.date_data[date]['features']

    def get_date_returns(self, date: str) -> Optional[np.ndarray]:
        """Get returns for a date."""
        if date not in self.date_data:
            return None
        return self.date_data[date]['returns']

    def get_date_tickers(self, date: str) -> Optional[List[str]]:
        """Get tickers for a date."""
        if date not in self.date_data:
            return None
        return self.date_data[date]['tickers']


def predict_batch(
    model: SimpleTransformerPredictor,
    features: np.ndarray,
    bin_edges: Optional[torch.Tensor],
    device: str = 'cuda',
    batch_size: int = 512,
) -> np.ndarray:
    """
    Run model inference on a batch of features.

    Args:
        model: The model
        features: Shape (n_stocks, input_dim)
        bin_edges: Bin edges for classification mode
        device: Device
        batch_size: Batch size for inference

    Returns:
        Predictions array of shape (n_stocks,)
    """
    model.eval()
    predictions = []

    n_stocks = features.shape[0]

    with torch.no_grad():
        for start in range(0, n_stocks, batch_size):
            end = min(start + batch_size, n_stocks)
            batch = torch.from_numpy(features[start:end]).float().to(device)

            # Add sequence dimension if needed
            if batch.dim() == 2:
                batch = batch.unsqueeze(1)  # (batch, 1, features)

            pred, _ = model(batch)

            # Convert to expected value
            if bin_edges is not None and pred.dim() == 3:
                # Classification mode: pred is (batch, num_bins, num_days)
                # Use first day (horizon_idx=0)
                pred_day = pred[:, :, 0]  # (batch, num_bins)
                probs = F.softmax(pred_day, dim=-1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                expected = (probs * bin_centers.to(device)).sum(dim=-1)
                predictions.extend(expected.cpu().numpy().tolist())
            else:
                # Regression mode
                if pred.dim() == 2:
                    predictions.extend(pred[:, 0].cpu().numpy().tolist())
                else:
                    predictions.extend(pred.cpu().numpy().tolist())

    return np.array(predictions)


def evaluate_checkpoint_with_cache(
    model: SimpleTransformerPredictor,
    checkpoint: dict,
    data_cache: FoldDataCache,
    device: str = 'cuda',
    top_k: int = 10,
    n_mc_trials: int = 50,
    mc_stocks: int = 100,
) -> CheckpointMetrics:
    """
    Evaluate a single checkpoint using cached data.

    Model inference runs ONCE on all cached data.
    Monte Carlo trials just resample from cached predictions (CPU only).
    """

    # Get bin edges
    bin_edges = checkpoint.get('bin_edges', None)
    if bin_edges is not None:
        bin_edges = bin_edges.to(device)

    dates = data_cache.get_dates()

    # =========================================================================
    # PHASE 1: Score ALL stocks on ALL dates - SINGLE FORWARD PASS per date
    # =========================================================================
    # Structure: date_predictions[date] = {'predictions': array, 'returns': array, 'tickers': list}
    date_predictions = {}

    for date in dates:
        features = data_cache.get_date_features(date)
        returns = data_cache.get_date_returns(date)
        tickers = data_cache.get_date_tickers(date)

        if features is None:
            continue

        # Run model inference ONCE for this date
        predictions = predict_batch(model, features, bin_edges, device)

        date_predictions[date] = {
            'predictions': predictions,
            'returns': returns,
            'tickers': tickers,
        }

    # =========================================================================
    # PHASE 2: Compute metrics from cached predictions (CPU only)
    # =========================================================================
    daily_ics = []
    daily_rank_ics = []
    daily_model_returns = []
    daily_random_returns = []
    top_decile_returns = []
    bottom_decile_returns = []

    for date, data in date_predictions.items():
        predictions = data['predictions']
        actual_returns = data['returns']

        n_stocks = len(predictions)
        if n_stocks < 20:
            continue

        # Compute IC
        if np.std(predictions) > 1e-8 and np.std(actual_returns) > 1e-8:
            ic, _ = pearsonr(predictions, actual_returns)
            rank_ic, _ = spearmanr(predictions, actual_returns)
            if not np.isnan(ic):
                daily_ics.append(ic)
            if not np.isnan(rank_ic):
                daily_rank_ics.append(rank_ic)

        # Quantile analysis (just re-sort cached predictions)
        sorted_idx = np.argsort(predictions)[::-1]
        decile_size = max(1, n_stocks // 10)

        top_ret = actual_returns[sorted_idx[:decile_size]].mean()
        bottom_ret = actual_returns[sorted_idx[-decile_size:]].mean()
        top_decile_returns.append(top_ret)
        bottom_decile_returns.append(bottom_ret)

        # Model selection (top-k) - just index into sorted predictions
        k = min(top_k, n_stocks)
        model_ret = actual_returns[sorted_idx[:k]].mean()

        # Random selection
        random_idx = np.random.choice(n_stocks, k, replace=False)
        random_ret = actual_returns[random_idx].mean()

        daily_model_returns.append(model_ret)
        daily_random_returns.append(random_ret)

    # Aggregate metrics
    daily_ics = np.array(daily_ics) if daily_ics else np.array([0.0])
    daily_rank_ics = np.array(daily_rank_ics) if daily_rank_ics else np.array([0.0])
    daily_model_returns = np.array(daily_model_returns) if daily_model_returns else np.array([0.0])
    daily_random_returns = np.array(daily_random_returns) if daily_random_returns else np.array([0.0])

    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics) if len(daily_ics) > 1 else 0.0
    ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    pct_positive_ic = np.mean(daily_ics > 0) * 100
    mean_rank_ic = np.mean(daily_rank_ics)

    # Quantile metrics
    top_decile = np.mean(top_decile_returns) * 100 if top_decile_returns else 0.0
    bottom_decile = np.mean(bottom_decile_returns) * 100 if bottom_decile_returns else 0.0
    spread = top_decile - bottom_decile

    # Compound returns
    model_capital = 100000.0
    random_capital = 100000.0
    for m_ret, r_ret in zip(daily_model_returns, daily_random_returns):
        model_capital *= (1 + m_ret)
        random_capital *= (1 + r_ret)

    model_total = (model_capital / 100000.0 - 1) * 100
    random_total = (random_capital / 100000.0 - 1) * 100

    # Sharpe ratio
    if len(daily_model_returns) > 1 and np.std(daily_model_returns) > 1e-8:
        sharpe = np.mean(daily_model_returns) / np.std(daily_model_returns) * np.sqrt(252 / data_cache.horizon_days)
    else:
        sharpe = 0.0

    # =========================================================================
    # PHASE 3: Monte Carlo validation (CPU only - just mask and re-sort)
    # =========================================================================
    mc_prob_beats_random = 0.0
    mc_excess_mean = 0.0
    mc_excess_std = 0.0

    if n_mc_trials > 0 and len(date_predictions) >= 5:
        all_tickers = set()
        for data in date_predictions.values():
            all_tickers.update(data['tickers'])
        all_tickers = list(all_tickers)

        if len(all_tickers) >= mc_stocks:
            trial_excesses = []

            for _ in range(n_mc_trials):
                # Sample random subset of stocks for this trial
                trial_tickers = set(np.random.choice(all_tickers, size=mc_stocks, replace=False))

                trial_model_capital = 100000.0
                trial_random_capital = 100000.0

                for date, data in date_predictions.items():
                    # Mask to trial tickers (CPU only - no model computation)
                    mask = np.array([t in trial_tickers for t in data['tickers']])
                    if mask.sum() < top_k:
                        continue

                    masked_preds = data['predictions'][mask]
                    masked_rets = data['returns'][mask]

                    # Model: sort by prediction, take top-k
                    sorted_idx = np.argsort(masked_preds)[::-1]
                    k = min(top_k, len(masked_preds))
                    model_ret = masked_rets[sorted_idx[:k]].mean()

                    # Random: random k stocks
                    random_idx = np.random.choice(len(masked_rets), k, replace=False)
                    random_ret = masked_rets[random_idx].mean()

                    trial_model_capital *= (1 + model_ret)
                    trial_random_capital *= (1 + random_ret)

                trial_model_total = (trial_model_capital / 100000.0 - 1) * 100
                trial_random_total = (trial_random_capital / 100000.0 - 1) * 100
                trial_excesses.append(trial_model_total - trial_random_total)

            trial_excesses = np.array(trial_excesses)
            mc_prob_beats_random = np.mean(trial_excesses > 0) * 100
            mc_excess_mean = np.mean(trial_excesses)
            mc_excess_std = np.std(trial_excesses)

    return CheckpointMetrics(
        epoch=checkpoint.get('epoch', 0),
        train_loss=checkpoint.get('train_loss', 0.0),
        val_loss=checkpoint.get('val_loss', 0.0),
        mean_ic=mean_ic,
        std_ic=std_ic,
        ir=ir,
        pct_positive_ic=pct_positive_ic,
        mean_rank_ic=mean_rank_ic,
        top_decile_return=top_decile,
        bottom_decile_return=bottom_decile,
        long_short_spread=spread,
        model_mean_return=np.mean(daily_model_returns) * 100,
        random_mean_return=np.mean(daily_random_returns) * 100,
        excess_vs_random=model_total - random_total,
        total_return_pct=model_total,
        sharpe_ratio=sharpe,
        mc_prob_beats_random=mc_prob_beats_random,
        mc_excess_mean=mc_excess_mean,
        mc_excess_std=mc_excess_std,
    )


def plot_progression(
    metrics_list: List[CheckpointMetrics],
    output_dir: str,
    fold_idx: int,
):
    """Generate comprehensive progression charts."""

    if len(metrics_list) < 2:
        log("  Not enough checkpoints to plot progression")
        return

    epochs = [m.epoch for m in metrics_list]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Loss Curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, [m.train_loss for m in metrics_list], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, [m.val_loss for m in metrics_list], 'r-o', label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. IC Progression
    ax2 = fig.add_subplot(gs[0, 1])
    ics = [m.mean_ic for m in metrics_list]
    ax2.plot(epochs, ics, 'g-o', markersize=4)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(epochs, 0, ics, alpha=0.3, color='green' if ics[-1] > 0 else 'red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean IC')
    ax2.set_title('Information Coefficient')
    ax2.grid(True, alpha=0.3)

    # 3. IR Progression
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, [m.ir for m in metrics_list], 'm-o', markersize=4)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Information Ratio')
    ax3.set_title('IC Consistency (IR = IC/std)')
    ax3.grid(True, alpha=0.3)

    # 4. % Positive IC
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, [m.pct_positive_ic for m in metrics_list], 'c-o', markersize=4)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('% Days with Positive IC')
    ax4.set_title('Prediction Consistency')
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Long-Short Spread
    ax5 = fig.add_subplot(gs[1, 1])
    spreads = [m.long_short_spread for m in metrics_list]
    colors = ['green' if s > 0 else 'red' for s in spreads]
    ax5.bar(epochs, spreads, color=colors, alpha=0.7)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Spread (%)')
    ax5.set_title('Top Decile - Bottom Decile')
    ax5.grid(True, alpha=0.3)

    # 6. Excess vs Random
    ax6 = fig.add_subplot(gs[1, 2])
    excess = [m.excess_vs_random for m in metrics_list]
    colors = ['green' if e > 0 else 'red' for e in excess]
    ax6.bar(epochs, excess, color=colors, alpha=0.7)
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Excess Return (%)')
    ax6.set_title('Model vs Random Selection')
    ax6.grid(True, alpha=0.3)

    # 7. Total Return
    ax7 = fig.add_subplot(gs[2, 0])
    model_returns = [m.total_return_pct for m in metrics_list]
    ax7.plot(epochs, model_returns, 'b-o', label='Model', markersize=4)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Total Return (%)')
    ax7.set_title('Cumulative Trading Return')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Sharpe Ratio
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(epochs, [m.sharpe_ratio for m in metrics_list], 'orange', marker='o', markersize=4)
    ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Sharpe Ratio')
    ax8.set_title('Risk-Adjusted Return')
    ax8.grid(True, alpha=0.3)

    # 9. Rank IC
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(epochs, [m.mean_rank_ic for m in metrics_list], 'purple', marker='o', markersize=4)
    ax9.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Mean Rank IC')
    ax9.set_title('Spearman Correlation')
    ax9.grid(True, alpha=0.3)

    # 10. Monte Carlo P(beats random)
    ax10 = fig.add_subplot(gs[3, 0])
    mc_probs = [m.mc_prob_beats_random for m in metrics_list]
    ax10.plot(epochs, mc_probs, 'darkgreen', marker='o', markersize=4)
    ax10.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('P(Model > Random) %')
    ax10.set_title('Monte Carlo: Win Rate')
    ax10.set_ylim(0, 100)
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # 11. Monte Carlo Excess Return
    ax11 = fig.add_subplot(gs[3, 1])
    mc_excess = [m.mc_excess_mean for m in metrics_list]
    mc_std = [m.mc_excess_std for m in metrics_list]
    ax11.errorbar(epochs, mc_excess, yerr=mc_std, fmt='o-', color='darkblue',
                  capsize=3, markersize=4, alpha=0.8)
    ax11.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('MC Excess Return (%)')
    ax11.set_title('Monte Carlo: Excess vs Random')
    ax11.grid(True, alpha=0.3)

    # 12. Combined IC and Loss (dual axis)
    ax12 = fig.add_subplot(gs[3, 2])
    ax12_twin = ax12.twinx()
    line1 = ax12.plot(epochs, [m.val_loss for m in metrics_list], 'r-o', label='Val Loss', markersize=4)
    line2 = ax12_twin.plot(epochs, [m.mean_ic for m in metrics_list], 'g-s', label='IC', markersize=4)
    ax12.set_xlabel('Epoch')
    ax12.set_ylabel('Validation Loss', color='red')
    ax12_twin.set_ylabel('Mean IC', color='green')
    ax12.set_title('Loss vs IC Correlation')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax12.legend(lines, labels, loc='upper right')
    ax12.grid(True, alpha=0.3)

    # 13. Summary table
    ax13 = fig.add_subplot(gs[4, :])
    ax13.axis('off')

    # Create summary text
    first = metrics_list[0]
    last = metrics_list[-1]
    best_ic_idx = np.argmax([m.mean_ic for m in metrics_list])
    best = metrics_list[best_ic_idx]

    # Compute correlation between loss and IC
    losses = np.array([m.val_loss for m in metrics_list])
    ics = np.array([m.mean_ic for m in metrics_list])
    if len(losses) > 2 and np.std(losses) > 1e-8 and np.std(ics) > 1e-8:
        loss_ic_corr, _ = pearsonr(losses, ics)
    else:
        loss_ic_corr = 0.0

    summary_text = f"""
    TRAINING PROGRESSION SUMMARY (Fold {fold_idx})
    {'='*80}

    Metric                Epoch 1         Final (Epoch {last.epoch})      Best (Epoch {best.epoch})       Change
    ────────────────────────────────────────────────────────────────────────────────────────────
    Val Loss              {first.val_loss:.4f}          {last.val_loss:.4f}              {best.val_loss:.4f}              {last.val_loss - first.val_loss:+.4f}
    Mean IC               {first.mean_ic:+.4f}          {last.mean_ic:+.4f}              {best.mean_ic:+.4f}              {last.mean_ic - first.mean_ic:+.4f}
    IR                    {first.ir:+.2f}            {last.ir:+.2f}                {best.ir:+.2f}                {last.ir - first.ir:+.2f}
    % Positive IC         {first.pct_positive_ic:.1f}%           {last.pct_positive_ic:.1f}%               {best.pct_positive_ic:.1f}%               {last.pct_positive_ic - first.pct_positive_ic:+.1f}%
    Long-Short Spread     {first.long_short_spread:+.3f}%         {last.long_short_spread:+.3f}%             {best.long_short_spread:+.3f}%             {last.long_short_spread - first.long_short_spread:+.3f}%
    Excess vs Random      {first.excess_vs_random:+.2f}%          {last.excess_vs_random:+.2f}%              {best.excess_vs_random:+.2f}%              {last.excess_vs_random - first.excess_vs_random:+.2f}%
    Sharpe Ratio          {first.sharpe_ratio:+.2f}            {last.sharpe_ratio:+.2f}                {best.sharpe_ratio:+.2f}                {last.sharpe_ratio - first.sharpe_ratio:+.2f}
    MC P(>Random)         {first.mc_prob_beats_random:.1f}%           {last.mc_prob_beats_random:.1f}%               {best.mc_prob_beats_random:.1f}%               {last.mc_prob_beats_random - first.mc_prob_beats_random:+.1f}%

    Loss-IC Correlation: {loss_ic_corr:+.3f} ({'Good: loss tracks IC inversely' if loss_ic_corr < -0.3 else 'OK' if abs(loss_ic_corr) < 0.3 else 'Warning: loss not tracking IC'})

    VALIDATION: {'PASS - IC improved during training' if last.mean_ic > first.mean_ic else 'WARNING - IC did not improve'}
    """

    ax13.text(0.02, 0.98, summary_text, transform=ax13.transAxes,
              fontfamily='monospace', fontsize=9, verticalalignment='top')

    plt.suptitle(f'Checkpoint Progression Analysis - Fold {fold_idx}', fontsize=14, fontweight='bold')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'checkpoint_progression_fold_{fold_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    log(f"  Saved progression chart: {output_path}")


def plot_all_folds_comparison(
    all_fold_metrics: Dict[int, List[CheckpointMetrics]],
    output_dir: str,
):
    """Generate comparison chart across all folds."""

    if len(all_fold_metrics) < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_fold_metrics)))

    for fold_idx, metrics_list in all_fold_metrics.items():
        if len(metrics_list) < 2:
            continue

        epochs = [m.epoch for m in metrics_list]
        color = colors[fold_idx % len(colors)]

        # IC progression
        axes[0, 0].plot(epochs, [m.mean_ic for m in metrics_list],
                       '-o', color=color, label=f'Fold {fold_idx}', markersize=3, alpha=0.8)

        # IR progression
        axes[0, 1].plot(epochs, [m.ir for m in metrics_list],
                       '-o', color=color, label=f'Fold {fold_idx}', markersize=3, alpha=0.8)

        # Loss progression
        axes[0, 2].plot(epochs, [m.val_loss for m in metrics_list],
                       '-o', color=color, label=f'Fold {fold_idx}', markersize=3, alpha=0.8)

        # Excess return progression
        axes[1, 0].plot(epochs, [m.excess_vs_random for m in metrics_list],
                       '-o', color=color, label=f'Fold {fold_idx}', markersize=3, alpha=0.8)

        # MC win rate
        axes[1, 1].plot(epochs, [m.mc_prob_beats_random for m in metrics_list],
                       '-o', color=color, label=f'Fold {fold_idx}', markersize=3, alpha=0.8)

        # MC excess
        axes[1, 2].plot(epochs, [m.mc_excess_mean for m in metrics_list],
                       '-o', color=color, label=f'Fold {fold_idx}', markersize=3, alpha=0.8)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean IC')
    axes[0, 0].set_title('Information Coefficient by Fold')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IR')
    axes[0, 1].set_title('Information Ratio by Fold')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Validation Loss')
    axes[0, 2].set_title('Validation Loss by Fold')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Excess Return (%)')
    axes[1, 0].set_title('Excess vs Random by Fold')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('P(Model > Random) %')
    axes[1, 1].set_title('Monte Carlo Win Rate by Fold')
    axes[1, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('MC Excess Return (%)')
    axes[1, 2].set_title('Monte Carlo Excess by Fold')
    axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Training Progression Across All Folds', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'checkpoint_progression_all_folds.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    log(f"Saved all-folds comparison: {output_path}")


def evaluate_fold_progression(
    checkpoint_dir: str,
    fold_idx: int,
    dataset_path: str,
    prices_path: str,
    device: str = 'cuda',
    max_stocks: int = 500,
    top_k: int = 10,
    n_mc_trials: int = 50,
    mc_stocks: int = 100,
    verbose: bool = True,
) -> List[CheckpointMetrics]:
    """
    Evaluate all checkpoints for a single fold.

    EFFICIENT: Loads data ONCE, then evaluates each checkpoint.
    """

    # Find checkpoints
    checkpoints = find_intermediate_checkpoints(checkpoint_dir, fold_idx)

    if len(checkpoints) == 0:
        log(f"  No intermediate checkpoints found for fold {fold_idx}")
        log(f"  Looking in: {os.path.join(checkpoint_dir, 'intermediate', f'fold_{fold_idx}')}")
        return []

    log(f"  Found {len(checkpoints)} intermediate checkpoints")

    # Load config from first checkpoint
    first_ckpt = torch.load(checkpoints[0], map_location=device, weights_only=False)
    model_config = first_ckpt.get('model_config', {})
    training_config = first_ckpt.get('training_config', {})

    input_dim = model_config.get('input_dim', 203)
    horizon_days = training_config.get('horizon_days', 5)

    # Try to get test dates from final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_idx}_best.pt')
    if os.path.exists(final_checkpoint_path):
        final_ckpt = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
        test_dates = final_ckpt.get('test_dates', None)
    else:
        test_dates = None

    # If no test dates, compute them from dataset
    if test_dates is None:
        log("  Computing test dates from dataset...")
        with h5py.File(dataset_path, 'r') as h5f:
            all_dates = set()
            for ticker in list(h5f.keys())[:100]:
                if 'dates' in h5f[ticker]:
                    dates = [d.decode('utf-8') for d in h5f[ticker]['dates'][:]]
                    all_dates.update(dates)
            all_dates = sorted(all_dates)
            split_idx = int(len(all_dates) * 0.8)
            test_dates = all_dates[split_idx:]

    # =========================================================================
    # LOAD DATA ONCE - This is the key optimization
    # =========================================================================
    log(f"\n  Creating data cache (one-time operation for fold {fold_idx})...")
    data_cache = FoldDataCache(
        dataset_path=dataset_path,
        prices_path=prices_path,
        test_dates=test_dates,
        horizon_days=horizon_days,
        max_stocks=max_stocks,
        input_dim=input_dim,
    )

    # =========================================================================
    # EVALUATE EACH CHECKPOINT - Only model inference, no data reloading
    # =========================================================================
    log(f"\n  Evaluating {len(checkpoints)} checkpoints...")
    metrics_list = []

    for ckpt_path in tqdm(checkpoints, desc="  Checkpoints"):
        model, checkpoint = load_checkpoint(ckpt_path, device)

        # Evaluate - uses cached data, only runs model inference once
        metrics = evaluate_checkpoint_with_cache(
            model=model,
            checkpoint=checkpoint,
            data_cache=data_cache,
            device=device,
            top_k=top_k,
            n_mc_trials=n_mc_trials,
            mc_stocks=mc_stocks,
        )

        metrics_list.append(metrics)

        # Log detailed metrics for each checkpoint if verbose
        if verbose:
            log_checkpoint_metrics(metrics, fold_idx)

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Log fold summary
    if len(metrics_list) > 0:
        log_fold_summary(metrics_list, fold_idx)

    return metrics_list


def main():
    global logger

    parser = argparse.ArgumentParser(description='Evaluate checkpoint progression')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset HDF5 file')
    parser.add_argument('--prices', type=str, default=None,
                       help='Path to prices HDF5 file')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold index to evaluate (default: 0)')
    parser.add_argument('--all-folds', action='store_true',
                       help='Evaluate all folds')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--max-stocks', type=int, default=500,
                       help='Maximum stocks to evaluate per day')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of stocks to select for trading simulation')
    parser.add_argument('--mc-trials', type=int, default=50,
                       help='Number of Monte Carlo trials per checkpoint')
    parser.add_argument('--mc-stocks', type=int, default=100,
                       help='Number of stocks per Monte Carlo trial')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: checkpoint_dir/progression)')
    parser.add_argument('--verbose', action='store_true',
                       help='Log detailed metrics for each checkpoint')

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output or os.path.join(args.checkpoint_dir, 'progression')
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging to both console and file
    logger = setup_logging(output_dir)
    log_file_path = os.path.join(output_dir, 'checkpoint_progression.log')

    log(f"\n{'='*70}")
    log("CHECKPOINT PROGRESSION EVALUATION")
    log(f"{'='*70}")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Checkpoint dir: {args.checkpoint_dir}")
    log(f"Dataset: {args.data}")
    log(f"Prices: {args.prices}")
    log(f"Output: {output_dir}")
    log(f"Log file: {log_file_path}")
    log(f"Device: {args.device}")
    log(f"Max stocks: {args.max_stocks}")
    log(f"Top-k: {args.top_k}")
    log(f"Monte Carlo: {args.mc_trials} trials x {args.mc_stocks} stocks")
    log(f"Verbose: {args.verbose}")

    all_fold_metrics = {}

    if args.all_folds:
        # Find all folds
        intermediate_dir = os.path.join(args.checkpoint_dir, 'intermediate')
        if os.path.exists(intermediate_dir):
            fold_dirs = sorted(glob.glob(os.path.join(intermediate_dir, 'fold_*')))
            fold_indices = [int(os.path.basename(d).split('_')[1]) for d in fold_dirs]
        else:
            fold_indices = []

        if len(fold_indices) == 0:
            log("\nNo intermediate checkpoints found!")
            log(f"Make sure to train with --save-intermediate-checkpoints flag")
            return

        log(f"Found {len(fold_indices)} folds with intermediate checkpoints")

        for fold_idx in fold_indices:
            log(f"\n{'='*70}")
            log(f"FOLD {fold_idx}")
            log(f"{'='*70}")

            metrics_list = evaluate_fold_progression(
                checkpoint_dir=args.checkpoint_dir,
                fold_idx=fold_idx,
                dataset_path=args.data,
                prices_path=args.prices,
                device=args.device,
                max_stocks=args.max_stocks,
                top_k=args.top_k,
                n_mc_trials=args.mc_trials,
                mc_stocks=args.mc_stocks,
                verbose=args.verbose,
            )

            if len(metrics_list) > 0:
                all_fold_metrics[fold_idx] = metrics_list
                plot_progression(metrics_list, output_dir, fold_idx)

        # Plot all folds comparison
        if len(all_fold_metrics) > 1:
            plot_all_folds_comparison(all_fold_metrics, output_dir)

    else:
        # Single fold
        log(f"\n{'='*70}")
        log(f"FOLD {args.fold}")
        log(f"{'='*70}")

        metrics_list = evaluate_fold_progression(
            checkpoint_dir=args.checkpoint_dir,
            fold_idx=args.fold,
            dataset_path=args.data,
            prices_path=args.prices,
            device=args.device,
            max_stocks=args.max_stocks,
            top_k=args.top_k,
            n_mc_trials=args.mc_trials,
            mc_stocks=args.mc_stocks,
            verbose=args.verbose,
        )

        if len(metrics_list) > 0:
            all_fold_metrics[args.fold] = metrics_list
            plot_progression(metrics_list, output_dir, args.fold)

    # Save metrics to JSON
    if all_fold_metrics:
        results = {}
        for fold_idx, metrics_list in all_fold_metrics.items():
            results[f'fold_{fold_idx}'] = [asdict(m) for m in metrics_list]

        results_path = os.path.join(output_dir, 'checkpoint_progression_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"\nSaved JSON results to: {results_path}")

    # Final summary
    log(f"\n{'='*70}")
    log("FINAL SUMMARY")
    log(f"{'='*70}")

    for fold_idx, metrics_list in all_fold_metrics.items():
        if len(metrics_list) < 2:
            continue

        first = metrics_list[0]
        last = metrics_list[-1]

        ic_improved = last.mean_ic > first.mean_ic
        status = "PASS" if ic_improved else "WARNING"

        log(f"\nFold {fold_idx}:")
        log(f"  Epochs evaluated: {len(metrics_list)}")
        log(f"  IC: {first.mean_ic:+.4f} -> {last.mean_ic:+.4f} ({last.mean_ic - first.mean_ic:+.4f})")
        log(f"  IR: {first.ir:+.2f} -> {last.ir:+.2f}")
        log(f"  MC P(>Random): {first.mc_prob_beats_random:.1f}% -> {last.mc_prob_beats_random:.1f}%")
        log(f"  Validation: {status}")

    log(f"\nCharts saved to: {output_dir}")
    log(f"Log file saved to: {log_file_path}")
    log(f"\n{'='*70}")
    log("EVALUATION COMPLETE")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
