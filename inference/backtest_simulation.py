#!/usr/bin/env python3
"""
Backtesting Simulation for Stock Prediction Model

Simulates trading on the last N stocks (alphabetically) over a test period,
using model predictions to select top-k stocks for investment.

Features:
- Modular design for easy extension
- Support for both pickle and HDF5 datasets
- Configurable test period, stock selection, and strategy
- Detailed performance reporting
- Dynamic daily subsampling: randomly select different stocks each trading day (more realistic)
- Feature preloading for fast backtesting
"""

import torch
import torch.nn.functional as F
import argparse
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sys
import os
import h5py

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import pic_load
from training.train_new_format import SimpleTransformerPredictor, convert_price_ratios_to_bins_vectorized

# ===== CPU Optimizations =====
# Set optimal number of threads for CPU inference
torch.set_num_threads(torch.get_num_threads())  # Use all available CPU threads
torch.set_num_interop_threads(4)  # Limit inter-op parallelism to reduce overhead

# Enable oneDNN (MKL-DNN) optimizations for CPU
if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# Set matmul precision for faster computation (medium = TF32 on Ampere GPUs)
torch.set_float32_matmul_precision('medium')

# Disable gradient computation globally (inference only)
torch.set_grad_enabled(False)


class DatasetLoader:
    """Load and manage test dataset for backtesting."""

    def __init__(self, dataset_path: str, num_test_stocks: int = 1000, subset_size: Optional[int] = None, prices_path: Optional[str] = None):
        """
        Args:
            dataset_path: Path to dataset (pickle or HDF5) with features
            num_test_stocks: Number of stocks from end of alphabet to use
            subset_size: If set, randomly sample this many stocks each trading day (dynamic subsampling)
            prices_path: Optional path to HDF5 file with actual prices (if features are normalized)
        """
        print(f"\n{'='*80}")
        print("LOADING DATASET")
        print(f"{'='*80}")

        # Detect format and load dataset
        print(f"📦 Loading dataset from: {dataset_path}")
        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.prices_file = None

        if dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5'):
            # HDF5 format
            print(f"  Format: HDF5")
            self.is_hdf5 = True
            self.h5_file = h5py.File(dataset_path, 'r')
            all_tickers = sorted(list(self.h5_file.keys()))
            print(f"  ✅ Loaded {len(all_tickers)} tickers")
        else:
            # Pickle format
            print(f"  Format: Pickle")
            self.is_hdf5 = False
            self.data = pic_load(dataset_path)
            all_tickers = sorted(list(self.data.keys()))
            print(f"  ✅ Loaded {len(all_tickers)} tickers")

        # Get last num_test_stocks tickers (alphabetically)
        if len(all_tickers) < num_test_stocks:
            print(f"  ⚠️  Dataset has only {len(all_tickers)} tickers, using all")
            self.test_tickers = all_tickers
        else:
            self.test_tickers = all_tickers[-num_test_stocks:]
            print(f"  📊 Selected last {num_test_stocks} tickers (alphabetically)")

        # Store full pool for daily subsampling (before any subsetting)
        self.full_pool = self.test_tickers.copy()

        # Store subset_size for daily dynamic subsampling (if enabled)
        self.subset_size = subset_size
        if subset_size is not None and subset_size < len(self.test_tickers):
            print(f"  🎲 Daily dynamic subsampling enabled: {subset_size} random stocks per day")
            print(f"  ℹ️  Total pool: {len(self.test_tickers)} stocks (will cache all)")
        else:
            print(f"  📈 Using all {len(self.test_tickers)} test stocks")

        print(f"  Range: {self.test_tickers[0]} to {self.test_tickers[-1]}")

        # Get available dates from first ticker
        sample_ticker = self.test_tickers[0]
        if self.is_hdf5:
            # HDF5: dates are stored as dataset
            dates_bytes = self.h5_file[sample_ticker]['dates'][:]
            self.all_dates = sorted([d.decode('utf-8') for d in dates_bytes])
        else:
            # Pickle: dates are dict keys
            self.all_dates = sorted(list(self.data[sample_ticker].keys()))

        print(f"  📅 Date range: {self.all_dates[0]} to {self.all_dates[-1]}")
        print(f"  Total dates: {len(self.all_dates)}")

        # Load separate prices file if provided (for actual dollar prices)
        if prices_path:
            print(f"\n📊 Loading actual prices from: {prices_path}")
            try:
                self.prices_file = h5py.File(prices_path, 'r')
                num_tickers_with_prices = len(list(self.prices_file.keys()))
                print(f"  ✅ Prices loaded (will use for return calculation)")
                print(f"  📊 Prices file has {num_tickers_with_prices} tickers")

                # Debug: Check a sample price
                sample_ticker = list(self.prices_file.keys())[0]
                sample_prices = self.prices_file[sample_ticker]['prices'][:5]
                print(f"  🔍 Debug - Sample ticker {sample_ticker} first 5 prices: {sample_prices}")

                print(f"  ℹ️  Using normalized features for inference, actual prices for returns")
            except Exception as e:
                print(f"  ❌ Failed to load prices file: {e}")
                self.prices_file = None
        else:
            print(f"  ℹ️  Using features[0] as price (assumes unnormalized)")

        # Feature cache for fast access
        self.feature_cache = {}  # {(ticker, date): (features, price)}
        self.cache_enabled = False

    def get_daily_subset(self) -> List[str]:
        """
        Get a random subset of tickers for a single trading day.

        If subset_size is set, returns a random sample from the full pool (different each call).
        Otherwise, returns current test_tickers (which may have been subset by external code).

        Returns:
            List of tickers to use for this trading day
        """
        if self.subset_size is not None and self.subset_size < len(self.full_pool):
            # Random sample WITHOUT seed - different every time
            # Sample from full_pool, not test_tickers (which may be overridden by statistical_comparison)
            return random.sample(self.full_pool, self.subset_size)
        else:
            # Use current test_tickers (may be full pool or subset by external code)
            return self.test_tickers

    def preload_features(self, date_range: List[str]):
        """
        Preload all features for test tickers in the date range (eliminates I/O bottleneck).

        When daily subsampling is enabled, caches the entire pool (not just test_tickers).

        Args:
            date_range: List of dates to preload
        """
        print(f"\n{'='*80}")
        print("PRELOADING FEATURES (eliminates I/O bottleneck)")
        print(f"{'='*80}")

        # If daily subsampling is enabled, cache the full pool
        # Otherwise cache test_tickers (which may be subset by external code)
        tickers_to_cache = self.full_pool if (self.subset_size is not None and self.subset_size < len(self.full_pool)) else self.test_tickers

        print(f"Loading {len(tickers_to_cache)} tickers × {len(date_range)} dates...")
        if tickers_to_cache == self.full_pool and self.subset_size is not None:
            print(f"  (Caching full pool for daily subsampling)")

        self.feature_cache = {}

        if self.is_hdf5:
            # HDF5: Load all features at once for each ticker
            debug_counter = 0
            for ticker in tqdm(tickers_to_cache, desc="  Loading"):
                if ticker not in self.h5_file:
                    continue

                # Load entire feature array once
                features_2d = self.h5_file[ticker]['features'][:]  # (num_dates, num_features)

                # Cache only the dates we need
                for date in date_range:
                    try:
                        date_idx = self.all_dates.index(date)
                        if date_idx < features_2d.shape[0]:
                            features = torch.from_numpy(features_2d[date_idx, :]).float()
                            # Get actual price from prices file or fall back to features[0]
                            debug = False  # Debug first 5
                            if debug:
                                print(f"\n  🔍 Debug preload #{debug_counter}: {ticker} on {date}")
                                print(f"      features[0] = {features[0].item():.4f}")
                            current_price = self._get_actual_price(ticker, date, debug=debug)
                            if current_price is None:
                                current_price = features[0].item()
                                if debug:
                                    print(f"      Using features[0] as fallback: ${current_price:.4f}")
                            else:
                                if debug:
                                    print(f"      Got actual price: ${current_price:.2f}")
                            self.feature_cache[(ticker, date)] = (features, current_price)
                            debug_counter += 1
                    except ValueError:
                        continue
        else:
            # Pickle: Already in memory, just create index
            for ticker in tqdm(tickers_to_cache, desc="  Indexing"):
                if ticker not in self.data:
                    continue

                for date in date_range:
                    if date in self.data[ticker]:
                        features = self.data[ticker][date]
                        # Get actual price from prices file or fall back to features[0]
                        current_price = self._get_actual_price(ticker, date)
                        if current_price is None:
                            current_price = features[0].item()
                        self.feature_cache[(ticker, date)] = (features, current_price)

        self.cache_enabled = True
        print(f"  ✅ Preloaded {len(self.feature_cache):,} ticker-date pairs")
        print(f"  💾 Cache size: ~{len(self.feature_cache) * 1400 * 4 / 1e6:.1f} MB (assuming 1400 features)")

    def __del__(self):
        """Close HDF5 files when object is destroyed."""
        if hasattr(self, 'is_hdf5') and self.is_hdf5:
            if hasattr(self, 'h5_file') and self.h5_file is not None:
                self.h5_file.close()
        if hasattr(self, 'prices_file') and self.prices_file is not None:
            self.prices_file.close()

    def get_trading_period(self, num_months: int = 2) -> List[str]:
        """
        Get the last num_months of trading days.

        Args:
            num_months: Number of months to backtest

        Returns:
            List of date strings in the trading period (only actual trading days)
        """
        # If we have a prices file, use its dates (actual trading days only)
        # Otherwise use all dates from features
        if self.prices_file is not None:
            # Get trading days from prices file (no weekends/holidays)
            sample_ticker = list(self.prices_file.keys())[0]
            prices_dates_bytes = self.prices_file[sample_ticker]['dates'][:]
            actual_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])

            print(f"\n📅 Using actual trading days from prices file")
            print(f"  Total trading days available: {len(actual_trading_days)}")
        else:
            # Fall back to all dates (may include weekends if dataset has them)
            actual_trading_days = self.all_dates
            print(f"\n⚠️  No prices file - using all dates (may include weekends)")

        # Estimate number of trading days (approx 21 per month)
        approx_days = num_months * 21

        if len(actual_trading_days) < approx_days:
            print(f"  ⚠️  Only {len(actual_trading_days)} days available, using all")
            trading_dates = actual_trading_days
        else:
            trading_dates = actual_trading_days[-approx_days:]

        print(f"\n📅 Trading Period:")
        print(f"  Start: {trading_dates[0]}")
        print(f"  End: {trading_dates[-1]}")
        print(f"  Trading days: {len(trading_dates)}")

        return trading_dates

    def _get_actual_price(self, ticker: str, date: str, debug: bool = False) -> Optional[float]:
        """
        Get actual price from prices file or features.

        STRICT MODE: When prices file is provided, requires exact date match.
        No weekend/holiday fallbacks - we should only be trading on actual trading days.

        Args:
            ticker: Stock ticker
            date: Date string (must be a trading day)
            debug: Print debug info

        Returns:
            Actual price or None

        Raises:
            ValueError: If prices file exists but date is not a trading day
        """
        if self.prices_file is not None:
            # Use separate prices file (for when features are normalized)
            if ticker not in self.prices_file:
                if debug:
                    print(f"    🔍 Debug _get_actual_price: {ticker} not in prices file")
                return None

            # IMPORTANT: Use dates from the PRICES file, not from the features dataset!
            # Require exact match - we should only be trading on actual trading days
            prices_dates_bytes = self.prices_file[ticker]['dates'][:]
            prices_dates = [d.decode('utf-8') for d in prices_dates_bytes]

            try:
                date_idx = prices_dates.index(date)
            except ValueError:
                # STRICT MODE: Crash if date not found
                # This means we're trying to trade on a weekend/holiday, which is a bug
                raise ValueError(
                    f"❌ ERROR: {date} is not a trading day for {ticker}!\n"
                    f"   This should never happen if get_trading_period() is working correctly.\n"
                    f"   Available dates: {prices_dates[0]} to {prices_dates[-1]}\n"
                    f"   Total trading days: {len(prices_dates)}"
                )

            prices_array = self.prices_file[ticker]['prices'][:]
            actual_price = float(prices_array[date_idx])
            if debug:
                print(f"    🔍 Debug _get_actual_price: {ticker} on {date} -> ${actual_price:.2f}")
            return actual_price
        else:
            # Fall back to features[0] (assumes unnormalized)
            # This path is used when no prices file is provided
            if debug:
                print(f"    🔍 Debug _get_actual_price: No prices file, returning None")
            return None  # Will be extracted from features

    def get_features_and_price(self, ticker: str, date: str) -> Optional[Tuple[torch.Tensor, float]]:
        """
        Get features and current price for a ticker on a date.

        Args:
            ticker: Stock ticker
            date: Date string

        Returns:
            (features, current_price) or None if not available
        """
        # Check cache first (fast path)
        if self.cache_enabled:
            return self.feature_cache.get((ticker, date))

        # Slow path: Load from disk
        if self.is_hdf5:
            # HDF5 format
            if ticker not in self.h5_file:
                return None

            # Get date index
            try:
                date_idx = self.all_dates.index(date)
            except ValueError:
                return None

            # Get features for this date
            features_2d = self.h5_file[ticker]['features'][:]  # (num_dates, num_features)
            if date_idx >= features_2d.shape[0]:
                return None

            features = torch.from_numpy(features_2d[date_idx, :]).float()

            # Get actual price (either from prices file or features[0])
            current_price = self._get_actual_price(ticker, date)
            if current_price is None:
                current_price = features[0].item()  # Fall back to features

            return features, current_price
        else:
            # Pickle format
            if ticker not in self.data:
                return None
            if date not in self.data[ticker]:
                return None

            features = self.data[ticker][date]

            # Get actual price (either from prices file or features[0])
            current_price = self._get_actual_price(ticker, date)
            if current_price is None:
                current_price = features[0].item()  # Fall back to features

            return features, current_price

    def get_future_price(self, ticker: str, date: str, horizon_days: int) -> Optional[float]:
        """
        Get price N trading days in the future.

        IMPORTANT: Uses actual trading days from prices file if available.
        horizon_days means N actual trading days, not calendar days.

        Args:
            ticker: Stock ticker
            date: Current date (must be a trading day)
            horizon_days: Number of trading days ahead

        Returns:
            Future price or None if not available
        """
        # If we have a prices file, use its dates (actual trading days)
        if self.prices_file is not None and ticker in self.prices_file:
            # Get trading days from prices file
            prices_dates_bytes = self.prices_file[ticker]['dates'][:]
            prices_dates = [d.decode('utf-8') for d in prices_dates_bytes]

            # Find current date index in trading days
            try:
                date_idx = prices_dates.index(date)
            except ValueError:
                # Date not found - this should crash in strict mode
                return self._get_actual_price(ticker, date)  # Will raise ValueError

            # Get future date (N trading days ahead)
            future_idx = date_idx + horizon_days
            if future_idx >= len(prices_dates):
                return None

            future_date = prices_dates[future_idx]

            # Get price for future date
            prices_array = self.prices_file[ticker]['prices'][:]
            future_price = float(prices_array[future_idx])
            return future_price
        else:
            # Fall back to features dataset dates
            try:
                date_idx = self.all_dates.index(date)
            except ValueError:
                return None

            # Get future date
            future_idx = date_idx + horizon_days
            if future_idx >= len(self.all_dates):
                return None

            future_date = self.all_dates[future_idx]

            # Check cache first (fast path)
            if self.cache_enabled:
                cached = self.feature_cache.get((ticker, future_date))
                if cached is not None:
                    return cached[1]  # Return price
                return None

            # Slow path: Load from disk (features)
            if self.is_hdf5:
                # HDF5 format
                if ticker not in self.h5_file:
                    return None

                # Get features for future date
                features_2d = self.h5_file[ticker]['features'][:]
                if future_idx >= features_2d.shape[0]:
                    return None

                future_price = features_2d[future_idx, 0]  # First feature is price
                return float(future_price)
            else:
                # Pickle format
                if ticker not in self.data:
                    return None

                if future_date not in self.data[ticker]:
                    return None

                future_price = self.data[ticker][future_date][0].item()
                return future_price


class ModelPredictor:
    """Run model inference and calculate expected returns."""

    def __init__(self, model_path: str, bin_edges_path: str, device: str = 'cuda', batch_size: int = 64, compile_model: bool = False):
        """
        Args:
            model_path: Path to model checkpoint
            bin_edges_path: Path to cached bin edges
            device: Device to run inference on
            batch_size: Batch size for inference (default: 64)
            compile_model: Use torch.compile for faster inference (default: False)
        """
        self.batch_size = batch_size
        self.compile_model = compile_model

        print(f"\n{'='*80}")
        print("LOADING MODEL")
        print(f"{'='*80}")

        self.device = device
        print(f"Inference batch size: {batch_size}")
        if compile_model:
            print(f"⚡ torch.compile enabled (max-autotune mode)")

        # Load checkpoint
        print(f"📦 Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Get model config
        config = checkpoint['config']

        # Infer input_dim and pred_mode from state_dict
        state_dict = checkpoint['model_state_dict']

        # Handle both compiled and non-compiled checkpoints
        # torch.compile adds '_orig_mod.' prefix to all keys
        if '_orig_mod.input_proj.0.weight' in state_dict:
            input_dim = state_dict['_orig_mod.input_proj.0.weight'].shape[1]
            # Find final pred_head layer for pred_mode
            pred_head_keys = [k for k in state_dict.keys() if 'pred_head' in k and 'weight' in k and '_orig_mod' in k]
            final_pred_key = sorted(pred_head_keys)[-1]
            final_output_size = state_dict[final_pred_key].shape[0]
        elif 'input_proj.0.weight' in state_dict:
            input_dim = state_dict['input_proj.0.weight'].shape[1]
            # Find final pred_head layer for pred_mode
            pred_head_keys = [k for k in state_dict.keys() if 'pred_head' in k and 'weight' in k]
            final_pred_key = sorted(pred_head_keys)[-1]
            final_output_size = state_dict[final_pred_key].shape[0]
        else:
            raise KeyError(f"Could not find input_proj layer in state_dict. Keys: {list(state_dict.keys())[:5]}")

        # Infer pred_mode from final output size
        # classification: 100 bins × 4 horizons = 400
        # regression: 4 horizons = 4
        if 'pred_mode' in config:
            pred_mode = config['pred_mode']
        elif final_output_size == 400:
            pred_mode = 'classification'
            print(f"  ℹ️  Inferred pred_mode='classification' from output size {final_output_size}")
        elif final_output_size == 4:
            pred_mode = 'regression'
            print(f"  ℹ️  Inferred pred_mode='regression' from output size {final_output_size}")
        else:
            raise ValueError(f"Cannot infer pred_mode from output size {final_output_size}")

        print(f"  Inferred input_dim: {input_dim}")

        # Create model
        print(f"🏗️  Building model...")
        self.model = SimpleTransformerPredictor(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            num_pred_days=4,  # Always predict [1, 5, 10, 20] days
            pred_mode=pred_mode  # Use inferred pred_mode
        )

        # Strip '_orig_mod.' prefix if present (from torch.compile)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print(f"  🔧 Detected compiled checkpoint, stripping '_orig_mod.' prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Load weights (strict=False allows new confidence head in new models)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"  ⚠️  Missing keys (randomly initialized): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys in checkpoint: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")

        self.model = self.model.to(device)
        self.model.eval()

        # Compile model for faster inference if requested
        if self.compile_model:
            print(f"  🔧 Compiling model with torch.compile (max-autotune mode)...")
            print(f"     This will autotune Triton kernels for best performance...")
            # max-autotune: Best performance with autotuned Triton kernels
            # First batch will be slower (compilation), then ~2-3x faster
            self.model = torch.compile(self.model, mode='max-autotune')
            print(f"  ✅ Model compiled successfully")

        # Print checkpoint info (handle both epoch-based and iteration-based checkpoints)
        if 'epoch' in checkpoint:
            epoch_info = f"epoch {checkpoint['epoch']}"
        elif 'iteration' in checkpoint:
            epoch_info = f"iteration {checkpoint['iteration']}"
        else:
            epoch_info = "unknown"

        if 'val_loss' in checkpoint:
            loss_info = f"val_loss: {checkpoint['val_loss']:.6f}"
        elif 'best_val_loss' in checkpoint:
            loss_info = f"best_val_loss: {checkpoint['best_val_loss']:.6f}"
        else:
            loss_info = "loss: unknown"

        print(f"  ✅ Model loaded ({epoch_info}, {loss_info})")

        # Load bin edges
        self.pred_mode = pred_mode  # Use inferred pred_mode
        self.bin_edges_path = bin_edges_path

        if self.pred_mode == 'classification':
            if os.path.exists(bin_edges_path):
                print(f"📊 Loading bin edges: {bin_edges_path}")
                self.bin_edges = torch.load(bin_edges_path).to(device)
                print(f"  ✅ Loaded {len(self.bin_edges)} bin edges")
            else:
                print(f"  ⚠️  Bin edges file not found: {bin_edges_path}")
                print(f"  Will compute bin edges when needed...")
                # Will compute later when we have the dataset path
                self.bin_edges = None
        else:
            self.bin_edges = None

    @torch.no_grad()
    def predict_expected_return(self, features: torch.Tensor, horizon_idx: int = 0) -> Tuple[float, float]:
        """
        Predict expected return for a single stock.

        Args:
            features: Feature tensor (feature_dim,)
            horizon_idx: Index of prediction horizon (0=1day, 1=5day, 2=10day, 3=20day)

        Returns:
            Tuple of (expected_return, confidence)
        """
        # Use batch prediction with batch_size=1
        expected_returns, confidences = self.predict_expected_return_batch([features], horizon_idx)
        return expected_returns[0], confidences[0]

    @torch.no_grad()
    def predict_expected_return_batch(self, features_list: List[torch.Tensor], horizon_idx: int = 0) -> Tuple[List[float], List[float]]:
        """
        Predict expected returns for a batch of stocks (faster).

        Args:
            features_list: List of feature tensors, each (feature_dim,)
            horizon_idx: Index of prediction horizon (0=1day, 1=5day, 2=10day, 3=20day)

        Returns:
            Tuple of (expected_returns, confidences)
        """
        if len(features_list) == 0:
            return [], []

        # Stack into batch and add sequence dimension
        features_batch = torch.stack(features_list).unsqueeze(1).to(self.device)  # (batch, 1, feature_dim)

        # Forward pass - model returns (predictions, confidence) tuple
        pred, confidence = self.model(features_batch)

        if self.pred_mode == 'classification':
            # pred shape: (batch, num_bins, 4)
            # Get distribution for the specified horizon
            logits = pred[:, :, horizon_idx]  # (batch, num_bins)
            probs = F.softmax(logits, dim=1)  # (batch, num_bins)

            # Calculate expected value using bin midpoints
            bin_midpoints = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2  # (num_bins,)
            expected_ratios = torch.sum(probs * bin_midpoints.unsqueeze(0), dim=1)  # (batch,)

            # Convert from price ratio to multiplicative return
            expected_returns = (1.0 + expected_ratios).cpu().tolist()
        else:
            # Regression mode: direct price ratio prediction
            expected_ratios = pred[:, horizon_idx]  # (batch,)
            expected_returns = (1.0 + expected_ratios).cpu().tolist()

        # Extract confidence for this horizon
        horizon_confidence = confidence[:, horizon_idx].cpu().tolist()

        return expected_returns, horizon_confidence


class EnsemblePredictor:
    """Ensemble predictor that averages predictions from multiple models."""

    def __init__(self, model_paths: List[str], bin_edges_path: str, device: str = 'cuda', batch_size: int = 64, compile_model: bool = False):
        """
        Args:
            model_paths: List of paths to model checkpoints
            bin_edges_path: Path to cached bin edges
            device: Device to run inference on
            batch_size: Batch size for inference (default: 64)
            compile_model: Use torch.compile for faster inference (default: False)
        """
        if len(model_paths) == 0:
            raise ValueError("Must provide at least one model path for ensemble")

        self.batch_size = batch_size
        self.device = device
        self.bin_edges_path = bin_edges_path

        print(f"\n{'='*80}")
        print(f"LOADING ENSEMBLE ({len(model_paths)} models)")
        print(f"{'='*80}")

        # Load all models
        self.predictors = []
        for i, model_path in enumerate(model_paths, 1):
            print(f"\n--- Model {i}/{len(model_paths)} ---")
            predictor = ModelPredictor(
                model_path=model_path,
                bin_edges_path=bin_edges_path,
                device=device,
                batch_size=batch_size,
                compile_model=compile_model
            )
            self.predictors.append(predictor)

        # All models should have same pred_mode and bin_edges
        self.pred_mode = self.predictors[0].pred_mode
        self.bin_edges = self.predictors[0].bin_edges

        # Verify all models use same mode
        for i, predictor in enumerate(self.predictors[1:], 2):
            if predictor.pred_mode != self.pred_mode:
                raise ValueError(f"Model {i} has pred_mode={predictor.pred_mode}, expected {self.pred_mode}")

        print(f"\n{'='*80}")
        print(f"✅ ENSEMBLE LOADED: {len(self.predictors)} models in {self.pred_mode} mode")
        print(f"{'='*80}")

    @torch.no_grad()
    def predict_expected_return(self, features: torch.Tensor, horizon_idx: int = 0) -> Tuple[float, float]:
        """
        Predict expected return for a single stock using ensemble averaging.

        Args:
            features: Feature tensor (feature_dim,)
            horizon_idx: Index of prediction horizon (0=1day, 1=5day, 2=10day, 3=20day)

        Returns:
            Tuple of (expected_return, confidence)
        """
        # Use batch prediction with batch_size=1
        expected_returns, confidences = self.predict_expected_return_batch([features], horizon_idx)
        return expected_returns[0], confidences[0]

    @torch.no_grad()
    def predict_expected_return_batch(self, features_list: List[torch.Tensor], horizon_idx: int = 0) -> Tuple[List[float], List[float]]:
        """
        Predict expected returns for a batch of stocks using ensemble averaging.

        For classification mode: averages probability distributions before computing expected value
        For regression mode: averages direct predictions

        Args:
            features_list: List of feature tensors, each (feature_dim,)
            horizon_idx: Index of prediction horizon (0=1day, 1=5day, 2=10day, 3=20day)

        Returns:
            Tuple of (expected_returns, confidences)
        """
        if len(features_list) == 0:
            return [], []

        batch_size = len(features_list)

        if self.pred_mode == 'classification':
            # For classification: average probability distributions
            # Collect predictions from all models
            all_probs = []
            all_confidences = []

            for predictor in self.predictors:
                # Stack into batch and add sequence dimension
                features_batch = torch.stack(features_list).unsqueeze(1).to(self.device)  # (batch, 1, feature_dim)

                # Forward pass
                pred, confidence = predictor.model(features_batch)

                # Get logits for this horizon and convert to probabilities
                logits = pred[:, :, horizon_idx]  # (batch, num_bins)
                probs = F.softmax(logits, dim=1)  # (batch, num_bins)

                all_probs.append(probs)
                all_confidences.append(confidence[:, horizon_idx])  # (batch,)

            # Average probability distributions across models
            avg_probs = torch.stack(all_probs).mean(dim=0)  # (batch, num_bins)

            # Average confidences across models
            avg_confidence = torch.stack(all_confidences).mean(dim=0)  # (batch,)

            # Calculate expected value using bin midpoints
            bin_midpoints = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2  # (num_bins,)
            expected_ratios = torch.sum(avg_probs * bin_midpoints.unsqueeze(0), dim=1)  # (batch,)

            # Convert from price ratio to multiplicative return
            expected_returns = (1.0 + expected_ratios).cpu().tolist()
            confidences = avg_confidence.cpu().tolist()

        else:
            # For regression: average direct predictions
            all_predictions = []
            all_confidences = []

            for predictor in self.predictors:
                # Stack into batch and add sequence dimension
                features_batch = torch.stack(features_list).unsqueeze(1).to(self.device)  # (batch, 1, feature_dim)

                # Forward pass
                pred, confidence = predictor.model(features_batch)

                # Get predictions for this horizon
                predictions = pred[:, horizon_idx]  # (batch,)

                all_predictions.append(predictions)
                all_confidences.append(confidence[:, horizon_idx])  # (batch,)

            # Average predictions across models
            avg_predictions = torch.stack(all_predictions).mean(dim=0)  # (batch,)

            # Average confidences across models
            avg_confidence = torch.stack(all_confidences).mean(dim=0)  # (batch,)

            # Convert from price ratio to multiplicative return
            expected_returns = (1.0 + avg_predictions).cpu().tolist()
            confidences = avg_confidence.cpu().tolist()

        return expected_returns, confidences


class TradingSimulator:
    """Simulate trading strategy based on model predictions."""

    def __init__(self,
                 data_loader: DatasetLoader,
                 predictor: ModelPredictor,
                 top_k: int = 10,
                 horizon_days: int = 5,
                 horizon_idx: int = 1,
                 initial_capital: float = 100000.0,
                 confidence_percentile: float = 0.6,
                 verbose: bool = True,
                 dynamic_cluster_filter = None):
        """
        Args:
            data_loader: Dataset loader
            predictor: Model predictor
            top_k: Number of stocks to buy
            horizon_days: Holding period in trading days
            horizon_idx: Index for prediction (0=1day, 1=5day, 2=10day, 3=20day)
            initial_capital: Starting capital
            confidence_percentile: Percentile for confidence filtering (0.6 = keep top 40%)
            verbose: Print detailed trade information (default: True)
            dynamic_cluster_filter: Optional DynamicClusterFilter for daily filtering
        """
        self.data_loader = data_loader
        self.predictor = predictor
        self.top_k = top_k
        self.horizon_days = horizon_days
        self.horizon_idx = horizon_idx
        self.initial_capital = initial_capital
        self.confidence_percentile = confidence_percentile
        self.verbose = verbose
        self.dynamic_cluster_filter = dynamic_cluster_filter

        # Track cluster filtering stats
        self.cluster_filter_stats = {
            'total_days': 0,
            'total_candidates': 0,
            'total_filtered': 0
        }

        # Compute bin edges if needed
        if predictor.pred_mode == 'classification' and predictor.bin_edges is None:
            print(f"\n{'='*80}")
            print("COMPUTING BIN EDGES")
            print(f"{'='*80}")
            self._compute_bin_edges()

        # Performance tracking
        self.capital_history = []
        self.trade_history = []
        self.daily_returns = []

    def _compute_bin_edges(self):
        """Compute adaptive bin edges from dataset."""
        # Import the function based on dataset type
        if self.data_loader.is_hdf5:
            from training.hdf5_data_loader import compute_adaptive_bin_edges
        else:
            from training.new_data_loader import compute_adaptive_bin_edges

        # Compute bin edges
        bin_edges = compute_adaptive_bin_edges(
            dataset_path=self.data_loader.dataset_path,
            num_bins=100,
            pred_days=[1, 5, 10, 20],
            max_samples=50000
        )

        # Save for future use
        save_path = self.predictor.bin_edges_path
        torch.save(bin_edges, save_path)
        print(f"  💾 Saved bin edges to: {save_path}")

        # Set in predictor
        self.predictor.bin_edges = bin_edges.to(self.predictor.device)
        print(f"  ✅ Bin edges computed and loaded")

    def select_top_stocks(self, date: str) -> List[Tuple[str, float, float, float]]:
        """
        Select top-k stocks based on predicted returns (uses batched inference).

        Args:
            date: Current date

        Returns:
            List of (ticker, expected_return, confidence, current_price) sorted by confidence-weighted return
        """
        # Get subset of tickers for this trading day (random each day if subsampling enabled)
        daily_tickers = self.data_loader.get_daily_subset()

        # Collect all valid stocks and features for this date
        valid_data = []
        for ticker in daily_tickers:
            result = self.data_loader.get_features_and_price(ticker, date)
            if result is not None:
                features, current_price = result
                valid_data.append((ticker, features, current_price))

        if len(valid_data) == 0:
            return []

        # DYNAMIC CLUSTER FILTERING: Filter stocks based on today's cluster assignment
        if self.dynamic_cluster_filter is not None:
            # Create features dict for cluster encoding
            features_dict = {ticker: features for ticker, features, _ in valid_data}

            # Get allowed stocks for today (pass date for cache lookup)
            allowed_tickers = self.dynamic_cluster_filter.filter_stocks_for_date(features_dict, date=date)

            # Track stats
            self.cluster_filter_stats['total_days'] += 1
            self.cluster_filter_stats['total_candidates'] += len(valid_data)
            self.cluster_filter_stats['total_filtered'] += len(allowed_tickers)

            # Filter valid_data to only allowed stocks
            valid_data = [(ticker, features, price) for ticker, features, price in valid_data
                         if ticker in allowed_tickers]

            if len(valid_data) == 0:
                return []

        # Separate into lists
        tickers = [item[0] for item in valid_data]
        features_list = [item[1] for item in valid_data]
        prices = [item[2] for item in valid_data]

        # Batch inference
        predictions = []
        batch_size = self.predictor.batch_size

        for i in range(0, len(features_list), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            batch_features = features_list[i:i+batch_size]
            batch_prices = prices[i:i+batch_size]

            try:
                # Get predictions for this batch (returns expected_returns, confidences)
                expected_returns, confidences = self.predictor.predict_expected_return_batch(
                    batch_features, self.horizon_idx
                )

                # Combine results (now includes confidence)
                for ticker, expected_return, confidence, price in zip(batch_tickers, expected_returns, confidences, batch_prices):
                    predictions.append((ticker, expected_return, confidence, price))
            except Exception as e:
                # If batch fails, fall back to individual predictions
                for ticker, features, price in zip(batch_tickers, batch_features, batch_prices):
                    try:
                        expected_return, confidence = self.predictor.predict_expected_return(features, self.horizon_idx)
                        predictions.append((ticker, expected_return, confidence, price))
                    except:
                        continue

        # Filter by confidence percentile (keep top (1-percentile)% by confidence)
        if len(predictions) > 0:
            all_confidences = [pred[2] for pred in predictions]
            confidence_threshold = sorted(all_confidences)[int(len(all_confidences) * self.confidence_percentile)]
            predictions = [pred for pred in predictions if pred[2] >= confidence_threshold]

        # Weight expected returns by confidence
        weighted_predictions = [(ticker, expected_return * confidence, confidence, price)
                               for ticker, expected_return, confidence, price in predictions]

        # Sort by confidence-weighted return (descending)
        weighted_predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top-k (ticker, expected_return, confidence, price)
        return [(ticker, expected_return, confidence, price)
                for ticker, _, confidence, price in weighted_predictions[:self.top_k]]

    def simulate_trade(self, date: str, capital: float, debug: bool = False) -> Tuple[float, Dict]:
        """
        Simulate a single trade on a given date.

        Args:
            date: Trade date
            capital: Available capital
            debug: Print debug info

        Returns:
            (new_capital, trade_info)
        """
        # Select stocks
        top_stocks = self.select_top_stocks(date)


        if len(top_stocks) == 0:
            return capital, {'skipped': True, 'reason': 'no_predictions'}

        # Split capital equally among selected stocks
        capital_per_stock = capital / len(top_stocks)

        # Calculate returns
        total_return = 0.0
        stock_returns = []

        for ticker, expected_return, confidence, buy_price in top_stocks:
            # Get actual future price
            sell_price = self.data_loader.get_future_price(ticker, date, self.horizon_days)

            if sell_price is None:
                # Stock data not available - assume no change
                actual_return = 1.0
            else:
                actual_return = sell_price / buy_price

            # Calculate profit for this stock
            stock_profit = capital_per_stock * actual_return
            total_return += stock_profit

            stock_returns.append({
                'ticker': ticker,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'expected_return': expected_return,
                'confidence': confidence,
                'actual_return': actual_return,
                'profit_pct': (actual_return - 1.0) * 100
            })

        new_capital = total_return

        trade_info = {
            'date': date,
            'num_stocks': len(top_stocks),
            'capital_invested': capital,
            'capital_returned': new_capital,
            'return_pct': (new_capital / capital - 1.0) * 100,
            'stocks': stock_returns
        }

        return new_capital, trade_info

    def _print_trade_details(self, trade_num: int, trade_info: Dict, current_capital: float):
        """Print detailed information about a single trade."""
        if not self.verbose:
            return

        print(f"{'─'*80}")
        print(f"📅 Trade #{trade_num} | Date: {trade_info['date']}")
        print(f"{'─'*80}")

        # Selected stocks
        print(f"\n💼 Selected {trade_info['num_stocks']} stocks:")
        for stock in trade_info['stocks']:
            # Format expected vs actual returns
            expected_pct = (stock['expected_return'] - 1.0) * 100
            actual_pct = stock['profit_pct']

            # Color coding for profit/loss (using terminal colors)
            profit_indicator = "✅" if actual_pct > 0 else "❌"

            sell_price = stock['sell_price'] if stock['sell_price'] else 0
            print(f"  {profit_indicator} {stock['ticker']:>6}: ", end="")
            print(f"Expected: {expected_pct:>+6.2f}% | ", end="")
            print(f"Actual: {actual_pct:>+6.2f}% | ", end="")
            print(f"Buy: ${stock['buy_price']:>8.2f} → Sell: ${sell_price:>8.2f}")

        # Trade summary
        print(f"\n📊 Trade Summary:")
        print(f"  Capital invested:  ${trade_info['capital_invested']:>12,.2f}")
        print(f"  Capital returned:  ${trade_info['capital_returned']:>12,.2f}")
        print(f"  Trade return:      {trade_info['return_pct']:>+12.2f}%")

        # Running totals
        total_profit = current_capital - self.initial_capital
        total_return_pct = (current_capital / self.initial_capital - 1.0) * 100
        win_rate = sum(1 for r in self.daily_returns if r > 0) / len(self.daily_returns) * 100 if self.daily_returns else 0

        print(f"\n💰 Running Totals:")
        print(f"  Current capital:   ${current_capital:>12,.2f}")
        print(f"  Total profit:      ${total_profit:>+12,.2f}")
        print(f"  Total return:      {total_return_pct:>+12.2f}%")
        print(f"  Win rate:          {win_rate:>12.1f}%")
        print()

    def run_simulation(self, trading_dates: List[str]) -> Dict:
        """
        Run full trading simulation over the period.

        Args:
            trading_dates: List of dates to trade on

        Returns:
            Simulation results
        """
        print(f"\n{'='*80}")
        print("RUNNING SIMULATION")
        print(f"{'='*80}")
        print(f"Strategy: Buy top-{self.top_k} stocks, hold for {self.horizon_days} days")
        print(f"Initial capital: ${self.initial_capital:,.2f}")

        capital = self.initial_capital
        self.capital_history = [capital]
        self.trade_history = []
        self.daily_returns = []

        # We need to skip last horizon_days dates (can't complete trades)
        tradeable_dates = trading_dates[:-self.horizon_days]

        if self.verbose:
            print(f"\nSimulating {len(tradeable_dates)} trades...")
            print(f"{'='*80}\n")
            iterator = tradeable_dates
        else:
            print(f"\nSimulating {len(tradeable_dates)} trades...")
            iterator = tqdm(tradeable_dates, desc="Trading")

        trade_num = 0
        for date in iterator:
            # Enable debug for first 3 trades
            debug = (trade_num < 3)
            new_capital, trade_info = self.simulate_trade(date, capital, debug=debug)

            if trade_info.get('skipped'):
                print(f"Skipping {date}")
                continue

            trade_num += 1
            capital = new_capital
            self.capital_history.append(capital)
            self.trade_history.append(trade_info)
            self.daily_returns.append(trade_info['return_pct'])

            # Print detailed trade information
            self._print_trade_details(trade_num, trade_info, capital)

        # Calculate summary statistics
        final_capital = capital
        total_return = (final_capital / self.initial_capital - 1.0) * 100

        # Win rate
        winning_trades = sum(1 for r in self.daily_returns if r > 0)
        win_rate = (winning_trades / len(self.daily_returns) * 100) if self.daily_returns else 0

        # Average returns
        avg_return = np.mean(self.daily_returns) if self.daily_returns else 0
        std_return = np.std(self.daily_returns) if self.daily_returns else 0

        # Sharpe ratio (assuming 252 trading days/year, 0% risk-free rate)
        if std_return > 0:
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for capital_point in self.capital_history:
            if capital_point > peak:
                peak = capital_point
            drawdown = (peak - capital_point) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'num_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_return_pct': avg_return,
            'std_return_pct': std_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'capital_history': self.capital_history,
            'trade_history': self.trade_history,
            'daily_returns': self.daily_returns
        }

        # Print final summary
        print(f"{'='*80}")
        print("🎯 SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Final Results:")
        print(f"  Initial capital:  ${self.initial_capital:>12,.2f}")
        print(f"  Final capital:    ${final_capital:>12,.2f}")
        print(f"  Total profit:     ${final_capital - self.initial_capital:>+12,.2f}")
        print(f"  Total return:     {total_return:>+12.2f}%")
        print(f"  Number of trades: {len(self.trade_history):>12}")
        print(f"  Win rate:         {win_rate:>12.1f}%")
        print(f"  Avg return/trade: {avg_return:>+12.2f}%")
        print(f"  Sharpe ratio:     {sharpe_ratio:>12.2f}")
        print(f"  Max drawdown:     {max_drawdown:>12.2f}%")
        print()

        return results


class PerformanceReporter:
    """Report simulation results."""

    @staticmethod
    def print_summary(results: Dict):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print("SIMULATION RESULTS")
        print(f"{'='*80}")

        print(f"\n💰 Capital Performance:")
        print(f"  Initial capital:  ${results['initial_capital']:>12,.2f}")
        print(f"  Final capital:    ${results['final_capital']:>12,.2f}")
        print(f"  Total return:     {results['total_return_pct']:>12.2f}%")

        print(f"\n📊 Trading Statistics:")
        print(f"  Number of trades: {results['num_trades']:>12}")
        print(f"  Win rate:         {results['win_rate']:>12.2f}%")
        print(f"  Avg return:       {results['avg_return_pct']:>12.2f}%")
        print(f"  Std deviation:    {results['std_return_pct']:>12.2f}%")
        print(f"  Sharpe ratio:     {results['sharpe_ratio']:>12.2f}")
        print(f"  Max drawdown:     {results['max_drawdown_pct']:>12.2f}%")

    @staticmethod
    def print_recent_trades(results: Dict, n: int = 10):
        """Print recent trades."""
        print(f"\n{'='*80}")
        print(f"LAST {n} TRADES")
        print(f"{'='*80}")

        recent_trades = results['trade_history'][-n:]

        for trade in recent_trades:
            print(f"\n📅 {trade['date']}")
            print(f"  Return: {trade['return_pct']:>6.2f}% | "
                  f"Capital: ${trade['capital_returned']:>10,.2f}")
            print(f"  Stocks ({trade['num_stocks']}):")

            for stock in trade['stocks']:
                profit_str = f"{stock['profit_pct']:+.2f}%"
                sell_price_str = f"${stock['sell_price']:.2f}" if stock['sell_price'] else 'N/A'
                print(f"    {stock['ticker']:>6}: {profit_str:>8} | "
                      f"Buy: ${stock['buy_price']:.2f} -> Sell: {sell_price_str}")

    @staticmethod
    def save_detailed_results(results: Dict, output_path: str):
        """Save detailed results to file."""
        print(f"\n💾 Saving detailed results to: {output_path}")
        torch.save(results, output_path)
        print(f"  ✅ Results saved")


def analyze_multi_trial_results(all_trial_results: List[Dict], args):
    """Analyze and visualize results from multiple trials."""
    import matplotlib.pyplot as plt
    from scipy import stats

    # Extract metrics from all trials
    metrics = {
        'total_return_pct': [r['total_return_pct'] for r in all_trial_results],
        'sharpe_ratio': [r['sharpe_ratio'] for r in all_trial_results],
        'win_rate': [r['win_rate'] for r in all_trial_results],
        'max_drawdown_pct': [r['max_drawdown_pct'] for r in all_trial_results],
        'avg_return_pct': [r['avg_return_pct'] for r in all_trial_results],
        'num_trades': [r['num_trades'] for r in all_trial_results],
    }

    # Compute statistics
    print("\nPerformance Statistics Across Trials:")
    print("="*60)

    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        median_val = np.median(values)

        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Mean:   {mean_val:>10.2f}")
        print(f"  Median: {median_val:>10.2f}")
        print(f"  Std:    {std_val:>10.2f}")
        print(f"  Min:    {min_val:>10.2f}")
        print(f"  Max:    {max_val:>10.2f}")

        # Confidence interval
        ci_95 = stats.t.interval(0.95, len(values)-1,
                                 loc=mean_val,
                                 scale=stats.sem(values))
        print(f"  95% CI: [{ci_95[0]:>+10.2f}, {ci_95[1]:>+10.2f}]")

    # Visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    import os
    output_dir = os.path.dirname(args.output) or '.'

    # Plot 1: Distribution histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_metrics = [
        ('total_return_pct', 'Total Return (%)', 'higher'),
        ('sharpe_ratio', 'Sharpe Ratio', 'higher'),
        ('win_rate', 'Win Rate (%)', 'higher'),
        ('max_drawdown_pct', 'Max Drawdown (%)', 'lower'),
        ('avg_return_pct', 'Avg Return per Trade (%)', 'higher'),
        ('num_trades', 'Number of Trades', 'neutral'),
    ]

    for idx, (metric_key, label, better) in enumerate(plot_metrics):
        ax = axes[idx]
        values = metrics[metric_key]
        mean_val = np.mean(values)

        # Histogram
        n, bins, patches = ax.hist(values, bins=12, alpha=0.7, color='#3498db',
                                    edgecolor='black', linewidth=1.2)

        # Mean line
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=3,
                  label=f'Mean: {mean_val:.2f}')

        # Color bars based on value
        if better != 'neutral':
            threshold = mean_val
            for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
                if better == 'higher':
                    if bin_center > threshold:
                        patch.set_facecolor('#2ecc71')  # Green for above mean
                    else:
                        patch.set_facecolor('#e74c3c')  # Red for below mean
                else:  # lower is better
                    if bin_center < threshold:
                        patch.set_facecolor('#2ecc71')
                    else:
                        patch.set_facecolor('#e74c3c')

        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}\n({better} is better)' if better != 'neutral' else label,
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle(f'Model Performance Distribution Across {len(all_trial_results)} Trials',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'multi_trial_distributions.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {plot_path}")
    plt.close()

    # Plot 2: Box plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    key_metrics = [
        ('total_return_pct', 'Total Return (%)', 'higher'),
        ('sharpe_ratio', 'Sharpe Ratio', 'higher'),
        ('win_rate', 'Win Rate (%)', 'higher'),
        ('max_drawdown_pct', 'Max Drawdown (%)', 'lower'),
    ]

    for idx, (metric_key, label, better) in enumerate(key_metrics):
        ax = axes[idx]
        values = metrics[metric_key]
        mean_val = np.mean(values)

        # Box plot
        bp = ax.boxplot([values], labels=['Model'], widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor='#3498db', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        # Add mean marker
        ax.plot(1, mean_val, 'D', color='darkred', markersize=10,
               label=f'Mean: {mean_val:.2f}', zorder=3)

        # Add horizontal line at zero for return metrics
        if 'return' in metric_key or 'Return' in label:
            ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label}\n({better} is better)',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add statistics text
        stats_text = f"Median: {np.median(values):.2f}\nStd: {np.std(values):.2f}"
        ax.text(0.98, 0.97, stats_text,
               transform=ax.transAxes,
               ha='right', va='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'Model Performance Summary ({len(all_trial_results)} Trials)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'multi_trial_boxplots.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {plot_path}")
    plt.close()

    # Plot 3: Summary scorecard
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    summary_text = f"MODEL PERFORMANCE SUMMARY\n"
    summary_text += f"({len(all_trial_results)} trials)\n"
    summary_text += "="*50 + "\n\n"

    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_95 = stats.t.interval(0.95, len(values)-1,
                                 loc=mean_val,
                                 scale=stats.sem(values))

        summary_text += f"{metric_name.replace('_', ' ').title()}:\n"
        summary_text += f"  Mean:   {mean_val:>10.2f}\n"
        summary_text += f"  Std:    {std_val:>10.2f}\n"
        summary_text += f"  95% CI: [{ci_95[0]:>+7.2f}, {ci_95[1]:>+7.2f}]\n\n"

    summary_text += "="*50 + "\n\n"

    # Overall assessment
    avg_return = np.mean(metrics['total_return_pct'])
    avg_sharpe = np.mean(metrics['sharpe_ratio'])
    avg_winrate = np.mean(metrics['win_rate'])

    if avg_return > 10 and avg_sharpe > 1.5 and avg_winrate > 55:
        conclusion = "✓ STRONG PERFORMANCE: Consistently profitable"
        conclusion_color = 'green'
    elif avg_return > 5 and avg_sharpe > 1.0 and avg_winrate > 50:
        conclusion = "○ MODERATE PERFORMANCE: Some profitability"
        conclusion_color = 'orange'
    else:
        conclusion = "✗ WEAK PERFORMANCE: Inconsistent or unprofitable"
        conclusion_color = 'red'

    summary_text += conclusion

    ax.text(0.5, 0.5, summary_text,
           transform=ax.transAxes,
           ha='center', va='center',
           fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor=conclusion_color, linewidth=3, alpha=0.9))

    plot_path = os.path.join(output_dir, 'multi_trial_summary.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {plot_path}")
    plt.close()

    # Save aggregated results
    aggregated_results = {
        'num_trials': len(all_trial_results),
        'config': vars(args),
        'statistics': {
            metric_name: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_95': [float(ci_95[0]), float(ci_95[1])]
            }
            for metric_name, values in metrics.items()
            for ci_95 in [stats.t.interval(0.95, len(values)-1,
                                          loc=np.mean(values),
                                          scale=stats.sem(values))]
        },
        'all_trials': all_trial_results
    }

    output_path = args.output.replace('.pt', '_multi_trial.pt')
    torch.save(aggregated_results, output_path)
    print(f"\n💾 Aggregated results saved to: {output_path}")

    print(f"\n{'='*60}")
    print("✅ MULTI-TRIAL ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Backtest stock prediction model')

    # Data args
    parser.add_argument('--data', type=str, default="data/all_complete_dataset.h5",
                       help='Path to dataset (pickle or HDF5)')
    parser.add_argument('--prices', type=str, default="data/actual_prices.h5",
                       help='Path to HDF5 file with actual prices (optional, for normalized features)')
    parser.add_argument('--model', type=str, default="checkpoints/best_model_100m_1.18.pt",
                       help='Path to model checkpoint (or first model if using --ensemble-models)')
    parser.add_argument('--ensemble-models', type=str, nargs='+', default=None,
                       help='Paths to multiple model checkpoints for ensemble prediction (overrides --model)')
    parser.add_argument('--bin-edges', type=str, default='data/adaptive_bin_edges.pt',
                       help='Path to bin edges cache')

    # Test set args
    parser.add_argument('--num-test-stocks', type=int, default=2000,
                       help='Number of stocks from end of alphabet to test on')
    parser.add_argument('--subset-size', type=int, default=1024,
                       help='Randomly sample this many stocks EACH DAY (dynamic subsampling - different stocks each trading day)')

    # Trading strategy args
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of stocks to buy each period')
    parser.add_argument('--horizon-idx', type=int, default=0,
                       help='Prediction horizon index (0=1day, 1=5day, 2=10day, 3=20day)')
    parser.add_argument('--confidence-percentile', type=float, default=0.2,
                       help='Confidence percentile for filtering (default: 0.6 = keep top 40%%)')
    parser.add_argument('--test-months', type=int, default=4,
                       help='Number of months to backtest')

    # Cluster filtering args (optional)
    parser.add_argument('--cluster-dir', type=str, default="./cluster_results",
                       help='Directory with cluster results (enables cluster filtering)')
    parser.add_argument('--best-clusters-file', type=str, default="./cluster_results/best_clusters_1d.txt",
                       help='File with best cluster IDs (required if --cluster-dir is set)')
    parser.add_argument('--cluster-batch-size', type=int, default=32,
                       help='Batch size for cluster encoding (default: 32, reduce if OOM)')
    parser.add_argument('--embeddings-cache', type=str, default=None,
                       help='Path to pre-computed embeddings cache (HDF5) for fast filtering')

    # Other args
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial capital for simulation')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for inference (default: 64)')
    parser.add_argument('--no-preload', action='store_true',
                       help='Disable feature preloading (saves RAM but slower)')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable detailed per-trade output (show progress bar instead)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile with max-autotune (~2-3x inference speedup)')
    parser.add_argument('--output', type=str, default='backtest_results.pt',
                       help='Path to save detailed results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-trials', type=int, default=1,
                       help='Number of trials to run (for statistical analysis)')

    args = parser.parse_args()

    # Auto-calculate horizon_days from horizon_idx to keep them in sync
    horizon_map = {0: 1, 1: 5, 2: 10, 3: 20}
    args.horizon_days = horizon_map.get(args.horizon_idx, 1)

    print(f"Using prediction horizon {args.horizon_idx}: holding for {args.horizon_days} days")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    data_loader = DatasetLoader(
        dataset_path=args.data,
        num_test_stocks=args.num_test_stocks,
        subset_size=args.subset_size,
        prices_path=args.prices
    )

    # Initialize dynamic cluster filter if enabled
    dynamic_cluster_filter = None
    if args.cluster_dir is not None:
        if args.best_clusters_file is None:
            raise ValueError("--best-clusters-file must be provided when using --cluster-dir")

        print(f"\n{'='*80}")
        print("INITIALIZING DYNAMIC CLUSTER FILTERING")
        print(f"{'='*80}")
        print(f"\nℹ️  Dynamic filtering: stocks will be encoded and assigned to clusters EVERY DAY")
        print(f"   Only stocks in best-performing clusters will be considered for trading")

        # Import dynamic cluster filter
        from cluster.dynamic_cluster_filter import DynamicClusterFilter

        # Initialize filter
        dynamic_cluster_filter = DynamicClusterFilter(
            model_path=args.model if args.ensemble_models is None else args.ensemble_models[0],
            cluster_dir=args.cluster_dir,
            best_clusters_file=args.best_clusters_file,
            device=args.device,
            batch_size=args.cluster_batch_size,
            embeddings_cache_path=args.embeddings_cache
        )
        if args.embeddings_cache is None:
            print(f"  ℹ️  Cluster encoding batch size: {args.cluster_batch_size} (use --cluster-batch-size to adjust)")
            print(f"  💡 Tip: Pre-compute embeddings cache for faster filtering:")
            print(f"     python -m cluster.cache_embeddings --output-path data/embeddings_cache.h5")

    # Get trading period
    trading_dates = data_loader.get_trading_period(num_months=args.test_months)

    # Preload features (eliminates I/O bottleneck)
    if not args.no_preload:
        data_loader.preload_features(trading_dates)
    else:
        print(f"\n⚠️  Preloading disabled - expect slower performance")

    # Load model (single or ensemble)
    if args.ensemble_models is not None:
        # Ensemble mode: use multiple models
        print(f"\n🎯 Ensemble mode: using {len(args.ensemble_models)} models")
        predictor = EnsemblePredictor(
            model_paths=args.ensemble_models,
            bin_edges_path=args.bin_edges,
            device=args.device,
            batch_size=args.batch_size,
            compile_model=args.compile
        )
    else:
        # Single model mode
        predictor = ModelPredictor(
            model_path=args.model,
            bin_edges_path=args.bin_edges,
            device=args.device,
            batch_size=args.batch_size,
            compile_model=args.compile
        )

    if args.num_trials == 1:
        # Single trial mode (original behavior)
        # Run simulation
        simulator = TradingSimulator(
            data_loader=data_loader,
            predictor=predictor,
            top_k=args.top_k,
            horizon_days=args.horizon_days,
            horizon_idx=args.horizon_idx,
            initial_capital=args.initial_capital,
            confidence_percentile=args.confidence_percentile,
            verbose=not args.quiet,
            dynamic_cluster_filter=dynamic_cluster_filter
        )

        results = simulator.run_simulation(trading_dates)

        # Report results
        reporter = PerformanceReporter()
        reporter.print_summary(results)
        reporter.print_recent_trades(results, n=10)
        reporter.save_detailed_results(results, args.output)

        # Print cluster filtering stats if enabled
        if dynamic_cluster_filter is not None:
            stats = simulator.cluster_filter_stats
            print(f"\n{'='*80}")
            print("DYNAMIC CLUSTER FILTERING STATISTICS")
            print(f"{'='*80}")
            print(f"\n  Trading days:            {stats['total_days']:>6}")
            print(f"  Total candidates:        {stats['total_candidates']:>6}")
            print(f"  Passed filter:           {stats['total_filtered']:>6} ({stats['total_filtered']/stats['total_candidates']*100:.1f}%)")
            print(f"  Avg candidates per day:  {stats['total_candidates']/stats['total_days']:>6.1f}")
            print(f"  Avg filtered per day:    {stats['total_filtered']/stats['total_days']:>6.1f}")
            print(f"\n  Good clusters used:      {len(dynamic_cluster_filter.best_cluster_ids)}")

        print(f"\n{'='*80}")
        print("✅ BACKTEST COMPLETE")
        print(f"{'='*80}\n")

    else:
        # Multi-trial mode with statistical analysis
        print(f"\n{'='*80}")
        print(f"RUNNING {args.num_trials} TRIALS FOR STATISTICAL ANALYSIS")
        print(f"{'='*80}")

        all_trial_results = []

        for trial_idx in tqdm(range(args.num_trials), desc="Running trials"):
            # Set different seed for each trial
            trial_seed = args.seed + trial_idx
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)

            # Create new subset for this trial
            trial_data_loader = DatasetLoader(
                dataset_path=args.data,
                num_test_stocks=args.num_test_stocks,
                subset_size=args.subset_size,
                prices_path=args.prices
            )

            # Preload features if needed
            if not args.no_preload:
                trial_data_loader.preload_features(trading_dates)

            # Run simulation (quiet mode for multi-trial)
            simulator = TradingSimulator(
                data_loader=trial_data_loader,
                predictor=predictor,
                top_k=args.top_k,
                horizon_days=args.horizon_days,
                horizon_idx=args.horizon_idx,
                initial_capital=args.initial_capital,
                confidence_percentile=args.confidence_percentile,
                verbose=False,
                dynamic_cluster_filter=dynamic_cluster_filter
            )

            results = simulator.run_simulation(trading_dates)
            all_trial_results.append(results)

        # Analyze and visualize results
        print(f"\n{'='*80}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*80}")

        analyze_multi_trial_results(all_trial_results, args)


if __name__ == '__main__':
    main()
