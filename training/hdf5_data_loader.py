"""
Fast Data Loader for HDF5 Dataset Format

Much faster than pickle format - doesn't load entire dataset into memory.
Uses on-the-fly loading from HDF5 file.

Usage:
    # Same API as new_data_loader.py
    from training.hdf5_data_loader import StockDataModule

    dm = StockDataModule('dataset.h5', batch_size=32)
    train_loader = dm.train_dataloader()
"""

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import os
import hashlib


class HDF5StockDataset(Dataset):
    """
    PyTorch Dataset for stock prediction using HDF5 format.

    Loads data on-the-fly from HDF5 - much faster than pickle format.
    """

    def __init__(self,
                 dataset_path: str,
                 seq_len: int = 60,
                 pred_days: List[int] = [1, 5, 10, 20],
                 min_future_days: int = 20,
                 news_dim: int = 768,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 val_max_size: int = None,
                 test_max_size: int = None,
                 seed: int = 42):
        """
        Initialize dataset.

        Args:
            dataset_path: Path to HDF5 file
            seq_len: Sequence length for training
            pred_days: List of future days to predict
            min_future_days: Minimum days of future data required
            news_dim: Dimension of news embeddings (default 768)
            split: 'train', 'val', or 'test'
            train_ratio: Ratio for train split (default 0.7 = 70%)
            val_ratio: Ratio for val split (default 0.15 = 15%, remaining 15% is test)
            val_max_size: Maximum validation set size (None = use val_ratio)
            test_max_size: Maximum test set size (None = use remaining after train/val)
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.pred_days = sorted(pred_days)
        self.min_future_days = min_future_days
        self.news_dim = news_dim
        self.split = split

        print(f"\n{'='*80}")
        print(f"Loading HDF5 dataset: {dataset_path}")
        print(f"{'='*80}")

        # Open HDF5 file (keep handle open for fast access)
        self.h5f = h5py.File(dataset_path, 'r')

        # Get metadata
        self.num_tickers = self.h5f.attrs['num_tickers']

        # Find maximum feature dimension across all stocks
        print(f"\n  🔍 Analyzing feature dimensions...")
        max_features = 0
        min_features = float('inf')

        for ticker in tqdm(list(self.h5f.keys())[:100], desc="  Scanning"):  # Sample first 100
            features_shape = self.h5f[ticker]['features'].shape
            num_features = features_shape[1]
            max_features = max(max_features, num_features)
            min_features = min(min_features, num_features)

        self.total_features = max_features
        self.base_features = self.total_features - self.news_dim

        print(f"\n  📊 Dataset Info:")
        print(f"    Tickers: {self.num_tickers}")
        print(f"    Maximum features: {max_features}")
        print(f"    Minimum features: {min_features}")
        if max_features != min_features:
            print(f"    ⚠️  VARIABLE DIMENSIONS DETECTED - will pad to {max_features}")
        print(f"    Using: {self.total_features} (padded)")
        print(f"    Base features: {self.base_features}")
        print(f"    News embeddings: {self.news_dim}")

        # Build index of valid sequences
        print(f"\n  📦 Building sequence index...")
        self.sequences = self._build_sequence_index()

        # ============================================================================
        # TEMPORAL SPLIT (prevents future leakage)
        # ============================================================================
        # Check if temporal split is cached
        cache_path = self._get_cache_path(dataset_path, seq_len, min_future_days, train_ratio, val_ratio)

        if os.path.exists(cache_path):
            print(f"\n  💾 Loading cached temporal split from: {cache_path}")
            with h5py.File(cache_path, 'r') as cache_f:
                # Load sorted dates
                sorted_dates = [d.decode('utf-8') for d in cache_f['sorted_dates'][:]]

                # Load sequences_by_date
                sequences_by_date = {}
                for date in sorted_dates:
                    # Store as tuples of (ticker_bytes, end_idx)
                    date_group = cache_f['sequences_by_date'][date]
                    tickers_bytes = date_group['tickers'][:]
                    end_indices = date_group['end_indices'][:]
                    sequences_by_date[date] = [(t.decode('utf-8'), int(idx))
                                               for t, idx in zip(tickers_bytes, end_indices)]

            print(f"  ✅ Loaded {len(sorted_dates)} dates from cache")
        else:
            print(f"\n  📅 Grouping sequences by date for temporal split...")
            print(f"  ⚠️  This will take a few minutes (only computed once)...")
            sequences_by_date = {}
            for ticker, end_idx in tqdm(self.sequences, desc="  Grouping"):
                group = self.h5f[ticker]
                date = group['dates'][end_idx - 1].decode('utf-8')
                if date not in sequences_by_date:
                    sequences_by_date[date] = []
                sequences_by_date[date].append((ticker, end_idx))

            # Sort dates chronologically (CRITICAL for preventing leakage)
            sorted_dates = sorted(sequences_by_date.keys())
            print(f"  ✅ Found {len(sorted_dates)} unique dates")

            # Save to cache
            print(f"\n  💾 Saving temporal split to cache: {cache_path}")
            with h5py.File(cache_path, 'w') as cache_f:
                # Save sorted dates
                sorted_dates_bytes = np.array([d.encode('utf-8') for d in sorted_dates])
                cache_f.create_dataset('sorted_dates', data=sorted_dates_bytes, compression='gzip')

                # Save sequences_by_date
                seq_group = cache_f.create_group('sequences_by_date')
                for date, sequences in tqdm(sequences_by_date.items(), desc="  Saving"):
                    date_group = seq_group.create_group(date)
                    tickers = [ticker.encode('utf-8') for ticker, _ in sequences]
                    end_indices = [end_idx for _, end_idx in sequences]
                    date_group.create_dataset('tickers', data=np.array(tickers), compression='gzip')
                    date_group.create_dataset('end_indices', data=np.array(end_indices, dtype=np.int32), compression='gzip')

            print(f"  ✅ Cache saved successfully!")

        print(f"  📊 Date range: {sorted_dates[0]} to {sorted_dates[-1]}")

        # Split dates temporally (not sequences!)
        train_date_idx = int(len(sorted_dates) * train_ratio)

        # Calculate val size
        if val_max_size is not None:
            # Estimate dates needed for val_max_size sequences
            avg_seqs_per_date = len(self.sequences) / len(sorted_dates)
            val_size_dates = int(val_max_size / avg_seqs_per_date) + 1
            val_size_dates = min(val_size_dates, int(len(sorted_dates) * val_ratio))
            val_date_idx = train_date_idx + val_size_dates
        else:
            val_date_idx = int(len(sorted_dates) * (train_ratio + val_ratio))

        # Calculate test size
        if test_max_size is not None:
            # Estimate dates needed for test_max_size sequences
            avg_seqs_per_date = len(self.sequences) / len(sorted_dates)
            test_size_dates = int(test_max_size / avg_seqs_per_date) + 1
            test_size_dates = min(test_size_dates, len(sorted_dates) - val_date_idx)
            test_date_end_idx = val_date_idx + test_size_dates
        else:
            test_date_end_idx = len(sorted_dates)

        # Extract date ranges for each split
        train_dates = sorted_dates[:train_date_idx]
        val_dates = sorted_dates[train_date_idx:val_date_idx]
        test_dates = sorted_dates[val_date_idx:test_date_end_idx]

        # Collect sequences from appropriate date range
        if split == 'train':
            self.sequences = [seq for date in train_dates
                              for seq in sequences_by_date[date]]
            date_range_str = f"{train_dates[0]} to {train_dates[-1]}"
        elif split == 'val':
            self.sequences = [seq for date in val_dates
                              for seq in sequences_by_date[date]]
            date_range_str = f"{val_dates[0]} to {val_dates[-1]}"
        elif split == 'test':
            self.sequences = [seq for date in test_dates
                              for seq in sequences_by_date[date]]
            date_range_str = f"{test_dates[0]} to {test_dates[-1]}"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # NOW it's safe to shuffle (only within temporal range - doesn't break ordering)
        random.seed(seed)
        random.shuffle(self.sequences)

        # Verify temporal split
        print(f"\n  🔍 TEMPORAL SPLIT VERIFICATION:")
        print(f"    Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
        print(f"    Val dates:   {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} dates)")
        print(f"    Test dates:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates)")

        # Critical check: ensure no temporal overlap
        if train_dates[-1] >= val_dates[0]:
            raise ValueError(f"❌ TEMPORAL LEAKAGE DETECTED: Train overlaps with val!")
        if val_dates[-1] >= test_dates[0]:
            raise ValueError(f"❌ TEMPORAL LEAKAGE DETECTED: Val overlaps with test!")

        print(f"    ✅ VERIFIED: max(train) < min(val) < min(test)")
        print(f"    ✅ NO TEMPORAL LEAKAGE\n")

        print(f"  ✅ {split.upper()} split: {len(self.sequences)} sequences")
        print(f"    Date range: {date_range_str}")
        print(f"{'='*80}\n")

    def _get_cache_path(self, dataset_path: str, seq_len: int, min_future_days: int,
                        train_ratio: float, val_ratio: float) -> str:
        """
        Generate cache file path based on dataset and parameters.

        Returns path like: 'all_complete_dataset_temporal_split_abc123.h5'
        """
        # Create hash of parameters to detect changes
        param_str = f"{seq_len}_{min_future_days}_{train_ratio}_{val_ratio}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        # Generate cache filename
        dataset_base = os.path.splitext(dataset_path)[0]
        cache_path = f"{dataset_base}_temporal_split_{param_hash}.h5"

        return cache_path

    def _build_sequence_index(self):
        """
        Build index of all valid sequences.

        Returns:
            List of (ticker, end_idx) tuples
        """
        sequences = []

        for ticker in tqdm(list(self.h5f.keys()), desc="  Indexing"):
            group = self.h5f[ticker]
            num_dates = group.attrs['num_dates']

            # Find valid sequence positions
            for i in range(self.seq_len, num_dates - self.min_future_days):
                sequences.append((ticker, i))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            features: (seq_len, total_features) tensor
            prices: (pred_days,) tensor with future price ratios
            ticker: str
            date: str (YYYY-MM-DD)
        """
        ticker, end_idx = self.sequences[idx]

        # Get data from HDF5 (fast random access!)
        group = self.h5f[ticker]
        dates = group['dates'][:]
        features_2d = group['features'][:]  # (num_dates, num_features)

        # Extract sequence
        start_idx = end_idx - self.seq_len
        seq_features = features_2d[start_idx:end_idx]  # (seq_len, num_features)
        end_date = dates[end_idx - 1].decode('utf-8')

        # Convert to torch tensor
        features = torch.from_numpy(seq_features).float()

        # Pad if necessary
        if features.shape[1] < self.total_features:
            padding = torch.zeros(features.shape[0], self.total_features - features.shape[1])
            features = torch.cat([features, padding], dim=1)

        # Get current and future prices
        current_price = features[-1, 0]  # First feature is close price

        # Get future prices
        future_prices = []
        for pred_day in self.pred_days:
            future_idx = end_idx + pred_day
            if future_idx < len(features_2d):
                future_price = features_2d[future_idx, 0]
                price_ratio = future_price / current_price.item()
            else:
                price_ratio = 1.0  # No change if no data

            future_prices.append(price_ratio)

        prices = torch.tensor(future_prices, dtype=torch.float32)

        return features, prices, ticker, end_date

    def get_feature_splits(self, features):
        """
        Split features into base and news components.

        Args:
            features: (seq_len, total_features) or (batch, seq_len, total_features)

        Returns:
            base_features: (..., seq_len, base_features)
            news_features: (..., seq_len, news_dim)
        """
        base = features[..., :self.base_features]
        news = features[..., self.base_features:]
        return base, news

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5f'):
            self.h5f.close()


class StockDataModule:
    """
    Convenience class for managing train/val dataloaders with HDF5.

    Drop-in replacement for the pickle-based StockDataModule.
    """

    def __init__(self,
                 dataset_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seq_len: int = 60,
                 pred_days: List[int] = [1, 5, 10, 20],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 val_max_size: int = None,
                 test_max_size: int = None,
                 seed: int = 42):
        """
        Initialize data module.

        Args:
            dataset_path: Path to HDF5 file
            batch_size: Batch size
            num_workers: Number of data loader workers
            seq_len: Sequence length
            pred_days: Days to predict into future
            train_ratio: Train split ratio (default 0.7 = 70%)
            val_ratio: Val split ratio (default 0.15 = 15%, remaining 15% is test)
            val_max_size: Maximum validation set size (None = use val_ratio)
            test_max_size: Maximum test set size (None = use remaining after train/val)
            seed: Random seed
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.pred_days = pred_days
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.val_max_size = val_max_size
        self.test_max_size = test_max_size
        self.seed = seed

        # Create datasets
        self.train_dataset = HDF5StockDataset(
            dataset_path=dataset_path,
            seq_len=seq_len,
            pred_days=pred_days,
            split='train',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            val_max_size=val_max_size,
            test_max_size=test_max_size,
            seed=seed
        )

        self.val_dataset = HDF5StockDataset(
            dataset_path=dataset_path,
            seq_len=seq_len,
            pred_days=pred_days,
            split='val',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            val_max_size=val_max_size,
            test_max_size=test_max_size,
            seed=seed
        )

        self.test_dataset = HDF5StockDataset(
            dataset_path=dataset_path,
            seq_len=seq_len,
            pred_days=pred_days,
            split='test',
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            val_max_size=val_max_size,
            test_max_size=test_max_size,
            seed=seed
        )

        # Store feature dimensions from train dataset
        self.total_features = self.train_dataset.total_features
        self.base_features = self.train_dataset.base_features
        self.news_dim = self.train_dataset.news_dim

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


# Helper functions for binning (for classification approach)

# Global cache for bin edges
_BIN_EDGES_CACHE = {}

def compute_adaptive_bin_edges(dataset_path: str, num_bins: int = 100,
                                pred_days: List[int] = [1, 5, 10, 20],
                                max_samples: int = 50000) -> torch.Tensor:
    """
    Compute adaptive bin edges based on quantiles of actual price change distribution.

    For HDF5 format - loads data efficiently from HDF5 file.

    This creates more bins for common price changes (near 0%) and fewer bins for
    rare extreme changes, acting as a natural class balancing mechanism.

    Args:
        dataset_path: Path to HDF5 dataset file
        num_bins: Number of bins to create
        pred_days: Prediction horizons to compute bins for
        max_samples: Maximum samples to use for computing quantiles

    Returns:
        bin_edges: (num_bins+1,) tensor with bin edges
    """
    # Check cache first
    cache_key = (dataset_path, num_bins, tuple(pred_days), max_samples)
    if cache_key in _BIN_EDGES_CACHE:
        return _BIN_EDGES_CACHE[cache_key]

    print(f"\n📊 Computing adaptive bin edges from HDF5 data distribution...")
    print(f"  Num bins: {num_bins}")
    print(f"  Max samples: {max_samples}")

    # Open HDF5 file
    with h5py.File(dataset_path, 'r') as f:
        # Collect price ratios from dataset
        all_ratios = []
        sample_count = 0

        tickers = list(f.keys())

        for ticker in tqdm(tickers, desc="  Sampling"):
            if sample_count >= max_samples:
                break

            ticker_group = f[ticker]

            # Load the features array for this ticker
            features_2d = ticker_group['features'][:]  # (num_dates, num_features)
            num_dates = features_2d.shape[0]

            for i in range(num_dates - max(pred_days)):
                if sample_count >= max_samples:
                    break

                # Get current price (first feature)
                current_price = features_2d[i, 0]

                # Get future prices for all prediction horizons
                for pred_day in pred_days:
                    future_idx = i + pred_day
                    if future_idx < num_dates:
                        future_price = features_2d[future_idx, 0]
                        ratio = future_price / current_price
                        all_ratios.append(float(ratio))
                        sample_count += 1

    all_ratios = np.array(all_ratios)

    if len(all_ratios) == 0:
        raise ValueError(
            f"No samples collected! Check that:\n"
            f"  1. HDF5 file exists and has data: {dataset_path}\n"
            f"  2. File has correct structure (ticker/features/dates)\n"
            f"  3. pred_days {pred_days} are valid for the dataset"
        )

    print(f"\n  📈 Distribution Statistics:")
    print(f"    Samples collected: {len(all_ratios):,}")
    print(f"    Mean ratio: {all_ratios.mean():.4f}")
    print(f"    Std ratio: {all_ratios.std():.4f}")
    print(f"    Min ratio: {all_ratios.min():.4f}")
    print(f"    Max ratio: {all_ratios.max():.4f}")
    print(f"    Median ratio: {np.median(all_ratios):.4f}")

    # Compute quantile-based bin edges (adaptive binning)
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(all_ratios, q=quantiles)
    bin_edges = torch.tensor(bin_edges, dtype=torch.float32)

    # Show bin width distribution
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    print(f"\n  📏 Bin Width Statistics:")
    print(f"    Min width: {bin_widths.min():.6f}")
    print(f"    Max width: {bin_widths.max():.6f}")
    print(f"    Median width: {bin_widths.median():.6f}")
    print(f"    Center bins (45-55) avg width: {bin_widths[45:55].mean():.6f}")
    print(f"    Edge bins (0-10, 90-100) avg width: {torch.cat([bin_widths[:10], bin_widths[-10:]]).mean():.6f}")
    print(f"    Ratio (edge/center): {torch.cat([bin_widths[:10], bin_widths[-10:]]).mean() / bin_widths[45:55].mean():.2f}x")

    # Cache the result
    _BIN_EDGES_CACHE[cache_key] = bin_edges

    return bin_edges


def convert_price_ratio_to_bins_adaptive(ratio: float, bin_edges: torch.Tensor) -> int:
    """
    Convert price ratio to bin index using adaptive (quantile-based) bins.

    Args:
        ratio: Price ratio (future_price / current_price)
        bin_edges: (num_bins+1,) tensor with bin edges from compute_adaptive_bin_edges

    Returns:
        bin_idx: Integer bin index [0, num_bins-1]

    Examples:
        With 100 bins and typical stock data:
        - Bin 0-20: Large losses (< -5%)
        - Bin 20-40: Medium losses (-5% to -1%)
        - Bin 40-60: Small changes (-1% to +1%)  <- Most granular bins
        - Bin 60-80: Medium gains (+1% to +5%)
        - Bin 80-99: Large gains (> +5%)
    """
    # Use searchsorted to find bin
    bin_idx = torch.searchsorted(bin_edges, ratio, right=False) - 1

    # Clip to valid range [0, num_bins-1]
    num_bins = len(bin_edges) - 1
    bin_idx = int(np.clip(bin_idx, 0, num_bins - 1))

    return bin_idx


def convert_price_ratio_to_one_hot(ratio: float, num_bins: int = 100,
                                    bin_edges: torch.Tensor = None) -> torch.Tensor:
    """
    Convert price ratio to one-hot encoded bin.

    Args:
        ratio: Price ratio (future_price / current_price)
        num_bins: Number of bins (only used if bin_edges not provided)
        bin_edges: Optional precomputed adaptive bin edges

    Returns:
        one_hot: (num_bins,) tensor with 1.0 at the predicted bin
    """
    if bin_edges is not None:
        # Use adaptive binning
        bin_idx = convert_price_ratio_to_bins_adaptive(ratio, bin_edges)
        num_bins = len(bin_edges) - 1
    else:
        # Fallback to uniform binning (not recommended)
        print("⚠️  WARNING: Using uniform binning. For better results, provide bin_edges.")
        pct_change = (ratio - 1.0)
        max_change = 0.2
        pct_change = np.clip(pct_change, -max_change, max_change)
        normalized = (pct_change + max_change) / (2 * max_change)
        bin_idx = int(normalized * (num_bins - 1))
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    one_hot = torch.zeros(num_bins)
    one_hot[bin_idx] = 1.0
    return one_hot


if __name__ == '__main__':
    """Test the HDF5 data loader"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hdf5_data_loader.py <dataset.h5>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    print("\n" + "="*80)
    print("TESTING HDF5 DATA LOADER")
    print("="*80)

    # Create data module
    dm = StockDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        seq_len=60,
        pred_days=[1, 5, 10, 20],
        num_workers=2
    )

    print(f"\n📊 Dataset Info:")
    print(f"  Total features: {dm.total_features}")
    print(f"  Base features: {dm.base_features}")
    print(f"  News features: {dm.news_dim}")
    print(f"  Train samples: {len(dm.train_dataset)}")
    print(f"  Val samples: {len(dm.val_dataset)}")

    # Test train loader
    print(f"\n🔄 Testing train loader...")
    train_loader = dm.train_dataloader()

    import time
    start = time.time()

    for batch_idx, (features, prices, tickers, dates) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Features shape: {features.shape}")  # (batch, seq_len, features)
        print(f"  Prices shape: {prices.shape}")      # (batch, pred_days)
        print(f"  Tickers: {tickers[:3]}")
        print(f"  Dates: {dates[:3]}")

        # Test feature splitting
        base, news = dm.train_dataset.get_feature_splits(features)
        print(f"  Base features shape: {base.shape}")
        print(f"  News features shape: {news.shape}")

        if batch_idx >= 2:
            break

    elapsed = time.time() - start
    print(f"\n⏱️  Loaded {batch_idx + 1} batches in {elapsed:.2f}s ({elapsed/(batch_idx+1):.3f}s per batch)")
    print(f"\n✅ HDF5 data loader test complete!")
