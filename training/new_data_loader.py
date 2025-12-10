"""
Data Loader for New Dataset Format

Handles the new dataset format from the enhanced FMP processor:
- Input: Dict[ticker, Dict[date, torch.Tensor]]
- Each tensor contains: [Base Features] + [Cross-Sectional Features] + [News Embeddings]
- Creates sequences for training
- Handles variable feature dimensions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm

from utils.utils import pic_load


class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset for stock prediction with new data format.

    Creates sequences of length seq_len from daily feature tensors.
    Supports both regression and classification targets.
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
            dataset_path: Path to pickle file with format {ticker: {date: tensor}}
            seq_len: Sequence length for training
            pred_days: List of future days to predict (e.g., [1, 5, 10, 20])
            min_future_days: Minimum days of future data required
            news_dim: Dimension of news embeddings (default 768)
            split: 'train', 'val', or 'test'
            train_ratio: Ratio for train split (default 0.7 = 70%)
            val_ratio: Ratio for val split (default 0.15 = 15%, remaining 15% is test)
            val_max_size: Maximum validation set size (None = use val_ratio)
            test_max_size: Maximum test set size (None = use remaining after train/val)
            seed: Random seed for reproducibility
        """
        self.seq_len = seq_len
        self.pred_days = sorted(pred_days)
        self.min_future_days = min_future_days
        self.news_dim = news_dim
        self.split = split

        print(f"\n{'='*80}")
        print(f"Loading dataset: {dataset_path}")
        print(f"{'='*80}")

        # Load data
        self.data = pic_load(dataset_path)
        print(f"  ✅ Loaded {len(self.data)} tickers")

        # Analyze feature dimensions
        self._analyze_features()

        # Build index of valid sequences
        print(f"\n  📦 Building sequence index...")
        self.sequences = self._build_sequence_index()

        # Train/val/test split
        random.seed(seed)
        random.shuffle(self.sequences)

        train_idx = int(len(self.sequences) * train_ratio)

        # Calculate val size (either from ratio or max_size)
        if val_max_size is not None:
            val_size = min(val_max_size, int(len(self.sequences) * val_ratio))
            val_idx = train_idx + val_size
        else:
            val_idx = int(len(self.sequences) * (train_ratio + val_ratio))

        # Calculate test size (either from max_size or remaining)
        if test_max_size is not None:
            test_size = min(test_max_size, len(self.sequences) - val_idx)
            test_end_idx = val_idx + test_size
        else:
            test_end_idx = len(self.sequences)

        if split == 'train':
            self.sequences = self.sequences[:train_idx]
        elif split == 'val':
            self.sequences = self.sequences[train_idx:val_idx]
        elif split == 'test':
            self.sequences = self.sequences[val_idx:test_end_idx]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        print(f"\n  ✅ {split.upper()} split: {len(self.sequences)} sequences")
        print(f"{'='*80}\n")

    def _analyze_features(self):
        """Analyze feature dimensions from sample data."""
        # Find maximum feature dimension across all stocks
        print(f"\n  🔍 Analyzing feature dimensions across all tickers...")
        max_features = 0
        min_features = float('inf')
        feature_dims = []

        for ticker, date_dict in tqdm(self.data.items(), desc="  Scanning"):
            if date_dict:
                sample_date = list(date_dict.keys())[0]
                num_features = date_dict[sample_date].shape[0]
                feature_dims.append(num_features)
                max_features = max(max_features, num_features)
                min_features = min(min_features, num_features)

        self.total_features = max_features
        self.base_features = self.total_features - self.news_dim

        print(f"\n  📊 Feature Dimensions:")
        print(f"    Maximum features: {max_features}")
        print(f"    Minimum features: {min_features}")
        if max_features != min_features:
            print(f"    ⚠️  VARIABLE DIMENSIONS DETECTED - will pad to {max_features}")
            print(f"    Feature range: {min_features} to {max_features}")
        print(f"    Using: {self.total_features} (padded)")
        print(f"    Base features: {self.base_features}")
        print(f"    News embeddings: {self.news_dim}")

    def _build_sequence_index(self):
        """
        Build index of all valid sequences.

        Returns:
            List of (ticker, end_date_idx) tuples
        """
        sequences = []

        for ticker, date_dict in tqdm(self.data.items(), desc="  Indexing"):
            # Get sorted dates
            dates = sorted(list(date_dict.keys()))

            # Find valid sequence positions
            # Need: seq_len days of history + min_future_days of future data
            for i in range(self.seq_len, len(dates) - self.min_future_days):
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
            date: datetime.date
        """
        ticker, end_idx = self.sequences[idx]

        # Get sorted dates for this ticker
        dates = sorted(list(self.data[ticker].keys()))

        # Get sequence dates
        seq_dates = dates[end_idx - self.seq_len:end_idx]
        end_date = dates[end_idx]

        # Build feature sequence
        features = []
        for date in seq_dates:
            tensor = self.data[ticker][date]

            # Pad if necessary
            if tensor.shape[0] < self.total_features:
                padding = torch.zeros(self.total_features - tensor.shape[0])
                tensor = torch.cat([tensor, padding])

            features.append(tensor)
        features = torch.stack(features)  # (seq_len, total_features)

        # Get current and future prices
        # Assumes last feature is the price (or we need to extract it differently)
        current_price = features[-1, 0]  # Assuming first feature is close price

        # Get future prices
        future_prices = []
        for pred_day in self.pred_days:
            future_date_idx = end_idx + pred_day
            if future_date_idx < len(dates):
                future_date = dates[future_date_idx]
                future_tensor = self.data[ticker][future_date]
                future_price = future_tensor[0]  # First feature is close price
                price_ratio = future_price / current_price
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


class StockDataModule:
    """
    Convenience class for managing train/val dataloaders.
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
        self.train_dataset = StockSequenceDataset(
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

        self.val_dataset = StockSequenceDataset(
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

        self.test_dataset = StockSequenceDataset(
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
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# Helper functions for binning (for classification approach)

# Global cache for bin edges
_BIN_EDGES_CACHE = {}

def compute_adaptive_bin_edges(dataset_path: str, num_bins: int = 100,
                                pred_days: List[int] = [1, 5, 10, 20],
                                max_samples: int = 50000) -> torch.Tensor:
    """
    Compute adaptive bin edges based on quantiles of actual price change distribution.

    This creates more bins for common price changes (near 0%) and fewer bins for
    rare extreme changes, acting as a natural class balancing mechanism.

    Args:
        dataset_path: Path to dataset pickle
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

    print(f"\n📊 Computing adaptive bin edges from data distribution...")
    print(f"  Num bins: {num_bins}")
    print(f"  Max samples: {max_samples}")

    # Load dataset
    data = pic_load(dataset_path)

    # Collect price ratios from dataset
    all_ratios = []
    sample_count = 0

    for ticker, date_dict in tqdm(data.items(), desc="  Sampling"):
        dates = sorted(list(date_dict.keys()))

        for i in range(len(dates) - max(pred_days)):
            if sample_count >= max_samples:
                break

            # Get current price
            current_price = date_dict[dates[i]][0]  # First feature is close price

            # Get future prices for all prediction horizons
            for pred_day in pred_days:
                future_idx = i + pred_day
                if future_idx < len(dates):
                    future_price = date_dict[dates[future_idx]][0]
                    ratio = future_price / current_price
                    all_ratios.append(ratio.item())
                    sample_count += 1

        if sample_count >= max_samples:
            break

    all_ratios = np.array(all_ratios)

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
    """Test the data loader"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python new_data_loader.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    print("\n" + "="*80)
    print("TESTING DATA LOADER")
    print("="*80)

    # Create data module
    dm = StockDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        seq_len=60,
        pred_days=[1, 5, 10, 20]
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

    print(f"\n✅ Data loader test complete!")
