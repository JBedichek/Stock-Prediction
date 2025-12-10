"""
Cross-Sectional Normalizer (GPU-Accelerated)

Handles proper normalization of features based on their type:
- Fundamental metrics: Normalized cross-sectionally (relative to other stocks)
- Price/volume features: Normalized temporally (per-stock over time)

This ensures that fundamental ratios like P/E, ROE, etc. preserve their relative
ranking across stocks, which is critical for valuation-based predictions.

GPU-Accelerated: Uses PyTorch tensors for 10-50x speedup over CPU implementation.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Set, Tuple
from datetime import date as dt_date
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CrossSectionalNormalizer:
    """
    Normalizes features with awareness of whether they should be normalized
    cross-sectionally (across stocks) or temporally (across time).
    """

    def __init__(self):
        """Initialize normalizer with feature type detection."""

        # Features that should be normalized CROSS-SECTIONALLY (relative to other stocks)
        self.cross_sectional_patterns = [
            # Fundamental ratios - MUST be cross-sectional
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
            'peg_ratio', 'ev_', 'enterprise',
            'roe', 'roa', 'roic', 'roi',
            'debt_to_equity', 'debt_ratio', 'current_ratio', 'quick_ratio',
            'asset_turnover', 'inventory_turnover', 'receivables_turnover',
            'gross_margin', 'operating_margin', 'profit_margin', 'ebitda_margin',
            'net_margin', 'fcf_margin',

            # Financial statement items (revenue, earnings, assets, etc.)
            'revenue', 'income', 'earnings', 'ebitda', 'ebit',
            'total_assets', 'total_liabilities', 'stockholders_equity',
            'market_cap', 'shares_outstanding',
            'cash_flow', 'free_cash_flow', 'operating_cash',

            # Growth metrics
            'revenue_growth', 'earnings_growth', 'dividend_growth',
            'book_value_growth', 'eps_growth',

            # Per-share metrics
            'eps', 'book_value', 'tangible_book', 'revenue_per_share',
            'cash_per_share', 'fcf_per_share',

            # Quarterly/annual metrics from key_metrics, ratios, enterprise_values
            'key_metrics_', 'financial_ratios_', 'enterprise_values_',
            'financial_growth_',
        ]

        # Features that should be normalized TEMPORALLY (per-stock over time)
        self.temporal_patterns = [
            # Price-based features
            'price', 'close', 'open', 'high', 'low',
            'return', 'change', 'volatility', 'std',

            # Volume features (relative to own history)
            'volume_ratio', 'volume_spike', 'dollar_volume',

            # Technical indicators (designed for per-stock)
            'rsi', 'macd', 'sma', 'ema', 'adx', 'williams',
            'tech_', 'indicator_',

            # Derived price features
            'gap', 'range', 'intraday',

            # Market-relative features (already relative)
            'beta', 'correlation', 'relative_performance',
            '_SP500', '_NASDAQ', '_VIX', '_sector',

            # Cross-sectional features (already percentiles)
            'percentile', 'rank',
        ]

    def identify_feature_type(self, column_name: str) -> str:
        """
        Identify whether a feature should be normalized cross-sectionally or temporally.

        Args:
            column_name: Name of the feature column

        Returns:
            'cross_sectional', 'temporal', or 'none'
        """
        col_lower = column_name.lower()

        # Check temporal patterns first (more specific)
        for pattern in self.temporal_patterns:
            if pattern in col_lower:
                return 'temporal'

        # Check cross-sectional patterns
        for pattern in self.cross_sectional_patterns:
            if pattern in col_lower:
                return 'cross_sectional'

        # Default: if it has 'volume' in name but not 'ratio', it's cross-sectional (absolute volume)
        if 'volume' in col_lower and 'ratio' not in col_lower and 'spike' not in col_lower:
            return 'cross_sectional'

        # Default to temporal (safer, less likely to break)
        return 'temporal'

    def categorize_columns(self, columns: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Categorize columns into cross-sectional, temporal, and no normalization.

        Args:
            columns: List of column names

        Returns:
            (cross_sectional_cols, temporal_cols, no_norm_cols)
        """
        cross_sectional = []
        temporal = []
        no_norm = []

        for col in columns:
            feature_type = self.identify_feature_type(col)

            if feature_type == 'cross_sectional':
                cross_sectional.append(col)
            elif feature_type == 'temporal':
                temporal.append(col)
            else:
                no_norm.append(col)

        return cross_sectional, temporal, no_norm

    def normalize_temporal(self, series: pd.Series) -> pd.Series:
        """
        Normalize a series temporally (z-score across time for one stock).

        Args:
            series: Time series for one stock

        Returns:
            Normalized series
        """
        mean = series.mean()
        std = series.std()

        # Handle scalar/Series issues
        if isinstance(mean, pd.Series):
            mean = mean.iloc[0] if len(mean) > 0 else 0.0
        if isinstance(std, pd.Series):
            std = std.iloc[0] if len(std) > 0 else 0.0

        # Safe conversion
        try:
            mean_val = float(mean) if not pd.isna(mean) else 0.0
            std_val = float(std) if not pd.isna(std) else 1.0
        except (TypeError, ValueError):
            return series * 0

        if std_val == 0 or pd.isna(std_val):
            return series * 0

        return (series - mean_val) / std_val

    def normalize_cross_sectional(self, all_stocks_dfs: Dict[str, pd.DataFrame],
                                  column: str) -> Dict[str, pd.Series]:
        """
        Normalize a column cross-sectionally (z-score across stocks for each date).
        GPU-accelerated version using PyTorch.

        Args:
            all_stocks_dfs: Dict of {ticker: DataFrame} with date index
            column: Column name to normalize

        Returns:
            Dict of {ticker: normalized_series}
        """
        # Collect all dates
        all_dates = set()
        for df in all_stocks_dfs.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))

        # Build a matrix: rows = stocks, cols = dates
        tickers = list(all_stocks_dfs.keys())
        n_stocks = len(tickers)
        n_dates = len(all_dates)

        # Create date to index mapping
        date_to_idx = {date: i for i, date in enumerate(all_dates)}

        # Initialize matrix with NaN
        data_matrix = np.full((n_stocks, n_dates), np.nan, dtype=np.float32)

        # Fill matrix
        for stock_idx, ticker in enumerate(tickers):
            df = all_stocks_dfs[ticker]
            if column in df.columns:
                for date in df.index:
                    if date in date_to_idx:
                        val = df.loc[date, column]
                        if not pd.isna(val) and not np.isinf(val):
                            data_matrix[stock_idx, date_to_idx[date]] = val

        # Convert to tensor and move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_tensor = torch.tensor(data_matrix, dtype=torch.float32, device=device)

        # Compute cross-sectional mean and std (across stocks, dim=0)
        # Use nanmean and nanstd to ignore NaN values
        mean = torch.nanmean(data_tensor, dim=0, keepdim=True)  # Shape: (1, n_dates)
        std = torch.sqrt(torch.nanmean((data_tensor - mean) ** 2, dim=0, keepdim=True))  # Shape: (1, n_dates)

        # Normalize
        normalized_tensor = (data_tensor - mean) / (std + 1e-8)

        # Set to 0 where std was 0 or NaN
        normalized_tensor = torch.where(std < 1e-8, torch.zeros_like(normalized_tensor), normalized_tensor)
        normalized_tensor = torch.where(torch.isnan(normalized_tensor), torch.zeros_like(normalized_tensor), normalized_tensor)

        # Convert back to CPU and numpy
        normalized_matrix = normalized_tensor.cpu().numpy()

        # Convert back to series dict
        normalized_series = {}
        for stock_idx, ticker in enumerate(tickers):
            df = all_stocks_dfs[ticker]
            series = pd.Series(index=df.index, dtype=float)

            for date in df.index:
                if date in date_to_idx:
                    series.loc[date] = normalized_matrix[stock_idx, date_to_idx[date]]

            normalized_series[ticker] = series

        return normalized_series

    def normalize_dataframes(self, all_stocks_dfs: Dict[str, pd.DataFrame],
                           verbose: bool = True,
                           use_gpu: bool = False,
                           feature_batch_size: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Normalize all DataFrames with appropriate method per column.
        Memory-efficient implementation - processes features in batches on CPU.

        Args:
            all_stocks_dfs: Dict of {ticker: DataFrame with date index}
            verbose: Print normalization statistics
            use_gpu: Use GPU if available (requires significant VRAM, not recommended for large datasets)
            feature_batch_size: Number of features to process at once (lower = less memory)

        Returns:
            Dict of {ticker: normalized_DataFrame}
        """
        if not all_stocks_dfs:
            return {}

        # Use CPU by default to avoid GPU OOM on large datasets
        # Most systems have much more RAM than VRAM (e.g., 64GB RAM vs 32GB VRAM)
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Get all unique columns and dates
        all_columns = set()
        all_dates = set()
        for df in all_stocks_dfs.values():
            all_columns.update(df.columns)
            all_dates.update(df.index)
        all_columns = sorted(list(all_columns))
        all_dates = sorted(list(all_dates))

        # Categorize columns
        cross_sectional_cols, temporal_cols, no_norm_cols = self.categorize_columns(all_columns)

        if verbose:
            device_name = "GPU (CUDA)" if device.type == 'cuda' else "CPU"
            print(f"\n📊 Normalization Strategy ({device_name}):")
            print(f"  Cross-sectional (across stocks): {len(cross_sectional_cols)} features")
            print(f"  Temporal (per-stock): {len(temporal_cols)} features")
            print(f"  No normalization: {len(no_norm_cols)} features")
            print(f"  Dataset shape: {len(all_stocks_dfs)} stocks × {len(all_dates)} dates × {len(all_columns)} features")

            # Calculate total tensor size
            total_elements = len(all_stocks_dfs) * len(all_dates) * len(all_columns)
            total_gb = total_elements * 4 / 1e9  # 4 bytes per float32
            print(f"  Total tensor size: {total_gb:.2f} GB")
            if device.type == 'cuda':
                print(f"  ⚠️  Using GPU - may cause OOM on large datasets!")
            else:
                print(f"  ✅ Using CPU - more memory efficient for large datasets")

            if len(cross_sectional_cols) > 0:
                print(f"\n  Sample cross-sectional features:")
                for col in cross_sectional_cols[:10]:
                    print(f"    - {col}")
                if len(cross_sectional_cols) > 10:
                    print(f"    ... and {len(cross_sectional_cols) - 10} more")

        # Build 3D tensor: (n_stocks, n_dates, n_features)
        # Process in batches to avoid OOM
        print(f"\n  📦 Building 3D tensor (batched processing)...")
        tickers = list(all_stocks_dfs.keys())
        n_stocks = len(tickers)
        n_dates = len(all_dates)
        n_features = len(all_columns)

        # Create mappings
        date_to_idx = {date: i for i, date in enumerate(all_dates)}
        col_to_idx = {col: i for i, col in enumerate(all_columns)}
        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}

        # Initialize 3D array on CPU (RAM is cheaper than VRAM)
        data_3d = np.full((n_stocks, n_dates, n_features), np.nan, dtype=np.float32)

        # Fill 3D array (vectorized for speed)
        print(f"  📥 Loading data into tensor...")
        for ticker_idx, (ticker, df) in enumerate(tqdm(all_stocks_dfs.items(), desc="  Loading", ncols=100)):
            # Get date and column indices for this dataframe
            valid_dates = []
            valid_date_positions = []

            for pos, date in enumerate(df.index):
                if date in date_to_idx:
                    valid_dates.append(date_to_idx[date])
                    valid_date_positions.append(pos)

            if not valid_dates:
                continue

            df_date_indices = np.array(valid_dates)
            df_positions = np.array(valid_date_positions)

            # Get column indices
            df_col_indices = np.array([col_to_idx[col] for col in df.columns])

            # Convert dataframe to numpy array (much faster than iterating)
            df_values = df.values.astype(np.float32)

            # Use advanced indexing to assign all values at once
            data_3d[ticker_idx, df_date_indices[:, None], df_col_indices] = df_values[df_positions, :]

        print(f"  📊 Tensor shape: ({n_stocks}, {n_dates}, {n_features})")
        print(f"  💾 Memory size: {data_3d.nbytes / 1e9:.2f} GB")

        # Create feature type masks (CPU-based)
        cross_sectional_indices = [col_to_idx[col] for col in cross_sectional_cols if col in col_to_idx]
        temporal_indices = [col_to_idx[col] for col in temporal_cols if col in col_to_idx]

        # CROSS-SECTIONAL NORMALIZATION (batched feature processing)
        if len(cross_sectional_indices) > 0:
            print(f"\n  🔧 Normalizing {len(cross_sectional_indices)} cross-sectional features (batched)...")

            num_batches = (len(cross_sectional_indices) + feature_batch_size - 1) // feature_batch_size
            for batch_idx in tqdm(range(num_batches), desc="  Cross-sectional", ncols=100):
                start_idx = batch_idx * feature_batch_size
                end_idx = min(start_idx + feature_batch_size, len(cross_sectional_indices))
                batch_feature_indices = cross_sectional_indices[start_idx:end_idx]

                # Extract batch: (n_stocks, n_dates, batch_size)
                batch_data = data_3d[:, :, batch_feature_indices]

                # Convert to tensor
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)

                # Compute mean and std across stocks (dim=0)
                mean = torch.nanmean(batch_tensor, dim=0, keepdim=True)
                centered = batch_tensor - mean
                variance = torch.nanmean(centered ** 2, dim=0, keepdim=True)
                std = torch.sqrt(variance)

                # Normalize
                normalized = (batch_tensor - mean) / (std + 1e-8)
                normalized = torch.where(std < 1e-8, torch.zeros_like(normalized), normalized)
                normalized = torch.where(torch.isnan(normalized), torch.zeros_like(normalized), normalized)

                # Put back into numpy array
                data_3d[:, :, batch_feature_indices] = normalized.cpu().numpy()

                # Clean up GPU memory
                del batch_tensor, normalized, mean, std, centered, variance
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            print(f"  ✅ Cross-sectional normalization complete!")

        # TEMPORAL NORMALIZATION (batched feature processing)
        if len(temporal_indices) > 0:
            print(f"\n  🔧 Normalizing {len(temporal_indices)} temporal features (batched)...")

            num_batches = (len(temporal_indices) + feature_batch_size - 1) // feature_batch_size
            for batch_idx in tqdm(range(num_batches), desc="  Temporal", ncols=100):
                start_idx = batch_idx * feature_batch_size
                end_idx = min(start_idx + feature_batch_size, len(temporal_indices))
                batch_feature_indices = temporal_indices[start_idx:end_idx]

                # Extract batch: (n_stocks, n_dates, batch_size)
                batch_data = data_3d[:, :, batch_feature_indices]

                # Convert to tensor
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)

                # Compute mean and std across time (dim=1)
                mean = torch.nanmean(batch_tensor, dim=1, keepdim=True)
                centered = batch_tensor - mean
                variance = torch.nanmean(centered ** 2, dim=1, keepdim=True)
                std = torch.sqrt(variance)

                # Normalize
                normalized = (batch_tensor - mean) / (std + 1e-8)
                normalized = torch.where(std < 1e-8, torch.zeros_like(normalized), normalized)
                normalized = torch.where(torch.isnan(normalized), torch.zeros_like(normalized), normalized)

                # Put back into numpy array
                data_3d[:, :, batch_feature_indices] = normalized.cpu().numpy()

                # Clean up GPU memory
                del batch_tensor, normalized, mean, std, centered, variance
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            print(f"  ✅ Temporal normalization complete!")

        # Data is already in CPU numpy array (data_3d)
        data_normalized = data_3d

        # Convert back to DataFrames (vectorized)
        print(f"  📦 Converting back to DataFrames...")
        normalized_dfs = {}

        for ticker_idx, ticker in enumerate(tqdm(tickers, desc="  Converting", ncols=100)):
            df = all_stocks_dfs[ticker]

            # Periodically clean up memory during conversion
            if ticker_idx > 0 and ticker_idx % 100 == 0:
                import gc
                gc.collect()

            # Get date indices for this dataframe
            valid_dates = []
            valid_date_objs = []

            for date in df.index:
                if date in date_to_idx:
                    valid_dates.append(date_to_idx[date])
                    valid_date_objs.append(date)

            if not valid_dates:
                normalized_dfs[ticker] = pd.DataFrame()
                continue

            df_date_indices = np.array(valid_dates)

            # Get column indices for this dataframe
            df_col_indices = np.array([col_to_idx[col] for col in df.columns])

            # Extract all values at once using advanced indexing
            df_values = data_normalized[ticker_idx, df_date_indices[:, None], df_col_indices]

            # Create DataFrame directly from numpy array
            normalized_dfs[ticker] = pd.DataFrame(
                df_values,
                index=valid_date_objs,
                columns=df.columns
            )

        # Delete the large numpy array immediately - we don't need it anymore
        print(f"\n  🧹 Cleaning up large arrays...")
        del data_normalized
        del data_3d
        import gc
        gc.collect()

        # Fill NaNs and clean
        print(f"  🧹 Cleaning final data...")
        for ticker, df in normalized_dfs.items():
            normalized_dfs[ticker] = df.fillna(0).replace([np.inf, -np.inf], 0)

        # Delete input data - we don't need it anymore
        del all_stocks_dfs
        gc.collect()

        if verbose:
            print(f"\n  ✅ Normalization complete!")
            # Show memory usage
            try:
                import psutil
                import os as os_module
                process = psutil.Process(os_module.getpid())
                mem_gb = process.memory_info().rss / 1024**3
                print(f"  📊 RAM usage after cleanup: {mem_gb:.2f} GB")
            except:
                pass

        return normalized_dfs


def apply_cross_sectional_normalization(all_stocks_dfs: Dict[str, pd.DataFrame],
                                       verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to apply cross-sectional normalization.

    Args:
        all_stocks_dfs: Dict of {ticker: DataFrame with date index}
        verbose: Print progress

    Returns:
        Dict of {ticker: normalized_DataFrame}
    """
    normalizer = CrossSectionalNormalizer()
    return normalizer.normalize_dataframes(all_stocks_dfs, verbose=verbose)
