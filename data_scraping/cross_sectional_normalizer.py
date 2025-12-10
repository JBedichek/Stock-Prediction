"""
Cross-Sectional Normalizer

Handles proper normalization of features based on their type:
- Fundamental metrics: Normalized cross-sectionally (relative to other stocks)
- Price/volume features: Normalized temporally (per-stock over time)

This ensures that fundamental ratios like P/E, ROE, etc. preserve their relative
ranking across stocks, which is critical for valuation-based predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from datetime import date as dt_date
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

        # For each date, collect values across all stocks
        normalized_series = {ticker: pd.Series(index=df.index, dtype=float)
                           for ticker, df in all_stocks_dfs.items()}

        for date in all_dates:
            # Collect values for this date across all stocks
            values_on_date = {}
            for ticker, df in all_stocks_dfs.items():
                if date in df.index and column in df.columns:
                    val = df.loc[date, column]
                    if not pd.isna(val) and not np.isinf(val):
                        values_on_date[ticker] = val

            if len(values_on_date) < 2:
                # Not enough data for normalization
                continue

            # Calculate cross-sectional mean and std
            values = list(values_on_date.values())
            cross_mean = np.mean(values)
            cross_std = np.std(values)

            if cross_std == 0 or np.isnan(cross_std):
                # No variation across stocks - set to 0
                for ticker in values_on_date:
                    if date in normalized_series[ticker].index:
                        normalized_series[ticker].loc[date] = 0.0
            else:
                # Normalize each stock's value
                for ticker, val in values_on_date.items():
                    if date in normalized_series[ticker].index:
                        normalized_series[ticker].loc[date] = (val - cross_mean) / cross_std

        return normalized_series

    def normalize_dataframes(self, all_stocks_dfs: Dict[str, pd.DataFrame],
                           verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Normalize all DataFrames with appropriate method per column.

        Args:
            all_stocks_dfs: Dict of {ticker: DataFrame with date index}
            verbose: Print normalization statistics

        Returns:
            Dict of {ticker: normalized_DataFrame}
        """
        if not all_stocks_dfs:
            return {}

        # Get all unique columns across all stocks
        all_columns = set()
        for df in all_stocks_dfs.values():
            all_columns.update(df.columns)
        all_columns = sorted(list(all_columns))

        # Categorize columns
        cross_sectional_cols, temporal_cols, no_norm_cols = self.categorize_columns(all_columns)

        if verbose:
            print(f"\n📊 Normalization Strategy:")
            print(f"  Cross-sectional (across stocks): {len(cross_sectional_cols)} features")
            print(f"  Temporal (per-stock): {len(temporal_cols)} features")
            print(f"  No normalization: {len(no_norm_cols)} features")

            if len(cross_sectional_cols) > 0:
                print(f"\n  Sample cross-sectional features:")
                for col in cross_sectional_cols[:10]:
                    print(f"    - {col}")
                if len(cross_sectional_cols) > 10:
                    print(f"    ... and {len(cross_sectional_cols) - 10} more")

        # Initialize normalized DataFrames
        normalized_dfs = {ticker: df.copy() for ticker, df in all_stocks_dfs.items()}

        # Normalize cross-sectional features
        if cross_sectional_cols:
            print(f"\n  🔄 Normalizing cross-sectional features...")
            for i, col in enumerate(cross_sectional_cols):
                if (i + 1) % 20 == 0 or (i + 1) == len(cross_sectional_cols):
                    print(f"    {i+1}/{len(cross_sectional_cols)}: {col}")

                # Normalize this column across all stocks
                normalized_series_dict = self.normalize_cross_sectional(all_stocks_dfs, col)

                # Update each stock's DataFrame
                for ticker, norm_series in normalized_series_dict.items():
                    if ticker in normalized_dfs and col in normalized_dfs[ticker].columns:
                        normalized_dfs[ticker][col] = norm_series

        # Normalize temporal features
        if temporal_cols:
            print(f"\n  🔄 Normalizing temporal features...")
            for ticker, df in normalized_dfs.items():
                if (list(normalized_dfs.keys()).index(ticker) + 1) % 50 == 0:
                    print(f"    Stock {list(normalized_dfs.keys()).index(ticker)+1}/{len(normalized_dfs)}")

                for col in temporal_cols:
                    if col in df.columns:
                        df[col] = self.normalize_temporal(df[col])

        # Fill NaNs
        for ticker, df in normalized_dfs.items():
            normalized_dfs[ticker] = df.fillna(0).replace([np.inf, -np.inf], 0)

        if verbose:
            print(f"\n  ✅ Normalization complete!")

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
