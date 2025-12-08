"""
Cross-Sectional Rankings Calculator

Calculates percentile ranks and relative metrics across all stocks for each date.
These features capture where a stock stands relative to the universe of stocks.

Features include:
- Return percentile ranks (1d, 5d, 20d)
- Volume percentile ranks
- Volatility percentile ranks
- Sector-relative metrics
- Momentum/strength rankings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import percentileofscore
from datetime import date as dt_date
import warnings
warnings.filterwarnings('ignore')


class CrossSectionalCalculator:
    """
    Calculate cross-sectional features (percentile ranks across all stocks).
    """

    def __init__(self, sector_dict: Optional[Dict[str, str]] = None):
        """
        Initialize calculator.

        Args:
            sector_dict: Optional dict of {ticker: sector} for sector-relative metrics
        """
        self.sector_dict = sector_dict

    def calculate_percentile_rank(self, value: float, all_values: List[float]) -> float:
        """
        Calculate percentile rank of a value within a list.

        Args:
            value: The value to rank
            all_values: List of all values

        Returns:
            Percentile rank (0-100)
        """
        if np.isnan(value) or len(all_values) == 0:
            return 50.0  # Default to median if NaN

        # Filter out NaN values
        all_values = [v for v in all_values if not np.isnan(v)]

        if len(all_values) == 0:
            return 50.0

        try:
            return percentileofscore(all_values, value, kind='rank')
        except:
            return 50.0

    def calculate_cross_sectional_features_for_date(self,
                                                   all_stocks_data: Dict[str, pd.DataFrame],
                                                   date: dt_date) -> Dict[str, Dict[str, float]]:
        """
        Calculate cross-sectional features for all stocks on a specific date.

        Args:
            all_stocks_data: Dict of {ticker: DataFrame with features}
            date: Date to calculate features for

        Returns:
            Dict of {ticker: {feature_name: value}}
        """
        # Collect data for all stocks on this date
        stocks_on_date = {}
        for ticker, df in all_stocks_data.items():
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date

            date_data = df[df['date'] == date]

            if not date_data.empty:
                stocks_on_date[ticker] = date_data.iloc[0].to_dict()

        if len(stocks_on_date) < 10:  # Need minimum stocks
            return {}

        # Features to rank
        features_to_rank = {
            'return_1d': 'return_1d_percentile',
            'return_5d': 'return_5d_percentile',
            'return_20d': 'return_20d_percentile',
            'volume': 'volume_percentile',
            'dollar_volume': 'dollar_volume_percentile',
            'volatility_20d': 'volatility_percentile',
            'volume_ratio_20d': 'volume_ratio_percentile',
        }

        # Calculate percentiles
        cross_sectional_features = {}

        for ticker in stocks_on_date:
            cross_sectional_features[ticker] = {}

            for feature, percentile_name in features_to_rank.items():
                # Get all values for this feature
                all_values = []
                for t, data in stocks_on_date.items():
                    if feature in data and not pd.isna(data[feature]):
                        all_values.append(data[feature])

                # Get this stock's value
                stock_value = stocks_on_date[ticker].get(feature, np.nan)

                # Calculate percentile
                percentile = self.calculate_percentile_rank(stock_value, all_values)
                cross_sectional_features[ticker][percentile_name] = percentile

        # Sector-relative features
        if self.sector_dict is not None:
            cross_sectional_features = self._add_sector_relative_features(
                cross_sectional_features, stocks_on_date, date
            )

        return cross_sectional_features

    def _add_sector_relative_features(self,
                                     cross_sectional_features: Dict[str, Dict[str, float]],
                                     stocks_on_date: Dict[str, dict],
                                     date: dt_date) -> Dict[str, Dict[str, float]]:
        """
        Add sector-relative features.

        Args:
            cross_sectional_features: Existing features dict
            stocks_on_date: Stock data for this date
            date: Current date

        Returns:
            Updated features dict with sector-relative metrics
        """
        # Group stocks by sector
        sector_groups = {}
        for ticker in stocks_on_date:
            sector = self.sector_dict.get(ticker, 'Unknown')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(ticker)

        # Calculate sector medians
        sector_stats = {}
        for sector, tickers in sector_groups.items():
            sector_stats[sector] = {
                'return_1d_median': np.nanmedian([stocks_on_date[t].get('return_1d', 0) for t in tickers]),
                'return_5d_median': np.nanmedian([stocks_on_date[t].get('return_5d', 0) for t in tickers]),
                'volume_ratio_median': np.nanmedian([stocks_on_date[t].get('volume_ratio_20d', 1) for t in tickers]),
            }

        # Add sector-relative features
        for ticker in cross_sectional_features:
            sector = self.sector_dict.get(ticker, 'Unknown')

            if sector in sector_stats:
                stock_data = stocks_on_date[ticker]

                # Relative to sector median
                cross_sectional_features[ticker]['return_1d_vs_sector'] = (
                    stock_data.get('return_1d', 0) - sector_stats[sector]['return_1d_median']
                )
                cross_sectional_features[ticker]['return_5d_vs_sector'] = (
                    stock_data.get('return_5d', 0) - sector_stats[sector]['return_5d_median']
                )
                cross_sectional_features[ticker]['volume_ratio_vs_sector'] = (
                    stock_data.get('volume_ratio_20d', 1) - sector_stats[sector]['volume_ratio_median']
                )

        return cross_sectional_features

    def calculate_momentum_rankings(self, all_stocks_data: Dict[str, pd.DataFrame],
                                   date: dt_date, lookback_periods: List[int] = [5, 10, 20, 60]) -> Dict[str, Dict[str, float]]:
        """
        Calculate momentum rankings (strength of recent returns).

        Args:
            all_stocks_data: Dict of {ticker: DataFrame}
            date: Date to calculate for
            lookback_periods: Periods for momentum calculation

        Returns:
            Dict of {ticker: {momentum_feature: value}}
        """
        momentum_features = {}

        # Collect returns for each lookback period
        for period in lookback_periods:
            returns_dict = {}

            for ticker, df in all_stocks_data.items():
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date

                # Get data for this date
                date_idx = df[df['date'] == date].index

                if len(date_idx) > 0 and date_idx[0] >= period:
                    # Calculate return over lookback period
                    current_idx = date_idx[0]
                    if current_idx >= period and 'close' in df.columns:
                        current_price = df.loc[current_idx, 'close']
                        past_price = df.loc[current_idx - period, 'close']

                        if past_price > 0:
                            period_return = (current_price - past_price) / past_price
                            returns_dict[ticker] = period_return

            # Rank stocks by this period's return
            if len(returns_dict) > 0:
                sorted_stocks = sorted(returns_dict.items(), key=lambda x: x[1], reverse=True)

                for rank, (ticker, return_val) in enumerate(sorted_stocks):
                    if ticker not in momentum_features:
                        momentum_features[ticker] = {}

                    # Percentile rank (higher is better)
                    percentile = (1 - rank / len(sorted_stocks)) * 100
                    momentum_features[ticker][f'momentum_{period}d_rank'] = percentile

        return momentum_features

    def add_decile_ranks(self, features: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Convert percentiles to decile ranks (0-9).

        Args:
            features: Features dict with percentiles

        Returns:
            Updated features dict with decile ranks
        """
        for ticker in features:
            for feature_name, value in list(features[ticker].items()):
                if 'percentile' in feature_name and not np.isnan(value):
                    decile_name = feature_name.replace('percentile', 'decile')
                    features[ticker][decile_name] = int(value / 10)  # 0-9

        return features


def create_cross_sectional_features_dataset(all_stocks_data: Dict[str, pd.DataFrame],
                                           dates: List[dt_date],
                                           sector_dict: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Create cross-sectional features for all stocks and all dates.

    Args:
        all_stocks_data: Dict of {ticker: DataFrame with OHLCV and derived features}
        dates: List of dates to process
        sector_dict: Optional sector mapping

    Returns:
        Dict of {ticker: DataFrame with cross-sectional features}
    """
    calculator = CrossSectionalCalculator(sector_dict)

    print(f"\n{'='*80}")
    print(f"CALCULATING CROSS-SECTIONAL FEATURES")
    print(f"{'='*80}")
    print(f"Stocks: {len(all_stocks_data)}")
    print(f"Dates: {len(dates)}")
    print(f"{'='*80}\n")

    # Results storage
    ticker_features = {ticker: [] for ticker in all_stocks_data}

    # Process each date
    from tqdm import tqdm
    for date in tqdm(dates, desc="Processing dates"):
        # Calculate cross-sectional features
        date_features = calculator.calculate_cross_sectional_features_for_date(
            all_stocks_data, date
        )

        # Calculate momentum rankings
        momentum_features = calculator.calculate_momentum_rankings(
            all_stocks_data, date
        )

        # Merge features
        for ticker in all_stocks_data:
            combined_features = {'date': date}

            if ticker in date_features:
                combined_features.update(date_features[ticker])

            if ticker in momentum_features:
                combined_features.update(momentum_features[ticker])

            ticker_features[ticker].append(combined_features)

    # Convert to DataFrames
    result = {}
    for ticker, features_list in ticker_features.items():
        if len(features_list) > 0:
            result[ticker] = pd.DataFrame(features_list)

    print(f"\n✅ Cross-sectional features calculated!")
    print(f"Features per stock: ~{len(result[list(result.keys())[0]].columns) if result else 0}")

    return result


if __name__ == '__main__':
    print("Cross-Sectional Features Calculator")
    print("="*80)
    print("\nFeatures calculated:")
    print("  - Return percentiles (1d, 5d, 20d)")
    print("  - Volume percentiles")
    print("  - Volatility percentiles")
    print("  - Momentum rankings (5d, 10d, 20d, 60d)")
    print("  - Sector-relative metrics (if sector dict provided)")
    print("  - Decile ranks (0-9)")
    print(f"\nTotal: ~15-20 cross-sectional features per stock per day")
