"""
FMP Data Processor

Converts comprehensive FMP data into time-series tensors for the transformer model.

This module takes the raw data collected by fmp_comprehensive_scraper.py and:
1. Aligns all data sources to daily frequency
2. Forward-fills quarterly/irregular data
3. Normalizes features
4. Creates time-varying feature tensors

Output format:
{
    'AAPL': {
        datetime.date(2020, 1, 1): tensor([...]),  # N-dimensional feature vector
        datetime.date(2020, 1, 2): tensor([...]),
        ...
    },
    ...
}
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import pic_load, save_pickle


class FMPDataProcessor:
    """
    Processor for converting raw FMP data into time-series tensors.
    """

    def __init__(self, raw_data: Dict):
        """
        Initialize processor.

        Args:
            raw_data: Dict from fmp_comprehensive_scraper.scrape_all_data()
        """
        self.raw_data = raw_data
        self.ticker = raw_data['ticker']

    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float."""
        if value is None or pd.isna(value) or np.isinf(value):
            return default
        try:
            return float(value)
        except:
            return default

    def _normalize_series(self, series: pd.Series, method='zscore') -> pd.Series:
        """
        Normalize a series.

        Args:
            series: Pandas series
            method: 'zscore', 'minmax', or 'log'
        """
        if method == 'zscore':
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                return series * 0
            return (series - mean) / std

        elif method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            if min_val == max_val or pd.isna(min_val) or pd.isna(max_val):
                return series * 0
            return (series - min_val) / (max_val - min_val)

        elif method == 'log':
            return np.log1p(series.clip(lower=0))

        return series

    def process_financial_statements(self) -> pd.DataFrame:
        """
        Process financial statements into quarterly features.

        Returns:
            DataFrame with date index and financial features
        """
        features = []

        statements = self.raw_data.get('financial_statements', {})
        if not statements:
            return pd.DataFrame()

        # Process each statement type
        for stmt_name, df in statements.items():
            if df is None or df.empty:
                continue

            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.set_index('date')

            # Select key columns (exclude metadata)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[f'{stmt_name}_{col}'] = df[col]

            features.append(df[[f'{stmt_name}_{col}' for col in numeric_cols]])

        if not features:
            return pd.DataFrame()

        # Combine all features
        combined = pd.concat(features, axis=1)
        return combined

    def process_key_metrics(self) -> pd.DataFrame:
        """Process key metrics into quarterly features."""
        df = self.raw_data.get('key_metrics')
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols]

    def process_financial_ratios(self) -> pd.DataFrame:
        """Process financial ratios into quarterly features."""
        df = self.raw_data.get('financial_ratios')
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols]

    def process_daily_prices(self) -> pd.DataFrame:
        """Process daily price data."""
        df = self.raw_data.get('daily_prices')
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')

        # Core OHLCV + computed features
        features = {}
        if 'close' in df.columns:
            features['price'] = df['close']
            features['price_change'] = df['close'].pct_change()

        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_change'] = df['volume'].pct_change()

        if all(col in df.columns for col in ['high', 'low', 'close']):
            features['daily_range'] = (df['high'] - df['low']) / df['close']

        if all(col in df.columns for col in ['open', 'close']):
            features['intraday_return'] = (df['close'] - df['open']) / df['open']

        return pd.DataFrame(features)

    def process_technical_indicators(self) -> pd.DataFrame:
        """Process technical indicators."""
        indicators = self.raw_data.get('technical_indicators', {})
        if not indicators:
            return pd.DataFrame()

        dfs = []
        for name, df in indicators.items():
            if df is None or df.empty:
                continue

            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.set_index('date')

            # Most technical indicators have the indicator value in a column matching the type
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[f'tech_{name}_{col}'] = df[col]

            dfs.append(df[[f'tech_{name}_{col}' for col in numeric_cols]])

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=1)

    def process_earnings(self) -> pd.DataFrame:
        """Process earnings data."""
        features = []

        # Earnings history
        df_earnings = self.raw_data.get('earnings_history')
        if df_earnings is not None and not df_earnings.empty:
            df = df_earnings.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.set_index('date')

            if 'eps' in df.columns:
                features.append(pd.DataFrame({'earnings_eps': df['eps']}))
            if 'epsEstimated' in df.columns:
                features.append(pd.DataFrame({'earnings_eps_est': df['epsEstimated']}))

        # Earnings surprises
        df_surprises = self.raw_data.get('earnings_surprises')
        if df_surprises is not None and not df_surprises.empty:
            df = df_surprises.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df = df.set_index('date')

                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    features.append(pd.DataFrame({f'earnings_surprise_{col}': df[col]}))

        if not features:
            return pd.DataFrame()

        return pd.concat(features, axis=1)

    def process_insider_trading(self) -> pd.DataFrame:
        """Process insider trading into daily aggregated features."""
        df = self.raw_data.get('insider_trading')
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        if 'transactionDate' not in df.columns:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['transactionDate'])
        df = df.sort_values('date')

        # Aggregate by date
        daily_features = df.groupby('date').agg({
            'securitiesTransacted': 'sum',  # Total shares traded
            'price': 'mean',  # Average price
        })

        # Add buy/sell indicators
        if 'acquistionOrDisposition' in df.columns:
            df['is_buy'] = (df['acquistionOrDisposition'] == 'A').astype(int)
            df['is_sell'] = (df['acquistionOrDisposition'] == 'D').astype(int)

            daily_buy_sell = df.groupby('date').agg({
                'is_buy': 'sum',
                'is_sell': 'sum'
            })
            daily_features = pd.concat([daily_features, daily_buy_sell], axis=1)

        return daily_features

    def process_analyst_ratings(self) -> pd.DataFrame:
        """Process analyst ratings."""
        features = []

        # Ratings
        df_ratings = self.raw_data.get('analyst_ratings')
        if df_ratings is not None and not df_ratings.empty:
            df = df_ratings.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df = df.set_index('date')

                # Encode rating as numeric
                if 'rating' in df.columns:
                    rating_map = {
                        'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1
                    }
                    df['rating_numeric'] = df['rating'].map(rating_map).fillna(3)
                    features.append(pd.DataFrame({'analyst_rating': df['rating_numeric']}))

        # Price targets
        df_targets = self.raw_data.get('price_targets')
        if df_targets is not None and not df_targets.empty:
            df = df_targets.copy()
            if 'publishedDate' in df.columns:
                df['date'] = pd.to_datetime(df['publishedDate'])
                df = df.sort_values('date')
                df = df.set_index('date')

                if 'priceTarget' in df.columns:
                    features.append(pd.DataFrame({'price_target': df['priceTarget']}))

        if not features:
            return pd.DataFrame()

        return pd.concat(features, axis=1)

    def combine_all_features(self, start_date: str = '2000-01-01',
                           end_date: str = None) -> pd.DataFrame:
        """
        Combine all feature sources into a single daily DataFrame.

        Args:
            start_date: Start date
            end_date: End date (defaults to today)

        Returns:
            DataFrame with daily features (forward-filled for quarterly data)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"  Processing {self.ticker}...")

        # Process all data sources
        financial_stmt = self.process_financial_statements()
        key_metrics = self.process_key_metrics()
        ratios = self.process_financial_ratios()
        daily_prices = self.process_daily_prices()
        technical = self.process_technical_indicators()
        earnings = self.process_earnings()
        insider = self.process_insider_trading()
        analyst = self.process_analyst_ratings()

        # Combine quarterly data (will be forward-filled)
        quarterly_data = [financial_stmt, key_metrics, ratios, earnings]
        quarterly_combined = pd.concat([df for df in quarterly_data if not df.empty], axis=1)

        # Combine daily data
        daily_data = [daily_prices, technical, insider, analyst]
        daily_combined = pd.concat([df for df in daily_data if not df.empty], axis=1)

        # Create daily date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex and forward-fill quarterly data to daily
        if not quarterly_combined.empty:
            quarterly_combined = quarterly_combined.reindex(date_range, method='ffill')

        # Reindex daily data
        if not daily_combined.empty:
            daily_combined = daily_combined.reindex(date_range)

        # Combine all
        if not quarterly_combined.empty and not daily_combined.empty:
            combined = pd.concat([quarterly_combined, daily_combined], axis=1)
        elif not quarterly_combined.empty:
            combined = quarterly_combined
        elif not daily_combined.empty:
            combined = daily_combined
        else:
            return pd.DataFrame()

        # Fill NaNs with 0
        combined = combined.fillna(0)

        # Handle infinities
        combined = combined.replace([np.inf, -np.inf], 0)

        return combined

    def to_tensor_dict(self, start_date: str = '2000-01-01',
                      end_date: str = None,
                      normalize: bool = True) -> Dict[date, torch.Tensor]:
        """
        Convert all features to dict of {date: tensor}.

        Args:
            start_date: Start date
            end_date: End date
            normalize: Whether to z-score normalize features

        Returns:
            Dict of {datetime.date: torch.Tensor}
        """
        df = self.combine_all_features(start_date, end_date)

        if df.empty:
            print(f"  ⚠️  No data for {self.ticker}")
            return {}

        # Normalize each column
        if normalize:
            for col in df.columns:
                df[col] = self._normalize_series(df[col], method='zscore')

        # Convert to dict of tensors
        tensor_dict = {}
        for date_idx, row in df.iterrows():
            date_obj = date_idx.date() if isinstance(date_idx, pd.Timestamp) else date_idx
            tensor_dict[date_obj] = torch.tensor(row.values, dtype=torch.float32)

        print(f"    ✅ {len(tensor_dict)} days, {len(df.columns)} features")

        return tensor_dict


def process_comprehensive_data(raw_data_file: str,
                               output_file: str,
                               start_date: str = '2000-01-01',
                               end_date: str = None) -> Dict[str, Dict[date, torch.Tensor]]:
    """
    Process comprehensive FMP data for all stocks.

    Args:
        raw_data_file: Path to pickle file from fmp_comprehensive_scraper
        output_file: Path to output pickle file
        start_date: Start date
        end_date: End date

    Returns:
        Dict of {ticker: {date: tensor}}
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING FMP COMPREHENSIVE DATA")
    print(f"{'='*80}")
    print(f"Input: {raw_data_file}")
    print(f"Output: {output_file}")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    print(f"{'='*80}\n")

    # Load raw data
    print("📂 Loading raw data...")
    raw_data_all = pic_load(raw_data_file)
    print(f"  ✅ Loaded data for {len(raw_data_all)} stocks\n")

    # Process each stock
    processed_data = {}
    for ticker, raw_data in raw_data_all.items():
        try:
            processor = FMPDataProcessor(raw_data)
            tensor_dict = processor.to_tensor_dict(start_date, end_date)

            if tensor_dict:
                processed_data[ticker] = tensor_dict

        except Exception as e:
            print(f"  ❌ Error processing {ticker}: {e}")
            continue

    # Save
    print(f"\n💾 Saving processed data...")
    save_pickle(processed_data, output_file)

    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Stocks processed: {len(processed_data)}")
    print(f"Output file: {output_file}")

    # Print feature count statistics
    if processed_data:
        sample_ticker = list(processed_data.keys())[0]
        sample_date = list(processed_data[sample_ticker].keys())[0]
        num_features = processed_data[sample_ticker][sample_date].shape[0]
        print(f"Features per day: {num_features}")

    return processed_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process FMP comprehensive data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input pickle file from fmp_comprehensive_scraper')
    parser.add_argument('--output', type=str, default=None,
                       help='Output pickle file (default: input with _processed suffix)')
    parser.add_argument('--start_date', type=str, default='2000-01-01',
                       help='Start date')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (default: today)')

    args = parser.parse_args()

    # Set output file
    if args.output is None:
        args.output = args.input.replace('.pkl', '_processed.pkl')

    # Process
    process_comprehensive_data(
        raw_data_file=args.input,
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date
    )

    print(f"\n✅ Done! Processed data saved to {args.output}")
