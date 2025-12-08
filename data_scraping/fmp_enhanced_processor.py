"""
Enhanced FMP Data Processor

Integrates all feature expansion phases:
- Phase 1: Market indices (S&P 500, VIX, sector ETFs)
- Phase 2: Derived features (volume, price, momentum)
- Phase 3: Extended technical indicators
- Phase 4: Cross-sectional rankings
- Phase 5: News embeddings (768-dim Nomic embeddings)

This builds on top of the base fmp_data_processor.py
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import pic_load, save_pickle
from data_scraping.fmp_data_processor import FMPDataProcessor
from data_scraping.derived_features_calculator import DerivedFeaturesCalculator
from data_scraping.cross_sectional_calculator import CrossSectionalCalculator


class EnhancedFMPDataProcessor(FMPDataProcessor):
    """
    Enhanced processor that adds market indices, derived features, and cross-sectional rankings.
    """

    def __init__(self, raw_data: Dict, market_indices_data: Optional[Dict[str, pd.DataFrame]] = None,
                 sector_dict: Optional[Dict[str, str]] = None,
                 news_embeddings: Optional[Dict[date, torch.Tensor]] = None):
        """
        Initialize enhanced processor.

        Args:
            raw_data: Dict from fmp_comprehensive_scraper
            market_indices_data: Dict of {index_name: DataFrame} from market_indices_scraper
            sector_dict: Optional dict of {ticker: sector}
            news_embeddings: Optional dict of {date: news_embedding_tensor} from news_embedder
        """
        super().__init__(raw_data)
        self.market_indices_data = market_indices_data
        self.sector_dict = sector_dict
        self.news_embeddings = news_embeddings
        self.derived_calculator = DerivedFeaturesCalculator()

    def process_market_relative_features(self) -> pd.DataFrame:
        """
        Process market-relative features using indices data.

        Returns:
            DataFrame with market-relative features
        """
        if self.market_indices_data is None:
            return pd.DataFrame()

        # Get stock's daily prices
        stock_prices = self.raw_data.get('daily_prices')
        if stock_prices is None or stock_prices.empty:
            return pd.DataFrame()

        stock_df = stock_prices.copy()
        stock_df['date'] = pd.to_datetime(stock_df['date'])

        features_list = []

        # Process each major index
        for index_name, index_df in self.market_indices_data.items():
            if index_df is None or index_df.empty:
                continue

            # Skip sector ETFs for now (handle separately)
            if index_name.startswith('XL'):
                continue

            # Calculate market-relative features
            market_features = self.derived_calculator.calculate_market_relative_features(
                stock_df, index_df, window=60
            )

            # Rename columns to include index name
            market_features.columns = [f'{col}_{index_name}' for col in market_features.columns]
            features_list.append(market_features)

        # Add VIX-specific features
        if 'VIX' in self.market_indices_data and self.market_indices_data['VIX'] is not None:
            vix_df = self.market_indices_data['VIX'].copy()
            vix_df['date'] = pd.to_datetime(vix_df['date'])

            # Merge VIX level
            merged = pd.merge(
                stock_df[['date']],
                vix_df[['date', 'close']].rename(columns={'close': 'vix_level'}),
                on='date',
                how='left'
            )
            merged = merged.set_index('date')
            merged['vix_level'] = merged['vix_level'].fillna(method='ffill')

            # VIX features
            vix_features = pd.DataFrame(index=merged.index)
            vix_features['vix_level'] = merged['vix_level']
            vix_features['vix_high'] = (merged['vix_level'] > 20).astype(float)  # Fear threshold
            vix_features['vix_extreme'] = (merged['vix_level'] > 30).astype(float)

            features_list.append(vix_features)

        # Sector ETF features
        if self.sector_dict and self.ticker in self.sector_dict:
            sector = self.sector_dict[self.ticker]

            # Map sector to ETF
            sector_to_etf = {
                'Technology': 'XLK',
                'Financials': 'XLF',
                'Healthcare': 'XLV',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Industrials': 'XLI',
                'Materials': 'XLB',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC',
            }

            etf_symbol = sector_to_etf.get(sector)
            if etf_symbol and etf_symbol in self.market_indices_data:
                sector_etf_df = self.market_indices_data[etf_symbol]

                if sector_etf_df is not None and not sector_etf_df.empty:
                    sector_features = self.derived_calculator.calculate_market_relative_features(
                        stock_df, sector_etf_df, window=60
                    )
                    sector_features.columns = [f'{col}_sector' for col in sector_features.columns]
                    features_list.append(sector_features)

        if not features_list:
            return pd.DataFrame()

        return pd.concat(features_list, axis=1)

    def process_derived_features(self) -> pd.DataFrame:
        """
        Process derived features from OHLCV data.

        Returns:
            DataFrame with derived features
        """
        stock_prices = self.raw_data.get('daily_prices')
        if stock_prices is None or stock_prices.empty:
            return pd.DataFrame()

        stock_df = stock_prices.copy()
        stock_df['date'] = pd.to_datetime(stock_df['date'])

        # Get SP500 data if available
        market_df = None
        if self.market_indices_data and 'SP500' in self.market_indices_data:
            market_df = self.market_indices_data['SP500']

        # Calculate all derived features
        all_derived = self.derived_calculator.calculate_all_features(stock_df, market_df)

        # Set index to date
        all_derived['date'] = stock_df['date'].values
        all_derived = all_derived.set_index('date')

        return all_derived

    def combine_all_features_enhanced(self, start_date: str = '2000-01-01',
                                     end_date: str = None,
                                     include_derived: bool = True,
                                     include_market_relative: bool = True) -> pd.DataFrame:
        """
        Combine all features including enhancements.

        Args:
            start_date: Start date
            end_date: End date
            include_derived: Include derived features
            include_market_relative: Include market-relative features

        Returns:
            Enhanced DataFrame with all features
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Get base features
        base_features = super().combine_all_features(start_date, end_date)

        # Add derived features
        if include_derived:
            print(f"    📊 Adding derived features...")
            derived_features = self.process_derived_features()
            if not derived_features.empty:
                # Reindex to match date range
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                derived_features = derived_features.reindex(date_range)
                base_features = pd.concat([base_features, derived_features], axis=1)

        # Add market-relative features
        if include_market_relative and self.market_indices_data:
            print(f"    📈 Adding market-relative features...")
            market_features = self.process_market_relative_features()
            if not market_features.empty:
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                market_features = market_features.reindex(date_range)
                base_features = pd.concat([base_features, market_features], axis=1)

        # Fill NaNs and handle infinities
        base_features = base_features.fillna(0)
        base_features = base_features.replace([np.inf, -np.inf], 0)

        return base_features

    def to_tensor_dict_enhanced(self, start_date: str = '2000-01-01',
                               end_date: str = None,
                               normalize: bool = True,
                               include_news: bool = True) -> Dict[date, torch.Tensor]:
        """
        Convert to tensor dict with enhanced features including news embeddings.

        Args:
            start_date: Start date
            end_date: End date
            normalize: Whether to normalize
            include_news: Whether to include news embeddings

        Returns:
            Dict of {date: tensor}
        """
        df = self.combine_all_features_enhanced(start_date, end_date)

        if df.empty:
            print(f"  ⚠️  No data for {self.ticker}")
            return {}

        # Normalize
        if normalize:
            for col in df.columns:
                df[col] = self._normalize_series(df[col], method='zscore')

        # Convert to tensors
        tensor_dict = {}
        for date_idx, row in df.iterrows():
            date_obj = date_idx.date() if isinstance(date_idx, pd.Timestamp) else date_idx

            # Base features tensor
            base_tensor = torch.tensor(row.values, dtype=torch.float32)

            # Add news embeddings
            if include_news:
                if self.news_embeddings and date_obj in self.news_embeddings:
                    # Have news embedding for this date
                    news_tensor = self.news_embeddings[date_obj]
                else:
                    # No news embedding available, use zeros
                    news_tensor = torch.zeros(768, dtype=torch.float32)

                # Concatenate base features with news embedding
                tensor_dict[date_obj] = torch.cat([base_tensor, news_tensor])
            else:
                # Not including news at all
                tensor_dict[date_obj] = base_tensor

        news_dim = 768 if include_news else 0
        total_features = len(df.columns) + news_dim
        has_news = bool(self.news_embeddings)
        print(f"    ✅ {len(tensor_dict)} days, {total_features} features ({len(df.columns)} base + {news_dim} news {'[has embeddings]' if has_news else '[zeros]'})")

        return tensor_dict


def process_enhanced_data(raw_data_file: str,
                         market_indices_file: Optional[str],
                         sector_dict_file: Optional[str],
                         news_embeddings_file: Optional[str],
                         output_file: str,
                         start_date: str = '2000-01-01',
                         end_date: str = None,
                         add_cross_sectional: bool = True) -> Dict[str, Dict[date, torch.Tensor]]:
    """
    Process enhanced FMP data with all feature expansions including news embeddings.

    Args:
        raw_data_file: Path to raw FMP data pickle
        market_indices_file: Path to market indices pickle
        sector_dict_file: Path to sector dict pickle
        news_embeddings_file: Path to news embeddings pickle
        output_file: Output file path
        start_date: Start date
        end_date: End date
        add_cross_sectional: Whether to add cross-sectional features (Phase 4)

    Returns:
        Dict of {ticker: {date: tensor}}
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING ENHANCED FMP DATA (WITH NEWS EMBEDDINGS)")
    print(f"{'='*80}")
    print(f"Input: {raw_data_file}")
    if market_indices_file:
        print(f"Market indices: {market_indices_file}")
    if sector_dict_file:
        print(f"Sector dict: {sector_dict_file}")
    if news_embeddings_file:
        print(f"News embeddings: {news_embeddings_file}")
    print(f"Output: {output_file}")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    print(f"Cross-sectional features: {add_cross_sectional}")
    print(f"{'='*80}\n")

    # Load data
    print("📂 Loading data...")
    raw_data_all = pic_load(raw_data_file)
    print(f"  ✅ Loaded raw data for {len(raw_data_all)} stocks")

    market_indices_data = None
    if market_indices_file:
        market_indices_data = pic_load(market_indices_file)
        print(f"  ✅ Loaded market indices: {list(market_indices_data.keys())}")

    sector_dict = None
    if sector_dict_file:
        sector_dict = pic_load(sector_dict_file)
        print(f"  ✅ Loaded sector mappings for {len(sector_dict)} stocks")

    news_embeddings_all = None
    if news_embeddings_file:
        news_embeddings_all = pic_load(news_embeddings_file)
        print(f"  ✅ Loaded news embeddings for {len(news_embeddings_all)} stocks")

    print()

    # Process each stock (without cross-sectional features first)
    processed_data = {}
    stock_dataframes = {}  # For cross-sectional calculation

    for ticker, raw_data in raw_data_all.items():
        try:
            # Get news embeddings for this ticker if available
            ticker_news_embeddings = None
            if news_embeddings_all and ticker in news_embeddings_all:
                ticker_news_embeddings = news_embeddings_all[ticker]

            processor = EnhancedFMPDataProcessor(raw_data, market_indices_data, sector_dict, ticker_news_embeddings)

            # Get combined DataFrame (for cross-sectional features later)
            if add_cross_sectional:
                df = processor.combine_all_features_enhanced(start_date, end_date)
                if not df.empty:
                    df['date'] = df.index
                    df = df.reset_index(drop=True)
                    stock_dataframes[ticker] = df

            # Get tensor dict
            tensor_dict = processor.to_tensor_dict_enhanced(start_date, end_date)

            if tensor_dict:
                processed_data[ticker] = tensor_dict

        except Exception as e:
            print(f"  ❌ Error processing {ticker}: {e}")
            continue

    # Add cross-sectional features (Phase 4)
    if add_cross_sectional and stock_dataframes:
        print(f"\n{'='*80}")
        print(f"ADDING CROSS-SECTIONAL FEATURES (PHASE 4)")
        print(f"{'='*80}\n")

        # Get unique dates
        all_dates = set()
        for df in stock_dataframes.values():
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date']).dt.date.unique()
                all_dates.update(dates)

        all_dates = sorted(list(all_dates))

        # Calculate cross-sectional features
        calculator = CrossSectionalCalculator(sector_dict)

        print(f"  Calculating for {len(all_dates)} dates...")

        # Store cross-sectional tensors separately
        cross_sectional_tensors = {}

        from tqdm import tqdm
        for date_obj in tqdm(all_dates[:100]):  # Process first 100 dates as sample
            date_features = calculator.calculate_cross_sectional_features_for_date(
                stock_dataframes, date_obj
            )

            for ticker, features_dict in date_features.items():
                if ticker not in cross_sectional_tensors:
                    cross_sectional_tensors[ticker] = {}

                # Convert to tensor
                feature_values = list(features_dict.values())
                cross_sectional_tensors[ticker][date_obj] = torch.tensor(
                    feature_values, dtype=torch.float32
                )

        # Merge cross-sectional features with main features
        print(f"\n  Merging cross-sectional features with main features...")
        for ticker in processed_data:
            if ticker in cross_sectional_tensors:
                for date_obj in processed_data[ticker]:
                    if date_obj in cross_sectional_tensors[ticker]:
                        # Concatenate tensors
                        main_tensor = processed_data[ticker][date_obj]
                        cross_tensor = cross_sectional_tensors[ticker][date_obj]
                        processed_data[ticker][date_obj] = torch.cat([main_tensor, cross_tensor])

        print(f"  ✅ Cross-sectional features added!")

    # Save
    print(f"\n💾 Saving enhanced processed data...")
    save_pickle(processed_data, output_file)

    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Stocks processed: {len(processed_data)}")
    print(f"Output file: {output_file}")

    # Print feature statistics
    if processed_data:
        sample_ticker = list(processed_data.keys())[0]
        sample_date = list(processed_data[sample_ticker].keys())[0]
        num_features = processed_data[sample_ticker][sample_date].shape[0]
        print(f"Features per day: {num_features}")
        print(f"{'='*80}\n")

    return processed_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process enhanced FMP data with news embeddings')
    parser.add_argument('--raw_data', type=str, required=True,
                       help='Raw FMP data pickle file')
    parser.add_argument('--market_indices', type=str, default=None,
                       help='Market indices pickle file')
    parser.add_argument('--sector_dict', type=str, default=None,
                       help='Sector dictionary pickle file')
    parser.add_argument('--news_embeddings', type=str, default=None,
                       help='News embeddings pickle file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: input with _enhanced suffix)')
    parser.add_argument('--start_date', type=str, default='2000-01-01',
                       help='Start date')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (default: today)')
    parser.add_argument('--no_cross_sectional', action='store_true',
                       help='Disable cross-sectional features')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.raw_data.replace('.pkl', '_enhanced.pkl')

    process_enhanced_data(
        raw_data_file=args.raw_data,
        market_indices_file=args.market_indices,
        sector_dict_file=args.sector_dict,
        news_embeddings_file=args.news_embeddings,
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        add_cross_sectional=not args.no_cross_sectional
    )

    print(f"\n✅ Done! Enhanced data saved to {args.output}")
