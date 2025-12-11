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
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import pic_load, save_pickle
from data_scraping.fmp_data_processor import FMPDataProcessor
from data_scraping.derived_features_calculator import DerivedFeaturesCalculator
from data_scraping.cross_sectional_calculator import CrossSectionalCalculator
from data_scraping.cross_sectional_normalizer import CrossSectionalNormalizer


class EnhancedFMPDataProcessor(FMPDataProcessor):
    """
    Enhanced processor that adds market indices, derived features, and cross-sectional rankings.
    """

    def __init__(self, raw_data: Dict, market_indices_data: Optional[Dict[str, pd.DataFrame]] = None,
                 sector_dict: Optional[Dict[str, str]] = None,
                 news_embeddings: Optional[Dict[date, torch.Tensor]] = None,
                 price_data: Optional[pd.DataFrame] = None):
        """
        Initialize enhanced processor.

        Args:
            raw_data: Dict from fmp_comprehensive_scraper
            market_indices_data: Dict of {index_name: DataFrame} from market_indices_scraper
            sector_dict: Optional dict of {ticker: sector}
            news_embeddings: Optional dict of {date: news_embedding_tensor} from news_embedder
            price_data: Optional DataFrame with daily price data from yfinance
        """
        super().__init__(raw_data, price_data=price_data)
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
            vix_features['vix_high'] = merged['vix_level'].gt(20).fillna(False).astype(float)  # Fear threshold
            vix_features['vix_extreme'] = merged['vix_level'].gt(30).fillna(False).astype(float)

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
        try:
            print(f"    📦 Getting base features...")
            base_features = super().combine_all_features(start_date, end_date)
        except Exception as e:
            print(f"    ❌ Error in base features: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Add derived features
        if include_derived:
            try:
                print(f"    📊 Adding derived features...")
                derived_features = self.process_derived_features()
                if not derived_features.empty:
                    # Reindex to match date range
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    derived_features = derived_features.reindex(date_range)
                    base_features = pd.concat([base_features, derived_features], axis=1)
            except Exception as e:
                print(f"    ❌ Error in derived features: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Add market-relative features
        if include_market_relative and self.market_indices_data:
            try:
                print(f"    📈 Adding market-relative features...")
                market_features = self.process_market_relative_features()
                if not market_features.empty:
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    market_features = market_features.reindex(date_range)
                    base_features = pd.concat([base_features, market_features], axis=1)
            except Exception as e:
                print(f"    ❌ Error in market-relative features: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Fill NaNs and handle infinities
        print(f"    🧹 Cleaning data...")
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
        try:
            df = self.combine_all_features_enhanced(start_date, end_date)
        except Exception as e:
            print(f"    ❌ Error in combine_all_features_enhanced: {e}")
            import traceback
            traceback.print_exc()
            raise

        if df.empty:
            print(f"  ⚠️  No data for {self.ticker}")
            return {}

        # Normalize
        if normalize:
            try:
                print(f"    🔢 Normalizing {len(df.columns)} features...")
                for i, col in enumerate(df.columns):
                    if i % 50 == 0 and i > 0:
                        print(f"      Normalized {i}/{len(df.columns)} features...")
                    df[col] = self._normalize_series(df[col], method='zscore')
            except Exception as e:
                print(f"    ❌ Error normalizing column '{col}': {e}")
                import traceback
                traceback.print_exc()
                raise

        # Convert to tensors
        try:
            print(f"    🔄 Converting to tensors...")
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
        except Exception as e:
            print(f"    ❌ Error converting to tensors: {e}")
            import traceback
            traceback.print_exc()
            raise

        news_dim = 768 if include_news else 0
        total_features = len(df.columns) + news_dim
        has_news = bool(self.news_embeddings)
        print(f"    ✅ {len(tensor_dict)} days, {total_features} features ({len(df.columns)} base + {news_dim} news {'[has embeddings]' if has_news else '[zeros]'})")

        return tensor_dict


def process_enhanced_data(raw_data_file: str,
                         market_indices_file: Optional[str],
                         sector_dict_file: Optional[str],
                         news_embeddings_file: Optional[str],
                         price_data_file: Optional[str],
                         output_file: str,
                         start_date: str = '2000-01-01',
                         end_date: str = None,
                         add_cross_sectional: bool = True,
                         use_cross_sectional_normalization: bool = True,
                         intermediate_output_file: Optional[str] = None) -> Dict[str, Dict[date, torch.Tensor]]:
    """
    Process enhanced FMP data with all feature expansions including news embeddings and price data.

    NEW: Now uses proper cross-sectional normalization for fundamental metrics!

    Args:
        raw_data_file: Path to raw FMP data pickle
        market_indices_file: Path to market indices pickle
        sector_dict_file: Path to sector dict pickle
        news_embeddings_file: Path to news embeddings pickle
        price_data_file: Path to price data pickle (yfinance)
        output_file: Output file path
        start_date: Start date
        end_date: End date
        add_cross_sectional: Whether to add cross-sectional features (Phase 4)
        use_cross_sectional_normalization: Whether to use cross-sectional normalization for fundamentals (recommended!)
        intermediate_output_file: Optional path to save/load intermediate DataFrames (after concatenation, before normalization)

    Returns:
        Dict of {ticker: {date: tensor}}
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING ENHANCED FMP DATA (WITH CROSS-SECTIONAL NORMALIZATION)")
    print(f"{'='*80}")
    print(f"Input: {raw_data_file}")
    if market_indices_file:
        print(f"Market indices: {market_indices_file}")
    if sector_dict_file:
        print(f"Sector dict: {sector_dict_file}")
    if news_embeddings_file:
        print(f"News embeddings: {news_embeddings_file}")
    if price_data_file:
        print(f"Price data: {price_data_file}")
    print(f"Output: {output_file}")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    print(f"Cross-sectional features: {add_cross_sectional}")
    print(f"Cross-sectional normalization: {use_cross_sectional_normalization}")
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

    price_data_all = None
    if price_data_file:
        price_data_all = pic_load(price_data_file)
        print(f"  ✅ Loaded price data for {len(price_data_all)} stocks")

    print()

    # Check for existing progress and auto-resume
    import os
    completed_tickers = set()
    if os.path.exists(output_file):
        try:
            existing_data = pic_load(output_file)
            completed_tickers = set(existing_data.keys())
            print(f"📂 Found existing progress: {len(completed_tickers)}/{len(raw_data_all)} stocks completed")
            if len(completed_tickers) > 0:
                last_ticker = list(completed_tickers)[-1]
                print(f"   Last completed: {last_ticker}")
                print(f"   Resuming from next stock...\n")
            del existing_data  # Free memory immediately
        except Exception as e:
            print(f"⚠️  Could not load existing file: {e}")
            print(f"   Starting fresh...\n")
            completed_tickers = set()

    # Cross-sectional percentile features
    # (These are different from cross-sectional normalization!)
    if add_cross_sectional:
        print("✅ Cross-sectional percentile features: ENABLED")
        print("   (Note: This is different from cross-sectional normalization)\n")
    else:
        print("⚠️  Cross-sectional percentile features: DISABLED")
        print("   (Cross-sectional normalization is still enabled for fundamentals)\n")

    import gc
    import psutil

    # ========================================================================
    # TWO-PASS APPROACH FOR CROSS-SECTIONAL NORMALIZATION
    # ========================================================================

    if use_cross_sectional_normalization:
        # Check if intermediate file exists and load it
        all_dataframes = {}
        if intermediate_output_file and os.path.exists(intermediate_output_file):
            print(f"\n{'='*80}")
            print(f"LOADING INTERMEDIATE DATAFRAMES")
            print(f"{'='*80}\n")
            print(f"  📂 Loading from: {intermediate_output_file}")
            all_dataframes = pic_load(intermediate_output_file)
            print(f"  ✅ Loaded {len(all_dataframes)} DataFrames from intermediate file")
            print(f"  ⏭️  Skipping Pass 1 (concatenation)\n")
        else:
            # Run Pass 1: Generate DataFrames
            print(f"\n{'='*80}")
            print(f"PASS 1: GENERATING DATAFRAMES FOR ALL STOCKS")
            print(f"{'='*80}\n")

            # Pass 1: Generate DataFrames (no normalization yet)
            failed_stocks = []

            for i, (ticker, raw_data) in enumerate(raw_data_all.items(), 1):
                # Skip already processed stocks
                if ticker in completed_tickers:
                    print(f"  ⏭️  [{i}/{len(raw_data_all)}] {ticker}: Already processed")
                    continue

                if i % 50 == 0:
                    print(f"\n  Progress: {i}/{len(raw_data_all)} stocks")

                try:
                    # Get news embeddings and price data for this ticker
                    ticker_news_embeddings = None
                    if news_embeddings_all and ticker in news_embeddings_all:
                        ticker_news_embeddings = news_embeddings_all[ticker]

                    ticker_price_data = None
                    if price_data_all and ticker in price_data_all:
                        ticker_price_data = price_data_all[ticker]

                    processor = EnhancedFMPDataProcessor(
                        raw_data, market_indices_data, sector_dict,
                        ticker_news_embeddings, ticker_price_data
                    )

                    # Generate DataFrame WITHOUT normalization
                    df = processor.combine_all_features_enhanced(start_date, end_date)

                    if not df.empty:
                        all_dataframes[ticker] = df
                        print(f"  ✅ [{i}/{len(raw_data_all)}] {ticker}: {len(df)} days, {len(df.columns)} features")
                    else:
                        print(f"  ⚠️  [{i}/{len(raw_data_all)}] {ticker}: No data")
                        failed_stocks.append(ticker)

                except Exception as e:
                    print(f"  ❌ [{i}/{len(raw_data_all)}] {ticker}: {str(e)[:100]}")
                    failed_stocks.append(ticker)
                    continue

                # Memory management
                if i % 100 == 0:
                    gc.collect()
                    process = psutil.Process(os.getpid())
                    mem_gb = process.memory_info().rss / 1024**3
                    print(f"   📊 RAM usage: {mem_gb:.2f} GB\n")

            print(f"\n  ✅ Pass 1 complete: {len(all_dataframes)} stocks with DataFrames")
            print(f"  ⚠️  Failed: {len(failed_stocks)} stocks\n")

            # Clean up raw data - we don't need it anymore
            print(f"  🧹 Cleaning up raw data...")
            del raw_data_all
            gc.collect()

            # Save intermediate DataFrames if requested
            if intermediate_output_file:
                print(f"  💾 Saving intermediate DataFrames to: {intermediate_output_file}")
                save_pickle(all_dataframes, intermediate_output_file)
                print(f"  ✅ Intermediate file saved ({len(all_dataframes)} stocks)\n")

                # If news embeddings not provided, stop here (only concatenation step)
                if not news_embeddings_file:
                    print(f"  ⏭️  Skipping Pass 2 & 3 (no news embeddings provided)")
                    print(f"  ℹ️  Run again with news embeddings to complete normalization & tensorization\n")

                    # Clean up before returning
                    del all_dataframes
                    gc.collect()
                    return {}

        # ========================================================================
        # PASS 2: APPLY CROSS-SECTIONAL NORMALIZATION
        # ========================================================================

        print(f"\n{'='*80}")
        print(f"PASS 2: APPLYING CROSS-SECTIONAL NORMALIZATION")
        print(f"{'='*80}\n")

        normalizer = CrossSectionalNormalizer()
        # Use CPU-based normalization (default) to avoid GPU OOM on large datasets
        # Most systems have 64GB+ RAM but only 16-32GB VRAM
        # Features are processed in batches of 50 to conserve memory
        all_dataframes = normalizer.normalize_dataframes(
            all_dataframes,
            verbose=True,
            use_gpu=False,  # CPU normalization to avoid OOM
            feature_batch_size=50  # Process 50 features at a time
        )

        print(f"\n  ✅ Pass 2 complete: All stocks normalized\n")

        # Clean up after normalization
        gc.collect()

        # ========================================================================
        # PASS 2.25: ALIGN ALL DATAFRAME COLUMNS (FIX VARIABLE DIMENSIONS)
        # ========================================================================

        print(f"\n{'='*80}")
        print(f"PASS 2.25: ALIGNING DATAFRAME COLUMNS")
        print(f"{'='*80}\n")

        # Find all unique columns across all DataFrames
        print(f"  🔍 Finding all unique columns...")
        all_columns = set()
        for ticker, df in all_dataframes.items():
            all_columns.update(df.columns)

        all_columns = sorted(list(all_columns))
        print(f"  ✅ Found {len(all_columns)} unique columns")

        # Align all DataFrames to have the same columns
        print(f"  🔧 Aligning DataFrames...")
        num_aligned = 0
        for ticker, df in tqdm(all_dataframes.items(), desc="  Aligning", ncols=100):
            missing_cols = set(all_columns) - set(df.columns)
            if missing_cols:
                # Add missing columns filled with zeros
                for col in missing_cols:
                    df[col] = 0.0
                # Reindex to match column order
                all_dataframes[ticker] = df[all_columns]
                num_aligned += 1

        print(f"  ✅ Aligned {num_aligned} DataFrames to {len(all_columns)} columns")
        print(f"  ✅ All DataFrames now have the same feature dimensions\n")

        gc.collect()

        # ========================================================================
        # PASS 2.5: ADD CROSS-SECTIONAL PERCENTILE FEATURES (OPTIONAL)
        # ========================================================================

        cross_sectional_features = {}
        if add_cross_sectional:
            print(f"\n{'='*80}")
            print(f"PASS 2.5: ADDING CROSS-SECTIONAL PERCENTILE FEATURES")
            print(f"{'='*80}\n")

            # Get unique dates across all stocks
            all_dates = set()
            for df in all_dataframes.values():
                all_dates.update(df.index)
            all_dates = sorted(list(all_dates))

            print(f"  Total unique dates: {len(all_dates)}")
            print(f"  Calculating percentile rankings across stocks for each date...")

            # Calculate cross-sectional features
            calculator = CrossSectionalCalculator(sector_dict)

            # Process dates in batches to manage memory
            batch_size = 100
            for batch_start in tqdm(range(0, len(all_dates), batch_size), desc="  Processing date batches", ncols=100):
                batch_dates = all_dates[batch_start:batch_start + batch_size]

                for date_obj in batch_dates:
                    try:
                        date_features = calculator.calculate_cross_sectional_features_for_date(
                            all_dataframes, date_obj
                        )

                        # Store features for each ticker
                        for ticker, features_dict in date_features.items():
                            if ticker not in cross_sectional_features:
                                cross_sectional_features[ticker] = {}

                            # Store as dict for now, will convert to tensor later
                            cross_sectional_features[ticker][date_obj] = features_dict

                    except Exception as e:
                        # Skip problematic dates
                        continue

                # Periodic cleanup
                if batch_start % 1000 == 0:
                    gc.collect()

            print(f"\n  ✅ Cross-sectional percentile features calculated for {len(cross_sectional_features)} stocks")
        else:
            print(f"\n  ⏭️  Skipping cross-sectional percentile features\n")

        # ========================================================================
        # PASS 3: CONVERT TO TENSORS AND ADD NEWS EMBEDDINGS
        # ========================================================================

        print(f"\n{'='*80}")
        print(f"PASS 3: CONVERTING TO TENSORS")
        print(f"{'='*80}\n")

        processed_data = {}
        save_every = 500  # Save every 500 stocks to reduce I/O overhead

        # Convert to list to allow deletion during iteration
        all_dataframes_items = list(all_dataframes.items())

        for i, (ticker, df) in enumerate(all_dataframes_items, 1):
            try:
                # Get news embeddings for this ticker
                ticker_news_embeddings = None
                if news_embeddings_all and ticker in news_embeddings_all:
                    ticker_news_embeddings = news_embeddings_all[ticker]

                # Get cross-sectional features for this ticker
                ticker_cross_sectional = None
                if cross_sectional_features and ticker in cross_sectional_features:
                    ticker_cross_sectional = cross_sectional_features[ticker]

                # Convert to tensors with news embeddings and cross-sectional features
                tensor_dict = {}
                for date_idx, row in df.iterrows():
                    date_obj = date_idx.date() if isinstance(date_idx, pd.Timestamp) else date_idx

                    # Base features tensor
                    base_tensor = torch.tensor(row.values, dtype=torch.float32)

                    # Add cross-sectional percentile features (if enabled)
                    if ticker_cross_sectional and date_obj in ticker_cross_sectional:
                        cs_features = ticker_cross_sectional[date_obj]
                        cs_values = list(cs_features.values())
                        cs_tensor = torch.tensor(cs_values, dtype=torch.float32)
                    else:
                        # If no cross-sectional features, use zeros (or skip if not enabled)
                        cs_tensor = None if not add_cross_sectional else torch.tensor([], dtype=torch.float32)

                    # Add news embeddings
                    if ticker_news_embeddings and date_obj in ticker_news_embeddings:
                        news_tensor = ticker_news_embeddings[date_obj]
                    else:
                        news_tensor = torch.zeros(768, dtype=torch.float32)

                    # Concatenate all features
                    if cs_tensor is not None and len(cs_tensor) > 0:
                        tensor_dict[date_obj] = torch.cat([base_tensor, cs_tensor, news_tensor])
                    else:
                        tensor_dict[date_obj] = torch.cat([base_tensor, news_tensor])

                if tensor_dict:
                    processed_data[ticker] = tensor_dict
                    completed_tickers.add(ticker)

                # Delete the dataframe immediately after processing to free memory
                del df
                all_dataframes[ticker] = None  # Remove reference

                if i % 10 == 0:
                    news_dim = 768
                    total_features = len(all_dataframes_items[i-1][1].columns) + news_dim if i > 0 else 0
                    print(f"  [{i}/{len(all_dataframes_items)}] {ticker}: {len(tensor_dict)} days, {total_features} features")

            except Exception as e:
                print(f"  ❌ [{i}/{len(all_dataframes_items)}] {ticker}: {e}")
                # Still delete the dataframe even on error
                if ticker in all_dataframes:
                    all_dataframes[ticker] = None
                continue

            # Periodic garbage collection (every 50 stocks) without saving
            if i % 50 == 0 and i % save_every != 0:
                gc.collect()

            # Save incrementally and clean up aggressively (every 500 stocks)
            if i % save_every == 0:
                print(f"\n💾 Saving progress ({len(processed_data)} stocks)...")
                save_pickle(processed_data, output_file)

                # Aggressive memory cleanup
                # Delete processed DataFrames from dict
                for j in range(max(0, i - save_every), i):
                    if j < len(all_dataframes_items):
                        ticker_to_delete = all_dataframes_items[j][0]
                        if ticker_to_delete in all_dataframes:
                            del all_dataframes[ticker_to_delete]

                # Force garbage collection
                gc.collect()
                gc.collect()  # Run twice for better cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                process = psutil.Process(os.getpid())
                mem_gb = process.memory_info().rss / 1024**3
                print(f"   📊 RAM usage: {mem_gb:.2f} GB\n")

        # Final save
        if processed_data:
            print(f"\n💾 Final save...")
            save_pickle(processed_data, output_file)

        # Aggressive final cleanup
        print(f"\n🧹 Final memory cleanup...")
        del all_dataframes
        del all_dataframes_items
        if news_embeddings_all:
            del news_embeddings_all
        gc.collect()
        gc.collect()  # Run twice

        # Show final memory usage
        try:
            process = psutil.Process(os.getpid())
            mem_gb = process.memory_info().rss / 1024**3
            print(f"  📊 Final RAM usage: {mem_gb:.2f} GB")
        except:
            pass

    else:
        # OLD APPROACH: Temporal normalization only (for backward compatibility)
        print(f"\n⚠️  Using OLD temporal-only normalization (not recommended!)")
        print(f"   Set use_cross_sectional_normalization=True for better results\n")

        # [Original code would go here - keeping for backward compatibility]
        # For now, raise an error to encourage using the new approach
        raise ValueError("Please use cross-sectional normalization! Set use_cross_sectional_normalization=True")

    # Show final memory usage
    import psutil
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"   📊 Final RAM usage: {mem_gb:.2f} GB\n")


    # Load final data for statistics
    print(f"\n📊 Loading final statistics...")
    final_data = pic_load(output_file)

    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Stocks processed: {len(final_data)}")
    print(f"Output file: {output_file}")

    # Print feature statistics
    if final_data:
        sample_ticker = list(final_data.keys())[0]
        sample_date = list(final_data[sample_ticker].keys())[0]
        num_features = final_data[sample_ticker][sample_date].shape[0]
        print(f"Features per day: {num_features}")

        # Show breakdown
        if use_cross_sectional_normalization:
            base_features = num_features - 768
            print(f"  - Base features (cross-sectionally normalized): {base_features}")
            print(f"  - News embeddings (pre-normalized, not touched): 768")
        print(f"{'='*80}\n")

    return final_data


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
    parser.add_argument('--price_data', type=str, default=None,
                       help='Price data pickle file (from yfinance)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: input with _enhanced suffix)')
    parser.add_argument('--start_date', type=str, default='2000-01-01',
                       help='Start date')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (default: today)')
    parser.add_argument('--no_cross_sectional', action='store_true',
                       help='Disable cross-sectional percentile features')
    parser.add_argument('--no_cross_sectional_norm', action='store_true',
                       help='Disable cross-sectional normalization (NOT recommended!)')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.raw_data.replace('.pkl', '_enhanced.pkl')

    process_enhanced_data(
        raw_data_file=args.raw_data,
        market_indices_file=args.market_indices,
        sector_dict_file=args.sector_dict,
        news_embeddings_file=args.news_embeddings,
        price_data_file=args.price_data,
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        add_cross_sectional=not args.no_cross_sectional,
        use_cross_sectional_normalization=not args.no_cross_sectional_norm
    )

    print(f"\n✅ Done! Enhanced data saved to {args.output}")
