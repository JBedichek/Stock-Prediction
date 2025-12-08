"""
Hybrid Data Scraper: yfinance (prices) + FMP (fundamentals)

This scraper combines the best of both worlds:
- yfinance: Daily OHLCV price data (free, reliable, 20+ years)
- FMP: Quarterly fundamental data (161 quarters = 40+ years of financial statements)

Output format: {ticker: {date: tensor([N features])}}
where N includes:
- Price features: ~10 (OHLCV, returns, volatility, etc.)
- Fundamental features: ~100+ (financial ratios, margins, growth rates, etc.)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append('/home/james/Desktop/Stock-Prediction')
from data_scraping.fmp_comprehensive_scraper import FMPComprehensiveScraper
from utils.utils import save_pickle, pic_load


class HybridScraper:
    """
    Combines yfinance price data with FMP fundamental data.
    """

    def __init__(self, fmp_api_key: str):
        """
        Initialize hybrid scraper.

        Args:
            fmp_api_key: FMP API key
        """
        self.fmp_scraper = FMPComprehensiveScraper(fmp_api_key, rate_limit_delay=0.2)

    def get_yfinance_data(self, ticker: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get price data from yfinance.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with daily OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                return None

            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date'})

            return df

        except Exception as e:
            print(f"  ❌ yfinance error for {ticker}: {e}")
            return None

    def get_fmp_fundamentals(self, ticker: str) -> Dict:
        """
        Get fundamental data from FMP (what's available with free tier).

        Args:
            ticker: Stock ticker

        Returns:
            Dict with DataFrames of fundamental data
        """
        fundamentals = {}

        # Get what we know works from the test results
        print(f"  📊 Fetching fundamentals from FMP...")

        # Financial statements (161 quarters!)
        statements = self.fmp_scraper.get_financial_statements(ticker, limit=400)
        if statements:
            fundamentals['financial_statements'] = statements

        # Key metrics
        key_metrics = self.fmp_scraper.get_key_metrics(ticker, limit=400)
        if key_metrics is not None:
            fundamentals['key_metrics'] = key_metrics

        # Financial ratios
        ratios = self.fmp_scraper.get_financial_ratios(ticker, limit=400)
        if ratios is not None:
            fundamentals['financial_ratios'] = ratios

        # Enterprise values
        enterprise = self.fmp_scraper.get_enterprise_values(ticker, limit=400)
        if enterprise is not None:
            fundamentals['enterprise_values'] = enterprise

        # Growth metrics
        growth = self.fmp_scraper.get_financial_growth(ticker, limit=400)
        if growth is not None:
            fundamentals['financial_growth'] = growth

        # Company profile
        profile = self.fmp_scraper.get_company_profile(ticker)
        if profile:
            fundamentals['company_profile'] = profile

        return fundamentals

    def process_fundamentals(self, fundamentals: Dict) -> pd.DataFrame:
        """
        Process FMP fundamentals into quarterly features.

        Returns:
            DataFrame with date index and fundamental features
        """
        dfs = []

        # Process financial statements
        if 'financial_statements' in fundamentals:
            statements = fundamentals['financial_statements']

            for stmt_type, df in statements.items():
                if df is None or df.empty:
                    continue

                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')

                # Select numeric columns and add prefix
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols]
                df = df.add_prefix(f'{stmt_type}_')

                dfs.append(df)

        # Process key metrics
        if 'key_metrics' in fundamentals:
            df = fundamentals['key_metrics'].copy()
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols].add_prefix('metric_')
                dfs.append(df)

        # Process ratios
        if 'financial_ratios' in fundamentals:
            df = fundamentals['financial_ratios'].copy()
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols].add_prefix('ratio_')
                dfs.append(df)

        # Process enterprise values
        if 'enterprise_values' in fundamentals:
            df = fundamentals['enterprise_values'].copy()
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols].add_prefix('ev_')
                dfs.append(df)

        # Process growth metrics
        if 'financial_growth' in fundamentals:
            df = fundamentals['financial_growth'].copy()
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols].add_prefix('growth_')
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Combine all fundamental data
        combined = pd.concat(dfs, axis=1)
        return combined

    def process_prices(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process yfinance price data into features.

        Args:
            price_df: DataFrame from yfinance

        Returns:
            DataFrame with daily price features
        """
        df = price_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        features = pd.DataFrame(index=df.index)

        # Core OHLCV
        features['price_open'] = df['Open']
        features['price_high'] = df['High']
        features['price_low'] = df['Low']
        features['price_close'] = df['Close']
        features['price_volume'] = df['Volume']

        # Returns
        features['price_return_1d'] = df['Close'].pct_change(1)
        features['price_return_5d'] = df['Close'].pct_change(5)
        features['price_return_20d'] = df['Close'].pct_change(20)

        # Volatility
        features['price_volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        features['price_volatility_20d'] = df['Close'].pct_change().rolling(20).std()

        # Volume changes
        features['volume_change_1d'] = df['Volume'].pct_change(1)

        # Intraday range
        features['intraday_range'] = (df['High'] - df['Low']) / df['Close']

        # Moving averages
        features['price_sma_10'] = df['Close'].rolling(10).mean()
        features['price_sma_20'] = df['Close'].rolling(20).mean()
        features['price_sma_50'] = df['Close'].rolling(50).mean()
        features['price_sma_200'] = df['Close'].rolling(200).mean()

        # Price relative to moving averages
        features['price_vs_sma_10'] = df['Close'] / features['price_sma_10'] - 1
        features['price_vs_sma_20'] = df['Close'] / features['price_sma_20'] - 1
        features['price_vs_sma_50'] = df['Close'] / features['price_sma_50'] - 1

        return features

    def combine_data(self, price_features: pd.DataFrame,
                    fundamental_features: pd.DataFrame,
                    start_date: str, end_date: str) -> pd.DataFrame:
        """
        Combine price and fundamental features into daily time series.

        Args:
            price_features: Daily price features
            fundamental_features: Quarterly fundamental features
            start_date: Start date
            end_date: End date

        Returns:
            Combined DataFrame with daily frequency
        """
        # Create daily date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex price data to daily (already daily, just ensures coverage)
        price_daily = price_features.reindex(date_range)

        # Forward-fill fundamentals from quarterly to daily
        if not fundamental_features.empty:
            fundamental_daily = fundamental_features.reindex(date_range, method='ffill')
        else:
            fundamental_daily = pd.DataFrame(index=date_range)

        # Combine
        combined = pd.concat([price_daily, fundamental_daily], axis=1)

        # Fill NaNs with 0
        combined = combined.fillna(0)

        # Replace inf with 0
        combined = combined.replace([np.inf, -np.inf], 0)

        return combined

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize each column."""
        normalized = df.copy()

        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()

            if std == 0 or pd.isna(std):
                normalized[col] = 0
            else:
                normalized[col] = (df[col] - mean) / std

        return normalized

    def scrape_ticker(self, ticker: str, start_date: str = '2000-01-01',
                     end_date: str = None) -> Dict[date, torch.Tensor]:
        """
        Scrape all data for a single ticker.

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date (defaults to today)

        Returns:
            Dict of {date: tensor} with daily features
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n{'='*80}")
        print(f"SCRAPING {ticker}")
        print(f"{'='*80}")

        # Get price data from yfinance
        print(f"  📈 Fetching price data from yfinance...")
        price_df = self.get_yfinance_data(ticker, start_date, end_date)

        if price_df is None or price_df.empty:
            print(f"  ❌ No price data available")
            return {}

        print(f"     ✅ Got {len(price_df)} days of price data")

        # Get fundamentals from FMP
        fundamentals = self.get_fmp_fundamentals(ticker)

        if not fundamentals:
            print(f"  ⚠️  No fundamental data available, using prices only")

        # Process both
        print(f"  🔧 Processing features...")
        price_features = self.process_prices(price_df)
        fundamental_features = self.process_fundamentals(fundamentals)

        print(f"     Price features: {len(price_features.columns)}")
        print(f"     Fundamental features: {len(fundamental_features.columns)}")

        # Combine
        combined = self.combine_data(price_features, fundamental_features,
                                     start_date, end_date)

        # Normalize
        normalized = self.normalize_features(combined)

        # Convert to tensor dict
        tensor_dict = {}
        for date_idx, row in normalized.iterrows():
            date_obj = date_idx.date() if isinstance(date_idx, pd.Timestamp) else date_idx
            tensor_dict[date_obj] = torch.tensor(row.values, dtype=torch.float32)

        print(f"  ✅ Complete! {len(tensor_dict)} days, {len(normalized.columns)} total features")

        return tensor_dict


def scrape_hybrid_dataset(stock_dict: Dict[str, str], fmp_api_key: str,
                          output_file: str = 'hybrid_data.pkl',
                          start_date: str = '2000-01-01',
                          end_date: str = None,
                          resume_from: str = None) -> Dict[str, Dict[date, torch.Tensor]]:
    """
    Scrape hybrid dataset for multiple stocks.

    Args:
        stock_dict: {ticker: company_name}
        fmp_api_key: FMP API key
        output_file: Output pickle file
        start_date: Start date
        end_date: End date
        resume_from: Ticker to resume from

    Returns:
        Dict of {ticker: {date: tensor}}
    """
    scraper = HybridScraper(fmp_api_key)

    all_data = {}
    tickers = list(stock_dict.keys())

    # Resume logic
    if resume_from and resume_from in tickers:
        start_idx = tickers.index(resume_from)
        tickers = tickers[start_idx:]
        print(f"🔄 Resuming from {resume_from}")

    print(f"\n{'='*80}")
    print(f"HYBRID DATA SCRAPING (yfinance + FMP)")
    print(f"{'='*80}")
    print(f"Total stocks: {len(tickers)}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date or 'today'}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    for i, ticker in enumerate(tickers):
        company_name = stock_dict[ticker]
        print(f"\n[{i+1}/{len(tickers)}] {ticker} - {company_name}")

        try:
            tensor_dict = scraper.scrape_ticker(ticker, start_date, end_date)

            if tensor_dict:
                all_data[ticker] = tensor_dict

            # Save progress every 10 stocks
            if (i + 1) % 10 == 0:
                print(f"\n💾 Saving progress... ({len(all_data)} stocks collected)")
                save_pickle(all_data, output_file)

        except Exception as e:
            print(f"  ❌ Error scraping {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final save
    print(f"\n💾 Saving final dataset...")
    save_pickle(all_data, output_file)

    print(f"\n{'='*80}")
    print(f"✅ SCRAPING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total stocks collected: {len(all_data)}")
    print(f"FMP API calls: {scraper.fmp_scraper.call_count}")
    print(f"Output file: {output_file}")

    # Print statistics
    if all_data:
        sample_ticker = list(all_data.keys())[0]
        sample_dates = list(all_data[sample_ticker].keys())
        sample_tensor = all_data[sample_ticker][sample_dates[0]]

        print(f"\nDataset Statistics:")
        print(f"  Sample ticker: {sample_ticker}")
        print(f"  Date range: {min(sample_dates)} to {max(sample_dates)}")
        print(f"  Features per day: {sample_tensor.shape[0]}")

    return all_data


if __name__ == '__main__':
    import argparse
    from data_scraping.Stock import a_lot_of_stocks, s_lot_of_stocks, all_stocks

    parser = argparse.ArgumentParser(description='Hybrid data scraper (yfinance + FMP)')
    parser.add_argument('--api_key', type=str, default="YOUR_FMP_API_KEY_HERE",
                       help='FMP API key')
    parser.add_argument('--dataset', type=str, default='s_lot',
                       choices=['s_lot', 'a_lot', 'all'],
                       help='Dataset to scrape')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: {dataset}_hybrid_data.pkl)')
    parser.add_argument('--start_date', type=str, default='2000-01-01',
                       help='Start date for time-series data')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (default: today)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Ticker to resume from')

    args = parser.parse_args()

    # Select dataset
    if args.dataset == 's_lot':
        stock_dict = s_lot_of_stocks
    elif args.dataset == 'a_lot':
        stock_dict = a_lot_of_stocks
    else:
        stock_dict = all_stocks

    # Set output file
    if args.output is None:
        args.output = f"{args.dataset}_hybrid_data.pkl"

    # Scrape
    data = scrape_hybrid_dataset(
        stock_dict=stock_dict,
        fmp_api_key=args.api_key,
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        resume_from=args.resume
    )

    print(f"\n✅ Done! Data saved to {args.output}")
