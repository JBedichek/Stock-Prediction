#!/usr/bin/env python3
"""
Dataset Postprocessing Script

Cleans up the assembled dataset by removing companies with:
- No price data (critical)
- Companies no longer listed (no recent trading activity)
- Completely empty fundamental data

Keeps companies with:
- Missing news data (acceptable - will use zero embeddings)
- Partial data (as long as price data exists)

Usage:
    python dataset_postprocessing.py --input all_complete_dataset.pkl --output all_complete_dataset_cleaned.pkl

    # Or process raw data files directly
    python dataset_postprocessing.py --comprehensive all_fmp_comprehensive.pkl --price all_price_data.pkl --output cleaned_data
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import pic_load, save_pickle


class DatasetPostprocessor:
    """
    Postprocessor to clean up assembled datasets.
    """

    def __init__(self, min_price_days: int = 30, min_recent_days: int = 60):
        """
        Initialize postprocessor.

        Args:
            min_price_days: Minimum number of days with price data required
            min_recent_days: Number of days to check for recent trading activity
        """
        self.min_price_days = min_price_days
        self.min_recent_days = min_recent_days
        self.removal_reasons = {
            'no_price_data': [],
            'insufficient_price_data': [],
            'delisted': [],
            'no_fundamental_data': [],
            'empty_data': []
        }

    def check_price_data(self, ticker: str, price_df: pd.DataFrame) -> tuple[bool, str]:
        """
        Check if price data is valid.

        Args:
            ticker: Stock ticker
            price_df: Price DataFrame

        Returns:
            (is_valid, reason)
        """
        # Check if DataFrame is empty
        if price_df is None or price_df.empty:
            return False, 'no_price_data'

        # Check if we have minimum number of days
        if len(price_df) < self.min_price_days:
            return False, 'insufficient_price_data'

        # Check for recent trading activity (company still listed)
        if 'date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['date'])
            latest_date = price_df['date'].max()
        else:
            # Assume index is date
            price_df.index = pd.to_datetime(price_df.index)
            latest_date = price_df.index.max()

        cutoff_date = datetime.now() - timedelta(days=self.min_recent_days)
        if latest_date < cutoff_date:
            return False, 'delisted'

        return True, None

    def check_fundamental_data(self, ticker: str, comprehensive_data: dict) -> tuple[bool, str]:
        """
        Check if fundamental data is valid.

        Args:
            ticker: Stock ticker
            comprehensive_data: Comprehensive data dict from FMP scraper

        Returns:
            (is_valid, reason)
        """
        if comprehensive_data is None or not isinstance(comprehensive_data, dict):
            return False, 'no_fundamental_data'

        # Check if we have at least some data
        has_any_data = False

        # Check critical fields
        critical_fields = [
            'daily_prices',
            'financial_statements',
            'company_profile'
        ]

        for field in critical_fields:
            if field in comprehensive_data:
                data = comprehensive_data[field]
                if data is not None:
                    if isinstance(data, (pd.DataFrame, list, dict)):
                        if len(data) > 0:
                            has_any_data = True
                            break

        if not has_any_data:
            return False, 'empty_data'

        return True, None

    def process_raw_data_files(self, comprehensive_file: str, price_file: str,
                               news_file: str = None, output_prefix: str = 'cleaned') -> dict:
        """
        Process raw data files and create cleaned versions.

        Args:
            comprehensive_file: Path to comprehensive FMP data file
            price_file: Path to price data file
            news_file: Optional path to news data file
            output_prefix: Prefix for output files

        Returns:
            Statistics dictionary
        """
        print("="*80)
        print("  DATASET POSTPROCESSING - RAW FILES MODE")
        print("="*80)

        # Load data
        print("\nLoading data files...")
        print(f"  📊 Comprehensive: {comprehensive_file}")
        comprehensive_data = pic_load(comprehensive_file)

        print(f"  💹 Price data: {price_file}")
        price_data = pic_load(price_file)

        news_data = None
        if news_file and os.path.exists(news_file):
            print(f"  📰 News data: {news_file}")
            news_data = pic_load(news_file)

        # Get all tickers
        all_tickers = set(comprehensive_data.keys()) | set(price_data.keys())
        print(f"\n📈 Total tickers found: {len(all_tickers)}")

        # Clean data
        cleaned_comprehensive = {}
        cleaned_price = {}
        cleaned_news = {}

        print("\n🔍 Processing tickers...")
        for ticker in tqdm(all_tickers):
            # Get ticker data
            ticker_comprehensive = comprehensive_data.get(ticker)
            ticker_price = price_data.get(ticker)

            # Check price data (CRITICAL)
            if ticker not in price_data:
                self.removal_reasons['no_price_data'].append(ticker)
                continue

            price_valid, price_reason = self.check_price_data(ticker, ticker_price)
            if not price_valid:
                self.removal_reasons[price_reason].append(ticker)
                continue

            # Check fundamental data
            if ticker not in comprehensive_data:
                self.removal_reasons['no_fundamental_data'].append(ticker)
                continue

            fundamental_valid, fundamental_reason = self.check_fundamental_data(
                ticker, ticker_comprehensive
            )
            if not fundamental_valid:
                self.removal_reasons[fundamental_reason].append(ticker)
                continue

            # Ticker passes all checks - keep it
            cleaned_comprehensive[ticker] = ticker_comprehensive
            cleaned_price[ticker] = ticker_price

            # News is optional - keep if available
            if news_data and ticker in news_data:
                cleaned_news[ticker] = news_data[ticker]

        # Print statistics
        stats = self._print_statistics(len(all_tickers), len(cleaned_comprehensive))

        # Save cleaned files
        print("\n💾 Saving cleaned files...")
        output_dir = Path(comprehensive_file).parent

        comprehensive_output = output_dir / f'{output_prefix}_fmp_comprehensive.pkl'
        price_output = output_dir / f'{output_prefix}_price_data.pkl'

        save_pickle(cleaned_comprehensive, str(comprehensive_output))
        print(f"  ✅ Comprehensive: {comprehensive_output} ({len(cleaned_comprehensive)} tickers)")

        save_pickle(cleaned_price, str(price_output))
        print(f"  ✅ Price data: {price_output} ({len(cleaned_price)} tickers)")

        if cleaned_news:
            news_output = output_dir / f'{output_prefix}_news_data.pkl'
            save_pickle(cleaned_news, str(news_output))
            print(f"  ✅ News data: {news_output} ({len(cleaned_news)} tickers)")

        return stats

    def process_complete_dataset(self, dataset_file: str,
                                 comprehensive_file: str = None,
                                 price_file: str = None,
                                 output_file: str = None) -> dict:
        """
        Process a complete dataset (output from generate_full_dataset.py).

        Args:
            dataset_file: Path to complete dataset file (e.g., all_complete_dataset.pkl)
            comprehensive_file: Optional path to comprehensive data for validation
            price_file: Optional path to price data for validation
            output_file: Path for cleaned output

        Returns:
            Statistics dictionary
        """
        print("="*80)
        print("  DATASET POSTPROCESSING - COMPLETE DATASET MODE")
        print("="*80)

        # Load complete dataset
        print(f"\nLoading complete dataset: {dataset_file}")
        complete_data = pic_load(dataset_file)

        # Load validation data if provided
        price_data = None
        comprehensive_data = None

        if price_file:
            print(f"Loading price data for validation: {price_file}")
            price_data = pic_load(price_file)

        if comprehensive_file:
            print(f"Loading comprehensive data for validation: {comprehensive_file}")
            comprehensive_data = pic_load(comprehensive_file)

        print(f"\n📈 Total tickers in dataset: {len(complete_data)}")

        # Clean dataset
        cleaned_data = {}

        print("\n🔍 Processing tickers...")
        for ticker in tqdm(complete_data.keys()):
            ticker_data = complete_data[ticker]

            # Check if ticker has any data at all
            if ticker_data is None or len(ticker_data) == 0:
                self.removal_reasons['empty_data'].append(ticker)
                continue

            # If validation data available, use it
            if price_data is not None:
                if ticker not in price_data:
                    self.removal_reasons['no_price_data'].append(ticker)
                    continue

                price_valid, price_reason = self.check_price_data(ticker, price_data[ticker])
                if not price_valid:
                    self.removal_reasons[price_reason].append(ticker)
                    continue

            if comprehensive_data is not None:
                if ticker not in comprehensive_data:
                    self.removal_reasons['no_fundamental_data'].append(ticker)
                    continue

                fundamental_valid, fundamental_reason = self.check_fundamental_data(
                    ticker, comprehensive_data[ticker]
                )
                if not fundamental_valid:
                    self.removal_reasons[fundamental_reason].append(ticker)
                    continue

            # Check number of days with data
            num_days = len(ticker_data)
            if num_days < self.min_price_days:
                self.removal_reasons['insufficient_price_data'].append(ticker)
                continue

            # Ticker passes all checks
            cleaned_data[ticker] = ticker_data

        # Print statistics
        stats = self._print_statistics(len(complete_data), len(cleaned_data))

        # Save cleaned dataset
        if output_file is None:
            output_file = dataset_file.replace('.pkl', '_cleaned.pkl')

        print(f"\n💾 Saving cleaned dataset: {output_file}")
        save_pickle(cleaned_data, output_file)
        print(f"  ✅ Saved {len(cleaned_data)} tickers")

        return stats

    def _print_statistics(self, original_count: int, cleaned_count: int) -> dict:
        """Print cleaning statistics."""
        print("\n" + "="*80)
        print("  CLEANING STATISTICS")
        print("="*80)

        total_removed = original_count - cleaned_count

        print(f"\n📊 Original tickers: {original_count}")
        print(f"✅ Cleaned tickers: {cleaned_count}")
        print(f"❌ Removed tickers: {total_removed} ({total_removed/original_count*100:.1f}%)")

        print("\n📋 Removal breakdown:")
        for reason, tickers in self.removal_reasons.items():
            if tickers:
                print(f"  • {reason.replace('_', ' ').title()}: {len(tickers)}")
                if len(tickers) <= 10:
                    print(f"    {', '.join(tickers)}")
                else:
                    print(f"    First 10: {', '.join(tickers[:10])}")

        stats = {
            'original_count': original_count,
            'cleaned_count': cleaned_count,
            'removed_count': total_removed,
            'removal_reasons': {k: len(v) for k, v in self.removal_reasons.items()}
        }

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Clean up stock prediction datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process complete dataset
  python dataset_postprocessing.py --dataset all_complete_dataset.pkl --output all_complete_dataset_cleaned.pkl

  # Process raw files
  python dataset_postprocessing.py --comprehensive all_fmp_comprehensive.pkl --price all_price_data.pkl --output cleaned

  # Process with validation
  python dataset_postprocessing.py --dataset all_complete_dataset.pkl --comprehensive all_fmp_comprehensive.pkl --price all_price_data.pkl

  # Adjust thresholds
  python dataset_postprocessing.py --dataset all_complete_dataset.pkl --min-days 60 --min-recent-days 90
        """
    )

    parser.add_argument('--dataset', type=str, default=None,
                       help='Complete dataset file (e.g., all_complete_dataset.pkl)')
    parser.add_argument('--comprehensive', type=str, default=None,
                       help='Comprehensive FMP data file (e.g., all_fmp_comprehensive.pkl)')
    parser.add_argument('--price', type=str, default=None,
                       help='Price data file (e.g., all_price_data.pkl)')
    parser.add_argument('--news', type=str, default=None,
                       help='News data file (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path or prefix')
    parser.add_argument('--min-days', type=int, default=30,
                       help='Minimum number of days with price data (default: 30)')
    parser.add_argument('--min-recent-days', type=int, default=60,
                       help='Maximum days since last trading activity (default: 60)')

    args = parser.parse_args()

    # Validate arguments
    if args.dataset is None and (args.comprehensive is None or args.price is None):
        parser.error("Either --dataset OR both --comprehensive and --price must be provided")

    # Create postprocessor
    postprocessor = DatasetPostprocessor(
        min_price_days=args.min_days,
        min_recent_days=args.min_recent_days
    )

    # Process based on mode
    if args.dataset:
        # Complete dataset mode
        stats = postprocessor.process_complete_dataset(
            dataset_file=args.dataset,
            comprehensive_file=args.comprehensive,
            price_file=args.price,
            output_file=args.output
        )
    else:
        # Raw files mode
        output_prefix = args.output if args.output else 'cleaned'
        stats = postprocessor.process_raw_data_files(
            comprehensive_file=args.comprehensive,
            price_file=args.price,
            news_file=args.news,
            output_prefix=output_prefix
        )

    print("\n✅ Dataset postprocessing complete!")
    return stats


if __name__ == '__main__':
    main()
