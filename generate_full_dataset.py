#!/usr/bin/env python3
"""
End-to-End Dataset Generation Script

This script generates the COMPLETE dataset for stock prediction with all features:
- Market indices (S&P 500, VIX, sector ETFs)
- Stock fundamentals (quarterly financial statements, ratios, metrics)
- Technical indicators (extended set)
- News embeddings (Nomic AI)
- Derived features (volume, price, momentum)
- Cross-sectional rankings

Handles:
- Progress saving and resume capability
- Error recovery
- All edge cases (missing news, failed scrapes, etc.)
- Multi-stock datasets (test, s_lot, a_lot, all_stocks)

Output: Complete feature tensors ready for training (1100-1200 features per day)
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.append('/home/james/Desktop/Stock-Prediction')

from data_scraping.market_indices_scraper import scrape_market_data
from data_scraping.fmp_comprehensive_scraper import scrape_dataset
from data_scraping.yfinance_price_scraper import scrape_price_dataset
from data_scraping.news_scraper import scrape_news_dataset as gnews_scrape
from data_scraping.alphavantage_news_scraper import scrape_news_dataset as av_scrape
from data_scraping.news_embedder import create_news_embeddings
from data_scraping.fmp_enhanced_processor import process_enhanced_data
from data_scraping.Stock import (
    all_stocks,
    s_lot_of_stocks,
    a_lot_of_stocks,
    test_stock_tickers
)
from utils.utils import pic_load, save_pickle


def print_banner(text):
    """Print a nice banner."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def check_file_exists(filepath):
    """Check if a file exists and show its info."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✅ Found existing file: {filepath} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  ⚠️  File not found: {filepath}")
        return False


class DatasetGenerator:
    """
    End-to-end dataset generator with resume capability.
    """

    def __init__(self, dataset_name: str, api_key: str, years: int = 25,
                 news_years: int = None, output_dir: str = '.',
                 news_source: str = 'gnews', av_api_key: str = None,
                 av_rate_limit: int = 75):
        """
        Initialize generator.

        Args:
            dataset_name: 'test', 's_lot', 'a_lot', or 'all'
            api_key: FMP API key
            years: Years of historical data
            news_years: Years of news (defaults to same as years)
            output_dir: Directory for output files
            news_source: 'gnews' or 'alphavantage'
            av_api_key: Alpha Vantage API key (required if news_source='alphavantage')
            av_rate_limit: Alpha Vantage rate limit (75 for Premium, 5 for Free)
        """
        self.dataset_name = dataset_name
        self.api_key = api_key
        self.years = years
        self.news_years = news_years if news_years else years
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.news_source = news_source
        self.av_api_key = av_api_key
        self.av_rate_limit = av_rate_limit

        # Select stock dictionary
        self.stock_dict = self._get_stock_dict()

        # Calculate dates
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data_start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        self.news_start_date = (datetime.now() - timedelta(days=self.news_years*365)).strftime('%Y-%m-%d')

        # Output filenames
        self.market_indices_file = self.output_dir / 'market_indices_data.pkl'
        self.fundamentals_file = self.output_dir / f'{dataset_name}_fmp_comprehensive.pkl'
        self.price_data_file = self.output_dir / f'{dataset_name}_price_data.pkl'
        self.news_data_file = self.output_dir / f'{dataset_name}_news_data.pkl'
        self.news_embeddings_file = self.output_dir / f'{dataset_name}_news_embeddings.pkl'
        self.sector_dict_file = self.output_dir / f'{dataset_name}_sector_dict'
        self.final_output_file = self.output_dir / f'{dataset_name}_complete_dataset.pkl'

        print_banner("DATASET GENERATOR INITIALIZED")
        print(f"Dataset: {dataset_name}")
        print(f"Stocks: {len(self.stock_dict)}")
        print(f"Historical data: {years} years ({self.data_start_date} to {self.end_date})")
        print(f"News data: {self.news_years} years ({self.news_start_date} to {self.end_date})")
        print(f"Output directory: {self.output_dir}")
        print(f"API Key: {api_key[:10]}...")

    def _get_stock_dict(self):
        """Get stock dictionary based on dataset name."""
        if self.dataset_name == 'test':
            return test_stock_tickers
        elif self.dataset_name == 's_lot':
            return s_lot_of_stocks
        elif self.dataset_name == 'a_lot':
            return a_lot_of_stocks
        elif self.dataset_name == 'all':
            return all_stocks
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def step1_market_indices(self, force_rescrape=False):
        """
        Step 1: Scrape market indices.

        Time: ~2-3 minutes
        API Calls: ~15 total
        """
        print_banner("STEP 1/5: MARKET INDICES")

        if check_file_exists(self.market_indices_file) and not force_rescrape:
            response = input("  Use existing file? [Y/n]: ").strip().lower()
            if response != 'n':
                print("  ✅ Using existing market indices")
                return True

        print(f"\nScraping market indices...")
        print(f"  Date range: {self.data_start_date} to {self.end_date}")

        try:
            scrape_market_data(
                api_key=self.api_key,
                from_date=self.data_start_date,
                to_date=self.end_date,
                output_file=str(self.market_indices_file)
            )
            print(f"\n✅ Step 1 Complete: {self.market_indices_file}")
            return True

        except Exception as e:
            print(f"\n❌ Step 1 Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step2_fundamentals(self, force_rescrape=False, resume_from=None):
        """
        Step 2: Scrape stock fundamentals.

        Time: ~30-60 min for s_lot, 5-10 hours for all_stocks
        API Calls: ~30 per stock
        """
        print_banner("STEP 2/5: STOCK FUNDAMENTALS")

        if check_file_exists(self.fundamentals_file) and not force_rescrape:
            response = input("  Use existing file? [Y/n]: ").strip().lower()
            if response != 'n':
                print("  ✅ Using existing fundamentals")
                return True

        print(f"\nScraping fundamentals for {len(self.stock_dict)} stocks...")
        print(f"  Date range: {self.data_start_date} to {self.end_date}")
        print(f"  Estimated time: {len(self.stock_dict) * 0.5:.0f} minutes")
        print(f"  Estimated API calls: {len(self.stock_dict) * 30}")

        if resume_from:
            print(f"  Resuming from: {resume_from}")

        try:
            scrape_dataset(
                stock_dict=self.stock_dict,
                api_key=self.api_key,
                output_file=str(self.fundamentals_file),
                start_date=self.data_start_date,
                resume_from=resume_from
            )
            print(f"\n✅ Step 2 Complete: {self.fundamentals_file}")
            return True

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted! Progress saved.")
            print("  Resume with: --resume_from TICKER")
            raise

        except Exception as e:
            print(f"\n❌ Step 2 Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step2_5_prices(self, force_rescrape=False):
        """
        Step 2.5: Scrape daily price data using yfinance.

        Time: ~5-15 minutes for most datasets
        API Calls: 0 (yfinance is free)
        """
        print_banner("STEP 2.5/6: DAILY PRICE DATA (YFINANCE)")

        if check_file_exists(self.price_data_file) and not force_rescrape:
            response = input("  Use existing file? [Y/n]: ").strip().lower()
            if response != 'n':
                print("  ✅ Using existing price data")
                return True

        print(f"\nScraping price data for {len(self.stock_dict)} stocks...")
        print(f"  Date range: {self.data_start_date} to {self.end_date}")
        print(f"  Source: Yahoo Finance (via yfinance)")
        print(f"  Estimated time: {len(self.stock_dict) * 0.5 / 60:.1f} minutes")

        try:
            scrape_price_dataset(
                stock_dict=self.stock_dict,
                start_date=self.data_start_date,
                end_date=self.end_date,
                output_file=str(self.price_data_file),
                batch_size=100
            )
            print(f"\n✅ Step 2.5 Complete: {self.price_data_file}")
            return True

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted! Progress saved.")
            raise

        except Exception as e:
            print(f"\n❌ Step 2.5 Failed: {e}")
            print("  WARNING: Proceeding without price data (technical indicators will be missing)")
            import traceback
            traceback.print_exc()
            # Create empty price dict as fallback
            save_pickle({ticker: pd.DataFrame() for ticker in self.stock_dict}, str(self.price_data_file))
            return True  # Continue even if price scraping fails

    def step3_news(self, force_rescrape=False):
        """
        Step 3: Scrape news articles.

        Time: Variable (depends on news source)
        - Alpha Vantage: ~5-25 minutes (depending on rate limit and chunking)
        - GNews: 30+ hours for 370 stocks with 25 years
        """
        print_banner("STEP 3/6: NEWS ARTICLES")

        if check_file_exists(self.news_data_file) and not force_rescrape:
            # Check if scraping is complete
            try:
                existing_news = pic_load(str(self.news_data_file))
                total_stocks = len(self.stock_dict)

                # Count stocks that are in the file
                scraped_stocks = len(existing_news)

                # Count stocks with actual articles (not just empty lists)
                stocks_with_articles = sum(1 for articles in existing_news.values() if len(articles) > 0)

                # Count missing stocks
                missing_stocks = [ticker for ticker in self.stock_dict if ticker not in existing_news]
                num_missing = len(missing_stocks)

                # Count failed stocks (in file but 0 articles)
                failed_stocks = [ticker for ticker, articles in existing_news.items() if len(articles) == 0]
                num_failed = len(failed_stocks)

                print(f"  Scraped: {scraped_stocks}/{total_stocks} stocks")
                print(f"  With articles: {stocks_with_articles}/{total_stocks} stocks")
                if num_failed > 0:
                    print(f"  ⚠️  Failed (0 articles): {num_failed} stocks")
                if num_missing > 0:
                    print(f"  ⚠️  Not yet scraped: {num_missing} stocks")

                if num_missing == 0:
                    # All stocks attempted
                    if num_failed == 0:
                        # Perfect - all stocks have articles
                        response = input("  Use existing file (complete)? [Y/n]: ").strip().lower()
                        if response != 'n':
                            print("  ✅ Using existing news data")
                            return True
                    else:
                        # All attempted, but some have 0 articles
                        print(f"  ⚠️  {num_failed} stocks have 0 articles (may not be in Alpha Vantage)")
                        print(f"     First 5: {', '.join(failed_stocks[:5])}")
                        print(f"\n  All stocks have been attempted.")
                        response = input("  [P]roceed with existing data, [R]etry failed stocks, or [C]ancel? [P/r/c]: ").strip().lower()

                        if response in ['', 'p']:
                            print("  ✅ Proceeding with existing data (stocks with 0 articles will use zero embeddings)")
                            return True
                        elif response == 'r':
                            print("  🔄 Retrying failed stocks...")
                            # Continue to scraping logic below
                        else:
                            print("  ❌ User cancelled. Exiting...")
                            return False
                else:
                    # Partial progress - not all stocks attempted
                    print(f"  ⚠️  Incomplete: {num_missing} stocks not yet attempted")
                    if num_failed > 0:
                        print(f"  ⚠️  {num_failed} stocks failed (0 articles)")
                        print(f"     First 5 failed: {', '.join(failed_stocks[:5])}")

                    response = input("  Resume scraping (will retry failed + scrape missing)? [Y/n]: ").strip().lower()
                    if response == 'n':
                        print("  ❌ User chose not to resume. Exiting...")
                        return False
                    else:
                        print("  ✅ Resuming scraping...")
                        # Continue to scraping logic below
            except Exception as e:
                print(f"  ⚠️  Could not check progress: {e}")
                response = input("  Use existing file? [Y/n]: ").strip().lower()
                if response != 'n':
                    print("  ✅ Using existing news data")
                    return True

        print(f"\nScraping news for {len(self.stock_dict)} stocks...")
        print(f"  Date range: {self.news_start_date} to {self.end_date}")
        print(f"  News source: {self.news_source}")

        try:
            if self.news_source == 'alphavantage':
                if not self.av_api_key:
                    raise ValueError("Alpha Vantage API key required! Use --av-api-key")

                print(f"  Alpha Vantage rate limit: {self.av_rate_limit} calls/min")
                print(f"  Note: Alpha Vantage may have limited historical coverage (~3-4 years)")
                print(f"  Estimated time: {len(self.stock_dict) * 6 * (60/self.av_rate_limit) / 60:.1f} minutes")

                av_scrape(
                    stock_dict=self.stock_dict,
                    start_date=self.news_start_date,
                    end_date=self.end_date,
                    output_file=str(self.news_data_file),
                    api_key=self.av_api_key,
                    rate_limit=self.av_rate_limit,
                    use_chunking=True,
                    chunk_years=5
                )

            else:  # gnews
                print(f"  Sources: Google News, Yahoo RSS")
                print(f"  Note: Slow but comprehensive (30+ hours for 370 stocks)")

                gnews_scrape(
                    stock_dict=self.stock_dict,
                    start_date=self.news_start_date,
                    end_date=self.end_date,
                    output_file=str(self.news_data_file),
                    fmp_api_key=self.api_key
                )

            print(f"\n✅ Step 3 Complete: {self.news_data_file}")
            return True

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted! Progress saved.")
            raise

        except Exception as e:
            print(f"\n❌ Step 3 Failed: {e}")
            print("  Continuing without news (will use zeros)")
            # Create empty news dict as fallback
            save_pickle({ticker: [] for ticker in self.stock_dict}, str(self.news_data_file))
            return True

    def step4_embed_news(self, force_reembed=False, device='cuda'):
        """
        Step 4: Embed news with Nomic AI.

        Time: ~5-20 minutes depending on GPU
        API Calls: 0 (local model)
        """
        print_banner("STEP 4/6: NEWS EMBEDDINGS")

        if check_file_exists(self.news_embeddings_file) and not force_reembed:
            response = input("  Use existing file? [Y/n]: ").strip().lower()
            if response != 'n':
                print("  ✅ Using existing embeddings")
                return True

        if not check_file_exists(self.news_data_file):
            print(f"\n⚠️  News data not found! Creating empty embeddings...")
            # Create empty embeddings
            save_pickle({}, str(self.news_embeddings_file))
            return True

        print(f"\nEmbedding news articles...")
        print(f"  Model: nomic-ai/nomic-embed-text-v1.5")
        print(f"  Device: {device}")
        print(f"  Embedding dimension: 768")

        try:
            create_news_embeddings(
                news_data_file=str(self.news_data_file),
                output_file=str(self.news_embeddings_file),
                start_date=self.news_start_date,
                end_date=self.end_date,
                device=device
            )
            print(f"\n✅ Step 4 Complete: {self.news_embeddings_file}")
            return True

        except Exception as e:
            print(f"\n❌ Step 4 Failed: {e}")
            print("  Creating empty embeddings as fallback...")
            save_pickle({}, str(self.news_embeddings_file))
            import traceback
            traceback.print_exc()
            return True  # Continue even if embedding fails

    def step5_process_enhanced_features(self, skip_cross_sectional=False,
                                        use_cross_sectional_normalization=True):
        """
        Step 5: Process enhanced features.

        Time: ~10-30 minutes depending on dataset size
        API Calls: 0 (all processing)
        """
        print_banner("STEP 5/6: ENHANCED FEATURES")

        print(f"\nProcessing enhanced features...")
        print(f"  Input files:")
        print(f"    - Fundamentals: {self.fundamentals_file.name}")
        print(f"    - Price data: {self.price_data_file.name}")
        print(f"    - Market indices: {self.market_indices_file.name}")
        print(f"    - News embeddings: {self.news_embeddings_file.name}")
        print(f"    - Sector dict: {self.sector_dict_file.name}")
        print(f"  Cross-sectional percentile features: {not skip_cross_sectional}")
        print(f"  Cross-sectional normalization: {use_cross_sectional_normalization}")
        print(f"  Output: {self.final_output_file}")

        # Check required files
        if not check_file_exists(self.fundamentals_file):
            print("\n❌ Missing fundamentals file!")
            return False

        try:
            process_enhanced_data(
                raw_data_file=str(self.fundamentals_file),
                market_indices_file=str(self.market_indices_file) if self.market_indices_file.exists() else None,
                sector_dict_file=str(self.sector_dict_file) if self.sector_dict_file.exists() else None,
                news_embeddings_file=str(self.news_embeddings_file) if self.news_embeddings_file.exists() else None,
                price_data_file=str(self.price_data_file) if self.price_data_file.exists() else None,
                output_file=str(self.final_output_file),
                start_date=self.data_start_date,
                end_date=self.end_date,
                add_cross_sectional=not skip_cross_sectional,
                use_cross_sectional_normalization=use_cross_sectional_normalization
            )
            print(f"\n✅ Step 5 Complete: {self.final_output_file}")
            return True

        except Exception as e:
            print(f"\n❌ Step 5 Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all(self, skip_steps=None, force_rescrape=False, resume_from=None,
                device='cuda', skip_cross_sectional=False,
                use_cross_sectional_normalization=True):
        """
        Run all steps in sequence.

        Args:
            skip_steps: List of step numbers to skip (e.g., [1, 2])
            force_rescrape: Force rescraping even if files exist
            resume_from: Ticker to resume from (for step 2)
            device: Device for embedding ('cuda' or 'cpu')
            skip_cross_sectional: Skip cross-sectional percentile features (faster, recommended)
            use_cross_sectional_normalization: Use cross-sectional normalization for fundamentals (recommended!)

        Returns:
            True if all steps completed successfully
        """
        skip_steps = skip_steps or []
        start_time = datetime.now()

        print_banner("STARTING COMPLETE DATASET GENERATION")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Cross-sectional normalization: {'ENABLED' if use_cross_sectional_normalization else 'DISABLED'}")

        steps = [
            (1, "Market Indices", lambda: self.step1_market_indices(force_rescrape)),
            (2, "Fundamentals", lambda: self.step2_fundamentals(force_rescrape, resume_from)),
            (2.5, "Price Data", lambda: self.step2_5_prices(force_rescrape)),
            (3, "News Scraping", lambda: self.step3_news(force_rescrape)),
            (4, "News Embedding", lambda: self.step4_embed_news(force_rescrape, device)),
            (5, "Enhanced Features", lambda: self.step5_process_enhanced_features(skip_cross_sectional, use_cross_sectional_normalization)),
        ]

        results = {}

        # Create global progress bar
        total_steps = len([s for s in steps if s[0] not in skip_steps])
        with tqdm(total=total_steps, desc="Overall Progress",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]',
                  position=0, leave=True) as pbar:

            for step_num, step_name, step_func in steps:
                if step_num in skip_steps:
                    print(f"\n⏭️  Skipping Step {step_num}: {step_name}")
                    results[step_num] = None
                    continue

                # Update progress bar description
                pbar.set_description(f"Overall Progress (Step {step_num}/5: {step_name})")

                try:
                    success = step_func()
                    results[step_num] = success

                    if not success:
                        print(f"\n❌ Pipeline stopped at Step {step_num}")
                        return False

                    # Update progress bar
                    pbar.update(1)

                except KeyboardInterrupt:
                    print(f"\n⚠️  Pipeline interrupted at Step {step_num}")
                    raise

                except Exception as e:
                    print(f"\n❌ Unexpected error in Step {step_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    return False

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print_banner("DATASET GENERATION COMPLETE!")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print(f"\nOutput files:")
        print(f"  📊 Market indices: {self.market_indices_file}")
        print(f"  📈 Fundamentals: {self.fundamentals_file}")
        print(f"  💹 Price data: {self.price_data_file}")
        print(f"  📰 News articles: {self.news_data_file}")
        print(f"  🤖 News embeddings: {self.news_embeddings_file}")
        print(f"  ✨ FINAL DATASET: {self.final_output_file}")

        # Load and show stats
        try:
            final_data = pic_load(str(self.final_output_file))
            sample_ticker = list(final_data.keys())[0]
            sample_date = list(final_data[sample_ticker].keys())[0]
            num_features = final_data[sample_ticker][sample_date].shape[0]

            print(f"\nDataset statistics:")
            print(f"  Stocks: {len(final_data)}")
            print(f"  Features per day: {num_features}")
            print(f"  Sample ticker: {sample_ticker}")
            print(f"  Sample date: {sample_date}")
            print(f"  File size: {os.path.getsize(self.final_output_file) / (1024**3):.2f} GB")

        except Exception as e:
            print(f"\n⚠️  Could not load final dataset for stats: {e}")

        print("\n🎉 Ready for training!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate complete stock prediction dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run with minimal data
  python generate_full_dataset.py --dataset test --years 1

  # Generate s_lot dataset (370 stocks, 25 years)
  python generate_full_dataset.py --dataset s_lot --years 25

  # Generate ALL stocks (4000+, 25 years) - LONG RUNNING!
  python generate_full_dataset.py --dataset all --years 25 --output-dir /data/stocks

  # Resume after interruption
  python generate_full_dataset.py --dataset s_lot --resume-from MSFT

  # Skip steps (use existing files)
  python generate_full_dataset.py --dataset s_lot --skip-steps 1 2 3

  # Force rescrape everything
  python generate_full_dataset.py --dataset s_lot --force-rescrape
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       choices=['test', 's_lot', 'a_lot', 'all'],
                       help='Dataset to generate')
    parser.add_argument('--api-key', type=str, required=False, default="placeholder",
                       help='FMP API key (required)')
    parser.add_argument('--years', type=int, default=25,
                       help='Years of historical data (default: 25)')
    parser.add_argument('--news-years', type=int, default=None,
                       help='Years of news (defaults to same as --years)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for files (default: current dir)')
    parser.add_argument('--skip-steps', type=int, nargs='+', default=None,
                       help='Step numbers to skip (e.g., --skip-steps 1 2)')
    parser.add_argument('--force-rescrape', action='store_true',
                       help='Force rescraping even if files exist')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume fundamentals scraping from this ticker')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for news embedding (default: cuda)')
    parser.add_argument('--skip-cross-sectional', action='store_true',
                       help='Skip cross-sectional percentile features (faster, recommended)')
    parser.add_argument('--no-cross-sectional-norm', action='store_true',
                       help='Disable cross-sectional normalization (NOT recommended!)')
    parser.add_argument('--news-source', type=str, default='alphavantage',
                       choices=['gnews', 'alphavantage'],
                       help='News scraping source (default: alphavantage)')
    parser.add_argument('--av-api-key', type=str, default="placeholder",
                       help='Alpha Vantage API key (required if news-source=alphavantage)')
    parser.add_argument('--av-rate-limit', type=int, default=75,
                       help='Alpha Vantage rate limit (75 for Premium, 5 for Free, default: 75)')

    args = parser.parse_args()

    # Validate Alpha Vantage settings
    if args.news_source == 'alphavantage' and not args.av_api_key:
        parser.error("--av-api-key is required when --news-source=alphavantage")

    # Create generator
    generator = DatasetGenerator(
        dataset_name=args.dataset,
        api_key=args.api_key,
        years=args.years,
        news_years=args.news_years,
        output_dir=args.output_dir,
        news_source=args.news_source,
        av_api_key=args.av_api_key,
        av_rate_limit=args.av_rate_limit
    )

    # Run pipeline
    try:
        success = generator.run_all(
            skip_steps=args.skip_steps,
            force_rescrape=args.force_rescrape,
            resume_from=args.resume_from,
            device=args.device,
            skip_cross_sectional=args.skip_cross_sectional,
            use_cross_sectional_normalization=not args.no_cross_sectional_norm
        )

        if success:
            print("\n✅ Dataset generation successful!")
            sys.exit(0)
        else:
            print("\n❌ Dataset generation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print("  Progress has been saved where possible")
        print("  Resume with appropriate --skip-steps or --resume-from flags")
        sys.exit(130)


if __name__ == '__main__':
    main()
