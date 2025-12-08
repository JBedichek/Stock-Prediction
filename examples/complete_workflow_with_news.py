"""
Complete Workflow: Stock Prediction with News Embeddings

This script demonstrates the complete end-to-end workflow for:
1. Scraping market data
2. Scraping stock fundamentals
3. Scraping news articles
4. Embedding news with Nomic
5. Integrating everything into enhanced features

Final output: Daily feature tensors with 1000-1200+ features per stock
- 200-300 quarterly fundamentals
- 20-25 technical indicators
- 40-50 derived features
- 15-25 market-relative features
- 15-20 cross-sectional rankings
- 768 news embeddings
"""

import sys
sys.path.append('/home/james/Desktop/Stock-Prediction')

from data_scraping.market_indices_scraper import scrape_market_data
from data_scraping.fmp_comprehensive_scraper import scrape_dataset
from data_scraping.news_scraper import scrape_news_dataset
from data_scraping.news_embedder import create_news_embeddings
from data_scraping.fmp_enhanced_processor import process_enhanced_data
from data_scraping.Stock import s_lot_of_stocks
from datetime import datetime, timedelta


def complete_workflow(api_key: str, dataset_name: str = 's_lot',
                     scrape_years: int = 5,
                     news_years: int = None):
    """
    Run complete workflow from scratch.

    Args:
        api_key: FMP API key
        dataset_name: Dataset to process ('s_lot', 'a_lot', etc.)
        scrape_years: Years of historical stock data to scrape
        news_years: Years of news to scrape (defaults to same as scrape_years)
    """
    # Default: scrape news for same period as stock data
    if news_years is None:
        news_years = scrape_years

    print("\n" + "="*80)
    print("COMPLETE WORKFLOW: STOCK PREDICTION WITH NEWS EMBEDDINGS")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Historical data: {scrape_years} years")
    print(f"News data: {news_years} years")
    print("="*80 + "\n")

    # Dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    data_start_date = (datetime.now() - timedelta(days=scrape_years*365)).strftime('%Y-%m-%d')
    news_start_date = (datetime.now() - timedelta(days=news_years*365)).strftime('%Y-%m-%d')

    # Stock dictionary
    stock_dict = s_lot_of_stocks  # You can change this

    # ===========================================================================
    # STEP 1: Scrape Market Indices (~2 minutes, 15 API calls)
    # ===========================================================================

    print("\n" + "="*80)
    print("STEP 1: SCRAPING MARKET INDICES")
    print("="*80 + "\n")

    market_file = 'market_indices_data.pkl'

    scrape_market_data(
        api_key=api_key,
        from_date=data_start_date,
        to_date=end_date,
        output_file=market_file
    )

    print(f"\n✅ Market indices saved to {market_file}")

    # ===========================================================================
    # STEP 2: Scrape Stock Fundamentals (~30-60 min for s_lot, ~30 calls/stock)
    # ===========================================================================

    print("\n" + "="*80)
    print("STEP 2: SCRAPING STOCK FUNDAMENTALS")
    print("="*80 + "\n")

    fundamentals_file = f'{dataset_name}_fmp_comprehensive.pkl'

    scrape_dataset(
        stock_dict=stock_dict,
        api_key=api_key,
        output_file=fundamentals_file,
        start_date=data_start_date
    )

    print(f"\n✅ Fundamentals saved to {fundamentals_file}")

    # ===========================================================================
    # STEP 3: Scrape News Articles (time varies by source availability)
    # ===========================================================================

    print("\n" + "="*80)
    print("STEP 3: SCRAPING NEWS ARTICLES")
    print("="*80 + "\n")

    news_data_file = f'{dataset_name}_news_data.pkl'

    scrape_news_dataset(
        stock_dict=stock_dict,
        start_date=news_start_date,
        end_date=end_date,
        output_file=news_data_file,
        fmp_api_key=api_key  # Will try FMP news, but likely 403
    )

    print(f"\n✅ News articles saved to {news_data_file}")

    # ===========================================================================
    # STEP 4: Embed News with Nomic (~5-15 min depending on GPU)
    # ===========================================================================

    print("\n" + "="*80)
    print("STEP 4: EMBEDDING NEWS ARTICLES")
    print("="*80 + "\n")

    news_embeddings_file = f'{dataset_name}_news_embeddings.pkl'

    create_news_embeddings(
        news_data_file=news_data_file,
        output_file=news_embeddings_file,
        start_date=news_start_date,
        end_date=end_date,
        device='cuda'  # or 'cpu'
    )

    print(f"\n✅ News embeddings saved to {news_embeddings_file}")

    # ===========================================================================
    # STEP 5: Process Enhanced Features (~10-20 min)
    # ===========================================================================

    print("\n" + "="*80)
    print("STEP 5: PROCESSING ENHANCED FEATURES")
    print("="*80 + "\n")

    enhanced_file = f'{dataset_name}_enhanced_with_news.pkl'

    process_enhanced_data(
        raw_data_file=fundamentals_file,
        market_indices_file=market_file,
        sector_dict_file=f'{dataset_name}_sector_dict',  # From Stock.py
        news_embeddings_file=news_embeddings_file,
        output_file=enhanced_file,
        start_date=data_start_date,
        end_date=end_date,
        add_cross_sectional=True
    )

    print(f"\n✅ Enhanced features saved to {enhanced_file}")

    # ===========================================================================
    # SUMMARY
    # ===========================================================================

    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. Market indices: {market_file}")
    print(f"  2. Fundamentals: {fundamentals_file}")
    print(f"  3. News articles: {news_data_file}")
    print(f"  4. News embeddings: {news_embeddings_file}")
    print(f"  5. Enhanced features: {enhanced_file}")

    print(f"\nFinal feature count:")
    print(f"  - Quarterly fundamentals: 200-300")
    print(f"  - Technical indicators: 20-25")
    print(f"  - Derived features: 40-50")
    print(f"  - Market-relative: 15-25")
    print(f"  - Cross-sectional: 15-20")
    print(f"  - News embeddings: 768")
    print(f"  - TOTAL: ~1100-1200 features per day")

    print(f"\nReady for training!")
    print(f"Use {enhanced_file} in your dataset class.")
    print("="*80 + "\n")


def quick_test_workflow():
    """Quick test with minimal data."""
    print("\n" + "="*80)
    print("QUICK TEST WORKFLOW")
    print("="*80 + "\n")

    from data_scraping.Stock import test_stock_tickers

    # API key must be provided via command line argument
    import sys
    if len(sys.argv) < 2:
        print("ERROR: API key required. Run with --api_key YOUR_KEY")
        sys.exit(1)

    # This will be set from command line args
    api_key = None  # Set via argparse below

    complete_workflow(
        api_key=api_key or "YOUR_API_KEY_HERE",
        dataset_name='test',
        scrape_years=1,   # Just 1 year
        news_years=1      # Same 1 year of news
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Complete workflow with news embeddings')
    parser.add_argument('--api_key', type=str, required=True,
                       help='FMP API key (required)')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['test', 's_lot', 'a_lot'],
                       help='Dataset to process')
    parser.add_argument('--years', type=int, default=25,
                       help='Years of historical stock data')
    parser.add_argument('--news_years', type=int, default=None,
                       help='Years of news to scrape (defaults to same as --years)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with minimal data')

    args = parser.parse_args()

    if args.quick_test:
        quick_test_workflow()
    else:
        complete_workflow(
            api_key=args.api_key,
            dataset_name=args.dataset,
            scrape_years=args.years,
            news_years=args.news_years
        )
