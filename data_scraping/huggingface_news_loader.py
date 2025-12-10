"""
Hugging Face Financial News Dataset Loader

Loads pre-built historical news data from Hugging Face's Financial News Multisource dataset.
Much faster than web scraping (2 hours vs 30 hours for 370 stocks).

Dataset: https://huggingface.co/datasets/Brianferrell787/financial-news-multisource
Coverage: 57.1M articles, 1990-2025, 4,775+ companies
"""

from datasets import load_dataset
from typing import Dict, List, Optional
from datetime import datetime
import sys

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import save_pickle


def load_news_from_huggingface(stock_dict: Dict[str, str],
                                start_date: str = '2000-01-01',
                                end_date: str = '2025-01-01',
                                output_file: str = 'huggingface_news_data.pkl',
                                verify_coverage: bool = True) -> Dict[str, List[Dict]]:
    """
    Load news from Hugging Face Financial News Multisource dataset.

    Args:
        stock_dict: Dict of {ticker: company_name}
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        output_file: Output pickle file
        verify_coverage: Whether to check ticker coverage first

    Returns:
        Dict of {ticker: [articles]}
    """
    print(f"\n{'='*80}")
    print(f"LOADING HUGGING FACE NEWS DATASET")
    print(f"{'='*80}")
    print(f"Stocks requested: {len(stock_dict)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    # Load dataset
    print("📥 Downloading/loading Hugging Face dataset (this may take a few minutes)...")
    try:
        dataset = load_dataset("Brianferrell787/financial-news-multisource", split='train')
        print(f"✅ Dataset loaded: {len(dataset)} total articles\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure you have 'datasets' installed: pip install datasets")
        return {}

    # Verify coverage
    if verify_coverage:
        print("🔍 Checking ticker coverage...")
        dataset_tickers = set()
        for row in dataset:
            ticker = row.get('ticker')
            if ticker:
                dataset_tickers.add(ticker)

        requested_tickers = set(stock_dict.keys())
        covered_tickers = requested_tickers.intersection(dataset_tickers)

        print(f"   Dataset has: {len(dataset_tickers)} unique tickers")
        print(f"   You requested: {len(requested_tickers)} tickers")
        print(f"   Coverage: {len(covered_tickers)}/{len(requested_tickers)} stocks ({100*len(covered_tickers)/len(requested_tickers):.1f}%)")

        # Show missing tickers
        missing = requested_tickers - dataset_tickers
        if missing:
            print(f"\n   ⚠️  Missing {len(missing)} tickers:")
            if len(missing) <= 20:
                print(f"   {', '.join(sorted(missing))}")
            else:
                print(f"   {', '.join(sorted(list(missing)[:20]))}... and {len(missing)-20} more")

        if len(covered_tickers) == 0:
            print("\n❌ No ticker overlap! Check ticker format (dataset may use different symbols)")
            return {}

        print()

    # Filter by tickers and date range
    print("🔎 Filtering articles by tickers and date range...")
    tickers = set(stock_dict.keys())
    news_dict = {ticker: [] for ticker in tickers}

    article_count = 0
    skipped_count = 0

    from tqdm import tqdm
    for row in tqdm(dataset, desc="Processing articles"):
        ticker = row.get('ticker')

        # Check ticker
        if ticker not in tickers:
            skipped_count += 1
            continue

        # Check date
        date = row.get('date', '')
        if not date:
            continue

        # Parse date (handle various formats)
        try:
            if isinstance(date, str):
                # Try various date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        date_obj = datetime.strptime(date, fmt)
                        date = date_obj.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
        except Exception:
            continue

        # Date range filter
        if date < start_date or date > end_date:
            continue

        # Create article in your format
        article = {
            'ticker': ticker,
            'title': row.get('headline', ''),
            'description': row.get('description', row.get('text', '')),
            'url': row.get('url', ''),
            'published_date': date,
            'publisher': row.get('source', ''),
            'full_text': row.get('text', row.get('description', '')),
            'source': 'huggingface'
        }

        news_dict[ticker].append(article)
        article_count += 1

    print(f"\n✅ Filtered {article_count} articles")
    print(f"   Skipped {skipped_count} articles (different tickers)")

    # Statistics
    print(f"\n{'='*80}")
    print("ARTICLE STATISTICS")
    print(f"{'='*80}")

    stocks_with_news = sum(1 for articles in news_dict.values() if len(articles) > 0)
    total_articles = sum(len(articles) for articles in news_dict.values())
    avg_articles = total_articles / stocks_with_news if stocks_with_news > 0 else 0

    print(f"Stocks with news: {stocks_with_news}/{len(stock_dict)}")
    print(f"Total articles: {total_articles:,}")
    print(f"Average per stock: {avg_articles:.1f}")

    # Show top 10 stocks by article count
    top_stocks = sorted([(t, len(a)) for t, a in news_dict.items()], key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 stocks by article count:")
    for ticker, count in top_stocks:
        print(f"   {ticker}: {count:,} articles")

    # Save to pickle
    print(f"\n💾 Saving to {output_file}...")
    save_pickle(news_dict, output_file)

    print(f"\n{'='*80}")
    print("✅ HUGGING FACE NEWS LOADING COMPLETE!")
    print(f"{'='*80}\n")

    return news_dict


def verify_ticker_coverage(stock_dict: Dict[str, str]) -> None:
    """
    Quick check to see how many of your tickers are in the Hugging Face dataset.

    Args:
        stock_dict: Dict of {ticker: company_name}
    """
    print("Loading dataset to check coverage...")
    dataset = load_dataset("Brianferrell787/financial-news-multisource", split='train')

    print("Extracting unique tickers from dataset...")
    dataset_tickers = set()
    for row in dataset:
        ticker = row.get('ticker')
        if ticker:
            dataset_tickers.add(ticker)

    requested_tickers = set(stock_dict.keys())
    covered_tickers = requested_tickers.intersection(dataset_tickers)
    missing_tickers = requested_tickers - dataset_tickers

    print(f"\n{'='*80}")
    print("TICKER COVERAGE REPORT")
    print(f"{'='*80}")
    print(f"Dataset has: {len(dataset_tickers)} unique tickers")
    print(f"You requested: {len(requested_tickers)} tickers")
    print(f"Coverage: {len(covered_tickers)}/{len(requested_tickers)} stocks ({100*len(covered_tickers)/len(requested_tickers):.1f}%)")

    if covered_tickers:
        print(f"\n✅ Covered tickers ({len(covered_tickers)}):")
        print(f"   {', '.join(sorted(list(covered_tickers)[:20]))}")
        if len(covered_tickers) > 20:
            print(f"   ... and {len(covered_tickers)-20} more")

    if missing_tickers:
        print(f"\n⚠️  Missing tickers ({len(missing_tickers)}):")
        print(f"   {', '.join(sorted(list(missing_tickers)[:20]))}")
        if len(missing_tickers) > 20:
            print(f"   ... and {len(missing_tickers)-20} more")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    from data_scraping.Stock import s_lot_of_stocks, a_lot_of_stocks, test_stock_tickers

    parser = argparse.ArgumentParser(description='Load news from Hugging Face dataset')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['test', 's_lot', 'a_lot'],
                       help='Dataset to load')
    parser.add_argument('--start_date', type=str, default='2000-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-01-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify ticker coverage, do not load news')

    args = parser.parse_args()

    # Select dataset
    if args.dataset == 'test':
        stock_dict = test_stock_tickers
    elif args.dataset == 's_lot':
        stock_dict = s_lot_of_stocks
    elif args.dataset == 'a_lot':
        stock_dict = a_lot_of_stocks

    # Set output file
    if args.output is None:
        args.output = f"{args.dataset}_huggingface_news.pkl"

    # Verify coverage only
    if args.verify_only:
        verify_ticker_coverage(stock_dict)
    else:
        # Load news
        news_data = load_news_from_huggingface(
            stock_dict=stock_dict,
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=args.output
        )

        print(f"\n✅ Done! News data saved to {args.output}")
