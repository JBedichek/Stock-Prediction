"""
Alpha Vantage News & Sentiments API Scraper

Fast, reliable news scraping with AI sentiment analysis included.
- 15+ years of historical news
- 75 calls/minute (Premium tier)
- Sentiment scores included
- ~5 minutes for 370 stocks

API Docs: https://www.alphavantage.co/documentation/#news-sentiment
Pricing: https://www.alphavantage.co/premium/
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import save_pickle


class AlphaVantageNewsScraper:
    """
    News scraper using Alpha Vantage News & Sentiments API.
    """

    def __init__(self, api_key: str, rate_limit_calls_per_minute: int = 75):
        """
        Initialize Alpha Vantage scraper.

        Args:
            api_key: Alpha Vantage API key
            rate_limit_calls_per_minute: API rate limit (75 for Premium, 25 for Free)
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

        # Calculate delay between requests
        self.delay = 60.0 / rate_limit_calls_per_minute  # seconds

    def scrape_news(self, ticker: str,
                   start_date: str = '20000101T0000',
                   end_date: str = None,
                   limit: int = 1000,
                   topics: Optional[List[str]] = None) -> List[Dict]:
        """
        Scrape news for a single ticker.

        Args:
            ticker: Stock ticker
            start_date: Start date in format 'YYYYMMDDTHHMM' (e.g., '20000101T0000')
            end_date: End date in format 'YYYYMMDDTHHMM' (defaults to now)
            limit: Max articles to fetch (1-1000)
            topics: Optional list of topics (e.g., ['earnings', 'ipo'])

        Returns:
            List of articles with sentiment scores
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%dT%H%M')

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'time_from': start_date,
            'time_to': end_date,
            'limit': min(limit, 1000),  # API max is 1000
            'apikey': self.api_key,
            'sort': 'EARLIEST'  # Get chronological order
        }

        if topics:
            params['topics'] = ','.join(topics)

        articles = []

        try:
            response = requests.get(self.base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Check for API errors
                if 'Note' in data:
                    print(f"      ⚠️  Rate limit hit: {data['Note']}")
                    return articles

                if 'Error Message' in data:
                    print(f"      ⚠️  API error: {data['Error Message']}")
                    return articles

                # Process feed items
                feed = data.get('feed', [])

                for item in feed:
                    # Get ticker-specific sentiment
                    ticker_sentiment = None
                    for ts in item.get('ticker_sentiment', []):
                        if ts.get('ticker') == ticker:
                            ticker_sentiment = ts
                            break

                    # Extract sentiment scores
                    overall_sentiment = item.get('overall_sentiment_score', 0.0)
                    overall_label = item.get('overall_sentiment_label', 'Neutral')

                    ticker_sentiment_score = None
                    ticker_sentiment_label = None
                    ticker_relevance = None

                    if ticker_sentiment:
                        ticker_sentiment_score = ticker_sentiment.get('ticker_sentiment_score', 0.0)
                        ticker_sentiment_label = ticker_sentiment.get('ticker_sentiment_label', 'Neutral')
                        ticker_relevance = ticker_sentiment.get('relevance_score', 0.0)

                    # Parse date
                    published_date = item.get('time_published', '')
                    try:
                        # Convert from '20230115T153000' to '2023-01-15'
                        dt = datetime.strptime(published_date[:8], '%Y%m%d')
                        published_date = dt.strftime('%Y-%m-%d')
                    except:
                        published_date = ''

                    article = {
                        'ticker': ticker,
                        'title': item.get('title', ''),
                        'description': item.get('summary', ''),
                        'url': item.get('url', ''),
                        'published_date': published_date,
                        'publisher': ', '.join(item.get('authors', [])) if item.get('authors') else item.get('source', ''),
                        'full_text': item.get('summary', ''),  # Alpha Vantage provides summaries
                        'source': 'alphavantage',

                        # Sentiment features (bonus!)
                        'overall_sentiment_score': overall_sentiment,
                        'overall_sentiment_label': overall_label,
                        'ticker_sentiment_score': ticker_sentiment_score,
                        'ticker_sentiment_label': ticker_sentiment_label,
                        'relevance_score': ticker_relevance,
                        'topics': item.get('topics', [])
                    }

                    articles.append(article)

            else:
                print(f"      ⚠️  HTTP {response.status_code}")

        except Exception as e:
            print(f"      ⚠️  Error: {type(e).__name__}: {e}")

        # Rate limiting
        time.sleep(self.delay)

        return articles

    def scrape_news_chunked(self, ticker: str,
                           start_date: str = '20000101T0000',
                           end_date: str = None,
                           chunk_years: int = 5,
                           verbose: bool = True) -> tuple[List[Dict], Dict]:
        """
        Scrape news for a single ticker with date chunking to bypass 1000 article limit.

        Args:
            ticker: Stock ticker
            start_date: Start date in format 'YYYYMMDDTHHMM'
            end_date: End date in format 'YYYYMMDDTHHMM' (defaults to now)
            chunk_years: Years per chunk (default: 5 years = ~1000 articles)
            verbose: Print detailed logging (default: True)

        Returns:
            Tuple of (articles list, stats dict)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%dT%H%M')

        # Parse dates
        start_dt = datetime.strptime(start_date[:8], '%Y%m%d')
        end_dt = datetime.strptime(end_date[:8], '%Y%m%d')

        # Create chunks
        chunks = []
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=365*chunk_years), end_dt)
            chunks.append((
                current_start.strftime('%Y%m%dT0000'),
                current_end.strftime('%Y%m%dT2359')
            ))
            current_start = current_end + timedelta(days=1)

        # Scrape each chunk
        all_articles = []
        chunk_stats = []

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_start_date = datetime.strptime(chunk_start[:8], '%Y%m%d').strftime('%Y-%m-%d')
            chunk_end_date = datetime.strptime(chunk_end[:8], '%Y%m%d').strftime('%Y-%m-%d')

            if verbose:
                print(f"     Chunk {i+1}/{len(chunks)}: {chunk_start_date} to {chunk_end_date}", end='')

            articles = self.scrape_news(
                ticker=ticker,
                start_date=chunk_start,
                end_date=chunk_end,
                limit=1000
            )

            # Calculate chunk statistics
            chunk_days = (datetime.strptime(chunk_end_date, '%Y-%m-%d') -
                         datetime.strptime(chunk_start_date, '%Y-%m-%d')).days

            unique_dates = set()
            for article in articles:
                if article.get('published_date'):
                    unique_dates.add(article['published_date'])

            days_with_articles = len(unique_dates)
            coverage_pct = (days_with_articles / chunk_days * 100) if chunk_days > 0 else 0

            chunk_stats.append({
                'period': f"{chunk_start_date} to {chunk_end_date}",
                'articles': len(articles),
                'days_with_articles': days_with_articles,
                'total_days': chunk_days,
                'coverage_pct': coverage_pct
            })

            if verbose:
                print(f" → {len(articles)} articles ({days_with_articles} days, {coverage_pct:.1f}% coverage)")

            all_articles.extend(articles)

        # Remove duplicates (in case there's overlap)
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
            elif not url:
                unique_articles.append(article)

        # Overall statistics
        total_days = (end_dt - start_dt).days
        unique_dates_overall = set()
        for article in unique_articles:
            if article.get('published_date'):
                unique_dates_overall.add(article['published_date'])

        stats = {
            'chunks': chunk_stats,
            'total_articles': len(unique_articles),
            'total_days': total_days,
            'days_with_articles': len(unique_dates_overall),
            'coverage_pct': (len(unique_dates_overall) / total_days * 100) if total_days > 0 else 0
        }

        return unique_articles, stats


def scrape_news_dataset(stock_dict: Dict[str, str],
                       start_date: str = '2000-01-01',
                       end_date: str = None,
                       output_file: str = 'alphavantage_news_data.pkl',
                       api_key: str = None,
                       rate_limit: int = 75,
                       use_chunking: bool = True,
                       chunk_years: int = 5) -> Dict[str, List[Dict]]:
    """
    Scrape news for a dataset of stocks using Alpha Vantage.

    Args:
        stock_dict: Dict of {ticker: company_name}
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (defaults to today)
        output_file: Output pickle file
        api_key: Alpha Vantage API key
        rate_limit: Calls per minute (75 for Premium, 25 for Free)
        use_chunking: Use date chunking to get >1000 articles (default: True)
        chunk_years: Years per chunk when using chunking (default: 5)

    Returns:
        Dict of {ticker: [articles]}
    """
    if not api_key:
        raise ValueError("Alpha Vantage API key required! Get one at: https://www.alphavantage.co/support/#api-key")

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Convert dates to Alpha Vantage format
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    av_start = start_dt.strftime('%Y%m%dT0000')
    av_end = end_dt.strftime('%Y%m%dT2359')

    # Calculate total API calls
    if use_chunking:
        years_total = (end_dt - start_dt).days / 365.25
        chunks_per_stock = max(1, int(years_total / chunk_years) + 1)
        total_calls = len(stock_dict) * chunks_per_stock
    else:
        total_calls = len(stock_dict)

    print(f"\n{'='*80}")
    print(f"ALPHA VANTAGE NEWS SCRAPING")
    print(f"{'='*80}")
    print(f"Stocks: {len(stock_dict)}")
    print(f"Date range: {start_date} to {end_date} ({(end_dt - start_dt).days / 365.25:.1f} years)")
    print(f"Chunking: {'Enabled' if use_chunking else 'Disabled'}")
    if use_chunking:
        print(f"Chunk size: {chunk_years} years")
        print(f"Chunks per stock: {chunks_per_stock}")
        print(f"Total API calls: {total_calls}")
    print(f"Rate limit: {rate_limit} calls/minute")
    print(f"Estimated time: {total_calls * (60/rate_limit) / 60:.1f} minutes")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    scraper = AlphaVantageNewsScraper(api_key=api_key, rate_limit_calls_per_minute=rate_limit)

    # Check for existing progress and auto-resume
    import os
    all_news = {}
    if os.path.exists(output_file):
        try:
            from utils.utils import pic_load
            all_news = pic_load(output_file)
            print(f"📂 Found existing progress: {len(all_news)}/{len(stock_dict)} stocks completed")

            if len(all_news) > 0:
                last_ticker = list(all_news.keys())[-1]
                print(f"   Last completed: {last_ticker}")
                print(f"   Resuming from next stock...")
        except Exception as e:
            print(f"⚠️  Could not load existing file: {e}")
            print(f"   Starting fresh...")
            all_news = {}

    from tqdm import tqdm
    for i, (ticker, company_name) in enumerate(tqdm(stock_dict.items(), desc="Scraping news")):
        # Skip already completed stocks (but retry if 0 articles)
        if ticker in all_news and len(all_news[ticker]) > 0:
            print(f"\n[{i+1}/{len(stock_dict)}] {ticker} - {company_name}")
            print(f"  ⏭️  Already scraped ({len(all_news[ticker])} articles), skipping...")
            continue

        if ticker in all_news and len(all_news[ticker]) == 0:
            print(f"\n[{i+1}/{len(stock_dict)}] {ticker} - {company_name}")
            print(f"  🔄 Retrying (previous attempt: 0 articles)...")
        else:
            print(f"\n[{i+1}/{len(stock_dict)}] {ticker} - {company_name}")

        try:
            if use_chunking:
                articles, stats = scraper.scrape_news_chunked(
                    ticker=ticker,
                    start_date=av_start,
                    end_date=av_end,
                    chunk_years=chunk_years,
                    verbose=True
                )
            else:
                articles = scraper.scrape_news(
                    ticker=ticker,
                    start_date=av_start,
                    end_date=av_end,
                    limit=1000  # Max per request
                )
                # Create simple stats for non-chunked mode
                unique_dates = set()
                for article in articles:
                    if article.get('published_date'):
                        unique_dates.add(article['published_date'])

                total_days = (end_dt - start_dt).days
                stats = {
                    'total_articles': len(articles),
                    'total_days': total_days,
                    'days_with_articles': len(unique_dates),
                    'coverage_pct': (len(unique_dates) / total_days * 100) if total_days > 0 else 0
                }

            all_news[ticker] = articles

            # Display coverage stats
            print(f"  ✅ Total: {len(articles)} articles")
            print(f"     Coverage: {stats['days_with_articles']}/{stats['total_days']} days ({stats['coverage_pct']:.1f}%)")

            # Show sentiment stats if available
            if articles:
                sentiments = [a.get('overall_sentiment_score', 0) for a in articles if a.get('overall_sentiment_score') is not None]
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    print(f"     Average sentiment: {avg_sentiment:.3f} ({'Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Neutral'})")

            # Save progress every 10 stocks
            if (i + 1) % 10 == 0:
                print(f"\n💾 Saving progress...")
                save_pickle(all_news, output_file)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_news[ticker] = []
            continue

    # Final save
    print(f"\n💾 Saving final data...")
    save_pickle(all_news, output_file)

    # Statistics
    print(f"\n{'='*80}")
    print(f"✅ NEWS SCRAPING COMPLETE!")
    print(f"{'='*80}")

    stocks_with_news = sum(1 for articles in all_news.values() if len(articles) > 0)
    total_articles = sum(len(articles) for articles in all_news.values())
    avg_articles = total_articles / stocks_with_news if stocks_with_news > 0 else 0

    print(f"Stocks with news: {stocks_with_news}/{len(stock_dict)}")
    print(f"Total articles: {total_articles:,}")
    print(f"Average per stock: {avg_articles:.1f}")

    # Sentiment statistics
    all_sentiments = []
    for articles in all_news.values():
        for article in articles:
            if article.get('overall_sentiment_score') is not None:
                all_sentiments.append(article['overall_sentiment_score'])

    if all_sentiments:
        avg_sentiment = sum(all_sentiments) / len(all_sentiments)
        positive = sum(1 for s in all_sentiments if s > 0.05)
        negative = sum(1 for s in all_sentiments if s < -0.05)
        neutral = len(all_sentiments) - positive - negative

        print(f"\nSentiment Analysis:")
        print(f"  Average sentiment: {avg_sentiment:.3f}")
        print(f"  Positive: {positive} ({100*positive/len(all_sentiments):.1f}%)")
        print(f"  Neutral: {neutral} ({100*neutral/len(all_sentiments):.1f}%)")
        print(f"  Negative: {negative} ({100*negative/len(all_sentiments):.1f}%)")

    print(f"\nOutput: {output_file}")
    print(f"{'='*80}\n")

    return all_news


if __name__ == '__main__':
    import argparse
    from data_scraping.Stock import s_lot_of_stocks, a_lot_of_stocks, test_stock_tickers

    parser = argparse.ArgumentParser(description='Scrape news using Alpha Vantage API')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['test', 's_lot', 'a_lot'],
                       help='Dataset to scrape')
    parser.add_argument('--start_date', type=str, default='2010-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (defaults to today)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file')
    parser.add_argument('--api_key', type=str, required=True,
                       help='Alpha Vantage API key (get at https://www.alphavantage.co/support/#api-key)')
    parser.add_argument('--rate_limit', type=int, default=75,
                       help='API calls per minute (75 for Premium, 25 for Free tier)')
    parser.add_argument('--no-chunking', action='store_true',
                       help='Disable date chunking (limited to 1000 articles per stock)')
    parser.add_argument('--chunk_years', type=int, default=5,
                       help='Years per chunk when using chunking (default: 5)')

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
        args.output = f"{args.dataset}_alphavantage_news.pkl"

    # Scrape
    news_data = scrape_news_dataset(
        stock_dict=stock_dict,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        use_chunking=not args.no_chunking,
        chunk_years=args.chunk_years
    )

    print(f"\n✅ Done! News data saved to {args.output}")
