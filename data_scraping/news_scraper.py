"""
News Scraper for Stocks

Scrapes news articles from multiple sources:
1. Google News (via gnews)
2. FMP API (if available)
3. Yahoo Finance RSS feeds

Collects news headlines and full text for embedding.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import sys
from gnews import GNews
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import save_pickle, pic_load


class NewsScraperMultiSource:
    """
    Multi-source news scraper for stocks.
    """

    def __init__(self, fmp_api_key: Optional[str] = None, rate_limit_delay: float = 1.0,
                 max_results_per_query: int = 100, fetch_full_articles: bool = True):
        """
        Initialize scraper.

        Args:
            fmp_api_key: Optional FMP API key
            rate_limit_delay: Delay between requests (seconds)
            max_results_per_query: Max results per query (will chunk dates if needed)
            fetch_full_articles: Whether to fetch full article text (SLOW - adds 25 hours per stock!)
        """
        self.fmp_api_key = fmp_api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_results_per_query = max_results_per_query
        self.fetch_full_articles = fetch_full_articles
        self.gnews_client = GNews(language='en', country='US', max_results=max_results_per_query)

    def scrape_google_news(self, ticker: str, company_name: str,
                          start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Scrape news from Google News with date chunking for comprehensive coverage.

        Args:
            ticker: Stock ticker
            company_name: Company name
            start_date: Start date
            end_date: End date

        Returns:
            List of news articles
        """
        articles = []

        try:
            # For long date ranges, chunk into monthly periods to get comprehensive coverage
            total_days = (end_date - start_date).days

            if total_days > 90:  # More than 3 months - chunk into monthly periods
                # Calculate total months for progress tracking
                total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
                month_count = 0

                current_start = start_date
                while current_start < end_date:
                    month_count += 1

                    # Get next month boundary
                    if current_start.month == 12:
                        next_month = datetime(current_start.year + 1, 1, 1)
                    else:
                        next_month = datetime(current_start.year, current_start.month + 1, 1)

                    current_end = min(next_month - timedelta(days=1), end_date)

                    # Query this month
                    self.gnews_client.start_date = current_start
                    self.gnews_client.end_date = current_end

                    search_query = f"{company_name} stock"

                    # Try to get news with timeout handling
                    try:
                        month_results = self.gnews_client.get_news(search_query)
                    except (TimeoutError, ConnectionError, Exception) as query_error:
                        # If query fails, skip this month and continue
                        print(f"      ⚠️  Failed to query {current_start.strftime('%Y-%m')}: {type(query_error).__name__}")
                        current_start = next_month
                        continue

                    if month_count % 12 == 0 or month_count == total_months:  # Print every year or at end
                        print(f"      Queried {month_count}/{total_months} months, {len(articles)} articles so far")

                    # Process articles from this month
                    for article in month_results:
                        try:
                            # Get full article text (optional - SLOW!)
                            if self.fetch_full_articles:
                                full_text = ''
                                try:
                                    full_article = self.gnews_client.get_full_article(article['url'])
                                    full_text = full_article.text if full_article else ''
                                    time.sleep(self.rate_limit_delay)  # Only delay if fetching
                                except (TimeoutError, ConnectionError, Exception) as article_error:
                                    # If full article fetch fails, use description instead
                                    full_text = article.get('description', '')
                            else:
                                # Fast mode: just use title + description (no HTTP requests)
                                full_text = article.get('description', '')

                            article_data = {
                                'ticker': ticker,
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'published_date': article.get('published date', ''),
                                'publisher': article.get('publisher', {}).get('title', ''),
                                'full_text': full_text,
                                'source': 'google_news'
                            }

                            articles.append(article_data)

                        except Exception as e:
                            # Skip articles that completely fail to process
                            continue

                    # Move to next month
                    current_start = next_month

            else:
                # Short date range - single query
                self.gnews_client.start_date = start_date
                self.gnews_client.end_date = end_date

                # Search by company name
                search_query = f"{company_name} stock"

                # Try to get news with timeout handling
                try:
                    results = self.gnews_client.get_news(search_query)
                except (TimeoutError, ConnectionError, Exception) as query_error:
                    # If query fails, return empty list
                    print(f"      ⚠️  Failed to query: {type(query_error).__name__}")
                    return articles

                for article in results:
                    try:
                        # Get full article text (optional - SLOW!)
                        if self.fetch_full_articles:
                            full_text = ''
                            try:
                                full_article = self.gnews_client.get_full_article(article['url'])
                                full_text = full_article.text if full_article else ''
                                time.sleep(self.rate_limit_delay)  # Only delay if fetching
                            except (TimeoutError, ConnectionError, Exception) as article_error:
                                # If full article fetch fails, use description instead
                                full_text = article.get('description', '')
                        else:
                            # Fast mode: just use title + description (no HTTP requests)
                            full_text = article.get('description', '')

                        article_data = {
                            'ticker': ticker,
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'published_date': article.get('published date', ''),
                            'publisher': article.get('publisher', {}).get('title', ''),
                            'full_text': full_text,
                            'source': 'google_news'
                        }

                        articles.append(article_data)

                    except Exception as e:
                        # Skip articles that completely fail to process
                        continue

        except Exception as e:
            print(f"  ⚠️  Google News error for {ticker}: {e}")

        return articles

    def scrape_fmp_news(self, ticker: str, limit: int = 10000) -> List[Dict]:
        """
        Scrape news from FMP API (if available).

        Args:
            ticker: Stock ticker
            limit: Number of articles to fetch

        Returns:
            List of news articles
        """
        if not self.fmp_api_key:
            return []

        articles = []

        try:
            url = f"https://financialmodelingprep.com/stable/stock_news"
            params = {
                'tickers': ticker,
                'limit': limit,
                'apikey': self.fmp_api_key
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()

                for item in data:
                    article_data = {
                        'ticker': ticker,
                        'title': item.get('title', ''),
                        'description': item.get('text', ''),
                        'url': item.get('url', ''),
                        'published_date': item.get('publishedDate', ''),
                        'publisher': item.get('site', ''),
                        'full_text': item.get('text', ''),  # FMP provides summary
                        'source': 'fmp'
                    }
                    articles.append(article_data)

        except Exception as e:
            print(f"  ⚠️  FMP News error for {ticker}: {e}")

        return articles

    def scrape_yahoo_finance_rss(self, ticker: str) -> List[Dict]:
        """
        Scrape news from Yahoo Finance RSS feed.

        Args:
            ticker: Stock ticker

        Returns:
            List of news articles
        """
        articles = []

        try:
            import feedparser

            # Yahoo Finance RSS feed
            rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            feed = feedparser.parse(rss_url)

            for entry in feed.entries:
                article_data = {
                    'ticker': ticker,
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'url': entry.get('link', ''),
                    'published_date': entry.get('published', ''),
                    'publisher': 'Yahoo Finance',
                    'full_text': entry.get('summary', ''),
                    'source': 'yahoo_rss'
                }
                articles.append(article_data)

        except Exception as e:
            print(f"  ⚠️  Yahoo RSS error for {ticker}: {e}")

        return articles

    def scrape_all_sources(self, ticker: str, company_name: str,
                          start_date: datetime, end_date: datetime,
                          use_google: bool = True,
                          use_fmp: bool = True,
                          use_yahoo: bool = True) -> List[Dict]:
        """
        Scrape news from all available sources.

        Args:
            ticker: Stock ticker
            company_name: Company name
            start_date: Start date
            end_date: End date
            use_google: Use Google News
            use_fmp: Use FMP API
            use_yahoo: Use Yahoo Finance RSS

        Returns:
            Combined list of articles from all sources
        """
        all_articles = []

        if use_google:
            print(f"    📰 Google News...")
            google_articles = self.scrape_google_news(ticker, company_name, start_date, end_date)
            all_articles.extend(google_articles)
            print(f"      Found {len(google_articles)} articles")

        if use_fmp and self.fmp_api_key:
            print(f"    📰 FMP News...")
            fmp_articles = self.scrape_fmp_news(ticker)
            all_articles.extend(fmp_articles)
            print(f"      Found {len(fmp_articles)} articles")

        if use_yahoo:
            print(f"    📰 Yahoo Finance RSS...")
            yahoo_articles = self.scrape_yahoo_finance_rss(ticker)
            all_articles.extend(yahoo_articles)
            print(f"      Found {len(yahoo_articles)} articles")

        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)

        return unique_articles


def scrape_news_dataset(stock_dict: Dict[str, str],
                       start_date: str,
                       end_date: str = None,
                       output_file: str = 'stock_news_data.pkl',
                       fmp_api_key: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    Scrape news for a dataset of stocks.

    Args:
        stock_dict: Dict of {ticker: company_name}
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (defaults to today)
        output_file: Output pickle file
        fmp_api_key: Optional FMP API key

    Returns:
        Dict of {ticker: [articles]}
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"SCRAPING NEWS DATA")
    print(f"{'='*80}")
    print(f"Stocks: {len(stock_dict)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    scraper = NewsScraperMultiSource(fmp_api_key=fmp_api_key, rate_limit_delay=1.0)

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
            articles = scraper.scrape_all_sources(
                ticker=ticker,
                company_name=company_name,
                start_date=start_dt,
                end_date=end_dt,
                use_google=True,
                use_fmp=(fmp_api_key is not None),
                use_yahoo=True
            )

            all_news[ticker] = articles
            print(f"  ✅ Total: {len(articles)} unique articles")

            # Save progress every 10 stocks
            if (i + 1) % 10 == 0:
                print(f"\n💾 Saving progress...")
                save_pickle(all_news, output_file)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Final save
    print(f"\n💾 Saving final data...")
    save_pickle(all_news, output_file)

    print(f"\n{'='*80}")
    print(f"✅ NEWS SCRAPING COMPLETE!")
    print(f"{'='*80}")
    print(f"Stocks with news: {len(all_news)}")

    # Statistics
    total_articles = sum(len(articles) for articles in all_news.values())
    avg_articles = total_articles / len(all_news) if all_news else 0

    print(f"Total articles: {total_articles}")
    print(f"Average per stock: {avg_articles:.1f}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    return all_news


if __name__ == '__main__':
    import argparse
    from data_scraping.Stock import s_lot_of_stocks, a_lot_of_stocks, test_stock_tickers

    parser = argparse.ArgumentParser(description='Scrape news for stocks')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['test', 's_lot', 'a_lot'],
                       help='Dataset to scrape')
    parser.add_argument('--start_date', type=str,
                       default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (defaults to today)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file')
    parser.add_argument('--fmp_key', type=str, default=None,
                       help='FMP API key (optional)')

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
        args.output = f"{args.dataset}_news_data.pkl"

    # Scrape
    news_data = scrape_news_dataset(
        stock_dict=stock_dict,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output,
        fmp_api_key=args.fmp_key
    )

    print(f"\n✅ Done! News data saved to {args.output}")
