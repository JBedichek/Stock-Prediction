"""
Test script for news scraping and embedding pipeline.

Tests:
1. News scraping from multiple sources
2. Nomic embedding generation
3. Daily aggregation
4. Integration with enhanced processor
"""

import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from data_scraping.news_scraper import NewsScraperMultiSource, scrape_news_dataset
from data_scraping.news_embedder import NomicNewsEmbedder, NewsAggregator, create_news_embeddings
from data_scraping.Stock import test_stock_tickers
from utils.utils import save_pickle, pic_load


def test_news_scraper():
    """Test news scraping."""
    print("\n" + "="*80)
    print("TEST 1: NEWS SCRAPING")
    print("="*80)

    # Test with one stock
    ticker = 'AAPL'
    company_name = 'Apple Inc.'

    scraper = NewsScraperMultiSource(rate_limit_delay=0.5)

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    print(f"\nScraping news for {ticker} ({company_name})")
    print(f"Date range: {start_date.date()} to {end_date.date()}")

    try:
        articles = scraper.scrape_all_sources(
            ticker=ticker,
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            use_google=True,
            use_fmp=False,  # FMP likely not available
            use_yahoo=True
        )

        print(f"\n✅ SUCCESS!")
        print(f"Total articles: {len(articles)}")

        if articles:
            print(f"\nSample article:")
            sample = articles[0]
            print(f"  Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"  Publisher: {sample.get('publisher', 'N/A')}")
            print(f"  Date: {sample.get('published_date', 'N/A')}")
            print(f"  Source: {sample.get('source', 'N/A')}")
            print(f"  Has full text: {bool(sample.get('full_text'))}")

        return articles

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_nomic_embedder():
    """Test Nomic embedding."""
    print("\n" + "="*80)
    print("TEST 2: NOMIC EMBEDDING")
    print("="*80)

    try:
        embedder = NomicNewsEmbedder(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Test embedding
        sample_texts = [
            "Apple Inc. announces record quarterly earnings.",
            "Microsoft releases new AI features in Windows.",
            "Tesla stock surges on delivery numbers."
        ]

        print(f"\nEmbedding {len(sample_texts)} sample texts...")

        embeddings = embedder.embed_texts(sample_texts)

        print(f"\n✅ SUCCESS!")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Expected: ({len(sample_texts)}, 768)")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"\nFirst embedding (first 10 values):")
        print(embeddings[0][:10])

        # Test article embedding
        sample_article = {
            'title': 'Apple announces new iPhone',
            'description': 'Apple unveils latest iPhone with improved camera',
            'full_text': 'In a major announcement today, Apple Inc. revealed their latest iPhone model featuring significant camera improvements and enhanced processing power.'
        }

        article_embedding = embedder.embed_article(sample_article)
        print(f"\nArticle embedding shape: {article_embedding.shape}")
        print(f"Expected: (768,)")

        return embedder

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_news_aggregation(articles, embedder):
    """Test news aggregation by date."""
    print("\n" + "="*80)
    print("TEST 3: NEWS AGGREGATION")
    print("="*80)

    if not articles or not embedder:
        print("⚠️  Skipping (no articles or embedder)")
        return None

    try:
        aggregator = NewsAggregator(embedder)

        print(f"\nAggregating {len(articles)} articles by date...")

        date_embeddings = aggregator.aggregate_by_date(articles, method='mean')

        print(f"\n✅ SUCCESS!")
        print(f"Unique dates with news: {len(date_embeddings)}")

        if date_embeddings:
            print(f"\nDates:")
            for date in sorted(date_embeddings.keys())[:5]:
                embedding = date_embeddings[date]
                print(f"  {date}: embedding shape {embedding.shape}")

        return date_embeddings

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline():
    """Test full pipeline from scraping to embeddings."""
    print("\n" + "="*80)
    print("TEST 4: FULL PIPELINE (Scrape + Embed)")
    print("="*80)

    # Use test tickers
    test_dict = {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation'}

    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nStep 1: Scraping news...")
    print(f"Stocks: {list(test_dict.keys())}")
    print(f"Date range: {start_date} to {end_date}")

    try:
        # Scrape news
        news_data = scrape_news_dataset(
            stock_dict=test_dict,
            start_date=start_date,
            end_date=end_date,
            output_file='test_news_data.pkl'
        )

        print(f"\nStep 2: Creating embeddings...")

        # Create embeddings
        embeddings = create_news_embeddings(
            news_data_file='test_news_data.pkl',
            output_file='test_news_embeddings.pkl',
            start_date=start_date,
            end_date=end_date
        )

        print(f"\n✅ FULL PIPELINE SUCCESS!")
        print(f"Stocks with embeddings: {len(embeddings)}")

        # Show sample
        for ticker, date_embeddings in list(embeddings.items())[:1]:
            print(f"\n{ticker}:")
            print(f"  Total dates: {len(date_embeddings)}")
            sample_date = list(date_embeddings.keys())[0]
            sample_embedding = date_embeddings[sample_date]
            print(f"  Sample date: {sample_date}")
            print(f"  Embedding shape: {sample_embedding.shape}")
            print(f"  Embedding type: {type(sample_embedding)}")

        return embeddings

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_integration_with_enhanced_processor():
    """Test integration with enhanced processor."""
    print("\n" + "="*80)
    print("TEST 5: INTEGRATION WITH ENHANCED PROCESSOR")
    print("="*80)

    print("\n⚠️  This test requires:")
    print("  1. Raw FMP data (from fmp_comprehensive_scraper)")
    print("  2. Market indices data")
    print("  3. News embeddings")

    print("\nExample command:")
    print("""
    python data_scraping/fmp_enhanced_processor.py \\
        --raw_data s_lot_fmp_comprehensive.pkl \\
        --market_indices market_indices_data.pkl \\
        --sector_dict s_lot_sector_dict.pkl \\
        --news_embeddings test_news_embeddings.pkl \\
        --output s_lot_with_news.pkl
    """)

    # If test files exist, try integration
    try:
        import os
        if os.path.exists('test_news_embeddings.pkl'):
            print("\n✅ Test news embeddings exist!")
            embeddings = pic_load('test_news_embeddings.pkl')
            print(f"   Stocks: {list(embeddings.keys())}")

            # Show tensor concatenation example
            if embeddings:
                ticker = list(embeddings.keys())[0]
                sample_date = list(embeddings[ticker].keys())[0]
                news_tensor = embeddings[ticker][sample_date]

                # Simulate base features (e.g., 300 features)
                base_features = torch.randn(300)

                # Concatenate
                combined = torch.cat([base_features, news_tensor])

                print(f"\n   Integration example ({ticker}, {sample_date}):")
                print(f"     Base features: {base_features.shape}")
                print(f"     News embedding: {news_tensor.shape}")
                print(f"     Combined: {combined.shape}")
                print(f"     Total features: {combined.shape[0]} (300 base + 768 news)")

    except Exception as e:
        print(f"\n⚠️  Could not test integration: {e}")


def main():
    print("\n" + "="*80)
    print("NEWS PIPELINE TEST SUITE")
    print("="*80)

    # Test 1: Scraping
    articles = test_news_scraper()

    # Test 2: Embedding
    embedder = test_nomic_embedder()

    # Test 3: Aggregation
    aggregated = test_news_aggregation(articles, embedder)

    # Test 4: Full pipeline
    full_pipeline_embeddings = test_full_pipeline()

    # Test 5: Integration
    test_integration_with_enhanced_processor()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
