"""
Test script for comprehensive FMP API scraper.

This tests the FMP API connection and scrapes a small dataset to verify everything works.
"""

import sys
sys.path.append('/home/james/Desktop/Stock-Prediction')

from data_scraping.fmp_comprehensive_scraper import FMPComprehensiveScraper, scrape_dataset
from data_scraping.fmp_data_processor import FMPDataProcessor, process_comprehensive_data
from utils.utils import save_pickle, pic_load


# ============================================================================
# SET YOUR API KEY HERE
# ============================================================================
API_KEY = "YOUR_FMP_API_KEY_HERE"  # ⚠️ Replace with your actual FMP API key
# ============================================================================


def test_api_connection():
    """Test basic API connection."""
    print("\n" + "="*80)
    print("TEST 1: API CONNECTION")
    print("="*80)

    scraper = FMPComprehensiveScraper(API_KEY)

    # Test simple endpoint
    print("\nTesting company profile for AAPL...")
    profile = scraper.get_company_profile('AAPL')

    if profile:
        print("  ✅ Success! Company profile:")
        print(f"     Company: {profile.get('companyName')}")
        print(f"     Sector: {profile.get('sector')}")
        print(f"     Industry: {profile.get('industry')}")
        print(f"     Market Cap: ${profile.get('mktCap', 0) / 1e9:.2f}B")
        return True
    else:
        print("  ❌ Failed to retrieve company profile")
        print("  Check your API key!")
        return False


def test_financial_statements():
    """Test financial statement retrieval."""
    print("\n" + "="*80)
    print("TEST 2: FINANCIAL STATEMENTS")
    print("="*80)

    scraper = FMPComprehensiveScraper(API_KEY)

    print("\nFetching financial statements for AAPL...")
    statements = scraper.get_financial_statements('AAPL', limit=20)

    if statements and 'income' in statements:
        income_df = statements['income']
        print(f"  ✅ Income Statement: {len(income_df)} periods")
        if not income_df.empty:
            latest = income_df.iloc[0]
            print(f"     Latest date: {latest.get('date')}")
            print(f"     Revenue: ${latest.get('revenue', 0) / 1e9:.2f}B")

    if statements and 'balance' in statements:
        balance_df = statements['balance']
        print(f"  ✅ Balance Sheet: {len(balance_df)} periods")

    if statements and 'cashflow' in statements:
        cashflow_df = statements['cashflow']
        print(f"  ✅ Cash Flow: {len(cashflow_df)} periods")

    return bool(statements)


def test_price_data():
    """Test daily price data retrieval."""
    print("\n" + "="*80)
    print("TEST 3: PRICE DATA")
    print("="*80)

    scraper = FMPComprehensiveScraper(API_KEY)

    print("\nFetching daily prices for AAPL (last 5 years)...")
    prices = scraper.get_daily_prices_full('AAPL', from_date='2020-01-01')

    if prices is not None and not prices.empty:
        print(f"  ✅ Retrieved {len(prices)} days of price data")
        print(f"     Date range: {prices['date'].min()} to {prices['date'].max()}")
        latest = prices.iloc[0]
        print(f"     Latest close: ${latest.get('close', 0):.2f}")
        return True
    else:
        print("  ❌ Failed to retrieve price data")
        return False


def test_technical_indicators():
    """Test technical indicator retrieval."""
    print("\n" + "="*80)
    print("TEST 4: TECHNICAL INDICATORS")
    print("="*80)

    scraper = FMPComprehensiveScraper(API_KEY)

    print("\nFetching RSI for AAPL...")
    rsi = scraper.get_technical_indicator('AAPL', 'daily', 'rsi', period=14)

    if rsi is not None and not rsi.empty:
        print(f"  ✅ Retrieved {len(rsi)} days of RSI data")
        return True
    else:
        print("  ❌ Failed to retrieve RSI")
        return False


def test_comprehensive_scrape():
    """Test comprehensive scraping for a small dataset."""
    print("\n" + "="*80)
    print("TEST 5: COMPREHENSIVE SCRAPING (Small Dataset)")
    print("="*80)

    test_stocks = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corp.',
        'GOOGL': 'Alphabet Inc.'
    }

    print(f"\nScraping {len(test_stocks)} stocks...")
    print("This will test all endpoints...\n")

    data = scrape_dataset(
        stock_dict=test_stocks,
        api_key=API_KEY,
        output_file='test_fmp_comprehensive.pkl',
        start_date='2020-01-01'
    )

    if data and len(data) > 0:
        print(f"\n  ✅ Successfully scraped {len(data)} stocks")

        # Show what was collected
        sample_ticker = list(data.keys())[0]
        sample_data = data[sample_ticker]

        print(f"\n  Sample data for {sample_ticker}:")
        print(f"    API calls: {sample_data.get('api_calls', 0)}")

        # Check each data source
        data_sources = [
            'financial_statements', 'key_metrics', 'financial_ratios',
            'daily_prices', 'technical_indicators', 'earnings_history',
            'insider_trading', 'analyst_ratings', 'company_profile'
        ]

        for source in data_sources:
            if source in sample_data and sample_data[source] is not None:
                if isinstance(sample_data[source], dict):
                    print(f"    ✓ {source}: {len(sample_data[source])} items")
                elif hasattr(sample_data[source], '__len__'):
                    print(f"    ✓ {source}: {len(sample_data[source])} records")
                else:
                    print(f"    ✓ {source}: available")

        return True
    else:
        print("  ❌ Comprehensive scrape failed")
        return False


def test_data_processing():
    """Test data processing pipeline."""
    print("\n" + "="*80)
    print("TEST 6: DATA PROCESSING")
    print("="*80)

    # Check if test data exists
    try:
        raw_data_all = pic_load('test_fmp_comprehensive.pkl')
    except:
        print("  ⚠️  No test data found. Run test_comprehensive_scrape first.")
        return False

    print(f"\nProcessing {len(raw_data_all)} stocks...")

    processed = process_comprehensive_data(
        raw_data_file='test_fmp_comprehensive.pkl',
        output_file='test_fmp_comprehensive_processed.pkl',
        start_date='2020-01-01'
    )

    if processed and len(processed) > 0:
        print(f"\n  ✅ Successfully processed {len(processed)} stocks")

        # Show feature statistics
        sample_ticker = list(processed.keys())[0]
        sample_dates = list(processed[sample_ticker].keys())

        if sample_dates:
            sample_tensor = processed[sample_ticker][sample_dates[0]]
            print(f"\n  Feature statistics:")
            print(f"    Ticker: {sample_ticker}")
            print(f"    Date range: {min(sample_dates)} to {max(sample_dates)}")
            print(f"    Total days: {len(sample_dates)}")
            print(f"    Features per day: {sample_tensor.shape[0]}")
            print(f"    Tensor shape: {sample_tensor.shape}")

        return True
    else:
        print("  ❌ Data processing failed")
        return False


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*80)
    print("FMP COMPREHENSIVE API TESTER")
    print("="*80)
    print(f"API Key: {API_KEY[:10]}...")
    print("="*80)

    results = {}

    # Run tests
    results['connection'] = test_api_connection()

    if not results['connection']:
        print("\n❌ API connection failed. Check your API key and try again.")
        return

    results['statements'] = test_financial_statements()
    results['prices'] = test_price_data()
    results['indicators'] = test_technical_indicators()
    results['comprehensive'] = test_comprehensive_scrape()
    results['processing'] = test_data_processing()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou're ready to scrape comprehensive FMP data!")
        print("\nNext steps:")
        print("1. Run: python data_scraping/fmp_comprehensive_scraper.py --api_key YOUR_KEY --dataset s_lot")
        print("2. Run: python data_scraping/fmp_data_processor.py --input s_lot_fmp_comprehensive.pkl")
        print("3. Update your training pipeline to use the processed data")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")


if __name__ == '__main__':
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ ERROR: Please set your API key in the script!")
        print("Edit this file and replace YOUR_API_KEY_HERE with your actual FMP API key.")
        print("\nGet your free key at: https://site.financialmodelingprep.com/developer/docs")
        sys.exit(1)

    run_all_tests()
