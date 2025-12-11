"""
Test script for hybrid scraper (yfinance + FMP).
"""

import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from data_scraping.hybrid_scraper import HybridScraper, scrape_hybrid_dataset
from utils.utils import save_pickle, pic_load


API_KEY = "YOUR_FMP_API_KEY_HERE"


def test_single_ticker():
    """Test scraping a single ticker."""
    print("\n" + "="*80)
    print("TEST: SINGLE TICKER (AAPL)")
    print("="*80)

    scraper = HybridScraper(API_KEY)

    # Scrape AAPL
    data = scraper.scrape_ticker('AAPL', start_date='2020-01-01', end_date='2024-12-31')

    if data and len(data) > 0:
        dates = list(data.keys())
        sample_tensor = data[dates[0]]

        print(f"\n✅ SUCCESS!")
        print(f"   Total days: {len(data)}")
        print(f"   Date range: {min(dates)} to {max(dates)}")
        print(f"   Features per day: {sample_tensor.shape[0]}")
        print(f"   Tensor shape: {sample_tensor.shape}")
        print(f"   Sample tensor (first 10 values): {sample_tensor[:10]}")

        return True
    else:
        print(f"\n❌ FAILED")
        return False


def test_multiple_tickers():
    """Test scraping multiple tickers."""
    print("\n" + "="*80)
    print("TEST: MULTIPLE TICKERS")
    print("="*80)

    test_stocks = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corp.',
        'GOOGL': 'Alphabet Inc.'
    }

    data = scrape_hybrid_dataset(
        stock_dict=test_stocks,
        fmp_api_key=API_KEY,
        output_file='test_hybrid_data.pkl',
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    if data and len(data) > 0:
        print(f"\n✅ SUCCESS!")
        print(f"   Stocks collected: {len(data)}")

        for ticker, ticker_data in data.items():
            if ticker_data:
                dates = list(ticker_data.keys())
                sample_tensor = ticker_data[dates[0]]
                print(f"\n   {ticker}:")
                print(f"     Days: {len(ticker_data)}")
                print(f"     Features: {sample_tensor.shape[0]}")

        return True
    else:
        print(f"\n❌ FAILED")
        return False


def test_load_and_verify():
    """Test loading saved data."""
    print("\n" + "="*80)
    print("TEST: LOAD AND VERIFY")
    print("="*80)

    try:
        data = pic_load('test_hybrid_data.pkl')

        if data and len(data) > 0:
            print(f"\n✅ Successfully loaded data")
            print(f"   Stocks: {len(data)}")

            # Verify tensor properties
            sample_ticker = list(data.keys())[0]
            sample_dates = list(data[sample_ticker].keys())
            sample_tensor = data[sample_ticker][sample_dates[0]]

            print(f"\n   Sample data ({sample_ticker}):")
            print(f"     Dates: {len(sample_dates)}")
            print(f"     Date range: {min(sample_dates)} to {max(sample_dates)}")
            print(f"     Features: {sample_tensor.shape[0]}")
            print(f"     Tensor dtype: {sample_tensor.dtype}")

            # Check for NaN/Inf
            has_nan = any(torch.isnan(data[ticker][d]).any()
                         for ticker in data
                         for d in list(data[ticker].keys())[:10])  # Check first 10 days

            has_inf = any(torch.isinf(data[ticker][d]).any()
                         for ticker in data
                         for d in list(data[ticker].keys())[:10])

            if has_nan:
                print(f"     ⚠️  Contains NaN values")
            else:
                print(f"     ✅ No NaN values")

            if has_inf:
                print(f"     ⚠️  Contains Inf values")
            else:
                print(f"     ✅ No Inf values")

            return True
        else:
            print(f"\n❌ No data loaded")
            return False

    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("HYBRID SCRAPER TEST SUITE")
    print("="*80)
    print(f"API Key: {API_KEY[:10]}...")
    print("="*80)

    results = {}

    # Run tests
    results['single_ticker'] = test_single_ticker()
    results['multiple_tickers'] = test_multiple_tickers()
    results['load_verify'] = test_load_and_verify()

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
        print("\nYou're ready to scrape your full dataset!")
        print("\nNext steps:")
        print("1. Run: python data_scraping/hybrid_scraper.py --dataset s_lot")
        print("2. Wait for completion (~30 min for 370 stocks)")
        print("3. Use the output in your training pipeline")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")


if __name__ == '__main__':
    run_all_tests()
