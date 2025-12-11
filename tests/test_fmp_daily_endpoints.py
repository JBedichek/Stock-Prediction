"""
Test script to verify which FMP daily-frequency endpoints are available
with your API key and what data they return.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your API key
API_KEY = "YOUR_FMP_API_KEY_HERE"
BASE_URL = "https://financialmodelingprep.com/stable"

def test_endpoint(endpoint, params=None, description=""):
    """Test if an endpoint works and show sample data."""
    if params is None:
        params = {}
    params['apikey'] = API_KEY

    url = f"{BASE_URL}/{endpoint}"

    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Endpoint: {endpoint}")
    print(f"{'='*80}")

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()

            if data:
                print(f"✅ SUCCESS - Got {len(data) if isinstance(data, list) else 1} records")

                # Show sample
                if isinstance(data, list) and len(data) > 0:
                    print(f"\nSample (first record):")
                    sample = data[0]
                    for key, value in list(sample.items())[:10]:  # Show first 10 fields
                        print(f"  {key}: {value}")
                    if len(sample) > 10:
                        print(f"  ... and {len(sample) - 10} more fields")

                    # Try converting to DataFrame
                    try:
                        df = pd.DataFrame(data)
                        print(f"\nDataFrame shape: {df.shape}")
                        print(f"Columns: {list(df.columns)[:10]}...")
                        if 'date' in df.columns:
                            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                    except:
                        pass

                elif isinstance(data, dict):
                    print(f"\nSample data:")
                    for key, value in list(data.items())[:10]:
                        print(f"  {key}: {value}")

                return True
            else:
                print("⚠️  Empty response (no data)")
                return False

        else:
            print(f"❌ FAILED - Status code: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("FMP DAILY ENDPOINTS TEST SUITE")
    print("="*80)

    results = {}

    # Test dates
    today = datetime.now().strftime('%Y-%m-%d')
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # =========================================================================
    # TIER 1: MARKET-WIDE DATA
    # =========================================================================

    print("\n" + "="*80)
    print("TIER 1: MARKET-WIDE DATA (High Priority)")
    print("="*80)

    # Treasury rates
    results['treasury'] = test_endpoint(
        "treasury",
        params={'from': one_year_ago, 'to': today},
        description="Treasury Rates (Daily)"
    )

    # Economic calendar
    results['economic_calendar'] = test_endpoint(
        "economic_calendar",
        params={'from': one_year_ago, 'to': today},
        description="Economic Calendar"
    )

    # Market risk premium
    results['market_risk_premium'] = test_endpoint(
        "market_risk_premium",
        description="Market Risk Premium"
    )

    # Sector performance (current)
    results['sector_performance'] = test_endpoint(
        "sectors-performance",
        description="Current Sector Performance"
    )

    # Historical sector performance
    results['historical_sector_performance'] = test_endpoint(
        "historical-sectors-performance",
        params={'limit': 1000},
        description="Historical Sector Performance"
    )

    # Sector P/E ratios
    results['sector_pe'] = test_endpoint(
        "sector-price-earning-ratio",
        params={'date': today},
        description="Sector P/E Ratios"
    )

    # Industry P/E ratios
    results['industry_pe'] = test_endpoint(
        "industry-price-earning-ratio",
        params={'date': today},
        description="Industry P/E Ratios"
    )

    # Indices (S&P 500, VIX, etc.)
    results['sp500'] = test_endpoint(
        "historical-price-eod/full",
        params={'symbol': '^GSPC', 'from': one_year_ago, 'to': today},
        description="S&P 500 Historical (^GSPC)"
    )

    results['vix'] = test_endpoint(
        "historical-price-eod/full",
        params={'symbol': '^VIX', 'from': one_year_ago, 'to': today},
        description="VIX Historical (^VIX)"
    )

    results['nasdaq'] = test_endpoint(
        "historical-price-eod/full",
        params={'symbol': '^IXIC', 'from': one_year_ago, 'to': today},
        description="Nasdaq Historical (^IXIC)"
    )

    # =========================================================================
    # TIER 2: PER-STOCK DAILY DATA
    # =========================================================================

    print("\n" + "="*80)
    print("TIER 2: PER-STOCK DAILY DATA (Medium Priority)")
    print("="*80)

    test_ticker = "AAPL"

    # Earnings calendar
    results['earnings_calendar'] = test_endpoint(
        "earning_calendar",
        params={'symbol': test_ticker, 'from': one_year_ago, 'to': today},
        description=f"Earnings Calendar for {test_ticker}"
    )

    # Dividend calendar
    results['dividend_calendar'] = test_endpoint(
        "stock_dividend_calendar",
        params={'symbol': test_ticker, 'from': one_year_ago, 'to': today},
        description=f"Dividend Calendar for {test_ticker}"
    )

    # Daily DCF
    results['daily_dcf'] = test_endpoint(
        "historical-daily-discounted-cash-flow",
        params={'symbol': test_ticker, 'limit': 1000},
        description=f"Daily DCF for {test_ticker}"
    )

    # Stock splits calendar
    results['splits_calendar'] = test_endpoint(
        "stock_split_calendar",
        params={'from': one_year_ago, 'to': today},
        description="Stock Splits Calendar"
    )

    # =========================================================================
    # TIER 3: MARKET ACTIVITY DATA
    # =========================================================================

    print("\n" + "="*80)
    print("TIER 3: MARKET ACTIVITY DATA (Lower Priority)")
    print("="*80)

    # Top gainers
    results['gainers'] = test_endpoint(
        "stock_market/gainers",
        description="Top Gainers Today"
    )

    # Top losers
    results['losers'] = test_endpoint(
        "stock_market/losers",
        description="Top Losers Today"
    )

    # Most active
    results['actives'] = test_endpoint(
        "stock_market/actives",
        description="Most Active Stocks Today"
    )

    # IPO calendar
    results['ipo_calendar'] = test_endpoint(
        "ipo_calendar",
        params={'from': one_year_ago, 'to': today},
        description="IPO Calendar"
    )

    # =========================================================================
    # POTENTIAL PREMIUM FEATURES
    # =========================================================================

    print("\n" + "="*80)
    print("POTENTIAL PREMIUM FEATURES")
    print("="*80)

    # Stock news
    results['stock_news'] = test_endpoint(
        "stock_news",
        params={'tickers': test_ticker, 'limit': 50},
        description=f"Stock News for {test_ticker}"
    )

    # Press releases
    results['press_releases'] = test_endpoint(
        "press-releases",
        params={'symbol': test_ticker, 'limit': 50},
        description=f"Press Releases for {test_ticker}"
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    working = [k for k, v in results.items() if v]
    not_working = [k for k, v in results.items() if not v]

    print(f"\n✅ WORKING ENDPOINTS ({len(working)}/{len(results)}):")
    for endpoint in working:
        print(f"  ✓ {endpoint}")

    if not_working:
        print(f"\n❌ NOT WORKING ({len(not_working)}/{len(results)}):")
        for endpoint in not_working:
            print(f"  ✗ {endpoint}")

    print(f"\n{'='*80}")
    print(f"Success rate: {len(working)}/{len(results)} ({100*len(working)/len(results):.1f}%)")
    print(f"{'='*80}\n")

    return results


if __name__ == '__main__':
    results = main()
