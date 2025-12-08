"""
Test FMP v3 API endpoints for daily data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "YOUR_FMP_API_KEY_HERE"
BASE_URL_V3 = "https://financialmodelingprep.com/api/v3"
BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"

def test_endpoint(base_url, endpoint, params=None, description=""):
    """Test if an endpoint works."""
    if params is None:
        params = {}
    params['apikey'] = API_KEY

    url = f"{base_url}/{endpoint}"

    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"URL: {endpoint}")
    print(f"{'='*80}")

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()

            if data:
                print(f"✅ SUCCESS - Got {len(data) if isinstance(data, list) else 1} records")

                # Show sample
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    print(f"\nSample:")
                    for key, value in list(sample.items())[:8]:
                        print(f"  {key}: {value}")

                    df = pd.DataFrame(data)
                    print(f"\nDataFrame: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    if 'date' in df.columns:
                        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

                elif isinstance(data, dict):
                    for key, value in list(data.items())[:8]:
                        print(f"  {key}: {value}")

                return True
            else:
                print("⚠️  Empty response")
                return False

        else:
            print(f"❌ FAILED - Status {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("FMP V3/V4 API DAILY ENDPOINTS TEST")
    print("="*80)

    results = {}
    today = datetime.now().strftime('%Y-%m-%d')
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Market indices
    print("\n\n### MARKET INDICES ###")
    results['sp500'] = test_endpoint(
        BASE_URL_V3, "historical-price-full/^GSPC",
        params={'from': one_year_ago, 'to': today},
        description="S&P 500"
    )

    results['vix'] = test_endpoint(
        BASE_URL_V3, "historical-price-full/^VIX",
        params={'from': one_year_ago, 'to': today},
        description="VIX"
    )

    results['nasdaq'] = test_endpoint(
        BASE_URL_V3, "historical-price-full/^IXIC",
        params={'from': one_year_ago, 'to': today},
        description="Nasdaq"
    )

    # Treasury rates
    print("\n\n### MACRO/ECONOMIC ###")
    results['treasury'] = test_endpoint(
        BASE_URL_V4, "treasury",
        params={'from': one_year_ago, 'to': today},
        description="Treasury Rates"
    )

    results['economic_calendar'] = test_endpoint(
        BASE_URL_V3, "economic_calendar",
        params={'from': one_year_ago, 'to': today},
        description="Economic Calendar"
    )

    # Sector performance
    print("\n\n### SECTOR PERFORMANCE ###")
    results['sector_performance'] = test_endpoint(
        BASE_URL_V3, "sector-performance",
        description="Sector Performance (Current)"
    )

    results['historical_sectors'] = test_endpoint(
        BASE_URL_V3, "historical-sectors-performance",
        params={'limit': 252},
        description="Historical Sector Performance"
    )

    results['sector_pe'] = test_endpoint(
        BASE_URL_V4, "sector_price_earning_ratio",
        params={'date': today},
        description="Sector P/E Ratios"
    )

    results['industry_pe'] = test_endpoint(
        BASE_URL_V4, "industry_price_earning_ratio",
        params={'date': today},
        description="Industry P/E Ratios"
    )

    # Market activity
    print("\n\n### MARKET ACTIVITY ###")
    results['gainers'] = test_endpoint(
        BASE_URL_V3, "stock_market/gainers",
        description="Top Gainers"
    )

    results['losers'] = test_endpoint(
        BASE_URL_V3, "stock_market/losers",
        description="Top Losers"
    )

    results['actives'] = test_endpoint(
        BASE_URL_V3, "stock_market/actives",
        description="Most Active"
    )

    # Per-stock data
    print("\n\n### PER-STOCK DATA (AAPL) ###")
    test_ticker = "AAPL"

    results['earnings_calendar'] = test_endpoint(
        BASE_URL_V3, f"historical/earning_calendar/{test_ticker}",
        params={'limit': 100},
        description="Earnings Calendar"
    )

    results['dividend_calendar'] = test_endpoint(
        BASE_URL_V3, f"stock_dividend_calendar",
        params={'symbol': test_ticker},
        description="Dividend Calendar"
    )

    results['daily_dcf'] = test_endpoint(
        BASE_URL_V3, f"historical-daily-discounted-cash-flow/{test_ticker}",
        params={'limit': 252},
        description="Daily DCF"
    )

    results['splits_calendar'] = test_endpoint(
        BASE_URL_V3, "stock_split_calendar",
        params={'from': one_year_ago, 'to': today},
        description="Stock Splits Calendar"
    )

    # Additional data sources
    print("\n\n### ADDITIONAL DATA ###")

    results['stock_news'] = test_endpoint(
        BASE_URL_V3, "stock_news",
        params={'tickers': test_ticker, 'limit': 10},
        description="Stock News"
    )

    results['press_releases'] = test_endpoint(
        BASE_URL_V3, f"press-releases/{test_ticker}",
        params={'limit': 10},
        description="Press Releases"
    )

    results['ipo_calendar'] = test_endpoint(
        BASE_URL_V3, "ipo_calendar",
        params={'from': one_year_ago, 'to': today},
        description="IPO Calendar"
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    working = [k for k, v in results.items() if v]
    not_working = [k for k, v in results.items() if not v]

    print(f"\n✅ WORKING ({len(working)}/{len(results)}):")
    for endpoint in working:
        print(f"  ✓ {endpoint}")

    if not_working:
        print(f"\n❌ NOT WORKING ({len(not_working)}/{len(results)}):")
        for endpoint in not_working:
            print(f"  ✗ {endpoint}")

    print(f"\nSuccess rate: {100*len(working)/len(results):.1f}%")

    return results


if __name__ == '__main__':
    results = main()
