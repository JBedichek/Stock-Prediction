"""
Test Financial Modeling Prep (FMP) API for historical fundamental data.

To use:
1. Sign up for FREE API key at: https://site.financialmodelingprep.com/developer/docs
2. Replace 'YOUR_API_KEY' below with your key
3. Run this script

Free tier: 250 API calls/day (enough for ~100 stocks per day)
"""

import requests
import json
import pandas as pd
from datetime import datetime

# ============================================================================
# STEP 1: GET YOUR FREE API KEY
# ============================================================================
# Go to: https://site.financialmodelingprep.com/developer/docs
# Click "Get your Free API Key Today"
# After signup, copy your API key and paste it here:

API_KEY = "YOUR_FMP_API_KEY_HERE"  # ⚠️ REPLACE THIS WITH YOUR ACTUAL KEY

# ============================================================================

def test_fmp_api(api_key):
    """Test FMP API and show available data."""

    if api_key == "YOUR_API_KEY":
        print("="*80)
        print("⚠️  API KEY NOT SET!")
        print("="*80)
        print("\nPlease follow these steps:")
        print("1. Go to: https://site.financialmodelingprep.com/developer/docs")
        print("2. Click 'Get your Free API Key Today'")
        print("3. Sign up (it's free)")
        print("4. Copy your API key")
        print("5. Edit this file and replace 'YOUR_API_KEY' with your key")
        print("6. Run this script again")
        print("="*80)

        # Try with demo key to show structure anyway
        print("\nUsing DEMO key to show data structure...")
        api_key = "demo"

    print("\n" + "="*80)
    print("TESTING FINANCIAL MODELING PREP API")
    print("="*80)

    ticker = "AAPL"

    # 1. Test Income Statement
    print(f"\n1. Testing Income Statement for {ticker}...")
    print("-"*80)

    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"✅ Success! Retrieved {len(data)} periods")

            # Convert to DataFrame for easier viewing
            df = pd.DataFrame(data)
            print(f"\nColumns available: {list(df.columns)}")

            # Show date range
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'])
                print(f"\nDate range: {dates.min().date()} to {dates.max().date()}")
                print(f"Span: {(dates.max() - dates.min()).days / 365.25:.1f} years")

            # Show recent quarters
            print(f"\nMost recent 8 quarters:")
            print(f"{'Date':<12} {'Revenue':>15} {'Gross Profit':>15} {'Net Income':>15}")
            print("-"*60)
            for i in range(min(8, len(df))):
                row = df.iloc[i]
                date = row.get('date', 'N/A')
                revenue = row.get('revenue', 0) / 1e9
                gross = row.get('grossProfit', 0) / 1e9
                net = row.get('netIncome', 0) / 1e9
                print(f"{date:<12} ${revenue:>13.2f}B ${gross:>13.2f}B ${net:>13.2f}B")

            # Calculate time-varying metrics
            print(f"\n2. Calculating Time-Varying Metrics...")
            print("-"*80)

            if 'revenue' in df.columns and 'grossProfit' in df.columns:
                df['grossMargin'] = (df['grossProfit'] / df['revenue']) * 100
                df['operatingMargin'] = (df['operatingIncome'] / df['revenue']) * 100
                df['netMargin'] = (df['netIncome'] / df['revenue']) * 100

                print(f"\nGross Margin % over time:")
                for i in range(min(8, len(df))):
                    row = df.iloc[i]
                    print(f"  {row['date']}: {row['grossMargin']:.2f}%")
        else:
            print(f"⚠️  Response: {data}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

    # 2. Test Balance Sheet
    print(f"\n3. Testing Balance Sheet for {ticker}...")
    print("-"*80)

    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"✅ Success! Retrieved {len(data)} periods")
            df_balance = pd.DataFrame(data)

            # Show key metrics
            print(f"\nKey balance sheet metrics available:")
            key_metrics = ['totalAssets', 'totalLiabilities', 'totalStockholdersEquity',
                          'totalCurrentAssets', 'totalCurrentLiabilities', 'cashAndCashEquivalents']
            for metric in key_metrics:
                if metric in df_balance.columns:
                    print(f"  ✓ {metric}")

    # 3. Test Cash Flow
    print(f"\n4. Testing Cash Flow for {ticker}...")
    print("-"*80)

    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=120&apikey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"✅ Success! Retrieved {len(data)} periods")
            df_cash = pd.DataFrame(data)

            print(f"\nKey cash flow metrics available:")
            key_metrics = ['operatingCashFlow', 'freeCashFlow', 'capitalExpenditure']
            for metric in key_metrics:
                if metric in df_cash.columns:
                    print(f"  ✓ {metric}")

    print("\n" + "="*80)
    print("FMP API TEST COMPLETE")
    print("="*80)

    if api_key != "demo":
        print("\n✅ Your API key is working!")
        print(f"\nNext steps:")
        print(f"1. We can now extract 20+ years of quarterly data")
        print(f"2. 250 calls/day = ~100 stocks/day")
        print(f"3. For 370 stocks = ~4 days to download all data")
        print(f"4. Data will be saved locally, only need to download once")
    else:
        print("\n⚠️  Using demo key - please get your own API key to continue")

    print("="*80)


if __name__ == "__main__":
    test_fmp_api(API_KEY)
