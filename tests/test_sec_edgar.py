"""
Test SEC EDGAR Company Facts API for historical fundamental data.

The SEC provides a Company Facts API that returns ALL financial data
from company filings in standardized JSON format.
"""

import requests
import json
import pandas as pd
from datetime import datetime

# SEC requires a User-Agent header
HEADERS = {
    'User-Agent': 'Stock-Prediction-Research james@example.com'  # SEC asks you identify yourself
}

def get_cik_from_ticker(ticker):
    """
    Get CIK (Central Index Key) for a ticker.
    SEC uses CIKs, not tickers.
    """
    # SEC provides a ticker-to-CIK mapping
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    for entry in data.values():
        if entry['ticker'] == ticker:
            # CIK needs to be 10 digits with leading zeros
            return str(entry['cik_str']).zfill(10)
    return None


def get_company_facts(cik):
    """
    Get all financial facts for a company from SEC EDGAR.

    Returns JSON with quarterly/annual data for ALL metrics going back 10+ years.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


print("="*80)
print("TESTING SEC EDGAR COMPANY FACTS API")
print("="*80)

# Test with Apple
ticker = "AAPL"
print(f"\n1. Getting CIK for {ticker}...")

cik = get_cik_from_ticker(ticker)
if cik:
    print(f"✅ CIK: {cik}")
else:
    print(f"❌ Could not find CIK for {ticker}")
    exit(1)

# Get company facts
print(f"\n2. Fetching company facts from SEC...")
facts = get_company_facts(cik)

if facts:
    print(f"✅ Successfully retrieved data")

    # Explore structure
    print(f"\n3. Exploring data structure...")
    print(f"Top-level keys: {list(facts.keys())}")

    entity_info = facts.get('entityName', 'Unknown')
    print(f"Entity: {entity_info}")

    # The 'facts' key contains all the financial data
    if 'facts' in facts:
        accounting_standards = list(facts['facts'].keys())
        print(f"\nAccounting standards available: {accounting_standards}")

        # US-GAAP is the main standard for US companies
        if 'us-gaap' in facts['facts']:
            us_gaap = facts['facts']['us-gaap']
            print(f"\nNumber of US-GAAP metrics: {len(us_gaap)}")

            # Show some key metrics
            print(f"\nSample metrics available:")
            key_metrics = [
                'Revenues',
                'GrossProfit',
                'OperatingIncomeLoss',
                'NetIncomeLoss',
                'Assets',
                'Liabilities',
                'StockholdersEquity',
                'OperatingCashFlow',
                'AssetsCurrent',
                'LiabilitiesCurrent'
            ]

            for metric in key_metrics:
                if metric in us_gaap:
                    # Each metric has units (USD, shares, etc.) and data points
                    units = list(us_gaap[metric]['units'].keys())
                    print(f"  ✓ {metric} (units: {units})")

            # Dive deep into one metric to see historical data
            print(f"\n4. Examining 'Revenues' in detail...")
            if 'Revenues' in us_gaap:
                revenue_data = us_gaap['Revenues']

                # USD data
                if 'USD' in revenue_data['units']:
                    usd_data = revenue_data['units']['USD']
                    print(f"\nTotal revenue data points: {len(usd_data)}")

                    # Convert to DataFrame for easier viewing
                    df = pd.DataFrame(usd_data)
                    print(f"\nColumns: {list(df.columns)}")

                    # Filter for 10-Q (quarterly) and 10-K (annual) filings
                    quarterly = df[df['form'] == '10-Q']
                    annual = df[df['form'] == '10-K']

                    print(f"\nQuarterly filings (10-Q): {len(quarterly)}")
                    print(f"Annual filings (10-K): {len(annual)}")

                    if not quarterly.empty:
                        print(f"\nQuarterly date range:")
                        print(f"  Oldest: {quarterly['end'].min()}")
                        print(f"  Newest: {quarterly['end'].max()}")

                        # Show recent quarterly revenues
                        recent_q = quarterly.sort_values('end', ascending=False).head(8)
                        print(f"\nRecent quarterly revenues:")
                        for _, row in recent_q.iterrows():
                            filing_date = row['filed']
                            end_date = row['end']
                            value = row['val'] / 1e9  # Convert to billions
                            print(f"  {end_date}: ${value:.2f}B (filed: {filing_date})")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nSEC EDGAR provides:")
print("  ✓ 10+ years of quarterly data")
print("  ✓ Completely FREE (no API key)")
print("  ✓ Official source (directly from SEC filings)")
print("  ✓ Comprehensive metrics")
print("\nNext step:")
print("  Build a parser to extract key metrics and align them to dates")
print("="*80)
