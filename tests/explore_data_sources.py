"""
Explore alternative data sources for historical fundamental data.

Testing:
1. Alpha Vantage (free tier)
2. Financial Modeling Prep (free tier)
3. SEC EDGAR API (completely free)
4. yfinance extended methods
"""

print("="*80)
print("EXPLORING ALTERNATIVE DATA SOURCES")
print("="*80)

# 1. Alpha Vantage
print("\n1. ALPHA VANTAGE")
print("-"*80)
print("Pros:")
print("  - Free tier: 25 API calls/day")
print("  - Historical fundamental data (income statement, balance sheet, cash flow)")
print("  - Annual AND quarterly data going back ~20 years")
print("  - Clean JSON API")
print("\nCons:")
print("  - Requires API key (free signup)")
print("  - Rate limited (25 calls/day on free tier)")
print("  - Would take ~15 days to get 370 stocks")
print("\nExample API call:")
print("  https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=YOUR_KEY")

# 2. Financial Modeling Prep
print("\n2. FINANCIAL MODELING PREP (FMP)")
print("-"*80)
print("Pros:")
print("  - Free tier: 250 API calls/day")
print("  - Comprehensive fundamental data")
print("  - Historical data going back 20+ years")
print("  - Fast API")
print("\nCons:")
print("  - Requires API key (free signup)")
print("  - Rate limited (250 calls/day)")
print("  - Would take ~2 days to get 370 stocks")
print("\nExample API call:")
print("  https://financialmodelingprep.com/api/v3/income-statement/AAPL?apikey=YOUR_KEY")

# 3. SEC EDGAR
print("\n3. SEC EDGAR API (XBRL)")
print("-"*80)
print("Pros:")
print("  - Completely FREE, no API key needed")
print("  - OFFICIAL source (direct from SEC filings)")
print("  - Historical data going back 10+ years")
print("  - No rate limits (but SEC asks for reasonable use)")
print("  - Most comprehensive data available")
print("\nCons:")
print("  - Complex data format (XBRL)")
print("  - Requires parsing XML/JSON")
print("  - Different companies use different tags")
print("  - More work to extract standardized metrics")
print("\nExample:")
print("  - SEC provides Company Facts API (JSON)")
print("  - https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json (Apple)")

# 4. Check if yfinance has other methods
print("\n4. YFINANCE EXTENDED")
print("-"*80)
print("Checking if yfinance has methods we haven't used...")

try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")

    # Check all available attributes
    attrs = [attr for attr in dir(ticker) if not attr.startswith('_')]

    financial_attrs = []
    for attr in attrs:
        if any(keyword in attr.lower() for keyword in ['financ', 'income', 'balance', 'cash', 'earnings', 'revenue']):
            financial_attrs.append(attr)

    print("Found financial-related attributes:")
    for attr in financial_attrs:
        print(f"  - ticker.{attr}")

    # Test 'earnings_dates' which might have historical data
    print("\nTesting ticker.earnings_dates:")
    try:
        earnings_dates = ticker.earnings_dates
        if earnings_dates is not None:
            print(f"  Shape: {earnings_dates.shape}")
            print(f"  Date range: {earnings_dates.index.min()} to {earnings_dates.index.max()}")
            print(f"  Columns: {list(earnings_dates.columns)}")
    except:
        print("  Not available")

    # Test 'earnings_history'
    print("\nTesting ticker.earnings_history:")
    try:
        earnings_history = ticker.earnings_history
        if earnings_history is not None:
            print(f"  Shape: {earnings_history.shape}")
            print(f"  Columns: {list(earnings_history.columns)}")
    except:
        print("  Not available")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nBest option for YOUR use case:")
print("\n  **SEC EDGAR API** (Company Facts)")
print("  - FREE, no API key")
print("  - 10+ years of quarterly data")
print("  - Official source")
print("  - Can download ALL 370 stocks in ~1 hour")
print("\nAlternative if SEC is too complex:")
print("\n  **Financial Modeling Prep** (FMP)")
print("  - Free tier sufficient")
print("  - 250 calls/day = ~2 days for 370 stocks")
print("  - Clean, standardized data")
print("  - Sign up at: https://site.financialmodelingprep.com/developer/docs")
print("="*80)
