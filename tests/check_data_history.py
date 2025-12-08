"""
Check how much historical fundamental data is available.
"""

import yfinance as yf
import pandas as pd

ticker = yf.Ticker("AAPL")

print("="*80)
print("CHECKING DATA HISTORY DEPTH")
print("="*80)

# Check quarterly data
print("\n1. QUARTERLY DATA")
print("-"*80)
quarterly_income = ticker.quarterly_income_stmt
if quarterly_income is not None and not quarterly_income.empty:
    dates = quarterly_income.columns
    print(f"Number of quarters: {len(dates)}")
    print(f"Date range: {dates[-1].date()} to {dates[0].date()}")
    print(f"Span: {(dates[0] - dates[-1]).days / 365.25:.1f} years")

# Check annual data
print("\n2. ANNUAL DATA")
print("-"*80)
annual_income = ticker.income_stmt
if annual_income is not None and not annual_income.empty:
    dates = annual_income.columns
    print(f"Number of years: {len(dates)}")
    print(f"Date range: {dates[-1].date()} to {dates[0].date()}")
    print(f"Span: {(dates[0] - dates[-1]).days / 365.25:.1f} years")
    print(f"\nAnnual dates:")
    for date in dates:
        print(f"  - {date.date()}")

# Check if we can get more quarterly data using financials API
print("\n3. TRYING ALTERNATIVE: ticker.financials")
print("-"*80)
try:
    financials = ticker.financials
    if financials is not None and not financials.empty:
        dates = financials.columns
        print(f"Number of periods: {len(dates)}")
        print(f"Date range: {dates[-1].date()} to {dates[0].date()}")
except:
    print("ticker.financials not available")

# Check quarterly_financials
print("\n4. TRYING: ticker.quarterly_financials")
print("-"*80)
try:
    q_financials = ticker.quarterly_financials
    if q_financials is not None and not q_financials.empty:
        dates = q_financials.columns
        print(f"Number of quarters: {len(dates)}")
        print(f"Date range: {dates[-1].date()} to {dates[0].date()}")
except:
    print("ticker.quarterly_financials not available")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("For time-varying fundamentals over 5-10 years, we have:")
print("  - Quarterly data: ~1.5-2 years (5-7 quarters)")
print("  - Annual data: ~4-5 years")
print()
print("Recommendation:")
print("  Use ANNUAL data (4-5 years) with forward-fill")
print("  OR")
print("  Mix: Annual for older periods + Quarterly for recent 2 years")
print("="*80)
