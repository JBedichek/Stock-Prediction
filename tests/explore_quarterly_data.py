"""
Explore what quarterly financial data is available from yfinance.
"""

import yfinance as yf
import pandas as pd

# Test with a well-known stock
ticker = yf.Ticker("AAPL")

print("="*80)
print("EXPLORING QUARTERLY FINANCIAL DATA FOR AAPL")
print("="*80)

# 1. Quarterly Income Statement
print("\n1. QUARTERLY INCOME STATEMENT")
print("-"*80)
income_stmt = ticker.quarterly_income_stmt
if income_stmt is not None and not income_stmt.empty:
    print(f"Shape: {income_stmt.shape}")
    print(f"Columns (dates): {list(income_stmt.columns)[:5]}")  # Show first 5 quarters
    print(f"\nAvailable metrics ({len(income_stmt.index)} total):")
    for metric in list(income_stmt.index)[:10]:
        print(f"  - {metric}")
    print("  ...")

    # Show sample data
    print(f"\nSample data for most recent quarter ({income_stmt.columns[0]}):")
    print(income_stmt[income_stmt.columns[0]].head(10))
else:
    print("No quarterly income statement data available")

# 2. Quarterly Balance Sheet
print("\n2. QUARTERLY BALANCE SHEET")
print("-"*80)
balance_sheet = ticker.quarterly_balance_sheet
if balance_sheet is not None and not balance_sheet.empty:
    print(f"Shape: {balance_sheet.shape}")
    print(f"Columns (dates): {list(balance_sheet.columns)[:5]}")
    print(f"\nAvailable metrics ({len(balance_sheet.index)} total):")
    for metric in list(balance_sheet.index)[:10]:
        print(f"  - {metric}")
    print("  ...")
else:
    print("No quarterly balance sheet data available")

# 3. Quarterly Cash Flow
print("\n3. QUARTERLY CASH FLOW")
print("-"*80)
cashflow = ticker.quarterly_cashflow
if cashflow is not None and not cashflow.empty:
    print(f"Shape: {cashflow.shape}")
    print(f"Columns (dates): {list(cashflow.columns)[:5]}")
    print(f"\nAvailable metrics ({len(cashflow.index)} total):")
    for metric in list(cashflow.index)[:10]:
        print(f"  - {metric}")
    print("  ...")
else:
    print("No quarterly cash flow data available")

# 4. Calculate sample derived metrics
print("\n4. CALCULATED TIME-VARYING METRICS")
print("-"*80)

if income_stmt is not None and not income_stmt.empty:
    try:
        # Extract key metrics
        total_revenue = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
        gross_profit = income_stmt.loc['Gross Profit'] if 'Gross Profit' in income_stmt.index else None
        operating_income = income_stmt.loc['Operating Income'] if 'Operating Income' in income_stmt.index else None
        net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None

        if total_revenue is not None and gross_profit is not None:
            gross_margin = (gross_profit / total_revenue) * 100
            print("\nGross Margin % over time:")
            for date, value in gross_margin.head(8).items():
                print(f"  {date.date()}: {value:.2f}%")

        if total_revenue is not None and operating_income is not None:
            operating_margin = (operating_income / total_revenue) * 100
            print("\nOperating Margin % over time:")
            for date, value in operating_margin.head(8).items():
                print(f"  {date.date()}: {value:.2f}%")

    except Exception as e:
        print(f"Error calculating metrics: {e}")

# 5. Check balance sheet for equity/assets
if balance_sheet is not None and not balance_sheet.empty:
    try:
        total_assets = balance_sheet.loc['Total Assets'] if 'Total Assets' in balance_sheet.index else None
        stockholder_equity = balance_sheet.loc['Stockholders Equity'] if 'Stockholders Equity' in balance_sheet.index else None
        total_debt = balance_sheet.loc['Total Debt'] if 'Total Debt' in balance_sheet.index else None

        if net_income is not None and stockholder_equity is not None:
            # ROE = Net Income / Stockholders Equity
            roe = (net_income / stockholder_equity) * 100
            print("\nReturn on Equity (ROE) % over time:")
            for date, value in roe.head(8).items():
                print(f"  {date.date()}: {value:.2f}%")

    except Exception as e:
        print(f"Error calculating ROE: {e}")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
