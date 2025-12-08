"""
Generate time-varying fundamental metrics using Financial Modeling Prep (FMP) API.

This extracts 20+ years of quarterly fundamental data.

Setup:
1. Get FREE API key: https://site.financialmodelingprep.com/developer/docs
2. Set API_KEY variable below
3. Run: python generate_fmp_fundamentals.py --dataset s_lot

Free tier limits: 250 API calls/day
- Each stock = 3 calls (income + balance + cash flow)
- Can process ~80 stocks per day
- For 370 stocks = ~5 days total
"""

import requests
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import sys

sys.path.append('/home/james/Desktop/Stock-Prediction')
from .Stock import a_lot_of_stocks, s_lot_of_stocks
from utils.utils import save_pickle

# ============================================================================
# SET YOUR API KEY HERE
# ============================================================================
API_KEY = "YOUR_API_KEY"  # ⚠️ Replace with your actual FMP API key
# ============================================================================


def get_fmp_data(ticker, statement_type, api_key, limit=120):
    """
    Fetch financial statement data from FMP.

    Args:
        ticker: Stock ticker
        statement_type: 'income-statement', 'balance-sheet-statement', or 'cash-flow-statement'
        api_key: FMP API key
        limit: Number of periods to retrieve (quarterly, max 120 = 30 years)

    Returns:
        DataFrame with financial data
    """
    url = f"https://financialmodelingprep.com/stable/{statement_type}?symbol={ticker}&period=quarter&limit={limit}&apikey={api_key}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return pd.DataFrame(data)
            else:
                return None
        else:
            print(f"  ⚠️  API Error {response.status_code} for {ticker}")
            return None
    except Exception as e:
        print(f"  ❌ Exception for {ticker}: {e}")
        return None


def calculate_fundamental_metrics(income_df, balance_df, cash_df):
    """
    Calculate 27 fundamental metrics from financial statements.

    Returns:
        dict[date: dict[metric: value]]
    """
    fundamentals_by_date = {}

    if income_df is None or income_df.empty:
        return fundamentals_by_date

    # Merge all statements on date
    merged = income_df.copy()

    if balance_df is not None and not balance_df.empty:
        merged = merged.merge(balance_df, on='date', how='left', suffixes=('', '_balance'))

    if cash_df is not None and not cash_df.empty:
        merged = merged.merge(cash_df, on='date', how='left', suffixes=('', '_cash'))

    # Process each quarter
    for _, row in merged.iterrows():
        date = pd.to_datetime(row['date']).date()
        metrics = {}

        # Helper to safely get values
        def safe_val(key):
            val = row.get(key)
            if pd.notna(val) and not np.isinf(val):
                return float(val)
            return None

        # Extract raw values
        revenue = safe_val('revenue')
        gross_profit = safe_val('grossProfit')
        operating_income = safe_val('operatingIncome')
        net_income = safe_val('netIncome')
        ebitda = safe_val('ebitda')

        total_assets = safe_val('totalAssets')
        stockholder_equity = safe_val('totalStockholdersEquity')
        total_debt = safe_val('totalDebt') or safe_val('totalLiabilities')
        current_assets = safe_val('totalCurrentAssets')
        current_liabilities = safe_val('totalCurrentLiabilities')
        cash = safe_val('cashAndCashEquivalents')

        operating_cashflow = safe_val('operatingCashFlow')
        free_cashflow = safe_val('freeCashFlow')

        # Calculate derived metrics (matching our 27-metric standard)

        # Profitability margins
        metrics['grossMargins'] = gross_profit / revenue if (gross_profit and revenue) else None
        metrics['operatingMargins'] = operating_income / revenue if (operating_income and revenue) else None
        metrics['profitMargins'] = net_income / revenue if (net_income and revenue) else None
        metrics['ebitdaMargins'] = ebitda / revenue if (ebitda and revenue) else None

        # Returns
        metrics['returnOnEquity'] = net_income / stockholder_equity if (net_income and stockholder_equity) else None
        metrics['returnOnAssets'] = net_income / total_assets if (net_income and total_assets) else None

        # Financial health
        metrics['currentRatio'] = current_assets / current_liabilities if (current_assets and current_liabilities) else None
        metrics['quickRatio'] = (current_assets - safe_val('inventory')) / current_liabilities if (current_liabilities and current_assets) else None
        metrics['debtToEquity'] = total_debt / stockholder_equity if (total_debt and stockholder_equity) else None

        # Absolute values (scaled to billions)
        metrics['totalCash'] = cash / 1e9 if cash else None
        metrics['totalDebt'] = total_debt / 1e9 if total_debt else None
        metrics['operatingCashflow'] = operating_cashflow / 1e9 if operating_cashflow else None

        # Per-share metrics (if shares available)
        shares = safe_val('weightedAverageShsOut') or safe_val('weightedAverageShsOutDil')
        metrics['revenuePerShare'] = revenue / shares if (revenue and shares) else None
        metrics['trailingEps'] = safe_val('eps')
        metrics['bookValue'] = stockholder_equity / shares if (stockholder_equity and shares) else None

        # Market metrics (from income statement if available)
        market_cap = safe_val('marketCap')
        metrics['marketCap'] = market_cap / 1e9 if market_cap else None

        # Valuation ratios
        price = safe_val('price')
        if price:
            metrics['priceToBook'] = price / (metrics['bookValue'] or 1)
            metrics['priceToSalesTrailing12Months'] = (price * shares) / revenue if (shares and revenue) else None
        else:
            metrics['priceToBook'] = None
            metrics['priceToSalesTrailing12Months'] = None

        metrics['enterpriseToRevenue'] = (market_cap + total_debt - cash) / revenue if all([market_cap, revenue]) else None
        metrics['enterpriseToEbitda'] = (market_cap + total_debt - cash) / ebitda if all([market_cap, ebitda]) else None

        # Placeholder for metrics not in statements
        metrics['forwardPE'] = None
        metrics['forwardEps'] = None
        metrics['beta'] = None
        metrics['fiftyTwoWeekHigh'] = None
        metrics['fiftyTwoWeekLow'] = None
        metrics['payoutRatio'] = safe_val('dividendsPaid') / net_income if (net_income and safe_val('dividendsPaid')) else None
        metrics['revenueGrowth'] = None  # Will calculate later

        fundamentals_by_date[date] = metrics

    # Calculate revenue growth (quarter-over-quarter)
    sorted_dates = sorted(fundamentals_by_date.keys())
    for i in range(1, len(sorted_dates)):
        curr_date = sorted_dates[i]
        prev_date = sorted_dates[i-1]

        curr_row = merged[merged['date'] == str(curr_date)]
        prev_row = merged[merged['date'] == str(prev_date)]

        if not curr_row.empty and not prev_row.empty:
            curr_rev = curr_row.iloc[0].get('revenue')
            prev_rev = prev_row.iloc[0].get('revenue')

            if curr_rev and prev_rev and prev_rev != 0:
                growth = (curr_rev - prev_rev) / prev_rev
                fundamentals_by_date[curr_date]['revenueGrowth'] = growth

    return fundamentals_by_date


METRIC_ORDER = [
    'ebitdaMargins', 'grossMargins', 'operatingMargins', 'profitMargins',
    'totalCash', 'totalDebt', 'currentRatio', 'quickRatio',
    'priceToBook', 'priceToSalesTrailing12Months', 'enterpriseToRevenue',
    'returnOnEquity', 'returnOnAssets', 'revenueGrowth',
    'operatingCashflow', 'trailingEps', 'bookValue', 'revenuePerShare',
    'beta', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'marketCap',
    'forwardPE', 'payoutRatio', 'enterpriseToEbitda', 'forwardEps',
    'debtToEquity'
]


def forward_fill_to_daily(fundamentals_by_date, start_date, end_date):
    """Forward-fill quarterly fundamentals to daily frequency."""
    if not fundamentals_by_date:
        return {}

    reporting_dates = sorted(fundamentals_by_date.keys())
    daily_fundamentals = {}
    current_date = start_date
    current_fundamentals = None
    reporting_idx = 0

    while current_date <= end_date:
        while reporting_idx < len(reporting_dates) and reporting_dates[reporting_idx] <= current_date:
            current_fundamentals = fundamentals_by_date[reporting_dates[reporting_idx]]
            reporting_idx += 1

        if current_fundamentals is not None:
            daily_fundamentals[current_date] = current_fundamentals.copy()

        current_date += timedelta(days=1)

    return daily_fundamentals


def generate_fmp_fundamentals(stock_dict, dataset_name, api_key, start_date='2015-01-01', end_date='2025-12-31', resume_from=None):
    """
    Generate time-varying fundamentals using FMP API.

    Args:
        stock_dict: Dict of {ticker: company_name}
        dataset_name: Name of dataset
        api_key: FMP API key
        start_date: Start date for forward-fill
        end_date: End date for forward-fill
        resume_from: Ticker to resume from (if interrupted)

    Returns:
        dict[ticker: dict[date: tensor]]
    """
    print(f"\n{'='*80}")
    print(f"GENERATING FMP FUNDAMENTALS FOR {dataset_name}")
    print(f"{'='*80}")
    print(f"Dataset: {len(stock_dict)} stocks")
    print(f"Date range: {start_date} to {end_date}")
    print(f"API calls per stock: 3 (income + balance + cash)")
    print(f"Total API calls needed: {len(stock_dict) * 3}")
    print(f"Free tier limit: 250 calls/day")
    print(f"Estimated days: {(len(stock_dict) * 3) / 250:.1f}")

    if resume_from:
        print(f"Resuming from: {resume_from}")

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    all_stock_fundamentals = {}
    api_call_count = 0
    resuming = resume_from is not None

    for ticker_symbol, company_name in tqdm(stock_dict.items(), desc="Processing"):
        # Skip until we reach resume point
        if resuming:
            if ticker_symbol == resume_from:
                resuming = False
            else:
                continue

        # Fetch financial statements
        income_df = get_fmp_data(ticker_symbol, 'income-statement', api_key)
        time.sleep(0.3)  # Rate limiting
        api_call_count += 1

        balance_df = get_fmp_data(ticker_symbol, 'balance-sheet-statement', api_key)
        time.sleep(0.3)
        api_call_count += 1

        cash_df = get_fmp_data(ticker_symbol, 'cash-flow-statement', api_key)
        time.sleep(0.3)
        api_call_count += 1

        if income_df is None:
            continue

        # Calculate metrics
        fundamentals_by_date = calculate_fundamental_metrics(income_df, balance_df, cash_df)

        if not fundamentals_by_date:
            continue

        # Forward-fill to daily
        daily_fundamentals = forward_fill_to_daily(fundamentals_by_date, start_dt, end_dt)

        # Convert to tensors
        daily_tensors = {}
        for date, metrics in daily_fundamentals.items():
            values = [metrics.get(m, 0.0) if metrics.get(m) is not None else 0.0 for m in METRIC_ORDER]
            daily_tensors[date] = torch.tensor(values, dtype=torch.float32)

        all_stock_fundamentals[ticker_symbol] = daily_tensors

        # Check if approaching daily limit
        if api_call_count >= 240:
            print(f"\n⚠️  Approaching daily API limit (240/250 calls used)")
            print(f"Processed {len(all_stock_fundamentals)} stocks so far")
            print(f"Last ticker processed: {ticker_symbol}")
            print(f"\nTo resume tomorrow, run:")
            print(f"  python generate_fmp_fundamentals.py --resume {ticker_symbol}")
            break

    print(f"\n✅ Generated fundamentals for {len(all_stock_fundamentals)} stocks")
    print(f"API calls used: {api_call_count}")

    return all_stock_fundamentals


def normalize_fundamentals(all_stock_fundamentals):
    """Normalize using robust scaling."""
    print("\nNormalizing...")

    all_values = [[] for _ in range(27)]
    for ticker, daily_tensors in all_stock_fundamentals.items():
        for date, tensor in daily_tensors.items():
            for i, val in enumerate(tensor):
                if not np.isnan(val) and not np.isinf(val):
                    all_values[i].append(val.item())

    medians = []
    iqrs = []
    for values in all_values:
        if values:
            median = np.median(values)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1 if (q3 - q1) != 0 else 1.0
        else:
            median, iqr = 0.0, 1.0
        medians.append(median)
        iqrs.append(iqr)

    medians = np.array(medians)
    iqrs = np.array(iqrs)

    normalized = {}
    for ticker, daily_tensors in all_stock_fundamentals.items():
        normalized[ticker] = {}
        for date, tensor in daily_tensors.items():
            norm_values = (tensor.numpy() - medians) / iqrs
            norm_values = np.clip(norm_values, -10, 10)
            normalized[ticker][date] = torch.tensor(norm_values, dtype=torch.float32)

    return normalized


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='s_lot', choices=['a_lot', 's_lot'])
    parser.add_argument('--api_key', type=str, default=API_KEY)
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2025-12-31')
    parser.add_argument('--resume', type=str, default=None, help='Resume from ticker')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.api_key == "YOUR_API_KEY":
        print("❌ Please set your FMP API key!")
        print("Edit this file or use: --api_key YOUR_KEY")
        return

    stock_dict = s_lot_of_stocks if args.dataset == 's_lot' else a_lot_of_stocks
    dataset_name = 's_lot_of_stocks' if args.dataset == 's_lot' else 'a_lot_of_stocks'

    fundamentals = generate_fmp_fundamentals(
        stock_dict, dataset_name, args.api_key,
        args.start, args.end, args.resume
    )

    fundamentals = normalize_fundamentals(fundamentals)

    output_path = args.output or f'/home/james/Desktop/Stock-Prediction/{dataset_name}_fmp_fundamentals.pkl'
    save_pickle(fundamentals, output_path)

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
