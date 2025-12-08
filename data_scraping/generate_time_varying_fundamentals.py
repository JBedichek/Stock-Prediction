"""
Generate time-varying fundamental metrics using quarterly/annual financial statements.

This creates a dict[ticker: dict[date: tensor]] where fundamentals change over time.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
import pickle

import sys
sys.path.append('/home/james/Desktop/Stock-Prediction')
from .Stock import a_lot_of_stocks, s_lot_of_stocks
from utils.utils import save_pickle


def extract_time_varying_fundamentals(ticker_symbol):
    """
    Extract time-varying fundamentals from yfinance for a single stock.

    Returns:
        dict[date: dict[metric: value]] - Fundamentals at each reporting date
    """
    try:
        ticker = yf.Ticker(ticker_symbol)

        # Get financial statements
        annual_income = ticker.income_stmt  # 4-5 years
        annual_balance = ticker.balance_sheet
        annual_cashflow = ticker.cashflow

        quarterly_income = ticker.quarterly_income_stmt  # 1 year
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow

        # Combine annual and quarterly (quarterly overrides annual for overlap)
        fundamentals_by_date = {}

        # Process annual data first (goes back further)
        if annual_income is not None and not annual_income.empty:
            for date in annual_income.columns:
                fundamentals_by_date[date] = calculate_metrics(
                    annual_income[date],
                    annual_balance[date] if annual_balance is not None and date in annual_balance.columns else None,
                    annual_cashflow[date] if annual_cashflow is not None and date in annual_cashflow.columns else None
                )

        # Process quarterly data (more recent, overrides annual)
        if quarterly_income is not None and not quarterly_income.empty:
            for date in quarterly_income.columns:
                fundamentals_by_date[date] = calculate_metrics(
                    quarterly_income[date],
                    quarterly_balance[date] if quarterly_balance is not None and date in quarterly_balance.columns else None,
                    quarterly_cashflow[date] if quarterly_cashflow is not None and date in quarterly_cashflow.columns else None
                )

        return fundamentals_by_date

    except Exception as e:
        print(f"  Error for {ticker_symbol}: {e}")
        return {}


def calculate_metrics(income, balance, cashflow):
    """
    Calculate fundamental metrics from financial statements.

    Returns 27 metrics matching the original static implementation.
    """
    metrics = {}

    # Helper to safely get values
    def safe_get(series, key):
        if series is None:
            return None
        try:
            if key in series.index:
                val = series[key]
                if pd.notna(val) and not np.isinf(val):
                    return float(val)
        except:
            pass
        return None

    # Income statement metrics
    total_revenue = safe_get(income, 'Total Revenue')
    gross_profit = safe_get(income, 'Gross Profit')
    operating_income = safe_get(income, 'Operating Income')
    net_income = safe_get(income, 'Net Income')
    ebitda = safe_get(income, 'EBITDA')

    # Balance sheet metrics
    total_assets = safe_get(balance, 'Total Assets')
    stockholder_equity = safe_get(balance, 'Stockholders Equity')
    total_debt = safe_get(balance, 'Total Debt')
    current_assets = safe_get(balance, 'Current Assets')
    current_liabilities = safe_get(balance, 'Current Liabilities')
    cash = safe_get(balance, 'Cash And Cash Equivalents')

    # Cash flow metrics
    operating_cashflow = safe_get(cashflow, 'Operating Cash Flow')
    free_cashflow = safe_get(cashflow, 'Free Cash Flow')

    # Calculate derived metrics (matching original 27 metrics)

    # Profitability margins
    metrics['grossMargins'] = gross_profit / total_revenue if (gross_profit and total_revenue) else None
    metrics['operatingMargins'] = operating_income / total_revenue if (operating_income and total_revenue) else None
    metrics['profitMargins'] = net_income / total_revenue if (net_income and total_revenue) else None
    metrics['ebitdaMargins'] = ebitda / total_revenue if (ebitda and total_revenue) else None

    # Returns
    metrics['returnOnEquity'] = net_income / stockholder_equity if (net_income and stockholder_equity) else None
    metrics['returnOnAssets'] = net_income / total_assets if (net_income and total_assets) else None

    # Financial health
    metrics['currentRatio'] = current_assets / current_liabilities if (current_assets and current_liabilities) else None
    metrics['quickRatio'] = (current_assets - safe_get(balance, 'Inventory')) / current_liabilities if current_liabilities else None
    metrics['debtToEquity'] = total_debt / stockholder_equity if (total_debt and stockholder_equity) else None

    # Absolute values (scaled down for normalization)
    metrics['totalCash'] = cash / 1e9 if cash else None  # Billions
    metrics['totalDebt'] = total_debt / 1e9 if total_debt else None  # Billions
    metrics['operatingCashflow'] = operating_cashflow / 1e9 if operating_cashflow else None  # Billions
    metrics['marketCap'] = None  # Not available in statements

    # Revenue/earnings metrics
    metrics['revenuePerShare'] = None  # Need share count
    metrics['trailingEps'] = None  # Need share count
    metrics['bookValue'] = stockholder_equity / safe_get(balance, 'Ordinary Shares Number') if stockholder_equity else None

    # Valuation ratios (not directly available from statements)
    metrics['forwardPE'] = None
    metrics['priceToBook'] = None
    metrics['priceToSalesTrailing12Months'] = None
    metrics['enterpriseToRevenue'] = None
    metrics['enterpriseToEbitda'] = None
    metrics['forwardEps'] = None
    metrics['payoutRatio'] = None

    # Market metrics (not available from statements)
    metrics['beta'] = None
    metrics['fiftyTwoWeekHigh'] = None
    metrics['fiftyTwoWeekLow'] = None

    # Revenue growth (need previous period - will calculate later)
    metrics['revenueGrowth'] = None

    return metrics


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
    """
    Forward-fill quarterly/annual fundamentals to daily frequency.

    Args:
        fundamentals_by_date: dict[reporting_date: dict[metric: value]]
        start_date: Start date for daily data
        end_date: End date for daily data

    Returns:
        dict[daily_date: dict[metric: value]]
    """
    if not fundamentals_by_date:
        return {}

    # Sort reporting dates
    reporting_dates = sorted(fundamentals_by_date.keys())

    # Create daily series
    daily_fundamentals = {}
    current_date = start_date
    current_fundamentals = None
    reporting_idx = 0

    while current_date <= end_date:
        # Check if we've passed a new reporting date
        while reporting_idx < len(reporting_dates) and reporting_dates[reporting_idx] <= pd.Timestamp(current_date):
            current_fundamentals = fundamentals_by_date[reporting_dates[reporting_idx]]
            reporting_idx += 1

        # Use current fundamentals (forward-filled)
        if current_fundamentals is not None:
            daily_fundamentals[current_date] = current_fundamentals.copy()

        current_date += timedelta(days=1)

    return daily_fundamentals


def generate_time_varying_fundamentals(stock_dict, dataset_name, start_date='2020-01-01', end_date='2025-12-31'):
    """
    Generate time-varying fundamentals for all stocks in dataset.

    Returns:
        dict[ticker: dict[date: tensor]] - Time-varying fundamental tensors
    """
    print(f"\n{'='*80}")
    print(f"GENERATING TIME-VARYING FUNDAMENTALS FOR {dataset_name}")
    print(f"{'='*80}")
    print(f"Dataset: {len(stock_dict)} stocks")
    print(f"Date range: {start_date} to {end_date}")

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    all_stock_fundamentals = {}

    for ticker_symbol, company_name in tqdm(stock_dict.items(), desc="Processing stocks"):
        # Extract quarterly/annual fundamentals
        fundamentals_by_date = extract_time_varying_fundamentals(ticker_symbol)

        if not fundamentals_by_date:
            continue

        # Forward-fill to daily
        daily_fundamentals = forward_fill_to_daily(fundamentals_by_date, start_dt, end_dt)

        # Convert to tensors
        daily_tensors = {}
        for date, metrics in daily_fundamentals.items():
            # Create tensor in consistent order
            values = []
            for metric in METRIC_ORDER:
                val = metrics.get(metric, None)
                values.append(val if val is not None else 0.0)
            daily_tensors[date] = torch.tensor(values, dtype=torch.float32)

        all_stock_fundamentals[ticker_symbol] = daily_tensors

    print(f"\n✅ Generated time-varying fundamentals for {len(all_stock_fundamentals)} stocks")

    # Show sample
    if all_stock_fundamentals:
        sample_ticker = list(all_stock_fundamentals.keys())[0]
        sample_dates = sorted(all_stock_fundamentals[sample_ticker].keys())
        print(f"\nSample ({sample_ticker}):")
        print(f"  Date range: {sample_dates[0]} to {sample_dates[-1]}")
        print(f"  Number of days: {len(sample_dates)}")
        print(f"  Tensor shape: {all_stock_fundamentals[sample_ticker][sample_dates[0]].shape}")

    return all_stock_fundamentals


def normalize_time_varying_fundamentals(all_stock_fundamentals):
    """
    Normalize fundamentals using robust scaling across all stocks and dates.
    """
    print("\nNormalizing time-varying fundamentals...")

    # Collect all values for each metric
    all_values = [[] for _ in range(27)]

    for ticker, daily_tensors in all_stock_fundamentals.items():
        for date, tensor in daily_tensors.items():
            for i, val in enumerate(tensor):
                if not np.isnan(val) and not np.isinf(val):
                    all_values[i].append(val.item())

    # Calculate robust statistics
    medians = []
    iqrs = []
    for values in all_values:
        if values:
            median = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            if iqr == 0:
                iqr = 1.0
        else:
            median = 0.0
            iqr = 1.0
        medians.append(median)
        iqrs.append(iqr)

    medians = np.array(medians)
    iqrs = np.array(iqrs)

    # Normalize all tensors
    normalized = {}
    for ticker, daily_tensors in all_stock_fundamentals.items():
        normalized[ticker] = {}
        for date, tensor in daily_tensors.items():
            norm_values = (tensor.numpy() - medians) / iqrs
            norm_values = np.clip(norm_values, -10, 10)
            normalized[ticker][date] = torch.tensor(norm_values, dtype=torch.float32)

    print("✅ Normalization complete")
    return normalized


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='s_lot', choices=['a_lot', 's_lot'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2025-12-31')
    args = parser.parse_args()

    # Select dataset
    if args.dataset == 'a_lot':
        stock_dict = a_lot_of_stocks
        dataset_name = 'a_lot_of_stocks'
    else:
        stock_dict = s_lot_of_stocks
        dataset_name = 's_lot_of_stocks'

    # Generate
    fundamentals = generate_time_varying_fundamentals(stock_dict, dataset_name, args.start, args.end)

    # Normalize
    fundamentals = normalize_time_varying_fundamentals(fundamentals)

    # Save
    output_path = args.output or f'/home/james/Desktop/Stock-Prediction/{dataset_name}_time_varying_fundamentals.pkl'
    print(f"\nSaving to: {output_path}")
    save_pickle(fundamentals, output_path)

    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
