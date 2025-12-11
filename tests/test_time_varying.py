"""
Quick test of time-varying fundamental extraction.
"""

import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_time_varying_fundamentals import (
    extract_time_varying_fundamentals,
    forward_fill_to_daily,
    METRIC_ORDER
)
import pandas as pd
import torch

# Test with a single stock
print("Testing time-varying fundamental extraction...")
print("="*80)

ticker = "AAPL"
print(f"\nExtracting fundamentals for {ticker}...")

fundamentals_by_date = extract_time_varying_fundamentals(ticker)

if fundamentals_by_date:
    print(f"✅ Found {len(fundamentals_by_date)} reporting dates")

    # Show dates and sample metrics
    for date in sorted(fundamentals_by_date.keys()):
        metrics = fundamentals_by_date[date]
        gross_margin = metrics.get('grossMargins')
        operating_margin = metrics.get('operatingMargins')
        roe = metrics.get('returnOnEquity')

        print(f"\n{date.date()}:")
        print(f"  Gross Margin: {gross_margin*100:.2f}%" if gross_margin else "  Gross Margin: N/A")
        print(f"  Operating Margin: {operating_margin*100:.2f}%" if operating_margin else "  Operating Margin: N/A")
        print(f"  ROE: {roe*100:.2f}%" if roe else "  ROE: N/A")

    # Test forward-fill
    print("\n" + "="*80)
    print("Testing forward-fill to daily...")
    print("="*80)

    start_date = pd.to_datetime('2023-01-01').date()
    end_date = pd.to_datetime('2025-12-31').date()

    daily_fundamentals = forward_fill_to_daily(fundamentals_by_date, start_date, end_date)

    print(f"\n✅ Forward-filled to {len(daily_fundamentals)} days")
    print(f"Date range: {min(daily_fundamentals.keys())} to {max(daily_fundamentals.keys())}")

    # Show how metrics change over time
    print("\nSample: Gross Margin over time (checking for changes):")
    sample_dates = sorted(daily_fundamentals.keys())[::180]  # Sample every ~6 months
    for date in sample_dates[:6]:
        metrics = daily_fundamentals[date]
        gm = metrics.get('grossMargins')
        print(f"  {date}: {gm*100:.2f}%" if gm else f"  {date}: N/A")

    # Convert to tensor
    print("\n" + "="*80)
    print("Testing tensor conversion...")
    print("="*80)

    sample_date = sorted(daily_fundamentals.keys())[0]
    sample_metrics = daily_fundamentals[sample_date]

    values = []
    for metric in METRIC_ORDER:
        val = sample_metrics.get(metric, None)
        values.append(val if val is not None else 0.0)

    tensor = torch.tensor(values, dtype=torch.float32)
    print(f"\n✅ Tensor shape: {tensor.shape}")
    print(f"First 5 values: {tensor[:5]}")

    print("\n" + "="*80)
    print("TEST SUCCESSFUL!")
    print("="*80)
else:
    print("❌ No fundamentals found")
