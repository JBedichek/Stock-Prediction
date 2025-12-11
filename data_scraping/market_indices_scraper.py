"""
Market Indices and Sector ETF Scraper

Collects daily OHLCV data for market indices and sector ETFs.
This data is used to create market-relative features for individual stocks.

Data collected ONCE for all stocks (not per-stock), making it very efficient.
"""

import requests
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import time
import sys
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import save_pickle, pic_load


class MarketIndicesScraper:
    """
    Scraper for market indices and sector ETFs using FMP API.
    """

    # Major market indices
    MARKET_INDICES = {
        'SP500': '^GSPC',      # S&P 500
        'NASDAQ': '^IXIC',      # Nasdaq Composite
        'DJI': '^DJI',          # Dow Jones Industrial
        'RUSSELL2000': '^RUT',  # Russell 2000 (small cap)
        'VIX': '^VIX',          # Volatility Index
    }

    # Sector ETFs (SPDR Select Sector)
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services',
    }

    # Additional useful ETFs
    ADDITIONAL_ETFS = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq-100 ETF',
        'IWM': 'Russell 2000 ETF',
        'DIA': 'Dow Jones ETF',
        'TLT': 'Long-term Treasury',
        'GLD': 'Gold',
        'USO': 'Oil',
    }

    def __init__(self, api_key: str, rate_limit_delay: float = 0.2):
        """
        Initialize scraper.

        Args:
            api_key: FMP API key
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        self.rate_limit_delay = rate_limit_delay
        self.call_count = 0

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[any]:
        """Make API request with error handling."""
        if params is None:
            params = {}

        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params, timeout=15)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()
                return data if data else None
            elif response.status_code == 429:
                print(f"  ⚠️  Rate limit hit. Waiting 60s...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                print(f"  ⚠️  API Error {response.status_code} for {endpoint}")
                return None
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return None

    def get_index_prices(self, symbol: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for an index/ETF.

        Args:
            symbol: Index symbol (e.g., '^GSPC', 'SPY')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, change, changePercent
        """
        endpoint = "historical-price-eod/full"
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date
        }

        data = self._make_request(endpoint, params)

        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                return df

        return None

    def scrape_all_indices(self, from_date: str, to_date: str = None,
                          include_sector_etfs: bool = True,
                          include_additional: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Scrape all market indices and ETFs.

        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (defaults to today)
            include_sector_etfs: Include sector ETFs
            include_additional: Include additional ETFs (treasuries, gold, etc.)

        Returns:
            Dict of {symbol: DataFrame}
        """
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n{'='*80}")
        print(f"SCRAPING MARKET INDICES AND SECTOR ETFs")
        print(f"{'='*80}")
        print(f"Date range: {from_date} to {to_date}")
        print(f"{'='*80}\n")

        all_data = {}
        symbols_to_scrape = {}

        # Collect symbols
        symbols_to_scrape.update({k: f"Index - {k}" for k in self.MARKET_INDICES.keys()})

        if include_sector_etfs:
            symbols_to_scrape.update({k: f"Sector ETF - {v}" for k, v in self.SECTOR_ETFS.items()})

        if include_additional:
            symbols_to_scrape.update({k: f"Additional ETF - {v}" for k, v in self.ADDITIONAL_ETFS.items()})

        # Get actual symbols (some are in dict values for indices)
        actual_symbols = {}
        for key, description in symbols_to_scrape.items():
            if key in self.MARKET_INDICES:
                actual_symbols[key] = (self.MARKET_INDICES[key], description)
            else:
                actual_symbols[key] = (key, description)

        total = len(actual_symbols)
        print(f"Scraping {total} symbols...\n")

        for i, (key, (symbol, description)) in enumerate(actual_symbols.items()):
            print(f"[{i+1}/{total}] {key} ({description})")

            df = self.get_index_prices(symbol, from_date, to_date)

            if df is not None and not df.empty:
                all_data[key] = df
                print(f"  ✅ Got {len(df)} days of data (from {df['date'].min().date()} to {df['date'].max().date()})")
            else:
                print(f"  ⚠️  No data retrieved")

        print(f"\n{'='*80}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*80}")
        print(f"Successfully scraped: {len(all_data)}/{total} symbols")
        print(f"Total API calls: {self.call_count}")
        print(f"{'='*80}\n")

        return all_data

    def save_data(self, data: Dict[str, pd.DataFrame], output_file: str):
        """Save scraped data to pickle file."""
        save_pickle(data, output_file)
        print(f"✅ Data saved to {output_file}")


def scrape_market_data(api_key: str, from_date: str, to_date: str = None,
                      output_file: str = 'market_indices_data.pkl') -> Dict[str, pd.DataFrame]:
    """
    Convenience function to scrape all market data.

    Args:
        api_key: FMP API key
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (defaults to today)
        output_file: Output pickle file

    Returns:
        Dict of {symbol: DataFrame}
    """
    scraper = MarketIndicesScraper(api_key)
    data = scraper.scrape_all_indices(
        from_date=from_date,
        to_date=to_date,
        include_sector_etfs=True,
        include_additional=True
    )

    scraper.save_data(data, output_file)
    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Scrape market indices and sector ETFs')
    parser.add_argument('--api_key', type=str, default="YOUR_FMP_API_KEY_HERE",
                       help='FMP API key')
    parser.add_argument('--from_date', type=str, default='2000-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to_date', type=str, default=None,
                       help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--output', type=str, default='market_indices_data.pkl',
                       help='Output pickle file')

    args = parser.parse_args()

    data = scrape_market_data(
        api_key=args.api_key,
        from_date=args.from_date,
        to_date=args.to_date,
        output_file=args.output
    )

    print(f"\n✅ Done! Market data saved to {args.output}")
    print(f"\nSymbols collected: {list(data.keys())}")
