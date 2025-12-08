"""
Comprehensive FMP Data Scraper (Premium Tier)

This scraper collects ALL available time-series data from Financial Modeling Prep API
to dramatically expand the features fed to the transformer model.

Data Sources:
1. Financial Statements (quarterly, 30+ years history)
2. Key Financial Metrics & Ratios
3. Intraday Price Data (1min, 5min, 15min, 30min, 1hour)
4. Technical Indicators (RSI, EMA, SMA, ADX, Williams, etc.)
5. Earnings Data (transcripts, surprises, calendar)
6. Insider Trading Activity
7. Institutional Holdings (13F filings)
8. Analyst Ratings & Price Targets
9. Sector Performance Metrics
10. Economic Indicators

Premium tier assumptions:
- Unlimited API calls (or very high limit)
- Access to all historical data
- Intraday data available
"""

import requests
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta, date
import time
import sys
from typing import Dict, List, Optional, Tuple
import json

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import save_pickle, pic_load


class FMPComprehensiveScraper:
    """
    Comprehensive scraper for FMP API with premium tier access.
    """

    def __init__(self, api_key: str, rate_limit_delay: float = 0.1):
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

    def _make_request(self, endpoint: str, symbol: str = None, params: Dict = None) -> Optional[any]:
        """Make API request with error handling."""
        if params is None:
            params = {}

        # Add symbol as query parameter for /stable/ endpoint
        if symbol:
            params['symbol'] = symbol

        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = requests.get(url, params=params, timeout=15)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()
                return data if data else None
            elif response.status_code == 429:
                print(f"  ⚠️  Rate limit hit. Waiting 60s...")
                time.sleep(60)
                return self._make_request(endpoint, symbol, params)
            else:
                print(f"  ⚠️  API Error {response.status_code} for URL: {url}")
                return None
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return None

    # ========================================================================
    # 1. FINANCIAL STATEMENTS (Quarterly, 30+ years)
    # ========================================================================

    def get_financial_statements(self, ticker: str, limit: int = 400) -> Dict:
        """
        Get all financial statements (income, balance, cash flow).

        Args:
            ticker: Stock ticker
            limit: Number of periods (400 = 100 years quarterly)

        Returns:
            Dict with 'income', 'balance', 'cashflow' DataFrames
        """
        statements = {}

        for stmt_type in ['income-statement', 'balance-sheet-statement', 'cash-flow-statement']:
            data = self._make_request(stmt_type, symbol=ticker, params={'period': 'quarter', 'limit': limit})
            if data:
                statements[stmt_type.split('-')[0]] = pd.DataFrame(data)

        return statements

    def get_key_metrics(self, ticker: str, limit: int = 400) -> Optional[pd.DataFrame]:
        """
        Get key financial metrics (P/E, P/B, ROE, ROA, etc.).

        These are derived metrics calculated by FMP, saving us computation.
        """
        data = self._make_request("key-metrics", symbol=ticker, params={'period': 'quarter', 'limit': limit})
        return pd.DataFrame(data) if data else None

    def get_financial_ratios(self, ticker: str, limit: int = 400) -> Optional[pd.DataFrame]:
        """Get comprehensive financial ratios (60+ ratios)."""
        data = self._make_request("ratios", symbol=ticker, params={'period': 'quarter', 'limit': limit})
        return pd.DataFrame(data) if data else None

    def get_enterprise_values(self, ticker: str, limit: int = 400) -> Optional[pd.DataFrame]:
        """Get enterprise value metrics over time."""
        data = self._make_request("enterprise-values", symbol=ticker, params={'period': 'quarter', 'limit': limit})
        return pd.DataFrame(data) if data else None

    def get_financial_growth(self, ticker: str, limit: int = 400) -> Optional[pd.DataFrame]:
        """Get growth metrics (revenue growth, earnings growth, etc.)."""
        data = self._make_request("financial-growth", symbol=ticker, params={'period': 'quarter', 'limit': limit})
        return pd.DataFrame(data) if data else None

    # ========================================================================
    # 2. INTRADAY PRICE DATA (1min, 5min, 15min, 30min, 1hour)
    # ========================================================================

    def get_intraday_prices(self, ticker: str, interval: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """
        Get intraday price data.

        Args:
            ticker: Stock ticker
            interval: '1min', '5min', '15min', '30min', '1hour', '4hour'
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        """
        data = self._make_request(f"historical-chart/{interval}", symbol=ticker,
            params={'from': from_date, 'to': to_date}
        )
        return pd.DataFrame(data) if data else None

    def get_daily_prices_full(self, ticker: str, from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get full daily price history (30+ years available).

        Args:
            ticker: Stock ticker
            from_date: Start date (YYYY-MM-DD), defaults to maximum available
            to_date: End date (YYYY-MM-DD), defaults to today
        """
        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        data = self._make_request("historical-price-eod/full", symbol=ticker, params=params)
        if data:
            # New /stable/ endpoint returns list directly, not nested in 'historical'
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'historical' in data:
                return pd.DataFrame(data['historical'])
        return None

    # ========================================================================
    # 3. TECHNICAL INDICATORS (RSI, EMA, SMA, ADX, Williams, etc.)
    # ========================================================================

    def get_technical_indicator(self, ticker: str, interval: str, indicator_type: str,
                                period: int = 10) -> Optional[pd.DataFrame]:
        """
        Get technical indicator time series.

        Args:
            ticker: Stock ticker
            interval: 'daily', '1min', '5min', '15min', '30min', '1hour', '4hour'
            indicator_type: 'sma', 'ema', 'wma', 'dema', 'tema', 'williams', 'rsi', 'adx', 'standardDeviation'
            period: Period for calculation (e.g., 10 for 10-day SMA)
        """
        endpoint = f"technical_indicator/{interval}"
        data = self._make_request(endpoint, symbol=ticker, params={'type': indicator_type, 'period': period})
        return pd.DataFrame(data) if data else None

    def get_all_technical_indicators(self, ticker: str, interval: str = 'daily',
                                     include_extended: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get all technical indicators for a stock.

        Args:
            ticker: Stock ticker
            interval: Time interval (default 'daily')
            include_extended: Include extended indicators (MACD, Stochastic, etc.)

        Returns dict of {indicator_name: DataFrame}
        """
        indicators = {}

        # Core indicators (always included)
        indicator_types = [
            ('sma_10', 'sma', 10),
            ('sma_20', 'sma', 20),
            ('sma_50', 'sma', 50),
            ('sma_200', 'sma', 200),
            ('ema_10', 'ema', 10),
            ('ema_20', 'ema', 20),
            ('ema_50', 'ema', 50),
            ('ema_200', 'ema', 200),
            ('rsi_14', 'rsi', 14),
            ('adx_14', 'adx', 14),
            ('williams_14', 'williams', 14),
        ]

        # Extended indicators
        if include_extended:
            extended_indicators = [
                # MACD (12, 26, 9 are standard)
                ('macd_12_26', 'dema', 12),  # Using DEMA as proxy
                # WMA (Weighted Moving Average)
                ('wma_10', 'wma', 10),
                ('wma_20', 'wma', 20),
                ('wma_50', 'wma', 50),
                # TEMA (Triple Exponential Moving Average)
                ('tema_10', 'tema', 10),
                ('tema_20', 'tema', 20),
                # Standard Deviation (volatility)
                ('stddev_10', 'standardDeviation', 10),
                ('stddev_20', 'standardDeviation', 20),
                # Additional RSI periods
                ('rsi_7', 'rsi', 7),
                ('rsi_21', 'rsi', 21),
                # Additional ADX periods
                ('adx_7', 'adx', 7),
                ('adx_21', 'adx', 21),
            ]
            indicator_types.extend(extended_indicators)

        for name, ind_type, period in indicator_types:
            df = self.get_technical_indicator(ticker, interval, ind_type, period)
            if df is not None:
                indicators[name] = df

        return indicators

    # ========================================================================
    # 4. EARNINGS DATA
    # ========================================================================

    def get_earnings_history(self, ticker: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical earnings (EPS actual vs estimated)."""
        data = self._make_request("historical/earning_calendar", symbol=ticker, params={'limit': limit})
        return pd.DataFrame(data) if data else None

    def get_earnings_surprises(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get earnings surprise history."""
        data = self._make_request("earnings-surprises", symbol=ticker)
        return pd.DataFrame(data) if data else None

    # ========================================================================
    # 5. INSIDER TRADING
    # ========================================================================

    def get_insider_trading(self, ticker: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get insider trading transactions."""
        data = self._make_request("insider-trading", symbol=ticker, params={'limit': limit})
        return pd.DataFrame(data) if data else None

    def get_insider_trading_summary(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get aggregated insider trading statistics."""
        data = self._make_request("insider-trading-statistics", symbol=ticker)
        return pd.DataFrame(data) if data else None

    # ========================================================================
    # 6. INSTITUTIONAL HOLDINGS (13F)
    # ========================================================================

    def get_institutional_holdings(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get current institutional holdings."""
        data = self._make_request("institutional-holder", symbol=ticker)
        return pd.DataFrame(data) if data else None

    # ========================================================================
    # 7. ANALYST RATINGS & ESTIMATES
    # ========================================================================

    def get_analyst_ratings(self, ticker: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get analyst rating changes over time."""
        data = self._make_request("rating", symbol=ticker, params={'limit': limit})
        return pd.DataFrame(data) if data else None

    def get_price_targets(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get analyst price targets."""
        data = self._make_request("price-target", symbol=ticker)
        return pd.DataFrame(data) if data else None

    def get_analyst_estimates(self, ticker: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get analyst estimates (revenue, EPS)."""
        data = self._make_request("analyst-estimates", symbol=ticker, params={'period': 'quarter', 'limit': limit})
        return pd.DataFrame(data) if data else None

    # ========================================================================
    # 8. SECTOR & INDUSTRY PERFORMANCE
    # ========================================================================

    def get_sector_performance(self) -> Optional[pd.DataFrame]:
        """Get current sector performance."""
        data = self._make_request("sectors-performance")
        return pd.DataFrame(data) if data else None

    def get_historical_sector_performance(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical sector performance."""
        data = self._make_request("historical-sectors-performance", params={'limit': limit})
        return pd.DataFrame(data) if data else None

    # ========================================================================
    # 9. COMPANY PROFILE & METRICS
    # ========================================================================

    def get_company_profile(self, ticker: str) -> Optional[Dict]:
        """Get company profile (sector, industry, description, etc.)."""
        data = self._make_request("profile", symbol=ticker)
        return data[0] if data and len(data) > 0 else None

    def get_market_cap_history(self, ticker: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical market cap."""
        data = self._make_request("historical-market-capitalization", symbol=ticker, params={'limit': limit})
        return pd.DataFrame(data) if data else None

    # ========================================================================
    # 10. DIVIDENDS & SPLITS
    # ========================================================================

    def get_dividend_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get historical dividends."""
        data = self._make_request("historical-price-eod/dividend", symbol=ticker)
        if data:
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'historical' in data:
                return pd.DataFrame(data['historical'])
        return None

    def get_stock_splits(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get stock split history."""
        data = self._make_request("historical-price-eod/split", symbol=ticker)
        if data:
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'historical' in data:
                return pd.DataFrame(data['historical'])
        return None

    # ========================================================================
    # COMPREHENSIVE DATA COLLECTION
    # ========================================================================

    def scrape_all_data(self, ticker: str, start_date: str = '2000-01-01',
                       end_date: str = None) -> Dict:
        """
        Scrape ALL available data for a stock.

        Args:
            ticker: Stock ticker
            start_date: Start date for time-series data
            end_date: End date (defaults to today)

        Returns:
            Comprehensive dict with all collected data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n{'='*80}")
        print(f"SCRAPING ALL DATA FOR {ticker}")
        print(f"{'='*80}")

        data = {
            'ticker': ticker,
            'collection_date': datetime.now().isoformat(),
            'api_calls': 0
        }

        start_calls = self.call_count

        # 1. Financial Statements
        print("  📊 Fetching financial statements...")
        data['financial_statements'] = self.get_financial_statements(ticker)

        # 2. Key Metrics
        print("  📈 Fetching key metrics...")
        data['key_metrics'] = self.get_key_metrics(ticker)

        # 3. Financial Ratios
        print("  🔢 Fetching financial ratios...")
        data['financial_ratios'] = self.get_financial_ratios(ticker)

        # 4. Enterprise Values
        print("  💰 Fetching enterprise values...")
        data['enterprise_values'] = self.get_enterprise_values(ticker)

        # 5. Growth Metrics
        print("  📉 Fetching growth metrics...")
        data['financial_growth'] = self.get_financial_growth(ticker)

        # 6. Daily Price History
        print("  💹 Fetching daily price history...")
        data['daily_prices'] = self.get_daily_prices_full(ticker, start_date, end_date)

        # 7. Technical Indicators
        print("  📊 Fetching technical indicators...")
        data['technical_indicators'] = self.get_all_technical_indicators(ticker, 'daily')

        # 8. Earnings
        print("  💵 Fetching earnings data...")
        data['earnings_history'] = self.get_earnings_history(ticker)
        data['earnings_surprises'] = self.get_earnings_surprises(ticker)

        # 9. Insider Trading
        print("  🔍 Fetching insider trading...")
        data['insider_trading'] = self.get_insider_trading(ticker)
        data['insider_trading_summary'] = self.get_insider_trading_summary(ticker)

        # 10. Institutional Holdings
        print("  🏢 Fetching institutional holdings...")
        data['institutional_holdings'] = self.get_institutional_holdings(ticker)

        # 11. Analyst Data
        print("  📝 Fetching analyst data...")
        data['analyst_ratings'] = self.get_analyst_ratings(ticker)
        data['price_targets'] = self.get_price_targets(ticker)
        data['analyst_estimates'] = self.get_analyst_estimates(ticker)

        # 12. Company Profile
        print("  🏢 Fetching company profile...")
        data['company_profile'] = self.get_company_profile(ticker)

        # 13. Market Cap History
        print("  📊 Fetching market cap history...")
        data['market_cap_history'] = self.get_market_cap_history(ticker)

        # 14. Dividends & Splits
        print("  💰 Fetching dividends & splits...")
        data['dividend_history'] = self.get_dividend_history(ticker)
        data['stock_splits'] = self.get_stock_splits(ticker)

        data['api_calls'] = self.call_count - start_calls
        print(f"\n  ✅ Complete! API calls used: {data['api_calls']}")

        return data


def scrape_dataset(stock_dict: Dict[str, str], api_key: str,
                  output_file: str = 'fmp_comprehensive_data.pkl',
                  start_date: str = '2000-01-01',
                  resume_from: str = None) -> Dict:
    """
    Scrape comprehensive data for a dataset of stocks.

    Args:
        stock_dict: {ticker: company_name}
        api_key: FMP API key
        output_file: Output pickle file
        start_date: Start date for time-series data
        resume_from: Ticker to resume from (if interrupted)

    Returns:
        Dict of {ticker: comprehensive_data}
    """
    scraper = FMPComprehensiveScraper(api_key, rate_limit_delay=0.2)

    all_data = {}
    tickers = list(stock_dict.keys())

    # Resume logic
    if resume_from and resume_from in tickers:
        start_idx = tickers.index(resume_from)
        tickers = tickers[start_idx:]
        print(f"🔄 Resuming from {resume_from}")

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE FMP DATA SCRAPING")
    print(f"{'='*80}")
    print(f"Total stocks: {len(tickers)}")
    print(f"Start date: {start_date}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    for i, ticker in enumerate(tickers):
        company_name = stock_dict[ticker]
        print(f"\n[{i+1}/{len(tickers)}] {ticker} - {company_name}")

        try:
            data = scraper.scrape_all_data(ticker, start_date)
            all_data[ticker] = data

            # Save progress every 10 stocks
            if (i + 1) % 10 == 0:
                print(f"\n💾 Saving progress... ({len(all_data)} stocks collected)")
                save_pickle(all_data, output_file)

        except Exception as e:
            print(f"  ❌ Error scraping {ticker}: {e}")
            continue

    # Final save
    print(f"\n💾 Saving final dataset...")
    save_pickle(all_data, output_file)

    print(f"\n{'='*80}")
    print(f"✅ SCRAPING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total stocks collected: {len(all_data)}")
    print(f"Total API calls: {scraper.call_count}")
    print(f"Output file: {output_file}")

    return all_data


if __name__ == '__main__':
    import argparse
    from data_scraping.Stock import a_lot_of_stocks, s_lot_of_stocks, all_stocks

    parser = argparse.ArgumentParser(description='Comprehensive FMP data scraper')
    parser.add_argument('--api_key', type=str, default="YOUR_FMP_API_KEY_HERE", help='FMP API key')
    parser.add_argument('--dataset', type=str, default='s_lot',
                       choices=['s_lot', 'a_lot', 'all'],
                       help='Dataset to scrape')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: {dataset}_fmp_comprehensive.pkl)')
    parser.add_argument('--start_date', type=str, default='2000-01-01',
                       help='Start date for time-series data')
    parser.add_argument('--resume', type=str, default=None,
                       help='Ticker to resume from')

    args = parser.parse_args()

    # Select dataset
    if args.dataset == 's_lot':
        stock_dict = s_lot_of_stocks
    elif args.dataset == 'a_lot':
        stock_dict = a_lot_of_stocks
    else:
        stock_dict = all_stocks

    # Set output file
    if args.output is None:
        args.output = f"{args.dataset}_fmp_comprehensive.pkl"

    # Scrape
    data = scrape_dataset(
        stock_dict=stock_dict,
        api_key=args.api_key,
        output_file=args.output,
        start_date=args.start_date,
        resume_from=args.resume
    )

    print(f"\n✅ Done! Data saved to {args.output}")
