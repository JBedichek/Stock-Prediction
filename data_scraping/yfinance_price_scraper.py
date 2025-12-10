"""
YFinance Price Data Scraper

Fetches daily OHLCV (Open, High, Low, Close, Volume) price data using yfinance.
This provides the critical missing price data needed for:
- Technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
- Price-derived features (returns, volatility, momentum)
- Target variables (future returns for prediction)

Advantages of yfinance:
- Free and reliable
- 20+ years of historical data
- Split-adjusted and dividend-adjusted
- No API key required
- Fast (parallel downloads)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import save_pickle, pic_load


def fetch_price_data(ticker: str, start_date: str, end_date: str,
                     retry_count: int = 3, retry_delay: float = 2.0) -> Optional[pd.DataFrame]:
    """
    Fetch daily price data for a single ticker using yfinance.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        retry_count: Number of retries on failure
        retry_delay: Delay between retries (seconds)

    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    for attempt in range(retry_count):
        try:
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)

            # Check if we got data
            if df.empty:
                return None

            # Rename columns to match FMP format
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            # Convert date to string format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Add ticker symbol
            df['symbol'] = ticker

            return df

        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
                continue
            else:
                # Final attempt failed
                return None

    return None


def scrape_price_dataset(stock_dict: Dict[str, str],
                         start_date: str,
                         end_date: str = None,
                         output_file: str = 'price_data.pkl',
                         batch_size: int = 100) -> Dict[str, pd.DataFrame]:
    """
    Scrape price data for multiple stocks.

    Args:
        stock_dict: Dict of {ticker: company_name}
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (defaults to today)
        output_file: Output pickle file
        batch_size: Number of stocks to process before saving

    Returns:
        Dict of {ticker: price_dataframe}
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"SCRAPING DAILY PRICE DATA (YFINANCE)")
    print(f"{'='*80}")
    print(f"Stocks: {len(stock_dict)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    # Check for existing progress
    import os
    all_prices = {}
    completed_tickers = set()

    if os.path.exists(output_file):
        try:
            all_prices = pic_load(output_file)
            completed_tickers = set(all_prices.keys())
            print(f"📂 Found existing progress: {len(completed_tickers)}/{len(stock_dict)} stocks completed")

            if len(completed_tickers) > 0:
                last_ticker = list(completed_tickers)[-1]
                print(f"   Last completed: {last_ticker}")
                print(f"   Resuming from next stock...\n")
        except Exception as e:
            print(f"⚠️  Could not load existing file: {e}")
            print(f"   Starting fresh...\n")
            all_prices = {}
            completed_tickers = set()

    # Track statistics
    successful = 0
    failed = 0
    failed_tickers = []

    # Process stocks
    stock_count = 0
    for ticker, company_name in tqdm(stock_dict.items(), desc="Fetching prices"):
        # Skip already completed
        if ticker in completed_tickers:
            continue

        stock_count += 1

        # Fetch price data
        df = fetch_price_data(ticker, start_date, end_date)

        if df is not None and not df.empty:
            all_prices[ticker] = df
            successful += 1
        else:
            # Save empty DataFrame for failed stocks
            all_prices[ticker] = pd.DataFrame()
            failed += 1
            failed_tickers.append(ticker)

        # Save progress periodically
        if stock_count % batch_size == 0:
            print(f"\n💾 Saving progress ({len(all_prices)} stocks)...")
            save_pickle(all_prices, output_file)

            # Show memory usage
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_gb = process.memory_info().rss / 1024**3
                print(f"   📊 RAM usage: {mem_gb:.2f} GB")
            except:
                pass

            print(f"   Progress: {len(all_prices)}/{len(stock_dict)} stocks")
            print(f"   Success: {successful}, Failed: {failed}\n")

    # Final save
    print(f"\n💾 Saving final data...")
    save_pickle(all_prices, output_file)

    print(f"\n{'='*80}")
    print(f"✅ PRICE SCRAPING COMPLETE!")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(stock_dict)} stocks ({successful/len(stock_dict)*100:.1f}%)")
    print(f"Failed: {failed}/{len(stock_dict)} stocks ({failed/len(stock_dict)*100:.1f}%)")

    if failed > 0 and failed <= 20:
        print(f"\nFailed tickers: {', '.join(failed_tickers)}")
    elif failed > 20:
        print(f"\nFirst 20 failed tickers: {', '.join(failed_tickers[:20])}")

    # Statistics
    if successful > 0:
        # Calculate average data points per stock
        total_days = 0
        for df in all_prices.values():
            if not df.empty:
                total_days += len(df)

        avg_days = total_days / successful if successful > 0 else 0
        print(f"\nAverage days per stock: {avg_days:.0f}")

    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    return all_prices


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from OHLCV data.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional technical indicator columns
    """
    df = df.copy()

    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by date
    df = df.sort_values('date')

    # Simple Moving Averages
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['sma_200'] = df['close'].rolling(200, min_periods=1).mean()

    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # RSI (14-day)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14, min_periods=1).mean()

    # On-Balance Volume (OBV)
    obv = (df['volume'] * ((df['close'] > df['close'].shift()).astype(int) -
                           (df['close'] < df['close'].shift()).astype(int))).cumsum()
    df['obv'] = obv

    # Volume ratios
    df['volume_sma_20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1)

    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)

    # Volatility
    df['volatility_20d'] = df['return_1d'].rolling(20, min_periods=1).std()

    return df


if __name__ == '__main__':
    import argparse
    from data_scraping.Stock import s_lot_of_stocks, a_lot_of_stocks, test_stock_tickers, all_stocks

    parser = argparse.ArgumentParser(description='Scrape daily price data using yfinance')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['test', 's_lot', 'a_lot', 'all'],
                       help='Dataset to scrape')
    parser.add_argument('--start_date', type=str,
                       default=(datetime.now() - timedelta(days=365*25)).strftime('%Y-%m-%d'),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (defaults to today)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Save progress every N stocks')

    args = parser.parse_args()

    # Select dataset
    if args.dataset == 'test':
        stock_dict = test_stock_tickers
    elif args.dataset == 's_lot':
        stock_dict = s_lot_of_stocks
    elif args.dataset == 'a_lot':
        stock_dict = a_lot_of_stocks
    elif args.dataset == 'all':
        stock_dict = all_stocks

    # Set output file
    if args.output is None:
        args.output = f"{args.dataset}_price_data.pkl"

    # Scrape
    price_data = scrape_price_dataset(
        stock_dict=stock_dict,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output,
        batch_size=args.batch_size
    )

    print(f"\n✅ Done! Price data saved to {args.output}")
