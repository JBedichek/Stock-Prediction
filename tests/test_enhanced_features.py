"""
Test script for enhanced FMP features.

Tests all 4 phases:
1. Market indices scraper
2. Derived features calculator
3. Extended technical indicators
4. Cross-sectional rankings

Usage:
    python tests/test_enhanced_features.py --api_key YOUR_KEY
"""

import sys
sys.path.append('/home/james/Desktop/Stock-Prediction')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.utils import save_pickle, pic_load
from data_scraping.market_indices_scraper import MarketIndicesScraper
from data_scraping.derived_features_calculator import DerivedFeaturesCalculator
from data_scraping.cross_sectional_calculator import CrossSectionalCalculator
from data_scraping.fmp_comprehensive_scraper import FMPComprehensiveScraper


def test_phase1_market_indices(api_key: str):
    """Test Phase 1: Market indices scraper."""
    print("\n" + "="*80)
    print("TEST PHASE 1: MARKET INDICES SCRAPER")
    print("="*80)

    scraper = MarketIndicesScraper(api_key)

    # Test with short date range
    from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\nScraping market data from {from_date} to {to_date}...")

    try:
        data = scraper.scrape_all_indices(
            from_date=from_date,
            to_date=to_date,
            include_sector_etfs=True,
            include_additional=True
        )

        print(f"\n✅ SUCCESS!")
        print(f"Collected {len(data)} indices/ETFs")

        # Show sample
        if 'SP500' in data:
            sp500_df = data['SP500']
            print(f"\nS&P 500 sample:")
            print(f"  Shape: {sp500_df.shape}")
            print(f"  Date range: {sp500_df['date'].min()} to {sp500_df['date'].max()}")
            print(f"  Columns: {list(sp500_df.columns)}")
            print(f"\n  Last 3 days:")
            print(sp500_df.tail(3)[['date', 'close', 'volume']])

        # Save test data
        save_pickle(data, 'test_market_indices.pkl')
        print(f"\n✅ Test data saved to test_market_indices.pkl")

        return data

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return None


def test_phase2_derived_features():
    """Test Phase 2: Derived features calculator."""
    print("\n" + "="*80)
    print("TEST PHASE 2: DERIVED FEATURES CALCULATOR")
    print("="*80)

    # Create sample stock data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 2) + 2,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 2) - 2,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })

    calculator = DerivedFeaturesCalculator()

    print("\nCalculating derived features...")

    try:
        # Volume features
        volume_features = calculator.calculate_volume_features(sample_data)
        print(f"\n✅ Volume features: {volume_features.shape[1]} features")
        print(f"  Columns: {list(volume_features.columns)[:5]}...")

        # Price features
        price_features = calculator.calculate_price_features(sample_data)
        print(f"\n✅ Price features: {price_features.shape[1]} features")
        print(f"  Columns: {list(price_features.columns)[:5]}...")

        # Bollinger Bands
        bb_features = calculator.calculate_bollinger_bands(sample_data)
        print(f"\n✅ Bollinger Bands: {bb_features.shape[1]} features")
        print(f"  Columns: {list(bb_features.columns)}")

        # Momentum
        momentum_features = calculator.calculate_momentum_features(sample_data)
        print(f"\n✅ Momentum features: {momentum_features.shape[1]} features")
        print(f"  Columns: {list(momentum_features.columns)}")

        # All features combined
        all_features = calculator.calculate_all_features(sample_data)
        print(f"\n✅ ALL DERIVED FEATURES: {all_features.shape[1]} total features")

        return all_features

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_phase3_extended_technicals(api_key: str):
    """Test Phase 3: Extended technical indicators."""
    print("\n" + "="*80)
    print("TEST PHASE 3: EXTENDED TECHNICAL INDICATORS")
    print("="*80)

    scraper = FMPComprehensiveScraper(api_key)

    print("\nFetching extended technical indicators for AAPL...")

    try:
        indicators = scraper.get_all_technical_indicators('AAPL', interval='daily', include_extended=True)

        print(f"\n✅ SUCCESS!")
        print(f"Collected {len(indicators)} technical indicators:")

        for name, df in indicators.items():
            if df is not None and not df.empty:
                print(f"  ✓ {name}: {len(df)} days")

        return indicators

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return None


def test_phase4_cross_sectional():
    """Test Phase 4: Cross-sectional rankings."""
    print("\n" + "="*80)
    print("TEST PHASE 4: CROSS-SECTIONAL RANKINGS")
    print("="*80)

    # Create sample multi-stock data
    dates = pd.date_range(start='2024-11-01', end='2024-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    all_stocks_data = {}

    np.random.seed(42)
    for ticker in tickers:
        df = pd.DataFrame({
            'date': dates,
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'return_1d': np.random.randn(len(dates)) * 0.02,
            'return_5d': np.random.randn(len(dates)) * 0.05,
            'return_20d': np.random.randn(len(dates)) * 0.10,
            'volatility_20d': np.random.rand(len(dates)) * 0.03,
            'volume_ratio_20d': np.random.rand(len(dates)) * 2,
        })
        df['dollar_volume'] = df['close'] * df['volume']
        all_stocks_data[ticker] = df

    calculator = CrossSectionalCalculator()

    print(f"\nCalculating cross-sectional features for {len(tickers)} stocks...")

    try:
        # Test single date
        test_date = dates[30].date()
        date_features = calculator.calculate_cross_sectional_features_for_date(
            all_stocks_data, test_date
        )

        print(f"\n✅ SUCCESS for date {test_date}!")
        print(f"Features calculated for {len(date_features)} stocks:")

        for ticker, features in date_features.items():
            print(f"\n  {ticker}:")
            for feature_name, value in list(features.items())[:5]:
                print(f"    {feature_name}: {value:.2f}")
            if len(features) > 5:
                print(f"    ... and {len(features) - 5} more features")

        # Test momentum rankings
        momentum_features = calculator.calculate_momentum_rankings(
            all_stocks_data, test_date
        )

        print(f"\n✅ Momentum rankings calculated!")
        for ticker in tickers:
            if ticker in momentum_features:
                print(f"  {ticker}: {momentum_features[ticker]}")

        return date_features

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline(api_key: str):
    """Test full enhanced pipeline on one stock."""
    print("\n" + "="*80)
    print("FULL PIPELINE TEST: ONE STOCK WITH ALL FEATURES")
    print("="*80)

    from data_scraping.fmp_enhanced_processor import EnhancedFMPDataProcessor

    print("\nStep 1: Scraping market indices...")
    market_data = test_phase1_market_indices(api_key)

    if not market_data:
        print("⚠️  Skipping full pipeline test (no market data)")
        return

    print("\nStep 2: Scraping stock data (AAPL)...")
    scraper = FMPComprehensiveScraper(api_key)

    from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    stock_data = scraper.scrape_all_data('AAPL', start_date=from_date, end_date=to_date)

    print("\nStep 3: Processing with enhanced processor...")

    try:
        processor = EnhancedFMPDataProcessor(stock_data, market_data)

        # Get enhanced features
        enhanced_df = processor.combine_all_features_enhanced(
            start_date=from_date,
            end_date=to_date
        )

        print(f"\n✅ ENHANCED FEATURES CREATED!")
        print(f"  Shape: {enhanced_df.shape}")
        print(f"  Date range: {enhanced_df.index.min()} to {enhanced_df.index.max()}")
        print(f"  Total features: {enhanced_df.shape[1]}")

        # Show feature breakdown
        feature_types = {}
        for col in enhanced_df.columns:
            if 'SP500' in col or 'VIX' in col or 'NASDAQ' in col:
                category = 'Market Indices'
            elif 'volume' in col.lower():
                category = 'Volume Features'
            elif 'bb_' in col or 'bollinger' in col.lower():
                category = 'Bollinger Bands'
            elif 'momentum' in col.lower() or 'roc' in col.lower():
                category = 'Momentum Features'
            elif 'beta' in col or 'correlation' in col or 'relative' in col:
                category = 'Market-Relative'
            elif 'tech_' in col:
                category = 'Technical Indicators'
            else:
                category = 'Other'

            feature_types[category] = feature_types.get(category, 0) + 1

        print(f"\nFeature breakdown:")
        for category, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} features")

        # Convert to tensors
        tensor_dict = processor.to_tensor_dict_enhanced(from_date, to_date)
        print(f"\n✅ Converted to {len(tensor_dict)} daily tensors")

        sample_date = list(tensor_dict.keys())[0]
        print(f"\nSample tensor for {sample_date}:")
        print(f"  Shape: {tensor_dict[sample_date].shape}")
        print(f"  First 10 values: {tensor_dict[sample_date][:10].numpy()}")

        return enhanced_df

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test enhanced FMP features')
    parser.add_argument('--api_key', type=str, default="YOUR_FMP_API_KEY_HERE",
                       help='FMP API key')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['1', '2', '3', '4', 'full', 'all'],
                       help='Which phase to test')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ENHANCED FMP FEATURES TEST SUITE")
    print("="*80)

    if args.phase in ['1', 'all']:
        test_phase1_market_indices(args.api_key)

    if args.phase in ['2', 'all']:
        test_phase2_derived_features()

    if args.phase in ['3', 'all']:
        test_phase3_extended_technicals(args.api_key)

    if args.phase in ['4', 'all']:
        test_phase4_cross_sectional()

    if args.phase in ['full', 'all']:
        test_full_pipeline(args.api_key)

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
