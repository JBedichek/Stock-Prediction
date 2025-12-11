"""
Derived Features Calculator

Calculates additional features from existing OHLCV price/volume data.
These features require NO additional API calls and provide significant value.

Features include:
- Volume features (ratios, trends, spikes)
- Price features (gaps, volatility, ranges)
- Technical patterns (Bollinger Bands, etc.)
- Market-relative features (beta, correlation)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DerivedFeaturesCalculator:
    """
    Calculate derived features from OHLCV data.
    """

    def __init__(self):
        """Initialize calculator."""
        pass

    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features.

        Args:
            df: DataFrame with columns: date, open, high, low, close, volume

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)

        # Ensure volume is numeric
        volume = pd.to_numeric(df['volume'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')

        # Volume moving averages
        volume_ma_5 = volume.rolling(5, min_periods=1).mean()
        volume_ma_10 = volume.rolling(10, min_periods=1).mean()
        volume_ma_20 = volume.rolling(20, min_periods=1).mean()
        volume_ma_50 = volume.rolling(50, min_periods=1).mean()

        # Volume ratios
        features['volume_ratio_5d'] = volume / (volume_ma_5 + 1e-8)
        features['volume_ratio_10d'] = volume / (volume_ma_10 + 1e-8)
        features['volume_ratio_20d'] = volume / (volume_ma_20 + 1e-8)
        features['volume_ratio_50d'] = volume / (volume_ma_50 + 1e-8)

        # Dollar volume
        features['dollar_volume'] = close * volume
        features['dollar_volume_ma_20'] = (close * volume).rolling(20, min_periods=1).mean()

        # Volume spike detection
        features['volume_spike_2x'] = volume.gt(2 * volume_ma_20).fillna(False).astype(float)
        features['volume_spike_3x'] = volume.gt(3 * volume_ma_20).fillna(False).astype(float)

        # Volume trend
        features['volume_trend_5d'] = volume.rolling(5, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )

        # Relative volume (z-score)
        volume_mean = volume.rolling(20, min_periods=1).mean()
        volume_std = volume.rolling(20, min_periods=1).std()
        features['volume_zscore'] = (volume - volume_mean) / (volume_std + 1e-8)

        return features

    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features.

        Args:
            df: DataFrame with columns: date, open, high, low, close

        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=df.index)

        # Ensure numeric
        open_price = pd.to_numeric(df['open'], errors='coerce')
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')

        # Intraday volatility
        features['intraday_volatility'] = (high - low) / (close + 1e-8)
        features['intraday_vol_ma_20'] = features['intraday_volatility'].rolling(20, min_periods=1).mean()

        # Gap analysis
        prev_close = close.shift(1)
        features['gap_pct'] = (open_price - prev_close) / (prev_close + 1e-8)
        features['gap_up'] = features['gap_pct'].gt(0.01).fillna(False).astype(float)  # >1% gap up
        features['gap_down'] = features['gap_pct'].lt(-0.01).fillna(False).astype(float)  # >1% gap down

        # Price ranges
        features['high_low_ratio'] = high / (low + 1e-8)
        features['close_to_high'] = (close - low) / (high - low + 1e-8)  # 0 = at low, 1 = at high

        # Distance from 52-week high/low
        high_52w = high.rolling(252, min_periods=1).max()
        low_52w = low.rolling(252, min_periods=1).min()
        features['dist_from_52w_high'] = (close - high_52w) / (high_52w + 1e-8)
        features['dist_from_52w_low'] = (close - low_52w) / (low_52w + 1e-8)

        # Distance from key moving averages
        sma_20 = close.rolling(20, min_periods=1).mean()
        sma_50 = close.rolling(50, min_periods=1).mean()
        sma_200 = close.rolling(200, min_periods=1).mean()

        features['dist_from_sma20'] = (close - sma_20) / (sma_20 + 1e-8)
        features['dist_from_sma50'] = (close - sma_50) / (sma_50 + 1e-8)
        features['dist_from_sma200'] = (close - sma_200) / (sma_200 + 1e-8)

        # Trend strength
        features['sma20_above_sma50'] = sma_20.gt(sma_50).fillna(False).astype(float)
        features['sma50_above_sma200'] = sma_50.gt(sma_200).fillna(False).astype(float)

        # Returns
        features['return_1d'] = close.pct_change(1)
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)

        # Realized volatility (std of returns)
        features['volatility_5d'] = close.pct_change().rolling(5, min_periods=1).std()
        features['volatility_20d'] = close.pct_change().rolling(20, min_periods=1).std()

        return features

    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands features.

        Args:
            df: DataFrame with close prices
            window: SMA window (default 20)
            num_std: Number of standard deviations (default 2)

        Returns:
            DataFrame with Bollinger Band features
        """
        features = pd.DataFrame(index=df.index)

        close = pd.to_numeric(df['close'], errors='coerce')

        # Middle band (SMA)
        sma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std()

        # Upper and lower bands
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std

        # Bollinger features
        features['bb_upper'] = upper_band
        features['bb_middle'] = sma
        features['bb_lower'] = lower_band
        features['bb_width'] = (upper_band - lower_band) / (sma + 1e-8)
        features['bb_position'] = (close - lower_band) / (upper_band - lower_band + 1e-8)  # 0-1

        # Bollinger squeeze
        bb_width_threshold = features['bb_width'].rolling(50, min_periods=1).quantile(0.25)
        features['bb_squeeze'] = features['bb_width'].lt(bb_width_threshold).fillna(False).astype(float)

        # Price touching bands
        features['touching_upper_bb'] = close.ge(upper_band * 0.99).fillna(False).astype(float)
        features['touching_lower_bb'] = close.le(lower_band * 1.01).fillna(False).astype(float)

        return features

    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based features.

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)

        close = pd.to_numeric(df['close'], errors='coerce')

        # Rate of change (ROC)
        features['roc_5'] = (close - close.shift(5)) / (close.shift(5) + 1e-8) * 100
        features['roc_10'] = (close - close.shift(10)) / (close.shift(10) + 1e-8) * 100
        features['roc_20'] = (close - close.shift(20)) / (close.shift(20) + 1e-8) * 100

        # Momentum (price difference)
        features['momentum_5'] = close - close.shift(5)
        features['momentum_10'] = close - close.shift(10)

        # Acceleration (change in momentum)
        features['acceleration_5'] = features['momentum_5'] - features['momentum_5'].shift(5)

        return features

    def calculate_market_relative_features(self, stock_df: pd.DataFrame, market_df: pd.DataFrame,
                                          window: int = 60) -> pd.DataFrame:
        """
        Calculate features relative to market index.

        Args:
            stock_df: Stock DataFrame with close prices
            market_df: Market index DataFrame with close prices
            window: Rolling window for calculations

        Returns:
            DataFrame with market-relative features
        """
        features = pd.DataFrame(index=stock_df.index)

        # Align dates
        stock_close = pd.to_numeric(stock_df['close'], errors='coerce')

        # Merge on date to align
        merged = pd.merge(
            stock_df[['date', 'close']].rename(columns={'close': 'stock_close'}),
            market_df[['date', 'close']].rename(columns={'close': 'market_close'}),
            on='date',
            how='left'
        )

        merged['stock_close'] = pd.to_numeric(merged['stock_close'], errors='coerce')
        merged['market_close'] = pd.to_numeric(merged['market_close'], errors='coerce')
        merged['market_close'] = merged['market_close'].fillna(method='ffill')  # Forward fill missing market data

        # Returns
        stock_returns = merged['stock_close'].pct_change()
        market_returns = merged['market_close'].pct_change()

        # Beta (rolling)
        def rolling_beta(x, y, window):
            """Calculate rolling beta."""
            result = pd.Series(index=x.index, dtype=float)
            for i in range(len(x)):
                start = max(0, i - window + 1)
                x_slice = x.iloc[start:i+1]
                y_slice = y.iloc[start:i+1]

                if len(x_slice) > 5:  # Need minimum data
                    valid = ~(x_slice.isna() | y_slice.isna())
                    if valid.sum() > 5:
                        slope, _, _, _, _ = stats.linregress(y_slice[valid], x_slice[valid])
                        result.iloc[i] = slope
            return result

        features['beta'] = rolling_beta(stock_returns, market_returns, window)

        # Correlation (rolling)
        features['correlation'] = stock_returns.rolling(window, min_periods=10).corr(market_returns)

        # Relative strength (stock return - market return)
        features['relative_return_1d'] = stock_returns - market_returns
        features['relative_return_5d'] = stock_returns.rolling(5).sum() - market_returns.rolling(5).sum()
        features['relative_return_20d'] = stock_returns.rolling(20).sum() - market_returns.rolling(20).sum()

        # Relative performance (cumulative)
        stock_perf = (1 + stock_returns).cumprod()
        market_perf = (1 + market_returns).cumprod()
        features['relative_performance'] = stock_perf / (market_perf + 1e-8)

        return features

    def calculate_all_features(self, df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all derived features.

        Args:
            df: Stock DataFrame with OHLCV data
            market_df: Optional market index DataFrame for relative features

        Returns:
            DataFrame with all derived features
        """
        all_features = pd.DataFrame(index=df.index)

        try:
            print("      📊 Calculating volume features...")
            volume_features = self.calculate_volume_features(df)
            all_features = pd.concat([all_features, volume_features], axis=1)
        except Exception as e:
            print(f"      ❌ Error in volume features: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            print("      📈 Calculating price features...")
            price_features = self.calculate_price_features(df)
            all_features = pd.concat([all_features, price_features], axis=1)
        except Exception as e:
            print(f"      ❌ Error in price features: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            print("      📉 Calculating Bollinger Bands...")
            bb_features = self.calculate_bollinger_bands(df)
            all_features = pd.concat([all_features, bb_features], axis=1)
        except Exception as e:
            print(f"      ❌ Error in Bollinger Bands: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            print("      🚀 Calculating momentum features...")
            momentum_features = self.calculate_momentum_features(df)
            all_features = pd.concat([all_features, momentum_features], axis=1)
        except Exception as e:
            print(f"      ❌ Error in momentum features: {e}")
            import traceback
            traceback.print_exc()
            raise

        if market_df is not None:
            try:
                print("      📊 Calculating market-relative features...")
                market_features = self.calculate_market_relative_features(df, market_df)
                all_features = pd.concat([all_features, market_features], axis=1)
            except Exception as e:
                print(f"      ❌ Error in market-relative features: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Replace inf with NaN, then fill NaN with 0
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(0)

        return all_features


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.utils import pic_load

    # Load sample data
    print("Loading sample stock data...")
    # This is just an example - in practice, use your actual data

    calculator = DerivedFeaturesCalculator()
    print("✅ Derived Features Calculator initialized")
    print(f"\nAvailable feature categories:")
    print("  - Volume features: 10 features")
    print("  - Price features: 18 features")
    print("  - Bollinger Bands: 7 features")
    print("  - Momentum features: 6 features")
    print("  - Market-relative features: 8 features (requires market data)")
    print(f"\nTotal: ~40-50 derived features per stock")
