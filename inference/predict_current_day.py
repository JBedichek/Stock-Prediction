#!/usr/bin/env python3
"""
Current Day Stock Prediction Script

Makes predictions for the current day (or latest available date) and outputs
a ranked list of stocks to buy.

This script:
1. Loads the dataset and finds the most recent date
2. Runs inference on all stocks for that date
3. Applies the same filtering and selection logic as backtest_simulation.py
4. Outputs a ranked list of stocks to buy

Usage:
    python inference/predict_current_day.py \
        --data all_complete_dataset.h5 \
        --prices actual_prices.h5 \
        --model checkpoints/best_model.pt \
        --bin-edges adaptive_bin_edges.pt \
        --top-k 10 \
        --horizon-idx 1

Parameters match backtest_simulation.py for consistency.
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import os
import h5py
from datetime import datetime, timedelta
from typing import List, Tuple
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train_new_format import SimpleTransformerPredictor

# ===== CPU Optimizations =====
torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(4)

if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

torch.set_float32_matmul_precision('medium')
torch.set_grad_enabled(False)


class CurrentDayPredictor:
    """Predict stocks to buy for the current/latest available day."""

    def __init__(self,
                 dataset_path: str,
                 prices_path: str,
                 model_path: str,
                 bin_edges_path: str,
                 device: str = 'cuda',
                 batch_size: int = 256,
                 compile_model: bool = False):
        """
        Args:
            dataset_path: Path to HDF5 dataset with features
            prices_path: Path to HDF5 file with actual prices
            model_path: Path to model checkpoint
            bin_edges_path: Path to cached bin edges
            device: Device to run inference on
            batch_size: Batch size for inference
            compile_model: Use torch.compile for faster inference
        """
        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.device = device
        self.batch_size = batch_size

        print(f"\n{'='*80}")
        print("LOADING DATASET")
        print(f"{'='*80}")

        # Open HDF5 files
        print(f"📦 Loading dataset: {dataset_path}")
        self.h5_file = h5py.File(dataset_path, 'r')
        self.all_tickers = sorted(list(self.h5_file.keys()))
        print(f"  ✅ Loaded {len(self.all_tickers)} tickers")

        # Get dates from first ticker
        sample_ticker = self.all_tickers[0]
        dates_bytes = self.h5_file[sample_ticker]['dates'][:]
        self.all_dates = sorted([d.decode('utf-8') for d in dates_bytes])
        print(f"  📅 Date range: {self.all_dates[0]} to {self.all_dates[-1]}")
        print(f"  Total dates: {len(self.all_dates)}")

        # Load prices file
        if prices_path and os.path.exists(prices_path):
            print(f"\n📊 Loading actual prices: {prices_path}")
            self.prices_file = h5py.File(prices_path, 'r')
            print(f"  ✅ Prices loaded")
        else:
            print(f"\n⚠️  No prices file provided - will use features[0]")
            self.prices_file = None

        # Load model
        print(f"\n{'='*80}")
        print("LOADING MODEL")
        print(f"{'='*80}")
        self._load_model(model_path, bin_edges_path, compile_model)

    def _load_model(self, model_path: str, bin_edges_path: str, compile_model: bool):
        """Load model and bin edges."""
        print(f"📦 Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get model config
        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']

        # Infer input_dim from state_dict
        if '_orig_mod.input_proj.0.weight' in state_dict:
            input_dim = state_dict['_orig_mod.input_proj.0.weight'].shape[1]
            pred_head_keys = [k for k in state_dict.keys() if 'pred_head' in k and 'weight' in k and '_orig_mod' in k]
            final_pred_key = sorted(pred_head_keys)[-1]
            final_output_size = state_dict[final_pred_key].shape[0]
        elif 'input_proj.0.weight' in state_dict:
            input_dim = state_dict['input_proj.0.weight'].shape[1]
            pred_head_keys = [k for k in state_dict.keys() if 'pred_head' in k and 'weight' in k]
            final_pred_key = sorted(pred_head_keys)[-1]
            final_output_size = state_dict[final_pred_key].shape[0]
        else:
            raise KeyError(f"Could not find input_proj layer in state_dict")

        # Infer pred_mode
        if 'pred_mode' in config:
            pred_mode = config['pred_mode']
        elif final_output_size == 400:
            pred_mode = 'classification'
        elif final_output_size == 4:
            pred_mode = 'regression'
        else:
            raise ValueError(f"Cannot infer pred_mode from output size {final_output_size}")

        print(f"  Input dimension: {input_dim}")
        print(f"  Prediction mode: {pred_mode}")

        # Create model
        self.model = SimpleTransformerPredictor(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            num_pred_days=4,
            pred_mode=pred_mode
        )

        # Strip '_orig_mod.' prefix if present
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Compile if requested
        if compile_model:
            print(f"  🔧 Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode='max-autotune')

        print(f"  ✅ Model loaded successfully")

        # Load bin edges
        self.pred_mode = pred_mode
        if pred_mode == 'classification':
            if os.path.exists(bin_edges_path):
                print(f"📊 Loading bin edges: {bin_edges_path}")
                self.bin_edges = torch.load(bin_edges_path).to(self.device)
                print(f"  ✅ Loaded {len(self.bin_edges)} bin edges")
            else:
                raise FileNotFoundError(f"Bin edges file not found: {bin_edges_path}")
        else:
            self.bin_edges = None

    def get_latest_date(self) -> str:
        """Get the most recent date available in the dataset with REAL data."""
        import numpy as np

        # Check if recent dates are forward-filled (identical data)
        print(f"\n  🔍 Checking for forward-filled data...")

        sample_tickers = list(self.h5_file.keys())[:10]
        forward_filled_count = 0

        for ticker in sample_tickers:
            dates_bytes = self.h5_file[ticker]['dates'][:]
            dates = [d.decode('utf-8') for d in dates_bytes]
            features = self.h5_file[ticker]['features'][:]

            if len(dates) < 2:
                continue

            # Compare last two dates
            last_features = features[-1, :]
            second_last_features = features[-2, :]

            # Check if identical (forward-filled)
            if np.allclose(last_features, second_last_features, rtol=1e-5):
                forward_filled_count += 1

        # If most tickers have forward-filled data, use second-to-last date
        if forward_filled_count >= len(sample_tickers) * 0.5:
            latest_date = self.all_dates[-2]  # Use second-to-last date
            print(f"  ⚠️  Latest date ({self.all_dates[-1]}) appears to be forward-filled")
            print(f"  ✅ Using last date with real data: {latest_date}")
        else:
            latest_date = self.all_dates[-1]
            print(f"  ✅ Latest date has real data: {latest_date}")

        # If prices file exists, check it too
        if self.prices_file is not None:
            sample_ticker = list(self.prices_file.keys())[0]
            prices_dates_bytes = self.prices_file[sample_ticker]['dates'][:]
            prices_dates = sorted([d.decode('utf-8') for d in prices_dates_bytes])
            prices_latest = prices_dates[-1]

            if prices_latest != latest_date:
                print(f"  📊 Price file date: {prices_latest}")

        return latest_date

    def get_features_and_price(self, ticker: str, date: str) -> Tuple[torch.Tensor, float]:
        """Get features and current price for a ticker on a date."""
        if ticker not in self.h5_file:
            return None

        # Get date index
        try:
            date_idx = self.all_dates.index(date)
        except ValueError:
            return None

        # Get features
        features_2d = self.h5_file[ticker]['features'][:]
        if date_idx >= features_2d.shape[0]:
            return None

        features = torch.from_numpy(features_2d[date_idx, :]).float()

        # Get price
        if self.prices_file is not None and ticker in self.prices_file:
            prices_dates_bytes = self.prices_file[ticker]['dates'][:]
            prices_dates = [d.decode('utf-8') for d in prices_dates_bytes]

            try:
                price_idx = prices_dates.index(date)
                prices_array = self.prices_file[ticker]['prices'][:]
                current_price = float(prices_array[price_idx])
            except (ValueError, IndexError):
                current_price = features[0].item()
        else:
            current_price = features[0].item()

        return features, current_price

    @torch.no_grad()
    def predict_batch(self, features_list: List[torch.Tensor], horizon_idx: int) -> Tuple[List[float], List[float]]:
        """
        Run batch inference on a list of features.

        Args:
            features_list: List of feature tensors
            horizon_idx: Prediction horizon index

        Returns:
            Tuple of (expected_returns, confidences)
        """
        if len(features_list) == 0:
            return [], []

        # Check if all tensors have the same size
        feature_sizes = [f.shape[0] for f in features_list]
        max_size = max(feature_sizes)
        min_size = min(feature_sizes)

        if max_size != min_size:
            # Pad tensors to same size
            print(f"  ⚠️  Feature dimension mismatch detected: {min_size} to {max_size}")
            print(f"     Padding all tensors to {max_size} dimensions")

            padded_features = []
            for f in features_list:
                if f.shape[0] < max_size:
                    # Pad with zeros
                    padding = torch.zeros(max_size - f.shape[0], dtype=f.dtype, device=f.device)
                    f_padded = torch.cat([f, padding])
                    padded_features.append(f_padded)
                else:
                    padded_features.append(f)
            features_list = padded_features

        # Stack into batch
        features_batch = torch.stack(features_list).unsqueeze(1).to(self.device)  # (batch, 1, features)

        # Forward pass
        pred, confidence = self.model(features_batch)

        if self.pred_mode == 'classification':
            # Get distribution for horizon
            logits = pred[:, :, horizon_idx]  # (batch, num_bins)
            probs = F.softmax(logits, dim=1)

            # Calculate expected value
            bin_midpoints = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            expected_ratios = torch.sum(probs * bin_midpoints.unsqueeze(0), dim=1)
            expected_returns = (1.0 + expected_ratios).cpu().tolist()
        else:
            # Regression mode
            expected_ratios = pred[:, horizon_idx]
            expected_returns = (1.0 + expected_ratios).cpu().tolist()

        # Extract confidence
        horizon_confidence = confidence[:, horizon_idx].cpu().tolist()

        return expected_returns, horizon_confidence

    def predict_all_stocks(self,
                          date: str,
                          horizon_idx: int,
                          confidence_percentile: float = 0.6) -> List[Tuple[str, float, float, float]]:
        """
        Run inference on all stocks for a given date.

        Args:
            date: Date to predict for
            horizon_idx: Prediction horizon (0=1day, 1=5day, 2=10day, 3=20day)
            confidence_percentile: Filter stocks below this confidence percentile

        Returns:
            List of (ticker, expected_return, confidence, current_price) sorted by confidence-weighted return
        """
        print(f"\n{'='*80}")
        print(f"RUNNING PREDICTIONS FOR {date}")
        print(f"{'='*80}")

        # Collect all valid stocks
        print(f"📊 Loading features for all stocks...")
        valid_data = []
        for ticker in tqdm(self.all_tickers, desc="  Loading"):
            result = self.get_features_and_price(ticker, date)
            if result is not None:
                features, current_price = result
                valid_data.append((ticker, features, current_price))

        print(f"  ✅ Loaded {len(valid_data)} stocks with valid data")

        # Separate into lists
        tickers = [item[0] for item in valid_data]
        features_list = [item[1] for item in valid_data]
        prices = [item[2] for item in valid_data]

        # Run batch inference
        print(f"\n🧠 Running model inference...")
        predictions = []

        for i in tqdm(range(0, len(features_list), self.batch_size), desc="  Inference"):
            batch_tickers = tickers[i:i+self.batch_size]
            batch_features = features_list[i:i+self.batch_size]
            batch_prices = prices[i:i+self.batch_size]

            # Get predictions
            expected_returns, confidences = self.predict_batch(batch_features, horizon_idx)

            # Combine results
            for ticker, expected_return, confidence, price in zip(batch_tickers, expected_returns, confidences, batch_prices):
                predictions.append((ticker, expected_return, confidence, price))

        print(f"  ✅ Generated {len(predictions)} predictions")

        # Filter by confidence percentile
        if len(predictions) > 0 and confidence_percentile > 0:
            all_confidences = [pred[2] for pred in predictions]
            confidence_threshold = sorted(all_confidences)[int(len(all_confidences) * confidence_percentile)]
            predictions = [pred for pred in predictions if pred[2] >= confidence_threshold]
            print(f"  🔍 Filtered to {len(predictions)} stocks (confidence >= {confidence_threshold:.4f})")

        # Weight expected returns by confidence and sort
        weighted_predictions = []
        for ticker, expected_return, confidence, price in predictions:
            weighted_return = expected_return * confidence
            weighted_predictions.append((ticker, expected_return, confidence, price, weighted_return))

        weighted_predictions.sort(key=lambda x: x[4], reverse=True)

        # Return (ticker, expected_return, confidence, price)
        return [(ticker, expected_return, confidence, price)
                for ticker, expected_return, confidence, price, _ in weighted_predictions]

    def __del__(self):
        """Close HDF5 files."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
        if hasattr(self, 'prices_file') and self.prices_file is not None:
            self.prices_file.close()


def print_stock_recommendations(stocks: List[Tuple[str, float, float, float]],
                                top_k: int,
                                horizon_days: int,
                                date: str):
    """Print formatted stock recommendations."""
    print(f"\n{'='*80}")
    print(f"📈 TOP {top_k} STOCK RECOMMENDATIONS FOR {date}")
    print(f"{'='*80}")
    print(f"\nHolding period: {horizon_days} trading days")
    print(f"Total candidates analyzed: {len(stocks)}")
    print()

    print(f"{'Rank':<6} {'Ticker':<8} {'Expected Return':<18} {'Confidence':<12} {'Current Price':<15}")
    print(f"{'-'*80}")

    for rank, (ticker, expected_return, confidence, price) in enumerate(stocks[:top_k], 1):
        expected_pct = (expected_return - 1.0) * 100
        print(f"{rank:<6} {ticker:<8} {expected_pct:>+6.2f}%{'':<10} {confidence:>6.4f}{'':<6} ${price:>10.2f}")

    print()
    print(f"{'='*80}")
    print(f"💡 INVESTMENT STRATEGY")
    print(f"{'='*80}")
    print(f"  • Buy these {min(top_k, len(stocks))} stocks today")
    print(f"  • Hold for {horizon_days} trading days")
    print(f"  • Split capital equally among selected stocks")
    print(f"  • Rebalance after holding period")
    print()


def save_recommendations(stocks: List[Tuple[str, float, float, float]],
                        output_path: str,
                        date: str,
                        horizon_days: int,
                        top_k: int):
    """Save recommendations to file."""
    results = {
        'date': date,
        'horizon_days': horizon_days,
        'top_k': top_k,
        'recommendations': [
            {
                'rank': rank,
                'ticker': ticker,
                'expected_return': expected_return,
                'expected_return_pct': (expected_return - 1.0) * 100,
                'confidence': confidence,
                'current_price': price
            }
            for rank, (ticker, expected_return, confidence, price) in enumerate(stocks[:top_k], 1)
        ],
        'all_predictions': [
            {
                'ticker': ticker,
                'expected_return': expected_return,
                'confidence': confidence,
                'current_price': price
            }
            for ticker, expected_return, confidence, price in stocks
        ]
    }

    torch.save(results, output_path)
    print(f"💾 Saved recommendations to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict stocks to buy for the current/latest day',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data args
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset with features')
    parser.add_argument('--prices', type=str, default=None,
                       help='Path to HDF5 file with actual prices (optional)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--bin-edges', type=str, required=True,
                       help='Path to bin edges cache')

    # Prediction args (same as backtest_simulation.py)
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of stocks to recommend')
    parser.add_argument('--horizon-idx', type=int, default=1,
                       help='Prediction horizon (0=1day, 1=5day, 2=10day, 3=20day)')
    parser.add_argument('--confidence-percentile', type=float, default=0.6,
                       help='Confidence percentile for filtering (0.6 = keep top 40%%)')

    # Optional args
    parser.add_argument('--date', type=str, default=None,
                       help='Date to predict for (default: latest available)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for faster inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save recommendations (optional)')

    args = parser.parse_args()

    # Map horizon_idx to days
    horizon_map = {0: 1, 1: 5, 2: 10, 3: 20}
    horizon_days = horizon_map.get(args.horizon_idx, 5)

    print(f"\n{'='*80}")
    print("CURRENT DAY STOCK PREDICTION")
    print(f"{'='*80}")
    print(f"  Dataset: {args.data}")
    print(f"  Model: {args.model}")
    print(f"  Prediction horizon: {horizon_days} days (horizon_idx={args.horizon_idx})")
    print(f"  Top-K stocks: {args.top_k}")
    print(f"  Confidence percentile: {args.confidence_percentile}")
    print()

    # Create predictor
    predictor = CurrentDayPredictor(
        dataset_path=args.data,
        prices_path=args.prices,
        model_path=args.model,
        bin_edges_path=args.bin_edges,
        device=args.device,
        batch_size=args.batch_size,
        compile_model=args.compile
    )

    # Get prediction date
    if args.date is None:
        prediction_date = predictor.get_latest_date()
        print(f"📅 Using latest available date: {prediction_date}")
    else:
        prediction_date = args.date
        print(f"📅 Using specified date: {prediction_date}")

    # Run predictions
    stocks = predictor.predict_all_stocks(
        date=prediction_date,
        horizon_idx=args.horizon_idx,
        confidence_percentile=args.confidence_percentile
    )

    # Print recommendations
    print_stock_recommendations(
        stocks=stocks,
        top_k=args.top_k,
        horizon_days=horizon_days,
        date=prediction_date
    )

    # Save if requested
    if args.output:
        save_recommendations(
            stocks=stocks,
            output_path=args.output,
            date=prediction_date,
            horizon_days=horizon_days,
            top_k=args.top_k
        )

    print(f"\n{'='*80}")
    print("✅ PREDICTION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
