#!/usr/bin/env python3
"""
Evaluate walk-forward training checkpoints.

This script handles checkpoints that may or may not have bin edges saved.
For checkpoints without bin edges, it recomputes them from the training data.
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import h5py

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.backtest_simulation import DatasetLoader, ModelPredictor, TradingSimulator


def compute_bin_edges_from_date_range(
    dataset_path: str,
    start_date: str,
    end_date: str,
    num_bins: int = 100,
    max_samples: int = 100000,
    pred_days: List[int] = [1, 5, 10, 20]
) -> torch.Tensor:
    """
    Compute adaptive bin edges from data within a specific date range.

    Args:
        dataset_path: Path to HDF5 dataset
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        num_bins: Number of bins
        max_samples: Max samples to use
        pred_days: Prediction horizons

    Returns:
        bin_edges tensor of shape (num_bins + 1,)
    """
    print(f"\n  Computing bin edges from {start_date} to {end_date}...")

    all_ratios = []
    sample_count = 0

    with h5py.File(dataset_path, 'r') as f:
        tickers = list(f.keys())

        for ticker in tickers:
            if sample_count >= max_samples:
                break

            ticker_group = f[ticker]

            # Get dates for this ticker
            if 'dates' in ticker_group:
                dates = [d.decode() if isinstance(d, bytes) else d
                        for d in ticker_group['dates'][:]]
            else:
                continue

            # Find indices for date range
            date_indices = []
            for i, d in enumerate(dates):
                if start_date <= d <= end_date:
                    date_indices.append(i)

            if len(date_indices) == 0:
                continue

            # Load features
            features = ticker_group['features'][:]

            # Compute price ratios
            # CRITICAL: Only use ratios where BOTH current AND future dates are within training range
            # This prevents data leakage from looking into the test period
            for i in date_indices:
                if sample_count >= max_samples:
                    break

                current_price = features[i, 0]
                if current_price <= 0:
                    continue

                for pred_day in pred_days:
                    future_idx = i + pred_day
                    if future_idx < len(dates):
                        # LEAKAGE FIX: Ensure future date is also within training bounds
                        future_date = dates[future_idx]
                        if future_date > end_date:
                            continue  # Skip - would leak test period information

                        future_price = features[future_idx, 0]
                        if future_price > 0:
                            ratio = (future_price / current_price) - 1.0
                            if not np.isnan(ratio) and not np.isinf(ratio):
                                all_ratios.append(ratio)
                                sample_count += 1

    all_ratios = np.array(all_ratios)

    if len(all_ratios) == 0:
        print("    Warning: No valid ratios found, using uniform bins")
        bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
    else:
        # Clip to reasonable range
        all_ratios = np.clip(all_ratios, -0.5, 0.5)

        # Compute percentile-based bin edges
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(all_ratios, percentiles)

        print(f"    Samples used: {len(all_ratios):,}")
        print(f"    Ratio range: [{all_ratios.min():.4f}, {all_ratios.max():.4f}]")
        print(f"    Bin edge range: [{bin_edges[0]:.4f}, {bin_edges[-1]:.4f}]")

    return torch.tensor(bin_edges, dtype=torch.float32)


def load_checkpoint_with_bin_edges(
    checkpoint_path: str,
    dataset_path: str,
    device: str = 'cuda'
) -> Tuple[Dict, torch.Tensor]:
    """
    Load checkpoint and get/compute bin edges.

    Returns:
        (checkpoint_dict, bin_edges)
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"  Fold: {checkpoint.get('fold_idx', 'unknown')}")
    print(f"  Train dates: {checkpoint.get('train_dates', 'unknown')}")
    print(f"  Test dates: {checkpoint.get('test_dates', 'unknown')}")
    print(f"  Config: {checkpoint.get('config', {})}")

    # Check if bin edges are in checkpoint
    if 'bin_edges' in checkpoint:
        print("  Bin edges: Found in checkpoint")
        bin_edges = checkpoint['bin_edges']
    else:
        print("  Bin edges: NOT in checkpoint - will recompute from training data")

        config = checkpoint.get('config', {})
        if config.get('pred_mode') != 'classification':
            print("    Model is not classification mode - bin edges not needed")
            bin_edges = None
        else:
            # Recompute from training data
            train_dates = checkpoint.get('train_dates')
            if train_dates is None:
                raise ValueError("Checkpoint has no train_dates, cannot compute bin edges")

            bin_edges = compute_bin_edges_from_date_range(
                dataset_path=dataset_path,
                start_date=train_dates[0],
                end_date=train_dates[1],
                num_bins=100
            )

    return checkpoint, bin_edges


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    prices_path: str,
    top_k: int = 5,
    horizon_idx: int = 0,
    horizon_days: int = 1,
    initial_capital: float = 100000.0,
    confidence_percentile: float = 0.0,
    device: str = 'cuda',
    test_dates: Optional[List[str]] = None,
    transaction_cost_pct: float = 0.001
) -> Dict:
    """
    Evaluate a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        dataset_path: Path to HDF5 dataset
        prices_path: Path to prices HDF5 file
        top_k: Number of stocks to hold
        horizon_idx: Which horizon to use (0=1d, 1=5d, etc.)
        horizon_days: Actual number of days for horizon
        initial_capital: Starting capital
        confidence_percentile: Min confidence percentile
        device: Device to use
        test_dates: Override test dates (None = use checkpoint's test dates)
        transaction_cost_pct: Round-trip transaction cost (default 0.1%)

    Returns:
        Dictionary of evaluation metrics
    """
    # Load checkpoint and bin edges
    checkpoint, bin_edges = load_checkpoint_with_bin_edges(
        checkpoint_path, dataset_path, device
    )

    # Use checkpoint's test dates if not overridden
    if test_dates is None:
        test_date_range = checkpoint.get('test_dates')
        if test_date_range is None:
            raise ValueError("No test dates specified and checkpoint has no test_dates")
    else:
        test_date_range = (test_dates[0], test_dates[-1])

    print(f"\n  Evaluation period: {test_date_range[0]} to {test_date_range[1]}")

    # Save temporary files for ModelPredictor
    temp_dir = os.path.dirname(checkpoint_path)
    temp_model_path = os.path.join(temp_dir, 'temp_eval_model.pt')
    temp_bin_edges_path = os.path.join(temp_dir, 'temp_eval_bin_edges.pt')

    try:
        # Save model checkpoint in format ModelPredictor expects
        torch.save({
            'model_state_dict': checkpoint['model_state_dict'],
            'config': checkpoint['config']
        }, temp_model_path)

        # Save bin edges
        if bin_edges is not None:
            torch.save(bin_edges.cpu(), temp_bin_edges_path)

        # Load data
        print("\n  Loading data...")
        data_loader = DatasetLoader(
            dataset_path=dataset_path,
            prices_path=prices_path
        )

        # Get test dates in range
        available_dates = [d for d in data_loader.all_dates
                         if test_date_range[0] <= d <= test_date_range[1]]

        if len(available_dates) == 0:
            print(f"    ERROR: No dates available in range!")
            return {'error': 'No dates available'}

        print(f"    Available test dates: {len(available_dates)}")
        print(f"    Date range: {available_dates[0]} to {available_dates[-1]}")

        # Preload features
        data_loader.preload_features(available_dates)

        # Create predictor
        print("\n  Loading model...")
        predictor = ModelPredictor(
            model_path=temp_model_path,
            bin_edges_path=temp_bin_edges_path if bin_edges is not None else None,
            device=device,
            batch_size=64
        )

        # Diagnostic: verify model loaded correctly
        print(f"\n  Model diagnostics:")
        print(f"    Model device: {next(predictor.model.parameters()).device}")
        print(f"    Model in eval mode: {not predictor.model.training}")

        # Create simulator
        print("\n  Running simulation...")
        simulator = TradingSimulator(
            data_loader=data_loader,
            predictor=predictor,
            top_k=top_k,
            horizon_days=horizon_days,
            horizon_idx=horizon_idx,
            initial_capital=initial_capital,
            confidence_percentile=confidence_percentile,
            verbose=True,
            transaction_cost_pct=transaction_cost_pct
        )

        # Run simulation
        results = simulator.run_simulation(available_dates)

        # Diagnostic: check for abnormal results
        sharpe = results.get('sharpe_ratio', 0)
        if abs(sharpe) > 100:
            print(f"\n  ⚠️  WARNING: Abnormal Sharpe ratio ({sharpe:.2f})")
            print(f"    This usually indicates:")
            print(f"    - Model weights not loaded correctly (check for missing keys)")
            print(f"    - Very few trades or near-identical returns")
            daily_returns = results.get('daily_returns', [])
            if daily_returns:
                print(f"    Daily returns: count={len(daily_returns)}, "
                      f"mean={np.mean(daily_returns):.6f}, std={np.std(daily_returns):.10f}")

        return results

    finally:
        # Clean up temp files
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if os.path.exists(temp_bin_edges_path):
            os.remove(temp_bin_edges_path)


def evaluate_all_folds(
    checkpoint_dir: str,
    dataset_path: str,
    prices_path: str,
    **kwargs
) -> List[Dict]:
    """
    Evaluate all fold checkpoints in a directory.

    Returns:
        List of result dictionaries, one per fold
    """
    # Find all fold checkpoints
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('fold_') and f.endswith('_best.pt')
    ])

    if not checkpoints:
        print(f"No fold checkpoints found in {checkpoint_dir}")
        return []

    print(f"\nFound {len(checkpoints)} fold checkpoints")

    all_results = []

    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}")
        print('='*60)

        try:
            results = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                dataset_path=dataset_path,
                prices_path=prices_path,
                **kwargs
            )
            results['checkpoint'] = ckpt_name
            all_results.append(results)

            # Print summary
            print(f"\n  Results for {ckpt_name}:")
            print(f"    Total Return: {results.get('total_return_pct', 0):+.2f}%")
            print(f"    Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"    Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
            print(f"    Win Rate: {results.get('win_rate', 0):.1f}%")

        except Exception as e:
            print(f"  ERROR evaluating {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'checkpoint': ckpt_name, 'error': str(e)})

    # Print aggregate summary
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print('='*60)

    valid_results = [r for r in all_results if 'error' not in r]
    if valid_results:
        avg_return = np.mean([r.get('total_return_pct', 0) for r in valid_results])
        avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in valid_results])
        avg_drawdown = np.mean([r.get('max_drawdown_pct', 0) for r in valid_results])
        avg_win_rate = np.mean([r.get('win_rate', 0) for r in valid_results])

        print(f"  Evaluated {len(valid_results)} folds successfully")
        print(f"  Avg Total Return: {avg_return:+.2f}%")
        print(f"  Avg Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"  Avg Max Drawdown: {avg_drawdown:.2f}%")
        print(f"  Avg Win Rate: {avg_win_rate:.1f}%")
    else:
        print("  No valid results!")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate walk-forward checkpoints')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='checkpoints/walk_forward',
                       help='Directory containing fold checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Single checkpoint to evaluate (overrides --checkpoint-dir)')
    parser.add_argument('--data', type=str,
                       default='all_complete_dataset.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--prices', type=str,
                       default='actual_prices_clean.h5',
                       help='Path to prices HDF5 file (use _clean.h5 for validated data)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of stocks to hold')
    parser.add_argument('--horizon-idx', type=int, default=0,
                       help='Horizon index (0=1d, 1=5d, 2=10d, 3=20d)')
    parser.add_argument('--horizon-days', type=int, default=1,
                       help='Actual horizon in days')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--confidence-percentile', type=float, default=0.0,
                       help='Minimum confidence percentile')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Round-trip transaction cost as decimal (default 0.1%% = 0.001)')

    args = parser.parse_args()

    kwargs = {
        'dataset_path': args.data,
        'prices_path': args.prices,
        'top_k': args.top_k,
        'horizon_idx': args.horizon_idx,
        'horizon_days': args.horizon_days,
        'initial_capital': args.initial_capital,
        'confidence_percentile': args.confidence_percentile,
        'device': args.device,
        'transaction_cost_pct': args.transaction_cost
    }

    if args.checkpoint:
        # Evaluate single checkpoint
        results = evaluate_checkpoint(checkpoint_path=args.checkpoint, **kwargs)
        print(f"\nFinal Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        # Evaluate all folds
        evaluate_all_folds(checkpoint_dir=args.checkpoint_dir, **kwargs)


if __name__ == '__main__':
    main()
