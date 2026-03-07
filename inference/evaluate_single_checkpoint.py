#!/usr/bin/env python3
"""
Evaluate a single checkpoint (from cross_sectional_model.py training).

Computes IC, Rank IC, IR over time and generates plots.

Usage:
    python -m inference.evaluate_single_checkpoint \
        --checkpoint checkpoints/cross_sectional/best_model.pt \
        --data all_complete_dataset.h5 \
        --prices actual_prices_clean.h5 \
        --test-start 2021-01-01 \
        --test-end 2022-12-31 \
        --device cuda
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import json
import os


def evaluate_checkpoint(
    checkpoint_path: str,
    data_path: str,
    prices_path: str,
    test_start: str,
    test_end: str,
    seq_len: int = 64,
    num_stocks: int = 100,
    device: str = 'cuda',
    output_dir: str = None,
    batch_size: int = 32,
):
    """
    Evaluate a checkpoint and compute IC metrics over time.
    """
    from training.cross_sectional_model import (
        DirectPricePredictor,
        CrossSectionalDataset,
    )
    from training.multimodal_model import FeatureConfig

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("CHECKPOINT EVALUATION")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint.get('config', {})
    args = checkpoint.get('args', {})

    pred_mode = config.get('pred_mode', 'regression')
    hidden_dim = config.get('hidden_dim', args.get('hidden_dim', 256))
    num_encoder_layers = config.get('num_encoder_layers', args.get('num_encoder_layers', 4))
    num_encoder_heads = config.get('num_encoder_heads', args.get('num_encoder_heads', 4))
    saved_seq_len = config.get('seq_len', args.get('seq_len', seq_len))

    if saved_seq_len != seq_len:
        print(f"  Using seq_len={saved_seq_len} from checkpoint")
        seq_len = saved_seq_len

    print(f"  Mode: {pred_mode}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_encoder_layers}, Heads: {num_encoder_heads}")

    # Create model
    if pred_mode == 'regression':
        model = DirectPricePredictor(
            feature_config=FeatureConfig(),
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_encoder_heads=num_encoder_heads,
            max_seq_len=seq_len,
        )
    else:
        raise ValueError(f"Evaluation for {pred_mode} mode not yet implemented")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"\nTest period: {test_start} to {test_end}")

    # Create dataset with many stocks per date (for IC computation)
    # IC requires multiple stocks per date to compute correlation
    dataset = CrossSectionalDataset(
        dataset_path=data_path,
        prices_path=prices_path,
        start_date=test_start,
        end_date=test_end,
        seq_len=seq_len,
        num_stocks=num_stocks,
        pred_day=1,
        min_stocks_per_date=min(10, num_stocks),  # Need at least some stocks per date
    )

    # Process multiple dates at a time for speed
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  Test dates: {len(dataset)}")
    print(f"  Stocks per date: {num_stocks}")

    # Collect predictions by date
    date_to_preds = defaultdict(list)
    date_to_targets = defaultdict(list)

    print("\nRunning inference...")
    with torch.no_grad():
        date_idx = 0
        for batch_features, batch_returns, batch_mask in tqdm(loader, desc="Evaluating"):
            # batch_features: (batch_size, num_stocks, seq_len, feat_dim)
            # batch_returns: (batch_size, num_stocks)
            # batch_mask: (batch_size, num_stocks)

            curr_batch_size = batch_features.shape[0]
            num_stocks_batch = batch_features.shape[1]

            # Flatten batch and stocks for efficient GPU processing
            # (batch_size, num_stocks, seq_len, feat_dim) -> (batch_size * num_stocks, seq_len, feat_dim)
            flat_features = batch_features.view(-1, batch_features.shape[2], batch_features.shape[3])
            flat_features = flat_features.to(device)

            # Single forward pass for all samples
            flat_preds = model(flat_features, return_tuple=False).cpu().numpy()  # (batch_size * num_stocks,)

            # Reshape back
            preds_batch = flat_preds.reshape(curr_batch_size, num_stocks_batch)
            returns_batch = batch_returns.numpy()
            mask_batch = batch_mask.numpy()

            # Assign to dates
            for b in range(curr_batch_size):
                date = dataset.dates[date_idx]
                date_idx += 1

                for i in range(num_stocks_batch):
                    if mask_batch[b, i]:
                        date_to_preds[date].append(preds_batch[b, i])
                        date_to_targets[date].append(returns_batch[b, i])

    # Compute daily IC
    dates = sorted(date_to_preds.keys())
    daily_ic = []
    daily_rank_ic = []
    valid_dates = []

    for date in dates:
        preds = np.array(date_to_preds[date])
        targets = np.array(date_to_targets[date])

        if len(preds) < 5:
            continue

        # Clean NaN
        mask = ~(np.isnan(preds) | np.isnan(targets))
        if mask.sum() < 5:
            continue

        preds = preds[mask]
        targets = targets[mask]

        ic = pearsonr(preds, targets)[0]
        rank_ic = spearmanr(preds, targets)[0]

        if not np.isnan(ic) and not np.isnan(rank_ic):
            daily_ic.append(ic)
            daily_rank_ic.append(rank_ic)
            valid_dates.append(date)

    daily_ic = np.array(daily_ic)
    daily_rank_ic = np.array(daily_rank_ic)

    # Compute aggregate metrics
    mean_ic = np.mean(daily_ic)
    std_ic = np.std(daily_ic)
    ir = mean_ic / std_ic if std_ic > 0 else 0

    mean_rank_ic = np.mean(daily_rank_ic)
    std_rank_ic = np.std(daily_rank_ic)
    rank_ir = mean_rank_ic / std_rank_ic if std_rank_ic > 0 else 0

    pct_positive_ic = (daily_ic > 0).mean() * 100
    pct_positive_rank_ic = (daily_rank_ic > 0).mean() * 100

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Mean IC':<25} {mean_ic:>+15.6f}")
    print(f"{'Std IC':<25} {std_ic:>15.6f}")
    print(f"{'IR (IC/std)':<25} {ir:>+15.4f}")
    print(f"{'% Positive IC':<25} {pct_positive_ic:>14.1f}%")
    print("-" * 42)
    print(f"{'Mean Rank IC':<25} {mean_rank_ic:>+15.6f}")
    print(f"{'Std Rank IC':<25} {std_rank_ic:>15.6f}")
    print(f"{'Rank IR':<25} {rank_ir:>+15.4f}")
    print(f"{'% Positive Rank IC':<25} {pct_positive_rank_ic:>14.1f}%")
    print("-" * 42)
    print(f"{'Test Days':<25} {len(daily_ic):>15}")
    print("=" * 60)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'checkpoint': checkpoint_path,
            'test_start': test_start,
            'test_end': test_end,
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            'ir': float(ir),
            'pct_positive_ic': float(pct_positive_ic),
            'mean_rank_ic': float(mean_rank_ic),
            'std_rank_ic': float(std_rank_ic),
            'rank_ir': float(rank_ir),
            'pct_positive_rank_ic': float(pct_positive_rank_ic),
            'num_days': len(daily_ic),
        }

        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Plot IC over time
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Convert dates to datetime for plotting
        plot_dates = [datetime.strptime(d, '%Y-%m-%d') for d in valid_dates]

        # IC over time
        ax1 = axes[0]
        ax1.plot(plot_dates, daily_ic, 'b-', alpha=0.5, linewidth=0.8, label='Daily IC')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.axhline(y=mean_ic, color='r', linestyle='--', linewidth=1.5, label=f'Mean IC: {mean_ic:+.4f}')
        # Rolling mean
        window = min(20, len(daily_ic) // 5)
        if window > 1:
            rolling_ic = np.convolve(daily_ic, np.ones(window)/window, mode='valid')
            rolling_dates = plot_dates[window-1:]
            ax1.plot(rolling_dates, rolling_ic, 'g-', linewidth=2, label=f'{window}-day Rolling IC')
        ax1.set_ylabel('IC (Pearson)')
        ax1.set_title(f'Information Coefficient Over Time | IR: {ir:+.3f}')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Rank IC over time
        ax2 = axes[1]
        ax2.plot(plot_dates, daily_rank_ic, 'b-', alpha=0.5, linewidth=0.8, label='Daily Rank IC')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axhline(y=mean_rank_ic, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_rank_ic:+.4f}')
        if window > 1:
            rolling_rank_ic = np.convolve(daily_rank_ic, np.ones(window)/window, mode='valid')
            ax2.plot(rolling_dates, rolling_rank_ic, 'g-', linewidth=2, label=f'{window}-day Rolling')
        ax2.set_ylabel('Rank IC (Spearman)')
        ax2.set_title(f'Rank IC Over Time | Rank IR: {rank_ir:+.3f}')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Cumulative IC
        ax3 = axes[2]
        cumsum_ic = np.cumsum(daily_ic)
        ax3.plot(plot_dates, cumsum_ic, 'b-', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Cumulative IC')
        ax3.set_xlabel('Date')
        ax3.set_title('Cumulative IC (should trend upward for good model)')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(output_dir, 'ic_over_time.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.close()

        # IC distribution histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        ax1.hist(daily_ic, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax1.axvline(x=mean_ic, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_ic:+.4f}')
        ax1.set_xlabel('IC')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'IC Distribution | {pct_positive_ic:.1f}% Positive')
        ax1.legend()

        ax2 = axes[1]
        ax2.hist(daily_rank_ic, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax2.axvline(x=mean_rank_ic, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_rank_ic:+.4f}')
        ax2.set_xlabel('Rank IC')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Rank IC Distribution | {pct_positive_rank_ic:.1f}% Positive')
        ax2.legend()

        plt.tight_layout()

        hist_path = os.path.join(output_dir, 'ic_distribution.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {hist_path}")
        plt.close()

    dataset.close()

    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'ir': ir,
        'mean_rank_ic': mean_rank_ic,
        'rank_ir': rank_ir,
        'daily_ic': daily_ic,
        'daily_rank_ic': daily_rank_ic,
        'dates': valid_dates,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate single checkpoint with IC metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data', type=str, default='all_complete_dataset.h5',
                       help='Path to features HDF5')
    parser.add_argument('--prices', type=str, default='actual_prices_clean.h5',
                       help='Path to prices HDF5')
    parser.add_argument('--test-start', type=str, default='2021-01-01',
                       help='Test period start date')
    parser.add_argument('--test-end', type=str, default='2022-12-31',
                       help='Test period end date')
    parser.add_argument('--seq-len', type=int, default=64,
                       help='Sequence length (overridden by checkpoint if saved)')
    parser.add_argument('--num-stocks', type=int, default=100,
                       help='Number of stocks per date for IC computation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for results and plots')

    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        prices_path=args.prices,
        test_start=args.test_start,
        test_end=args.test_end,
        seq_len=args.seq_len,
        num_stocks=args.num_stocks,
        device=args.device,
        output_dir=args.output,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
