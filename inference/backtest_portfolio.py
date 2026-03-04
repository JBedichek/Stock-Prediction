#!/usr/bin/env python3
"""
Backtesting for Differentiable Portfolio Model

Uses the trained PortfolioModel with its built-in Gumbel-softmax top-k selection
to simulate trading and measure performance.

Usage:
    python -m inference.backtest_portfolio \
        --checkpoint checkpoints/portfolio/best_portfolio_model.pt \
        --data data/all_complete_dataset.h5 \
        --prices data/actual_prices.h5 \
        --start-date 2023-01-01 \
        --end-date 2023-12-31
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_portfolio_differentiable import (
    PortfolioDataset,
    PortfolioModel,
    collate_portfolio_batch
)
from training.model import SimpleTransformerPredictor


def load_portfolio_model(checkpoint_path: str, device: str = 'cuda') -> PortfolioModel:
    """Load trained portfolio model from checkpoint.

    Supports two checkpoint formats:
    - train_portfolio_differentiable.py: has 'args' key
    - walk_forward_portfolio.py: has 'config' key
    """
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both checkpoint formats
    if 'args' in checkpoint:
        # From train_portfolio_differentiable.py
        args = checkpoint['args']
    elif 'config' in checkpoint:
        # From walk_forward_portfolio.py - normalize key names
        config = checkpoint['config']
        args = {
            'hidden_dim': config.get('hidden_dim', 256),
            'num_layers': config.get('num_layers', 4),
            'num_heads': config.get('num_heads', 8),
            'top_k': config.get('top_k', 20),
            'selection': config.get('selection_method', 'gumbel'),
            'selection_method': config.get('selection_method', 'gumbel'),
            'min_temp': config.get('min_temperature', 0.2),
            'horizon_idx': config.get('horizon_idx', 0),
            'horizon_days': config.get('horizon_days', 1),
            'seq_len': config.get('seq_len', 60),
            'input_dim': config.get('input_dim'),
        }
    else:
        raise ValueError(f"Checkpoint missing 'args' or 'config' key. Found keys: {list(checkpoint.keys())}")

    # Reconstruct the encoder
    # We need to get input_dim from the checkpoint
    state_dict = checkpoint['model_state_dict']

    # Find input_dim from the encoder's input projection
    input_dim = args.get('input_dim')
    if input_dim is None:
        for key in state_dict:
            if 'encoder.input_proj.0.weight' in key:
                input_dim = state_dict[key].shape[1]
                break
        else:
            raise ValueError("Could not determine input_dim from checkpoint")

    # Extract args with defaults for backwards compatibility
    hidden_dim = args.get('hidden_dim', 256)
    num_layers = args.get('num_layers', 4)
    num_heads = args.get('num_heads', 8)
    top_k = args.get('top_k', 20)
    selection = args.get('selection') or args.get('selection_method', 'gumbel')

    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Top-k: {top_k}")
    print(f"  Selection method: {selection}")

    encoder = SimpleTransformerPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,  # No dropout for inference
        num_pred_days=4,
        pred_mode='regression'
    )

    model = PortfolioModel(
        encoder=encoder,
        k=top_k,
        selection_method=selection,
        initial_temperature=args.get('min_temp', 0.2),  # Use final temperature
        min_temperature=args.get('min_temp', 0.2),
        horizon_idx=args.get('horizon_idx', 0)
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully")
    return model, args


@torch.no_grad()
def run_backtest(
    model: PortfolioModel,
    dataset: PortfolioDataset,
    device: str = 'cuda',
    transaction_cost: float = 0.001
) -> dict:
    """
    Run backtest on the portfolio model.

    Args:
        model: Trained PortfolioModel
        dataset: PortfolioDataset for the test period
        device: Device to run on
        transaction_cost: Round-trip transaction cost (default 0.1%)

    Returns:
        Dictionary of backtest results
    """
    model.eval()

    daily_returns = []
    portfolio_values = [1.0]  # Start with $1 for easy percentage tracking
    trade_details = []

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Process one day at a time for detailed tracking
        shuffle=False,
        collate_fn=collate_portfolio_batch,
        num_workers=0
    )

    print(f"\nRunning backtest on {len(dataset)} trading days...")

    if len(dataset) == 0:
        raise ValueError("Dataset has no valid trading days in the specified date range")

    for idx, (features, returns, masks) in enumerate(tqdm(dataloader, desc="Backtesting")):
        date = dataset.valid_dates[idx]

        features = features.to(device)
        returns = returns.to(device)
        masks = masks.to(device)

        # Get model's stock selection
        scores, weights, confidence = model(features, masks, hard=True)

        # Compute portfolio return using model's method (ensures consistency)
        portfolio_return = model.compute_portfolio_return(weights, returns, masks).item()

        # Track which stocks were selected (move to CPU first)
        weights_cpu = weights[0].cpu()
        returns_cpu = returns[0].cpu()
        confidence_cpu = confidence[0].cpu()

        selected_mask = weights_cpu.numpy() > 0.5
        num_selected = selected_mask.sum()

        # Apply transaction costs (only if we actually traded)
        if num_selected > 0:
            portfolio_return_net = portfolio_return - transaction_cost
        else:
            portfolio_return_net = 0.0  # No trades, no return, no cost

        daily_returns.append(portfolio_return_net)
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return_net))

        # Extract details for selected stocks
        selected_returns = returns_cpu.numpy()[selected_mask]
        selected_confidence = confidence_cpu.numpy()[selected_mask]

        trade_details.append({
            'date': date,
            'num_stocks': int(num_selected),
            'gross_return': portfolio_return,
            'net_return': portfolio_return_net,
            'selected_returns': selected_returns.tolist() if num_selected > 0 else [],
            'avg_confidence': float(selected_confidence.mean()) if num_selected > 0 else 0.0,
        })

    # Calculate summary statistics
    daily_returns = np.array(daily_returns)
    portfolio_values = np.array(portfolio_values)

    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    avg_daily_return = daily_returns.mean() * 100
    std_daily_return = daily_returns.std() * 100

    # Sharpe ratio (annualized, assuming 252 trading days)
    if std_daily_return > 1e-6:
        sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Win rate
    win_rate = (daily_returns > 0).mean() * 100

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = drawdown.max() * 100

    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 1e-6:
        sortino_ratio = (avg_daily_return / (downside_returns.std() * 100)) * np.sqrt(252)
    else:
        sortino_ratio = 0.0

    # Calmar ratio (return / max drawdown)
    if max_drawdown > 0:
        # Annualize the return
        num_years = len(daily_returns) / 252
        annualized_return = ((1 + total_return/100) ** (1/num_years) - 1) * 100 if num_years > 0 else total_return
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0

    results = {
        'total_return_pct': total_return,
        'avg_daily_return_pct': avg_daily_return,
        'std_daily_return_pct': std_daily_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate_pct': win_rate,
        'max_drawdown_pct': max_drawdown,
        'num_trading_days': len(daily_returns),
        'daily_returns': daily_returns.tolist(),
        'portfolio_values': portfolio_values.tolist(),
        'trade_details': trade_details,
    }

    return results


def print_results(results: dict, args: dict):
    """Print backtest results."""
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")

    print(f"\nStrategy: Top-{args['top_k']} selection, {args.get('horizon_days', 1)}-day horizon")
    print(f"Period: {results['num_trading_days']} trading days")

    print(f"\n📈 Performance Metrics:")
    print(f"  Total Return:      {results['total_return_pct']:>+8.2f}%")
    print(f"  Avg Daily Return:  {results['avg_daily_return_pct']:>+8.4f}%")
    print(f"  Std Daily Return:  {results['std_daily_return_pct']:>8.4f}%")

    print(f"\n📊 Risk-Adjusted Metrics:")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:     {results['sortino_ratio']:>8.2f}")
    print(f"  Calmar Ratio:      {results['calmar_ratio']:>8.2f}")

    print(f"\n🎯 Trading Statistics:")
    print(f"  Win Rate:          {results['win_rate_pct']:>8.1f}%")
    print(f"  Max Drawdown:      {results['max_drawdown_pct']:>8.2f}%")

    # Best and worst days
    daily_returns = np.array(results['daily_returns'])
    if len(daily_returns) > 0:
        best_day_idx = daily_returns.argmax()
        worst_day_idx = daily_returns.argmin()

        print(f"\n📅 Notable Days:")
        print(f"  Best Day:  {results['trade_details'][best_day_idx]['date']} ({daily_returns[best_day_idx]*100:+.2f}%)")
        print(f"  Worst Day: {results['trade_details'][worst_day_idx]['date']} ({daily_returns[worst_day_idx]*100:+.2f}%)")

    print(f"\n{'='*60}\n")


def plot_results(results: dict, output_path: str):
    """Generate performance plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Portfolio value over time
    ax = axes[0, 0]
    portfolio_values = np.array(results['portfolio_values'])
    ax.plot(portfolio_values, linewidth=1.5, color='#2ecc71')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Portfolio Value (starting = 1.0)')
    ax.grid(alpha=0.3)

    # 2. Daily returns distribution
    ax = axes[0, 1]
    daily_returns = np.array(results['daily_returns']) * 100
    ax.hist(daily_returns, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(daily_returns.mean(), color='green', linestyle='-', linewidth=2,
               label=f'Mean: {daily_returns.mean():.3f}%')
    ax.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Drawdown over time
    ax = axes[1, 0]
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    ax.fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.7, color='#e74c3c')
    ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(alpha=0.3)

    # 4. Rolling Sharpe ratio (30-day)
    ax = axes[1, 1]
    window = 30
    if len(daily_returns) >= window:
        rolling_mean = np.convolve(daily_returns, np.ones(window)/window, mode='valid')
        rolling_std = np.array([daily_returns[i:i+window].std() for i in range(len(daily_returns)-window+1)])
        rolling_sharpe = np.where(rolling_std > 0, (rolling_mean / rolling_std) * np.sqrt(252), 0)
        ax.plot(rolling_sharpe, linewidth=1.5, color='#9b59b6')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(results['sharpe_ratio'], color='green', linestyle='-', alpha=0.7,
                   label=f'Overall: {results["sharpe_ratio"]:.2f}')
    ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Portfolio Backtest Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Backtest Portfolio Model')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to portfolio model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to features HDF5 file')
    parser.add_argument('--prices', type=str, required=True,
                       help='Path to prices HDF5 file')

    # Date range
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Backtest start date')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='Backtest end date')

    # Options
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Round-trip transaction cost (default: 0.1%%)')
    parser.add_argument('--cache-dir', type=str, default='/tmp/portfolio_backtest_cache',
                       help='Cache directory for dataset')
    parser.add_argument('--output', type=str, default='backtest_results.pt',
                       help='Output file for results')
    parser.add_argument('--plot', type=str, default='backtest_plot.png',
                       help='Output file for plot')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    model, model_args = load_portfolio_model(args.checkpoint, args.device)

    # Create test dataset
    print(f"\nLoading test data from {args.start_date} to {args.end_date}...")
    dataset = PortfolioDataset(
        dataset_path=args.data,
        prices_path=args.prices,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_len=model_args.get('seq_len', 60),
        horizon_days=model_args.get('horizon_days', 1),
        max_stocks_per_day=model_args.get('max_stocks', 300),
        cache_dir=args.cache_dir,
        rank=0,
        world_size=1,
    )

    # Verify feature dimensions match
    # Get expected input_dim from model
    for key in model.state_dict():
        if 'encoder.input_proj.0.weight' in key:
            expected_input_dim = model.state_dict()[key].shape[1]
            break
    else:
        expected_input_dim = None

    if expected_input_dim is not None and dataset.feature_dim != expected_input_dim:
        raise ValueError(
            f"Feature dimension mismatch: model expects {expected_input_dim}, "
            f"but dataset has {dataset.feature_dim}"
        )

    # Run backtest
    results = run_backtest(
        model=model,
        dataset=dataset,
        device=args.device,
        transaction_cost=args.transaction_cost
    )

    # Print results
    print_results(results, model_args)

    # Save results
    torch.save({
        'results': results,
        'model_args': model_args,
        'backtest_args': vars(args),
    }, args.output)
    print(f"💾 Saved results to {args.output}")

    # Generate plot
    plot_results(results, args.plot)


if __name__ == '__main__':
    main()
