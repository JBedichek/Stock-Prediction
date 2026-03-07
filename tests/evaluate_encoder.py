#!/usr/bin/env python3
"""
Encoder Quality Evaluation via Linear Probe

Evaluates pretrained encoder quality by comparing linear probe IC on:
1. Pretrained encoder embeddings
2. Raw features (baseline)

This helps determine if contrastive pretraining learned useful representations.

Usage:
    python -m tests.evaluate_encoder \
        --encoder checkpoints/pretrained_encoder.pt \
        --data all_complete_dataset.h5 \
        --prices actual_prices_clean.h5 \
        --device cuda
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, pearsonr


def compute_ic(pred: np.ndarray, actual: np.ndarray) -> tuple:
    """
    Compute IC (Pearson) and Rank IC (Spearman) between predictions and actuals.

    Args:
        pred: Predicted values
        actual: Actual values

    Returns:
        (ic, rank_ic) tuple
    """
    # Remove any NaN
    mask = ~(np.isnan(pred) | np.isnan(actual))
    pred = pred[mask]
    actual = actual[mask]

    if len(pred) < 10:
        return 0.0, 0.0

    ic = pearsonr(pred, actual)[0]
    rank_ic = spearmanr(pred, actual)[0]
    return ic, rank_ic


def collect_embeddings(
    encoder: nn.Module,
    dataset,
    batch_size: int,
    max_samples: int,
    device: torch.device,
    desc: str = "Encoding",
) -> tuple:
    """
    Collect encoder embeddings, raw features, and returns from dataset.

    Args:
        encoder: Encoder model (in eval mode)
        dataset: CrossSectionalDataset instance
        batch_size: Batch size for encoding
        max_samples: Maximum samples to collect
        device: Device for encoder
        desc: Progress bar description

    Returns:
        (embeddings, raw_features, returns) numpy arrays
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    embeddings_list = []
    raw_features_list = []
    returns_list = []

    total = 0
    pbar = tqdm(loader, desc=desc)

    with torch.no_grad():
        for features, returns, mask in pbar:
            # features: (batch, 1, seq_len, feat_dim) -> squeeze to (batch, seq_len, feat_dim)
            features = features.squeeze(1)
            returns = returns.squeeze(1)

            # Get encoder embeddings
            features_gpu = features.to(device)
            emb = encoder(features_gpu).cpu().numpy()
            embeddings_list.append(emb)

            # Raw features: use last timestep
            raw = features[:, -1, :].numpy()  # (batch, feat_dim)
            raw_features_list.append(raw)

            returns_list.append(returns.numpy())

            total += features.shape[0]
            if total >= max_samples:
                break

    embeddings = np.concatenate(embeddings_list, axis=0)[:max_samples]
    raw_features = np.concatenate(raw_features_list, axis=0)[:max_samples]
    returns = np.concatenate(returns_list, axis=0)[:max_samples]

    # Clean NaN/Inf
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    return embeddings, raw_features, returns


def run_linear_probe_eval(
    encoder: nn.Module,
    data_path: str,
    prices_path: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    num_samples: int = 5000,
    num_stocks_per_date: int = 50,
    epoch_num: int = 0,
    verbose: bool = False,
) -> Dict:
    """
    Run linear probe evaluation on encoder.

    This is the core evaluation function used both during training
    and for standalone evaluation.

    Args:
        encoder: Encoder model
        data_path: Path to features HDF5
        prices_path: Path to prices HDF5
        train_start/end: Date range for fitting linear probe
        test_start/end: Date range for evaluation
        seq_len: Sequence length
        batch_size: Batch size for encoding
        device: Device for encoder
        num_samples: Max samples to use
        num_stocks_per_date: Stocks per date for cross-sectional IC
        epoch_num: Current epoch (for logging)
        verbose: Print detailed results

    Returns:
        Dict with evaluation metrics
    """
    # Import here to avoid circular imports
    from training.cross_sectional_model import CrossSectionalDataset
    from collections import defaultdict

    was_training = encoder.training
    encoder.eval()

    # Create eval datasets with multiple stocks per date for proper IC
    train_dataset = CrossSectionalDataset(
        dataset_path=data_path,
        prices_path=prices_path,
        start_date=train_start,
        end_date=train_end,
        seq_len=seq_len,
        num_stocks=num_stocks_per_date,
        pred_day=1,
        min_stocks_per_date=10,
    )

    test_dataset = CrossSectionalDataset(
        dataset_path=data_path,
        prices_path=prices_path,
        start_date=test_start,
        end_date=test_end,
        seq_len=seq_len,
        num_stocks=num_stocks_per_date,
        pred_day=1,
        min_stocks_per_date=10,
    )

    # Collect embeddings with dates
    train_emb, train_raw, train_ret, train_dates = collect_embeddings_with_dates(
        encoder, train_dataset, batch_size, num_samples, device,
        desc="Encoding train" if verbose else "Train"
    )
    test_emb, test_raw, test_ret, test_dates = collect_embeddings_with_dates(
        encoder, test_dataset, batch_size, num_samples, device,
        desc="Encoding test" if verbose else "Test"
    )

    if verbose:
        print(f"  Train samples: {len(train_ret)}")
        print(f"  Test samples: {len(test_ret)}")

    # Fit linear probes
    ridge_emb = Ridge(alpha=1.0)
    ridge_emb.fit(train_emb, train_ret)
    pred_emb = ridge_emb.predict(test_emb)

    ridge_raw = Ridge(alpha=1.0)
    ridge_raw.fit(train_raw, train_ret)
    pred_raw = ridge_raw.predict(test_raw)

    # Compute daily IC (cross-sectional within each date)
    ic_emb, rank_ic_emb = compute_daily_ic(pred_emb, test_ret, test_dates)
    ic_raw, rank_ic_raw = compute_daily_ic(pred_raw, test_ret, test_dates)

    # Cleanup
    train_dataset.close()
    test_dataset.close()

    # Restore training mode
    if was_training:
        encoder.train()

    return {
        'epoch': epoch_num,
        'encoder_ic': ic_emb,
        'encoder_rank_ic': rank_ic_emb,
        'raw_ic': ic_raw,
        'raw_rank_ic': rank_ic_raw,
        'improvement_ic': ic_emb - ic_raw,
        'improvement_rank_ic': rank_ic_emb - rank_ic_raw,
        'train_samples': len(train_ret),
        'test_samples': len(test_ret),
    }


def collect_embeddings_with_dates(
    encoder: nn.Module,
    dataset,
    batch_size: int,
    max_samples: int,
    device: torch.device,
    desc: str = "Encoding",
) -> tuple:
    """
    Collect encoder embeddings, raw features, returns, and dates from dataset.
    """
    import torch.utils.data

    embeddings_list = []
    raw_features_list = []
    returns_list = []
    dates_list = []

    total = 0
    num_dates = min(len(dataset), max_samples // dataset.num_stocks + 1)

    with torch.no_grad():
        for idx in tqdm(range(num_dates), desc=desc):
            features, returns, mask = dataset[idx]
            date = dataset.dates[idx] if hasattr(dataset, 'dates') else f"date_{idx}"

            # features: (num_stocks, seq_len, feat_dim)
            # returns: (num_stocks,)
            # mask: (num_stocks,)

            features_gpu = features.to(device)
            emb = encoder(features_gpu).cpu().numpy()  # (num_stocks, hidden_dim)

            raw = features[:, -1, :].numpy()  # (num_stocks, feat_dim)
            returns_np = returns.numpy()
            mask_np = mask.numpy().astype(bool)

            # Only keep valid stocks
            for i in range(len(returns_np)):
                if mask_np[i]:
                    embeddings_list.append(emb[i])
                    raw_features_list.append(raw[i])
                    returns_list.append(returns_np[i])
                    dates_list.append(date)
                    total += 1

            if total >= max_samples:
                break

    embeddings = np.concatenate([e.reshape(1, -1) for e in embeddings_list], axis=0)[:max_samples]
    raw_features = np.concatenate([r.reshape(1, -1) for r in raw_features_list], axis=0)[:max_samples]
    returns = np.array(returns_list[:max_samples])
    dates = dates_list[:max_samples]

    # Clean NaN/Inf
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    return embeddings, raw_features, returns, dates


def compute_daily_ic(pred: np.ndarray, actual: np.ndarray, dates: list) -> tuple:
    """
    Compute mean daily IC (cross-sectional correlation within each date).

    This is the proper way to evaluate ranking models - IC should be computed
    within each cross-section (date), then averaged.
    """
    from collections import defaultdict

    # Group by date
    date_preds = defaultdict(list)
    date_actuals = defaultdict(list)

    for p, a, d in zip(pred, actual, dates):
        date_preds[d].append(p)
        date_actuals[d].append(a)

    daily_ics = []
    daily_rank_ics = []

    for date in date_preds:
        p = np.array(date_preds[date])
        a = np.array(date_actuals[date])

        if len(p) < 5:
            continue

        mask = ~(np.isnan(p) | np.isnan(a))
        if mask.sum() < 5:
            continue

        p, a = p[mask], a[mask]

        ic = pearsonr(p, a)[0]
        rank_ic = spearmanr(p, a)[0]

        if not np.isnan(ic):
            daily_ics.append(ic)
        if not np.isnan(rank_ic):
            daily_rank_ics.append(rank_ic)

    mean_ic = np.mean(daily_ics) if daily_ics else 0.0
    mean_rank_ic = np.mean(daily_rank_ics) if daily_rank_ics else 0.0

    return mean_ic, mean_rank_ic


def evaluate_encoder(
    encoder_path: str,
    data_path: str,
    prices_path: str,
    train_start: str = '2010-01-01',
    train_end: str = '2019-12-31',
    test_start: str = '2020-01-01',
    test_end: str = '2021-12-31',
    seq_len: int = 64,
    batch_size: int = 256,
    device: str = 'cuda',
    num_samples: int = 10000,
) -> Dict:
    """
    Evaluate encoder quality by comparing linear probe IC on:
    1. Pretrained encoder embeddings
    2. Raw features (baseline)

    Uses a simple Ridge regression as the linear probe.

    Args:
        encoder_path: Path to pretrained encoder checkpoint
        data_path: Path to features HDF5
        prices_path: Path to prices HDF5
        train_start/end: Date range for fitting linear probe
        test_start/end: Date range for evaluation
        seq_len: Sequence length
        batch_size: Batch size for encoding
        device: Device for encoder
        num_samples: Max samples to use (for speed)

    Returns:
        Dict with evaluation metrics
    """
    # Import here to avoid circular imports
    from training.cross_sectional_model import StockEncoder, CrossSectionalDataset
    from training.multimodal_model import FeatureConfig

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("ENCODER QUALITY EVALUATION")
    print("=" * 60)

    # Load encoder
    print(f"\nLoading encoder from: {encoder_path}")
    checkpoint = torch.load(encoder_path, map_location='cpu')

    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 256)
    num_layers = config.get('num_layers', 4)
    num_heads = config.get('num_heads', 4)
    saved_seq_len = config.get('seq_len', seq_len)

    # Use saved seq_len if available
    if saved_seq_len != seq_len:
        print(f"  Note: Using seq_len={saved_seq_len} from checkpoint (not {seq_len})")
        seq_len = saved_seq_len

    encoder = StockEncoder(
        feature_config=FeatureConfig(),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len,
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()

    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}, Heads: {num_heads}")
    print(f"  Seq len: {seq_len}")
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Create datasets
    print(f"\nCreating datasets...")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Test:  {test_start} to {test_end}")

    # Run evaluation
    metrics = run_linear_probe_eval(
        encoder=encoder,
        data_path=data_path,
        prices_path=prices_path,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        num_samples=num_samples,
        num_stocks_per_date=50,  # Use 50 stocks for proper cross-sectional IC
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Method':<25} {'IC':>10} {'Rank IC':>10}")
    print("-" * 47)
    print(f"{'Raw Features (baseline)':<25} {metrics['raw_ic']:>10.4f} {metrics['raw_rank_ic']:>10.4f}")
    print(f"{'Pretrained Encoder':<25} {metrics['encoder_ic']:>10.4f} {metrics['encoder_rank_ic']:>10.4f}")
    print("-" * 47)
    print(f"{'Improvement':<25} {metrics['improvement_ic']:>+10.4f} {metrics['improvement_rank_ic']:>+10.4f}")

    if metrics['improvement_ic'] > 0:
        print(f"\nEncoder embeddings have {metrics['improvement_ic']:.4f} higher IC than raw features")
    else:
        print(f"\nRaw features outperform encoder by {-metrics['improvement_ic']:.4f} IC")

    print("=" * 60)

    return metrics


def print_results_table(metrics: Dict) -> None:
    """Print evaluation results in a formatted table."""
    print(f"\n{'Method':<25} {'IC':>10} {'Rank IC':>10}")
    print("-" * 47)
    print(f"{'Raw Features (baseline)':<25} {metrics['raw_ic']:>10.4f} {metrics['raw_rank_ic']:>10.4f}")
    print(f"{'Pretrained Encoder':<25} {metrics['encoder_ic']:>10.4f} {metrics['encoder_rank_ic']:>10.4f}")
    print("-" * 47)
    print(f"{'Improvement':<25} {metrics['improvement_ic']:>+10.4f} {metrics['improvement_rank_ic']:>+10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate encoder quality via linear probe',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to pretrained encoder checkpoint')
    parser.add_argument('--data', type=str, default='all_complete_dataset.h5',
                       help='Path to features HDF5 file')
    parser.add_argument('--prices', type=str, default='actual_prices_clean.h5',
                       help='Path to prices HDF5 file')
    parser.add_argument('--train-start', type=str, default='2010-01-01',
                       help='Start date for linear probe training')
    parser.add_argument('--train-end', type=str, default='2018-12-31',
                       help='End date for linear probe training')
    parser.add_argument('--test-start', type=str, default='2019-01-01',
                       help='Start date for evaluation')
    parser.add_argument('--test-end', type=str, default='2020-12-31',
                       help='End date for evaluation')
    parser.add_argument('--seq-len', type=int, default=64,
                       help='Sequence length (overridden by checkpoint if saved)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for encoding')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Maximum samples to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda, cuda:0, cpu)')

    args = parser.parse_args()

    evaluate_encoder(
        encoder_path=args.encoder,
        data_path=args.data,
        prices_path=args.prices,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        num_samples=args.num_samples,
    )


if __name__ == '__main__':
    main()
