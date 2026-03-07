#!/usr/bin/env python3
"""
Train traditional ML models on encoder embeddings.

Extracts embeddings from a pretrained encoder and trains:
- XGBoost
- LightGBM
- Random Forest
- Ridge Regression (baseline)

Evaluates with IC, Rank IC, and IR metrics.

Usage:
    python -m training.train_ml_on_embeddings \
        --encoder checkpoints/pretrained_encoder.pt \
        --data all_complete_dataset.h5 \
        --prices actual_prices_clean.h5 \
        --train-start 2010-01-01 \
        --train-end 2019-12-31 \
        --test-start 2020-01-01 \
        --test-end 2021-12-31 \
        --device cuda
"""

import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Optional: XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed. Install with: pip install lightgbm")


def extract_embeddings(
    encoder,
    dataset,
    batch_size: int,
    device: torch.device,
    max_samples: int = None,
    desc: str = "Extracting embeddings",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract embeddings from encoder - streaming to avoid memory issues.

    Processes data in chunks: load -> encode -> discard raw features -> keep embeddings.
    Embeddings are much smaller than raw features (hidden_dim vs seq_len * feat_dim).

    Returns:
        embeddings: (N, hidden_dim) array
        returns: (N,) array of next-day returns
        dates: List of date strings
    """
    embeddings_list = []
    returns_list = []
    dates_list = []

    encoder.eval()
    total_samples = 0

    num_stocks = dataset.num_stocks if hasattr(dataset, 'num_stocks') else 50
    max_dates = (max_samples // num_stocks + 1) if max_samples else len(dataset)
    max_dates = min(max_dates, len(dataset))

    # Process in chunks to balance memory and speed
    chunk_size = batch_size  # Number of dates per chunk

    with torch.no_grad():
        for chunk_start in tqdm(range(0, max_dates, chunk_size), desc=desc):
            chunk_end = min(chunk_start + chunk_size, max_dates)

            # Load chunk of data
            chunk_features = []
            chunk_returns = []
            chunk_masks = []
            chunk_dates = []

            for idx in range(chunk_start, chunk_end):
                features, returns, mask = dataset[idx]
                chunk_features.append(features)
                chunk_returns.append(returns)
                chunk_masks.append(mask)
                chunk_dates.append(dataset.dates[idx] if hasattr(dataset, 'dates') else f"date_{idx}")

            # Stack into batch
            batch_features = torch.stack(chunk_features)  # (chunk, num_stocks, seq, feat)
            batch_returns = torch.stack(chunk_returns)    # (chunk, num_stocks)
            batch_masks = torch.stack(chunk_masks)        # (chunk, num_stocks)

            # Free chunk lists immediately
            del chunk_features, chunk_returns, chunk_masks

            chunk_size_actual = batch_features.shape[0]
            num_stocks_batch = batch_features.shape[1]

            # Flatten for encoding: (chunk * num_stocks, seq, feat)
            flat_features = batch_features.view(-1, batch_features.shape[2], batch_features.shape[3])

            # Free batch_features before GPU transfer
            del batch_features

            # Move to GPU and encode
            flat_features = flat_features.to(device)
            emb = encoder(flat_features).cpu().numpy()  # (chunk * num_stocks, hidden_dim)

            # Free GPU memory
            del flat_features
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Reshape embeddings
            emb = emb.reshape(chunk_size_actual, num_stocks_batch, -1)
            returns_np = batch_returns.numpy()
            mask_np = batch_masks.numpy()

            del batch_returns, batch_masks

            # Collect valid samples - vectorized
            for b in range(chunk_size_actual):
                valid_mask = mask_np[b].astype(bool)
                valid_emb = emb[b, valid_mask]
                valid_returns = returns_np[b, valid_mask]

                n_valid = len(valid_returns)
                if n_valid > 0:
                    embeddings_list.append(valid_emb)
                    returns_list.extend(valid_returns.tolist())
                    dates_list.extend([chunk_dates[b]] * n_valid)
                    total_samples += n_valid

            if max_samples and total_samples >= max_samples:
                break

    # Concatenate all embeddings
    if embeddings_list:
        embeddings = np.vstack(embeddings_list)[:max_samples] if max_samples else np.vstack(embeddings_list)
        returns = np.array(returns_list[:max_samples] if max_samples else returns_list)
        dates = dates_list[:max_samples] if max_samples else dates_list
    else:
        embeddings = np.array([])
        returns = np.array([])
        dates = []

    return embeddings, returns, dates


def compute_daily_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: List[str],
) -> Dict[str, float]:
    """Compute daily IC metrics."""
    # Group by date
    date_to_preds = defaultdict(list)
    date_to_targets = defaultdict(list)

    for pred, target, date in zip(predictions, targets, dates):
        date_to_preds[date].append(pred)
        date_to_targets[date].append(target)

    daily_ic = []
    daily_rank_ic = []

    for date in sorted(date_to_preds.keys()):
        preds = np.array(date_to_preds[date])
        targs = np.array(date_to_targets[date])

        if len(preds) < 5:
            continue

        # Clean NaN
        mask = ~(np.isnan(preds) | np.isnan(targs))
        if mask.sum() < 5:
            continue

        preds = preds[mask]
        targs = targs[mask]

        ic = pearsonr(preds, targs)[0]
        rank_ic = spearmanr(preds, targs)[0]

        if not np.isnan(ic) and not np.isnan(rank_ic):
            daily_ic.append(ic)
            daily_rank_ic.append(rank_ic)

    daily_ic = np.array(daily_ic)
    daily_rank_ic = np.array(daily_rank_ic)

    if len(daily_ic) == 0:
        return {'mean_ic': 0, 'std_ic': 0, 'ir': 0, 'mean_rank_ic': 0, 'rank_ir': 0, 'pct_positive': 0}

    mean_ic = np.mean(daily_ic)
    std_ic = np.std(daily_ic)
    ir = mean_ic / std_ic if std_ic > 0 else 0

    mean_rank_ic = np.mean(daily_rank_ic)
    std_rank_ic = np.std(daily_rank_ic)
    rank_ir = mean_rank_ic / std_rank_ic if std_rank_ic > 0 else 0

    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'ir': ir,
        'mean_rank_ic': mean_rank_ic,
        'std_rank_ic': std_rank_ic,
        'rank_ir': rank_ir,
        'pct_positive_ic': (daily_ic > 0).mean() * 100,
        'pct_positive_rank_ic': (daily_rank_ic > 0).mean() * 100,
        'num_days': len(daily_ic),
    }


def train_and_evaluate_models(
    train_emb: np.ndarray,
    train_returns: np.ndarray,
    test_emb: np.ndarray,
    test_returns: np.ndarray,
    test_dates: List[str],
    models_to_train: List[str] = None,
) -> Dict[str, Dict]:
    """
    Train multiple ML models and evaluate them.

    Args:
        train_emb: Training embeddings (N_train, hidden_dim)
        train_returns: Training returns (N_train,)
        test_emb: Test embeddings (N_test, hidden_dim)
        test_returns: Test returns (N_test,)
        test_dates: Test dates for daily IC computation
        models_to_train: List of model names to train (default: all available)

    Returns:
        Dict mapping model name to metrics dict
    """
    # Clean data
    train_emb = np.nan_to_num(train_emb, nan=0.0, posinf=0.0, neginf=0.0)
    test_emb = np.nan_to_num(test_emb, nan=0.0, posinf=0.0, neginf=0.0)
    train_returns = np.nan_to_num(train_returns, nan=0.0, posinf=0.0, neginf=0.0)
    test_returns = np.nan_to_num(test_returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features
    scaler = StandardScaler()
    train_emb_scaled = scaler.fit_transform(train_emb)
    test_emb_scaled = scaler.transform(test_emb)

    # Define models
    all_models = {}

    # Linear models (fast baselines)
    all_models['Ridge'] = Ridge(alpha=1.0)
    all_models['Lasso'] = Lasso(alpha=0.001, max_iter=2000)
    all_models['ElasticNet'] = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000)

    # Tree-based models
    all_models['RandomForest'] = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    all_models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
    )

    if HAS_XGB:
        all_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )

    if HAS_LGB:
        all_models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )

    # Filter models if specified
    if models_to_train:
        all_models = {k: v for k, v in all_models.items() if k in models_to_train}

    results = {}

    print("\n" + "=" * 70)
    print("TRAINING ML MODELS ON ENCODER EMBEDDINGS")
    print("=" * 70)
    print(f"\nTraining samples: {len(train_returns):,}")
    print(f"Test samples: {len(test_returns):,}")
    print(f"Embedding dim: {train_emb.shape[1]}")
    print(f"\nModels to train: {list(all_models.keys())}")

    for name, model in all_models.items():
        print(f"\n--- Training {name} ---")

        try:
            # Train
            model.fit(train_emb_scaled, train_returns)

            # Predict
            train_preds = model.predict(train_emb_scaled)
            test_preds = model.predict(test_emb_scaled)

            # Compute metrics
            train_ic = pearsonr(train_preds, train_returns)[0]
            test_metrics = compute_daily_ic(test_preds, test_returns, test_dates)

            results[name] = {
                'train_ic': train_ic,
                **test_metrics,
            }

            print(f"  Train IC: {train_ic:+.4f}")
            print(f"  Test IC:  {test_metrics['mean_ic']:+.4f} (IR: {test_metrics['ir']:+.3f})")
            print(f"  Rank IC:  {test_metrics['mean_rank_ic']:+.4f} (IR: {test_metrics['rank_ir']:+.3f})")
            print(f"  % Positive IC: {test_metrics['pct_positive_ic']:.1f}%")

        except Exception as e:
            print(f"  Error training {name}: {e}")
            results[name] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train ML models on encoder embeddings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to pretrained encoder checkpoint')
    parser.add_argument('--data', type=str, default='all_complete_dataset.h5',
                       help='Path to features HDF5')
    parser.add_argument('--prices', type=str, default='actual_prices_clean.h5',
                       help='Path to prices HDF5')

    # Date ranges
    parser.add_argument('--train-start', type=str, default='2010-01-01',
                       help='Training start date')
    parser.add_argument('--train-end', type=str, default='2019-12-31',
                       help='Training end date')
    parser.add_argument('--test-start', type=str, default='2020-01-01',
                       help='Test start date')
    parser.add_argument('--test-end', type=str, default='2021-12-31',
                       help='Test end date')

    # Model config
    parser.add_argument('--seq-len', type=int, default=64,
                       help='Sequence length (should match encoder)')
    parser.add_argument('--num-stocks', type=int, default=50,
                       help='Number of stocks per date for embedding extraction')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding extraction')
    parser.add_argument('--max-train-samples', type=int, default=100000,
                       help='Max training samples (for speed)')
    parser.add_argument('--max-test-samples', type=int, default=50000,
                       help='Max test samples')

    # Which models to train
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to train (default: all available). Options: Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM')

    # Output
    parser.add_argument('--output', type=str, default='results/ml_on_embeddings',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for encoder')

    args = parser.parse_args()

    # Import here to avoid circular imports
    from training.cross_sectional_model import StockEncoder, CrossSectionalDataset
    from training.multimodal_model import FeatureConfig

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load encoder
    print(f"\nLoading encoder from: {args.encoder}")
    checkpoint = torch.load(args.encoder, map_location='cpu')

    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 256)
    num_layers = config.get('num_layers', 4)
    num_heads = config.get('num_heads', 4)
    saved_seq_len = config.get('seq_len', args.seq_len)

    if saved_seq_len != args.seq_len:
        print(f"  Using seq_len={saved_seq_len} from checkpoint")
        args.seq_len = saved_seq_len

    encoder = StockEncoder(
        feature_config=FeatureConfig(),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=args.seq_len,
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()

    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}, Heads: {num_heads}")
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Create datasets
    print(f"\nCreating train dataset: {args.train_start} to {args.train_end}")
    train_dataset = CrossSectionalDataset(
        dataset_path=args.data,
        prices_path=args.prices,
        start_date=args.train_start,
        end_date=args.train_end,
        seq_len=args.seq_len,
        num_stocks=args.num_stocks,
        pred_day=1,
        min_stocks_per_date=10,
    )

    print(f"Creating test dataset: {args.test_start} to {args.test_end}")
    test_dataset = CrossSectionalDataset(
        dataset_path=args.data,
        prices_path=args.prices,
        start_date=args.test_start,
        end_date=args.test_end,
        seq_len=args.seq_len,
        num_stocks=args.num_stocks,
        pred_day=1,
        min_stocks_per_date=10,
    )

    # Extract embeddings
    print("\nExtracting embeddings...")
    train_emb, train_returns, train_dates = extract_embeddings(
        encoder, train_dataset, args.batch_size, device,
        max_samples=args.max_train_samples,
        desc="Train embeddings"
    )

    test_emb, test_returns, test_dates = extract_embeddings(
        encoder, test_dataset, args.batch_size, device,
        max_samples=args.max_test_samples,
        desc="Test embeddings"
    )

    print(f"\nExtracted {len(train_returns):,} train and {len(test_returns):,} test samples")

    # Train and evaluate models
    results = train_and_evaluate_models(
        train_emb, train_returns,
        test_emb, test_returns, test_dates,
        models_to_train=args.models,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Test IC':>10} {'IR':>8} {'Rank IC':>10} {'Rank IR':>8} {'% Pos':>8}")
    print("-" * 70)

    for name, metrics in sorted(results.items(), key=lambda x: x[1].get('ir', -999), reverse=True):
        if 'error' in metrics:
            print(f"{name:<20} ERROR: {metrics['error']}")
        else:
            print(f"{name:<20} {metrics['mean_ic']:>+10.4f} {metrics['ir']:>+8.3f} "
                  f"{metrics['mean_rank_ic']:>+10.4f} {metrics['rank_ir']:>+8.3f} "
                  f"{metrics['pct_positive_ic']:>7.1f}%")

    print("=" * 70)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, 'ml_results.json')

    # Convert numpy types to Python types for JSON serialization
    json_results = {}
    for name, metrics in results.items():
        json_results[name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in metrics.items()}

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Cleanup
    train_dataset.close()
    test_dataset.close()

    return results


if __name__ == '__main__':
    main()
