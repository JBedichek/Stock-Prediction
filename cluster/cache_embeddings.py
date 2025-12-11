"""
Cache Stock Embeddings for Fast Cluster Filtering

Pre-computes transformer embeddings for all stocks over a date range and saves
to HDF5 for fast lookup during backtesting.

This eliminates the need to re-encode stocks every day during backtesting.

Usage:
    python -m cluster.cache_embeddings \
        --model-path checkpoints/best_model.pt \
        --dataset-path data/all_complete_dataset.h5 \
        --output-path data/embeddings_cache.h5 \
        --num-months 6 \
        --batch-size 32
"""

import torch
import h5py
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime, timedelta


class EmbeddingCacheBuilder:
    """Build cache of stock embeddings for fast cluster filtering."""

    def __init__(self, model_path: str, device: str = 'cuda', batch_size: int = 32):
        """
        Initialize cache builder.

        Args:
            model_path: Path to trained model
            device: Device for encoding
            batch_size: Batch size for encoding
        """
        self.device = device
        self.batch_size = batch_size

        print(f"\n{'='*80}")
        print("INITIALIZING EMBEDDING CACHE BUILDER")
        print(f"{'='*80}")

        # Load model
        print(f"\n📦 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            from training.train_new_format import SimpleTransformerPredictor

            config = checkpoint.get('config', {})
            state_dict = checkpoint['model_state_dict']

            # Auto-detect input_dim
            input_dim = state_dict['input_proj.0.weight'].shape[1]

            self.model = SimpleTransformerPredictor(
                input_dim=input_dim,
                hidden_dim=config.get('hidden_dim', 1024),
                num_layers=config.get('num_layers', 10),
                num_heads=config.get('num_heads', 16),
                dropout=config.get('dropout', 0.15),
                num_pred_days=4,
                pred_mode=config.get('pred_mode', 'classification')
            )
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model = checkpoint

        self.model.eval()
        self.model.to(device)
        print(f"  ✅ Model loaded")

    def encode_batch(self, features_batch: List[torch.Tensor]) -> np.ndarray:
        """
        Encode a batch of feature tensors.

        Args:
            features_batch: List of feature tensors (each is feature_dim)

        Returns:
            Embeddings array (batch_size, embedding_dim)
        """
        # Find max feature dimension
        max_dim = max(f.shape[0] for f in features_batch)

        # Pad to consistent dimension
        padded_batch = []
        for features in features_batch:
            current_dim = features.shape[0]
            if current_dim < max_dim:
                padding = torch.zeros(max_dim - current_dim, dtype=torch.float32)
                features = torch.cat([features, padding])
            elif current_dim > max_dim:
                features = features[:max_dim]

            padded_batch.append(features)

        # Stack into batch tensor
        batch_tensor = torch.stack(padded_batch).to(self.device)

        # Adjust to model input_dim
        model_input_dim = self.model.input_dim
        current_dim = batch_tensor.shape[1]

        if current_dim < model_input_dim:
            padding = torch.zeros(batch_tensor.shape[0], model_input_dim - current_dim, device=self.device)
            batch_tensor = torch.cat([batch_tensor, padding], dim=1)
        elif current_dim > model_input_dim:
            batch_tensor = batch_tensor[:, :model_input_dim]

        # Add sequence dimension
        seq_len = 2000
        batch_tensor = batch_tensor.unsqueeze(1).expand(-1, seq_len, -1)

        # Extract transformer embeddings
        with torch.no_grad():
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output
                return hook

            handle = self.model.transformer.register_forward_hook(get_activation('transformer'))
            _ = self.model(batch_tensor)
            handle.remove()

            transformer_out = activation['transformer']
            embeddings = transformer_out.mean(dim=1)

        return embeddings.cpu().numpy()

    def build_cache(self,
                   dataset_path: str,
                   output_path: str,
                   num_months: int = 6,
                   tickers: List[str] = None):
        """
        Build embedding cache for all stocks and dates.

        Args:
            dataset_path: Path to features dataset
            output_path: Path to save embeddings cache
            num_months: Number of months to cache
            tickers: List of tickers to cache (None = all)
        """
        print(f"\n{'='*80}")
        print("BUILDING EMBEDDING CACHE")
        print(f"{'='*80}")

        # Open dataset
        print(f"\n📂 Opening dataset: {dataset_path}")
        dataset = h5py.File(dataset_path, 'r')

        # Get tickers
        if tickers is None:
            tickers = sorted(list(dataset.keys()))
        print(f"  ✓ Processing {len(tickers)} tickers")

        # Get date range from first ticker
        first_ticker = tickers[0]
        all_dates = [d.decode('utf-8') for d in dataset[first_ticker]['dates'][:]]
        all_dates = sorted(all_dates)

        # Get last N months
        end_date = datetime.strptime(all_dates[-1], '%Y-%m-%d')
        start_date = end_date - timedelta(days=num_months * 30)
        start_date_str = start_date.strftime('%Y-%m-%d')

        cache_dates = [d for d in all_dates if d >= start_date_str]
        print(f"  ✓ Caching {len(cache_dates)} dates ({cache_dates[0]} to {cache_dates[-1]})")

        # Create output file
        print(f"\n💾 Creating cache file: {output_path}")
        cache_file = h5py.File(output_path, 'w')

        # Process each ticker
        print(f"\n🔄 Processing tickers...")
        for ticker in tqdm(tickers, desc="Tickers"):
            if ticker not in dataset:
                continue

            try:
                # Get features and dates for this ticker
                features = dataset[ticker]['features'][:]
                dates = [d.decode('utf-8') for d in dataset[ticker]['dates'][:]]

                # Create date -> index mapping
                date_to_idx = {d: i for i, d in enumerate(dates)}

                # Create ticker group in cache
                ticker_group = cache_file.create_group(ticker)

                # Process dates in batches
                batch_dates = []
                batch_features = []

                for date in cache_dates:
                    if date not in date_to_idx:
                        continue

                    idx = date_to_idx[date]
                    if idx >= len(features):
                        continue

                    feature_vec = features[idx]
                    batch_dates.append(date)
                    batch_features.append(torch.tensor(feature_vec, dtype=torch.float32))

                    # Process batch when full
                    if len(batch_features) >= self.batch_size:
                        embeddings = self.encode_batch(batch_features)

                        for date, emb in zip(batch_dates, embeddings):
                            ticker_group.create_dataset(date, data=emb, compression='gzip')

                        batch_dates = []
                        batch_features = []

                # Process remaining
                if len(batch_features) > 0:
                    embeddings = self.encode_batch(batch_features)

                    for date, emb in zip(batch_dates, embeddings):
                        ticker_group.create_dataset(date, data=emb, compression='gzip')

            except Exception as e:
                print(f"\n  ⚠️  Error processing {ticker}: {e}")
                continue

        # Close files
        dataset.close()
        cache_file.close()

        print(f"\n✅ Cache created successfully!")
        print(f"   Saved to: {output_path}")

        # Print cache stats
        cache_file = h5py.File(output_path, 'r')
        total_embeddings = sum(len(cache_file[ticker].keys()) for ticker in cache_file.keys())
        cache_file.close()

        print(f"\n📊 Cache Statistics:")
        print(f"   Tickers:    {len(tickers):>8,}")
        print(f"   Dates:      {len(cache_dates):>8,}")
        print(f"   Embeddings: {total_embeddings:>8,}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute embeddings cache for cluster filtering')

    # Required
    parser.add_argument('--model-path', type=str, default='./checkpoints/best_model_100m_1.18.pt',
                       help='Path to trained model')
    parser.add_argument('--dataset-path', type=str, default='./data/all_complete_dataset.h5',
                       help='Path to features dataset')
    parser.add_argument('--output-path', type=str, default='./data/embeddings_cache.h5',
                       help='Path to save embeddings cache')

    # Optional
    parser.add_argument('--num-months', type=int, default=6,
                       help='Number of months to cache (from end of dataset)')
    parser.add_argument('--num-stocks', type=int, default=None,
                       help='Number of stocks to process (None = all)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for encoding')

    args = parser.parse_args()

    # Create builder
    builder = EmbeddingCacheBuilder(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )

    # Load dataset to get tickers
    with h5py.File(args.dataset_path, 'r') as f:
        all_tickers = sorted(list(f.keys()))

    if args.num_stocks:
        tickers = all_tickers[-args.num_stocks:]
        print(f"\nℹ️  Processing last {args.num_stocks} tickers")
    else:
        tickers = all_tickers

    # Build cache
    builder.build_cache(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_months=args.num_months,
        tickers=tickers
    )

    print(f"\n{'='*80}")
    print("CACHE BUILDING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTo use the cache during backtesting:")
    print(f"  python -m inference.backtest_simulation \\")
    print(f"    --cluster-dir cluster_results \\")
    print(f"    --best-clusters-file cluster_results/best_clusters_5d.txt \\")
    print(f"    --embeddings-cache {args.output_path}")


if __name__ == '__main__':
    main()
