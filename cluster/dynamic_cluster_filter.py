"""
Dynamic Cluster-Based Stock Filtering for Inference

This module provides DYNAMIC cluster filtering where stocks are encoded and assigned
to clusters EVERY DAY based on their current features, rather than static assignment.

Key difference from static filtering:
- Static: "AAPL is always in cluster 5"
- Dynamic: "Today, given current market conditions, AAPL is in cluster 5"

This captures regime changes - a stock might be in different clusters on different days.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
import h5py
from tqdm import tqdm


class DynamicClusterFilter:
    """
    Dynamically assign stocks to clusters based on current features.

    Usage:
        # Initialize once
        filter = DynamicClusterFilter(
            model_path='checkpoints/best_model.pt',
            cluster_dir='cluster_results',
            best_cluster_ids=[1, 5, 8, 12]  # From analysis
        )

        # Every trading day
        for date in trading_dates:
            # Get candidate stocks for this day
            stocks = ['AAPL', 'GOOGL', 'MSFT', ...]
            features = {ticker: features_for_ticker for ticker, ...}

            # Filter to stocks in good clusters TODAY
            allowed_stocks = filter.filter_stocks_for_date(features)

            # Now pick top-k from allowed_stocks only
            predictions = model.predict(allowed_stocks)
            top_k = select_top_k(predictions)
    """

    def __init__(self,
                 model_path: str,
                 cluster_dir: str,
                 best_cluster_ids: Optional[List[int]] = None,
                 best_clusters_file: Optional[str] = None,
                 device: str = 'cuda',
                 batch_size: int = 32):
        """
        Initialize dynamic cluster filter.

        Args:
            model_path: Path to trained model checkpoint
            cluster_dir: Directory with clustering results
            best_cluster_ids: List of good cluster IDs
            best_clusters_file: File with good cluster IDs (alternative to list)
            device: Device for encoding
            batch_size: Batch size for encoding (default: 32)
        """
        self.device = device
        self.batch_size = batch_size

        print(f"\n{'='*80}")
        print("INITIALIZING DYNAMIC CLUSTER FILTER")
        print(f"{'='*80}")

        # Load model for encoding
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

        # Load clustering model and scaler
        cluster_dir = Path(cluster_dir)

        print(f"\n📂 Loading clustering artifacts from: {cluster_dir}")

        with open(cluster_dir / 'clustering_model.pkl', 'rb') as f:
            self.clustering_model = pickle.load(f)
        print(f"  ✅ Loaded clustering model")

        # Load scaler if it exists
        scaler_path = cluster_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"  ✅ Loaded scaler")
        else:
            self.scaler = None
            print(f"  ℹ️  No scaler found")

        # Load PCA if it exists
        pca_path = cluster_dir / 'pca.pkl'
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            print(f"  ✅ Loaded PCA")
        else:
            self.pca = None
            print(f"  ℹ️  No PCA found")

        # Load best cluster IDs
        if best_clusters_file:
            self.best_cluster_ids = self._load_best_clusters_from_file(best_clusters_file)
        elif best_cluster_ids:
            self.best_cluster_ids = set(best_cluster_ids)
        else:
            raise ValueError("Must provide either best_clusters_file or best_cluster_ids")

        print(f"  ✅ Loaded {len(self.best_cluster_ids)} best cluster IDs: {sorted(self.best_cluster_ids)}")

        print(f"\n✅ Dynamic cluster filter ready!")

    def _load_best_clusters_from_file(self, filepath: str) -> Set[int]:
        """Load best cluster IDs from file."""
        cluster_ids = set()

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    cluster_ids.add(int(line))

        return cluster_ids

    def encode_features(self, features_dict: Dict[str, torch.Tensor], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Encode stocks using transformer mean pooling (batched for memory efficiency).

        Args:
            features_dict: {ticker: features_tensor} where features is (seq_len, feature_dim) or (feature_dim,)
            batch_size: Batch size for encoding (default: 32)

        Returns:
            {ticker: embedding} dictionary
        """
        if len(features_dict) == 0:
            return {}

        # First pass: find max feature dimension and prepare data
        tickers = []
        features_list = []
        max_feature_dim = 0

        for ticker, features in features_dict.items():
            # Ensure features is a tensor
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            # Get feature dimension
            if len(features.shape) == 1:
                feature_dim = features.shape[0]
            else:
                feature_dim = features.shape[-1]

            max_feature_dim = max(max_feature_dim, feature_dim)
            tickers.append(ticker)
            features_list.append(features)

        # Process in batches to avoid OOM
        all_embeddings = {}

        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            batch_features = features_list[i:i+batch_size]

            # Pad batch to consistent dimension
            padded_batch = []
            for features in batch_features:
                # Pad if needed
                if len(features.shape) == 1:
                    current_dim = features.shape[0]
                    if current_dim < max_feature_dim:
                        padding = torch.zeros(max_feature_dim - current_dim, dtype=torch.float32)
                        features = torch.cat([features, padding])
                    elif current_dim > max_feature_dim:
                        features = features[:max_feature_dim]
                else:
                    current_dim = features.shape[-1]
                    if current_dim < max_feature_dim:
                        padding_shape = list(features.shape)
                        padding_shape[-1] = max_feature_dim - current_dim
                        padding = torch.zeros(padding_shape, dtype=torch.float32)
                        features = torch.cat([features, padding], dim=-1)
                    elif current_dim > max_feature_dim:
                        features = features[..., :max_feature_dim]

                padded_batch.append(features)

            # Stack and prepare for model
            batch_tensor = torch.stack(padded_batch).to(self.device)

            # Handle dimension adjustment if needed
            model_input_dim = self.model.input_dim

            if len(batch_tensor.shape) == 2:
                # (batch, feature_dim) - need to add sequence dimension
                current_feature_dim = batch_tensor.shape[1]

                # Pad/truncate to match model input_dim
                if current_feature_dim < model_input_dim:
                    padding = torch.zeros(batch_tensor.shape[0], model_input_dim - current_feature_dim, device=self.device)
                    batch_tensor = torch.cat([batch_tensor, padding], dim=1)
                elif current_feature_dim > model_input_dim:
                    batch_tensor = batch_tensor[:, :model_input_dim]

                # Add sequence dimension
                seq_len = 2000  # Match model's seq_len
                batch_tensor = batch_tensor.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, feature_dim)

            # Extract transformer embeddings using forward hook
            with torch.no_grad():
                activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = output
                    return hook

                # Register hook
                handle = self.model.transformer.register_forward_hook(get_activation('transformer'))

                # Forward pass
                _ = self.model(batch_tensor)

                # Remove hook
                handle.remove()

                # Mean pool transformer output
                transformer_out = activation['transformer']  # (batch, seq_len, hidden_dim)
                embeddings = transformer_out.mean(dim=1)  # (batch, hidden_dim)

            # Convert to numpy and store
            embeddings_np = embeddings.cpu().numpy()

            for ticker, emb in zip(batch_tickers, embeddings_np):
                all_embeddings[ticker] = emb

        return all_embeddings

    def assign_to_clusters(self, embeddings_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Assign stocks to clusters based on embeddings.

        Args:
            embeddings_dict: {ticker: embedding}

        Returns:
            {ticker: cluster_id} dictionary
        """
        if len(embeddings_dict) == 0:
            return {}

        # Convert to matrix
        tickers = list(embeddings_dict.keys())
        X = np.array([embeddings_dict[ticker] for ticker in tickers])

        # Apply same transformations used during clustering
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.pca is not None:
            X = self.pca.transform(X)

        # Predict cluster assignments
        cluster_labels = self.clustering_model.predict(X)

        # Create assignment dictionary
        assignments = {ticker: int(label) for ticker, label in zip(tickers, cluster_labels)}

        return assignments

    def filter_stocks_for_date(self,
                               features_dict: Dict[str, torch.Tensor],
                               return_assignments: bool = False) -> List[str]:
        """
        Filter stocks to only those in good clusters for this specific day.

        This is the main method for dynamic filtering during backtesting.

        Args:
            features_dict: {ticker: features} for all candidate stocks on this day
            return_assignments: If True, also return cluster assignments

        Returns:
            List of tickers in good clusters (or tuple if return_assignments=True)
        """
        # Step 1: Encode all stocks (batched for memory efficiency)
        embeddings = self.encode_features(features_dict, batch_size=self.batch_size)

        # Step 2: Assign to clusters
        assignments = self.assign_to_clusters(embeddings)

        # Step 3: Filter to good clusters
        allowed_tickers = [
            ticker for ticker, cluster_id in assignments.items()
            if cluster_id in self.best_cluster_ids
        ]

        if return_assignments:
            return allowed_tickers, assignments
        else:
            return allowed_tickers

    def get_cluster_stats_for_date(self, assignments: Dict[str, int]) -> Dict:
        """Get statistics about cluster distribution for a day."""
        from collections import Counter

        cluster_counts = Counter(assignments.values())

        total_stocks = len(assignments)
        stocks_in_good_clusters = sum(
            count for cluster_id, count in cluster_counts.items()
            if cluster_id in self.best_cluster_ids
        )

        return {
            'total_stocks': total_stocks,
            'stocks_in_good_clusters': stocks_in_good_clusters,
            'filter_rate': stocks_in_good_clusters / total_stocks if total_stocks > 0 else 0,
            'cluster_distribution': dict(cluster_counts),
            'good_clusters_represented': sum(
                1 for cluster_id in cluster_counts.keys()
                if cluster_id in self.best_cluster_ids
            )
        }


def main():
    """Example usage of dynamic cluster filter."""
    import argparse

    parser = argparse.ArgumentParser(description='Test dynamic cluster filter')

    parser.add_argument('--model-path', type=str, default='./checkpoints/best_model_100m_1.18.pt')
    parser.add_argument('--cluster-dir', type=str, default='./cluster_results')
    parser.add_argument('--best-clusters-file', type=str, default='./cluster_results/best_clusters_1d.txt')
    parser.add_argument('--dataset-path', type=str, default='./data/all_complete_dataset.h5')
    parser.add_argument('--test-date', type=str, default=None, help='Date to test (YYYY-MM-DD)')
    parser.add_argument('--num-stocks', type=int, default=100, help='Number of stocks to test')

    args = parser.parse_args()

    # Initialize filter
    filter = DynamicClusterFilter(
        model_path=args.model_path,
        cluster_dir=args.cluster_dir,
        best_clusters_file=args.best_clusters_file
    )

    # Load test data
    print(f"\n📂 Loading test data from: {args.dataset_path}")
    h5_file = h5py.File(args.dataset_path, 'r')

    tickers = sorted(list(h5_file.keys()))[:args.num_stocks]
    print(f"  Testing with {len(tickers)} stocks")

    # Get a test date
    if args.test_date:
        sample_ticker = tickers[0]
        dates = [d.decode('utf-8') for d in h5_file[sample_ticker]['dates'][:]]

        if args.test_date in dates:
            test_date_idx = dates.index(args.test_date)
        else:
            test_date_idx = len(dates) // 2
            print(f"  Date {args.test_date} not found, using middle date: {dates[test_date_idx]}")
    else:
        test_date_idx = -1  # Use last date

    # Extract features for all stocks on this date
    features_dict = {}
    for ticker in tickers:
        try:
            features = h5_file[ticker]['features'][test_date_idx]
            features_dict[ticker] = torch.tensor(features, dtype=torch.float32)
        except:
            continue

    print(f"  ✅ Loaded features for {len(features_dict)} stocks")

    # Apply dynamic filtering
    print(f"\n🔍 Applying dynamic cluster filter...")
    allowed_stocks, assignments = filter.filter_stocks_for_date(
        features_dict,
        return_assignments=True
    )

    # Show results
    print(f"\n{'='*80}")
    print("FILTERING RESULTS")
    print(f"{'='*80}")

    stats = filter.get_cluster_stats_for_date(assignments)

    print(f"\n  Total stocks:              {stats['total_stocks']:>6}")
    print(f"  Stocks in good clusters:   {stats['stocks_in_good_clusters']:>6} ({stats['filter_rate']*100:.1f}%)")
    print(f"  Good clusters represented: {stats['good_clusters_represented']:>6}")

    print(f"\n  Cluster distribution:")
    for cluster_id in sorted(stats['cluster_distribution'].keys()):
        count = stats['cluster_distribution'][cluster_id]
        is_good = "✓" if cluster_id in filter.best_cluster_ids else " "
        print(f"    {is_good} Cluster {cluster_id:>2}: {count:>4} stocks ({count/stats['total_stocks']*100:>5.1f}%)")

    print(f"\n  Sample allowed stocks: {allowed_stocks[:10]}")

    h5_file.close()


if __name__ == '__main__':
    main()
