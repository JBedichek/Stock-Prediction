"""
Create Clusters from Dataset Encodings

Encodes the entire dataset using transformer mean pooling and clusters the embeddings.

Process:
1. Load trained price predictor
2. Encode all stocks using transformer activations (mean pool over sequence)
3. Cluster the encodings using K-means or other methods
4. Save cluster assignments and centroids
"""

import torch
import h5py
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class ClusterEncoder:
    """
    Encode dataset using transformer mean pooling and create clusters.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize encoder.

        Args:
            model_path: Path to trained price predictor checkpoint
            device: Device to run on
        """
        self.device = device
        print(f"\n{'='*80}")
        print("INITIALIZING CLUSTER ENCODER")
        print(f"{'='*80}")

        print(f"\n📦 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Handle both direct model saves and checkpoint dicts
        if isinstance(checkpoint, dict):
            # Checkpoint dict format - need to reconstruct model
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # SimpleTransformerPredictor from train_new_format.py
                from training.train_new_format import SimpleTransformerPredictor

                config = checkpoint.get('config', {})

                # Get input_dim from model state dict
                state_dict = checkpoint['model_state_dict']
                # Find input_dim from first layer (input_proj.0.weight)
                for key in state_dict.keys():
                    if 'input_proj.0.weight' in key:
                        input_dim = state_dict[key].shape[1]
                        break
                else:
                    input_dim = 245  # Default from dataset (218 price + 27 summary)

                self.model = SimpleTransformerPredictor(
                    input_dim=input_dim,
                    hidden_dim=config.get('hidden_dim', 1024),
                    num_layers=config.get('num_layers', 10),
                    num_heads=config.get('num_heads', 16),
                    dropout=config.get('dropout', 0.15),
                    num_pred_days=4,  # [1, 5, 10, 20]
                    pred_mode=config.get('pred_mode', 'classification')
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError("Checkpoint does not contain 'model' or 'model_state_dict'")
        else:
            # Direct model save
            self.model = checkpoint

        self.model.eval()
        self.model.to(device)

        print(f"✅ Model loaded successfully")

    def encode_dataset(self, dataset_path: str,
                      max_stocks: Optional[int] = None,
                      batch_size: int = 64) -> Dict[str, np.ndarray]:
        """
        Encode entire dataset using transformer mean pooling.

        Args:
            dataset_path: Path to HDF5 dataset
            max_stocks: Optional limit on number of stocks to encode
            batch_size: Batch size for encoding

        Returns:
            Dictionary mapping ticker -> embedding (mean pooled transformer activation)
        """
        print(f"\n{'='*80}")
        print("ENCODING DATASET")
        print(f"{'='*80}")

        print(f"\n📂 Opening dataset: {dataset_path}")
        h5_file = h5py.File(dataset_path, 'r')
        tickers = sorted(list(h5_file.keys()))

        if max_stocks:
            tickers = tickers[:max_stocks]

        print(f"📊 Encoding {len(tickers)} stocks...")

        embeddings = {}

        # Process in batches
        for i in tqdm(range(0, len(tickers), batch_size), desc="Encoding"):
            batch_tickers = tickers[i:i+batch_size]
            batch_embeddings = self._encode_batch(h5_file, batch_tickers)

            for ticker, emb in zip(batch_tickers, batch_embeddings):
                embeddings[ticker] = emb

        h5_file.close()

        print(f"\n✅ Encoded {len(embeddings)} stocks")
        print(f"   Embedding dimension: {list(embeddings.values())[0].shape[0]}")

        return embeddings

    def _encode_batch(self, h5_file: h5py.File, tickers: List[str]) -> np.ndarray:
        """
        Encode a batch of stocks.

        Args:
            h5_file: Open HDF5 file
            tickers: List of tickers to encode

        Returns:
            Batch of embeddings (batch_size, embedding_dim)
        """
        batch_features = []

        for ticker in tickers:
            if ticker not in h5_file:
                continue

            # Get features (take first timestep or average across time)
            features = h5_file[ticker]['features'][:]  # (num_dates, num_features)

            # Use most recent timestep's features (or could average)
            if len(features.shape) == 2:
                features = features[-1]  # Take last timestep

            batch_features.append(features)

        if len(batch_features) == 0:
            return np.array([])

        # Convert to tensor
        batch_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32).to(self.device)

        # Reshape to (batch, seq_len, feature_dim) if needed
        # Assuming features are (batch, feature_dim), we need to add seq dimension
        if len(batch_tensor.shape) == 2:
            # Expand to create sequence (use same features for all timesteps)
            seq_len = 2000  # Match model's seq_len
            batch_tensor = batch_tensor.unsqueeze(1).expand(-1, seq_len, -1)

        # Get transformer activations (SimpleTransformerPredictor)
        # Extract mean-pooled transformer output using a forward hook
        with torch.no_grad():
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output
                return hook

            # Register hook on transformer output
            handle = self.model.transformer.register_forward_hook(get_activation('transformer'))

            # Forward pass
            _ = self.model(batch_tensor)

            # Remove hook
            handle.remove()

            # Get transformer output and mean pool
            transformer_out = activation['transformer']  # (batch, seq_len, hidden_dim)
            t_act = transformer_out.mean(dim=1)  # (batch, hidden_dim)

        # t_act shape: (batch, transformer_dim) - already mean pooled
        embeddings = t_act.cpu().numpy()

        return embeddings


def create_clusters_from_dataset(embeddings: Dict[str, np.ndarray],
                                 n_clusters: int = 50,
                                 method: str = 'kmeans',
                                 standardize: bool = True,
                                 use_pca: bool = False,
                                 pca_components: int = 50) -> Tuple[Dict[str, int], object, Optional[StandardScaler], Optional[PCA]]:
    """
    Create clusters from embeddings.

    Args:
        embeddings: Dictionary mapping ticker -> embedding
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'dbscan', 'agglomerative')
        standardize: Whether to standardize embeddings
        use_pca: Whether to use PCA for dimensionality reduction
        pca_components: Number of PCA components

    Returns:
        (cluster_assignments, clustering_model, scaler, pca)
    """
    print(f"\n{'='*80}")
    print("CREATING CLUSTERS")
    print(f"{'='*80}")

    # Convert to matrix
    tickers = list(embeddings.keys())
    X = np.array([embeddings[ticker] for ticker in tickers])

    print(f"\n📊 Dataset shape: {X.shape}")
    print(f"   Method: {method}")
    print(f"   Clusters: {n_clusters}")

    # Standardize
    scaler = None
    if standardize:
        print(f"   Standardizing embeddings...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # PCA
    pca = None
    if use_pca:
        print(f"   Applying PCA ({pca_components} components)...")
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Cluster
    print(f"\n🔬 Clustering...")
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'dbscan':
        clustering = DBSCAN(eps=0.5, min_samples=5)
    elif method == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown method: {method}")

    labels = clustering.fit_predict(X)

    # Create assignments
    cluster_assignments = {ticker: int(label) for ticker, label in zip(tickers, labels)}

    # Print cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n✅ Clustering complete!")
    print(f"   Number of clusters: {len(unique_labels)}")
    print(f"   Cluster sizes (min/mean/max): {counts.min()}/{counts.mean():.1f}/{counts.max()}")

    # Show largest clusters
    sorted_idx = np.argsort(-counts)
    print(f"\n   Top 5 largest clusters:")
    for i in range(min(5, len(unique_labels))):
        cluster_id = unique_labels[sorted_idx[i]]
        size = counts[sorted_idx[i]]
        print(f"      Cluster {cluster_id}: {size} stocks ({size/len(tickers)*100:.1f}%)")

    return cluster_assignments, clustering, scaler, pca


def visualize_clusters(embeddings: Dict[str, np.ndarray],
                      cluster_assignments: Dict[str, int],
                      output_path: str,
                      max_points: int = 5000):
    """
    Visualize clusters using PCA projection.

    Args:
        embeddings: Embedding dictionary
        cluster_assignments: Cluster assignments
        output_path: Path to save plot
        max_points: Maximum points to plot (for performance)
    """
    print(f"\n📊 Creating visualization...")

    # Sample if too many points
    tickers = list(embeddings.keys())
    if len(tickers) > max_points:
        tickers = np.random.choice(tickers, max_points, replace=False).tolist()

    X = np.array([embeddings[ticker] for ticker in tickers])
    labels = np.array([cluster_assignments[ticker] for ticker in tickers])

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=[color], label=f'Cluster {label}',
                  alpha=0.6, s=20)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Stock Clusters (PCA Projection)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")
    plt.close()


def save_cluster_results(cluster_assignments: Dict[str, int],
                         embeddings: Dict[str, np.ndarray],
                         clustering_model: object,
                         scaler: Optional[StandardScaler],
                         pca: Optional[PCA],
                         output_dir: str):
    """
    Save clustering results.

    Args:
        cluster_assignments: Cluster assignments
        embeddings: Embeddings
        clustering_model: Fitted clustering model
        scaler: Fitted scaler (if used)
        pca: Fitted PCA (if used)
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving results to: {output_dir}/")

    # Save assignments
    with open(output_dir / 'cluster_assignments.pkl', 'wb') as f:
        pickle.dump(cluster_assignments, f)
    print(f"   ✓ cluster_assignments.pkl")

    # Save embeddings
    with open(output_dir / 'embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"   ✓ embeddings.pkl")

    # Save model
    with open(output_dir / 'clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)
    print(f"   ✓ clustering_model.pkl")

    # Save scaler
    if scaler:
        with open(output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   ✓ scaler.pkl")

    # Save PCA
    if pca:
        with open(output_dir / 'pca.pkl', 'wb') as f:
            pickle.dump(pca, f)
        print(f"   ✓ pca.pkl")

    print(f"\n✅ All results saved!")


def main():
    parser = argparse.ArgumentParser(description='Create clusters from dataset encodings')

    # Required
    parser.add_argument('--model-path', type=str, default="./checkpoints/best_model_100m_1.18.pt", help='Path to trained predictor')
    parser.add_argument('--dataset-path', type=str, default="./data/all_complete_dataset.h5", help='Path to HDF5 dataset')

    # Encoding
    parser.add_argument('--max-stocks', type=int, default=100000, help='Max stocks to encode')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for encoding')

    # Clustering
    parser.add_argument('--n-clusters', type=int, default=50, help='Number of clusters')
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'dbscan', 'agglomerative'])
    parser.add_argument('--standardize', action='store_true', default=True, help='Standardize embeddings')
    parser.add_argument('--use-pca', action='store_true', help='Use PCA before clustering')
    parser.add_argument('--pca-components', type=int, default=50, help='Number of PCA components')

    # Output
    parser.add_argument('--output-dir', type=str, default='./cluster_results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', default=True, help='Create visualization')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create encoder
    encoder = ClusterEncoder(args.model_path, args.device)

    # Encode dataset
    embeddings = encoder.encode_dataset(
        args.dataset_path,
        max_stocks=args.max_stocks,
        batch_size=args.batch_size
    )

    # Create clusters
    cluster_assignments, clustering_model, scaler, pca = create_clusters_from_dataset(
        embeddings,
        n_clusters=args.n_clusters,
        method=args.method,
        standardize=args.standardize,
        use_pca=args.use_pca,
        pca_components=args.pca_components
    )

    # Visualize
    if args.visualize:
        output_path = Path(args.output_dir) / 'cluster_visualization.png'
        visualize_clusters(embeddings, cluster_assignments, str(output_path))

    # Save results
    save_cluster_results(
        cluster_assignments,
        embeddings,
        clustering_model,
        scaler,
        pca,
        args.output_dir
    )

    print(f"\n{'='*80}")
    print("CLUSTERING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext step: Analyze cluster performance")
    print(f"  python -m cluster.analyze_clusters \\")
    print(f"    --cluster-dir {args.output_dir} \\")
    print(f"    --dataset-path {args.dataset_path} \\")
    print(f"    --prices-path path/to/prices.h5")


if __name__ == '__main__':
    main()
