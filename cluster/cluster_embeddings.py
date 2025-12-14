"""
Cluster Embeddings

Takes pre-computed embeddings and creates clusters using various clustering algorithms.

Process:
1. Load embeddings from disk
2. Optionally standardize and apply PCA
3. Cluster the embeddings using K-means or other methods
4. Save cluster assignments and centroids
5. Optionally visualize clusters
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import GPU K-means from shared module
from cluster.gpu_kmeans import GPUKMeans


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from disk.

    Args:
        embeddings_path: Path to embeddings file (pickle format)

    Returns:
        Dictionary mapping key -> embedding
    """
    print(f"\n{'='*80}")
    print("LOADING EMBEDDINGS")
    print(f"{'='*80}")
    print(f"Loading from: {embeddings_path}")

    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    print(f"   ✅ Loaded {len(embeddings):,} embeddings")
    if len(embeddings) > 0:
        first_emb = next(iter(embeddings.values()))
        print(f"   Embedding dimension: {first_emb.shape[0]}")

    return embeddings


def create_clusters_from_embeddings(embeddings: Dict[str, np.ndarray],
                                    n_clusters: int = 50,
                                    method: str = 'kmeans',
                                    standardize: bool = True,
                                    use_pca: bool = False,
                                    pca_components: int = 50,
                                    device: str = 'cuda') -> Tuple[Dict[str, int], object, Optional[StandardScaler], Optional[PCA]]:
    """
    Create clusters from embeddings.

    Args:
        embeddings: Dictionary mapping key -> embedding
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'dbscan', 'agglomerative')
        standardize: Whether to standardize embeddings
        use_pca: Whether to use PCA for dimensionality reduction
        pca_components: Number of PCA components
        device: Device to run on ('cuda' or 'cpu') - only used for kmeans

    Returns:
        (cluster_assignments, clustering_model, scaler, pca)
    """
    print(f"\n{'='*80}")
    print("CREATING CLUSTERS")
    print(f"{'='*80}")

    # Convert to matrix
    keys = list(embeddings.keys())
    X = np.array([embeddings[key] for key in keys])

    print(f"\n📊 Dataset shape: {X.shape}")
    print(f"   Method: {method}")
    print(f"   Clusters: {n_clusters}")
    if method == 'kmeans':
        print(f"   Device: {device}")

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
        # Use GPU-accelerated K-means
        clustering = GPUKMeans(n_clusters=n_clusters, device=device, max_iter=300, n_init=10)
    elif method == 'dbscan':
        print(f"   ⚠️  DBSCAN runs on CPU only")
        clustering = DBSCAN(eps=0.5, min_samples=5)
    elif method == 'agglomerative':
        print(f"   ⚠️  Agglomerative clustering runs on CPU only")
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown method: {method}")

    labels = clustering.fit_predict(X)

    # Create assignments
    cluster_assignments = {key: int(label) for key, label in zip(keys, labels)}

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
        print(f"      Cluster {cluster_id}: {size} samples ({size/len(keys)*100:.1f}%)")

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
    keys = list(embeddings.keys())
    if len(keys) > max_points:
        keys = np.random.choice(keys, max_points, replace=False).tolist()

    X = np.array([embeddings[key] for key in keys])
    labels = np.array([cluster_assignments[key] for key in keys])

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
        embeddings: Embeddings (saved for reference)
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

    # Save embeddings (for reference)
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
    parser = argparse.ArgumentParser(description='Cluster pre-computed embeddings')

    # Required
    parser.add_argument('--embeddings-path', type=str, default='./cluster_results/embeddings.pkl',
                       help='Path to embeddings file (pickle format)')
    parser.add_argument('--output-dir', type=str, default='./cluster_results',
                       help='Output directory for cluster results')

    # Clustering parameters
    parser.add_argument('--n-clusters', type=int, default=500,
                       help='Number of clusters')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'agglomerative'],
                       help='Clustering method')
    parser.add_argument('--standardize', action='store_true', default=True,
                       help='Standardize embeddings before clustering')
    parser.add_argument('--use-pca', action='store_true',
                       help='Use PCA before clustering')
    parser.add_argument('--pca-components', type=int, default=50,
                       help='Number of PCA components')

    # Visualization
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='Create cluster visualization')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda or cpu) - only affects kmeans clustering')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load embeddings
    embeddings = load_embeddings(args.embeddings_path)

    # Create clusters
    cluster_assignments, clustering_model, scaler, pca = create_clusters_from_embeddings(
        embeddings,
        n_clusters=args.n_clusters,
        method=args.method,
        standardize=args.standardize,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        device=args.device
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
    print(f"    --dataset-path path/to/dataset.h5 \\")
    print(f"    --prices-path path/to/prices.h5")


if __name__ == '__main__':
    main()
