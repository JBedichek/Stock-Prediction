"""
Cluster-based Stock Filtering

This module implements cluster-based stock selection using transformer embeddings.

The approach:
1. Encode entire dataset using transformer mean pooling
2. Cluster the encodings (K-means, DBSCAN, etc.)
3. Analyze which clusters have highest returns and win rates
4. During inference, only trade stocks from profitable clusters

This reduces the search space and focuses on stocks with profitable patterns.
"""

from .create_clusters import ClusterEncoder, create_clusters_from_dataset
from .analyze_clusters import ClusterAnalyzer, analyze_cluster_performance
from .cluster_filter import ClusterFilter

__all__ = [
    'ClusterEncoder',
    'create_clusters_from_dataset',
    'ClusterAnalyzer',
    'analyze_cluster_performance',
    'ClusterFilter'
]
