"""
GPU-Accelerated K-means Clustering

PyTorch-based K-means implementation that runs on GPU for faster clustering.
"""

import torch
import numpy as np
from typing import Tuple
from tqdm import tqdm


def kmeans_gpu(X: np.ndarray, n_clusters: int, device: str = 'cuda', max_iters: int = 300, tol: float = 1e-4, n_init: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated K-means clustering using PyTorch.

    Args:
        X: Data matrix (n_samples, n_features)
        n_clusters: Number of clusters
        device: Device to run on ('cuda' or 'cpu')
        max_iters: Maximum iterations
        tol: Tolerance for convergence
        n_init: Number of random initializations (pick best result)

    Returns:
        (labels, centroids) - cluster labels and centroid locations
    """
    print(f"   Running K-means on {device.upper()}...")

    # Convert to torch tensor
    X_tensor = torch.from_numpy(X).float().to(device)
    n_samples, n_features = X_tensor.shape

    best_inertia = float('inf')
    best_labels = None
    best_centroids = None

    # Multiple random initializations (pick best)
    pbar_init = tqdm(range(n_init), desc="   K-means init", leave=True)
    for init_idx in pbar_init:
        # Initialize centroids using k-means++
        centroids = kmeans_plusplus_init(X_tensor, n_clusters, device)

        # EM algorithm
        converged = False
        pbar_iter = tqdm(range(max_iters), desc=f"     Init {init_idx+1}/{n_init}", leave=False)
        for iteration in pbar_iter:
            # E-step: Assign points to nearest centroid
            # Compute pairwise distances: (n_samples, n_clusters)
            distances = torch.cdist(X_tensor, centroids)  # (n_samples, n_clusters)
            labels = torch.argmin(distances, dim=1)  # (n_samples,)

            # M-step: Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X_tensor[mask].mean(dim=0)
                else:
                    # If cluster is empty, reinitialize with random point
                    new_centroids[k] = X_tensor[torch.randint(0, n_samples, (1,))]

            # Check convergence
            centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
            centroids = new_centroids

            # Update progress bar with convergence info
            pbar_iter.set_postfix({'shift': f'{centroid_shift:.6f}'})

            if centroid_shift < tol:
                converged = True
                pbar_iter.set_postfix({'shift': f'{centroid_shift:.6f}', 'status': 'converged'})
                break

        pbar_iter.close()

        # Compute inertia (sum of squared distances to nearest centroid)
        final_distances = torch.cdist(X_tensor, centroids)
        min_distances = torch.min(final_distances, dim=1)[0]
        inertia = (min_distances ** 2).sum().item()

        # Update outer progress bar with inertia
        pbar_init.set_postfix({'best_inertia': f'{best_inertia:.2f}', 'current': f'{inertia:.2f}'})

        # Keep best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    pbar_init.close()

    # Convert back to numpy
    labels_np = best_labels.cpu().numpy()
    centroids_np = best_centroids.cpu().numpy()

    print(f"   ✅ Best inertia: {best_inertia:.2f}")

    return labels_np, centroids_np


def kmeans_plusplus_init(X: torch.Tensor, n_clusters: int, device: str) -> torch.Tensor:
    """
    K-means++ initialization for better cluster initialization.

    Args:
        X: Data tensor (n_samples, n_features)
        n_clusters: Number of clusters
        device: Device to run on

    Returns:
        Initial centroids (n_clusters, n_features)
    """
    n_samples = X.shape[0]
    centroids = []

    # Choose first centroid randomly
    first_idx = torch.randint(0, n_samples, (1,), device=device)
    centroids.append(X[first_idx])

    # Choose remaining centroids
    for _ in range(n_clusters - 1):
        # Compute distances to nearest existing centroid
        centroids_tensor = torch.cat(centroids, dim=0)
        distances = torch.cdist(X, centroids_tensor)  # (n_samples, len(centroids))
        min_distances = torch.min(distances, dim=1)[0]  # (n_samples,)

        # Sample next centroid with probability proportional to distance^2
        probs = min_distances ** 2
        probs = probs / probs.sum()
        next_idx = torch.multinomial(probs, 1)
        centroids.append(X[next_idx])

    return torch.cat(centroids, dim=0)


class GPUKMeans:
    """
    Wrapper class to make GPU K-means compatible with sklearn interface.
    """
    def __init__(self, n_clusters: int, device: str = 'cuda', max_iter: int = 300, n_init: int = 10):
        self.n_clusters = n_clusters
        self.device = device
        self.max_iter = max_iter
        self.n_init = n_init
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X: np.ndarray):
        """Fit K-means on data."""
        self.labels_, self.cluster_centers_ = kmeans_gpu(
            X, self.n_clusters, self.device, self.max_iter, n_init=self.n_init
        )
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit K-means and return labels."""
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        X_tensor = torch.from_numpy(X).float().to(self.device)
        centroids_tensor = torch.from_numpy(self.cluster_centers_).float().to(self.device)
        distances = torch.cdist(X_tensor, centroids_tensor)
        labels = torch.argmin(distances, dim=1)
        return labels.cpu().numpy()
