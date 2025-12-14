#!/bin/bash
# Example: Using top-k percent cluster filtering in backtest simulation
#
# Instead of hard cluster assignment (stock is either IN or OUT of best clusters),
# this selects the top-k% of stocks closest to the best cluster centroids.
#
# This provides more flexibility and can improve performance by:
# - Including stocks near cluster boundaries
# - Allowing finer control over the number of candidate stocks
# - Adapting better to market conditions

# Example 1: Select top 30% of stocks closest to best cluster centroids
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model_100m_1.18.pt \
    --bin-edges adaptive_bin_edges.pt \
    --num-test-stocks 1000 \
    --subset-size 500 \
    --top-k 10 \
    --horizon-idx 0 \
    --test-months 6 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_1d.txt \
    --cluster-top-k-percent 0.3 \
    --embeddings-cache data/embeddings_cache.h5 \
    --output backtest_cluster_top30pct.pt

# Example 2: Select top 50% for more candidates
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model_100m_1.18.pt \
    --bin-edges adaptive_bin_edges.pt \
    --num-test-stocks 1000 \
    --subset-size 500 \
    --top-k 10 \
    --horizon-idx 0 \
    --test-months 6 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_1d.txt \
    --cluster-top-k-percent 0.5 \
    --embeddings-cache data/embeddings_cache.h5 \
    --output backtest_cluster_top50pct.pt

# Example 3: For comparison, hard assignment (original behavior)
# Just omit --cluster-top-k-percent to use hard cluster assignment
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model_100m_1.18.pt \
    --bin-edges adaptive_bin_edges.pt \
    --num-test-stocks 1000 \
    --subset-size 500 \
    --top-k 10 \
    --horizon-idx 0 \
    --test-months 6 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_1d.txt \
    --embeddings-cache data/embeddings_cache.h5 \
    --output backtest_cluster_hard.pt

echo "Results saved to:"
echo "  - backtest_cluster_top30pct.pt (top 30% by distance)"
echo "  - backtest_cluster_top50pct.pt (top 50% by distance)"
echo "  - backtest_cluster_hard.pt (hard assignment)"
