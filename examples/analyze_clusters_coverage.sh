#!/bin/bash
# Example: Using coverage-based cluster selection
#
# Instead of arbitrary thresholds (min return, win rate, Sharpe),
# this selects the top-performing clusters that cover a target percentage of the total dataset.
#
# Benefits:
# - No need to tune threshold values
# - Consistent proportion of dataset
# - More data-driven approach
# - Automatically adapts to cluster distribution

# Example 1: Select clusters covering top 30% of stocks (recommended)
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --horizons 1 5 10 20 \
    --top-k-percent-coverage 0.3 \
    --ranking-metric mean_return \
    --output-dir cluster_results

# Example 2: Select clusters covering top 50% of stocks (more aggressive)
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --horizons 1 5 10 20 \
    --top-k-percent-coverage 0.5 \
    --ranking-metric mean_return \
    --output-dir cluster_results

# Example 3: Select by Sharpe ratio instead of mean return
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --horizons 1 5 10 20 \
    --top-k-percent-coverage 0.3 \
    --ranking-metric sharpe \
    --output-dir cluster_results

# Example 4: For comparison, threshold-based selection (legacy)
# Just omit --top-k-percent-coverage to use threshold mode
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --horizons 1 5 10 20 \
    --min-return 0.01 \
    --min-win-rate 0.5 \
    --min-sharpe 0.1 \
    --top-k 10 \
    --output-dir cluster_results

echo "Best clusters saved to cluster_results/best_clusters_*d.txt"
