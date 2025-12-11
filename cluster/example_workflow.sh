#!/bin/bash
# Complete workflow for cluster-based stock filtering
#
# This script demonstrates the full pipeline:
# 1. Create clusters from dataset encodings
# 2. Analyze cluster performance
# 3. Identify best clusters
# 4. (Optional) Use for filtering in RL/backtesting
#
# Before running:
# 1. Update paths below
# 2. Make executable: chmod +x example_workflow.sh

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model and data paths
MODEL_PATH="checkpoints/best_model.pt"
DATASET_PATH="data/all_complete_dataset.h5"
PRICES_PATH="data/actual_prices.h5"

# Output directory
OUTPUT_DIR="./cluster_results"

# Clustering parameters
N_CLUSTERS=50
METHOD="kmeans"  # kmeans, dbscan, or agglomerative
BATCH_SIZE=64
MAX_STOCKS=""  # Leave empty for all stocks, or set limit like "1000"

# Analysis parameters
HORIZONS="1 5 10 20"  # Time horizons to analyze (in days)
MIN_RETURN=0.005      # 0.5% minimum return
MIN_WIN_RATE=0.52     # 52% minimum win rate
MIN_SHARPE=0.1        # 0.1 minimum Sharpe ratio
TOP_K=20              # Select top 20 clusters

# Device
DEVICE="cuda"

# =============================================================================
# STEP 1: CREATE CLUSTERS
# =============================================================================

echo "========================================================================"
echo "STEP 1: CREATE CLUSTERS"
echo "========================================================================"
echo ""
echo "Encoding dataset and creating clusters..."
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Clusters: $N_CLUSTERS"
echo "  Method: $METHOD"
echo ""

CMD="python -m cluster.create_clusters \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --n-clusters $N_CLUSTERS \
    --method $METHOD \
    --batch-size $BATCH_SIZE \
    --standardize \
    --output-dir $OUTPUT_DIR \
    --visualize \
    --device $DEVICE"

# Add max-stocks if specified
if [ -n "$MAX_STOCKS" ]; then
    CMD="$CMD --max-stocks $MAX_STOCKS"
fi

# Run clustering
$CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Clustering failed!"
    exit 1
fi

echo ""
echo "✓ Clustering complete!"
echo ""

# =============================================================================
# STEP 2: ANALYZE CLUSTER PERFORMANCE
# =============================================================================

echo "========================================================================"
echo "STEP 2: ANALYZE CLUSTER PERFORMANCE"
echo "========================================================================"
echo ""
echo "Computing performance metrics for each cluster..."
echo "  Prices: $PRICES_PATH"
echo "  Horizons: $HORIZONS"
echo "  Criteria:"
echo "    - Min return: ${MIN_RETURN} ($(echo "$MIN_RETURN * 100" | bc)%)"
echo "    - Min win rate: ${MIN_WIN_RATE} ($(echo "$MIN_WIN_RATE * 100" | bc)%)"
echo "    - Min Sharpe: ${MIN_SHARPE}"
echo "    - Top K: ${TOP_K} clusters"
echo ""

python -m cluster.analyze_clusters \
    --cluster-dir $OUTPUT_DIR \
    --dataset-path $DATASET_PATH \
    --prices-path $PRICES_PATH \
    --horizons $HORIZONS \
    --min-return $MIN_RETURN \
    --min-win-rate $MIN_WIN_RATE \
    --min-sharpe $MIN_SHARPE \
    --top-k $TOP_K

if [ $? -ne 0 ]; then
    echo "ERROR: Analysis failed!"
    exit 1
fi

echo ""
echo "✓ Analysis complete!"
echo ""

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo "========================================================================"
echo "WORKFLOW COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
echo "  - cluster_assignments.pkl         Ticker → cluster mapping"
echo "  - embeddings.pkl                  Ticker → embedding mapping"
echo "  - cluster_visualization.png       Visual representation"
echo "  - cluster_ranking_*d.csv          Ranked clusters per horizon"
echo "  - best_clusters_*d.txt            Best cluster IDs per horizon"
echo "  - cluster_performance_*d.png      Performance plots"
echo ""
echo "Key results for 5-day horizon:"
echo ""

# Show top 5 clusters from ranking
if [ -f "$OUTPUT_DIR/cluster_ranking_5d.csv" ]; then
    echo "Top 5 Clusters (5-day returns):"
    head -6 "$OUTPUT_DIR/cluster_ranking_5d.csv" | column -t -s,
    echo ""
fi

# Show best clusters
if [ -f "$OUTPUT_DIR/best_clusters_5d.txt" ]; then
    BEST_COUNT=$(grep -v '^#' "$OUTPUT_DIR/best_clusters_5d.txt" | wc -l)
    echo "Selected $BEST_COUNT best clusters for filtering"
    echo "  (See: $OUTPUT_DIR/best_clusters_5d.txt)"
    echo ""
fi

echo "========================================================================"
echo "NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Review visualizations:"
echo "   - $OUTPUT_DIR/cluster_visualization.png"
echo "   - $OUTPUT_DIR/cluster_performance_5d.png"
echo ""
echo "2. Check cluster rankings:"
echo "   cat $OUTPUT_DIR/cluster_ranking_5d.csv"
echo ""
echo "3. Use cluster filter in your code:"
echo "   "
echo "   from cluster import ClusterFilter"
echo "   "
echo "   filter = ClusterFilter("
echo "       cluster_dir='$OUTPUT_DIR',"
echo "       best_clusters_file='$OUTPUT_DIR/best_clusters_5d.txt'"
echo "   )"
echo "   "
echo "   allowed_tickers = filter.filter_tickers(all_tickers)"
echo ""
echo "4. Integrate with RL training:"
echo "   python -m rl.train_rl_phase2 \\"
echo "       --dataset-path $DATASET_PATH \\"
echo "       --predictor-checkpoint $MODEL_PATH \\"
echo "       ... (add cluster filtering in code)"
echo ""
echo "5. Or integrate with backtesting:"
echo "   python -m inference.backtest_simulation \\"
echo "       --dataset-path $DATASET_PATH \\"
echo "       --model-path $MODEL_PATH \\"
echo "       ... (add cluster filtering in code)"
echo ""
echo "Happy trading! 🚀"
echo ""
