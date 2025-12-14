#!/bin/bash
#
# Compare Statistical Performance Across Different Pool Sizes
#
# This script runs the statistical comparison for multiple pool sizes
# and generates a summary graph comparing performance across all sizes.
#
# Pool sizes tested: 50, 100, 256, 512, 1024, 1500
#

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
POOL_SIZES=(50 100 256 512 1024 1500)
DATA="data/all_complete_dataset.h5"
PRICES="data/actual_prices.h5"
MODEL="./checkpoints/best_model_100m_1.18.pt"
BIN_EDGES="data/adaptive_bin_edges.pt"
NUM_TRIALS=20
NUM_RANDOM_PER_TRIAL=25
TOP_K=5
HORIZON_IDX=0
CONFIDENCE_PERCENTILE=0.002
TEST_MONTHS=3
INITIAL_CAPITAL=100000
BATCH_SIZE=128
SEED=42

# Create output directory
OUTPUT_BASE="output/pool_size_comparison"
mkdir -p "${OUTPUT_BASE}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Pool Size Comparison - Statistical Analysis             ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo ""
echo -e "${YELLOW}Testing pool sizes: ${POOL_SIZES[@]}${NC}"
echo -e "${YELLOW}Number of trials per size: ${NUM_TRIALS}${NC}"
echo -e "${YELLOW}Random portfolios per trial: ${NUM_RANDOM_PER_TRIAL}${NC}"
echo ""

# Run comparison for each pool size
for pool_size in "${POOL_SIZES[@]}"; do
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Running analysis for pool size: ${pool_size}${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Create output directory for this pool size
    OUTPUT_DIR="${OUTPUT_BASE}/pool_${pool_size}"
    mkdir -p "${OUTPUT_DIR}"

    OUTPUT_FILE="${OUTPUT_DIR}/results.pt"

    # Run statistical comparison
    # (plots are automatically saved to OUTPUT_DIR by plot_results())
    python inference/statistical_comparison.py \
        --data "${DATA}" \
        --prices "${PRICES}" \
        --model "${MODEL}" \
        --bin-edges "${BIN_EDGES}" \
        --num-trials ${NUM_TRIALS} \
        --num-random-per-trial ${NUM_RANDOM_PER_TRIAL} \
        --subset-size ${pool_size} \
        --top-k ${TOP_K} \
        --horizon-idx ${HORIZON_IDX} \
        --confidence-percentile ${CONFIDENCE_PERCENTILE} \
        --test-months ${TEST_MONTHS} \
        --initial-capital ${INITIAL_CAPITAL} \
        --batch-size ${BATCH_SIZE} \
        --output "${OUTPUT_FILE}" \
        --seed ${SEED}

    echo -e "${GREEN}✓ Completed pool size ${pool_size}${NC}"
    echo ""
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Generating summary comparison graph...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Generate summary graph
python - <<EOF
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Pool sizes tested
pool_sizes = [50, 100, 256, 512, 1024, 1500]
output_base = "${OUTPUT_BASE}"

# Collect results
model_means = []
model_stds = []
random_means = []
random_stds = []

for pool_size in pool_sizes:
    results_path = f"{output_base}/pool_{pool_size}/results.pt"

    if not os.path.exists(results_path):
        print(f"Warning: Results not found for pool size {pool_size}")
        model_means.append(0)
        model_stds.append(0)
        random_means.append(0)
        random_stds.append(0)
        continue

    results = torch.load(results_path, weights_only=False)
    trial_results = results['trial_results']

    # Extract final capital from all trials
    model_finals = [trial['model']['final_capital'] for trial in trial_results]
    random_finals_all = []
    for trial in trial_results:
        for random_result in trial['random']:
            random_finals_all.append(random_result['final_capital'])

    model_means.append(np.mean(model_finals))
    model_stds.append(np.std(model_finals))
    random_means.append(np.mean(random_finals_all))
    random_stds.append(np.std(random_finals_all))

# Create summary plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Mean final portfolio value
x_pos = np.arange(len(pool_sizes))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, model_means, width, yerr=model_stds,
               label='Model', capsize=5, color='#2ecc71', alpha=0.8,
               edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x_pos + width/2, random_means, width, yerr=random_stds,
               label='Random', capsize=5, color='#95a5a6', alpha=0.8,
               edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Pool Size (Number of Stocks)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Final Portfolio Value (\$)', fontsize=13, fontweight='bold')
ax1.set_title('Final Portfolio Value vs Pool Size\n(Mean ± Std Dev)',
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(pool_sizes, fontsize=11)
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Add horizontal line for initial capital
ax1.axhline(y=${INITIAL_CAPITAL}, color='red', linestyle='--', linewidth=2,
           label=f'Initial Capital (\$${INITIAL_CAPITAL})', alpha=0.7)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'\${height:,.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Return percentage
model_returns = [(m / ${INITIAL_CAPITAL} - 1) * 100 for m in model_means]
random_returns = [(r / ${INITIAL_CAPITAL} - 1) * 100 for r in random_means]
model_return_stds = [(s / ${INITIAL_CAPITAL}) * 100 for s in model_stds]
random_return_stds = [(s / ${INITIAL_CAPITAL}) * 100 for s in random_stds]

bars3 = ax2.bar(x_pos - width/2, model_returns, width, yerr=model_return_stds,
               label='Model', capsize=5, color='#2ecc71', alpha=0.8,
               edgecolor='black', linewidth=1.5)
bars4 = ax2.bar(x_pos + width/2, random_returns, width, yerr=random_return_stds,
               label='Random', capsize=5, color='#95a5a6', alpha=0.8,
               edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Pool Size (Number of Stocks)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Total Return (%)', fontsize=13, fontweight='bold')
ax2.set_title('Total Return vs Pool Size\n(Mean ± Std Dev)',
             fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(pool_sizes, fontsize=11)
ax2.legend(fontsize=11, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

plt.tight_layout()
summary_path = f"{output_base}/pool_size_comparison_summary.png"
plt.savefig(summary_path, dpi=200, bbox_inches='tight')
print(f"✅ Summary graph saved to: {summary_path}")
plt.close()

# Create detailed statistics table
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Prepare table data
table_data = [['Pool Size', 'Model Mean', 'Model Std', 'Random Mean', 'Random Std',
               'Model Return', 'Improvement']]

for i, pool_size in enumerate(pool_sizes):
    model_return = model_returns[i]
    random_return = random_returns[i]
    improvement = model_return - random_return

    table_data.append([
        str(pool_size),
        f'\${model_means[i]:,.0f}',
        f'\${model_stds[i]:,.0f}',
        f'\${random_means[i]:,.0f}',
        f'\${random_stds[i]:,.0f}',
        f'{model_return:+.2f}%',
        f'{improvement:+.2f}%'
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.13])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        cell.set_edgecolor('gray')
        cell.set_linewidth(0.5)

ax.set_title('Pool Size Comparison - Detailed Statistics',
            fontsize=16, fontweight='bold', pad=20)

table_path = f"{output_base}/pool_size_statistics_table.png"
plt.savefig(table_path, dpi=200, bbox_inches='tight')
print(f"✅ Statistics table saved to: {table_path}")
plt.close()

print("")
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
for i, pool_size in enumerate(pool_sizes):
    print(f"\nPool Size: {pool_size}")
    print(f"  Model:  Mean = \${model_means[i]:>10,.0f},  Std = \${model_stds[i]:>10,.0f},  Return = {model_returns[i]:>+7.2f}%")
    print(f"  Random: Mean = \${random_means[i]:>10,.0f},  Std = \${random_stds[i]:>10,.0f},  Return = {random_returns[i]:>+7.2f}%")
    print(f"  Improvement: {model_returns[i] - random_returns[i]:+.2f}%")

EOF

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     ANALYSIS COMPLETE!                         ║${NC}"
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo ""
echo -e "${BLUE}Results saved to: ${OUTPUT_BASE}/${NC}"
echo -e "${BLUE}Summary graph: ${OUTPUT_BASE}/pool_size_comparison_summary.png${NC}"
echo -e "${BLUE}Statistics table: ${OUTPUT_BASE}/pool_size_statistics_table.png${NC}"
echo ""
