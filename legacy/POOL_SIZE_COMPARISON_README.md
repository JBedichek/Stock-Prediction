# Pool Size Comparison Analysis

This script compares the statistical performance of the stock prediction model across different pool sizes (number of stocks to select from).

## What It Does

The `compare_pool_size_stats.sh` script:

1. **Runs Statistical Comparison** for multiple pool sizes:
   - 50 stocks
   - 100 stocks
   - 256 stocks
   - 512 stocks
   - 1024 stocks
   - 1500 stocks

2. **For Each Pool Size**:
   - Runs 20 independent trials
   - Each trial compares model vs 25 random portfolios
   - Selects top 5 stocks from the pool
   - Tests over 3 months
   - Saves results and visualizations to `output/pool_size_comparison/pool_<size>/`

3. **Generates Summary Visualizations**:
   - **Summary Graph**: Shows mean ± std of final portfolio values across all pool sizes
   - **Return Comparison**: Total return percentage for each pool size
   - **Statistics Table**: Detailed breakdown of all metrics

## Usage

```bash
# Run the full comparison (takes significant time)
./compare_pool_size_stats.sh
```

## Output Structure

```
output/pool_size_comparison/
├── pool_50/
│   ├── results.pt
│   ├── comparison_summary.png
│   ├── distributions_detailed.png
│   └── summary_scorecard.png
├── pool_100/
│   └── ...
├── pool_256/
│   └── ...
├── pool_512/
│   └── ...
├── pool_1024/
│   └── ...
├── pool_1500/
│   └── ...
├── pool_size_comparison_summary.png  # Main summary graph
└── pool_size_statistics_table.png    # Detailed statistics
```

## Configuration

You can edit the script to modify:

- **Pool sizes**: Change the `POOL_SIZES` array
- **Number of trials**: Modify `NUM_TRIALS` (default: 20)
- **Random portfolios per trial**: Change `NUM_RANDOM_PER_TRIAL` (default: 25)
- **Top-K selection**: Adjust `TOP_K` (default: 5)
- **Test period**: Modify `TEST_MONTHS` (default: 3)
- **Initial capital**: Change `INITIAL_CAPITAL` (default: $100,000)

## Key Metrics Compared

For each pool size, the analysis compares:

1. **Final Portfolio Value**: Mean and standard deviation
2. **Total Return**: Percentage gain/loss
3. **Model vs Random**: How much better (or worse) the model performs
4. **Consistency**: Standard deviation shows variability across trials

## Expected Runtime

With the default settings:
- Each pool size: ~5-15 minutes (depends on GPU)
- Total for 6 pool sizes: ~30-90 minutes

## Interpretation

The summary graphs help answer:

1. **Does pool size matter?**
   - Do larger pools give better returns?
   - Or is a smaller, focused pool more effective?

2. **What's the optimal pool size?**
   - Balance between diversity and quality

3. **How consistent are results?**
   - Look at standard deviations across pool sizes

4. **Model effectiveness across scales**
   - Does the model perform better with more or fewer choices?

## Tips

- **Initial Run**: Start with fewer trials (e.g., `NUM_TRIALS=5`) to test quickly
- **Pool Sizes**: Adjust based on your total stock universe size
- **Parallel Execution**: Results are independent - you could run pool sizes in parallel on different machines
- **Resource Usage**: Larger pool sizes require more memory and computation time

## Example Output

The summary graph will show:
- Bar charts comparing model vs random for each pool size
- Error bars showing variability (standard deviation)
- Clear visualization of which pool size yields best results
- Percentage improvement over random selection

## Dependencies

- Python 3.x
- PyTorch
- matplotlib
- numpy
- All dependencies from `requirements.txt`
