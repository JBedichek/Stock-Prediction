# Create Inference Dataset Guide

## Overview

The `create_inference_dataset.py` script creates a dataset for inference that includes all stocks over a specified time period (default: past year). It reuses the existing dataset processing logic from `dataset_processing.py`.

## Features

- ✅ **Automatic date range calculation** - Defaults to past 12 months
- ✅ **All stocks included** - Uses all companies in your data by default
- ✅ **Configurable time range** - Adjust `--months-back` to change the period
- ✅ **Short date format** - Outputs files named like `inference_dataset_12-10-23-12-10-24.pt`
- ✅ **Reuses existing logic** - Calls `QTrainingData` with proper inference parameters

## Quick Start

### Basic Usage

```bash
# Create inference dataset for the past year
python -m dataset_creation.create_inference_dataset \
    --bulk-prices unnorm_price_series_prediction_-5y-2024-12-15__.pickle \
    --summaries pred_summary_embs.pickle \
    --sectors prediction_sector_dict \
    --data-dict pred_DataDict_2024-12-15
```

### Custom Time Range

```bash
# Create dataset for the past 6 months
python -m dataset_creation.create_inference_dataset \
    --bulk-prices unnorm_price_series_prediction_-5y-2024-12-15__.pickle \
    --summaries pred_summary_embs.pickle \
    --sectors prediction_sector_dict \
    --data-dict pred_DataDict_2024-12-15 \
    --months-back 6
```

### Subset of Companies

```bash
# Use only 50% of companies (for testing/debugging)
python -m dataset_creation.create_inference_dataset \
    --bulk-prices unnorm_price_series_prediction_-5y-2024-12-15__.pickle \
    --summaries pred_summary_embs.pickle \
    --sectors prediction_sector_dict \
    --data-dict pred_DataDict_2024-12-15 \
    --company-keep-rate 0.5
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--bulk-prices` | Path to bulk prices pickle file | `unnorm_price_series_prediction_-5y-2024-12-15__.pickle` |
| `--summaries` | Path to company summaries pickle file | `pred_summary_embs.pickle` |
| `--sectors` | Path to sectors dictionary | `prediction_sector_dict` |
| `--data-dict` | Path to data dictionary (no .pt extension) | `pred_DataDict_2024-12-15` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seq-len` | 600 | Sequence length for the model |
| `--months-back` | 12 | Number of months to go back from today |
| `--company-keep-rate` | 1.0 | Proportion of companies to keep (0.0-1.0) |
| `--output-dir` | `.` | Directory to save the output file |

## Output

The script creates a file named `inference_dataset_{start_date}-{end_date}.pt` where dates are in `MM-DD-YY` format.

### Example Output Filename

```
inference_dataset_12-10-23-12-10-24.pt
```

This represents a dataset from December 10, 2023 to December 10, 2024.

### Output Format

The output file contains a PyTorch dictionary with the following structure:

```python
{
    date1: {
        'AAPL': [data_tensor, summary_tensor, price],
        'MSFT': [data_tensor, summary_tensor, price],
        ...
    },
    date2: {
        'AAPL': [data_tensor, summary_tensor, price],
        'MSFT': [data_tensor, summary_tensor, price],
        ...
    },
    ...
}
```

## Loading the Dataset

To use the inference dataset in your code:

```python
import torch

# Load the inference dataset
inference_data = torch.load('inference_dataset_12-10-23-12-10-24.pt')

# Access data for a specific date and company
date = list(inference_data.keys())[0]  # Get first date
company = list(inference_data[date].keys())[0]  # Get first company

data_tensor, summary_tensor, price = inference_data[date][company]
print(f"Data shape: {data_tensor.shape}")
print(f"Summary shape: {summary_tensor.shape}")
print(f"Price: {price}")
```

## Example Workflow

### 1. Create Inference Dataset

```bash
python -m dataset_creation.create_inference_dataset \
    --bulk-prices unnorm_price_series_prediction_-5y-2024-12-15__.pickle \
    --summaries pred_summary_embs.pickle \
    --sectors prediction_sector_dict \
    --data-dict pred_DataDict_2024-12-15
```

**Output:**
```
Creating inference dataset from 2023-12-10 to 2024-12-10
  Bulk prices: unnorm_price_series_prediction_-5y-2024-12-15__.pickle
  Summaries: pred_summary_embs.pickle
  Sectors: prediction_sector_dict
  Data dict: pred_DataDict_2024-12-15
  Sequence length: 600
  Company keep rate: 100.0%

Initializing QTrainingData with inference mode...
Loading Summaries...
Loading Sectors...
Loading Fundamentals...
Loading Data Dict...
Generating inference dataset...
100%|██████████| 250/250 [05:23<00:00,  1.29s/it]

Saving inference dataset to: inference_dataset_12-10-23-12-10-24.pt

✅ Dataset created successfully!
  Number of dates: 250
  Number of companies: 3,500
  Output file: inference_dataset_12-10-23-12-10-24.pt
  File size: 2.34 GB
```

### 2. Use in Inference Script

```python
from inference.inference import stock_inference
import datetime

# Initialize inference with the new dataset
model_pths = ['DistPred_m_2_E6']
inf = stock_inference(
    model_pths,
    None, None, None,
    start_date=None,
    end_date=None,
    load_dataset=True,
    n=5,
    custom_dataset='inference_dataset_12-10-23-12-10-24'
)

# Run trading simulation
inf.run_trading_sim(
    n=15,
    date=datetime.date(2024, 1, 15),
    period_len=40,
    low=False,
    entropy=True
)
```

## File Requirements

Before running the script, ensure you have the following files:

1. **Bulk Prices File** - Contains historical price data for all companies
   - Format: `unnorm_price_series_*.pickle`
   - Generated by: `data_scraping/Stock.py`

2. **Summaries File** - Contains BERT embeddings of company descriptions
   - Format: `*_summary_embs.pickle`
   - Generated by: Data scraping scripts

3. **Sectors Dictionary** - Maps companies to their sector/industry
   - Format: `*_sector_dict`
   - Generated by: Data scraping scripts

4. **Data Dictionary** - Preprocessed data dictionary
   - Format: `DataDict_*.pt`
   - Generated by: `GenerateDataDict` class

## Troubleshooting

### Error: "File not found"

**Solution:** Check that all input files exist and paths are correct:
```bash
ls -lh unnorm_price_series_prediction_-5y-2024-12-15__.pickle
ls -lh pred_summary_embs.pickle
ls -lh prediction_sector_dict
ls -lh pred_DataDict_2024-12-15.pt
```

### Error: "Dataset is empty"

**Solution:** This means the date range doesn't overlap with your data. Try:
1. Reducing `--months-back` (e.g., `--months-back 6`)
2. Checking the date range in your bulk prices file
3. Ensuring your DataDict contains recent dates

### Error: "KeyError" or "Missing company"

**Solution:** Some companies may not have data for all dates. This is normal and handled by the script. The output dataset will only contain companies with complete data.

## Performance Tips

1. **Large Datasets** - Creating inference datasets can take 5-30 minutes depending on:
   - Number of dates (controlled by `--months-back`)
   - Number of companies (controlled by `--company-keep-rate`)
   - Your disk I/O speed

2. **Memory Usage** - The script loads the entire DataDict into memory. Ensure you have:
   - At least 8GB RAM for 1 year of data
   - 16GB+ RAM for longer periods

3. **Testing** - For quick testing, use:
   ```bash
   --months-back 1 --company-keep-rate 0.1
   ```

## Differences from Training Dataset

| Feature | Training Dataset | Inference Dataset |
|---------|-----------------|-------------------|
| **Time sampling** | Every k-th date | Every date |
| **Company sampling** | Random subset | All companies |
| **Data structure** | List of tuples | Dict[date][company] |
| **Purpose** | Model training | Backtesting/prediction |
| **Size** | Smaller (sampled) | Larger (complete) |

## Integration with Existing Code

The inference dataset format is compatible with the `stock_inference` class:

```python
# Old way (generating on-the-fly)
dataset = QTrainingData(
    summaries_pth, bulk_prices_pth, 600,
    inference=(start_date, end_date),
    ...
)

# New way (using pre-generated dataset)
inference_data = torch.load('inference_dataset_12-10-23-12-10-24.pt')
```

Both produce the same data structure and can be used interchangeably.

## Advanced Usage

### Generate Multiple Time Periods

```bash
# Q1 2024
python -m dataset_creation.create_inference_dataset \
    --bulk-prices <files> \
    --months-back 3 \
    --output-dir ./datasets/q1_2024

# Q2 2024
python -m dataset_creation.create_inference_dataset \
    --bulk-prices <files> \
    --months-back 6 \
    --output-dir ./datasets/q2_2024
```

### Batch Processing

```bash
#!/bin/bash
# Create datasets for different time periods

for months in 3 6 12 24; do
    echo "Creating dataset for past $months months..."
    python -m dataset_creation.create_inference_dataset \
        --bulk-prices unnorm_price_series_prediction_-5y-2024-12-15__.pickle \
        --summaries pred_summary_embs.pickle \
        --sectors prediction_sector_dict \
        --data-dict pred_DataDict_2024-12-15 \
        --months-back $months \
        --output-dir ./datasets
done
```

## See Also

- `dataset_processing.py` - Core dataset processing logic
- `inference/inference.py` - Inference and backtesting code
- `data_scraping/Stock.py` - Data collection scripts
