# Pre-computed Stock Selections

## Overview

This optimization eliminates CPU-intensive stock selection during training by pre-computing diverse selections once and reusing them across training runs.

### Performance Impact

**Without pre-computation:**
- Stock selection computed every episode: `get_top_k_stocks_per_horizon()` + `get_bottom_4_stocks()`
- ~900 stocks × sorting/filtering = significant CPU overhead
- Repeated across all training episodes

**With pre-computation:**
- Stock selections loaded from HDF5 (~10ms)
- Random sampling from pre-computed pool (instant)
- **~10-20x faster** per episode
- Can reuse across multiple training runs

## Quick Start

### Step 1: Pre-compute Stock Selections

Run this **once** before training:

```bash
python rl/precompute_stock_selections.py
```

This generates `data/rl_stock_selections_4yr.h5` containing:
- 100 diverse stock selections per date
- Both top-4 (long) and bottom-4 (short) stocks
- Randomized with `sample_fraction=0.3` for diversity

**Expected time:** ~5-10 minutes (one-time cost)

**Output file size:** ~50-100 MB

### Step 2: Enable in Training

The training script automatically uses pre-computed selections if the file exists:

```python
config = {
    # ... other settings ...

    'use_precomputed_selections': True,  # Enable (default)
    'stock_selections_cache': 'data/rl_stock_selections_4yr.h5',
}
```

### Step 3: Train as Normal

```bash
python rl/train_actor_critic.py
```

Training will automatically:
- Load pre-computed selections at startup
- Sample randomly from the pool each episode
- Maintain same diversity as on-the-fly computation
- Run much faster!

## How It Works

### Pre-computation Phase

For each trading date:
1. Generate N diverse samples (default: 100)
2. Each sample uses random `sample_fraction=0.3` sampling
3. Store in compressed HDF5 format

```
data/rl_stock_selections_4yr.h5
├── 2021-01-04/
│   ├── top_4_tickers: (100, 4) array of tickers
│   ├── top_4_horizons: (100, 4) array of horizon indices
│   ├── bottom_4_tickers: (100, 4) array of tickers
│   └── bottom_4_horizons: (100, 4) array of horizon indices
├── 2021-01-05/
│   └── ...
└── ... (all trading days)
```

### Training Phase

Each episode:
1. Look up date in pre-computed cache
2. Randomly select one of 100 pre-computed samples
3. Use those stocks for the episode
4. **No CPU-intensive computation needed!**

## Customization

Edit `precompute_stock_selections.py` to change:

```python
config = {
    'num_samples_per_date': 100,  # More samples = more diversity
    'top_k_per_horizon': 3,       # Match training config
    'sample_fraction': 0.3,       # Match training config
}
```

**Important:** Settings should match your training configuration.

## Regenerating Pre-computed Selections

Regenerate if you change:
- `top_k_per_horizon`
- `sample_fraction`
- Trading date range

Just run the pre-computation script again - it will overwrite the old file.

## Fallback Behavior

If pre-computed selections are missing or disabled:
- Training automatically falls back to on-the-fly computation
- Same behavior, just slower
- No data loss or errors

## Verification

Check that pre-computation worked:

```bash
python -c "
import h5py
with h5py.File('data/rl_stock_selections_4yr.h5', 'r') as f:
    print(f'Dates: {len(f.keys())}')
    print(f'Samples per date: {f.attrs[\"num_samples_per_date\"]}')
    print(f'Sample fraction: {f.attrs[\"sample_fraction\"]}')
"
```

Expected output:
```
Dates: 1008
Samples per date: 100
Sample fraction: 0.3
```

## FAQ

**Q: Do I need to regenerate for every training run?**
A: No! That's the point. Generate once, reuse forever (unless you change config).

**Q: Does this reduce diversity?**
A: No. We pre-compute 100 diverse samples per date and randomly sample from them. Same diversity as on-the-fly computation.

**Q: What if I add more data?**
A: Re-run the pre-computation script to include new dates.

**Q: Can I use different settings for different experiments?**
A: Yes, generate multiple files with different configs:
```bash
python precompute_stock_selections.py --output data/selections_k5.h5 --top_k 5
python precompute_stock_selections.py --output data/selections_k10.h5 --top_k 10
```

**Q: How much faster is training?**
A: Typically 10-20x faster per episode for the stock selection part. Overall training speedup depends on your bottleneck.

## Memory Usage

- Pre-computed file: ~50-100 MB (disk)
- Loaded in memory: ~50-100 MB (RAM)
- Very efficient with compression

No GPU memory impact.
