# Plan: Flexible Multi-Feature Stock Prediction Architecture

## Executive Summary

**Goal**: Refactor Stock-Predictor to support:
- Multiple timescales (hourly/4-hour over 5-10 years)
- Multiple feature types (fundamentals, news, analyst, financial statements)
- Flexible architecture that makes adding features easy

**Key Challenge**: Current 600 daily points → ~50,000 hourly points (impossible for standard transformers)

**Solution**: Hierarchical temporal encoding + parametric architecture + modular feature pipeline

---

## Current State Analysis

### What You're Currently Using
- **Price Data**: 600 days × 5 OHLCV features (Open, High, Low, Close, Volume)
- **Company Metadata**: RoBERTa embeddings of sector + industry + business summary (768 dims → padded to 988)
- **News**: Headlines are scraped but NOT integrated into training pipeline
- **Market Stats**: PCA-based market-wide statistics (10 dims)
- **Feature Engineering**: Gaussian normalization, cube normalization, relative price changes, sector/industry one-hot encoding
- **Final Input Shape**: (batch, 354, 218) = 350 price tokens + 4 summary tokens

### Key Architectural Constraints
- **Summary embedding**: Hardcoded to 988 dimensions with forced reshape to 4×218 tokens
- **Linear head**: Hardcoded input size of 604×2 = 1208 dimensions
- **Limited flexibility**: Many magic numbers (988, 220, 872, 604) that cascade through architecture
- **Moderate-High difficulty** to add features due to tight coupling

---

## Available Financial Data Sources (Not Currently Used)

### From yfinance (FREE - Already Available)

#### 1. Fundamental Financial Metrics (HIGH IMPACT)
**Valuation Ratios:**
- P/E ratio (trailing & forward)
- Price-to-Book, Price-to-Sales
- PEG ratio
- Enterprise Value / Revenue, EV/EBITDA

**Profitability Metrics:**
- Profit margins (gross, operating, net)
- Return on Equity (ROE), Return on Assets (ROA)
- EBITDA margins

**Financial Health:**
- Debt-to-Equity ratio
- Current ratio, Quick ratio
- Free cash flow, Operating cash flow
- Total cash, Total debt
- Book value per share

**Growth Indicators:**
- Revenue growth rate
- Earnings growth rate
- EPS (Earnings Per Share) trend

**Example extraction:**
```python
ticker = yf.Ticker("AAPL")
info = ticker.info
pe_ratio = info.get('trailingPE')
roe = info.get('returnOnEquity')
debt_equity = info.get('debtToEquity')
revenue_growth = info.get('revenueGrowth')
```

#### 2. Analyst Consensus Data (MEDIUM-HIGH IMPACT)
- Analyst recommendations (Strong Buy to Sell scale)
- Price targets (mean, median, high, low)
- Number of analysts covering
- Earnings estimates (EPS, revenue)
- Estimate revisions trend

**Example:**
```python
recommendations = ticker.recommendations
price_targets = ticker.analyst_price_targets
earnings_est = ticker.earnings_estimate
```

#### 3. Insider Trading & Holdings (MEDIUM IMPACT)
- Net insider buys vs sells
- Insider ownership percentage
- Recent insider transaction activity
- Institutional ownership
- Major shareholders

**Example:**
```python
insider_purchases = ticker.insider_purchases
institutional = ticker.institutional_holders
```

#### 4. Market Sensitivity & Risk (MEDIUM IMPACT)
- Beta coefficient (volatility vs market)
- 52-week high/low distances
- Average volume trends
- Short interest ratio

#### 5. Financial Statements (COMPLEX but POWERFUL)
- Income statement (quarterly & annual)
- Balance sheet (quarterly & annual)
- Cash flow statement (quarterly & annual)
- Trailing Twelve Months (TTM) data

**Example:**
```python
income_stmt = ticker.income_stmt
balance_sheet = ticker.quarterly_balance_sheet
cash_flow = ticker.ttm_cash_flow
```

#### 6. Corporate Actions & Distributions
- Dividend history and yield
- Stock split history
- Earnings dates
- Calendar events

---

## Recommended Implementation Approach

### Phase 1: Add Fundamental Metrics (EASIEST + HIGH IMPACT)

**Target features (~20-30 dimensions):**
1. Valuation: P/E, P/B, P/S, PEG
2. Profitability: ROE, ROA, profit margin, gross margin
3. Growth: Revenue growth, earnings growth, EPS trend
4. Financial health: Debt/Equity, current ratio, free cash flow
5. Market: Beta, 52-week position, relative strength

**Implementation:**
- Extract from `ticker.info` dictionary
- Add to summary embedding (currently 768 dims)
- Handle missing data with reasonable defaults
- Normalize/standardize metrics

**Advantages:**
- Relatively simple to extract
- Stable data (changes quarterly, not daily)
- Can be cached and updated infrequently
- Proven predictive value in finance

**Challenges:**
- Requires updating summary embedding size (currently 988)
- Need to handle missing data (small-cap stocks)
- Some metrics are industry-specific

### Phase 2: Integrate News Embeddings (MEDIUM DIFFICULTY)

**Current state:** News is scraped but not used in training

**Proposed approach:**
1. Embed news headlines using existing RoBERTa encoder
2. Aggregate daily news embeddings (mean/max pooling)
3. Add as additional sequence features or summary component

**Advantages:**
- Already have news scraping infrastructure
- News has strong predictive signal for short-term movements
- Can capture sentiment and events

**Challenges:**
- News availability varies by company size
- Need to handle days with no news
- Embeddings are high-dimensional (768 dims per article)

### Phase 3: Add Analyst & Earnings Data (MEDIUM DIFFICULTY)

**Target features:**
- Analyst recommendation score (1-5 scale)
- Price target vs current price (%)
- Number of analysts covering
- EPS estimate vs actual (surprise metric)
- Earnings date proximity

**Advantages:**
- Strong signal for institutional sentiment
- Captures forward-looking expectations

**Challenges:**
- Not available for all stocks (small-caps)
- Requires careful handling of temporal alignment

### Phase 4: Financial Statement Ratios (COMPLEX)

**Derived metrics from statements:**
- Debt service coverage ratio
- Asset turnover
- Inventory turnover
- Operating leverage
- Working capital trends

**Advantages:**
- Deep fundamental analysis
- Captures business model differences

**Challenges:**
- Quarterly updates only
- Complex feature engineering
- Many stocks lack complete data

---

## Alternative Data Sources (Beyond yfinance)

### If you need more than yfinance provides:

1. **Finnhub** (Free tier: 60 calls/min)
   - Sentiment scores on news
   - Social media mentions
   - Insider sentiment

2. **Alpha Vantage** (Free: 5 calls/min)
   - Technical indicators pre-computed
   - Forex and commodities
   - Economic indicators

3. **Polygon.io** (Limited free tier)
   - Options flow data
   - Real-time aggregates

4. **Reddit/Twitter Sentiment**
   - WallStreetBets mentions
   - Social media sentiment scores
   - Retail investor interest

---

## Architectural Modifications Required

### Option A: Minimal Changes (Keep Current Architecture)

**Increase summary embedding size:**
- Current: 768 (BERT) + 220 padding = 988 → reshape to (4, 218)
- New: 768 (BERT) + X (fundamentals) + padding = Y → reshape to (N, 218)

**Steps:**
1. Add fundamental features to summary (increase from 768 to ~800-850)
2. Recalculate padding to reach next multiple of 218×K
3. Update reshape dimensions in models.py (lines 256-258)
4. Update linear head input size (line 212)

**Pros:** Minimal code changes
**Cons:** Still has hardcoded constraints, not flexible

### Option B: Parametric Architecture (Recommended)

**Make dimensions configurable:**
```python
class t_Dist_Pred:
    def __init__(self, seq_len=350, data_dim=218, summary_dim=None, ...):
        self.summary_dim = summary_dim or 988
        self.num_summary_tokens = self.summary_dim // data_dim
        # Compute linear input dynamically
        self.linear_input_dim = (seq_len + self.num_summary_tokens) * 2
```

**Pros:** Flexible, maintainable, can add features easily
**Cons:** Requires more extensive refactoring

### Option C: Feature-Specific Embeddings (Advanced)

**Separate embedding layers for different feature types:**
- Price sequence → Transformer
- Company summary → Projection layer
- Fundamentals → Separate MLP
- News → Separate embedding layer
- Combine with cross-attention or concatenation

**Pros:** Maximum flexibility, can add any feature type
**Cons:** Significant architectural redesign

---

## Critical Files to Modify

1. **Stock.py** (data scraping)
   - Add methods to extract fundamental metrics
   - Add methods to fetch analyst data
   - Implement caching for quarterly data

2. **training.py**
   - Modify `GenerateDataDict` to include new features
   - Update `QTrainingData` to handle additional dimensions
   - Adjust normalization for new feature types

3. **models.py**
   - Update `t_Dist_Pred.__init__()` summary size handling
   - Modify forward() reshape logic (lines 256-258)
   - Update linear head input dimension (line 212)

4. **utils.py**
   - Add normalization functions for fundamental ratios
   - Add feature extraction utilities
   - Handle missing data imputation

---

## Recommended Priority Order

### Immediate (Week 1-2):
1. **Add 20-30 fundamental metrics from ticker.info**
   - Start with P/E, P/B, ROE, debt ratios, beta
   - Easy to extract, high signal value
   - Minimal architectural changes

2. **Integrate existing news embeddings**
   - You're already scraping news
   - Low-hanging fruit for improvement

### Short-term (Week 3-4):
3. **Add analyst consensus data**
   - Recommendations, price targets
   - Strong forward-looking signal

4. **Parametrize architecture for flexibility**
   - Remove hardcoded dimensions
   - Make it easier to iterate

### Medium-term (Month 2):
5. **Add financial statement ratios**
   - More complex but powerful
   - Requires careful feature engineering

6. **Explore alternative data sources**
   - Sentiment, social mentions
   - Options flow, insider trading

---

## User Requirements (Confirmed)

✅ **Goal**: Flexible system - refactor to make adding features easy
✅ **Data types**: ALL - fundamentals, news, analyst data, financial statements
✅ **Missing data**: Impute with zeros or industry averages
✅ **Time scale**: Both longer period (5-10 years) AND higher resolution (hourly/4-hour)

---

## Core Architecture Design

### 1. Hierarchical Multi-Scale Temporal Architecture (HMSTA)

**Problem**: 87,600 hourly points over 10 years breaks transformers (memory O(N²))

**Solution**: Three-tier hierarchical encoding
```
Hourly data (50K points) → Local encoder → 64-dim compression
Daily data (3.6K points)  → Local encoder → 64-dim compression
Weekly data (520 points)  → Local encoder → 64-dim compression
                                ↓
                    Fusion Transformer (192 dims from temporal)
                                ↓
                    Add fundamentals, news, analyst, statements
                                ↓
                         Final predictions
```

**Memory Savings**:
- Standard: 50K × 50K attention = 10GB per layer (impossible)
- Hierarchical: 3 × (1K × 1K) = 12MB per layer (feasible!)

### 2. Flexible Model Architecture

**Remove ALL hardcoded dimensions**:

Current hardcoded constraints:
- Line 212 (models.py): `nn.Linear(604*2, ...)` - assumes 600 + 4 tokens
- Line 256-258 (models.py): `s[:,:218*4]` - hardcoded 872-dim summary
- Line 367 (training.py): `torch.ones((988-768))` - hardcoded padding

**New parametric approach**:
```python
class FlexibleDistPredictor(nn.Module):
    def __init__(
        self,
        price_feature_dim: int,      # e.g., 12
        fundamental_dim: int,         # e.g., 30
        news_dim: int,                # e.g., 768
        analyst_dim: int,             # e.g., 10
        statement_dim: int,           # e.g., 50
        summary_dim: int,             # e.g., 988
        hidden_dim: int = 512,
        # ... all dimensions configurable
    ):
        # Feature-specific embedding layers
        self.price_embedding = nn.Linear(price_feature_dim, hidden_dim)
        self.fundamental_embedding = nn.Linear(fundamental_dim, hidden_dim)
        self.news_embedding = nn.Linear(news_dim, hidden_dim)
        # ... separate embeddings for each feature type

        # NO HARDCODING - everything computed dynamically
```

### 3. Feature Integration Pipeline

**New Features** (~860 total, vs current ~200):

| Feature Type | Dimensions | Update Frequency | Imputation Strategy |
|--------------|------------|------------------|---------------------|
| Price (OHLCV + engineered) | 12 | Hourly/Daily | None (always present) |
| Fundamentals | 30 | Quarterly | Forward-fill / Industry avg |
| News embeddings | 768 | Daily (sporadic) | Zero-fill |
| Analyst data | 10 | Weekly | Forward-fill |
| Financial statements | 50 | Quarterly | Forward-fill |
| Company summary | 988 | Static | N/A |

**Data Alignment**:
- Align all features to common timeline (daily)
- Forward-fill quarterly/weekly data to daily
- Zero-fill sporadic data (news)
- Create mask tensor indicating which features are present

### 4. Data Collection Refactor

**Critical Issue**: yfinance only provides 730 days of intraday data

**Solutions**:
1. **Hybrid approach** (recommended): Daily for 10 years + hourly for recent 2 years
2. **Alternative APIs**: Polygon.io ($200/month) for full historical intraday
3. **Focus on daily first**: Start with daily data, add hourly when budget allows

**New FlexibleStockDataCollector** (stock.py - currently MISSING):
```python
class FlexibleStockDataCollector:
    def collect_all_data(self, ticker):
        # Price data (hourly + daily)
        hourly_data = self._get_intraday(ticker, '1h', period='730d')
        daily_data = ticker.history(start='2015-01-01', interval='1d')

        # Fundamentals (~30 features)
        fundamentals = self._extract_fundamentals(ticker.info)
        # P/E, ROE, ROA, margins, debt ratios, growth rates

        # Financial statements (~50 features)
        statements = self._extract_ratios(
            ticker.quarterly_income_stmt,
            ticker.quarterly_balance_sheet,
            ticker.quarterly_cashflow
        )

        # Analyst data
        analyst = ticker.recommendations
        price_targets = ticker.analyst_price_targets

        # News embeddings
        news = self._collect_and_embed_news(ticker)

        # Company summary
        summary = self._get_summary_embedding(ticker.info)

        return {all features aligned and cached}
```

---

## Implementation Roadmap

### Phase 0: Infrastructure (Week 1)
**Goal**: Set up configuration system and base classes

- [ ] Create `/configs/` directory with YAML/JSON configs
- [ ] Define DataConfig, FeatureConfig, SequenceConfig classes
- [ ] Set up HDF5/Parquet data storage structure
- [ ] Create base FeaturePipeline framework

**Deliverables**:
- Configuration templates
- Data storage format defined
- Base classes (no feature-specific logic yet)

### Phase 1: Data Collection (Weeks 2-3)
**Goal**: Collect all feature types for subset of stocks

**Priority 1: Fundamentals** (easiest, high impact)
- [x] Implement fundamental extraction from ticker.info
- [x] Extract ~30 metrics: P/E, ROE, margins, debt ratios, etc.
- [ ] Add industry-average imputation for missing values
- [ ] Test with 100 tickers, validate data quality

**Priority 2: Financial Statements**
- [x] Extract quarterly income, balance sheet, cash flow
- [x] Calculate ~50 derived ratios
- [x] Implement quarterly→daily forward-filling

**Priority 3: News Integration**
- [ ] Find existing news scraper (referenced but not in repo)
- [ ] Implement RoBERTa embedding pipeline
- [ ] Add daily aggregation (mean pooling)
- [ ] Handle missing news days (zero vectors)

**Priority 4: Analyst Data**
- [ ] Extract recommendations, ratings, price targets
- [ ] Implement weekly→daily alignment
- [ ] Handle missing analyst coverage

**Deliverables**:
- New `/stock.py` (refactored data collector)
- Sample dataset: 100 tickers × 2 years with all features
- Data validation report

### Phase 2: Feature Processing (Weeks 4-5)
**Goal**: Build modular feature processing pipeline

- [ ] Create `feature_processors/` module
- [ ] Implement PriceFeatureProcessor (technical indicators)
- [ ] Implement FundamentalProcessor (imputation + normalization)
- [ ] Implement NewsProcessor (embedding aggregation)
- [ ] Implement AnalystProcessor (alignment)
- [ ] Implement StatementProcessor (quarterly→daily)
- [ ] Create FeatureAligner class (align all to common timeline)
- [ ] Refactor QTrainingData to FlexibleQTrainingData

**Deliverables**:
- Modular feature processors
- Updated training.py with flexible dataset class
- Test dataset with aligned multi-frequency features

### Phase 3: Model Architecture (Weeks 6-7)
**Goal**: Implement flexible, hierarchical model

**Priority 1: Remove Hardcoding**
- [ ] Parametrize all dimensions in models.py
- [ ] Remove lines 212, 256-258 hardcoding
- [ ] Create FlexibleDistPredictor base class
- [ ] Add feature-specific embedding layers

**Priority 2: Hierarchical Encoding**
- [ ] Implement LocalTemporalEncoder (1K sequence → 64 dims)
- [ ] Implement HierarchicalTemporalEncoder (multi-scale)
- [ ] Add fusion transformer layer
- [ ] Test compression with synthetic data

**Priority 3: Efficient Attention**
- [ ] Integrate Flash Attention 2 (2-4x speedup)
- [ ] Add gradient checkpointing (2-3x memory reduction)
- [ ] Implement mixed precision training (FP16/BF16)

**Deliverables**:
- New `models_v2.py` (flexible architecture)
- `hierarchical_encoding.py` module
- Model tests with varying sequence lengths (100, 1K, 10K)
- Memory profiling report

### Phase 4: Training Infrastructure (Weeks 8-9)
**Goal**: Update training loop for new architecture

- [ ] Modify Train_Dist_Direct_Predictor for new model
- [ ] Update loss function for new dimensions
- [ ] Add validation for all feature types
- [ ] Implement gradient accumulation for large batches
- [ ] Add WandB logging for new features
- [ ] Add feature importance tracking

**Deliverables**:
- Updated training loop
- Training configs for different setups
- Training run scripts

### Phase 5: Testing & Migration (Weeks 10-11)
**Goal**: Validate new system and migrate from old

- [ ] Create data migration script (old → new format)
- [ ] Test with 600-day daily setup (verify no regression)
- [ ] Test with 1000-day daily + fundamentals
- [ ] Test with 1000-day daily + all features
- [ ] Benchmark: training time, memory, inference latency
- [ ] Compare prediction accuracy on holdout set

**Deliverables**:
- Migration scripts
- Comprehensive test suite
- Performance benchmark report

### Phase 6: Production Deployment (Week 12+)
**Goal**: Deploy to production with monitoring

- [ ] Update stock_inference class for new model
- [ ] Modify trading simulation for new features
- [ ] Add data quality monitoring
- [ ] Add feature drift detection
- [ ] Update documentation

**Deliverables**:
- Updated inference.py
- Production monitoring dashboard
- Comprehensive documentation

---

## Critical Implementation Details

### Memory Management

**Challenge**: Even with hierarchical encoding, 10 years of hourly data is massive

**Memory Budget** (single A100 24GB):
```
Model parameters: ~200M × 2 bytes (FP16) = 400 MB
Optimizer states (AdamW): 400 MB × 3 = 1.2 GB
Gradients: 400 MB
Activations (with checkpointing): ~3 MB per layer × 24 = 72 MB
Batch data: Variable (depends on batch size)

Total: ~2 GB (leaves 22GB for data batches)
```

**Optimizations**:
1. Gradient checkpointing (2-3x memory reduction)
2. Mixed precision FP16/BF16 (2x memory reduction)
3. Flash Attention 2 (constant memory attention)
4. Batch size 1 with gradient accumulation
5. HDF5 loading (load on demand, not all in memory)

### Data Source Strategy

**yfinance Limitation**: Only 730 days of intraday data

**Phase 1 Approach** (FREE):
- Daily data: 10 years (2015-2025)
- Hourly data: Last 2 years from yfinance
- Focus on daily first, add hourly later

**Phase 2 Approach** (if budget allows):
- Polygon.io: $200/month for full historical intraday
- Or use Alpha Vantage free tier (limited)
- Or synthesize intraday from daily OHLC (lower quality)

### Imputation Strategies

**By feature type**:
```python
imputation_config = {
    'fundamentals': 'industry_average',  # Use sector avg P/E, ROE, etc.
    'statements': 'forward_fill',         # Quarterly → forward fill to daily
    'analyst': 'forward_fill',            # Weekly → forward fill to daily
    'news': 'zero',                       # No news = zero vector
    'price': 'none',                      # Should always exist
}
```

**Missing data masks**:
```python
feature_masks = {
    'has_fundamentals': bool,
    'has_analyst_coverage': bool,
    'has_news': bool,
    'has_complete_statements': bool,
}
```

Feed masks to model so it learns to handle incomplete data.

---

## Success Metrics

### Data Quality
- [ ] 80%+ stocks have fundamentals
- [ ] 50%+ stocks have news coverage
- [ ] 30%+ stocks have analyst coverage
- [ ] 100% stocks have 2-year hourly data

### Model Performance
- [ ] Validation loss ≤ current best
- [ ] Sharpe ratio ≥ 1.5 in backtest
- [ ] Win rate ≥ 55%
- [ ] Max drawdown ≤ 20%

### System Performance
- [ ] Training time ≤ 1 week per epoch
- [ ] Inference latency ≤ 2 min for 4000 stocks
- [ ] Memory usage ≤ 20 GB during training

### Code Quality
- [ ] All magic numbers removed
- [ ] Test coverage ≥ 70%
- [ ] Documentation complete

---

## Critical Files to Modify

### Priority 1: Core Architecture
1. **/home/james/Desktop/Stock-Prediction/models.py** (HIGH PRIORITY)
   - Remove hardcoded dimensions (lines 212, 256-258, 367)
   - Implement FlexibleDistPredictor class
   - Add feature-specific embedding layers
   - ~800 lines added, ~200 modified

2. **/home/james/Desktop/Stock-Prediction/training.py** (HIGH PRIORITY)
   - Refactor QTrainingData to FlexibleQTrainingData
   - Remove hardcoded 600 sequence length, 988 summary dims
   - Add support for multi-frequency feature alignment
   - ~1000 lines added/modified

3. **/home/james/Desktop/Stock-Prediction/stock.py** (CURRENTLY MISSING - HIGH PRIORITY)
   - Implement FlexibleStockDataCollector
   - Add fundamental extraction (~30 features)
   - Add financial statement ratio calculation (~50 features)
   - Add news scraping + RoBERTa embedding
   - Add analyst data extraction
   - ~1200 lines (new file)

### Priority 2: New Modules
4. **/home/james/Desktop/Stock-Prediction/configs/** (NEW - MEDIUM-HIGH PRIORITY)
   - DataConfig, FeatureConfig, SequenceConfig, ImputationConfig
   - YAML/JSON configuration templates
   - ~500 lines across multiple files

5. **/home/james/Desktop/Stock-Prediction/feature_processors/** (NEW - MEDIUM PRIORITY)
   - PriceFeatureProcessor, FundamentalProcessor, NewsProcessor
   - AnalystProcessor, StatementProcessor, FeatureAligner
   - ~800 lines across multiple files

6. **/home/james/Desktop/Stock-Prediction/hierarchical_encoding.py** (NEW - MEDIUM PRIORITY)
   - LocalTemporalEncoder, HierarchicalTemporalEncoder
   - Multi-scale temporal compression
   - ~600 lines

### Priority 3: Supporting Files
7. **/home/james/Desktop/Stock-Prediction/inference.py**
   - Update stock_inference class for new model
   - ~400 lines modified

8. **/home/james/Desktop/Stock-Prediction/utils.py**
   - Add normalization for new features
   - Add imputation utilities
   - ~200 lines added

---

## Quick Start Recommendation

**Week 1 - Proof of Concept**:
1. Start with daily data only (skip hourly for now)
2. Add just fundamentals (30 features) - easiest win
3. Parametrize models.py to remove hardcoding
4. Test with 100 tickers, 1 year
5. Validate improvement over baseline

**If successful → proceed with full roadmap**
**If not → reassess feature selection and architecture**

This incremental approach reduces risk and provides fast feedback.

---

## Current Progress Update

### ✅ Completed: Phase 1 - Fundamental Metrics (FMP Integration)

**What was accomplished:**
1. **Time-Varying Fundamentals** - Solved the critical requirement that metrics must change over time
   - Created `generate_fmp_fundamentals.py` - extracts 20+ years of quarterly fundamental data
   - Data structure: `dict[ticker: dict[date: tensor(27)]]` - fundamentals actually vary by date!
   - 27 metrics including margins, ratios, valuations, growth indicators

2. **Training Integration** - Fundamentals automatically included in model
   - Modified `training.py` to look up fundamentals by specific date (time-varying!)
   - Embedded fundamentals into summary at positions 768-794 (maintains 988-dim compatibility)
   - Updated all 3 data preparation functions (prepare_dataset, prepare_dataset_layer, inference_dataset)

3. **Documentation**
   - Created `FMP_SETUP_GUIDE.md` - step-by-step user guide
   - Created `FUNDAMENTAL_INTEGRATION_SUMMARY.md` - complete technical documentation
   - Updated `models.py` with clear documentation of summary structure

**User Action Required:**
- Get FMP API key (free tier: 250 calls/day)
- Collect fundamentals over 5 days (~370 stocks × 3 API calls each = 1,110 total)
- Training pipeline is ready - will automatically use fundamentals once data is collected

**Next Steps (from roadmap):**
- Phase 1 Priority 2: News Integration (news scraper exists but not integrated)
- Phase 1 Priority 4: Analyst Data (recommendations, price targets)
- Phase 2: Feature Processing pipeline (modular processors)
- Phase 3: Remove hardcoded dimensions, parametric architecture
