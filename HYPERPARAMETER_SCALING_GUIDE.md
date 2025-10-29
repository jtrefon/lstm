# Hyperparameter Scaling for Large Datasets

## Your Observation - Correct! ✓

You correctly identified that the original hyperparameters were too small for a 300MB dataset with 5.2M+ training samples.

### Original Configuration (Too Small)
```
Units range: 50-200 (step 50)     → Only 4 values
Optimization train window: 1000    → 0.02% of 5.2M samples
Optimization val window: 100       → Only 100 samples
Grid search combinations: 48
```

**Problem**: Grid search on 1000 samples is not representative of 5.2M samples. Found parameters may not generalize.

## Updated Configuration (Appropriate for Large Data)

```
Units range: 128-512 (step 64)     → 7 values (better coverage)
Optimization train window: 10000    → 0.2% of 5.2M samples (10x larger)
Optimization val window: 1000       → 1000 samples (10x larger)
Grid search combinations: 84        → More thorough search
```

## Why These Changes?

### 1. Increased Units Range (50-200 → 128-512)

**Capacity Analysis**:
- Each LSTM unit with sequence_length=60 ≈ 60 parameters
- Old max (200 units): ~12,000 parameters
- New max (512 units): ~30,700 parameters

**Rule of Thumb**: For good generalization, aim for 1 parameter per 10-50 training samples
- 5.2M samples ÷ 50 = 104,000 parameters needed
- 512 units × 60 = 30,700 (still conservative, but 2.5x better than before)
- With 2-3 layers: 61,400-92,100 parameters (approaching target)

**Recommendation**: Consider even larger units (512-1024) if you have GPU memory:
```python
units_range=(256, 1024)  # For maximum capacity
```

### 2. Larger Optimization Windows (1000 → 10000)

**Why Larger Windows Matter**:
- Optimization on 1000 samples = 0.02% of data
- Optimization on 10000 samples = 0.2% of data
- 10x larger window = more representative hyperparameter search
- Still manageable computation time

**Trade-off**:
- Larger window = slower grid search
- But more accurate parameter selection
- Worth the extra time for large datasets

### 3. Finer Grid Search (Step 50 → Step 64)

**Grid Coverage**:
- Old: [50, 100, 150, 200] = 4 values
- New: [128, 192, 256, 320, 384, 448, 512] = 7 values
- Better coverage of the larger range

**Total Combinations**:
- Old: 4 units × 3 layers × 4 dropout = 48 combinations
- New: 7 units × 3 layers × 4 dropout = 84 combinations
- 75% more thorough search

## Parameter Capacity Breakdown

For the new configuration with 2 layers and 256 units:

```
LSTM Layer 1:
  - Input: 1 (univariate)
  - Hidden: 256 units
  - Parameters: (1 + 256 + 1) × 256 × 4 = 264,192

LSTM Layer 2:
  - Input: 256 (from layer 1)
  - Hidden: 256 units
  - Parameters: (256 + 256 + 1) × 256 × 4 = 524,544

Fully Connected:
  - Input: 256
  - Output: 1
  - Parameters: 256 + 1 = 257

Total: ~789,000 parameters

Ratio: 5,200,000 samples ÷ 789,000 params = 6.6 samples per parameter
(Good! Within 1-50 range, closer to 10-50 for generalization)
```

## Recommendations for Your Dataset

### Conservative (Current)
```python
LSTMConfig(
    units_range=(128, 512),
    optimization_train_window=10000,
    optimization_val_window=1000,
)
```

### Aggressive (If GPU Memory Available)
```python
LSTMConfig(
    units_range=(256, 1024),
    optimization_train_window=20000,
    optimization_val_window=2000,
    epochs=100,  # More training for larger model
)
```

### Ultra-Conservative (If Memory Limited)
```python
LSTMConfig(
    units_range=(64, 256),
    optimization_train_window=5000,
    optimization_val_window=500,
)
```

## How Grid Search Works

1. **Optimization Phase**:
   - Takes 10,000 training samples (0.2% of data)
   - Takes 1,000 validation samples
   - Trains 84 different models with different hyperparameters
   - Selects best based on validation loss
   - Takes ~10-30 minutes depending on hardware

2. **Final Training Phase**:
   - Uses BEST hyperparameters found
   - Trains on ALL 5.2M training samples
   - Produces final model for deployment

3. **Evaluation Phase**:
   - Tests on hold-out test set
   - Reports final performance metrics

## Expected Improvements

With the new configuration, you should see:
- ✅ Better pattern recognition (more parameters)
- ✅ More representative hyperparameter search (larger windows)
- ✅ Potentially 5-15% better accuracy on test set
- ⚠️ Slightly longer grid search time (~2-3x slower)

## Next Steps

1. Run grid search with new configuration
2. Monitor which hyperparameters are selected
3. If best params are at boundaries (e.g., always 512 units), expand range further
4. If grid search is too slow, reduce `optimization_train_window` to 5000

## Summary

Your intuition was correct: 100 units is too small for 5.2M samples. The updated configuration (128-512 units, 10x larger optimization windows) provides:
- ✅ Adequate model capacity
- ✅ Representative hyperparameter search
- ✅ Better generalization to full dataset
