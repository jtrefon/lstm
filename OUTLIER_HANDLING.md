# Outlier Handling - Prevent Clipping in Data Normalization

## Problem Statement

When using `MinMaxScaler` for normalization, extreme outliers cause most normal values to be compressed into a tiny range:

```
Data: [1, 2, 3, 4, 5, 1000000]
MinMaxScaler: [0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 1.0]
Result: Normal values clipped to near-zero, losing information
```

This severely impacts model training and prediction accuracy.

## Solution: Configurable Outlier Handling

Added outlier detection and clipping before normalization to prevent extreme values from distorting the scale.

## Implementation

### 1. Outlier Handler Utility (`src/infrastructure/data/outlier_handler.py`)

Provides two methods for outlier detection and clipping:

**IQR Method (Interquartile Range)**:
- Clips values outside `[Q1 - multiplier*IQR, Q3 + multiplier*IQR]`
- Standard multiplier: 1.5 (removes ~0.7% of data)
- More lenient: 3.0 (removes ~0.02% of data)
- Good for: General-purpose outlier removal

**Z-Score Method**:
- Clips values with `|z-score| > threshold`
- Standard threshold: 3.0 (removes ~0.3% of data)
- Stricter: 2.0 (removes ~5% of data)
- Good for: Normally-distributed data

### 2. Configuration (`config/data_config.yaml`)

```yaml
preprocessing:
  selected_columns: [close]
  handle_missing: forward_fill
  sort_by_index: true
  outlier_method: iqr  # 'none', 'iqr', 'zscore'
  outlier_threshold: 1.5  # IQR multiplier or z-score threshold
```

### 3. Validator Integration (`src/infrastructure/torch/lstm_trainer.py`)

```python
# Handle outliers if configured
if self.preprocessing_config and self.preprocessing_config.outlier_method != 'none':
    combined_values = OutlierHandler.handle_outliers(
        combined_values,
        method=self.preprocessing_config.outlier_method,
        threshold=self.preprocessing_config.outlier_threshold,
    )

# Then scale normally
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_values.reshape(-1, 1))
```

## Configuration Options

### No Outlier Handling

```yaml
preprocessing:
  outlier_method: none
```

Use when: Data is already clean or you want to preserve all values.

### IQR Method (Recommended)

```yaml
preprocessing:
  outlier_method: iqr
  outlier_threshold: 1.5  # Standard (removes ~0.7%)
```

Use when: You want robust outlier removal that preserves most data.

**Threshold Guide**:
- 1.5 (standard): Removes ~0.7% of data
- 2.0: Removes ~0.2% of data
- 3.0 (lenient): Removes ~0.02% of data

### Z-Score Method

```yaml
preprocessing:
  outlier_method: zscore
  outlier_threshold: 3.0  # Standard (removes ~0.3%)
```

Use when: Data is approximately normally distributed.

**Threshold Guide**:
- 2.0 (strict): Removes ~5% of data
- 2.5: Removes ~1.2% of data
- 3.0 (standard): Removes ~0.3% of data

## How It Works

### Before Outlier Handling

```
Raw data: [100, 101, 102, ..., 105, 50000000]
                                    ↑ extreme outlier
MinMaxScaler result: [0.000002, 0.000002, ..., 0.000002, 1.0]
                      ↑ all normal values compressed to near-zero
```

### After Outlier Handling (IQR)

```
Raw data: [100, 101, 102, ..., 105, 50000000]
                                    ↑ clipped to ~105
Clipped data: [100, 101, 102, ..., 105, 105]
MinMaxScaler result: [0.0, 0.01, 0.02, ..., 1.0, 1.0]
                      ↑ normal values spread across full range
```

## Performance Impact

| Method | Data Removed | Computation | Use Case |
|--------|-------------|-------------|----------|
| none | 0% | Minimal | Clean data |
| iqr (1.5) | ~0.7% | Very fast | Most cases |
| iqr (3.0) | ~0.02% | Very fast | Preserve outliers |
| zscore (3.0) | ~0.3% | Fast | Normal distribution |
| zscore (2.0) | ~5% | Fast | Strict filtering |

## Logging Output

When outliers are handled, you'll see:

```
IQR: Clipped 42 outliers (bounds: [95.5, 105.3])
```

or

```
Z-score: Clipped 15 outliers (threshold: 3.0)
```

This helps you understand what's being removed.

## Examples

### Example 1: Gold Price Data (Recommended)

```yaml
preprocessing:
  outlier_method: iqr
  outlier_threshold: 1.5
```

Gold prices rarely have extreme spikes, so IQR with standard threshold works well.

### Example 2: Volatile Crypto Data

```yaml
preprocessing:
  outlier_method: iqr
  outlier_threshold: 3.0  # More lenient
```

Crypto can have sudden spikes, so use lenient IQR to preserve more data.

### Example 3: Clean Synthetic Data

```yaml
preprocessing:
  outlier_method: none
```

If data is already clean, skip outlier handling.

## Verification

Check outlier handling in action:

```bash
python3 src/cli/optimize_cli.py --max-trials 1 --verbose
```

Output should show:
```
Loaded 1000 samples from data source
IQR: Clipped 5 outliers (bounds: [1234.5, 1456.3])
Train samples: 950, Val samples: 50
```

## Architecture

Outlier handling is applied:
1. **After data loading** (samples_cap applied)
2. **Before normalization** (MinMaxScaler)
3. **Before sequence building** (LSTM sequences)

This ensures:
- Outliers don't distort the scale
- Normal values use full [0,1] range
- Model trains on properly normalized data

## Related Configuration

- **samples_cap** (data_config.yaml): Limits input data loading
- **outlier_method/threshold** (data_config.yaml): Handles outliers
- **train_window/val_window** (lstm_config.yaml): Further slices data

All work together for robust data preprocessing.

## Troubleshooting

### "Too many outliers clipped"

**Cause**: Threshold too strict
**Solution**: Increase threshold (e.g., 1.5 → 3.0 for IQR)

### "Not enough outliers clipped"

**Cause**: Threshold too lenient
**Solution**: Decrease threshold (e.g., 3.0 → 1.5 for IQR)

### "Model performance degraded"

**Cause**: Legitimate data removed as outliers
**Solution**: Set `outlier_method: none` to preserve all data

## Best Practices

1. **Start with IQR (1.5)**: Good default for most data
2. **Monitor clipping**: Check logs to see how many outliers are removed
3. **Validate results**: Compare model performance with/without outlier handling
4. **Adjust threshold**: Fine-tune based on your domain knowledge
5. **Document choice**: Record which method you used for reproducibility
