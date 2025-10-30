# Detrended Outlier Handling - Preserve Trends, Remove Anomalies

## Problem Statement

With multi-year gold price data ($100 → $4000), standard outlier detection fails:

```
Standard IQR on raw data:
- $100 prices (2010) flagged as outliers (too low)
- $4000 prices (2024) flagged as outliers (too high)
- Result: Legitimate trend removed, model can't learn price progression
```

But real anomalies exist:
- COVID pandemic spike (2020)
- Ship sinking with gold bars (specific event)
- Mine flood (specific event)
- Post-pandemic economic spike

These are **true anomalies** (deviations from trend), not just old/new prices.

## Solution: Detrended IQR

Removes the trend, detects anomalies on detrended data, then restores the trend:

```
Step 1: Remove trend
  Raw: [100, 101, 102, ..., 3900, 4000]
  Detrended: [-50, -49, -48, ..., 50, 100]  (anomalies now visible)

Step 2: Detect outliers on detrended data
  COVID spike: +500 (anomaly!)
  Ship sinking: -300 (anomaly!)
  Flood: -200 (anomaly!)
  Normal variation: ±100 (not anomalies)

Step 3: Clip anomalies in detrended space
  Clipped: [-50, -49, -48, ..., 50, 100]  (anomalies removed)

Step 4: Restore trend
  Result: [100, 101, 102, ..., 3900, 4000]  (trend preserved, anomalies gone)
```

## How It Works

### Algorithm

```python
# 1. Detrend: remove linear trend
detrended = signal.detrend(data)

# 2. Detect outliers on detrended data (not affected by trend)
q1 = percentile(detrended, 25)
q3 = percentile(detrended, 75)
iqr = q3 - q1
bounds = [q1 - 1.5*iqr, q3 + 1.5*iqr]

# 3. Clip anomalies
clipped_detrended = clip(detrended, bounds)

# 4. Restore trend
trend = data - detrended
result = clipped_detrended + trend
```

### Key Insight

- **Trend** = underlying price progression ($100 → $4000)
- **Anomalies** = deviations from trend (COVID spike, ship sinking, etc.)
- **Detrended IQR** = detects anomalies, preserves trend

## Configuration

### Enable Detrended IQR

```yaml
preprocessing:
  outlier_method: detrended_iqr
  outlier_threshold: 1.5  # IQR multiplier (standard)
```

### Threshold Guide

| Threshold | Data Removed | Use Case |
|-----------|------------|----------|
| 1.5 | ~0.7% | Standard (removes COVID spike, ship sinking, flood) |
| 2.0 | ~0.2% | More lenient (keeps more anomalies) |
| 3.0 | ~0.02% | Very lenient (only extreme anomalies) |

For gold prices with COVID/ship/flood events: **1.5 is recommended**.

## Example: Gold Price Data

### Raw Data (Multi-Year)

```
2010: $100-$150 (normal variation)
2015: $1000-$1200 (normal variation)
2020: $1500-$2500 (COVID spike anomaly)
2021: $1700-$1900 (ship sinking anomaly)
2022: $1600-$1800 (mine flood anomaly)
2024: $3900-$4000 (normal variation)
```

### With Detrended IQR (1.5)

```
Detrended analysis:
- 2010 prices: Normal variation around trend
- 2015 prices: Normal variation around trend
- 2020 COVID spike: +500 above trend → CLIPPED
- 2021 ship sinking: -300 below trend → CLIPPED
- 2022 mine flood: -200 below trend → CLIPPED
- 2024 prices: Normal variation around trend

Result:
- Trend preserved: $100 → $4000
- Anomalies removed: COVID, ship, flood
- Model trains on stable patterns
```

## Logging Output

When detrended IQR is applied:

```
Detrended IQR: Clipped 47 anomalies (bounds: [-150.5, 200.3])
  Trend preserved: min=100.00, max=4000.00
```

This shows:
- **47 anomalies** removed (COVID spike, ship sinking, flood, etc.)
- **Bounds** in detrended space (±150-200 is normal variation)
- **Trend** preserved from $100 to $4000

## Comparison: Methods

| Method | Trend | Anomalies | Use Case |
|--------|-------|-----------|----------|
| none | ✓ | ✗ | Keep all data |
| iqr | ✗ | ✗ | Fails on trending data |
| zscore | ✗ | ✗ | Fails on trending data |
| **detrended_iqr** | **✓** | **✓** | **Gold prices (RECOMMENDED)** |

## Implementation Details

### What Gets Clipped

- **COVID pandemic spike** (2020): Sudden price jump above trend
- **Ship sinking** (specific date): Price drop below trend
- **Mine flood** (specific date): Price drop below trend
- **Post-pandemic spike** (2021-2022): Gradual recovery (part of trend, not clipped)

### What's Preserved

- **Long-term trend**: $100 (2010) → $4000 (2024)
- **Normal seasonal variation**: ±5-10% around trend
- **Gradual price movements**: Recovery periods, bull/bear markets

## Verification

Run optimizer to see detrended IQR in action:

```bash
python3 src/cli/optimize_cli.py --max-trials 1 --verbose
```

Expected output:

```
Loaded 1000 samples from data source
Detrended IQR: Clipped 47 anomalies (bounds: [-150.5, 200.3])
  Trend preserved: min=100.00, max=4000.00
Train samples: 950, Val samples: 50
```

## Why Detrended IQR for Gold Prices

1. **Multi-year trend**: Prices naturally increase over time
2. **Real anomalies**: COVID, ship sinking, flood are deviations from trend
3. **Preserves learning**: Model learns $100→$4000 progression
4. **Removes noise**: Event-based spikes don't distort training
5. **Precision**: Targets specific anomalies, not entire price ranges

## Configuration Recommendations

### Conservative (Keep More Data)

```yaml
outlier_method: detrended_iqr
outlier_threshold: 2.0  # Removes only extreme anomalies
```

### Standard (Recommended)

```yaml
outlier_method: detrended_iqr
outlier_threshold: 1.5  # Removes COVID, ship, flood
```

### Aggressive (Remove More Anomalies)

```yaml
outlier_method: detrended_iqr
outlier_threshold: 1.0  # Removes more variation
```

## Related Configuration

- **samples_cap**: Limits input data loading
- **outlier_method**: Detrended IQR (NEW, recommended for trending data)
- **outlier_threshold**: 1.5 (removes ~0.7% anomalies)
- **train_window/val_window**: Further slices data for optimization

All work together for robust data preprocessing.

## Troubleshooting

### "Too many anomalies clipped"

**Cause**: Threshold too strict
**Solution**: Increase threshold (1.5 → 2.0)

### "Not enough anomalies clipped"

**Cause**: Threshold too lenient
**Solution**: Decrease threshold (2.0 → 1.5)

### "Trend looks wrong"

**Cause**: Detrending removed legitimate trend
**Solution**: This shouldn't happen - detrending preserves linear trend

### "Model performance degraded"

**Cause**: Legitimate price movements removed
**Solution**: Increase threshold to preserve more data

## Technical Notes

- Uses `scipy.signal.detrend()` for linear detrending
- Detrending removes linear trend, preserves non-linear patterns
- IQR calculated on detrended data (not affected by trend)
- Trend restored after clipping (original progression preserved)
- Fully reversible: can always revert to original data
