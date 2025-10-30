# Bug Fixes - Type Conversion Issues

## Issues Fixed

### 1. Type Mismatch in GridSearchService

**Error**: `'<=' not supported between instances of 'float' and 'str'`

**Root Cause**: When loading search state from JSON, `best_loss` was loaded as a string but compared with float metrics.

**Fix**: Added explicit type conversion in `src/domain/services.py`:
```python
# Load previous state
best_loss_raw = state.get('best_loss', float('inf'))
# Ensure best_loss is float (JSON loads it as string sometimes)
try:
    best_loss = float(best_loss_raw)
except (TypeError, ValueError):
    best_loss = float('inf')
```

### 2. Format String Error

**Error**: `Unknown format code 'f' for object of type 'str'`

**Root Cause**: When saving state to JSON, `best_loss` was not being explicitly converted to float, causing format string errors later.

**Fix**: Added explicit float conversion when saving state in `src/domain/services.py`:
```python
state['best_loss'] = float(best_loss)  # Ensure it's a float for JSON serialization
```

### 3. Validator Metrics Type Inconsistency

**Error**: Metrics returned from validator could be numpy types or other non-float types.

**Root Cause**: The `_train_model` method returns dict with numpy values, which weren't being converted to Python floats.

**Fix**: Added explicit float conversion in `src/infrastructure/torch/lstm_trainer.py`:
```python
metrics = ValidationMetrics(
    val_loss=float(metrics['val_loss']),
    val_rmse=float(metrics['val_rmse']),
    val_mae=float(metrics['val_mae']),
    duration_seconds=float(time.time() - start_time),
)
```

## Files Modified

1. `src/domain/services.py`
   - Added try/except for best_loss type conversion on load
   - Added explicit float() conversion on save

2. `src/infrastructure/torch/lstm_trainer.py`
   - Added explicit float() conversion for all metrics

## Testing

Run the optimizer again:
```bash
cd /Users/jack/Projects/simple-lstm
python3 src/cli/optimize_cli.py --max-trials 5 --verbose
```

Expected behavior:
- No type conversion errors
- Grid search runs successfully
- Results saved to `models/` directory

## Root Cause Analysis

The issue stems from JSON serialization/deserialization:
- JSON stores all numbers as floats or ints
- When loaded back, they're strings if they were formatted as strings
- NumPy types (np.float64, etc.) are not JSON-serializable
- Python's float type must be used consistently

## Prevention

To prevent similar issues in the future:
1. Always explicitly convert to Python native types (float, int, str) before JSON serialization
2. Always validate types when loading from external sources
3. Use type hints and dataclasses to enforce type safety
4. Add unit tests for persistence layer
