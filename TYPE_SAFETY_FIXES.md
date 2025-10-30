# Type Safety Fixes - Comprehensive Solution

## Problem Statement

The optimizer was failing with type mismatch errors:
- `'<=' not supported between instances of 'float' and 'str'`
- `Unknown format code 'f' for object of type 'str'`

Root cause: NumPy types and JSON serialization creating type inconsistencies throughout the validation pipeline.

## Solution: Comprehensive Type Conversion

### 1. GridSearchService (`src/domain/services.py`)

**Issue**: `best_loss` loaded from JSON as string, compared with float metrics

**Fixes**:
- Lines 45-50: Added try/except type conversion when loading state
- Line 89: Explicit float() conversion when saving state

```python
# Load with type safety
best_loss_raw = state.get('best_loss', float('inf'))
try:
    best_loss = float(best_loss_raw)
except (TypeError, ValueError):
    best_loss = float('inf')

# Save with type safety
state['best_loss'] = float(best_loss)
```

### 2. PyTorchLSTMValidator (`src/infrastructure/torch/lstm_trainer.py`)

**Issue**: NumPy types not JSON-serializable, inconsistent type handling

**Fixes**:
- Lines 174-177: Initialize best_val_loss/rmse/mae as Python floats
- Line 220: Explicit float() conversion for mean_val_loss before scheduler
- Lines 132-135: Explicit float() conversion for all returned metrics
- Lines 141-143: Added traceback printing for better error diagnostics

```python
# Initialize as Python floats
best_val_loss = float('inf')
best_val_rmse = float('inf')
best_val_mae = float('inf')

# Ensure type before scheduler
mean_val_loss = float(mean_val_loss)
scheduler.step(mean_val_loss)

# Return with explicit conversion
metrics = ValidationMetrics(
    val_loss=float(metrics['val_loss']),
    val_rmse=float(metrics['val_rmse']),
    val_mae=float(metrics['val_mae']),
    duration_seconds=float(time.time() - start_time),
)
```

## Type Conversion Strategy

### Principle: Convert at Boundaries

1. **When loading from external sources** (JSON, CSV, etc.)
   - Always convert to Python native types
   - Use try/except for robustness

2. **When passing to external libraries** (PyTorch scheduler, etc.)
   - Ensure Python float type
   - Avoid NumPy types

3. **When returning from functions**
   - Explicitly convert to expected types
   - Document type expectations

### NumPy vs Python Types

| Operation | NumPy Type | Python Type | Issue |
|-----------|-----------|-----------|-------|
| JSON serialization | np.float64 | float | NumPy not JSON-serializable |
| Comparison | np.float64 | float | May fail with string |
| Format string | np.float64 | float | Format code 'f' may fail |
| PyTorch scheduler | np.float64 | float | Scheduler expects Python float |

## Files Modified

1. **`src/domain/services.py`**
   - Lines 45-50: Type conversion on load
   - Line 89: Type conversion on save

2. **`src/infrastructure/torch/lstm_trainer.py`**
   - Lines 174-177: Initialize as Python floats
   - Line 220: Convert before scheduler
   - Lines 132-135: Convert on return
   - Lines 141-143: Better error diagnostics

## Testing

Run the optimizer:
```bash
cd /Users/jack/Projects/simple-lstm
python3 src/cli/optimize_cli.py --max-trials 5 --verbose
```

Expected behavior:
- No type conversion errors
- Grid search runs successfully
- Results saved to `models/` directory
- Traceback printed if validation fails (for debugging)

## Best Practices

1. **Always use Python native types** for JSON serialization
2. **Convert NumPy types** at function boundaries
3. **Use try/except** for type conversions from external sources
4. **Add type hints** to document expected types
5. **Test with edge cases** (inf, nan, very small/large numbers)

## Prevention Checklist

- [ ] All JSON values converted to Python types
- [ ] All NumPy operations converted to Python types before external calls
- [ ] All function returns have explicit type conversion
- [ ] All comparisons use same type
- [ ] All format strings have correct types
- [ ] Error handling includes type conversion fallbacks
