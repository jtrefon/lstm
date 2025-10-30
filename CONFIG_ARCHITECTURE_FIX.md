# Configuration Architecture Fix

## Problem Statement

The refactored architecture had a critical flaw:

1. **Duplicate Configuration**: Values defined in both YAML config files AND as default values in dataclass definitions
2. **Silent Fallbacks**: Missing config values silently fell back to hardcoded defaults instead of failing fast
3. **Confusion & Errors**: Unclear which values were being used - config or implementation defaults
4. **No Samples Cap**: No way to limit the number of optimization trials
5. **Maintenance Nightmare**: Changes to defaults required updating both config AND code

## Solution: Fail-Fast Architecture

### 1. Remove All Default Values from Dataclasses

**Before**:
```python
@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 500
    early_stop_patience: int = 10
    log_interval_batches: int = 200
```

**After**:
```python
@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    early_stop_patience: int
    log_interval_batches: int
```

**Impact**: Missing config values now raise `KeyError` immediately, making configuration errors obvious.

### 2. Replace `.get()` with Direct Dict Access

**Before**:
```python
max_epochs = int(t.get('max_epochs', 500))  # Silent fallback to 500
```

**After**:
```python
max_epochs = int(t['max_epochs'])  # Raises KeyError if missing
```

**Impact**: Configuration errors are caught at startup, not hidden during runtime.

### 3. Add Samples Cap to Grid Search Config

**New field in `GridSearchConfig`**:
```python
samples_cap: Optional[int]  # Max number of trials to run (None = all)
```

**In `optimization_config.yaml`**:
```yaml
grid_search:
  sequence_length: [64, 128, 256]
  learning_rate: [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4]
  batch_size: [64, 128, 256]
  units: [256, 512, 768]
  layers: [2, 3, 4]
  dropout: [0.1, 0.2, 0.3]
  samples_cap: null  # null = run all combinations
```

**Impact**: Users can now cap optimization trials via config, not just CLI args.

## Files Modified

### 1. `src/config/config_loader.py`

**Changes**:
- Removed all default values from dataclasses (LSTMModelConfig, TrainingConfig, LearningRateSchedulerConfig, OptimizationWindowsConfig, ForecastConfig, LoggingConfig, PreprocessingConfig)
- Replaced all `.get(key, default)` with direct `[key]` access in config loaders
- Added `samples_cap` field to `GridSearchConfig`
- Added parsing for `samples_cap` in `load_optimization_config()`

**Result**: Config loading now fails fast on missing values.

### 2. `src/domain/services.py`

**Changes**:
- Updated `run()` method signature: `max_trials` → `samples_cap`
- Updated docstring to clarify `samples_cap` comes from config

**Result**: Service uses config-driven samples cap.

### 3. `src/application/grid_search_orchestrator.py`

**Changes**:
- Updated service call: `self.service.run(samples_cap=request.max_trials)`

**Result**: Orchestrator passes samples cap to service.

### 4. `src/cli/optimize_cli.py`

**Changes**:
- Added logic to use `samples_cap` from config
- CLI `--max-trials` arg overrides config value if provided

**Result**: Config is primary source, CLI is override.

### 5. `config/optimization_config.yaml`

**Changes**:
- Added `samples_cap: null` field to grid_search section

**Result**: Users can now control trial count via config.

## Architecture Principles Applied

### 1. Single Source of Truth

- **Before**: Values in both YAML and Python code
- **After**: Values ONLY in YAML config files

### 2. Fail-Fast on Configuration Errors

- **Before**: Missing config silently used hardcoded defaults
- **After**: Missing config raises `KeyError` at startup

### 3. Explicit Over Implicit

- **Before**: Unclear which values were being used
- **After**: All values explicitly come from config

### 4. Configuration-Driven Design

- **Before**: Code had hardcoded defaults
- **After**: Code reads ONLY from config, no defaults

## Usage Examples

### Set Samples Cap via Config

Edit `config/optimization_config.yaml`:
```yaml
grid_search:
  samples_cap: 100  # Run only 100 trials
```

### Override via CLI

```bash
python3 src/cli/optimize_cli.py --max-trials 50
```

This overrides the config value.

### Run All Combinations

```yaml
grid_search:
  samples_cap: null  # null = all combinations
```

## Error Handling

### Missing Required Config

```
KeyError: 'max_epochs'
```

**Fix**: Add the missing field to the YAML config file.

### Invalid Type

```
ValueError: invalid literal for int() with base 10: 'abc'
```

**Fix**: Ensure the config value is the correct type (int, float, list, etc.).

## Testing Configuration

Verify config loading works:

```bash
cd /Users/jack/Projects/simple-lstm
python3 -c "
from src.config.config_loader import ConfigLoader
lstm_cfg = ConfigLoader.load_lstm_config('config/lstm_config.yaml')
opt_cfg = ConfigLoader.load_optimization_config('config/optimization_config.yaml')
data_cfg = ConfigLoader.load_data_config('config/data_config.yaml')
print('✓ All configs loaded successfully')
print(f'  Samples cap: {opt_cfg.grid_search.samples_cap}')
"
```

## Benefits

1. **Clarity**: All values in one place (YAML config)
2. **Safety**: Missing config caught at startup
3. **Maintainability**: No duplicate values to keep in sync
4. **Flexibility**: Easy to adjust parameters without code changes
5. **Auditability**: Config file is the source of truth

## Migration Guide

If you have old code using hardcoded defaults:

1. Move all defaults to YAML config files
2. Remove default values from dataclasses
3. Replace `.get(key, default)` with `[key]`
4. Test that missing config raises `KeyError`

## Next Steps

- Run optimizer to verify config loading works
- Check that missing config values raise errors
- Adjust `samples_cap` in config as needed
- Monitor that all values come from config, not code
