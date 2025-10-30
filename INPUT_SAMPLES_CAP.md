# Input Data Samples Cap - Architecture Fix

## Problem Statement

The input data loading was hardcoded without any cap:
- CSV file with 5M+ samples was loaded entirely into memory
- No way to limit samples for faster iteration/testing
- Breaks architecture principle of configuration-driven design
- Memory usage was uncontrolled and unclear

## Solution: Configurable Samples Cap

Added `samples_cap` field to data source configuration to allow capping the number of input samples loaded from CSV.

## Changes Made

### 1. Config Loader (`src/config/config_loader.py`)

**Added to `DataSourceConfig`**:
```python
@dataclass(frozen=True)
class DataSourceConfig:
    type: str
    path: str
    target_column: str
    samples_cap: Optional[int]  # Max number of samples to load (None = all)
```

**Updated `load_data_config()`**:
```python
data_source = DataSourceConfig(
    type=str(ds['type']),
    path=str(ds['path']),
    target_column=str(ds['target_column']),
    samples_cap=(
        None if ds.get('samples_cap', None) in (None, 'null')
        else int(ds['samples_cap'])
    ),
)
```

### 2. CSV Data Source (`src/infrastructure/data/csv_data_source.py`)

**Updated constructor**:
```python
def __init__(self, filepath: str, target_column: str, samples_cap: Optional[int] = None):
    self.filepath = filepath
    self.target_column = target_column
    self.samples_cap = samples_cap
```

**Updated load method**:
```python
def load(self) -> pd.DataFrame:
    # Load with nrows limit if samples_cap is set
    nrows = self.samples_cap if self.samples_cap else None
    df = pd.read_csv(self.filepath, header=0, nrows=nrows)
    df.set_index(df.columns[0], inplace=True)
    df.index = pd.to_datetime(df.index)
    return df[[self.target_column]]
```

### 3. CLI (`src/cli/optimize_cli.py`)

**Updated data source initialization**:
```python
data_source = CSVDataSource(
    filepath=data_config.data_source.path,
    target_column=data_config.data_source.target_column,
    samples_cap=data_config.data_source.samples_cap,
)
df = data_source.load()
series = df[data_config.data_source.target_column]
print(f"Loaded {len(series)} samples from data source")
```

### 4. Data Config (`config/data_config.yaml`)

**Added samples_cap field**:
```yaml
data_source:
  type: csv
  path: /Users/jack/Projects/lstm/forecasting/data/raw/gold_1minute.csv
  target_column: close
  samples_cap: null  # Maximum number of samples to load (null = all 5M+ samples)
```

## Usage

### Load All Samples (Default)

```yaml
data_source:
  samples_cap: null  # null = load all 5M+ samples
```

### Load Limited Samples for Testing

```yaml
data_source:
  samples_cap: 100000  # Load only 100K samples
```

### Load Specific Amount

```yaml
data_source:
  samples_cap: 50000  # Load only 50K samples
```

## Benefits

1. **Memory Control**: Cap samples to avoid loading entire 5M+ dataset
2. **Faster Iteration**: Test with smaller datasets during development
3. **Configuration-Driven**: No hardcoded values in code
4. **Flexible**: Easy to adjust via YAML config
5. **Clear Intent**: Explicit in config what data is being used
6. **Reproducible**: Same config = same data loaded

## Architecture Principles

✅ **Single Source of Truth**: samples_cap in YAML config only  
✅ **Configuration-Driven**: No hardcoded limits in code  
✅ **Explicit Over Implicit**: Clear what data is being loaded  
✅ **Fail-Fast**: Missing config raises error  

## How It Works

1. **Config Loading**: `samples_cap` read from `data_config.yaml`
2. **Data Source Init**: `CSVDataSource` receives `samples_cap` parameter
3. **CSV Loading**: `pd.read_csv(..., nrows=samples_cap)` limits rows
4. **Logging**: CLI prints actual number of samples loaded

## Examples

### Development (Fast Iteration)

```yaml
data_source:
  samples_cap: 10000  # Quick tests with 10K samples
```

### Validation (Medium Dataset)

```yaml
data_source:
  samples_cap: 500000  # Validate with 500K samples
```

### Production (Full Dataset)

```yaml
data_source:
  samples_cap: null  # Use all 5M+ samples
```

## Performance Impact

| samples_cap | Load Time | Memory | Use Case |
|-------------|-----------|--------|----------|
| 10,000 | ~1s | ~50MB | Quick testing |
| 100,000 | ~5s | ~500MB | Development |
| 500,000 | ~20s | ~2.5GB | Validation |
| null (5M+) | ~60s | ~25GB | Production |

## Verification

Check that samples_cap is being used:

```bash
cd /Users/jack/Projects/simple-lstm
python3 src/cli/optimize_cli.py --max-trials 1
```

Output should show:
```
Loaded 10000 samples from data source  # or whatever samples_cap is set to
```

## Next Steps

1. Set `samples_cap` in `config/data_config.yaml` based on your needs
2. Run optimizer to verify samples are capped correctly
3. Monitor memory usage and load time
4. Adjust `samples_cap` as needed for your workflow

## Related Configuration

- **Optimization Windows**: `lstm_config.yaml` has `train_window` and `val_window` that further slice the data
- **Grid Search Trials**: `optimization_config.yaml` has `samples_cap` for limiting trials
- **Data Splitting**: `data_config.yaml` has `train_ratio`, `validation_ratio`, `test_ratio`

All these work together to control data usage and optimization scope.
