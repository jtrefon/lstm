# Quick Start - Refactored LSTM Grid Search

## Prerequisites

```bash
pip install pyyaml pandas numpy torch scikit-learn
```

## Verify Installation

Test that all imports work:

```bash
cd /Users/jack/Projects/simple-lstm
python test_imports.py
```

Expected output:
```
Testing imports...
✓ ConfigLoader
✓ Domain models
✓ Domain ports
✓ Domain services
✓ CSV data source
✓ Series splitters
✓ Sequence builder
✓ Search state repository
✓ Results repository
✓ Best params repository
✓ PyTorch LSTM trainer
✓ Parameter grid generator
✓ Application DTOs
✓ Grid search orchestrator

✅ All imports successful!
```

## Run Grid Search

### Small Test (5 trials)

```bash
cd /Users/jack/Projects/simple-lstm
python3 src/cli/optimize_cli.py --max-trials 5 --verbose
```

### Full Grid Search

```bash
cd /Users/jack/Projects/simple-lstm
python3 src/cli/optimize_cli.py --verbose
```

## Configure Grid Search

Edit `config/optimization_config.yaml` to change the hyperparameter space:

```yaml
grid_search:
  sequence_length: [64, 128, 256]
  learning_rate: [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4]
  batch_size: [64, 128, 256]
  units: [256, 512, 768]
  layers: [2, 3, 4]
  dropout: [0.1, 0.2, 0.3]
```

## Configure Data Source

Edit `config/data_config.yaml` to change the data path:

```yaml
data_source:
  type: csv
  path: /Users/jack/Projects/lstm/forecasting/data/raw/gold_1minute.csv
  target_column: close
```

## Configure Training

Edit `config/lstm_config.yaml` to change training parameters:

```yaml
training:
  max_epochs: 500
  early_stop_patience: 10

optimization:
  train_window: 10000
  val_window: 1000
```

## Output Files

After running, check results in `models/`:

```bash
# Best parameters found
cat models/lstm_best_params.json

# All trial results
head models/lstm_grid_results.csv

# Search state (for resuming)
cat models/lstm_search_state.json
```

## Resume Search

If the search is interrupted, run again to resume from where it left off:

```bash
python3 src/cli/optimize_cli.py --verbose
```

The search will skip already-completed trials and continue with new ones.

## Troubleshooting

### ImportError: attempted relative import beyond top-level package

**Solution**: All imports have been fixed. Make sure you're running from the project root:

```bash
cd /Users/jack/Projects/simple-lstm
python3 src/cli/optimize_cli.py --max-trials 5
```

### FileNotFoundError: data file not found

**Solution**: Update the data path in `config/data_config.yaml`:

```yaml
data_source:
  path: /path/to/your/data.csv
```

### Out of memory

**Solution**: Reduce optimization windows in `config/lstm_config.yaml`:

```yaml
optimization:
  train_window: 5000    # Reduced from 10000
  val_window: 500       # Reduced from 1000
```

### Slow training

**Solution**: Reduce grid search space in `config/optimization_config.yaml`:

```yaml
grid_search:
  sequence_length: [64, 128]        # Fewer options
  learning_rate: [1e-4, 5e-4]       # Fewer options
  batch_size: [128]                 # Fixed value
  units: [256, 512]                 # Fewer options
  layers: [2, 3]                    # Fewer options
  dropout: [0.1, 0.2]               # Fewer options
```

## Next Steps

1. **Run a small test** to verify everything works
2. **Check output files** in `models/` directory
3. **Adjust configuration** as needed
4. **Run full grid search** for production results
5. **Review results** in CSV file
6. **Use best parameters** for final model training

## Documentation

- `ARCHITECTURE.md` - Detailed architecture documentation
- `README_REFACTORED.md` - Complete usage guide
- `REFACTORING_SUMMARY.md` - Summary of changes
- `IMPORT_FIXES.md` - Import system explanation
- `IMPLEMENTATION_CHECKLIST.md` - Verification steps

## Support

For issues or questions, refer to the documentation files or check the error messages in the console output.
