# Simple LSTM - Refactored with Hexagonal Architecture

A production-grade LSTM time series forecasting system built with **hexagonal (ports & adapters) architecture**, following **SOLID principles** and **clean architecture** best practices.

## Quick Start

### Prerequisites

```bash
pip install pyyaml pandas numpy torch scikit-learn
```

### Run Grid Search

```bash
cd /Users/jack/Projects/simple-lstm
python src/cli/optimize_cli.py --max-trials 100 --verbose
```

### Configure Grid Search

Edit `config/optimization_config.yaml`:

```yaml
grid_search:
  sequence_length: [64, 128, 256]
  learning_rate: [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4]
  batch_size: [64, 128, 256]
  units: [256, 512, 768]
  layers: [2, 3, 4]
  dropout: [0.1, 0.2, 0.3]
```

### Configure Data Source

Edit `config/data_config.yaml`:

```yaml
data_source:
  type: csv
  path: /Users/jack/Projects/lstm/forecasting/data/raw/gold_1minute.csv
  target_column: close
```

### Configure Training

Edit `config/lstm_config.yaml`:

```yaml
training:
  max_epochs: 500
  early_stop_patience: 10
  log_interval_batches: 200
  log_interval_seconds: 30.0

learning_rate_scheduler:
  factor: 0.5
  patience: 5
  min_lr: 1e-6

optimization:
  train_window: 10000
  val_window: 1000
```

## Architecture

See `ARCHITECTURE.md` for detailed documentation of the hexagonal architecture design.

### Key Components

- **Domain Layer**: Core business logic (ports, models, services)
- **Infrastructure Layer**: External adapters (persistence, data, PyTorch)
- **Application Layer**: Use cases and orchestration
- **Configuration Layer**: YAML-based configuration
- **CLI Layer**: Command-line interface

## Features

✅ **Hexagonal Architecture**: Clean separation of concerns  
✅ **SOLID Principles**: Single responsibility, dependency inversion, etc.  
✅ **Configuration-Driven**: All settings in YAML files  
✅ **Resumable Search**: Save/load search state  
✅ **Comprehensive Logging**: CSV results + JSON state  
✅ **Testable**: Mock/fake adapters for unit testing  
✅ **Extensible**: Add new adapters without changing core logic  
✅ **Type-Safe**: Dataclasses and type hints throughout  

## Project Structure

```
simple-lstm/
├── config/                          # Configuration files
│   ├── lstm_config.yaml
│   ├── optimization_config.yaml
│   └── data_config.yaml
├── src/
│   ├── config/                      # Configuration adapter
│   ├── domain/                      # Core business logic
│   ├── infrastructure/              # External adapters
│   ├── application/                 # Use cases
│   └── cli/                         # Command-line interface
├── models/                          # Output directory (auto-created)
│   ├── lstm_search_state.json       # Search state
│   ├── lstm_grid_results.csv        # Trial results
│   └── lstm_best_params.json        # Best parameters
├── ARCHITECTURE.md                  # Architecture documentation
└── README_REFACTORED.md             # This file
```

## Output Files

After running grid search, the following files are created in `models/`:

### `lstm_search_state.json`
Resumable search state:
```json
{
  "completed": [
    {"sequence_length": 64, "learning_rate": 5e-5, ...},
    ...
  ],
  "best_params": {
    "sequence_length": 128,
    "learning_rate": 1e-4,
    ...
  },
  "best_loss": 0.0123
}
```

### `lstm_grid_results.csv`
All trial results:
```
trial_number,sequence_length,learning_rate,batch_size,units,layers,dropout,val_loss,val_rmse,val_mae,duration_seconds,timestamp
1,64,5e-05,64,256,2,0.1,0.0234,0.1523,0.0891,1234.5,1698000000
2,64,5e-05,64,256,2,0.2,0.0198,0.1407,0.0812,1245.3,1698001245
...
```

### `lstm_best_params.json`
Best parameters found:
```json
{
  "best_params": {
    "sequence_length": 128,
    "learning_rate": 1e-4,
    "batch_size": 128,
    "units": 512,
    "layers": 3,
    "dropout": 0.2
  },
  "best_loss": 0.0123,
  "timestamp": 1698000000
}
```

## Extending the System

### Add a New Data Source

1. Create adapter implementing `DataSource` port:

```python
# src/infrastructure/data/parquet_data_source.py
from ...domain.ports import DataSource
import pandas as pd

class ParquetDataSource(DataSource):
    def __init__(self, filepath: str, target_column: str):
        self.filepath = filepath
        self.target_column = target_column
    
    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.filepath)
        return df[[self.target_column]]
```

2. Update CLI to use new adapter:

```python
# In src/cli/optimize_cli.py
if data_config.data_source.type == 'parquet':
    data_source = ParquetDataSource(...)
else:
    data_source = CSVDataSource(...)
```

### Add a New Persistence Backend

1. Create adapter implementing `SearchStateRepository` port:

```python
# src/infrastructure/persistence/postgres_repository.py
from ...domain.ports import SearchStateRepository
import psycopg2

class PostgresSearchStateRepository(SearchStateRepository):
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
    
    def load(self) -> Dict[str, Any]:
        # Query database
        pass
    
    def save(self, state: Dict[str, Any]) -> None:
        # Update database
        pass
```

2. Update CLI to use new adapter:

```python
# In src/cli/optimize_cli.py
if optimization_config.persistence.backend == 'postgres':
    state_repo = PostgresSearchStateRepository(...)
else:
    state_repo = JSONSearchStateRepository(...)
```

### Add a New Validator

1. Create adapter implementing `LSTMValidator` port:

```python
# src/infrastructure/torch/tensorflow_validator.py
from ...domain.ports import LSTMValidator
import tensorflow as tf

class TensorFlowLSTMValidator(LSTMValidator):
    def validate(self, parameters: ParameterSet) -> ValidationMetrics:
        # TensorFlow training loop
        pass
```

2. Update CLI to use new validator:

```python
# In src/cli/optimize_cli.py
if lstm_config.framework == 'tensorflow':
    validator = TensorFlowLSTMValidator(...)
else:
    validator = PyTorchLSTMValidator(...)
```

## Testing

### Unit Test Example

```python
# tests/test_grid_search_service.py
from src.domain.services import GridSearchService
from src.domain.models import ParameterSet, ValidationMetrics

class FakeValidator:
    def validate(self, params: ParameterSet) -> ValidationMetrics:
        return ValidationMetrics(
            val_loss=0.01,
            val_rmse=0.1,
            val_mae=0.05,
            duration_seconds=1.0,
        )

def test_grid_search():
    validator = FakeValidator()
    # ... setup other fakes ...
    
    service = GridSearchService(
        grid_generator=grid_gen,
        validator=validator,
        state_repo=state_repo,
        results_repo=results_repo,
        best_params_repo=best_params_repo,
    )
    
    results = list(service.run(max_trials=5))
    assert len(results) == 5
```

## Performance Tuning

### Reduce Search Time

1. **Smaller optimization windows** in `lstm_config.yaml`:
   ```yaml
   optimization:
     train_window: 5000    # Reduced from 10000
     val_window: 500       # Reduced from 1000
   ```

2. **Smaller grid** in `optimization_config.yaml`:
   ```yaml
   grid_search:
     sequence_length: [64, 128]        # Reduced options
     learning_rate: [1e-4, 5e-4]       # Reduced options
     batch_size: [128]                 # Fixed value
     units: [256, 512]                 # Reduced options
     layers: [2, 3]                    # Reduced options
     dropout: [0.1, 0.2]               # Reduced options
   ```

3. **Limit trials**:
   ```bash
   python src/cli/optimize_cli.py --max-trials 50
   ```

### Increase Search Quality

1. **Larger optimization windows**:
   ```yaml
   optimization:
     train_window: 50000
     val_window: 5000
   ```

2. **Finer grid**:
   ```yaml
   grid_search:
     learning_rate: [1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3]
   ```

3. **Longer training**:
   ```yaml
   training:
     max_epochs: 1000
     early_stop_patience: 20
   ```

## Troubleshooting

### Grid search crashes during validation

Check `models/lstm_grid_results.csv` for patterns in failing trials.

### Out of memory

Reduce `optimization.train_window` and `optimization.val_window` in config.

### Slow training

- Reduce `max_epochs` in `lstm_config.yaml`
- Reduce `optimization_train_window` in `lstm_config.yaml`
- Use smaller `batch_size` values in `optimization_config.yaml`

## Migration from Old Code

The old `optimize_lstm.py` and `train_lstm.py` are still available. To migrate:

1. Update imports to use new modules
2. Update configuration in YAML files
3. Run new CLI: `python src/cli/optimize_cli.py`

The old code can be deprecated once validation confirms identical behavior.

## References

- **Hexagonal Architecture**: https://alistair.cockburn.us/hexagonal-architecture/
- **SOLID Principles**: https://en.wikipedia.org/wiki/SOLID
- **Clean Architecture**: https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html
- **Ports & Adapters**: https://en.wikipedia.org/wiki/Hexagonal_architecture_(software)

## License

MIT
