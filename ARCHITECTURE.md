# Hexagonal Architecture - LSTM Grid Search

This document describes the refactored architecture following **hexagonal (ports & adapters)** pattern with strong adherence to **SOLID principles**, **SRP (Single Responsibility Principle)**, **DRY (Don't Repeat Yourself)**, and **clean architecture** principles.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer                                 │
│                   (optimize_cli.py)                              │
│              Wires all components together                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   Application Layer                              │
│            (GridSearchOrchestrator, DTOs)                        │
│         Implements use cases, coordinates domain logic           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Domain Layer (Core)                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Ports (Abstract Interfaces)                              │   │
│  │ - DataSource, SeriesSplitter, SequenceBuilder            │   │
│  │ - LSTMValidator, ParameterGridGenerator                  │   │
│  │ - SearchStateRepository, ResultsRepository, etc.         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Models (Value Objects)                                   │   │
│  │ - ParameterSet, ValidationMetrics                        │   │
│  │ - SearchTrialResult, SearchSummary                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Services (Business Logic)                                │   │
│  │ - GridSearchService (orchestrates validation trials)     │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              Infrastructure Layer (Adapters)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Persistence Adapters                                     │   │
│  │ - JSONSearchStateRepository                              │   │
│  │ - CSVResultsRepository                                   │   │
│  │ - JSONBestParamsRepository                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Data Adapters                                            │   │
│  │ - CSVDataSource (I/O)                                    │   │
│  │ - TimeSeriesSplitter (data processing)                   │   │
│  │ - OptimizationWindowSplitter (data processing)           │   │
│  │ - NumpySequenceBuilder (data processing)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PyTorch Adapters                                         │   │
│  │ - PyTorchLSTMValidator (training loop)                   │   │
│  │ - LSTMModel (neural network)                             │   │
│  │ - ConfigBasedParameterGridGenerator                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              Configuration Layer (External)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ YAML Configuration Files                                 │   │
│  │ - lstm_config.yaml (model & training)                    │   │
│  │ - optimization_config.yaml (grid search)                 │   │
│  │ - data_config.yaml (data source & splitting)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ConfigLoader (Adapter)                                   │   │
│  │ - Parses YAML into typed config objects                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
simple-lstm/
├── config/                          # Configuration files (external)
│   ├── lstm_config.yaml             # Model & training config
│   ├── optimization_config.yaml     # Grid search hyperparameter space
│   └── data_config.yaml             # Data source & splitting config
│
├── src/                             # Source code (hexagonal architecture)
│   ├── __init__.py
│   │
│   ├── config/                      # Configuration adapter layer
│   │   ├── __init__.py
│   │   └── config_loader.py         # YAML parsing → typed objects
│   │
│   ├── domain/                      # Core business logic (no external deps)
│   │   ├── __init__.py
│   │   ├── models.py                # Value objects (ParameterSet, etc.)
│   │   ├── ports.py                 # Abstract interfaces (ports)
│   │   └── services.py              # Business logic (GridSearchService)
│   │
│   ├── infrastructure/              # External adapters
│   │   ├── __init__.py
│   │   ├── persistence/             # I/O adapters for persistence
│   │   │   ├── __init__.py
│   │   │   ├── search_state_repository.py      # JSON state persistence
│   │   │   ├── results_repository.py           # CSV results persistence
│   │   │   └── best_params_repository.py       # JSON best params persistence
│   │   ├── data/                    # Data processing adapters
│   │   │   ├── __init__.py
│   │   │   ├── csv_data_source.py              # CSV loading (I/O)
│   │   │   ├── series_splitter.py              # Train/val/test splitting
│   │   │   └── sequence_builder.py             # Sequence creation
│   │   └── torch/                   # PyTorch adapters
│   │       ├── __init__.py
│   │       ├── lstm_trainer.py                 # Training loop & validator
│   │       └── parameter_grid_generator.py     # Config → parameter combinations
│   │
│   ├── application/                 # Use cases & orchestration
│   │   ├── __init__.py
│   │   ├── dto.py                   # Application DTOs
│   │   └── grid_search_orchestrator.py # Grid search use case
│   │
│   └── cli/                         # Command-line interface
│       ├── __init__.py
│       └── optimize_cli.py          # CLI entry point
│
├── ARCHITECTURE.md                  # This file
└── README.md                        # Usage guide
```

## SOLID Principles Adherence

### 1. **Single Responsibility Principle (SRP)**

Each class has **one reason to change**:

- **`CSVDataSource`**: Only responsible for loading CSV files
- **`TimeSeriesSplitter`**: Only responsible for splitting time series
- **`NumpySequenceBuilder`**: Only responsible for creating sequences
- **`PyTorchLSTMValidator`**: Only responsible for training & validation
- **`JSONSearchStateRepository`**: Only responsible for state persistence
- **`CSVResultsRepository`**: Only responsible for results persistence
- **`GridSearchService`**: Only responsible for orchestrating search logic
- **`GridSearchOrchestrator`**: Only responsible for use case coordination

### 2. **Open/Closed Principle (OCP)**

Classes are **open for extension, closed for modification**:

- All external adapters implement abstract `Port` interfaces
- New persistence backends (e.g., PostgreSQL) can be added without modifying domain logic
- New data sources (e.g., Parquet) can be added without changing core logic
- New validators (e.g., TensorFlow) can be added by implementing `LSTMValidator` port

### 3. **Liskov Substitution Principle (LSP)**

All implementations are **substitutable** for their interfaces:

- Any `DataSource` implementation works with the same code
- Any `SeriesSplitter` implementation works identically
- Any `LSTMValidator` implementation produces compatible results

### 4. **Interface Segregation Principle (ISP)**

Interfaces are **focused and minimal**:

- `DataSource` has only `load()` method
- `SeriesSplitter` has only `split()` method
- `SequenceBuilder` has only `build()` method
- No "fat" interfaces with unrelated methods

### 5. **Dependency Inversion Principle (DIP)**

High-level modules depend on **abstractions, not concrete implementations**:

- `GridSearchService` depends on `Port` interfaces, not concrete adapters
- `GridSearchOrchestrator` depends on `GridSearchService`, not implementation details
- CLI wires concrete implementations at the boundary

## Configuration-Driven Design

All hyperparameters and settings are **externalized to YAML**:

### `lstm_config.yaml` - Model & Training
```yaml
model:
  input_size: 1
  sequence_stride: 1

training:
  max_epochs: 500
  early_stop_patience: 10
  ...

learning_rate_scheduler:
  factor: 0.5
  patience: 5
  min_lr: 1e-6

optimization:
  train_window: 10000
  val_window: 1000
```

### `optimization_config.yaml` - Grid Search
```yaml
grid_search:
  sequence_length: [64, 128, 256]
  learning_rate: [5e-5, 1e-4, ...]
  batch_size: [64, 128, 256]
  units: [256, 512, 768]
  layers: [2, 3, 4]
  dropout: [0.1, 0.2, 0.3]

persistence:
  state_file: models/lstm_search_state.json
  results_file: models/lstm_grid_results.csv
  best_params_file: models/lstm_best_params.json
```

### `data_config.yaml` - Data Source & Splitting
```yaml
data_source:
  type: csv
  path: /path/to/data.csv
  target_column: close

splitting:
  train_ratio: 0.95
  validation_ratio: 0.035
  test_ratio: 0.015

preprocessing:
  selected_columns: [close]
  handle_missing: forward_fill
  sort_by_index: true
```

**Benefits:**
- No hardcoded values in code
- Easy to experiment with different configurations
- Reproducible runs
- Clear separation of concerns

## DRY (Don't Repeat Yourself)

### Eliminated Duplication

**Before:**
- `optimize_lstm.py` had hardcoded grid values
- `train_lstm.py` had hardcoded config values
- Multiple files parsing/storing JSON/CSV

**After:**
- Single source of truth: YAML config files
- Reusable `ConfigLoader` for all config parsing
- Centralized `ParameterSet` model for all parameter handling
- Shared persistence adapters

### Shared Utilities

- **`ParameterSet`**: Single model for all parameter handling (no dict duplication)
- **`ValidationMetrics`**: Single model for metrics (no tuple unpacking)
- **`SearchTrialResult`**: Single model for trial results (no scattered fields)
- **`ConfigLoader`**: Single parser for all YAML files

## Testability

The hexagonal architecture enables **easy unit testing**:

```python
# Example: Test grid search without I/O
def test_grid_search():
    # Use in-memory fake repositories
    state_repo = FakeSearchStateRepository()
    results_repo = FakeResultsRepository()
    best_params_repo = FakeBestParamsRepository()
    
    # Use mock validator
    validator = MockLSTMValidator()
    
    # Test service in isolation
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

## Usage

### Run Grid Search

```bash
cd simple-lstm
python src/cli/optimize_cli.py --max-trials 100 --verbose
```

### Modify Configuration

Edit YAML files in `config/` directory:

```bash
# Change grid search space
vim config/optimization_config.yaml

# Change data source
vim config/data_config.yaml

# Change training parameters
vim config/lstm_config.yaml
```

### Add New Adapter

1. Create new class implementing a `Port` interface
2. Register in CLI wiring code
3. No changes to domain logic needed

Example: Add PostgreSQL persistence

```python
# src/infrastructure/persistence/postgres_repository.py
class PostgresSearchStateRepository(SearchStateRepository):
    def load(self) -> Dict[str, Any]:
        # Query database
        pass
    
    def save(self, state: Dict[str, Any]) -> None:
        # Update database
        pass

# In CLI, swap repositories
state_repo = PostgresSearchStateRepository(connection_string)
```

## Key Design Patterns

1. **Hexagonal Architecture**: Ports & adapters for external dependencies
2. **Dependency Injection**: All dependencies injected at boundaries
3. **Value Objects**: Immutable models (`ParameterSet`, `ValidationMetrics`, etc.)
4. **Repository Pattern**: Abstract persistence behind interfaces
5. **Factory Pattern**: `ConfigLoader` creates typed config objects
6. **Strategy Pattern**: Swappable validators, splitters, builders
7. **Facade Pattern**: `GridSearchOrchestrator` simplifies use case

## Benefits

✅ **Maintainability**: Clear separation of concerns, easy to understand  
✅ **Testability**: Mock/fake adapters for unit testing  
✅ **Extensibility**: Add new adapters without changing core logic  
✅ **Flexibility**: Swap implementations (e.g., different databases)  
✅ **Configuration**: All settings in YAML, no code changes needed  
✅ **Decoupling**: Domain logic independent of external frameworks  
✅ **Reproducibility**: Config-driven, deterministic runs  
✅ **Scalability**: Easy to parallelize or distribute trials  

## Migration from Old Code

The old `optimize_lstm.py` and `train_lstm.py` are still available for reference. The new architecture coexists with the old code. To fully migrate:

1. Update imports to use new modules
2. Update configuration in YAML files
3. Run new CLI: `python src/cli/optimize_cli.py`

The old code can be deprecated once validation confirms identical behavior.
