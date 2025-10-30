# Import Fixes - Hexagonal Architecture

## Problem

The CLI was failing with:
```
ImportError: attempted relative import beyond top-level package
```

This occurred because:
1. The CLI adds `src/` to `sys.path` to enable absolute imports
2. Infrastructure modules were using relative imports (`from ...domain.ports`)
3. When `src/` is added to path, relative imports from infrastructure modules fail

## Solution

Converted all relative imports to absolute imports throughout the codebase.

### Files Modified

#### Domain Layer
- `src/domain/services.py`
  - Changed: `from .models` → `from domain.models`
  - Changed: `from .ports` → `from domain.ports`

#### Infrastructure Layer - Data
- `src/infrastructure/data/csv_data_source.py`
  - Changed: `from ...domain.ports` → `from domain.ports`

- `src/infrastructure/data/series_splitter.py`
  - Changed: `from ...domain.ports` → `from domain.ports`

- `src/infrastructure/data/sequence_builder.py`
  - Changed: `from ...domain.ports` → `from domain.ports`

#### Infrastructure Layer - Persistence
- `src/infrastructure/persistence/search_state_repository.py`
  - Changed: `from ...domain.ports` → `from domain.ports`

- `src/infrastructure/persistence/results_repository.py`
  - Changed: `from ...domain.models` → `from domain.models`
  - Changed: `from ...domain.ports` → `from domain.ports`

- `src/infrastructure/persistence/best_params_repository.py`
  - Changed: `from ...domain.models` → `from domain.models`
  - Changed: `from ...domain.ports` → `from domain.ports`

#### Infrastructure Layer - PyTorch
- `src/infrastructure/torch/lstm_trainer.py`
  - Changed: `from ...config.config_loader` → `from config.config_loader`
  - Changed: `from ...domain.models` → `from domain.models`
  - Changed: `from ...domain.ports` → `from domain.ports`

- `src/infrastructure/torch/parameter_grid_generator.py`
  - Changed: `from ...config.config_loader` → `from config.config_loader`
  - Changed: `from ...domain.models` → `from domain.models`
  - Changed: `from ...domain.ports` → `from domain.ports`

#### Application Layer
- `src/application/dto.py`
  - Changed: `from ..domain.models` → `from domain.models`

- `src/application/grid_search_orchestrator.py`
  - Changed: `from ..domain.models` → `from domain.models`
  - Changed: `from ..domain.services` → `from domain.services`
  - Changed: `from .dto` → `from application.dto`

## How It Works

The CLI (`src/cli/optimize_cli.py`) adds the `src/` directory to Python's path:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This allows all modules to use absolute imports from the `src/` root:

```python
# Instead of:
from ...domain.ports import DataSource

# Use:
from domain.ports import DataSource
```

## Testing

Run the import test:
```bash
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

## Running the CLI

Now you can run the optimizer:

```bash
cd /Users/jack/Projects/simple-lstm
python3 src/cli/optimize_cli.py --max-trials 5 --verbose
```

Or with full path:
```bash
python3 /Users/jack/Projects/simple-lstm/src/cli/optimize_cli.py --max-trials 5 --verbose
```

## Architecture Principle

This import structure follows the **hexagonal architecture** principle:
- **CLI layer** (entry point) is responsible for wiring dependencies
- **All modules** use absolute imports from the `src/` root
- **No circular dependencies** because imports flow in one direction
- **Easy to test** because imports are explicit and can be mocked

## Future Improvements

If you want to make this even cleaner, you could:

1. Create a `__main__.py` in `src/cli/` to make it a package:
   ```bash
   python -m src.cli
   ```

2. Or create a setup.py to install the package:
   ```bash
   pip install -e .
   python -m simple_lstm.cli.optimize_cli
   ```

But the current approach works well and is simple to understand.
