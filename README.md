# simple-lstm

A clean, production-ready LSTM forecasting pipeline built on a hexagonal architecture (domain, application, infrastructure, CLI, config). Includes reproducibility controls, structured logging, persistence (JSON/CSV, model packages), unit tests, and GitHub Actions CI.

## Requirements

- Python 3.11+ recommended
- macOS/Unix
- Optional GPU: CUDA (Linux) or Metal/MPS (macOS) supported by PyTorch

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

- Configs live in `config/`:
  - `lstm_config.yaml`: model/training settings, reproducibility, optimization windows
  - `optimization_config.yaml`: grid search space and persistence
  - `data_config.yaml`: data source, splitting, preprocessing
- You can also add reproducibility:

```yaml
reproducibility:
  seed: 42
  deterministic: true
training:
  num_workers: 0
  grad_clip_norm: null
```

## Commands

- Optimize (grid search):

```bash
python -m src.cli.optimize_cli --verbose --max-trials 10
```

- Train single model (using `use_config_only` precedence and/or best_params.json):

```bash
python -m src.cli.train_cli --verbose
```

- Predict with a saved package:

```bash
python -m src.cli.predict_cli --verbose
```

- Analyze grid search results:

```bash
python -m src.cli.analyze_results_cli --results ./results/results.csv
```

## Model Packages

- Saved under `models/TS_HASH_LOSS` with:
  - `model.pt` (state_dict)
  - `scaler.pkl` (MinMaxScaler)
  - `params.json` (hyperparameters)
  - `meta.json` (metrics and metadata)

## Testing & CI

```bash
pytest -q --cov=src --cov-report=term-missing
```

- CI workflow at `.github/workflows/ci.yml`.

## Pre-commit Hooks

1) Install pre-commit once:

```bash
pip install pre-commit
pre-commit install
```

2) Run on all files:

```bash
pre-commit run --all-files
```

Hooks configured in `.pre-commit-config.yaml` (formatting/lint).

## Notes

- For deterministic runs, set `reproducibility.seed` and `reproducibility.deterministic: true` in `lstm_config.yaml`.
- Large datasets: tune `training.num_workers` and consider `optimization.train_window`/`val_window` for faster search.
