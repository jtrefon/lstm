# Production Release Tracker

This checklist tracks the implementation of the production readiness plan.

## Must Fix Before Launch
- [x] Fix OutlierHandler tuple return on std==0 (predict unpack bug)
- [x] Make GridSearchResponse.summary Optional
- [x] Add reproducibility support (seed + deterministic mode) in trainer
- [x] Replace print statements with Python logging in trainer and CLIs
- [x] Add pinned dependencies (requirements.txt)
- [x] Add GitHub Actions CI (pytest + coverage)
- [x] Improve JSON portability in search state (avoid Infinity)
- [x] Switch model package hashing to SHA-256

## High-Value Improvements
- [x] Centralize device selection utility (infrastructure/torch/device.py) and reuse
- [x] Consolidate dropout/layers validation to model factory (removed duplicate in train_cli)
- [x] Use ConfigLoader.get_config_path everywhere (remove local helpers)
- [x] Add DataLoader num_workers and optional gradient clipping (configurable)

## Tests (Initial Coverage)
- [x] OutlierHandler edge cases
- [x] SequenceBuilder stride logic
- [ ] SearchStateRepository save/load portability and corruption recovery
- [ ] ResultsRepository header + append integrity
- [ ] GridSearchService resume and best-params update
- [ ] Minimal integration test for optimize_cli on tiny grid

## Remaining (Post-Launch Nice-to-Have)
- [ ] Pre-commit hooks (black/ruff/mypy)
- [ ] README with usage examples and config schema
- [ ] Optional: metrics/experiment tracking (MLflow)

## Developer Notes
- Reproducibility settings can be added to config/lstm_config.yaml:
  
  ```yaml
  reproducibility:
    seed: 42
    deterministic: true
  training:
    max_epochs: 500
    early_stop_patience: 10
    log_interval_batches: 200
    log_interval_seconds: 30.0
    max_batches_per_epoch: null
    num_workers: 0
    grad_clip_norm: null
  ```

- Run tests locally:
  
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  pytest -q --cov=src --cov-report=term-missing
  ```

- CI workflow: .github/workflows/ci.yml

