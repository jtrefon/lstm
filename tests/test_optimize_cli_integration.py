import io
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from config.config_loader import ConfigLoader
from src.cli import optimize_cli


def _make_tiny_csv(path: Path, n: int = 200):
    start = datetime(2020, 1, 1)
    times = [start + timedelta(minutes=i) for i in range(n)]
    # Simple increasing signal
    values = [float(i) for i in range(n)]
    df = pd.DataFrame({
        'timestamp': times,
        'value': values,
    })
    df.to_csv(path, index=False)


def _write_yaml(p: Path, content: str):
    p.write_text(content)


@pytest.mark.timeout(60)
def test_optimize_cli_runs_on_tiny_grid(monkeypatch, tmp_path: Path):
    # Prepare tiny dataset
    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / 'tiny.csv'
    _make_tiny_csv(csv_path, n=300)

    # Prepare temp config files
    cfg_dir = tmp_path / 'config'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    lstm_cfg = cfg_dir / 'lstm_config.yaml'
    opt_cfg = cfg_dir / 'optimization_config.yaml'
    data_cfg = cfg_dir / 'data_config.yaml'

    _write_yaml(lstm_cfg, f"""
model:
  input_size: 1
  sequence_stride: 1

model_defaults:
  sequence_length: 5
  learning_rate: 0.001
  batch_size: 8
  units: 8
  layers: 1
  dropout: 0.0

use_config_only: true

training:
  max_epochs: 2
  early_stop_patience: 1
  log_interval_batches: 100
  log_interval_seconds: 10.0
  max_batches_per_epoch: null
  num_workers: 0
  grad_clip_norm: null

learning_rate_scheduler:
  factor: 0.5
  patience: 1
  min_lr: 1e-5

optimization:
  train_window: null
  val_window: null

forecast:
  plot_max_points: 100

reproducibility:
  seed: 123
  deterministic: true
""")

    results_csv = str(tmp_path / 'results.csv')
    state_json = str(tmp_path / 'state.json')
    best_json = str(tmp_path / 'best.json')

    _write_yaml(opt_cfg, f"""
grid_search:
  sequence_length: [5]
  learning_rate: [0.001]
  batch_size: [8]
  units: [8]
  layers: [1]
  dropout: [0.0]
  samples_cap: 1
  data_samples_cap: 200
  data_cap_from: end

persistence:
  state_file: {state_json}
  results_file: {results_csv}
  best_params_file: {best_json}

logging:
  verbose: false
  save_interval: 1
""")

    _write_yaml(data_cfg, f"""
data_source:
  type: csv
  path: {csv_path}
  target_column: value

splitting:
  train_ratio: 0.8
  validation_ratio: 0.2
  test_ratio: 0.0

preprocessing:
  selected_columns: [value]
  handle_missing: drop
  sort_by_index: true
  outlier_method: none
  outlier_threshold: 1.5
""")

    # Monkeypatch config path resolver to use our temp configs
    def _fake_get_config_path(filename: str) -> str:
        mapping = {
            'lstm_config.yaml': str(lstm_cfg),
            'optimization_config.yaml': str(opt_cfg),
            'data_config.yaml': str(data_cfg),
        }
        return mapping[filename]

    monkeypatch.setattr(ConfigLoader, 'get_config_path', staticmethod(_fake_get_config_path))

    # Monkeypatch argv
    monkeypatch.setenv('PYTHONWARNINGS', 'ignore')
    monkeypatch.setattr(sys, 'argv', ['optimize_cli.py', '--max-trials', '1'])

    # Run CLI main; should not raise and should produce outputs
    optimize_cli.main()

    assert Path(results_csv).exists()
    assert Path(state_json).exists()
    assert Path(best_json).exists()
