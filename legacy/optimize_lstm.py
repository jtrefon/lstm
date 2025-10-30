import os
import json
import csv
import itertools
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from data_loader import DataLoader as CSVDataLoader
from train_lstm import LSTMConfig, LSTMValidator

# Output paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_PARAMS_JSON = os.path.join(MODELS_DIR, 'lstm_best_params.json')
RESULTS_CSV = os.path.join(MODELS_DIR, 'lstm_grid_results.csv')
STATE_JSON = os.path.join(MODELS_DIR, 'lstm_search_state.json')

os.makedirs(MODELS_DIR, exist_ok=True)


def default_grid_from_config(cfg: LSTMConfig) -> Dict[str, List[Any]]:
    # Construct a discrete grid from ranges following our stronger defaults
    seqs = [64, 128, 256]
    lrs = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4]
    bsz = [64, 128, 256]
    units = [256, 512, 768]
    layers = [2, 3, 4]
    drop = [0.1, 0.2, 0.3]
    return {
        'sequence_length': seqs,
        'learning_rate': lrs,
        'batch_size': bsz,
        'units': units,
        'layers': layers,
        'dropout': drop,
    }


def iter_param_grid(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON, 'r') as f:
            return json.load(f)
    return {
        'completed': [],  # list of param dicts serialized to sorted tuples
        'best_params': None,
        'best_loss': float('inf'),
    }


def save_state(state: Dict[str, Any]):
    with open(STATE_JSON, 'w') as f:
        json.dump(state, f)


def serialize_params(p: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return sorted(p.items())


def append_csv_header_if_needed():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sequence_length','learning_rate','batch_size','units','layers','dropout','val_loss','val_rmse','val_mae','seconds'])


def append_csv_row(params: Dict[str, Any], val_loss: float, val_rmse: float, val_mae: float, seconds: float):
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            params['sequence_length'], params['learning_rate'], params['batch_size'],
            params['units'], params['layers'], params['dropout'],
            f"{val_loss:.6f}", f"{val_rmse:.6f}", f"{val_mae:.6f}", f"{seconds:.1f}"
        ])


def save_best(params: Dict[str, Any], loss: float):
    payload = {
        'best_params': params,
        'best_loss': loss,
        'timestamp': time.time()
    }
    with open(BEST_PARAMS_JSON, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    print('Using separate LSTM grid search script with resume and reporting.')
    dl = CSVDataLoader()
    cfg = LSTMConfig()

    # Use small windows for validation compute to keep runs tractable
    print(f"Optimization windows: train={cfg.optimization_train_window} val={cfg.optimization_val_window}")

    # Build grid
    grid = default_grid_from_config(cfg)
    total = 1
    for v in grid.values():
        total *= len(v)
    print(f"Total combinations: {total}")

    # Load resume state
    state = load_state()
    completed = set(tuple(serialize_params(p)) for p in state.get('completed', []))
    best_params = state.get('best_params', None)
    best_loss = state.get('best_loss', float('inf'))

    append_csv_header_if_needed()

    # Prepare validator datasets
    train_series = dl.get_train_data()[cfg.target_column]
    val_series = dl.get_validation_data()[cfg.target_column]
    validator = LSTMValidator(train_series, val_series, cfg)

    tried = len(completed)
    for params in iter_param_grid(grid):
        key = tuple(serialize_params(params))
        if key in completed:
            continue
        tried += 1
        print(f"[{tried}/{total}] Testing params: {params}")
        t0 = time.time()
        # validator.validate returns best_val_loss; we also print rmse/mae inside
        val_loss = validator.validate(params)
        seconds = time.time() - t0
        print(f"    → val_loss={val_loss:.6f} in {seconds:.1f}s")

        # Append CSV row (rmse/mae printed in logs only; for CSV we store loss)
        append_csv_row(params, val_loss, np.nan, np.nan, seconds)

        # Update best and state
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params
            save_best(best_params, best_loss)
            print(f"    ✓ New best: loss={best_loss:.6f} params={best_params}")

        state['completed'].append(dict(key))
        state['best_params'] = best_params
        state['best_loss'] = best_loss
        save_state(state)

    print('Search complete.')
    if best_params is not None:
        print(f"Best params: {best_params} with val_loss={best_loss:.6f}")
        save_best(best_params, best_loss)
    else:
        print('No successful trials.')


if __name__ == '__main__':
    main()
