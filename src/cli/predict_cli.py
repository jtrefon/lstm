import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from infrastructure.data.csv_data_source import CSVDataSource
from infrastructure.data.series_splitter import TimeSeriesSplitter
from infrastructure.data.sequence_builder import NumpySequenceBuilder
from infrastructure.data.outlier_handler import OutlierHandler
from infrastructure.persistence.model_repository import ModelPackageRepository
from infrastructure.torch.FactoryPattern_LSTMModelFactory import build_lstm
from infrastructure.torch.device import get_device


logger = logging.getLogger(__name__)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mse = float(np.mean((y_true - y_pred) ** 2)) if len(y_true) else float('inf')
    rmse = float(np.sqrt(mse)) if mse != float('inf') else float('inf')
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float('inf')
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0) if len(y_true) else float('inf')
    # R2
    denom = np.sum((y_true - np.mean(y_true)) ** 2) if len(y_true) else 0.0
    r2 = float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom)) if denom > 0 else 0.0
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape_pct': mape,
        'r2': r2,
    }


def _resolve_package_dir(path: str) -> str:
    required = ['model.pt', 'scaler.pkl', 'params.json']
    if all(os.path.exists(os.path.join(path, f)) for f in required):
        return path
    if not os.path.isdir(path):
        raise SystemExit(f"Path not found or not a directory: {path}")
    candidates = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and all(os.path.exists(os.path.join(full, f)) for f in required):
            try:
                with open(os.path.join(full, 'meta.json'), 'r') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            try:
                mtime = os.path.getmtime(os.path.join(full, 'model.pt'))
            except Exception:
                mtime = 0.0
            candidates.append((mtime, full, meta))
    if not candidates:
        raise SystemExit(f"No valid model packages found under: {path}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) == 1:
        return candidates[0][1]
    logger.info("\nAvailable model packages:")
    for i, (_, d, meta) in enumerate(candidates, start=1):
        metrics = meta.get('metrics', {}) if isinstance(meta, dict) else {}
        v = metrics.get('val_loss', 'na')
        logger.info(f"  [{i}] {os.path.basename(d)}  val_loss={v}")
    while True:
        choice = input(f"Select package [1-{len(candidates)}] (default 1): ").strip()
        if choice == "":
            return candidates[0][1]
        try:
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1][1]
        except ValueError:
            pass
        logger.warning("Invalid selection. Try again.")


def main() -> None:
    p = argparse.ArgumentParser(description='Predict using a saved LSTM model package on the test split')
    p.add_argument('--package-dir', default='./models/', type=str, help='Directory containing model.pt, scaler.pkl, params.json')
    p.add_argument('--verbose', action='store_true', default=True)
    args = p.parse_args()

    logging.basicConfig(
        level=(logging.INFO if args.verbose else logging.WARNING),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    )

    lstm_config = ConfigLoader.load_lstm_config(ConfigLoader.get_config_path('lstm_config.yaml'))
    data_config = ConfigLoader.load_data_config(ConfigLoader.get_config_path('data_config.yaml'))

    ds = CSVDataSource(
        filepath=data_config.data_source.path,
        target_column=data_config.data_source.target_column,
    )
    df = ds.load()
    series = df[data_config.data_source.target_column]

    splitter = TimeSeriesSplitter(
        train_ratio=data_config.splitting.train_ratio,
        validation_ratio=data_config.splitting.validation_ratio,
        test_ratio=data_config.splitting.test_ratio,
    )
    _, _, test_series = splitter.split(series)

    combined = test_series
    if data_config.preprocessing.sort_by_index:
        combined = combined.sort_index()
    hm = data_config.preprocessing.handle_missing
    if hm == 'forward_fill':
        combined = combined.ffill().dropna()
    elif hm == 'interpolate':
        try:
            combined = combined.interpolate(method='time')
        except Exception:
            combined = combined.interpolate()
        combined = combined.ffill().bfill()
    elif hm == 'drop':
        combined = combined.dropna()

    values = combined.values.flatten()
    if data_config.preprocessing.outlier_method != 'none':
        values, _ = OutlierHandler.handle_outliers(
            values,
            method=data_config.preprocessing.outlier_method,
            threshold=data_config.preprocessing.outlier_threshold,
        )

    pkg_dir = _resolve_package_dir(args.package_dir)
    pkg_repo = ModelPackageRepository()
    state_dict, scaler, params, saved_metrics = pkg_repo.load_package(pkg_dir)

    scaled = scaler.transform(values.reshape(-1, 1)).reshape(-1)

    builder = NumpySequenceBuilder()
    X_test, y_test = builder.build(
        scaled,
        sequence_length=int(params.sequence_length),
        stride=int(lstm_config.model.sequence_stride),
    )

    if len(X_test) == 0:
        logger.error('No test sequences could be built. Check sequence_length and test split size.')
        return

    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, -1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    device = get_device()
    model = build_lstm(lstm_config, params, device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds = model(X_test_t.to(device)).squeeze(-1).cpu().numpy()

    # Inverse scale back to original units
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)

    report = _metrics(y_true, y_pred)

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION REPORT")
    logger.info("=" * 80)
    logger.info(f"Model package: {pkg_dir}")
    logger.info(f"Saved val metrics: {saved_metrics}")
    logger.info(f"Test samples: {len(y_true)}")
    logger.info("Metrics:")
    logger.info(f"  RMSE:      {report['rmse']:.6f}")
    logger.info(f"  MAE:       {report['mae']:.6f}")
    logger.info(f"  MSE:       {report['mse']:.6f}")
    logger.info(f"  MAPE(%):   {report['mape_pct']:.2f}")
    logger.info(f"  R2:        {report['r2']:.6f}")

    k = min(10, len(y_true))
    sample_df = pd.DataFrame({
        'actual': y_true[:k],
        'predicted': y_pred[:k],
        'error': (y_pred - y_true)[:k],
    })
    logger.info("\nSample predictions (first {}):".format(k))
    logger.info("\n" + sample_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == '__main__':
    main()
