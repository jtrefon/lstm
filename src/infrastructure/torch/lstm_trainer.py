"""LSTM training loop - PyTorch-based implementation."""
import time
from typing import Tuple
import copy
import logging
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from config.config_loader import LSTMConfig, TrainingConfig, LearningRateSchedulerConfig, PreprocessingConfig
from domain.models import ParameterSet, ValidationMetrics
from domain.ports import LSTMValidator, SequenceBuilder
from infrastructure.data.outlier_handler import OutlierHandler
from infrastructure.torch.FactoryPattern_LSTMModelFactory import build_lstm
from infrastructure.torch.device import get_device

logger = logging.getLogger(__name__)


# Device selection
DEVICE = get_device()


class PyTorchLSTMValidator(LSTMValidator):
    """Validates LSTM hyperparameters using PyTorch."""

    def __init__(
        self,
        train_series: pd.Series,
        val_series: pd.Series,
        sequence_builder: SequenceBuilder,
        lstm_config: LSTMConfig,
        preprocessing_config: PreprocessingConfig = None,
        log_preprocessing: bool = False,
    ):
        """Initialize validator.
        
        Args:
            train_series: Training time series
            val_series: Validation time series
            sequence_builder: Sequence builder adapter
            lstm_config: LSTM configuration
            preprocessing_config: Data preprocessing config (outlier handling, etc.)
        """
        self.train_series = train_series
        self.val_series = val_series
        self.sequence_builder = sequence_builder
        self.lstm_config = lstm_config
        self.preprocessing_config = preprocessing_config
        self._log_preprocessing = log_preprocessing
        self._last_model = None
        self._scaler = None
        self._last_val_metrics_orig = {}
        self._prep_stats = {
            'raw_len': int(len(self.train_series) + len(self.val_series)),
            'na_removed': 0,
            'outliers_clipped': 0,
        }
        self._seq_stats = {'train_sequences': 0, 'val_sequences': 0}

        # Apply reproducibility settings (seeding/determinism)
        self._apply_reproducibility()

        # Preprocess once per run: sorting, missing handling, outlier handling, scaling
        combined_raw = pd.concat([self.train_series, self.val_series])
        na_before = int(combined_raw.isna().sum())
        combined = combined_raw
        # Sort if requested
        if self.preprocessing_config and self.preprocessing_config.sort_by_index:
            combined = combined.sort_index()
        # Handle missing values
        if self.preprocessing_config:
            hm = self.preprocessing_config.handle_missing
            if hm == 'forward_fill':
                combined = combined.ffill()
                # Drop any remaining NaNs (e.g., at the start)
                combined = combined.dropna()
            elif hm == 'interpolate':
                try:
                    combined = combined.interpolate(method='time')
                except Exception:
                    combined = combined.interpolate()
                combined = combined.ffill().bfill()
            elif hm == 'drop':
                combined = combined.dropna()
        # Final guard: if any NaNs remain, drop them
        combined = combined.dropna()
        self._prep_stats['na_removed'] = na_before  # everything removed by cleaning
        values = combined.values.flatten()

        if self.preprocessing_config and self.preprocessing_config.outlier_method != 'none':
            if self._log_preprocessing:
                logger.info(
                    f"Outlier handling: detecting anomalies with {self.preprocessing_config.outlier_method}..."
                )
            values, n_clipped = OutlierHandler.handle_outliers(
                values,
                method=self.preprocessing_config.outlier_method,
                threshold=self.preprocessing_config.outlier_threshold,
                window=getattr(self.preprocessing_config, 'outlier_window', None),
            )
            self._prep_stats['outliers_clipped'] = int(n_clipped)
            if n_clipped > 0 and self._log_preprocessing:
                logger.info(
                    f"  Clipped {n_clipped} points using {self.preprocessing_config.outlier_method} "
                    f"(threshold={self.preprocessing_config.outlier_threshold})"
                )

        # Final cleaned length after preprocessing/outlier handling
        self._prep_stats['cleaned_len'] = int(len(values))

        if self._log_preprocessing:
            logger.info(f"Scaling {len(values)} samples...")

        # Select scaler based on config (default: standard)
        scaler_kind = 'standard'
        margin_pct = 0.0
        try:
            if self.preprocessing_config is not None:
                scaler_kind = str(getattr(self.preprocessing_config, 'scaler', 'standard')).lower()
                margin_pct = float(getattr(self.preprocessing_config, 'minmax_margin_pct', 0.0) or 0.0)
        except Exception:
            scaler_kind = 'standard'
            margin_pct = 0.0

        if scaler_kind == 'minmax':
            scaler = MinMaxScaler()
            scaled_all = scaler.fit_transform(values.reshape(-1, 1))
            # Optionally extend bounds by margin to reduce OOD at inference
            if margin_pct > 0.0:
                try:
                    fr_min, fr_max = scaler.feature_range
                except Exception:
                    fr_min, fr_max = (0.0, 1.0)
                vmin = float(scaler.data_min_[0])
                vmax = float(scaler.data_max_[0])
                span = vmax - vmin if vmax > vmin else 1.0
                vmin2 = vmin - margin_pct * span
                vmax2 = vmax + margin_pct * span
                data_range = vmax2 - vmin2
                scale = (fr_max - fr_min) / data_range
                min_shift = fr_min - vmin2 * scale
                scaler.data_min_ = np.array([vmin2], dtype=float)
                scaler.data_max_ = np.array([vmax2], dtype=float)
                scaler.data_range_ = np.array([data_range], dtype=float)
                scaler.scale_ = np.array([scale], dtype=float)
                scaler.min_ = np.array([min_shift], dtype=float)
                if self._log_preprocessing:
                    logger.info(
                        f"MinMaxScaler with margin: {margin_pct:.2%} → bounds=({vmin2:.6f}, {vmax2:.6f})"
                    )
        elif scaler_kind == 'robust':
            scaler = RobustScaler()
            scaled_all = scaler.fit_transform(values.reshape(-1, 1))
        else:
            scaler = StandardScaler()
            scaled_all = scaler.fit_transform(values.reshape(-1, 1))
        if self._log_preprocessing:
            logger.info("Scaling complete.")
        self._scaler = scaler
        split = len(self.train_series)
        self._train_scaled = scaled_all[:split]
        self._val_scaled = scaled_all[split:]

    def _apply_reproducibility(self) -> None:
        """Apply reproducibility settings from config (seeding and deterministic mode)."""
        try:
            repro = getattr(self.lstm_config, 'reproducibility', None)
            if repro is None:
                return
            seed = repro.seed
            deterministic = bool(getattr(repro, 'deterministic', False))
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                try:
                    torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass
                logger.info(f"Applied reproducibility seed: {seed}")
            if deterministic:
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
                logger.info("Deterministic mode enabled (may impact performance)")
        except Exception:
            # Never fail initialization due to reproducibility settings
            logger.exception("Failed to apply reproducibility settings")

    def get_preprocessing_stats(self) -> dict:
        return dict(self._prep_stats)

    def validate(self, parameters: ParameterSet) -> ValidationMetrics:
        """Validate LSTM with given parameters."""
        try:
            start_time = time.time()
            
            # Use precomputed scaled arrays
            train_scaled = self._train_scaled
            val_scaled = self._val_scaled

            # Build sequences
            if self._log_preprocessing:
                logger.info(
                    f"Building sequences (seq_len={parameters.sequence_length}, stride={self.lstm_config.model.sequence_stride})..."
                )
            X_train, y_train = self.sequence_builder.build(
                train_scaled,
                parameters.sequence_length,
                stride=self.lstm_config.model.sequence_stride,
            )
            X_val, y_val = self.sequence_builder.build(
                val_scaled,
                parameters.sequence_length,
                stride=self.lstm_config.model.sequence_stride,
            )
            if self._log_preprocessing:
                logger.info(f"Sequences built: train={len(X_train)} val={len(X_val)}")

            if len(X_train) == 0 or len(X_val) == 0:
                if self._log_preprocessing:
                    logger.warning(
                        f"No sequences built: train_len={len(self._train_scaled)}, val_len={len(self._val_scaled)}, "
                        f"seq_len={parameters.sequence_length}, stride={self.lstm_config.model.sequence_stride}"
                    )
                return ValidationMetrics(
                    val_loss=float('inf'),
                    val_rmse=float('inf'),
                    val_mae=float('inf'),
                    duration_seconds=time.time() - start_time,
                )

            # Track sequence counts for transparency
            try:
                self._seq_stats = {
                    'train_sequences': int(len(X_train)),
                    'val_sequences': int(len(X_val)),
                }
            except Exception:
                self._seq_stats = {'train_sequences': 0, 'val_sequences': 0}

            # Optional target: delta of last step
            target_is_delta = False
            try:
                target_is_delta = str(getattr(self.lstm_config.model, 'target_mode', 'value')).lower() == 'delta'
            except Exception:
                target_is_delta = False

            if target_is_delta:
                try:
                    y_train = (y_train - X_train[:, -1]).astype(np.float32)
                    y_val = (y_val - X_val[:, -1]).astype(np.float32)
                except Exception:
                    pass

            # Ensure correct shapes: (batch, seq_len, 1) for inputs, (batch, 1) for targets
            if X_train.ndim == 2:
                X_train = np.expand_dims(X_train, -1)
            if X_val.ndim == 2:
                X_val = np.expand_dims(X_val, -1)
            if y_train.ndim == 1:
                y_train = np.expand_dims(y_train, -1)
            if y_val.ndim == 1:
                y_val = np.expand_dims(y_val, -1)

            # Convert to tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            pin = (DEVICE.type == 'cuda')
            # Use seeded generator for determinism if provided
            generator = None
            repro = getattr(self.lstm_config, 'reproducibility', None)
            if repro and repro.seed is not None:
                try:
                    generator = torch.Generator(device='cpu')
                    generator.manual_seed(int(repro.seed))
                except Exception:
                    generator = None
            num_workers = 0
            try:
                nw = getattr(self.lstm_config.training, 'num_workers', None)
                if nw is not None:
                    num_workers = int(nw)
            except Exception:
                num_workers = 0
            train_loader = DataLoader(
                train_dataset,
                batch_size=parameters.batch_size,
                shuffle=True,
                pin_memory=pin,
                generator=generator,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=parameters.batch_size,
                shuffle=False,
                pin_memory=pin,
                num_workers=num_workers,
            )

            # Create model via factory (DRY, uses input_size from config)
            model = build_lstm(self.lstm_config, parameters, DEVICE)

            # Training
            metrics = self._train_model(
                model,
                train_loader,
                val_loader,
                parameters.learning_rate,
                target_is_delta=target_is_delta,
            )

            metrics = ValidationMetrics(
                val_loss=float(metrics['val_loss']),
                val_rmse=float(metrics['val_rmse']),
                val_mae=float(metrics['val_mae']),
                duration_seconds=float(time.time() - start_time),
            )

            self._last_model = model
            return metrics

        except Exception as e:
            logger.exception(f"Validation failed: {e}")
            return ValidationMetrics(
                val_loss=float('inf'),
                val_rmse=float('inf'),
                val_mae=float('inf'),
                duration_seconds=time.time() - start_time,
            )

    def get_last_model(self):
        return self._last_model

    def get_scaler(self):
        return self._scaler

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        target_is_delta: bool = False,
    ) -> dict:
        """Train model and return best validation metrics."""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(self.lstm_config.lr_scheduler.factor),
            patience=int(self.lstm_config.lr_scheduler.patience),
            min_lr=float(self.lstm_config.lr_scheduler.min_lr),
        )

        best_val_loss = float('inf')
        best_val_rmse = float('inf')
        best_val_mae = float('inf')
        patience_counter = 0
        best_state = None
        best_epoch = -1
        
        # Ensure all are Python floats, not numpy types
        best_val_loss = float(best_val_loss)
        best_val_rmse = float(best_val_rmse)
        best_val_mae = float(best_val_mae)

        model.train()
        for epoch in range(self.lstm_config.training.max_epochs):
            # Training phase
            running_loss = 0.0
            batches_seen = 0
            total_batches = None
            try:
                total_batches = len(train_loader)
            except Exception:
                total_batches = None
            sec_interval = 0.0
            batch_interval = 0
            try:
                sec_interval = float(getattr(self.lstm_config.training, 'log_interval_seconds', 0.0) or 0.0)
            except Exception:
                sec_interval = 0.0
            try:
                batch_interval = int(getattr(self.lstm_config.training, 'log_interval_batches', 0) or 0)
            except Exception:
                batch_interval = 0
            last_log_t = time.time()
            is_tty = False
            try:
                is_tty = sys.stdout.isatty()
            except Exception:
                is_tty = False
            progress_len = 0

            for batch_idx, (X_batch, y_batch) in enumerate(train_loader, start=1):
                X_batch = X_batch.to(DEVICE, non_blocking=train_loader.pin_memory)
                y_batch = y_batch.to(DEVICE, non_blocking=train_loader.pin_memory)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                # Optional gradient clipping
                gc = getattr(self.lstm_config.training, 'grad_clip_norm', None)
                if gc is not None:
                    try:
                        clip_grad_norm_(model.parameters(), max_norm=float(gc))
                    except Exception:
                        pass
                optimizer.step()

                running_loss += loss.item()
                batches_seen = batch_idx

                if self._log_preprocessing and (
                    (batch_interval and (batch_idx % batch_interval == 0)) or
                    (sec_interval and ((time.time() - last_log_t) >= sec_interval))
                ):
                    try:
                        mean_so_far = float(running_loss / batches_seen) if batches_seen else float('inf')
                    except Exception:
                        mean_so_far = float('inf')
                    if is_tty:
                        if total_batches:
                            msg = f"\rEpoch {epoch + 1:03d} [{batch_idx}/{total_batches}] train_loss={mean_so_far:.10g}"
                        else:
                            msg = f"\rEpoch {epoch + 1:03d} [batch {batch_idx}] train_loss={mean_so_far:.10g}"
                        pad = max(0, progress_len - len(msg))
                        try:
                            sys.stdout.write(msg + (" " * pad))
                            sys.stdout.flush()
                            progress_len = len(msg)
                        except Exception:
                            pass
                    # In non-TTY (e.g., CI) avoid noisy logs; rely on end-of-epoch summary
                    last_log_t = time.time()

                if self.lstm_config.training.max_batches_per_epoch and batch_idx >= self.lstm_config.training.max_batches_per_epoch:
                    break

            # Validation phase
            model.eval()
            val_losses = []
            val_errors = []
            val_preds_batches = []
            val_targets_batches = []
            with torch.no_grad():
                for v_idx, (X_batch, y_batch) in enumerate(val_loader, start=1):
                    X_batch = X_batch.to(DEVICE, non_blocking=val_loader.pin_memory)
                    y_batch = y_batch.to(DEVICE, non_blocking=val_loader.pin_memory)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
                    # For metrics, compare absolute predictions in scaled space
                    if target_is_delta:
                        last_x = X_batch[:, -1, 0].unsqueeze(-1)  # (B,1)
                        preds_abs = (outputs + last_x).squeeze(-1).cpu().numpy()
                        targets_abs = (y_batch + last_x).squeeze(-1).cpu().numpy()
                    else:
                        preds_abs = outputs.squeeze(-1).cpu().numpy()
                        targets_abs = y_batch.squeeze(-1).cpu().numpy()
                    val_preds_batches.append(np.atleast_1d(preds_abs))
                    val_targets_batches.append(np.atleast_1d(targets_abs))
                    val_errors.append(preds_abs - targets_abs)

                    if self.lstm_config.training.max_batches_per_epoch and v_idx >= self.lstm_config.training.max_batches_per_epoch:
                        break

            mean_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')

            if val_errors:
                val_errors_np = np.concatenate([np.atleast_1d(err) for err in val_errors])
                val_rmse = float(np.sqrt(np.mean(np.square(val_errors_np))))
                val_mae = float(np.mean(np.abs(val_errors_np)))
            else:
                val_rmse = float('inf')
                val_mae = float('inf')

            # Ensure mean_val_loss is a Python float for scheduler
            mean_val_loss = float(mean_val_loss)
            prev_lr = float(optimizer.param_groups[0]['lr'])
            scheduler.step(mean_val_loss)
            new_lr = float(optimizer.param_groups[0]['lr'])

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                patience_counter = 0
                try:
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                except Exception:
                    best_state = None
                # Compute original-unit metrics for the best epoch
                try:
                    if self._scaler is not None and val_preds_batches and val_targets_batches:
                        preds_all = np.concatenate(val_preds_batches).reshape(-1, 1)
                        targets_all = np.concatenate(val_targets_batches).reshape(-1, 1)
                        y_pred_orig = self._scaler.inverse_transform(preds_all).reshape(-1)
                        y_true_orig = self._scaler.inverse_transform(targets_all).reshape(-1)
                        err = y_pred_orig - y_true_orig
                        best_val_rmse_orig = float(np.sqrt(np.mean(np.square(err))))
                        best_val_mae_orig = float(np.mean(np.abs(err)))
                        self._last_val_metrics_orig = {
                            'val_rmse_orig': best_val_rmse_orig,
                            'val_mae_orig': best_val_mae_orig,
                        }
                except Exception:
                    pass
            else:
                patience_counter += 1
                if patience_counter >= self.lstm_config.training.early_stop_patience:
                    break

            # Epoch summary logging
            mean_train_loss = float(running_loss / batches_seen) if batches_seen else float('inf')
            if self._log_preprocessing:
                # Terminate any in-place progress line with a newline once per epoch
                if is_tty:
                    try:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    except Exception:
                        pass
                logger.info(
                    f"Epoch {epoch + 1:03d}: train_loss={mean_train_loss:.10g} | "
                    f"val_loss={mean_val_loss:.10g} (rmse={val_rmse:.10g}, mae={val_mae:.10g}) | "
                    f"lr={new_lr:.6f}{' (↓)' if new_lr < prev_lr else ''} | "
                    f"patience={patience_counter}/{self.lstm_config.training.early_stop_patience}"
                )

            model.train()

        if best_state is not None:
            try:
                model.load_state_dict(best_state)
            except Exception:
                pass

        result = {
            'val_loss': best_val_loss,
            'val_rmse': best_val_rmse,
            'val_mae': best_val_mae,
        }
        # Attach original-unit metrics if computed
        try:
            if self._last_val_metrics_orig:
                result.update(self._last_val_metrics_orig)
        except Exception:
            pass
        return result

    def get_sequence_stats(self) -> dict:
        return dict(self._seq_stats)

    def get_last_val_metrics_orig(self) -> dict:
        return dict(self._last_val_metrics_orig)
