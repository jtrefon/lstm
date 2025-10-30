"""LSTM training loop - PyTorch-based implementation."""
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config.config_loader import LSTMConfig, TrainingConfig, LearningRateSchedulerConfig, PreprocessingConfig
from domain.models import ParameterSet, ValidationMetrics
from domain.ports import LSTMValidator, SequenceBuilder
from infrastructure.data.outlier_handler import OutlierHandler


# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print(f"Using device: {DEVICE}")
else:
    DEVICE = torch.device('cpu')
    print(f"Using device: {DEVICE}")


class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting."""

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1, dropout: float = 0.0):
        """Initialize LSTM model."""
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


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

        # Preprocess once per run: sorting, missing handling, outlier handling, scaling
        combined = pd.concat([self.train_series, self.val_series])
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
        values = combined.values.flatten()

        if self.preprocessing_config and self.preprocessing_config.outlier_method != 'none':
            values, n_clipped = OutlierHandler.handle_outliers(
                values,
                method=self.preprocessing_config.outlier_method,
                threshold=self.preprocessing_config.outlier_threshold,
            )
            if n_clipped > 0 and self._log_preprocessing:
                print(
                    f"Outlier handling: clipped {n_clipped} points using "
                    f"{self.preprocessing_config.outlier_method} "
                    f"(threshold={self.preprocessing_config.outlier_threshold})"
                )

        scaler = MinMaxScaler()
        scaled_all = scaler.fit_transform(values.reshape(-1, 1))
        split = len(self.train_series)
        self._train_scaled = scaled_all[:split]
        self._val_scaled = scaled_all[split:]

    def validate(self, parameters: ParameterSet) -> ValidationMetrics:
        """Validate LSTM with given parameters."""
        try:
            start_time = time.time()
            
            # Use precomputed scaled arrays
            train_scaled = self._train_scaled
            val_scaled = self._val_scaled

            # Build sequences
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

            if len(X_train) == 0 or len(X_val) == 0:
                if self._log_preprocessing:
                    print(
                        f"No sequences built: train_len={len(self._train_scaled)}, val_len={len(self._val_scaled)}, "
                        f"seq_len={parameters.sequence_length}, stride={self.lstm_config.model.sequence_stride}"
                    )
                return ValidationMetrics(
                    val_loss=float('inf'),
                    val_rmse=float('inf'),
                    val_mae=float('inf'),
                    duration_seconds=time.time() - start_time,
                )

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
            X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
            y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
            X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            pin = (DEVICE.type == 'cuda')
            train_loader = DataLoader(train_dataset, batch_size=parameters.batch_size, shuffle=True, pin_memory=pin)
            val_loader = DataLoader(val_dataset, batch_size=parameters.batch_size, shuffle=False, pin_memory=pin)

            # Create model
            model = LSTMModel(
                hidden_size=parameters.units,
                num_layers=parameters.layers,
                dropout=parameters.dropout,
            ).to(DEVICE)

            # Training
            metrics = self._train_model(
                model,
                train_loader,
                val_loader,
                parameters.learning_rate,
            )

            metrics = ValidationMetrics(
                val_loss=float(metrics['val_loss']),
                val_rmse=float(metrics['val_rmse']),
                val_mae=float(metrics['val_mae']),
                duration_seconds=float(time.time() - start_time),
            )

            return metrics

        except Exception as e:
            import traceback
            print(f"Validation failed: {e}")
            traceback.print_exc()
            return ValidationMetrics(
                val_loss=float('inf'),
                val_rmse=float('inf'),
                val_mae=float('inf'),
                duration_seconds=time.time() - start_time,
            )

    def _train_model(
        self,
        model: LSTMModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
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
        
        # Ensure all are Python floats, not numpy types
        best_val_loss = float(best_val_loss)
        best_val_rmse = float(best_val_rmse)
        best_val_mae = float(best_val_mae)

        model.train()
        for epoch in range(self.lstm_config.training.max_epochs):
            # Training phase
            running_loss = 0.0
            batches_seen = 0

            for batch_idx, (X_batch, y_batch) in enumerate(train_loader, start=1):
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batches_seen = batch_idx

                if self.lstm_config.training.max_batches_per_epoch and batch_idx >= self.lstm_config.training.max_batches_per_epoch:
                    break

            # Validation phase
            model.eval()
            val_losses = []
            val_errors = []
            with torch.no_grad():
                for v_idx, (X_batch, y_batch) in enumerate(val_loader, start=1):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
                    preds = outputs.squeeze(-1).cpu().numpy()
                    targets = y_batch.squeeze(-1).cpu().numpy()
                    val_errors.append(preds - targets)

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
            scheduler.step(mean_val_loss)

            # Early stopping
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.lstm_config.training.early_stop_patience:
                    break

            model.train()

        return {
            'val_loss': best_val_loss,
            'val_rmse': best_val_rmse,
            'val_mae': best_val_mae,
        }
