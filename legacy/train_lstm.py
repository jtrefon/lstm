import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from data_loader import DataLoader as CSVDataLoader
import time
import warnings
warnings.filterwarnings('ignore')

# Set device: prefer CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = 'cuda'
    print(f"Using device: {device} ({name})")
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: {device}")
else:
    device = torch.device('cpu')
    print(f"Using device: {device}")

@dataclass(frozen=True)
class LSTMConfig:
    """LSTM configuration for time series forecasting.
    
    For large datasets (>1M samples):
    - Tune only essential hyperparameters (sequence_length, learning_rate, batch_size, architecture)
    - Use early stopping instead of fixed epochs (stops when validation loss plateaus)
    - Use learning rate scheduling instead of fixed learning rate (reduces LR when loss plateaus)
    - This reduces grid search from 5000+ to ~200 combinations
    
    Hyperparameters to tune (grid search):
    - sequence_length: Historical context window (30-180 days)
    - learning_rate: Initial optimizer step size (0.0001-0.005)
    - batch_size: Gradient batch size (16-128)
    - units: LSTM hidden units (128-512)
    - layers: LSTM depth (1-3)
    - dropout: Regularization (0.0-0.3)
    
    Adaptive parameters (NOT tuned, handled automatically):
    - epochs: Determined by early stopping (no fixed limit)
    - learning_rate scheduling: ReduceLROnPlateau reduces LR when loss plateaus
    """
    target_column: str = 'close'
    
    # Sequence length range: How many past timesteps to use for prediction
    # Longer sequences capture long-term patterns but increase computation
    sequence_length_range: Tuple[int, int] = (64, 256)
    
    # Batch size range: Samples per gradient update
    # Smaller batches = noisier gradients but better generalization
    batch_size_range: Tuple[int, int] = (64, 256)
    
    # Learning rate range: Initial optimizer step size
    # Will be automatically reduced by ReduceLROnPlateau when loss plateaus
    learning_rate_range: Tuple[float, float] = (0.00005, 0.0008)
    
    # LSTM architecture parameters
    units_range: Tuple[int, int] = (256, 768)
    layers_range: Tuple[int, int] = (2, 4)
    dropout_range: Tuple[float, float] = (0.1, 0.35)
    
    # Early stopping: Stop training when validation loss doesn't improve
    early_stop_patience: int = 10
    
    # Learning rate scheduling: Reduce LR when validation loss plateaus
    # factor: Multiply LR by this when plateau detected
    # patience: Wait this many epochs before reducing
    # min_lr: Don't reduce below this value
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-6
    
    # Sequence stride for sequence creation (use >1 to downsample and reduce steps per epoch)
    sequence_stride: int = 1

    # Logging cadence: show progress every N batches or seconds (whichever comes first)
    log_interval_batches: int = 200
    log_interval_seconds: float = 30.0

    # Maximum epochs (safety limit, early stopping usually stops earlier)
    max_epochs: int = 500

    # Limit number of batches per epoch (for very large datasets). None = no cap.
    max_batches_per_epoch: Optional[int] = None

    # Fraction of combined (train+val) used as internal validation during final training
    final_val_fraction: float = 0.05
    
    # Optimization windows: Use larger windows for representative grid search
    # With 5.2M training samples, 10k samples = 0.2% (still manageable)
    optimization_train_window: Optional[int] = 10000
    optimization_val_window: Optional[int] = 1000
    plot_max_points: int = 500

@dataclass
class ParameterSearchStep:
    """Records a single hyperparameter combination and its validation performance."""
    params: Dict[str, Any]  # All hyperparameters: seq_len, lr, batch_size, epochs, units, layers, dropout
    val_loss: float
    train_loss: float

@dataclass
class ParameterSearchResult:
    best_params: Dict[str, Any]
    best_model: nn.Module
    best_loss: float
    steps: List[ParameterSearchStep]
    early_stopped: bool

@dataclass
class ForecastResult:
    forecast: pd.Series
    actual: pd.Series
    mse: float
    rmse: float
    mae: float
    valid_actual: np.ndarray
    valid_forecast: np.ndarray

class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting.
    
    Architecture:
    - LSTM layers for sequence processing
    - Fully connected layer for single-step prediction
    - Uses last hidden state to predict next value
    """
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.0):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features (default 1 for univariate)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate between layers (prevents overfitting)
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass: process sequence and predict next value.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length, stride: int = 1):
    xs, ys = [], []
    if stride < 1:
        stride = 1
    limit = len(data) - seq_length
    for i in range(0, max(0, limit), stride):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class LSTMValidator:
    """Encapsulates validation logic for LSTM.
    
    Trains LSTM models with different hyperparameters on train data and evaluates
    on validation data. Uses MinMaxScaler fit only on train+val to prevent data leakage.
    """

    def __init__(self, train_series: pd.Series, validation_series: pd.Series, config: LSTMConfig):
        """
        Initialize validator.
        
        Args:
            train_series: Training data for model fitting
            validation_series: Validation data for performance evaluation
            config: LSTMConfig with model and training parameters
            
        Raises:
            ValueError: If either series is empty
        """
        if len(train_series) == 0:
            raise ValueError("Training series is empty; cannot validate LSTM parameters.")
        if len(validation_series) == 0:
            raise ValueError("Validation series is empty; cannot validate LSTM parameters.")

        self._train_series = train_series
        self._validation_series = validation_series
        self.config = config

    def validate(self, params: Dict[str, Any]) -> float:
        """
        Validate LSTM model with given hyperparameters.
        
        Trains model on training data and evaluates on validation data.
        Scaler is fit only on train+val to prevent test data leakage.
        
        Args:
            params: Dict with hyperparameters:
                - sequence_length: Historical context window
                - learning_rate: Initial optimizer step size (auto-reduced by scheduler)
                - batch_size: Gradient batch size
                - units: LSTM hidden units
                - layers: LSTM depth
                - dropout: Regularization
                (Note: epochs removed - determined by early stopping)
            
        Returns:
            Mean validation loss, or inf if training fails
        """
        # Extract hyperparameters from params dict
        seq_length = params['sequence_length']
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        units = params['units']
        layers = params['layers']
        dropout = params['dropout']
        
        # Prepare data
        scaler = MinMaxScaler()
        combined = pd.concat([self._train_series, self._validation_series])
        scaled_data = scaler.fit_transform(combined.values.reshape(-1, 1))

        train_scaled = scaled_data[:len(self._train_series)]
        val_scaled = scaled_data[len(self._train_series):]

        X_train, y_train = create_sequences(train_scaled, seq_length, stride=self.config.sequence_stride)
        X_val, y_val = create_sequences(val_scaled, seq_length, stride=self.config.sequence_stride)

        if len(X_train) == 0 or len(X_val) == 0:
            return float('inf')

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        pin = (device.type == 'cuda')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)

        model = LSTMModel(hidden_size=units, num_layers=layers, dropout=dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler: Reduce LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.lr_scheduler_min_lr
        )

        best_val_loss = float('inf')
        best_val_mae = float('inf')
        best_val_rmse = float('inf')
        patience_counter = 0
        
        model.train()
        total_batches = len(train_loader)
        for epoch in range(self.config.max_epochs):
            # Training phase
            epoch_start = time.time()
            last_log_time = epoch_start
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

                time_elapsed = time.time() - last_log_time
                if (
                    batch_idx == 1
                    or batch_idx % self.config.log_interval_batches == 0
                    or time_elapsed >= self.config.log_interval_seconds
                    or batch_idx == total_batches
                ):
                    avg_loss = running_loss / batches_seen
                    progress_pct = (batch_idx / total_batches * 100) if total_batches else 0.0
                    print(
                        "\r"
                        f"      [train] epoch {epoch + 1:03d} batch {batch_idx}/{total_batches} "
                        f"({progress_pct:5.1f}%) loss={loss.item():.4f} "
                        f"avg={avg_loss:.4f} elapsed={time.time() - epoch_start:.1f}s",
                        end='',
                        flush=True,
                    )
                    last_log_time = time.time()

                # Optional cap on batches per epoch
                if self.config.max_batches_per_epoch is not None and batch_idx >= self.config.max_batches_per_epoch:
                    break

            mean_train_loss = running_loss / batches_seen if batches_seen > 0 else float('inf')
            if batches_seen:
                print()  # newline after the final batch log

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
                    if self.config.max_batches_per_epoch is not None and v_idx >= self.config.max_batches_per_epoch:
                        break

            mean_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
            if val_errors:
                val_errors_np = np.concatenate([np.atleast_1d(err) for err in val_errors])
                val_rmse = float(np.sqrt(np.mean(np.square(val_errors_np))))
                val_mae = float(np.mean(np.abs(val_errors_np)))
            else:
                val_rmse = float('inf')
                val_mae = float('inf')

            prev_lr = optimizer.param_groups[0]['lr']
            # Learning rate scheduling: Reduce LR if loss plateaus
            scheduler.step(mean_val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            lr_msg = f"lr={new_lr:.6f}"
            if new_lr < prev_lr:
                lr_msg += f" (↓ from {prev_lr:.6f})"

            print(
                f"    Epoch {epoch + 1:03d}: train_loss={mean_train_loss:.4f} | "
                f"val_loss={mean_val_loss:.4f} (rmse={val_rmse:.4f}, mae={val_mae:.4f}) | "
                f"{lr_msg} | patience={patience_counter}/{self.config.early_stop_patience}"
            )

            # Early stopping: Stop if validation loss doesn't improve
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                patience_counter = 0
                print(
                    f"      ✓ Validation improved. best val_loss={best_val_loss:.4f} "
                    f"(rmse={best_val_rmse:.4f}, mae={best_val_mae:.4f}). Reset patience."
                )
            else:
                patience_counter += 1
                print(
                    f"      ✗ No improvement. Patience counter={patience_counter}/"
                    f"{self.config.early_stop_patience}"
                )
                if patience_counter >= self.config.early_stop_patience:
                    print(
                        f"      → Early stopping triggered at epoch {epoch + 1}. "
                        f"Best val_loss={best_val_loss:.4f}."
                    )
                    break

        return best_val_loss

class LSTMOptimizer:
    """Coordinates LSTM hyperparameter search using a validator."""

    def __init__(self, validator: LSTMValidator, config: LSTMConfig):
        self.validator = validator
        self.config = config

    def _parameter_grid(self) -> Iterable[Dict[str, Any]]:
        """Generate hyperparameter combinations for grid search.
        
        Searches over 6 critical hyperparameters (epochs removed - uses early stopping instead):
        - sequence_length: Historical context window
        - learning_rate: Initial optimizer step size (will be auto-reduced by scheduler)
        - batch_size: Gradient batch size
        - units: LSTM hidden units
        - layers: LSTM depth
        - dropout: Regularization
        
        Early stopping: Automatically stops when validation loss plateaus
        Learning rate scheduling: Automatically reduces LR when loss plateaus
        
        Grid size: 3 × 3 × 2 × 4 × 3 × 3 = 216 combinations (~4-6 hours on GPU)
        """
        # Sequence length: 30, 90, 180 (3 values)
        seq_length_values = [30, 90, 180]
        
        # Learning rate: 0.0001, 0.0005, 0.001, 0.005 (3 values, log scale)
        # Will be auto-reduced by ReduceLROnPlateau when loss plateaus
        learning_rate_values = [0.0005, 0.001, 0.005]
        
        # Batch size: 32, 64, 128 (3 values, powers of 2)
        batch_size_values = [32, 64, 128]
        
        # LSTM architecture
        units_values = range(self.config.units_range[0], self.config.units_range[1] + 1, 128)  # 128, 256, 384, 512
        layers_values = range(self.config.layers_range[0], self.config.layers_range[1] + 1)  # 1, 2, 3
        dropout_values = [0.1, 0.2, 0.3]  # 3 values
        
        # Generate all combinations (NO EPOCHS - determined by early stopping)
        for seq_len, lr, batch_size, units, layers, dropout in product(
            seq_length_values,
            learning_rate_values,
            batch_size_values,
            units_values,
            layers_values,
            dropout_values
        ):
            yield {
                'sequence_length': seq_len,
                'learning_rate': lr,
                'batch_size': batch_size,
                'units': units,
                'layers': layers,
                'dropout': dropout,
            }

    def run(self) -> ParameterSearchResult:
        """Run grid search with early stopping and learning rate scheduling.
        
        For each hyperparameter combination:
        1. Train model with early stopping (stops when val loss plateaus)
        2. Use learning rate scheduling (reduces LR when loss plateaus)
        3. Track best model based on validation loss
        """
        print("Starting LSTM hyperparameter optimization...")

        history: List[ParameterSearchStep] = []
        best_loss = float('inf')
        best_params: Optional[Dict[str, Any]] = None
        best_model: Optional[nn.Module] = None
        no_improvement = 0

        grid = list(self._parameter_grid())
        print(f"Total parameter combinations: {len(grid)}")

        for idx, params in enumerate(grid, start=1):
            try:
                val_loss = self.validator.validate(params)
                train_loss = val_loss  # Approximation
                step = ParameterSearchStep(
                    params=params,
                    val_loss=val_loss,
                    train_loss=train_loss,
                )
                history.append(step)
                print(
                    f"Step {idx}/{len(grid)}: "
                    f"SeqLen={params['sequence_length']}, "
                    f"LR={params['learning_rate']}, "
                    f"Batch={params['batch_size']}, "
                    f"Units={params['units']}, "
                    f"Layers={params['layers']}, "
                    f"Dropout={params['dropout']:.2f} "
                    f"-> Val Loss: {val_loss:.4f}"
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    no_improvement = 0
                    print(f"  ✓ New best model! Loss: {val_loss:.4f}")
                else:
                    no_improvement += 1
                    print(f"  ✗ No improvement. Count: {no_improvement}/{self.config.early_stop_patience}")

                if no_improvement >= self.config.early_stop_patience:
                    print(
                        f"\nEarly stopping at step {idx}. "
                        f"No improvement for {self.config.early_stop_patience} iterations."
                    )
                    break

            except Exception as exc:
                print(f"Error with params {params}: {exc}")

        if best_params is None:
            raise RuntimeError("Hyperparameter optimization failed to identify viable parameters.")

        print(f"\nOptimization complete. Best parameters: {best_params}")
        print(f"Best validation loss: {best_loss:.4f}")

        early_stopped = len(history) < len(grid)
        return ParameterSearchResult(
            best_params=best_params,
            best_model=best_model,
            best_loss=best_loss,
            steps=history,
            early_stopped=early_stopped,
        )

class LSTMEvaluator:
    """Generates rolling forecasts on a hold-out split."""

    def __init__(
        self,
        train_series: pd.Series,
        validation_series: pd.Series,
        test_series: pd.Series,
        model: nn.Module,
        scaler: MinMaxScaler,
        config: LSTMConfig,
        sequence_length: Optional[int] = None,
    ):
        self.history = pd.concat([train_series, validation_series]) if len(validation_series) > 0 else train_series
        self.test_series = test_series
        self.model = model
        self.scaler = scaler
        self.config = config
        # Use provided sequence_length or fall back to config (for backward compatibility)
        self.sequence_length = sequence_length or getattr(config, 'sequence_length', 60)

    def evaluate(self) -> ForecastResult:
        if len(self.test_series) == 0:
            print("No test data available for evaluation.")
            empty = pd.Series(dtype=float)
            return ForecastResult(empty, empty, float('inf'), float('inf'), float('inf'), np.array([]), np.array([]))

        # Scale data
        scaled_history = self.scaler.transform(self.history.values.reshape(-1, 1))
        scaled_test = self.scaler.transform(self.test_series.values.reshape(-1, 1))

        predictions = []
        current_sequence = scaled_history[-self.sequence_length:].copy()

        self.model.eval()
        with torch.no_grad():
            for i in range(len(self.test_series)):
                X = torch.tensor(current_sequence.reshape(1, self.sequence_length, 1), dtype=torch.float32).to(device)
                pred = self.model(X).cpu().numpy().flatten()[0]
                predictions.append(pred)
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = scaled_test[i]

        # Inverse transform
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        forecast_series = pd.Series(predictions, index=self.test_series.index)
        actual_series = self.test_series

        valid_mask = ~np.isnan(forecast_series.values) & ~np.isnan(actual_series.values)
        if np.any(valid_mask):
            valid_actual = actual_series.values[valid_mask]
            valid_forecast = forecast_series.values[valid_mask]
            mse = mean_squared_error(valid_actual, valid_forecast)
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(valid_actual, valid_forecast))
            print("\nTest Set Forecasting Performance:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Evaluated on {len(valid_forecast)} valid samples")
        else:
            print("\nNo valid forecast samples available for evaluation.")
            mse = rmse = mae = float('inf')
            valid_actual = np.array([])
            valid_forecast = np.array([])

        preview_count = min(5, len(actual_series))
        if preview_count > 0:
            comparison = pd.DataFrame(
                {
                    'actual': actual_series.iloc[:preview_count].values,
                    'predicted': forecast_series.iloc[:preview_count].values,
                },
                index=actual_series.index[:preview_count],
            )
            print("\nSample actual vs predicted (first {0} points):".format(preview_count))
            print(comparison)
            print(
                f"Actual range: [{actual_series.min():.2f}, {actual_series.max():.2f}] | "
                f"Predicted range: [{forecast_series.min():.2f}, {forecast_series.max():.2f}]"
            )

        return ForecastResult(
            forecast=forecast_series,
            actual=actual_series,
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            valid_actual=valid_actual,
            valid_forecast=valid_forecast,
        )

class LSTMPlotter:
    """Responsible for plotting LSTM forecasts."""

    def __init__(self, max_points: int):
        self.max_points = max_points

    def plot(self, forecast: ForecastResult, target_column: str) -> None:
        if forecast.forecast.empty or forecast.actual.empty:
            print("Forecast or actual data empty. Skipping plot.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting.")
            return

        steps = min(len(forecast.actual), self.max_points)
        forecast_series = forecast.forecast.iloc[:steps]
        actual_series = forecast.actual.iloc[:steps]

        plt.figure(figsize=(12, 5))
        plt.plot(actual_series.index, actual_series.values, label='Actual', color='tab:blue')
        plt.plot(
            forecast_series.index,
            forecast_series.values,
            label='Predicted',
            color='tab:orange',
            linestyle='--',
        )
        plt.title('LSTM Forecast vs Actual (Test Set)')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class LSTMTrainer:
    """Facade coordinating data preparation, optimization, training, and evaluation.
    
    Workflow:
    1. Load and split data via DataLoader
    2. Optimize LSTM hyperparameters using validation set (or use predefined params)
    3. Train final model on combined train+validation data
    4. Evaluate on hold-out test set with rolling forecasts
    
    IMPORTANT: Scaler is fit ONLY on train+val data to prevent test data leakage.
    """

    ENABLE_PLOTTING: bool = False
    ENABLE_GRID_SEARCH: bool = False  # Set to True to run grid search, False to use PREDEFINED_PARAMS
    
    # Best parameters from last grid search execution
    PREDEFINED_PARAMS: Dict[str, Any] = {
        'sequence_length': 128,
        'learning_rate': 0.0003,
        'batch_size': 128,
        'units': 512,
        'layers': 3,
        'dropout': 0.2,
    }

    def __init__(self, data_loader: CSVDataLoader, config: Optional[LSTMConfig] = None, target_column: Optional[str] = None):
        """
        Initialize LSTM trainer with data and configuration.
        
        Args:
            data_loader: CSVDataLoader instance providing train/val/test splits
            config: LSTMConfig instance (uses defaults if None)
            target_column: Override target column from config (for backward compatibility)
        """
        self.config = config or LSTMConfig()
        if target_column is not None:
            # Support legacy API
            self.config = LSTMConfig(target_column=target_column)
        
        self.data_loader = data_loader

        self.train_series = data_loader.get_train_data()[self.config.target_column].copy()
        self.val_series = data_loader.get_validation_data()[self.config.target_column].copy()
        self.test_series = data_loader.get_test_data()[self.config.target_column].copy()

        if len(self.train_series) == 0:
            raise ValueError("Training split is empty. Adjust DataLoader.SPLIT_RATIOS or verify source data.")
        # Allow empty validation split for small datasets
        if len(self.val_series) == 0:
            print("Warning: Validation split is empty. Using training data for validation.")

        # CRITICAL: Fit scaler ONLY on train+val data to prevent test leakage
        self.scaler = MinMaxScaler()
        combined_for_scaling = pd.concat([self.train_series, self.val_series]) if len(self.val_series) > 0 else self.train_series
        self.scaler.fit(combined_for_scaling.values.reshape(-1, 1))

        # Legacy attribute names for backward compatibility
        self.train_data = self.train_series
        self.test_data = self.test_series
        self.target_column = self.config.target_column
        self.best_model = None

        self.optimization_result: Optional[ParameterSearchResult] = None
        self.trained_model: Optional[nn.Module] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_loss: Optional[float] = None
        self.last_forecast: Optional[ForecastResult] = None

    # --------------------------------------------------------------
    # Optimization
    # --------------------------------------------------------------
    def _prepare_optimization_series(self) -> Tuple[pd.Series, pd.Series]:
        """Prepare train and validation series for optimization, applying window limits."""
        train_series = self.train_series
        val_series = self.val_series

        if self.config.optimization_train_window and len(train_series) > self.config.optimization_train_window:
            train_series = train_series.tail(self.config.optimization_train_window)

        if self.config.optimization_val_window and len(val_series) > self.config.optimization_val_window:
            val_series = val_series.tail(self.config.optimization_val_window)
        
        # If validation is empty, use a small portion of training data for validation
        if len(val_series) == 0 and len(train_series) > 1:
            split_idx = max(1, len(train_series) // 2)
            val_series = train_series.iloc[split_idx:]
            train_series = train_series.iloc[:split_idx]

        return train_series, val_series

    def optimize_parameters(self) -> ParameterSearchResult:
        """Select hyperparameters for training.

        Behavior:
        - If models/lstm_best_params.json exists, load and use it.
        - Else, fall back to PREDEFINED_PARAMS.
        - If ENABLE_GRID_SEARCH is True, print a notice to use optimize_lstm.py.
        """
        import os, json
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        best_json = os.path.join(models_dir, 'lstm_best_params.json')

        if self.ENABLE_GRID_SEARCH:
            print("NOTE: In-script grid search is deprecated. Use optimize_lstm.py to run grid search.")

        # Prefer loading best params produced by optimize_lstm.py
        if os.path.exists(best_json):
            try:
                with open(best_json, 'r') as f:
                    payload = json.load(f)
                loaded = payload.get('best_params') or payload
                if isinstance(loaded, dict):
                    self.best_params = {
                        'sequence_length': int(loaded['sequence_length']),
                        'learning_rate': float(loaded['learning_rate']),
                        'batch_size': int(loaded['batch_size']),
                        'units': int(loaded['units']),
                        'layers': int(loaded['layers']),
                        'dropout': float(loaded['dropout']),
                    }
                    self.best_loss = float(payload.get('best_loss', float('inf')))
                    print(f"Loaded best params from {best_json}: {self.best_params} (best_loss={self.best_loss})")
                else:
                    raise ValueError("Malformed best params JSON.")
            except Exception as e:
                print(f"Warning: failed to load {best_json}: {e}. Falling back to PREDEFINED_PARAMS.")
                self.best_params = self.PREDEFINED_PARAMS.copy()
                self.best_loss = float('inf')
        else:
            print("Best params file not found. Using predefined parameters...")
            print(f"Predefined params: {self.PREDEFINED_PARAMS}")
            self.best_params = self.PREDEFINED_PARAMS.copy()
            self.best_loss = float('inf')

        self.trained_model = None
        self.best_model = None

        # Return a simple ParameterSearchResult for compatibility
        result = ParameterSearchResult(
            best_params=self.best_params,
            best_model=None,
            best_loss=self.best_loss,
            steps=[],
            early_stopped=False,
        )
        self.optimization_result = result
        return result

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------
    def train_final_model(self) -> nn.Module:
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize_parameters first.")

        # Extract best hyperparameters
        seq_length = self.best_params['sequence_length']
        batch_size = self.best_params['batch_size']
        learning_rate = self.best_params['learning_rate']
        units = self.best_params['units']
        layers = self.best_params['layers']
        dropout = self.best_params['dropout']

        combined = pd.concat([self.train_series, self.val_series]) if len(self.val_series) > 0 else self.train_series
        scaled_data = self.scaler.transform(combined.values.reshape(-1, 1))

        # Internal chronological split for validation during final training
        val_frac = min(max(self.config.final_val_fraction, 0.0), 0.5)
        split_idx = int(len(scaled_data) * (1.0 - val_frac)) if val_frac > 0 else len(scaled_data)
        scaled_train_final = scaled_data[:split_idx]
        scaled_val_internal = scaled_data[split_idx:]

        X_train, y_train = create_sequences(scaled_train_final, seq_length, stride=self.config.sequence_stride)
        X_val_int, y_val_int = create_sequences(scaled_val_internal, seq_length, stride=self.config.sequence_stride)

        if len(X_train) == 0 or len(X_val_int) == 0:
            raise ValueError("Not enough data for sequence creation in final training.")

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_int = torch.tensor(X_val_int, dtype=torch.float32).to(device)
        y_val_int = torch.tensor(y_val_int, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset_int = TensorDataset(X_val_int, y_val_int)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader_int = DataLoader(val_dataset_int, batch_size=batch_size, shuffle=False)

        self.trained_model = LSTMModel(
            hidden_size=units,
            num_layers=layers,
            dropout=dropout
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.trained_model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler for final training
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.lr_scheduler_min_lr
        )

        print(f"\nTraining final model with best params on combined train+val data...")
        print(f"  Sequence length: {seq_length}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Units: {units}, Layers: {layers}, Dropout: {dropout}")
        
        self.trained_model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        total_batches = len(train_loader)

        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            last_log_time = epoch_start
            running_loss = 0.0
            batches_seen = 0

            # Training
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader, start=1):
                optimizer.zero_grad()
                outputs = self.trained_model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batches_seen = batch_idx

                time_elapsed = time.time() - last_log_time
                if (
                    batch_idx == 1
                    or batch_idx % self.config.log_interval_batches == 0
                    or time_elapsed >= self.config.log_interval_seconds
                    or batch_idx == total_batches
                ):
                    avg_loss = running_loss / batches_seen
                    progress_pct = (batch_idx / total_batches * 100) if total_batches else 0.0
                    print(
                        "\r"
                        f"    [train] epoch {epoch + 1:03d} batch {batch_idx}/{total_batches} "
                        f"({progress_pct:5.1f}%) loss={loss.item():.4f} "
                        f"avg={avg_loss:.4f} elapsed={time.time() - epoch_start:.1f}s",
                        end='',
                        flush=True,
                    )
                    last_log_time = time.time()

                if self.config.max_batches_per_epoch is not None and batch_idx >= self.config.max_batches_per_epoch:
                    break

            mean_train_loss = running_loss / batches_seen if batches_seen > 0 else float('inf')
            if batches_seen:
                print()

            # Internal validation evaluation
            self.trained_model.eval()
            val_losses = []
            with torch.no_grad():
                for v_idx, (Xb, yb) in enumerate(val_loader_int, start=1):
                    out_val = self.trained_model(Xb)
                    v_loss = criterion(out_val.squeeze(), yb)
                    val_losses.append(v_loss.item())
                    if self.config.max_batches_per_epoch is not None and v_idx >= self.config.max_batches_per_epoch:
                        break
            mean_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
            self.trained_model.train()

            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(mean_val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            lr_msg = f"lr={new_lr:.6f}"
            if new_lr < prev_lr:
                lr_msg += f" (↓ from {prev_lr:.6f})"

            improved = mean_val_loss < best_val_loss
            if improved:
                best_val_loss = mean_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"    Epoch {epoch + 1:03d}: train_loss={mean_train_loss:.4f} | "
                f"val_loss={mean_val_loss:.4f} | {lr_msg} | "
                f"patience={patience_counter}/{self.config.early_stop_patience}"
            )

            if patience_counter >= self.config.early_stop_patience:
                print(
                    f"    → Early stopping (final training) at epoch {epoch + 1}. "
                    f"Best val_loss={best_val_loss:.4f}."
                )
                break

        print("Final model trained.")
        return self.trained_model

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------
    def evaluate_on_test(self, test_window: Optional[int] = 500) -> ForecastResult:
        if self.trained_model is None:
            raise ValueError("No trained model available. Run optimize_parameters and train_final_model first.")

        if test_window is not None and test_window < len(self.test_series):
            test_slice = self.test_series.tail(test_window)
            print(f"Evaluating on last {test_window} test points (out of {len(self.test_series)} total).")
        else:
            test_slice = self.test_series
            print(f"Evaluating on all {len(self.test_series)} test points. This may take a long time.")

        # Pass sequence_length from best_params to evaluator
        seq_length = self.best_params['sequence_length'] if self.best_params else 60
        evaluator = LSTMEvaluator(
            self.train_series, 
            self.val_series, 
            test_slice, 
            self.trained_model, 
            self.scaler, 
            self.config,
            sequence_length=seq_length
        )
        forecast = evaluator.evaluate()
        self.last_forecast = forecast
        return forecast

    # --------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------
    def get_training_statistics(self) -> Dict[str, object]:
        if self.optimization_result is None:
            return {"message": "No training history available."}

        stats: Dict[str, object] = {
            'best_params': self.optimization_result.best_params,
            'best_val_loss': self.optimization_result.best_loss,
            'total_steps': len(self.optimization_result.steps),
            'early_stopped': self.optimization_result.early_stopped,
            'history': [step.__dict__ for step in self.optimization_result.steps],
        }

        if self.last_forecast is not None:
            stats['test_mse'] = self.last_forecast.mse
            stats['test_rmse'] = self.last_forecast.rmse
            stats['test_mae'] = self.last_forecast.mae

        return stats

    def plot_results(self) -> None:
        if not self.ENABLE_PLOTTING:
            print("Plotting disabled. Set LSTMTrainer.ENABLE_PLOTTING = True to enable plots.")
            return

        if self.last_forecast is None:
            print("No cached forecast found. Evaluating on test data before plotting...")
            forecast = self.evaluate_on_test()
        else:
            forecast = self.last_forecast

        plotter = LSTMPlotter(self.config.plot_max_points)
        plotter.plot(forecast, self.config.target_column)

def main() -> None:
    """
    Main training pipeline for LSTM time series forecasting.
    
    Workflow:
    1. Load and split data via DataLoader
    2. Optimize LSTM hyperparameters (units, layers, dropout) using validation set
    3. Train final model on combined train+validation data
    4. Evaluate on hold-out test set with rolling forecasts
    5. Report statistics and visualize results
    
    Data scaling:
    - MinMaxScaler is fit ONLY on train+val data to prevent test leakage
    - Predictions are inverse-transformed back to original scale
    """
    print("Loading data...")
    data_loader = CSVDataLoader()

    config = LSTMConfig(
        target_column='close',
        # Grid search ranges (no fixed values - determined adaptively)
        sequence_length_range=(30, 180),
        batch_size_range=(16, 128),
        learning_rate_range=(0.0001, 0.005),
        units_range=(128, 512),
        layers_range=(1, 3),
        dropout_range=(0.0, 0.3),
        # Early stopping and learning rate scheduling
        early_stop_patience=10,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=5,
        lr_scheduler_min_lr=1e-6,
        max_epochs=500,
        # Optimization windows for representative grid search
        optimization_train_window=10000,
        optimization_val_window=1000,
    )

    trainer = LSTMTrainer(data_loader, config)

    print(f"Train data shape: {trainer.train_series.shape}")
    print(f"Validation data shape: {trainer.val_series.shape}")
    print(f"Test data shape: {trainer.test_series.shape}")

    optimization_result = trainer.optimize_parameters()
    trainer.train_final_model()
    forecast = trainer.evaluate_on_test()

    stats = trainer.get_training_statistics()
    print("\nTraining Statistics:")
    for key, value in stats.items():
        if key != 'history':
            print(f"{key}: {value}")

    trainer.plot_results()

if __name__ == "__main__":
    main()