"""Configuration loader - parses YAML config files into typed objects."""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class LSTMModelConfig:
    """LSTM model architecture configuration."""
    input_size: int
    sequence_stride: int


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop configuration."""
    max_epochs: int
    early_stop_patience: int
    log_interval_batches: int
    log_interval_seconds: float
    max_batches_per_epoch: Optional[int]


@dataclass(frozen=True)
class LearningRateSchedulerConfig:
    """Learning rate scheduler configuration."""
    factor: float
    patience: int
    min_lr: float


@dataclass(frozen=True)
class OptimizationWindowsConfig:
    """Optimization windows for grid search."""
    train_window: Optional[int]
    val_window: Optional[int]


@dataclass(frozen=True)
class ForecastConfig:
    """Forecasting configuration."""
    plot_max_points: int


@dataclass(frozen=True)
class LSTMConfig:
    """Complete LSTM configuration."""
    model: LSTMModelConfig
    training: TrainingConfig
    lr_scheduler: LearningRateSchedulerConfig
    optimization: OptimizationWindowsConfig
    forecast: ForecastConfig
    # Optional defaults for single-training when best_params/CLI not provided
    model_defaults: "ModelDefaultsConfig | None" = None
    # Control parameter precedence: True = config only, False = config > best_params.json
    use_config_only: bool = False


@dataclass(frozen=True)
class ModelDefaultsConfig:
    """Default per-trial hyperparameters for single training."""
    sequence_length: int
    learning_rate: float
    batch_size: int
    units: int
    layers: int
    dropout: float


@dataclass(frozen=True)
class GridSearchConfig:
    """Grid search hyperparameter space."""
    sequence_length: List[int]
    learning_rate: List[float]
    batch_size: List[int]
    units: List[int]
    layers: List[int]
    dropout: List[float]
    samples_cap: Optional[int]  # Max number of trials to run (None = all)
    data_samples_cap: Optional[int]  # Max number of data points to use for grid search (None = all)
    data_cap_from: str  # 'start' or 'end' (use most recent when 'end')


@dataclass(frozen=True)
class PersistenceConfig:
    """Persistence layer configuration."""
    state_file: str
    results_file: str
    best_params_file: str


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    verbose: bool
    save_interval: int


@dataclass(frozen=True)
class OptimizationConfig:
    """Complete optimization configuration."""
    grid_search: GridSearchConfig
    persistence: PersistenceConfig
    logging: LoggingConfig


@dataclass(frozen=True)
class DataSourceConfig:
    """Data source configuration."""
    type: str  # 'csv', 'parquet', etc.
    path: str
    target_column: str


@dataclass(frozen=True)
class SplittingConfig:
    """Data splitting configuration."""
    train_ratio: float
    validation_ratio: float
    test_ratio: float


@dataclass(frozen=True)
class PreprocessingConfig:
    """Data preprocessing configuration."""
    selected_columns: List[str]
    handle_missing: str  # 'forward_fill', 'drop', 'interpolate'
    sort_by_index: bool
    outlier_method: str  # 'none', 'iqr', 'zscore', 'detrended_iqr'
    outlier_threshold: float  # IQR multiplier (1.5) or z-score threshold (3.0)


@dataclass(frozen=True)
class DataConfig:
    """Complete data configuration."""
    data_source: DataSourceConfig
    splitting: SplittingConfig
    preprocessing: PreprocessingConfig


class ConfigLoader:
    """Loads and parses YAML configuration files."""

    @staticmethod
    def load_lstm_config(config_path: str) -> LSTMConfig:
        """Load LSTM configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Model config (ensure ints) - fail fast if missing
        m = data['model']
        model = LSTMModelConfig(
            input_size=int(m['input_size']),
            sequence_stride=int(m['sequence_stride']),
        )

        # Training config (ensure ints/floats) - fail fast if missing
        t = data['training']
        training = TrainingConfig(
            max_epochs=int(t['max_epochs']),
            early_stop_patience=int(t['early_stop_patience']),
            log_interval_batches=int(t['log_interval_batches']),
            log_interval_seconds=float(t['log_interval_seconds']),
            max_batches_per_epoch=(
                None if t['max_batches_per_epoch'] in (None, 'null')
                else int(t['max_batches_per_epoch'])
            ),
        )

        # LR scheduler (ensure floats/ints) - fail fast if missing
        s = data['learning_rate_scheduler']
        lr_scheduler = LearningRateSchedulerConfig(
            factor=float(s['factor']),
            patience=int(s['patience']),
            min_lr=float(s['min_lr']),
        )

        # Optimization windows (ensure ints/None) - fail fast if missing
        o = data['optimization']
        train_window_raw = o['train_window']
        val_window_raw = o['val_window']
        optimization = OptimizationWindowsConfig(
            train_window=(None if train_window_raw in (None, 'null') else int(train_window_raw)),
            val_window=(None if val_window_raw in (None, 'null') else int(val_window_raw)),
        )

        # Forecast (ensure int) - fail fast if missing
        fconf = data['forecast']
        forecast = ForecastConfig(
            plot_max_points=int(fconf['plot_max_points']),
        )

        # Optional model defaults for single-training
        defaults_raw = data.get('model_defaults')
        defaults = None
        if defaults_raw is not None:
            defaults = ModelDefaultsConfig(
                sequence_length=int(defaults_raw['sequence_length']),
                learning_rate=float(defaults_raw['learning_rate']),
                batch_size=int(defaults_raw['batch_size']),
                units=int(defaults_raw['units']),
                layers=int(defaults_raw['layers']),
                dropout=float(defaults_raw['dropout']),
            )

        # Optional flag to enforce config-only mode (ignore best_params.json)
        use_config_only = bool(data.get('use_config_only', False))

        return LSTMConfig(
            model=model,
            training=training,
            lr_scheduler=lr_scheduler,
            optimization=optimization,
            forecast=forecast,
            model_defaults=defaults,
            use_config_only=use_config_only,
        )

    @staticmethod
    def load_optimization_config(config_path: str) -> OptimizationConfig:
        """Load optimization configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Normalize grid lists to correct types - fail fast if missing
        gs = data['grid_search']
        grid_search = GridSearchConfig(
            sequence_length=[int(x) for x in gs['sequence_length']],
            learning_rate=[float(x) for x in gs['learning_rate']],
            batch_size=[int(x) for x in gs['batch_size']],
            units=[int(x) for x in gs['units']],
            layers=[int(x) for x in gs['layers']],
            dropout=[float(x) for x in gs['dropout']],
            samples_cap=(
                None if gs.get('samples_cap', None) in (None, 'null')
                else int(gs['samples_cap'])
            ),
            data_samples_cap=(
                None if gs.get('data_samples_cap', None) in (None, 'null')
                else int(gs['data_samples_cap'])
            ),
            data_cap_from=str(gs.get('data_cap_from', 'end')),
        )

        persistence = PersistenceConfig(**data['persistence'])

        lg = data['logging']
        logging = LoggingConfig(
            verbose=bool(lg['verbose']),
            save_interval=int(lg['save_interval']),
        )

        return OptimizationConfig(
            grid_search=grid_search,
            persistence=persistence,
            logging=logging,
        )

    @staticmethod
    def load_data_config(config_path: str) -> DataConfig:
        """Load data configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        ds = data['data_source']
        data_source = DataSourceConfig(
            type=str(ds['type']),
            path=str(ds['path']),
            target_column=str(ds['target_column']),
        )

        sp = data['splitting']
        splitting = SplittingConfig(
            train_ratio=float(sp['train_ratio']),
            validation_ratio=float(sp['validation_ratio']),
            test_ratio=float(sp['test_ratio']),
        )

        pp = data['preprocessing']
        preprocessing = PreprocessingConfig(
            selected_columns=[str(x) for x in pp['selected_columns']],
            handle_missing=str(pp['handle_missing']),
            sort_by_index=bool(pp['sort_by_index']),
            outlier_method=str(pp['outlier_method']),
            outlier_threshold=float(pp['outlier_threshold']),
        )

        return DataConfig(
            data_source=data_source,
            splitting=splitting,
            preprocessing=preprocessing,
        )

    @staticmethod
    def get_config_path(filename: str) -> str:
        """Get absolute path to config file."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, '..', 'config', filename)
