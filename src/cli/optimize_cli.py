"""CLI for grid search optimization - wires all components together."""
import argparse
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from domain.services import GridSearchService
from infrastructure.data.csv_data_source import CSVDataSource
from infrastructure.data.sequence_builder import NumpySequenceBuilder
from infrastructure.data.series_splitter import OptimizationWindowSplitter, TimeSeriesSplitter
from infrastructure.persistence.best_params_repository import JSONBestParamsRepository
from infrastructure.persistence.results_repository import CSVResultsRepository
from infrastructure.persistence.search_state_repository import JSONSearchStateRepository
from infrastructure.torch.lstm_trainer import PyTorchLSTMValidator
from infrastructure.torch.parameter_grid_generator import ConfigBasedParameterGridGenerator
from application.dto import GridSearchRequest
from application.grid_search_orchestrator import GridSearchOrchestrator


def main():
    """Main entry point for grid search CLI."""
    parser = argparse.ArgumentParser(description='LSTM Hyperparameter Grid Search')
    parser.add_argument('--max-trials', type=int, default=None, help='Maximum number of trials')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    args = parser.parse_args()

    logging.basicConfig(
        level=(logging.INFO if args.verbose else logging.WARNING),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading configurations...")
    lstm_config = ConfigLoader.load_lstm_config(ConfigLoader.get_config_path('lstm_config.yaml'))
    optimization_config = ConfigLoader.load_optimization_config(ConfigLoader.get_config_path('optimization_config.yaml'))
    data_config = ConfigLoader.load_data_config(ConfigLoader.get_config_path('data_config.yaml'))

    logger.info("Initializing data pipeline...")
    # Data loading (no cap here; grid-search-only cap applied below)
    data_source = CSVDataSource(
        filepath=data_config.data_source.path,
        target_column=data_config.data_source.target_column,
    )
    df = data_source.load()
    series = df[data_config.data_source.target_column]
    logger.info(f"Loaded {len(series)} samples from data source")

    # Apply grid-search-only data cap (latest samples by default)
    gs_cap = optimization_config.grid_search.data_samples_cap
    gs_cap_from = optimization_config.grid_search.data_cap_from
    if gs_cap is not None:
        if gs_cap_from == 'end':
            series = series.iloc[-gs_cap:]
            logger.info(f"Grid search data cap: using last {gs_cap} samples")
        else:
            series = series.iloc[:gs_cap]
            logger.info(f"Grid search data cap: using first {gs_cap} samples")

    # Data splitting - single source of truth for capping
    base_splitter = TimeSeriesSplitter(
        train_ratio=data_config.splitting.train_ratio,
        validation_ratio=data_config.splitting.validation_ratio,
        test_ratio=data_config.splitting.test_ratio,
    )
    if gs_cap is not None:
        # When grid-search data cap is set, ignore train/val windows to avoid double clipping
        if lstm_config.optimization.train_window is not None or lstm_config.optimization.val_window is not None:
            logger.info("NOTE: Ignoring train_window/val_window because grid_search.data_samples_cap is set")

        # Ensure splits can support the maximum sequence length
        n = len(series)
        max_seq_len = max(optimization_config.grid_search.sequence_length)
        tr_ratio = data_config.splitting.train_ratio
        va_ratio = data_config.splitting.validation_ratio
        te_ratio = data_config.splitting.test_ratio

        # Initial lengths from ratios
        te_len = int(n * te_ratio)
        va_len = max(int(n * va_ratio), max_seq_len + 1)
        tr_len = n - va_len - te_len

        # If training too small, rebalance from validation
        if tr_len < (max_seq_len + 1):
            deficit = (max_seq_len + 1) - tr_len
            take_from_val = min(deficit, max(va_len - 1, 0))
            va_len -= take_from_val
            tr_len = n - va_len - te_len

        # Final guards
        if va_len < 1:
            va_len = 1
            tr_len = max(n - va_len - te_len, max_seq_len + 1)
        if tr_len < (max_seq_len + 1):
            tr_len = max_seq_len + 1
            va_len = max(n - tr_len - te_len, 1)

        # Slice
        train_series = series.iloc[:tr_len]
        val_series = series.iloc[tr_len:tr_len + va_len]
        logger.info(f"Grid search split: train={len(train_series)} val={len(val_series)} (max_seq_len={max_seq_len})")
    else:
        if (
            lstm_config.optimization.train_window is None
            and lstm_config.optimization.val_window is None
        ):
            train_series, val_series, _ = base_splitter.split(series)
        else:
            splitter = OptimizationWindowSplitter(
                base_splitter=base_splitter,
                train_window=lstm_config.optimization.train_window,
                val_window=lstm_config.optimization.val_window,
            )
            train_series, val_series, _ = splitter.split(series)

    logger.info(f"Train samples: {len(train_series)}, Val samples: {len(val_series)}")

    logger.info("Initializing persistence layer...")
    # Persistence
    state_repo = JSONSearchStateRepository(optimization_config.persistence.state_file)
    results_repo = CSVResultsRepository(optimization_config.persistence.results_file)
    best_params_repo = JSONBestParamsRepository(optimization_config.persistence.best_params_file)

    logger.info("Initializing validator...")
    # Validator with preprocessing config for outlier handling
    sequence_builder = NumpySequenceBuilder()
    validator = PyTorchLSTMValidator(
        train_series=train_series,
        val_series=val_series,
        sequence_builder=sequence_builder,
        lstm_config=lstm_config,
        preprocessing_config=data_config.preprocessing,
    )

    logger.info("Initializing grid search...")
    # Grid search
    grid_generator = ConfigBasedParameterGridGenerator(optimization_config.grid_search)
    service = GridSearchService(
        grid_generator=grid_generator,
        validator=validator,
        state_repo=state_repo,
        results_repo=results_repo,
        best_params_repo=best_params_repo,
    )

    logger.info("Running grid search...")
    orchestrator = GridSearchOrchestrator(service)
    # Use samples_cap from config, override with CLI arg if provided
    samples_cap = args.max_trials if args.max_trials else optimization_config.grid_search.samples_cap
    request = GridSearchRequest(
        max_trials=samples_cap,
        verbose=args.verbose,
    )
    response = orchestrator.execute(request)

    if response.success:
        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH COMPLETE")
        logger.info("="*80)
        logger.info(f"Total trials: {response.summary.total_trials}")
        logger.info(f"Duration: {response.summary.duration_seconds:.1f}s")
        logger.info(f"Throughput: {response.summary.trials_per_second:.2f} trials/sec")
        logger.info(f"\nBest parameters found:")
        logger.info(f"  sequence_length: {response.summary.best_parameters.sequence_length}")
        logger.info(f"  learning_rate: {response.summary.best_parameters.learning_rate}")
        logger.info(f"  batch_size: {response.summary.best_parameters.batch_size}")
        logger.info(f"  units: {response.summary.best_parameters.units}")
        logger.info(f"  layers: {response.summary.best_parameters.layers}")
        logger.info(f"  dropout: {response.summary.best_parameters.dropout}")
        logger.info(f"\nBest validation loss: {response.summary.best_loss:.6f}")
        logger.info("="*80)
    else:
        logger.error(f"Grid search failed: {response.error_message}")
        sys.exit(1)


if __name__ == '__main__':
    main()
