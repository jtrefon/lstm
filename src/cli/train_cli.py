import argparse
import os
import sys
import logging
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from domain.models import ParameterSet
from infrastructure.data.csv_data_source import CSVDataSource
from infrastructure.data.series_splitter import OptimizationWindowSplitter, TimeSeriesSplitter
from infrastructure.data.sequence_builder import NumpySequenceBuilder
from infrastructure.persistence.best_params_repository import JSONBestParamsRepository
from infrastructure.torch.lstm_trainer import PyTorchLSTMValidator
from infrastructure.persistence.model_repository import TorchModelRepository, ModelPackageRepository


def resolve_default_artifacts(best_params_path: str) -> tuple[str, str]:
    # Place artifacts next to best_params file by default
    models_dir = os.path.dirname(os.path.abspath(best_params_path)) or 'models'
    os.makedirs(models_dir, exist_ok=True)
    return (
        os.path.join(models_dir, 'lstm_model.pt'),
        os.path.join(models_dir, 'lstm_scaler.pkl'),
    )


def choose_parameters(
    best_repo: JSONBestParamsRepository,
    cli: argparse.Namespace,
    lstm_config,
) -> tuple[ParameterSet, str]:
    loaded = best_repo.load()
    best_params: Optional[ParameterSet] = loaded[0] if loaded else None
    defaults = getattr(lstm_config, 'model_defaults', None)
    use_config_only = getattr(lstm_config, 'use_config_only', False)

    # Determine parameter precedence based on use_config_only flag
    if use_config_only:
        # Config-only mode: CLI > model_defaults, ignore best_params.json
        params_source = best_params if best_params else None
        best_params = None
    else:
        # Auto mode: CLI > model_defaults > best_params.json
        params_source = best_params

    # Apply CLI overrides (if provided) or require params
    seq_len = (
        cli.sequence_length
        if cli.sequence_length is not None
        else (defaults.sequence_length if defaults else (params_source.sequence_length if params_source else None))
    )
    lr = (
        cli.learning_rate
        if cli.learning_rate is not None
        else (defaults.learning_rate if defaults else (params_source.learning_rate if params_source else None))
    )
    batch = (
        cli.batch_size
        if cli.batch_size is not None
        else (defaults.batch_size if defaults else (params_source.batch_size if params_source else None))
    )
    units = (
        cli.units
        if cli.units is not None
        else (defaults.units if defaults else (params_source.units if params_source else None))
    )
    layers = (
        cli.layers
        if cli.layers is not None
        else (defaults.layers if defaults else (params_source.layers if params_source else None))
    )
    dropout = (
        cli.dropout
        if cli.dropout is not None
        else (defaults.dropout if defaults else (params_source.dropout if params_source else None))
    )

    if None in (seq_len, lr, batch, units, layers, dropout):
        raise SystemExit(
            'No best_params found and not all hyperparameters were provided. '
            'Pass overrides via CLI or run optimize first.'
        )

    source = (
        'cli' if any(x is not None for x in [cli.sequence_length, cli.learning_rate, cli.batch_size, cli.units, cli.layers, cli.dropout])
        else ('model_defaults' if defaults is not None else 'best_params.json')
    )

    return ParameterSet(
        sequence_length=int(seq_len),
        learning_rate=float(lr),
        batch_size=int(batch),
        units=int(units),
        layers=int(layers),
        dropout=float(dropout),
    ), source


def main() -> None:
    p = argparse.ArgumentParser(description='Train single LSTM model with fixed hyperparameters')
    p.add_argument('--sequence-length', type=int)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--units', type=int)
    p.add_argument('--layers', type=int)
    p.add_argument('--dropout', type=float)
    p.add_argument('--model-out', type=str, default=None)
    p.add_argument('--scaler-out', type=str, default=None)
    p.add_argument('--out-dir', type=str, default=None)
    p.add_argument('--verbose', action='store_true', default=True)
    args = p.parse_args()

    logging.basicConfig(
        level=(logging.INFO if args.verbose else logging.WARNING),
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    # Load configs
    lstm_config = ConfigLoader.load_lstm_config(ConfigLoader.get_config_path('lstm_config.yaml'))
    optimization_config = ConfigLoader.load_optimization_config(ConfigLoader.get_config_path('optimization_config.yaml'))
    data_config = ConfigLoader.load_data_config(ConfigLoader.get_config_path('data_config.yaml'))

    # Data loading
    data_source = CSVDataSource(
        filepath=data_config.data_source.path,
        target_column=data_config.data_source.target_column,
    )
    df = data_source.load()
    series = df[data_config.data_source.target_column]
    logger.info(f"Loaded {len(series)} samples from data source")

    # Splitting: use full dataset with train/val ratios (optimization windows are for grid search only)
    splitter = TimeSeriesSplitter(
        train_ratio=data_config.splitting.train_ratio,
        validation_ratio=data_config.splitting.validation_ratio,
        test_ratio=data_config.splitting.test_ratio,
    )
    train_series, val_series, _ = splitter.split(series)
    logger.info(f"Split sizes → train={len(train_series)} val={len(val_series)}")

    # Validator
    sequence_builder = NumpySequenceBuilder()
    validator = PyTorchLSTMValidator(
        train_series=train_series,
        val_series=val_series,
        sequence_builder=sequence_builder,
        lstm_config=lstm_config,
        preprocessing_config=data_config.preprocessing,
        log_preprocessing=args.verbose,
    )

    # Parameters
    best_repo = JSONBestParamsRepository(optimization_config.persistence.best_params_file)
    params, source = choose_parameters(best_repo, args, lstm_config)

    # Train
    metrics = validator.validate(params)

    # Save artifacts
    model = validator.get_last_model()
    scaler = validator.get_scaler()
    saved_package_dir = None
    if model is not None and scaler is not None:
        base_dir = args.out_dir
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(optimization_config.persistence.best_params_file))
        pkg_repo = ModelPackageRepository()
        saved_package_dir = pkg_repo.save_package(
            base_dir=base_dir,
            state_dict=model.state_dict(),
            scaler=scaler,
            params=params,
            metrics={
                'val_loss': float(metrics.val_loss),
                'val_rmse': float(metrics.val_rmse),
                'val_mae': float(metrics.val_mae),
            },
        )

    # Optionally also write direct files if user asked
    if args.model_out or args.scaler_out:
        repo = TorchModelRepository()
        if model is not None and args.model_out:
            repo.save_model(model.state_dict(), args.model_out)
        if scaler is not None and args.scaler_out:
            repo.save_scaler(scaler, args.scaler_out)

    prep = validator.get_preprocessing_stats()
    seqs = validator.get_sequence_stats()

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    if saved_package_dir:
        logger.info(f"Saved package: {saved_package_dir}")
    if args.model_out:
        logger.info(f"Saved model:   {args.model_out}")
    if args.scaler_out:
        logger.info(f"Saved scaler:  {args.scaler_out}")
    logger.info(f"Parameter source: {source}")
    logger.info("Parameters:")
    logger.info(f"  sequence_length: {params.sequence_length}")
    logger.info(f"  learning_rate:  {params.learning_rate}")
    logger.info(f"  batch_size:     {params.batch_size}")
    logger.info(f"  units:          {params.units}")
    logger.info(f"  layers:         {params.layers}")
    logger.info(f"  dropout:        {params.dropout}")
    logger.info("Data/Preprocessing:")
    logger.info(f"  raw_len:        {prep.get('raw_len', 0)}")
    logger.info(f"  cleaned_len:    {prep.get('cleaned_len', 0)}")
    logger.info(f"  na_removed:     {prep.get('na_removed', 0)}")
    logger.info(f"  outliers_clipped: {prep.get('outliers_clipped', 0)}")
    logger.info("Sequences:")
    logger.info(f"  train_sequences: {seqs.get('train_sequences', 0)}")
    logger.info(f"  val_sequences:   {seqs.get('val_sequences', 0)}")
    logger.info("Validation metrics:")
    logger.info(f"  val_loss:       {metrics.val_loss:.6f}")
    logger.info(f"  val_rmse:       {metrics.val_rmse:.6f}")
    logger.info(f"  val_mae:        {metrics.val_mae:.6f}")


if __name__ == '__main__':
    main()
