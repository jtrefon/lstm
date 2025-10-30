#!/usr/bin/env python3
"""Quick test to verify all imports work correctly."""
import sys
import os

# Add src to path (same as CLI does)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    from config.config_loader import ConfigLoader
    print("✓ ConfigLoader")
except Exception as e:
    print(f"✗ ConfigLoader: {e}")
    sys.exit(1)

try:
    from domain.models import ParameterSet, ValidationMetrics, SearchTrialResult, SearchSummary
    print("✓ Domain models")
except Exception as e:
    print(f"✗ Domain models: {e}")
    sys.exit(1)

try:
    from domain.ports import (
        DataSource, SeriesSplitter, SequenceBuilder, LSTMValidator,
        ParameterGridGenerator, SearchStateRepository, ResultsRepository, BestParamsRepository
    )
    print("✓ Domain ports")
except Exception as e:
    print(f"✗ Domain ports: {e}")
    sys.exit(1)

try:
    from domain.services import GridSearchService
    print("✓ Domain services")
except Exception as e:
    print(f"✗ Domain services: {e}")
    sys.exit(1)

try:
    from infrastructure.data.csv_data_source import CSVDataSource
    print("✓ CSV data source")
except Exception as e:
    print(f"✗ CSV data source: {e}")
    sys.exit(1)

try:
    from infrastructure.data.series_splitter import TimeSeriesSplitter, OptimizationWindowSplitter
    print("✓ Series splitters")
except Exception as e:
    print(f"✗ Series splitters: {e}")
    sys.exit(1)

try:
    from infrastructure.data.sequence_builder import NumpySequenceBuilder
    print("✓ Sequence builder")
except Exception as e:
    print(f"✗ Sequence builder: {e}")
    sys.exit(1)

try:
    from infrastructure.persistence.search_state_repository import JSONSearchStateRepository
    print("✓ Search state repository")
except Exception as e:
    print(f"✗ Search state repository: {e}")
    sys.exit(1)

try:
    from infrastructure.persistence.results_repository import CSVResultsRepository
    print("✓ Results repository")
except Exception as e:
    print(f"✗ Results repository: {e}")
    sys.exit(1)

try:
    from infrastructure.persistence.best_params_repository import JSONBestParamsRepository
    print("✓ Best params repository")
except Exception as e:
    print(f"✗ Best params repository: {e}")
    sys.exit(1)

try:
    from infrastructure.torch.lstm_trainer import PyTorchLSTMValidator, LSTMModel
    print("✓ PyTorch LSTM trainer")
except Exception as e:
    print(f"✗ PyTorch LSTM trainer: {e}")
    sys.exit(1)

try:
    from infrastructure.torch.parameter_grid_generator import ConfigBasedParameterGridGenerator
    print("✓ Parameter grid generator")
except Exception as e:
    print(f"✗ Parameter grid generator: {e}")
    sys.exit(1)

try:
    from application.dto import GridSearchRequest, GridSearchResponse
    print("✓ Application DTOs")
except Exception as e:
    print(f"✗ Application DTOs: {e}")
    sys.exit(1)

try:
    from application.grid_search_orchestrator import GridSearchOrchestrator
    print("✓ Grid search orchestrator")
except Exception as e:
    print(f"✗ Grid search orchestrator: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
