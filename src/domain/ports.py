"""Domain ports - abstract interfaces for external adapters."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from domain.models import ParameterSet, SearchTrialResult, ValidationMetrics


class DataSource(ABC):
    """Port for loading time series data."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load raw data from source."""
        pass


class SeriesSplitter(ABC):
    """Port for splitting time series into train/val/test."""

    @abstractmethod
    def split(self, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Split series into (train, validation, test)."""
        pass


class SequenceBuilder(ABC):
    """Port for creating sequences from time series data."""

    @abstractmethod
    def build(self, data: Any, sequence_length: int, stride: int = 1) -> tuple[Any, Any]:
        """Build sequences from data. Returns (X, y) arrays."""
        pass


class LSTMValidator(ABC):
    """Port for validating LSTM with given hyperparameters."""

    @abstractmethod
    def validate(self, parameters: ParameterSet) -> ValidationMetrics:
        """Validate model with given parameters. Returns metrics."""
        pass


class ParameterGridGenerator(ABC):
    """Port for generating hyperparameter combinations."""

    @abstractmethod
    def generate(self) -> Iterable[ParameterSet]:
        """Generate parameter combinations."""
        pass


class SearchStateRepository(ABC):
    """Port for persisting and loading search state."""

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load search state. Returns dict with 'completed', 'best_params', 'best_loss'."""
        pass

    @abstractmethod
    def save(self, state: Dict[str, Any]) -> None:
        """Save search state."""
        pass


class ResultsRepository(ABC):
    """Port for persisting trial results."""

    @abstractmethod
    def append_header_if_needed(self) -> None:
        """Ensure results file has header."""
        pass

    @abstractmethod
    def append_result(self, result: SearchTrialResult) -> None:
        """Append a trial result."""
        pass


class BestParamsRepository(ABC):
    """Port for persisting best parameters found."""

    @abstractmethod
    def save(self, parameters: ParameterSet, loss: float) -> None:
        """Save best parameters."""
        pass

    @abstractmethod
    def load(self) -> Optional[tuple[ParameterSet, float]]:
        """Load best parameters. Returns (ParameterSet, loss) or None."""
        pass
