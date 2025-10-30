"""Domain models - value objects representing core concepts."""
from dataclasses import dataclass
from typing import Any, Dict
import time


@dataclass(frozen=True)
class ParameterSet:
    """Immutable set of hyperparameters for a single trial."""
    sequence_length: int
    learning_rate: float
    batch_size: int
    units: int
    layers: int
    dropout: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sequence_length': self.sequence_length,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ParameterSet':
        """Create from dictionary."""
        return ParameterSet(
            sequence_length=int(data['sequence_length']),
            learning_rate=float(data['learning_rate']),
            batch_size=int(data['batch_size']),
            units=int(data['units']),
            layers=int(data['layers']),
            dropout=float(data['dropout']),
        )


@dataclass(frozen=True)
class ValidationMetrics:
    """Immutable validation metrics from a single trial."""
    val_loss: float
    val_rmse: float
    val_mae: float
    duration_seconds: float


@dataclass(frozen=True)
class SearchTrialResult:
    """Immutable result of a single grid search trial."""
    trial_number: int
    parameters: ParameterSet
    metrics: ValidationMetrics
    timestamp: float

    @staticmethod
    def create(
        trial_number: int,
        parameters: ParameterSet,
        metrics: ValidationMetrics,
    ) -> 'SearchTrialResult':
        """Factory method to create a trial result with current timestamp."""
        return SearchTrialResult(
            trial_number=trial_number,
            parameters=parameters,
            metrics=metrics,
            timestamp=time.time(),
        )


@dataclass(frozen=True)
class SearchSummary:
    """Immutable summary of grid search results."""
    total_trials: int
    completed_trials: int
    best_parameters: ParameterSet
    best_loss: float
    early_stopped: bool
    start_time: float
    end_time: float

    @property
    def duration_seconds(self) -> float:
        """Total search duration."""
        return self.end_time - self.start_time

    @property
    def trials_per_second(self) -> float:
        """Throughput metric."""
        if self.duration_seconds == 0:
            return 0.0
        return self.completed_trials / self.duration_seconds
