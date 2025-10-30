"""Results persistence - CSV-based implementation."""
import csv
import os
from typing import List

from domain.models import SearchTrialResult
from domain.ports import ResultsRepository


class CSVResultsRepository(ResultsRepository):
    """Persists trial results to CSV file."""

    HEADERS = [
        'trial_number',
        'sequence_length',
        'learning_rate',
        'batch_size',
        'units',
        'layers',
        'dropout',
        'val_loss',
        'val_rmse',
        'val_mae',
        'duration_seconds',
        'timestamp',
    ]

    def __init__(self, filepath: str):
        """Initialize with file path."""
        self.filepath = filepath
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure parent directory exists."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def append_header_if_needed(self) -> None:
        """Write header if file doesn't exist."""
        if not os.path.exists(self.filepath):
            self._ensure_directory()
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def append_result(self, result: SearchTrialResult) -> None:
        """Append a trial result to CSV."""
        self._ensure_directory()
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.trial_number,
                int(result.parameters.sequence_length),
                float(result.parameters.learning_rate),
                int(result.parameters.batch_size),
                int(result.parameters.units),
                int(result.parameters.layers),
                float(result.parameters.dropout),
                f"{float(result.metrics.val_loss):.6f}",
                f"{float(result.metrics.val_rmse):.6f}",
                f"{float(result.metrics.val_mae):.6f}",
                f"{float(result.metrics.duration_seconds):.1f}",
                f"{float(result.timestamp):.0f}",
            ])
