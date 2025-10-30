"""Best parameters persistence - JSON-based implementation."""
import json
import os
import time
from typing import Optional

from domain.models import ParameterSet
from domain.ports import BestParamsRepository


class JSONBestParamsRepository(BestParamsRepository):
    """Persists best parameters to JSON file."""

    def __init__(self, filepath: str):
        """Initialize with file path."""
        self.filepath = filepath
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure parent directory exists."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def save(self, parameters: ParameterSet, loss: float) -> None:
        """Save best parameters to JSON file."""
        self._ensure_directory()
        payload = {
            'best_params': parameters.to_dict(),
            'best_loss': loss,
            'timestamp': time.time(),
        }
        with open(self.filepath, 'w') as f:
            json.dump(payload, f, indent=2)

    def load(self) -> Optional[tuple[ParameterSet, float]]:
        """Load best parameters from JSON file."""
        if not os.path.exists(self.filepath):
            return None

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            params = ParameterSet.from_dict(data['best_params'])
            loss = float(data['best_loss'])
            return (params, loss)
        except (json.JSONDecodeError, KeyError, IOError, ValueError):
            return None
