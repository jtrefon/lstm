"""Search state persistence - JSON-based implementation."""
import json
import os
from typing import Any, Dict

from domain.ports import SearchStateRepository


class JSONSearchStateRepository(SearchStateRepository):
    """Persists search state to JSON file."""

    def __init__(self, filepath: str):
        """Initialize with file path."""
        self.filepath = filepath
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure parent directory exists."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """Load search state from JSON file."""
        if not os.path.exists(self.filepath):
            return {
                'completed': [],
                'best_params': None,
                'best_loss': float('inf'),
            }

        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Return default state if file is corrupted
            return {
                'completed': [],
                'best_params': None,
                'best_loss': float('inf'),
            }

    def save(self, state: Dict[str, Any]) -> None:
        """Save search state to JSON file."""
        self._ensure_directory()
        with open(self.filepath, 'w') as f:
            json.dump(state, f, indent=2)
