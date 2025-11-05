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
                data = json.load(f)
            # Coerce defaults and fix best_loss portability
            completed = data.get('completed') or []
            best_params = data.get('best_params') if 'best_params' in data else None
            raw_best_loss = data.get('best_loss', None)
            if raw_best_loss in (None, 'null'):
                best_loss = float('inf')
            else:
                try:
                    best_loss = float(raw_best_loss)
                except (TypeError, ValueError):
                    best_loss = float('inf')
            return {
                'completed': completed,
                'best_params': best_params,
                'best_loss': best_loss,
            }
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
        payload = dict(state)
        # Ensure JSON portability: write None instead of Infinity
        try:
            bl = payload.get('best_loss', None)
            if bl is None:
                pass
            else:
                try:
                    blf = float(bl)
                except (TypeError, ValueError):
                    blf = None
                if blf is None or blf == float('inf'):
                    payload['best_loss'] = None
                else:
                    payload['best_loss'] = blf
        except Exception:
            payload['best_loss'] = None
        with open(self.filepath, 'w') as f:
            json.dump(payload, f, indent=2)
