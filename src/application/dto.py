"""Application DTOs - data transfer objects for use cases."""
from dataclasses import dataclass
from typing import Optional

from domain.models import ParameterSet, SearchSummary


@dataclass
class GridSearchRequest:
    """Request to run grid search."""
    max_trials: Optional[int] = None
    verbose: bool = True


@dataclass
class GridSearchResponse:
    """Response from grid search."""
    summary: Optional[SearchSummary]
    success: bool
    error_message: Optional[str] = None
