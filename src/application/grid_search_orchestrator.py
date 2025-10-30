"""Grid search orchestrator - application use case."""
import time
from typing import Optional

from domain.models import ParameterSet, SearchSummary
from domain.services import GridSearchService
from application.dto import GridSearchRequest, GridSearchResponse


class GridSearchOrchestrator:
    """Orchestrates grid search use case."""

    def __init__(self, service: GridSearchService):
        """Initialize with grid search service."""
        self.service = service

    def execute(self, request: GridSearchRequest) -> GridSearchResponse:
        """Execute grid search."""
        try:
            start_time = time.time()
            best_params: Optional[ParameterSet] = None
            best_loss = float('inf')
            completed_trials = 0

            for result in self.service.run(samples_cap=request.max_trials):
                completed_trials += 1

                if request.verbose:
                    print(
                        f"[{result.trial_number}] "
                        f"seq_len={result.parameters.sequence_length} "
                        f"lr={result.parameters.learning_rate:.6f} "
                        f"batch={result.parameters.batch_size} "
                        f"units={result.parameters.units} "
                        f"layers={result.parameters.layers} "
                        f"dropout={result.parameters.dropout:.2f} "
                        f"→ loss={result.metrics.val_loss:.6f} "
                        f"({result.metrics.duration_seconds:.1f}s)"
                    )

                if result.metrics.val_loss < best_loss:
                    best_loss = result.metrics.val_loss
                    best_params = result.parameters
                    if request.verbose:
                        print(f"  ✓ New best: loss={best_loss:.6f}")

            end_time = time.time()

            if best_params is None:
                return GridSearchResponse(
                    summary=None,
                    success=False,
                    error_message="No valid trials completed",
                )

            summary = SearchSummary(
                total_trials=completed_trials,
                completed_trials=completed_trials,
                best_parameters=best_params,
                best_loss=best_loss,
                early_stopped=False,
                start_time=start_time,
                end_time=end_time,
            )

            return GridSearchResponse(
                summary=summary,
                success=True,
            )

        except Exception as e:
            return GridSearchResponse(
                summary=None,
                success=False,
                error_message=str(e),
            )
