"""Grid search orchestrator - application use case."""
import time
import logging
from typing import Optional

from domain.models import ParameterSet, SearchSummary
from domain.services import GridSearchService
from application.dto import GridSearchRequest, GridSearchResponse

logger = logging.getLogger(__name__)


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

            state_snapshot = self.service.state_repo.load()
            completed_before_run = len(state_snapshot.get('completed', []))
            total_permutations = self.service.grid_generator.total()
            remaining_permutations = max(total_permutations - completed_before_run, 0)
            planned_new_trials = (
                remaining_permutations
                if request.max_trials is None
                else min(request.max_trials, remaining_permutations)
            )
            progress_denominator = (
                completed_before_run + planned_new_trials
                if planned_new_trials > 0
                else (total_permutations or 1)
            )

            if request.verbose:
                logger.info(
                    "Grid permutations: "
                    f"total={total_permutations} "
                    f"completed={completed_before_run} "
                    f"remaining={remaining_permutations} "
                    f"planned_this_run={planned_new_trials}"
                )

            for result in self.service.run(samples_cap=request.max_trials):
                completed_trials += 1

                if request.verbose:
                    global_completed = completed_before_run + completed_trials
                    progress_pct = (global_completed / progress_denominator) * 100.0
                    progress_summary = (
                        f"{global_completed}/{progress_denominator} "
                        f"({progress_pct:.1f}%)"
                    )
                    logger.info(
                        f"[{result.trial_number} | {progress_summary}] "
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
                        logger.info(f"  ✓ New best: loss={best_loss:.6f}")

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
