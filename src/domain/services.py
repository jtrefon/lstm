"""Domain services - core business logic using ports."""
from typing import Iterable

from domain.models import ParameterSet, SearchTrialResult, ValidationMetrics
from domain.ports import (
    ParameterGridGenerator,
    LSTMValidator,
    SearchStateRepository,
    ResultsRepository,
    BestParamsRepository,
)


class GridSearchService:
    """Orchestrates grid search using injected dependencies."""

    def __init__(
        self,
        grid_generator: ParameterGridGenerator,
        validator: LSTMValidator,
        state_repo: SearchStateRepository,
        results_repo: ResultsRepository,
        best_params_repo: BestParamsRepository,
    ):
        """Initialize with all required adapters."""
        self.grid_generator = grid_generator
        self.validator = validator
        self.state_repo = state_repo
        self.results_repo = results_repo
        self.best_params_repo = best_params_repo

    def run(self, samples_cap: int = None) -> Iterable[SearchTrialResult]:
        """
        Run grid search, yielding results as they complete.
        
        Supports resumable search - skips already-completed trials.
        
        Args:
            samples_cap: Maximum number of trials to run (from config). None = all.
        """
        # Load previous state
        state = self.state_repo.load()
        completed_keys = set(
            tuple(sorted(p.items()))
            for p in state.get('completed', [])
        )
        best_params = state.get('best_params')
        # Normalize best_params from JSON dict to ParameterSet if necessary
        if isinstance(best_params, dict):
            try:
                best_params = ParameterSet.from_dict(best_params)
            except Exception:
                best_params = None
        best_loss_raw = state.get('best_loss', float('inf'))
        # Ensure best_loss is float (JSON loads it as string sometimes)
        try:
            best_loss = float(best_loss_raw)
        except (TypeError, ValueError):
            best_loss = float('inf')

        # Ensure results file has header
        self.results_repo.append_header_if_needed()

        # Generate and test parameters
        trial_number = 0
        for params in self.grid_generator.generate():
            # Skip already-completed trials
            param_key = tuple(sorted(params.to_dict().items()))
            if param_key in completed_keys:
                continue

            trial_number += 1
            if samples_cap and trial_number > samples_cap:
                break

            # Validate parameters
            metrics = self.validator.validate(params)

            # Create result
            result = SearchTrialResult.create(
                trial_number=trial_number,
                parameters=params,
                metrics=metrics,
            )

            # Persist result
            self.results_repo.append_result(result)

            # Update best if improved
            if metrics.val_loss < best_loss:
                best_loss = metrics.val_loss
                best_params = params
                self.best_params_repo.save(params, best_loss)

            # Update state
            state['completed'].append(params.to_dict())
            # Persist best_params as dict if available
            state['best_params'] = best_params.to_dict() if isinstance(best_params, ParameterSet) else None
            state['best_loss'] = float(best_loss)  # Ensure it's a float for JSON serialization
            self.state_repo.save(state)

            yield result
