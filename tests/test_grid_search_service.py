from pathlib import Path
from typing import Iterable, List

from domain.models import ParameterSet, ValidationMetrics
from domain.services import GridSearchService
from infrastructure.persistence.results_repository import CSVResultsRepository
from infrastructure.persistence.search_state_repository import JSONSearchStateRepository
from infrastructure.persistence.best_params_repository import JSONBestParamsRepository


class FakeGridGenerator:
    def __init__(self, params: List[ParameterSet]):
        self._params = params

    def generate(self) -> Iterable[ParameterSet]:
        for p in self._params:
            yield p

    def total(self) -> int:
        return len(self._params)


class FakeValidator:
    def __init__(self, losses_by_key: dict[tuple, float]):
        self._losses = losses_by_key

    def validate(self, params: ParameterSet) -> ValidationMetrics:
        key = tuple(sorted(params.to_dict().items()))
        loss = float(self._losses[key])
        return ValidationMetrics(val_loss=loss, val_rmse=loss ** 0.5, val_mae=loss, duration_seconds=0.01)


def _key(ps: ParameterSet) -> tuple:
    return tuple(sorted(ps.to_dict().items()))


def test_resume_skips_completed_and_updates_best(tmp_path: Path):
    # Params
    p1 = ParameterSet(sequence_length=10, learning_rate=0.01, batch_size=32, units=64, layers=2, dropout=0.0)
    p2 = ParameterSet(sequence_length=20, learning_rate=0.02, batch_size=64, units=128, layers=2, dropout=0.0)

    # Losses: p1 worse, p2 better
    losses = {
        _key(p1): 0.9,
        _key(p2): 0.1,
    }

    # Repositories
    state_path = tmp_path / 'state.json'
    results_path = tmp_path / 'results.csv'
    best_path = tmp_path / 'best.json'

    state_repo = JSONSearchStateRepository(str(state_path))
    results_repo = CSVResultsRepository(str(results_path))
    best_repo = JSONBestParamsRepository(str(best_path))

    # Pre-populate state with p1 completed
    state = state_repo.load()
    state['completed'] = [p1.to_dict()]
    state['best_params'] = None
    state['best_loss'] = float('inf')
    state_repo.save(state)

    # Service with generator including both p1 and p2
    gen = FakeGridGenerator([p1, p2])
    val = FakeValidator(losses)
    svc = GridSearchService(gen, val, state_repo, results_repo, best_repo)

    # Run service and collect results
    results = list(svc.run())

    # Only p2 should be run (p1 skipped)
    assert len(results) == 1
    assert _key(results[0].parameters) == _key(p2)

    # State should include both completed now and best should be p2 with loss 0.1
    loaded_state = state_repo.load()
    completed_keys = {tuple(sorted(d.items())) for d in loaded_state['completed']}
    assert _key(p1) in completed_keys
    assert _key(p2) in completed_keys
    assert loaded_state['best_loss'] == 0.1

    # Best params repo should match p2
    best_loaded = best_repo.load()
    assert best_loaded is not None
    best_params, best_loss = best_loaded
    assert best_loss == 0.1
    assert best_params == p2
