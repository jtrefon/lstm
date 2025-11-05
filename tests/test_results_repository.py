from pathlib import Path
import csv

from domain.models import ParameterSet, SearchTrialResult, ValidationMetrics
from infrastructure.persistence.results_repository import CSVResultsRepository


def test_results_repository_header_and_append(tmp_path: Path):
    out = tmp_path / 'results.csv'
    repo = CSVResultsRepository(str(out))

    # Header should be created once
    repo.append_header_if_needed()
    repo.append_header_if_needed()

    with out.open('r', newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSVResultsRepository.HEADERS

    # Append one result
    params = ParameterSet(
        sequence_length=10,
        learning_rate=0.01,
        batch_size=32,
        units=64,
        layers=2,
        dropout=0.0,
    )
    metrics = ValidationMetrics(val_loss=0.123456, val_rmse=0.3333, val_mae=0.1111, duration_seconds=1.2)
    result = SearchTrialResult.create(trial_number=1, parameters=params, metrics=metrics)
    repo.append_result(result)

    with out.open('r', newline='') as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2
    data = rows[1]
    # trial number
    assert data[0] == '1'
    # sequence_length
    assert data[1] == str(params.sequence_length)
    # learning_rate formatted as float
    assert float(data[2]) == params.learning_rate
    # batch_size
    assert data[3] == str(params.batch_size)
    # units
    assert data[4] == str(params.units)
    # layers
    assert data[5] == str(params.layers)
    # dropout
    assert float(data[6]) == params.dropout
    # val_loss formatted to 6 decimals
    assert data[7] == f"{metrics.val_loss:.6f}"
    # duration formatted to 1 decimal
    assert data[10] == f"{metrics.duration_seconds:.1f}"
