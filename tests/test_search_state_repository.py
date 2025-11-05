import json
from pathlib import Path
from infrastructure.persistence.search_state_repository import JSONSearchStateRepository

def test_search_state_defaults_and_portability(tmp_path: Path):
    p = tmp_path / 'state.json'
    repo = JSONSearchStateRepository(str(p))

    # Default load
    state = repo.load()
    assert state['completed'] == []
    assert state['best_params'] is None
    assert state['best_loss'] == float('inf')

    # Save with inf should write None in JSON
    repo.save({'completed': [], 'best_params': None, 'best_loss': float('inf')})
    data = json.loads(p.read_text())
    assert data['best_loss'] is None

    # Load should convert None back to inf
    loaded = repo.load()
    assert loaded['best_loss'] == float('inf')
