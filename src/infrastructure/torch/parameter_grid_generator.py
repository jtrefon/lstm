"""Parameter grid generator adapter."""
from functools import reduce
from itertools import product
from operator import mul
from typing import Iterable

from config.config_loader import GridSearchConfig
from domain.models import ParameterSet
from domain.ports import ParameterGridGenerator


class ConfigBasedParameterGridGenerator(ParameterGridGenerator):
    """Generates parameter combinations from config."""

    def __init__(self, grid_config: GridSearchConfig):
        """Initialize with grid search config."""
        self.grid_config = grid_config

    def generate(self) -> Iterable[ParameterSet]:
        """Generate all parameter combinations."""
        for seq_len, lr, batch_size, units, layers, dropout in product(
            self.grid_config.sequence_length,
            self.grid_config.learning_rate,
            self.grid_config.batch_size,
            self.grid_config.units,
            self.grid_config.layers,
            self.grid_config.dropout,
        ):
            # Ensure types are correct even if YAML parsed values as strings
            yield ParameterSet(
                sequence_length=int(seq_len),
                learning_rate=float(lr),
                batch_size=int(batch_size),
                units=int(units),
                layers=int(layers),
                dropout=float(dropout),
            )

    def total(self) -> int:
        """Return total combinations available in the search grid."""
        dimensions = (
            len(self.grid_config.sequence_length),
            len(self.grid_config.learning_rate),
            len(self.grid_config.batch_size),
            len(self.grid_config.units),
            len(self.grid_config.layers),
            len(self.grid_config.dropout),
        )
        return reduce(mul, dimensions, 1)
