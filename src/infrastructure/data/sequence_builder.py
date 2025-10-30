"""Sequence building adapter - creates training sequences from time series."""
from typing import Tuple

import numpy as np

from domain.ports import SequenceBuilder


class NumpySequenceBuilder(SequenceBuilder):
    """Builds sequences from numpy arrays."""

    def build(
        self,
        data: np.ndarray,
        sequence_length: int,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences from 1D array.
        
        Args:
            data: 1D numpy array of values
            sequence_length: Length of each sequence
            stride: Step size between sequences
            
        Returns:
            (X, y) where X is (n_samples, sequence_length) and y is (n_samples,)
        """
        if stride < 1:
            stride = 1

        xs, ys = [], []
        limit = len(data) - sequence_length

        for i in range(0, max(0, limit), stride):
            x = data[i : i + sequence_length]
            y = data[i + sequence_length]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)
