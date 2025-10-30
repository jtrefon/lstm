from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
from sklearn.preprocessing import StandardScaler


class SeriesTransformerStrategy(ABC):
    """Defines the contract for series transformation strategies."""

    @abstractmethod
    def fit(self, values: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LogStandardScalerStrategy(SeriesTransformerStrategy):
    """Applies log scaling prior to standard normalization."""

    def __init__(self) -> None:
        self._scaler: Final[StandardScaler] = StandardScaler()
        self._fitted = False

    def fit(self, values: np.ndarray) -> None:
        self._scaler.fit(self._log(values))
        self._fitted = True

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._ensure_fit()
        return self._scaler.transform(self._log(values))

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._ensure_fit()
        return np.expm1(self._scaler.inverse_transform(values))

    def _log(self, values: np.ndarray) -> np.ndarray:
        return np.log1p(values)

    def _ensure_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before use.")
