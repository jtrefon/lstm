"""Series splitting adapter."""
from typing import Tuple

import pandas as pd

from domain.ports import SeriesSplitter


class TimeSeriesSplitter(SeriesSplitter):
    """Splits time series into train/val/test using ratios."""

    def __init__(
        self,
        train_ratio: float = 0.95,
        validation_ratio: float = 0.035,
        test_ratio: float = 0.015,
    ):
        """Initialize with split ratios."""
        total = train_ratio + validation_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

    def split(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split series into (train, validation, test)."""
        n = len(series)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.validation_ratio)

        train = series.iloc[:train_end]
        validation = series.iloc[train_end:val_end]
        test = series.iloc[val_end:]

        return train, validation, test


class OptimizationWindowSplitter(SeriesSplitter):
    """Splits series and applies optimization windows for faster grid search."""

    def __init__(
        self,
        base_splitter: SeriesSplitter,
        train_window: int = None,
        val_window: int = None,
    ):
        """Initialize with base splitter and window sizes."""
        self.base_splitter = base_splitter
        self.train_window = train_window
        self.val_window = val_window

    def split(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split series and apply windows."""
        train, validation, test = self.base_splitter.split(series)

        # Apply windows (use tail to get most recent data)
        if self.train_window and len(train) > self.train_window:
            train = train.tail(self.train_window)

        if self.val_window and len(validation) > self.val_window:
            validation = validation.tail(self.val_window)

        return train, validation, test
