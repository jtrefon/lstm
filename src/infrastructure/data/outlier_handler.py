"""Outlier handling utilities for data preprocessing."""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal


class OutlierHandler:
    """Handles outlier detection and removal/clipping."""

    @staticmethod
    def handle_outliers(
        data: np.ndarray,
        method: str = 'none',
        threshold: float = 1.5,
        detrend: bool = False,
    ) -> tuple[np.ndarray, int]:
        """
        Handle outliers in data.

        Args:
            data: 1D numpy array of values
            method: 'none' (no handling), 'iqr' (interquartile range), 'zscore', 'detrended_iqr'
            threshold: IQR multiplier (1.5) or z-score threshold (3.0)
            detrend: If True, remove trend before outlier detection (preserves trend)

        Returns:
            (processed_data, n_clipped)
            processed_data: data with outliers handled (clipped or removed), trend preserved
            n_clipped: number of points clipped as outliers/anomalies
        """
        if method == 'none':
            return data, 0

        if detrend or method == 'detrended_iqr':
            return OutlierHandler._handle_detrended_iqr(data, threshold)
        elif method == 'iqr':
            return OutlierHandler._handle_iqr(data, threshold)
        elif method == 'zscore':
            return OutlierHandler._handle_zscore(data, threshold)
        else:
            raise ValueError(f"Unknown outlier method: {method}")

    @staticmethod
    def _handle_detrended_iqr(data: np.ndarray, multiplier: float = 1.5) -> tuple[np.ndarray, int]:
        """
        Handle outliers using detrended IQR method.

        Removes trend, detects outliers on detrended data, then restores trend.
        This preserves the underlying trend while removing anomalies.

        Args:
            data: 1D numpy array
            multiplier: IQR multiplier (1.5 = standard)

        Returns:
            Data with trend preserved and anomalies clipped
        """
        # Detrend: remove linear trend
        detrended = signal.detrend(data)
        
        # Detect outliers on detrended data
        q1 = np.percentile(detrended, 25)
        q3 = np.percentile(detrended, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Clip outliers in detrended space
        clipped_detrended = np.clip(detrended, lower_bound, upper_bound)

        # Restore trend by adding back the original trend
        trend = data - detrended
        result = clipped_detrended + trend

        n_clipped = np.sum((detrended < lower_bound) | (detrended > upper_bound))
        return result, int(n_clipped)

    @staticmethod
    def _handle_iqr(data: np.ndarray, multiplier: float = 1.5) -> tuple[np.ndarray, int]:
        """
        Handle outliers using Interquartile Range (IQR) method.

        Clips values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]

        Args:
            data: 1D numpy array
            multiplier: IQR multiplier (1.5 = standard, 3.0 = more lenient)

        Returns:
            Data with outliers clipped
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Clip outliers
        clipped = np.clip(data, lower_bound, upper_bound)

        n_clipped = np.sum((data < lower_bound) | (data > upper_bound))
        return clipped, int(n_clipped)

    @staticmethod
    def _handle_zscore(data: np.ndarray, threshold: float = 3.0) -> tuple[np.ndarray, int]:
        """
        Handle outliers using Z-score method.

        Clips values with |z-score| > threshold

        Args:
            data: 1D numpy array
            threshold: Z-score threshold (3.0 = standard, 2.0 = stricter)

        Returns:
            Data with outliers clipped
        """
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return data

        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > threshold

        # Clip outliers to threshold bounds
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        clipped = np.clip(data, lower_bound, upper_bound)

        n_clipped = np.sum(outlier_mask)
        return clipped, int(n_clipped)

    @staticmethod
    def get_bounds(
        data: np.ndarray,
        method: str = 'none',
        threshold: float = 1.5,
    ) -> Tuple[float, float]:
        """
        Get outlier bounds without modifying data.

        Args:
            data: 1D numpy array
            method: 'none', 'iqr', 'zscore', 'detrended_iqr'
            threshold: IQR multiplier or z-score threshold

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if method == 'none':
            return (np.min(data), np.max(data))

        if method == 'iqr':
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            return (q1 - threshold * iqr, q3 + threshold * iqr)

        elif method == 'detrended_iqr':
            detrended = signal.detrend(data)
            q1 = np.percentile(detrended, 25)
            q3 = np.percentile(detrended, 75)
            iqr = q3 - q1
            return (q1 - threshold * iqr, q3 + threshold * iqr)

        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            return (mean - threshold * std, mean + threshold * std)

        else:
            raise ValueError(f"Unknown outlier method: {method}")
