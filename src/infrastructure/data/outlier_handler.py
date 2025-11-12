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
        window: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Handle outliers in data.

        Args:
            data: 1D numpy array of values
            method: 'none' (no handling), 'iqr' (interquartile range), 'zscore', 'detrended_iqr', 'rolling_mad'
            threshold: IQR multiplier (1.5), z-score threshold (3.0), or MAD multiplier (e.g., 5.0)
            window: rolling window size for 'rolling_mad'
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
        elif method == 'rolling_mad':
            return OutlierHandler._handle_rolling_mad(data, window=window, k=threshold)
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
            return data, 0

        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > threshold

        # Clip outliers to threshold bounds
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        clipped = np.clip(data, lower_bound, upper_bound)

        n_clipped = np.sum(outlier_mask)
        return clipped, int(n_clipped)

    @staticmethod
    def _handle_rolling_mad(data: np.ndarray, window: int | None = None, k: float = 5.0) -> tuple[np.ndarray, int]:
        """
        Handle outliers using a rolling Median Absolute Deviation (MAD) approach.

        Uses rolling median m_t and MAD_t = median(|x_t - m_t|) over a centered window.
        Clips values outside m_t ± k * 1.4826 * MAD_t. The constant 1.4826 scales MAD to
        be a consistent estimator for the standard deviation under normality.

        Args:
            data: 1D numpy array
            window: rolling window size (required). If None or <3, falls back to global MAD.
            k: MAD multiplier (e.g., 5.0 is common to avoid over-clipping)

        Returns:
            (clipped_data, n_clipped)
        """
        if window is None or window < 3:
            # Global MAD fallback
            s = pd.Series(data)
            med = float(s.median())
            mad = float((s - med).abs().median())
            scale = 1.4826 * mad if mad > 0 else 0.0
            if scale == 0.0:
                return data, 0
            lower = med - k * scale
            upper = med + k * scale
            clipped = np.clip(data, lower, upper)
            n_clipped = int(((data < lower) | (data > upper)).sum())
            return clipped, n_clipped

        s = pd.Series(data)
        med = s.rolling(window=window, center=True, min_periods=max(3, window // 4)).median()
        abs_dev = (s - med).abs()
        mad = abs_dev.rolling(window=window, center=True, min_periods=max(3, window // 4)).median()
        # Where MAD is zero or NaN (e.g., flat segments), fallback to global MAD
        global_med = float(s.median())
        global_mad = float((s - global_med).abs().median())
        global_scale = 1.4826 * global_mad if global_mad > 0 else 0.0
        scale = 1.4826 * mad
        scale = scale.fillna(global_scale).replace(0.0, global_scale)
        med_filled = med.fillna(global_med)
        lower = med_filled - k * scale
        upper = med_filled + k * scale
        lower_np = lower.to_numpy()
        upper_np = upper.to_numpy()
        clipped = np.clip(data, lower_np, upper_np)
        n_clipped = int(((data < lower_np) | (data > upper_np)).sum())
        return clipped, n_clipped

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
