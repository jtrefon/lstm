import numpy as np
from infrastructure.data.outlier_handler import OutlierHandler


def test_zscore_std_zero_returns_tuple_and_zero_clipped():
    data = np.ones(10)
    processed, n_clipped = OutlierHandler.handle_outliers(data, method='zscore', threshold=3.0)
    assert isinstance(processed, np.ndarray)
    assert n_clipped == 0
    assert processed.shape == data.shape
    assert np.allclose(processed, data)


def test_iqr_clips_extremes():
    data = np.array([1, 1, 1, 1, 1000, 1, 1, 1], dtype=float)
    processed, n_clipped = OutlierHandler.handle_outliers(data, method='iqr', threshold=1.5)
    assert isinstance(processed, np.ndarray)
    assert n_clipped >= 1
    assert processed.shape == data.shape
