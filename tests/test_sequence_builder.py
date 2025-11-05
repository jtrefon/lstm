import numpy as np
from infrastructure.data.sequence_builder import NumpySequenceBuilder


def test_sequence_builder_basic_stride_one():
    data = np.arange(6, dtype=float)
    builder = NumpySequenceBuilder()
    X, y = builder.build(data, sequence_length=3, stride=1)
    # sequences: [0,1,2]->3, [1,2,3]->4, [2,3,4]->5
    assert X.shape == (3, 3)
    assert y.shape == (3,)
    assert np.allclose(X[0], [0, 1, 2])
    assert y[0] == 3


def test_sequence_builder_with_stride_two():
    data = np.arange(8, dtype=float)
    builder = NumpySequenceBuilder()
    X, y = builder.build(data, sequence_length=3, stride=2)
    # indices start at 0 with step 2 until limit 5 (exclusive): i=0,2,4
    assert X.shape == (3, 3)
    assert y.shape == (3,)
    assert np.allclose(X[1], [2, 3, 4])
    assert y[1] == 5
