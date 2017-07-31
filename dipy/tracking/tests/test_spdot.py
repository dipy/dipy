import numpy as np
import numpy.testing as npt
from dipy.tracking.spdot import spdot, spdot_t

def test_spdot():
    matrix = np.random.randn(3, 3)
    row, col = np.where(matrix)
    X = matrix[row, col]
    h = np.random.randn(3)
    out = spdot(row, col, X, h, row.shape[0], 3)
    npt.assert_almost_equal(out, np.dot(matrix, h))


def test_spdot_t():
    matrix = np.random.randn(3, 3)
    row, col = np.where(matrix)
    X = matrix[row, col]
    h = np.random.randn(3)
    out = spdot_t(row, col, X, h, row.shape[0], 3)
    npt.assert_almost_equal(out, np.dot(matrix.T, h))
