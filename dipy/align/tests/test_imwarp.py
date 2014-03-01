import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal)


def test_feature():

    A = np.ones((3, 3))
    B = A + 2

    assert_array_equal(A, B)
