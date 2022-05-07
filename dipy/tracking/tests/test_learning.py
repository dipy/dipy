""" Testing track_metrics module """

import numpy as np
from numpy.testing import assert_array_equal
from dipy.tracking import learning as tl


def test_det_corr_tracks():

    A = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    B = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    C = np.array([[0, 0, -1], [0, 0, -2], [0, 0, -3]])

    bundle1 = [A, B, C]
    bundle2 = [B, A]
    indices = [0, 1]
    print(A)
    print(B)
    print(C)

    arr = tl.detect_corresponding_tracks(indices, bundle1, bundle2)
    print(arr)
    assert_array_equal(arr, np.array([[0, 1], [1, 0]]))

    indices2 = [0, 1]
    arr2 = tl.detect_corresponding_tracks_plus(indices, bundle1,
                                               indices2, bundle2)
    print(arr2)
    assert_array_equal(arr, arr2)
