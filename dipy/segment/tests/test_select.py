import numpy as np
import numpy.testing as npt

from dipy.segment.select import select_by_roi, _multi_or

def test_multi_or():
    my_arr = np.array([[True, True, True, False],
                       [True, False, False, False],
                       [False, False, False, False]])

    npt.assert_array_equal(_multi_or(my_arr),
                           np.array([True, True, True, False]))

    my_arr = np.array([[False, False, True, False],
                       [True, False, False, False],
                       [False, False, False, False]])

    npt.assert_array_equal(_multi_or(my_arr),
                           np.array([True, False, True, False]))


def test_select_by_roi():
    streamlines = [np.array([[0.5, 0., 0.],
                             [1.5, 0., 0.]]),
                   np.array([[0., 0., 0],
                             [0, 1., 1.],
                             [0, 2., 2.]]),
                   np.array([[2, 2, 2],
                             [3, 3, 3]])]

    affine = np.eye(4)
    # Make two ROIs:
    mask1 = np.zeros((4, 4, 4), dtype=bool)
    mask2 = np.zeros_like(mask1)
    mask1[0, 0, 0] = True
    mask2[1, 0, 0] = True

    selection = select_by_roi(streamlines, [mask1, mask2], [True, True], tol=0)

    npt.assert_array_equal(selection, np.array([False, True, False]))

    selection = select_by_roi(streamlines, [mask1, mask2], [True, True], tol=0.5)

    npt.assert_array_equal(selection, np.array([True, True, False]))

    mask3 = np.zeros_like(mask1)
    mask3[0, 2, 2] = 1
    selection = select_by_roi(streamlines, [mask1, mask2, mask3],
                              [True, True, False], tol=0.5)

    npt.assert_array_equal(selection, np.array([True, False, False]))
