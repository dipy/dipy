import numpy as np
import numpy.testing as npt

from dipy.segment.select import select_by_roi

def test_select_by_roi():
    streamlines = [np.array([[0, 0., 0.9],
                             [1.9, 0., 0.]]),
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

    selection = select_by_roi(streamlines, [mask1, mask2], [True, True], tol=1)

    npt.assert_array_equal(list(selection), [streamlines[0],
                                            streamlines[1]])

    selection = select_by_roi(streamlines, [mask1, mask2], [True, False],
                              tol=0.87)

    npt.assert_array_equal(list(selection), [streamlines[1]])

    selection = select_by_roi(streamlines, [mask1, mask2], [True, True],
                                            tol=0.87)

    npt.assert_array_equal(list(selection), [streamlines[1]])

    mask3 = np.zeros_like(mask1)
    mask3[0, 2, 2] = 1
    selection = select_by_roi(streamlines, [mask1, mask2, mask3],
                              [True, True, False], tol=1.0)

    npt.assert_array_equal(list(selection), [streamlines[0]])

    # Select using only one ROI
    selection = select_by_roi(streamlines, [mask1], [True], tol=0.87)
    npt.assert_array_equal(list(selection), [streamlines[1]])

    selection = select_by_roi(streamlines, [mask1], [True], tol=1.0)
    npt.assert_array_equal(list(selection), [streamlines[0],
                                            streamlines[1]])

    # Use different modes:
    selection = select_by_roi(streamlines, [mask1, mask2, mask3],
                              [True, True, False],
                              mode="all",
                              tol=1.0)
    npt.assert_array_equal(list(selection), [streamlines[0]])


    selection = select_by_roi(streamlines, [mask1, mask2, mask3],
                              [True, True, False],
                              mode="either_end",
                              tol=1.0)
    npt.assert_array_equal(list(selection), [streamlines[0]])

    selection = select_by_roi(streamlines, [mask1, mask2, mask3],
                              [True, True, False],
                              mode="both_end",
                              tol=1.0)
    npt.assert_array_equal(list(selection), [streamlines[0]])

    mask2[0, 2, 2] = True
    selection = select_by_roi(streamlines, [mask1, mask2, mask3],
                              [True, True, False],
                              mode="both_end",
                              tol=1.0)

    npt.assert_array_equal(list(selection), [streamlines[0],
                                             streamlines[1]])


    # Test with generator input:
    def generate_sl(streamlines):
        for sl in streamlines:
            yield sl

    selection = select_by_roi(generate_sl(streamlines), [mask1], [True],
                              tol=1.0)
    npt.assert_array_equal(list(selection), [streamlines[0],
                                            streamlines[1]])
