import numpy as np
import numpy.testing as npt
import scipy.ndimage

from dipy.tracking.local import (BinaryTissueClassifier,
                                 ThresholdTissueClassifier,
                                 ActTissueClassifier)


def test_binary_tissue_classifier():
    """This tests that the binary tissue classifier return expected
     tissue types:

        OUTSIDEIMAGE = -1
        INVALIDPOINT = 0
        TRACKPOINT = 1
        ENDPOINT = 2
    """

    mask = np.random.random((4, 4, 4))
    mask[mask < 0.4] = 0.0

    btc_boolean = BinaryTissueClassifier(mask > 0)
    btc_float64 = BinaryTissueClassifier(mask)

    # test voxel center
    for ind in np.ndindex(mask.shape):
        pts = np.array(ind, dtype='float64')
        state_boolean = btc_boolean.check_point(pts)
        state_float64 = btc_float64.check_point(pts)
        if mask[ind] > 0:
            npt.assert_equal(state_boolean, 1)
            npt.assert_equal(state_float64, 1)
        else:
            npt.assert_equal(state_boolean, 2)
            npt.assert_equal(state_float64, 2)

    # test random points in voxel
    for ind in np.ndindex(mask.shape):
        for _ in range(10):
            pts = np.array(ind, dtype='float64') + np.random.random(3) - 0.5
            state_boolean = btc_boolean.check_point(pts)
            state_float64 = btc_float64.check_point(pts)
            if mask[ind] > 0:
                npt.assert_equal(state_boolean, 1)
                npt.assert_equal(state_float64, 1)
            else:
                npt.assert_equal(state_boolean, 2)
                npt.assert_equal(state_float64, 2)

    # test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state_boolean = btc_boolean.check_point(pts)
        state_float64 = btc_float64.check_point(pts)
        npt.assert_equal(state_boolean, -1)
        npt.assert_equal(state_float64, -1)


def test_threshold_tissue_classifier():
    """This tests that the thresholdy tissue classifier return expected
     tissue types:

        OUTSIDEIMAGE = -1
        INVALIDPOINT = 0
        TRACKPOINT = 1
        ENDPOINT = 2
    """

    tissue_map = np.random.random((4, 4, 4))

    ttc = ThresholdTissueClassifier(tissue_map.astype('float32'), 0.5)

    # test voxel center
    for ind in np.ndindex(tissue_map.shape):
        pts = np.array(ind, dtype='float64')
        state = ttc.check_point(pts)
        if tissue_map[ind] > 0.5:
            npt.assert_equal(state, 1)
        else:
            npt.assert_equal(state, 2)

    # test random points in voxel
    inds = [[0, 1.4, 2.2], [0, 2.3, 2.3], [0, 2.2, 1.3], [0, 0.9, 2.2],
            [0, 2.8, 1.1], [0, 1.1, 3.3], [0, 2.1, 1.9], [0, 3.1, 3.1]]
    for pts in inds:
        pts = np.array(pts, dtype='float64')
        state = ttc.check_point(pts)
        res = scipy.ndimage.map_coordinates(
            tissue_map, np.reshape(pts, (3, 1)), order=1, mode='nearest')
        if res > 0.5:
            npt.assert_equal(state, 1)
        else:
            npt.assert_equal(state, 2)

    # test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state = ttc.check_point(pts)
        npt.assert_equal(state, -1)


def test_act_tissue_classifier():
    """This tests that the act tissue classifier return expected
     tissue types:

        OUTSIDEIMAGE = -1
        INVALIDPOINT = 0
        TRACKPOINT = 1
        ENDPOINT = 2
    """

    gm = np.random.random((4, 4, 4))
    wm = np.random.random((4, 4, 4))
    csf = np.random.random((4, 4, 4))
    tissue_sum = gm + wm + csf
    gm /= tissue_sum
    wm /= tissue_sum
    csf /= tissue_sum

    act_tc = ActTissueClassifier(include_map=gm, exclude_map=csf)

    # test voxel center
    for ind in np.ndindex(wm.shape):
        pts = np.array(ind, dtype='float64')
        state = act_tc.check_point(pts)
        if csf[ind] > 0.5:
            npt.assert_equal(state, 0)
        elif gm[ind] > 0.5:
            npt.assert_equal(state, 2)
        else:
            npt.assert_equal(state, 1)

    # test random points in voxel
    inds = [[0, 1.4, 2.2], [0, 2.3, 2.3], [0, 2.2, 1.3], [0, 0.9, 2.2],
            [0, 2.8, 1.1], [0, 1.1, 3.3], [0, 2.1, 1.9], [0, 3.1, 3.1]]
    for pts in inds:
        pts = np.array(pts, dtype='float64')
        state = act_tc.check_point(pts)
        gm_res = scipy.ndimage.map_coordinates(
            gm, np.reshape(pts, (3, 1)), order=1, mode='nearest')
        csf_res = scipy.ndimage.map_coordinates(
            csf, np.reshape(pts, (3, 1)), order=1, mode='nearest')
        if csf_res > 0.5:
            npt.assert_equal(state, 0)
        elif gm_res > 0.5:
            npt.assert_equal(state, 2)
        else:
            npt.assert_equal(state, 1)

    # test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state = act_tc.check_point(pts)
        npt.assert_equal(state, -1)

if __name__ == '__main__':
    run_module_suite()
