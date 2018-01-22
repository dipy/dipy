
import numpy as np
import numpy.testing as npt
import scipy.ndimage

from dipy.core.ndindex import ndindex
from dipy.tracking.local import (BinaryTissueClassifier,
                                 ThresholdTissueClassifier,
                                 ActTissueClassifier,
                                 CmcTissueClassifier)
from dipy.tracking.local.localtracking import TissueTypes


def test_binary_tissue_classifier():
    """This tests that the binary tissue classifier returns expected
     tissue types.
    """

    mask = np.random.random((4, 4, 4))
    mask[mask < 0.4] = 0.0

    btc_boolean = BinaryTissueClassifier(mask > 0)
    btc_float64 = BinaryTissueClassifier(mask)

    # Test voxel center
    for ind in ndindex(mask.shape):
        pts = np.array(ind, dtype='float64')
        state_boolean = btc_boolean.check_point(pts)
        state_float64 = btc_float64.check_point(pts)
        if mask[ind] > 0:
            npt.assert_equal(state_boolean, TissueTypes.TRACKPOINT)
            npt.assert_equal(state_float64, TissueTypes.TRACKPOINT)
        else:
            npt.assert_equal(state_boolean, TissueTypes.ENDPOINT)
            npt.assert_equal(state_float64, TissueTypes.ENDPOINT)

    # Test random points in voxel
    for ind in ndindex(mask.shape):
        for _ in range(50):
            pts = np.array(ind, dtype='float64') + np.random.random(3) - 0.5
            state_boolean = btc_boolean.check_point(pts)
            state_float64 = btc_float64.check_point(pts)
            if mask[ind] > 0:
                npt.assert_equal(state_boolean, TissueTypes.TRACKPOINT)
                npt.assert_equal(state_float64, TissueTypes.TRACKPOINT)
            else:
                npt.assert_equal(state_boolean, TissueTypes.ENDPOINT)
                npt.assert_equal(state_float64, TissueTypes.ENDPOINT)

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1], [4, 0, 0]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state_boolean = btc_boolean.check_point(pts)
        state_float64 = btc_float64.check_point(pts)
        npt.assert_equal(state_boolean, TissueTypes.OUTSIDEIMAGE)
        npt.assert_equal(state_float64, TissueTypes.OUTSIDEIMAGE)


def test_threshold_tissue_classifier():
    """This tests that the thresholdy tissue classifier returns expected
     tissue types.
    """

    tissue_map = np.random.random((4, 4, 4))

    ttc = ThresholdTissueClassifier(tissue_map.astype('float32'), 0.5)

    # Test voxel center
    for ind in ndindex(tissue_map.shape):
        pts = np.array(ind, dtype='float64')
        state = ttc.check_point(pts)
        if tissue_map[ind] > 0.5:
            npt.assert_equal(state, TissueTypes.TRACKPOINT)
        else:
            npt.assert_equal(state, TissueTypes.ENDPOINT)

    # Test random points in voxel
    inds = [[0, 1.4, 2.2], [0, 2.3, 2.3], [0, 2.2, 1.3], [0, 0.9, 2.2],
            [0, 2.8, 1.1], [0, 1.1, 3.3], [0, 2.1, 1.9], [0, 3.1, 3.1],
            [0, 0.1, 0.1], [0, 0.9, 0.5], [0, 0.9, 0.5], [0, 2.9, 0.1]]
    for pts in inds:
        pts = np.array(pts, dtype='float64')
        state = ttc.check_point(pts)
        res = scipy.ndimage.map_coordinates(
            tissue_map, np.reshape(pts, (3, 1)), order=1, mode='nearest')
        if res > 0.5:
            npt.assert_equal(state, TissueTypes.TRACKPOINT)
        else:
            npt.assert_equal(state, TissueTypes.ENDPOINT)

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state = ttc.check_point(pts)
        npt.assert_equal(state, TissueTypes.OUTSIDEIMAGE)


def test_act_tissue_classifier():
    """This tests that the act tissue classifier returns expected
     tissue types.
    """

    gm = np.random.random((4, 4, 4))
    wm = np.random.random((4, 4, 4))
    csf = np.random.random((4, 4, 4))
    tissue_sum = gm + wm + csf
    gm /= tissue_sum
    wm /= tissue_sum
    csf /= tissue_sum

    act_tc = ActTissueClassifier(include_map=gm, exclude_map=csf)

    # Test voxel center
    for ind in ndindex(wm.shape):
        pts = np.array(ind, dtype='float64')
        state = act_tc.check_point(pts)
        if csf[ind] > 0.5:
            npt.assert_equal(state, TissueTypes.INVALIDPOINT)
        elif gm[ind] > 0.5:
            npt.assert_equal(state, TissueTypes.ENDPOINT)
        else:
            npt.assert_equal(state, TissueTypes.TRACKPOINT)

    # Test random points in voxel
    inds = [[0, 1.4, 2.2], [0, 2.3, 2.3], [0, 2.2, 1.3], [0, 0.9, 2.2],
            [0, 2.8, 1.1], [0, 1.1, 3.3], [0, 2.1, 1.9], [0, 3.1, 3.1],
            [0, 0.1, 0.1], [0, 0.9, 0.5], [0, 0.9, 0.5], [0, 2.9, 0.1]]
    for pts in inds:
        pts = np.array(pts, dtype='float64')
        state = act_tc.check_point(pts)
        gm_res = scipy.ndimage.map_coordinates(
            gm, np.reshape(pts, (3, 1)), order=1, mode='nearest')
        csf_res = scipy.ndimage.map_coordinates(
            csf, np.reshape(pts, (3, 1)), order=1, mode='nearest')
        if csf_res > 0.5:
            npt.assert_equal(state, TissueTypes.INVALIDPOINT)
        elif gm_res > 0.5:
            npt.assert_equal(state, TissueTypes.ENDPOINT)
        else:
            npt.assert_equal(state, TissueTypes.TRACKPOINT)

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state = act_tc.check_point(pts)
        npt.assert_equal(state, TissueTypes.OUTSIDEIMAGE)


def test_cmc_tissue_classifier():
    """This tests that the cmc tissue classifier returns expected
     tissue types.
    """

    gm = np.array([[[1, 1], [0, 0], [0, 0]]])
    wm = np.array([[[0, 0], [1, 1], [0, 0]]])
    csf = np.array([[[0, 0], [0, 0], [1, 1]]])
    include_map = gm
    exclude_map = csf

    cmc_tc = CmcTissueClassifier(include_map=include_map,
                                 exclude_map=exclude_map,
                                 step_size=1,
                                 average_voxel_size=1)
    cmc_tc_from_pve = CmcTissueClassifier.from_pve(wm_map=wm,
                                                   gm_map=gm,
                                                   csf_map=csf,
                                                   step_size=1,
                                                   average_voxel_size=1)

    # Test contructors
    for idx in np.ndindex(wm.shape):
        idx = np.asarray(idx, dtype="float64")
        npt.assert_almost_equal(cmc_tc.get_include(idx),
                                cmc_tc_from_pve.get_include(idx))
        npt.assert_almost_equal(cmc_tc.get_exclude(idx),
                                cmc_tc_from_pve.get_exclude(idx))

    # Test voxel center
    for ind in ndindex(wm.shape):
        pts = np.array(ind, dtype='float64')
        state = cmc_tc.check_point(pts)
        if csf[ind] == 1:
            npt.assert_equal(state, TissueTypes.INVALIDPOINT)
        elif gm[ind] == 1:
            npt.assert_equal(state, TissueTypes.ENDPOINT)
        else:
            npt.assert_equal(state, TissueTypes.TRACKPOINT)

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        npt.assert_equal(cmc_tc.check_point(pts), TissueTypes.OUTSIDEIMAGE)
        npt.assert_equal(cmc_tc.get_exclude(pts), 0)
        npt.assert_equal(cmc_tc.get_include(pts), 0)


if __name__ == '__main__':
    npt.run_module_suite()
