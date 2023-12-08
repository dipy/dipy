import numpy as np
import numpy.testing as npt
import scipy.ndimage

from dipy.core.ndindex import ndindex
from dipy.tracking.stopping_criterion import (ActStoppingCriterion,
                                              BinaryStoppingCriterion,
                                              CmcStoppingCriterion,
                                              ThresholdStoppingCriterion,
                                              StreamlineStatus)
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_binary_stopping_criterion(rng):
    """This tests that the binary stopping criterion returns expected
    streamline statuses.
    """

    mask = rng.random((4, 4, 4))
    mask[mask < 0.4] = 0.0

    btc_boolean = BinaryStoppingCriterion(mask > 0)
    btc_float64 = BinaryStoppingCriterion(mask)

    # Test voxel center
    for ind in ndindex(mask.shape):
        pts = np.array(ind, dtype='float64')
        state_boolean = btc_boolean.check_point(pts)
        state_float64 = btc_float64.check_point(pts)
        if mask[ind] > 0:
            npt.assert_equal(state_boolean, int(StreamlineStatus.TRACKPOINT))
            npt.assert_equal(state_float64, int(StreamlineStatus.TRACKPOINT))
        else:
            npt.assert_equal(state_boolean, int(StreamlineStatus.ENDPOINT))
            npt.assert_equal(state_float64, int(StreamlineStatus.ENDPOINT))

    # Test random points in voxel
    for ind in ndindex(mask.shape):
        for _ in range(50):
            pts = np.array(ind, dtype='float64') + rng.random(3) - 0.5
            state_boolean = btc_boolean.check_point(pts)
            state_float64 = btc_float64.check_point(pts)
            if mask[ind] > 0:
                npt.assert_equal(state_boolean,
                                 int(StreamlineStatus.TRACKPOINT))
                npt.assert_equal(state_float64,
                                 int(StreamlineStatus.TRACKPOINT))
            else:
                npt.assert_equal(state_boolean, int(StreamlineStatus.ENDPOINT))
                npt.assert_equal(state_float64, int(StreamlineStatus.ENDPOINT))

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1], [4, 0, 0]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state_boolean = btc_boolean.check_point(pts)
        state_float64 = btc_float64.check_point(pts)
        npt.assert_equal(state_boolean, int(StreamlineStatus.OUTSIDEIMAGE))
        npt.assert_equal(state_float64, int(StreamlineStatus.OUTSIDEIMAGE))


@set_random_number_generator()
def test_threshold_stopping_criterion(rng):
    """This tests that the threshold stopping criterion returns expected
    streamline statuses.
    """

    tissue_map = rng.random((4, 4, 4))

    ttc = ThresholdStoppingCriterion(tissue_map.astype('float32'), 0.5)

    # Test voxel center
    for ind in ndindex(tissue_map.shape):
        pts = np.array(ind, dtype='float64')
        state = ttc.check_point(pts)
        if tissue_map[ind] > 0.5:
            npt.assert_equal(state, int(StreamlineStatus.TRACKPOINT))
        else:
            npt.assert_equal(state, int(StreamlineStatus.ENDPOINT))

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
            npt.assert_equal(state, int(StreamlineStatus.TRACKPOINT))
        else:
            npt.assert_equal(state, int(StreamlineStatus.ENDPOINT))

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state = ttc.check_point(pts)
        npt.assert_equal(state, int(StreamlineStatus.OUTSIDEIMAGE))


@set_random_number_generator()
def test_act_stopping_criterion(rng):
    """This tests that the act stopping criterion returns expected
    streamline statuses.
    """

    gm = rng.random((4, 4, 4))
    wm = rng.random((4, 4, 4))
    csf = rng.random((4, 4, 4))
    tissue_sum = gm + wm + csf
    gm /= tissue_sum
    wm /= tissue_sum
    csf /= tissue_sum

    act_tc = ActStoppingCriterion(include_map=gm, exclude_map=csf)

    # Test voxel center
    for ind in ndindex(wm.shape):
        pts = np.array(ind, dtype='float64')
        state = act_tc.check_point(pts)
        if csf[ind] > 0.5:
            npt.assert_equal(state, int(StreamlineStatus.INVALIDPOINT))
        elif gm[ind] > 0.5:
            npt.assert_equal(state, int(StreamlineStatus.ENDPOINT))
        else:
            npt.assert_equal(state, int(StreamlineStatus.TRACKPOINT))

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
            npt.assert_equal(state, int(StreamlineStatus.INVALIDPOINT))
        elif gm_res > 0.5:
            npt.assert_equal(state, int(StreamlineStatus.ENDPOINT))
        else:
            npt.assert_equal(state, int(StreamlineStatus.TRACKPOINT))

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        state = act_tc.check_point(pts)
        npt.assert_equal(state, int(StreamlineStatus.OUTSIDEIMAGE))


def test_cmc_stopping_criterion():
    """This tests that the cmc stopping criterion returns expected
    streamline statuses.
    """

    gm = np.array([[[1, 1], [0, 0], [0, 0]]])
    wm = np.array([[[0, 0], [1, 1], [0, 0]]])
    csf = np.array([[[0, 0], [0, 0], [1, 1]]])
    include_map = gm
    exclude_map = csf

    cmc_tc = CmcStoppingCriterion(include_map=include_map,
                                  exclude_map=exclude_map,
                                  step_size=1,
                                  average_voxel_size=1)
    cmc_tc_from_pve = CmcStoppingCriterion.from_pve(wm_map=wm,
                                                    gm_map=gm,
                                                    csf_map=csf,
                                                    step_size=1,
                                                    average_voxel_size=1)

    # Test constructors
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
            npt.assert_equal(state, int(StreamlineStatus.INVALIDPOINT))
        elif gm[ind] == 1:
            npt.assert_equal(state, int(StreamlineStatus.ENDPOINT))
        else:
            npt.assert_equal(state, int(StreamlineStatus.TRACKPOINT))

    # Test outside points
    outside_pts = [[100, 100, 100], [0, -1, 1], [0, 10, 2],
                   [0, 0.5, -0.51], [0, -0.51, 0.1]]
    for pts in outside_pts:
        pts = np.array(pts, dtype='float64')
        npt.assert_equal(cmc_tc.check_point(pts),
                         int(StreamlineStatus.OUTSIDEIMAGE))
        npt.assert_equal(cmc_tc.get_exclude(pts), 0)
        npt.assert_equal(cmc_tc.get_include(pts), 0)

