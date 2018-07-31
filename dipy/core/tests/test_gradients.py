import warnings

from nose.tools import assert_true, assert_raises
import numpy as np
import numpy.testing as npt

from dipy.data import get_data
from dipy.core.gradients import (gradient_table, GradientTable,
                                 gradient_table_from_bvals_bvecs,
                                 reorient_bvecs, generate_bvecs,
                                 check_multi_b)
from dipy.io.gradients import read_bvals_bvecs


def test_btable_prepare():

    sq2 = np.sqrt(2) / 2.
    bvals = 1500 * np.ones(7)
    bvals[0] = 0
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2]])
    bt = gradient_table(bvals, bvecs)
    npt.assert_array_equal(bt.bvecs, bvecs)
    # bt.info
    fimg, fbvals, fbvecs = get_data('small_64D')
    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)
    bvecs = np.where(np.isnan(bvecs), 0, bvecs)
    bt = gradient_table(bvals, bvecs)
    npt.assert_array_equal(bt.bvecs, bvecs)
    bt2 = gradient_table(bvals, bvecs.T)
    npt.assert_array_equal(bt2.bvecs, bvecs)
    btab = np.concatenate((bvals[:, None], bvecs), axis=1)
    bt3 = gradient_table(btab)
    npt.assert_array_equal(bt3.bvecs, bvecs)
    npt.assert_array_equal(bt3.bvals, bvals)
    bt4 = gradient_table(btab.T)
    npt.assert_array_equal(bt4.bvecs, bvecs)
    npt.assert_array_equal(bt4.bvals, bvals)
    # Test for proper inputs (expects either bvals/bvecs or 4 by n):
    assert_raises(ValueError, gradient_table, bvecs)


def test_GradientTable():

    gradients = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 1],
                          [3, 4, 0],
                          [5, 0, 12]], 'float')

    expected_bvals = np.array([0, 1, 1, 5, 13])
    expected_b0s_mask = expected_bvals == 0
    expected_bvecs = gradients / (expected_bvals + expected_b0s_mask)[:, None]

    gt = GradientTable(gradients, b0_threshold=0)
    npt.assert_array_almost_equal(gt.bvals, expected_bvals)
    npt.assert_array_equal(gt.b0s_mask, expected_b0s_mask)
    npt.assert_array_almost_equal(gt.bvecs, expected_bvecs)
    npt.assert_array_almost_equal(gt.gradients, gradients)

    gt = GradientTable(gradients, b0_threshold=1)
    npt.assert_array_equal(gt.b0s_mask, [1, 1, 1, 0, 0])
    npt.assert_array_equal(gt.bvals, expected_bvals)
    npt.assert_array_equal(gt.bvecs, expected_bvecs)

    npt.assert_raises(ValueError, GradientTable, np.ones((6, 2)))
    npt.assert_raises(ValueError, GradientTable, np.ones((6,)))


def test_gradient_table_from_bvals_bvecs():

    sq2 = np.sqrt(2) / 2
    bvals = [0, 1, 2, 3, 4, 5, 6, 0]
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2],
                      [0, 0, 0]])

    gt = gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=0)
    npt.assert_array_equal(gt.bvecs, bvecs)
    npt.assert_array_equal(gt.bvals, bvals)
    npt.assert_array_equal(gt.gradients, np.reshape(bvals, (-1, 1)) * bvecs)
    npt.assert_array_equal(gt.b0s_mask, [1, 0, 0, 0, 0, 0, 0, 1])

    # Test nans are replaced by 0
    new_bvecs = bvecs.copy()
    new_bvecs[[0, -1]] = np.nan
    gt = gradient_table_from_bvals_bvecs(bvals, new_bvecs, b0_threshold=0)
    npt.assert_array_equal(gt.bvecs, bvecs)

    # Bvalue > 0 for non-unit vector
    bad_bvals = [2, 1, 2, 3, 4, 5, 6, 0]
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bad_bvals,
                      bvecs, b0_threshold=0.)
    # num_gard inconsistent bvals, bvecs
    bad_bvals = np.ones(7)
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bad_bvals,
                      bvecs, b0_threshold=0.)
    # bvals not 1d
    bad_bvals = np.ones((1, 8))
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bad_bvals,
                      bvecs, b0_threshold=0.)
    # bvec not 2d
    bad_bvecs = np.ones((1, 8, 3))
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bvals,
                      bad_bvecs, b0_threshold=0.)
    # bvec not (N, 3)
    bad_bvecs = np.ones((8, 2))
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bvals,
                      bad_bvecs, b0_threshold=0.)
    # bvecs not unit vectors
    bad_bvecs = bvecs * 2
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bvals,
                      bad_bvecs, b0_threshold=0.)

    # Test **kargs get passed along
    gt = gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=0,
                                         big_delta=5, small_delta=2)
    npt.assert_equal(gt.big_delta, 5)
    npt.assert_equal(gt.small_delta, 2)


def test_b0s():

    sq2 = np.sqrt(2) / 2.
    bvals = 1500 * np.ones(8)
    bvals[0] = 0
    bvals[7] = 0
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2],
                      [0, 0, 0]])
    bt = gradient_table(bvals, bvecs)
    npt.assert_array_equal(np.where(bt.b0s_mask > 0)[0], np.array([0, 7]))
    npt.assert_array_equal(np.where(bt.b0s_mask == 0)[0], np.arange(1, 7))


def test_gtable_from_files():
    fimg, fbvals, fbvecs = get_data('small_101D')
    gt = gradient_table(fbvals, fbvecs)
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    npt.assert_array_equal(gt.bvals, bvals)
    npt.assert_array_equal(gt.bvecs, bvecs)


def test_deltas():
    sq2 = np.sqrt(2) / 2.
    bvals = 1500 * np.ones(7)
    bvals[0] = 0
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2]])
    bt = gradient_table(bvals, bvecs, big_delta=5, small_delta=2)
    npt.assert_equal(bt.big_delta, 5)
    npt.assert_equal(bt.small_delta, 2)


def test_qvalues():
    sq2 = np.sqrt(2) / 2.
    bvals = 1500 * np.ones(7)
    bvals[0] = 0
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2]])
    qvals = np.sqrt(bvals / 6) / (2 * np.pi)
    bt = gradient_table(bvals, bvecs, big_delta=8, small_delta=6)
    npt.assert_almost_equal(bt.qvals, qvals)


def test_reorient_bvecs():
    sq2 = np.sqrt(2) / 2
    bvals = np.concatenate([[0], np.ones(6) * 1000])
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2]])

    gt = gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=0)
    # The simple case: all affines are identity
    affs = np.zeros((6, 4, 4))
    for i in range(4):
        affs[:, i, i] = 1

    # We should get back the same b-vectors
    new_gt = reorient_bvecs(gt, affs)
    npt.assert_equal(gt.bvecs, new_gt.bvecs)

    # Now apply some rotations
    rotation_affines = []
    rotated_bvecs = bvecs[:]
    for i in np.where(~gt.b0s_mask)[0]:
        rot_ang = np.random.rand()
        cos_rot = np.cos(rot_ang)
        sin_rot = np.sin(rot_ang)
        rotation_affines.append(np.array([[1, 0, 0, 0],
                                          [0, cos_rot, -sin_rot, 0],
                                          [0, sin_rot, cos_rot, 0],
                                          [0, 0, 0, 1]]))
        rotated_bvecs[i] = np.dot(rotation_affines[-1][:3, :3],
                                  bvecs[i])

    # Copy over the rotation affines
    full_affines = rotation_affines[:]
    # And add some scaling and translation to each one to make this harder
    for i in range(len(full_affines)):
        full_affines[i] = np.dot(full_affines[i],
                                 np.array([[2.5, 0, 0, -10],
                                           [0, 2.2, 0, 20],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]]))

    gt_rot = gradient_table_from_bvals_bvecs(bvals,
                                             rotated_bvecs, b0_threshold=0)
    new_gt = reorient_bvecs(gt_rot, full_affines)
    # At the end of all this, we should be able to recover the original
    # vectors
    npt.assert_almost_equal(gt.bvecs, new_gt.bvecs)

    # We should be able to pass just the 3-by-3 rotation components to the same
    # effect
    new_gt = reorient_bvecs(gt_rot, np.array(rotation_affines)[:, :3, :3])
    npt.assert_almost_equal(gt.bvecs, new_gt.bvecs)

    # Verify that giving the wrong number of affines raises an error:
    full_affines.append(np.zeros((4, 4)))
    assert_raises(ValueError, reorient_bvecs, gt_rot, full_affines)


def test_nan_bvecs():
    """
    Test that the presence of nan's in b-vectors doesn't raise warnings.

    In previous versions, the presence of NaN in b-vectors was taken to
    indicate a 0 b-value, but also raised a warning when testing for the length
    of these vectors. This checks that it doesn't happen.
    """
    fdata, fbvals, fbvecs = get_data()
    with warnings.catch_warnings(record=True) as w:
        gradient_table(fbvals, fbvecs)
        npt.assert_(len(w) == 0)


def test_generate_bvecs():
    """Tests whether we have properly generated bvecs.
    """
    # Test if the generated bvectors are unit vectors
    bvecs = generate_bvecs(100)
    norm = [np.linalg.norm(v) for v in bvecs]
    npt.assert_almost_equal(norm, np.ones(100))

    # Test if two generated vectors are almost orthogonal
    bvecs_2 = generate_bvecs(2)
    cos_theta = np.dot(bvecs_2[0], bvecs_2[1])
    npt.assert_almost_equal(cos_theta, 0., decimal=6)


def test_check_multi_b():
    bvals = np.array([1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 0])
    bvecs = generate_bvecs(bvals.shape[-1])
    gtab = gradient_table(bvals, bvecs)
    npt.assert_(check_multi_b(gtab, 2, non_zero=False))

    # We don't consider differences this small to be sufficient:
    bvals = np.array([1995, 1995, 1995, 1995, 2005, 2005, 2005, 2005, 0])
    bvecs = generate_bvecs(bvals.shape[-1])
    gtab = gradient_table(bvals, bvecs)
    npt.assert_(not check_multi_b(gtab, 2, non_zero=True))

    # Unless you specify that you are interested in this magnitude of changes:
    npt.assert_(check_multi_b(gtab, 2, non_zero=True, bmag=1))

    # Or if you consider zero to be one of your b-values:
    npt.assert_(check_multi_b(gtab, 2, non_zero=False))


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
