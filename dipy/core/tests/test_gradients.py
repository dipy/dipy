from nose.tools import assert_true

import numpy as np
import numpy.testing as npt

from dipy.data import get_data
from dipy.core.gradients import (gradient_table, GradientTable,
                                 gradient_table_from_bvals_bvecs)
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
    bt.info
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

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
