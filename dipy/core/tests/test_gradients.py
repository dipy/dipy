import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.core.gradients import (b0_threshold_empty_gradient_message,
                                 b0_threshold_update_slicing_message,
                                 gradient_table, GradientTable,
                                 gradient_table_from_bvals_bvecs,
                                 gradient_table_from_qvals_bvecs,
                                 gradient_table_from_gradient_strength_bvecs,
                                 WATER_GYROMAGNETIC_RATIO,
                                 mask_non_weighted_bvals,
                                 orientation_to_string,
                                 reorient_bvecs, generate_bvecs,
                                 check_multi_b, round_bvals, get_bval_indices,
                                 unique_bvals_magnitude,
                                 unique_bvals_tolerance, unique_bvals,
                                 params_to_btens, btens_to_params,
                                 orientation_from_string, reorient_vectors)
from dipy.core.geometry import vec2vec_rotmat, vector_norm
from dipy.io.gradients import read_bvals_bvecs
from dipy.utils.deprecator import ExpiredDeprecationError
from dipy.testing import clear_and_catch_warnings
from dipy.testing.decorators import set_random_number_generator


def test_unique_bvals_deprecated():
    npt.assert_raises(ExpiredDeprecationError, unique_bvals,
                      np.array([0, 800, 1400, 1401, 1405]))


def test_mask_non_weighted_bvals():

    bvals = np.array([0., 100., 200., 300., 400.])
    b0_threshold = 0.
    expected_val = np.asarray([True, False, False, False, False])
    obtained_val = mask_non_weighted_bvals(bvals, b0_threshold)
    assert np.array_equal(obtained_val, expected_val)

    b0_threshold = 50
    obtained_val = mask_non_weighted_bvals(bvals, b0_threshold)
    assert np.array_equal(obtained_val, expected_val)

    b0_threshold = 200.
    expected_val = np.asarray([True, True, True, False, False])
    obtained_val = mask_non_weighted_bvals(bvals, b0_threshold)
    assert np.array_equal(obtained_val, expected_val)


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
    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
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
    npt.assert_raises(ValueError, gradient_table, bvecs)


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
    npt.assert_('B-values shape (5,)' in gt.__str__())
    npt.assert_array_almost_equal(gt.bvals, expected_bvals)
    npt.assert_array_equal(gt.b0s_mask, expected_b0s_mask)
    npt.assert_array_almost_equal(gt.bvecs, expected_bvecs)
    npt.assert_array_almost_equal(gt.gradients, gradients)

    gt = GradientTable(gradients, b0_threshold=1)
    npt.assert_array_equal(gt.b0s_mask, [1, 1, 1, 0, 0])
    npt.assert_array_equal(gt.bvals, expected_bvals)
    npt.assert_array_equal(gt.bvecs, expected_bvecs)

    # checks negative values in gtab
    npt.assert_raises(ValueError, GradientTable, -1)
    npt.assert_raises(ValueError, GradientTable, np.ones((6, 2)))
    npt.assert_raises(ValueError, GradientTable, np.ones((6,)))

    with warnings.catch_warnings(record=True) as l_warns:
        bad_gt = gradient_table(expected_bvals, expected_bvecs,
                                b0_threshold=200)

        # Select only UserWarning
        selected_w = [w for w in l_warns
                      if issubclass(w.category, UserWarning)]
        assert len(selected_w) >= 1
        msg = [str(m.message) for m in selected_w]
        npt.assert_equal('b0_threshold has a value > 199' in msg, True)


def test_GradientTable_btensor_calculation():

    # Generate a gradient table without specifying b-tensors
    gradients = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [3, 4, 0],
                          [5, 0, 12]], 'float')

    # Check that when btens attribute not specified it takes the value of None
    gt = GradientTable(gradients)
    npt.assert_equal(gt.btens, None)

    # Check that btens are correctly created if specified
    gt = GradientTable(gradients, btens='LTE')

    # Check that the number of b tensors is correct
    npt.assert_equal(gt.btens.shape[0], gt.bvals.shape[0])
    for i, (bval, bten) in enumerate(zip(gt.bvals, gt.btens)):
        # Check that the b tensor magnitude is correct
        npt.assert_almost_equal(np.trace(bten), bval)
        # Check that the b tensor orientation is correct
        if bval != 0:
            evals, evecs = np.linalg.eig(bten)
            dot_prod = np.dot(np.real(evecs[:, np.argmax(evals)]), gt.bvecs[i])
            npt.assert_almost_equal(np.abs(dot_prod), 1)

    # Check btens input option 1
    for btens in ['LTE', 'PTE', 'STE', 'CTE']:
        gt = GradientTable(gradients, btens=btens)
        # Check that the number of b tensors is correct
        npt.assert_equal(gt.bvals.shape[0], gt.btens.shape[0])
        for i, (bval, bvec, bten) in enumerate(zip(gt.bvals, gt.bvecs,
                                                   gt.btens)):
            # Check that the b tensor magnitude is correct
            npt.assert_almost_equal(np.trace(bten), bval)
            # Check that the b tensor orientation is correct
            if btens == ('LTE' or 'CTE'):
                if bval != 0:
                    evals, evecs = np.linalg.eig(bten)
                    dot_prod = np.dot(np.real(evecs[:, np.argmax(evals)]),
                                      bvec)
                    npt.assert_almost_equal(np.abs(dot_prod), 1)
            elif btens == 'PTE':
                if bval != 0:
                    evals, evecs = np.linalg.eig(bten)
                    dot_prod = np.dot(np.real(evecs[:, np.argmin(evals)]),
                                      bvec)
                    npt.assert_almost_equal(np.abs(dot_prod), 1)

    # Check btens input option 2
    btens = np.array(['LTE', 'PTE', 'STE', 'CTE', 'LTE', 'PTE'])
    gt = GradientTable(gradients, btens=btens)
    # Check that the number of b tensors is correct
    npt.assert_equal(gt.bvals.shape[0], gt.btens.shape[0])
    for i, (bval, bvec, bten) in enumerate(zip(gt.bvals, gt.bvecs,
                                               gt.btens)):
        # Check that the b tensor magnitude is correct
        npt.assert_almost_equal(np.trace(bten), bval)
        # Check that the b tensor orientation is correct
        if btens[i] == ('LTE' or 'CTE'):
            if bval != 0:
                evals, evecs = np.linalg.eig(bten)
                dot_prod = np.dot(np.real(evecs[:, np.argmax(evals)]), bvec)
                npt.assert_almost_equal(np.abs(dot_prod), 1)
        elif btens[i] == 'PTE':
            if bval != 0:
                evals, evecs = np.linalg.eig(bten)
                dot_prod = np.dot(np.real(evecs[:, np.argmin(evals)]), bvec)
                npt.assert_almost_equal(np.abs(dot_prod), 1)

    # Check btens input option 3
    btens = np.array([np.eye(3, 3) for i in range(6)])
    gt = GradientTable(gradients, btens=btens)
    npt.assert_equal(btens, gt.btens)
    npt.assert_equal(gt.bvals.shape[0], gt.btens.shape[0])

    # Check invalid input
    npt.assert_raises(ValueError, GradientTable, gradients=gradients,
                      btens='PPP')
    npt.assert_raises(ValueError, GradientTable, gradients=gradients,
                      btens=np.array([np.eye(3, 3) for i in range(10)]))
    npt.assert_raises(ValueError, GradientTable, gradients=gradients,
                      btens=np.zeros((10, 10)))


def test_gradient_table_from_qvals_bvecs():
    qvals = 30. * np.ones(7)
    big_delta = .03  # pulse separation of 30ms
    small_delta = 0.01  # pulse duration of 10ms
    qvals[0] = 0
    sq2 = np.sqrt(2) / 2
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2]])
    gt = gradient_table_from_qvals_bvecs(qvals, bvecs,
                                         big_delta, small_delta)

    bvals_expected = (qvals * 2 * np.pi) ** 2 * (big_delta - small_delta / 3.)
    gradient_strength_expected = qvals * 2 * np.pi /\
        (small_delta * WATER_GYROMAGNETIC_RATIO)
    npt.assert_almost_equal(gt.gradient_strength, gradient_strength_expected)
    npt.assert_almost_equal(gt.bvals, bvals_expected)


def test_gradient_table_from_gradient_strength_bvecs():
    gradient_strength = .03e-3 * np.ones(7)  # clinical strength at 30 mT/m
    big_delta = .03  # pulse separation of 30ms
    small_delta = 0.01  # pulse duration of 10ms
    gradient_strength[0] = 0
    sq2 = np.sqrt(2) / 2
    bvecs = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [sq2, sq2, 0],
                      [sq2, 0, sq2],
                      [0, sq2, sq2]])
    gt = gradient_table_from_gradient_strength_bvecs(gradient_strength, bvecs,
                                                     big_delta, small_delta)
    qvals_expected = (gradient_strength * WATER_GYROMAGNETIC_RATIO *
                      small_delta / (2 * np.pi))
    bvals_expected = (qvals_expected * 2 * np.pi) ** 2 *\
                     (big_delta - small_delta / 3.)
    npt.assert_almost_equal(gt.qvals, qvals_expected)
    npt.assert_almost_equal(gt.bvals, bvals_expected)


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

    # b-value > 0 for non-unit vector
    bad_bvals = [2, 1, 2, 3, 4, 5, 6, 0]
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bad_bvals,
                      bvecs, b0_threshold=0.)
    # num_gard inconsistent bvals, bvecs
    bad_bvals = np.ones(7)
    npt.assert_raises(ValueError, gradient_table_from_bvals_bvecs, bad_bvals,
                      bvecs, b0_threshold=0.)
    # negative bvals
    bad_bvals = [-1, -1, -1, -5, -6, -10]
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


def test_gradient_table_special_bvals_bvecs_case():
    bvals = [0, 1]
    bvecs = np.array(
        [[0, 0, 0],
         [0, 0, 1]]
    )
    gt = gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=0)
    npt.assert_array_equal(gt.bvecs, bvecs)
    npt.assert_array_equal(gt.bvals, bvals)


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
    fimg, fbvals, fbvecs = get_fnames('small_101D')
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


@set_random_number_generator()
def test_reorient_bvecs(rng):
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
    affs = np.zeros((4, 4, 6))
    for i in range(4):
        affs[i, i, :] = 1

    # We should get back the same b-vectors
    new_gt = reorient_bvecs(gt, affs)
    npt.assert_equal(gt.bvecs, new_gt.bvecs)

    # Now apply some rotations
    rotation_affines = []
    rotated_bvecs = bvecs[:]
    for i in np.where(~gt.b0s_mask)[0]:
        rot_ang = rng.random()
        cos_rot = np.cos(rot_ang)
        sin_rot = np.sin(rot_ang)
        rotation_affines.append(np.array([[1, 0, 0, 0],
                                          [0, cos_rot, -sin_rot, 0],
                                          [0, sin_rot, cos_rot, 0],
                                          [0, 0, 0, 1]]))
        rotated_bvecs[i] = np.dot(rotation_affines[-1][:3, :3],
                                  bvecs[i])

    rotation_affines = np.stack(rotation_affines, axis=-1)
    # Copy over the rotation affines
    full_affines = rotation_affines[:]
    # And add some scaling and translation to each one to make this harder
    for i in range(full_affines.shape[-1]):
        full_affines[..., i] = np.dot(full_affines[..., i],
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
    new_gt = reorient_bvecs(gt_rot, np.array(rotation_affines)[:3, :3])
    npt.assert_almost_equal(gt.bvecs, new_gt.bvecs)

    # Verify that giving the wrong number of affines raises an error:
    full_affines = np.concatenate([full_affines, np.zeros((4, 4, 1))], axis=-1)
    npt.assert_raises(ValueError, reorient_bvecs, gt_rot, full_affines)

    # Shear components in the matrix need to be decomposed into rotation only,
    # and should not lead to scaling of the bvecs
    shear_affines = []
    for i in np.where(~gt.b0s_mask)[0]:
        shear_affines.append(np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]))
    shear_affines = np.stack(shear_affines, axis=-1)
    # atol is set to 1 here to do the scaling verification here,
    # so that the reorient_bvecs function does not throw an error itself
    new_gt = reorient_bvecs(gt, shear_affines[:3, :3], atol=1)
    bvecs_close_to_1 = abs(vector_norm(new_gt.bvecs[~gt.b0s_mask]) - 1) <= 0.001
    npt.assert_(np.all(bvecs_close_to_1))


def test_nan_bvecs():
    """
    Test that the presence of nan's in b-vectors doesn't raise warnings.

    In previous versions, the presence of NaN in b-vectors was taken to
    indicate a 0 b-value, but also raised a warning when testing for the length
    of these vectors. This checks that it doesn't happen.
    """
    fdata, fbvals, fbvecs = get_fnames()
    with warnings.catch_warnings(record=True) as l_warns:
        gradient_table(fbvals, fbvecs)
        # Select only UserWarning
        selected_w = [w for w in l_warns
                      if issubclass(w.category, UserWarning)]
        npt.assert_(len(selected_w) == 0)


@set_random_number_generator()
def test_generate_bvecs(rng):
    """Tests whether we have properly generated bvecs.
    """
    # Test if the generated b-vectors are unit vectors
    bvecs = generate_bvecs(100, rng=rng)
    norm = [np.linalg.norm(v) for v in bvecs]
    npt.assert_almost_equal(norm, np.ones(100))

    # Test if two generated vectors are almost orthogonal
    bvecs_2 = generate_bvecs(2, rng=rng)
    cos_theta = np.dot(bvecs_2[0], bvecs_2[1])
    npt.assert_almost_equal(cos_theta, 0., decimal=6)


def test_getitem_idx():
    # Create a GradientTable object with some test b-values and b-vectors
    bvals = np.array([0., 100., 200., 300., 400.])
    # value should be in increasing order as b-value affects the diffusion
    # weighting of the image, and the amount of diffusion weighting increases
    # with increasing b-value.
    bvecs = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    # the b-vectors should be unit-length vectors
    gradients = bvals[:, None] * bvecs

    # Test a too large b0 threshold value
    b0_threshold = 100
    gtab = GradientTable(gradients, b0_threshold=b0_threshold)

    idx = 1
    with pytest.raises(ValueError) as excinfo:
        _ = gtab[idx]
        assert str(excinfo.value) == b0_threshold_empty_gradient_message(
            bvals, [idx], b0_threshold)

    gtab = GradientTable(gradients)

    # Test with a single index
    gtab_slice1 = gtab[1]
    assert np.array_equal(gtab_slice1.bvals, np.array([100.]))
    assert np.array_equal(gtab_slice1.bvecs, np.array([[1., 0., 0.]]))

    # Test with a range of indices
    gtab = GradientTable(gradients)
    idx_start = 2
    idx_end = 5
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=b0_threshold_update_slicing_message(idx_start),
            category=UserWarning)
        gtab_slice2 = gtab[idx_start:idx_end]
        assert np.array_equal(gtab_slice2.bvals, np.array([200., 300., 400.]))
        assert np.array_equal(gtab_slice2.bvecs,
                              np.array([[0., 1., 0.], [0., 0., 1.],
                                        [1., 0., 0.]]))


def test_round_bvals():
    bvals_gt = np.array([1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 0])
    b = round_bvals(bvals_gt)
    npt.assert_array_almost_equal(bvals_gt, b)

    bvals = np.array([995, 995, 995, 995, 2005, 2005, 2005, 2005, 0])
    # We don't consider differences this small to be sufficient:
    b = round_bvals(bvals)
    npt.assert_array_almost_equal(bvals_gt, b)

    # Unless you specify that you are interested in this magnitude of changes:
    b = round_bvals(bvals, bmag=0)
    npt.assert_array_almost_equal(bvals, b)

    # Case that b-values are in ms/um2
    bvals = np.array([0.995, 0.995, 0.995, 0.995, 2.005, 2.005, 2.005, 2.005,
                      0])
    b = round_bvals(bvals)
    bvals_gt = np.array([1, 1, 1, 1, 2, 2, 2, 2, 0])
    npt.assert_array_almost_equal(bvals_gt, b)


def test_unique_bvals_tolerance():
    bvals = np.array([1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 0])
    ubvals_gt = np.array([0, 1000, 2000])
    b = unique_bvals_tolerance(bvals)
    npt.assert_array_almost_equal(ubvals_gt, b)

    # Testing the tolerance factor on many b-values that are within tol.
    bvals = np.array([950, 980, 995, 1000, 1000, 1010, 1999, 2000, 2001, 0])
    ubvals_gt = np.array([0, 950, 1000, 2001])
    b = unique_bvals_tolerance(bvals)
    npt.assert_array_almost_equal(ubvals_gt, b)

    # All unique b-values are kept if tolerance is set to zero:
    bvals = np.array([990, 990, 1000, 1000, 2000, 2000, 2050, 2050, 0])
    ubvals_gt = np.array([0, 990, 1000, 2000, 2050])
    b = unique_bvals_tolerance(bvals, 0)
    npt.assert_array_almost_equal(ubvals_gt, b)

    # Case that b-values are in ms/um2
    bvals = np.array([0.995, 0.995, 0.995, 0.995, 2.005, 2.005, 2.005, 2.005,
                      0])
    b = unique_bvals_tolerance(bvals, 0.5)
    ubvals_gt = np.array([0, 0.995, 2.005])
    npt.assert_array_almost_equal(ubvals_gt, b)


def test_get_bval_indices():
    bvals = np.array([1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 0])
    indices_gt = np.array([0, 1, 2, 3])
    indices = get_bval_indices(bvals, 1000)
    npt.assert_array_almost_equal(indices_gt, indices)

    # Testing the tolerance factor on many b-values that are within tol.
    bvals = np.array([950, 980, 995, 1000, 1000, 1010, 1999, 2000, 2001, 0])
    indices_gt = np.array([0])
    indices = get_bval_indices(bvals, 950, 20)
    npt.assert_array_almost_equal(indices_gt, indices)
    indices_gt = np.array([1, 2, 3, 4, 5])
    indices = get_bval_indices(bvals, 1000, 20)
    npt.assert_array_almost_equal(indices_gt, indices)
    indices_gt = np.array([6, 7, 8])
    indices = get_bval_indices(bvals, 2001, 20)
    npt.assert_array_almost_equal(indices_gt, indices)

    # All unique b-values indices are returned if tolerance is set to zero:
    bvals = np.array([990, 990, 1000, 1000, 2000, 2000, 2050, 2050, 0])
    indices_gt = np.array([2, 3])
    indices = get_bval_indices(bvals, 1000, 0)
    npt.assert_array_almost_equal(indices_gt, indices)

    # Case that b-values are in ms/um2
    bvals = np.array([0.995, 0.995, 0.995, 0.995, 2.005, 2.005, 2.005, 2.005,
                      0])
    indices_gt = np.array([0, 1, 2, 3])
    indices = get_bval_indices(bvals, 0.995, 0.5)
    npt.assert_array_almost_equal(indices_gt, indices)


def test_unique_bvals_magnitude():
    bvals = np.array([1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 0])
    ubvals_gt = np.array([0, 1000, 2000])
    b = unique_bvals_magnitude(bvals)
    npt.assert_array_almost_equal(ubvals_gt, b)

    bvals = np.array([995, 995, 995, 995, 2005, 2005, 2005, 2005, 0])
    # Case that b-values are rounded:
    b = unique_bvals_magnitude(bvals)
    npt.assert_array_almost_equal(ubvals_gt, b)

    # b-values are not rounded if you specific the magnitude of the values
    # precision:
    b = unique_bvals_magnitude(bvals, bmag=0)
    npt.assert_array_almost_equal(b, np.array([0, 995, 2005]))

    # Case that b-values are in ms/um2
    bvals = np.array([0.995, 0.995, 0.995, 0.995, 2.005, 2.005, 2.005, 2.005,
                      0])
    b = unique_bvals_magnitude(bvals)
    ubvals_gt = np.array([0, 1, 2])
    npt.assert_array_almost_equal(ubvals_gt, b)

    # Test case that optional parameter round_bvals is set to true
    bvals = np.array([995, 1000, 1004, 1000, 2001, 2000, 1988, 2017, 0])
    ubvals_gt = np.array([0, 1000, 2000])
    rbvals_gt = np.array([1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 0])
    ub, rb = unique_bvals_magnitude(bvals, rbvals=True)
    npt.assert_array_almost_equal(ubvals_gt, ub)
    npt.assert_array_almost_equal(rbvals_gt, rb)


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
    npt.assert_(check_multi_b(gtab, 2, non_zero=True, bmag=0))

    # Or if you consider zero to be one of your b-values:
    npt.assert_(check_multi_b(gtab, 2, non_zero=False))

    # Case that b-values are in ms/um2 (this should successfully pass)
    bvals = np.array([0.995, 0.995, 0.995, 0.995, 2.005, 2.005, 2.005, 2.005,
                      0])
    bvecs = generate_bvecs(bvals.shape[-1])
    gtab = gradient_table(bvals, bvecs)
    npt.assert_(check_multi_b(gtab, 2, non_zero=False))


@set_random_number_generator()
def test_btens_to_params(rng):
    """
    Checks if bvals, bdeltas and b_etas are as expected for 4 b-tensor shapes
    (LTE, PTE, STE, CTE) as well as scaled and rotated versions of them

    This function intrinsically tests the function `_btens_to_params_2d` as
    `_btens_to_params_2d` is only meant to be called by `btens_to_params`

    """
    n_rotations = 30
    n_scales = 3

    expected_bvals = np.array([1, 1, 1, 1])
    expected_bdeltas = np.array([1, -0.5, 0, 0.5])
    expected_b_etas = np.array([0, 0, 0, 0])

    # Baseline tensors to test
    linear_tensor = np.array([[1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
    planar_tensor = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]) / 2
    spherical_tensor = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]]) / 3
    cigar_tensor = np.array([[2, 0, 0],
                             [0, .5, 0],
                             [0, 0, .5]]) / 3

    base_tensors = [linear_tensor, planar_tensor,
                    spherical_tensor, cigar_tensor]

    # ---------------------------------
    # Test function on baseline tensors
    # ---------------------------------

    # Loop through each tensor type and check results
    for i, tensor in enumerate(base_tensors):
        i_bval, i_bdelta, i_b_eta = btens_to_params(tensor)

        npt.assert_array_almost_equal(i_bval, expected_bvals[i])
        npt.assert_array_almost_equal(i_bdelta, expected_bdeltas[i])
        npt.assert_array_almost_equal(i_b_eta, expected_b_etas[i])

    # Test function on a 3D input
    base_tensors_array = np.empty((4, 3, 3))
    base_tensors_array[0, :, :] = linear_tensor
    base_tensors_array[1, :, :] = planar_tensor
    base_tensors_array[2, :, :] = spherical_tensor
    base_tensors_array[3, :, :] = cigar_tensor

    bvals, bdeltas, b_etas = btens_to_params(base_tensors_array)

    npt.assert_array_almost_equal(bvals, expected_bvals)
    npt.assert_array_almost_equal(bdeltas, expected_bdeltas)
    npt.assert_array_almost_equal(b_etas, expected_b_etas)

    # -----------------------------------------------------
    # Test function after rotating+scaling baseline tensors
    # -----------------------------------------------------

    scales = np.concatenate((np.array([1]), rng.random(n_scales)))

    for scale in scales:

        ebs = expected_bvals*scale

        # Generate `n_rotations` random 3-element vectors of norm 1
        v = rng.random((n_rotations, 3)) - 0.5
        u = np.apply_along_axis(lambda w: w/np.linalg.norm(w), axis=1, arr=v)

        for rot_idx in range(n_rotations):

            # Get rotation matrix for current iteration
            u_i = u[rot_idx, :]
            R_i = vec2vec_rotmat(np.array([1, 0, 0]), u_i)

            # Rotate each of the baseline test tensors and check results
            for i, tensor in enumerate(base_tensors):

                tensor_rot_i = np.matmul(np.matmul(R_i, tensor), R_i.T)
                i_bval, i_bdelta, i_b_eta = btens_to_params(tensor_rot_i*scale)

                npt.assert_array_almost_equal(i_bval, ebs[i])
                npt.assert_array_almost_equal(i_bdelta, expected_bdeltas[i])
                npt.assert_array_almost_equal(i_b_eta, expected_b_etas[i])

    # Input can't be string
    npt.assert_raises(ValueError, btens_to_params, 'LTE')

    # Input can't be list of strings
    npt.assert_raises(ValueError, btens_to_params, ['LTE', 'LTE'])

    # Input can't be 1D nor 4D
    npt.assert_raises(ValueError, btens_to_params, np.zeros((3,)))
    npt.assert_raises(ValueError, btens_to_params, np.zeros((3, 3, 3, 3)))

    # Input shape must be (3, 3) OR (N, 3, 3)
    npt.assert_raises(ValueError, btens_to_params, np.zeros((4, 4)))
    npt.assert_raises(ValueError, btens_to_params, np.zeros((2, 2, 2)))


def test_params_to_btens():
    """
    Checks if `params_to_btens` generates the expected b-tensors from provided
    `bvals`, `bdeltas`, `b_etas`.

    """
    # Test parameters that should generate "baseline" b-tensors
    bvals = [1, 1, 1, 1]
    bdeltas = [0, -0.5, 0.5, 1]
    b_etas = [0, 0, 0, 0]

    expected_btens = [
        np.eye(3) / 3,
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) / 2,
        np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 2]]) / 3,
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    ]

    for i, (bval, bdelta, b_eta) in enumerate(zip(bvals, bdeltas, b_etas)):
        btens = params_to_btens(bval, bdelta, b_eta)
        npt.assert_array_almost_equal(btens, expected_btens[i])

    # Additional tests
    bvals = [1.7, 0.4, 2.3]
    bdeltas = [0.6, -0.2, 0]
    b_etas = [0.3, 0.8, 0.7]

    expected_btens = [
        np.array([[0.12466667, 0, 0],
                  [0, 0.32866667, 0],
                  [0, 0, 1.24666667]]),
        np.array([[0.18133333, 0, 0],
                  [0, 0.13866667, 0],
                  [0, 0, 0.08]]),
        np.array([[0.76666667, 0, 0],
                  [0, 0.76666667, 0],
                  [0, 0, 0.76666667]])
    ]

    for i, (bval, bdelta, b_eta) in enumerate(zip(bvals, bdeltas, b_etas)):
        btens = params_to_btens(bval, bdelta, b_eta)
        npt.assert_array_almost_equal(btens, expected_btens[i])

    # Tests to trigger value errors
    # 1: wrong input type
    # 2: bval out of valid range
    # 3: bdelta out of valid range
    # 4: b_eta out of valid range
    bvals = [np.array([1]), -1, 1, 1]
    bdeltas = [0, 0, -1, 1]
    b_etas = [0, 0, 0, -1]

    for i, (bval, bdelta, b_eta) in enumerate(zip(bvals, bdeltas, b_etas)):
        npt.assert_raises(ValueError, params_to_btens, bval, bdelta, b_eta)


def test_orientation_from_to_string():
    with clear_and_catch_warnings():
        ras = np.array(((0, 1), (1, 1), (2, 1)))
        lps = np.array(((0, -1), (1, -1), (2, 1)))
        asl = np.array(((1, 1), (2, 1), (0, -1)))
        npt.assert_array_equal(orientation_from_string('ras'), ras)
        npt.assert_array_equal(orientation_from_string('lps'), lps)
        npt.assert_array_equal(orientation_from_string('asl'), asl)
        npt.assert_raises(ValueError, orientation_from_string, 'aasl')

        assert orientation_to_string(ras) == 'ras'
        assert orientation_to_string(lps) == 'lps'
        assert orientation_to_string(asl) == 'asl'


def test_reorient_vectors():
    with clear_and_catch_warnings():
        bvec = np.arange(12).reshape((3, 4))
        npt.assert_array_equal(reorient_vectors(bvec, 'ras', 'ras'), bvec)
        npt.assert_array_equal(reorient_vectors(bvec, 'ras', 'lpi'), -bvec)
        result = bvec[[1, 2, 0]]
        npt.assert_array_equal(reorient_vectors(bvec, 'ras', 'asr'), result)
        bvec = result
        result = bvec[[1, 0, 2]] * [[-1], [1], [-1]]
        npt.assert_array_equal(reorient_vectors(bvec, 'asr', 'ial'), result)
        result = bvec[[1, 0, 2]] * [[-1], [1], [1]]
        npt.assert_array_equal(reorient_vectors(bvec, 'asr', 'iar'), result)
        npt.assert_raises(ValueError, reorient_vectors, bvec, 'ras', 'ra')

        bvec = np.arange(12).reshape((3, 4))
        bvec = bvec.T
        npt.assert_array_equal(reorient_vectors(bvec, 'ras', 'ras', axis=1),
                               bvec)
        npt.assert_array_equal(reorient_vectors(bvec, 'ras', 'lpi', axis=1),
                               -bvec)
        result = bvec[:, [1, 2, 0]]
        npt.assert_array_equal(reorient_vectors(bvec, 'ras', 'asr', axis=1),
                               result)
        bvec = result
        result = bvec[:, [1, 0, 2]] * [-1, 1, -1]
        npt.assert_array_equal(reorient_vectors(bvec, 'asr', 'ial', axis=1),
                               result)
        result = bvec[:, [1, 0, 2]] * [-1, 1, 1]
        npt.assert_array_equal(reorient_vectors(bvec, 'asr', 'iar', axis=1),
                               result)
    bvec = np.arange(12).reshape((3, 4))


def test_affine_input_change():
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
    
    # Wrong affine dimension
    affs = np.zeros((6, 4, 4))
    for i in range(4):
        affs[:, i, i] = 1

    npt.assert_warns(Warning, reorient_bvecs, gt, affs)

    # Check if list still works
    affs = [np.eye(4) for _ in range(6)]
    _ = reorient_bvecs(gt, affs)
