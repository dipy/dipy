import os

import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec
import scipy.sparse as sps
import scipy.linalg as la

import nibabel as nib

import dipy.tracking.life as life
import dipy.core.sphere as dps
import dipy.core.gradients as dpg
from dipy.data import get_data


import numpy as np
import numpy.testing as npt


def test_sgd():
    # Set up the regression:
    beta = np.random.rand(10)
    X = np.random.randn(1000,10)
    y = np.dot(X, beta)
    beta_hat = life.sparse_sgd(y,X, plot=False, lamda=0)
    beta_hat_sparse = life.sparse_sgd(y, sps.csr_matrix(X), plot=False, lamda=0)
    # We should be able to get back the right answer for this simple case
    npt.assert_array_almost_equal(beta, beta_hat, decimal=1)
    npt.assert_array_almost_equal(beta, beta_hat_sparse, decimal=1)


def test_sl_gradients():
    sl = [[1,2,3], [4,5,6], [5,6,7]]
    grads = np.array([[3,3,3], [2,2,2], [1,1,1]])
    npt.assert_array_equal(life.sl_gradients(sl), grads)


def test_sl_tensors():
    # Small streamline
    sl = [[1,2,3], [4,5,3], [5,6,3]]
    # Non-default eigenvalues:
    evals=[0.0012, 0.0006, 0.0004]
    sl_tensors = life.sl_tensors(sl, evals=evals)
    # Get the resulting eigenvalues:
    eigvals = la.eigvals(sl_tensors[0])
    # la.eigvals returns things in a strange order, so reorder them:
    idx = np.argsort(eigvals)[::-1]
    # The eigenvalues are the same:
    npt.assert_array_almost_equal(eigvals[idx], evals)
    eigvecs = la.eig(sl_tensors[0])[1][idx]
    # The rotation on the first vector is 45 degrees:
    npt.assert_almost_equal(np.rad2deg(np.arccos(np.dot(eigvecs[0], [1, 0, 0]))),
                            45)

    # The rotation on the first vector is 135 degrees:
    npt.assert_almost_equal(np.rad2deg(np.arccos(np.dot(eigvecs[1], [0, 1, 0]))),
                            135)

    # The rotation on the last vector is 0 degrees (the same coordinate in all
    # three z components):
    npt.assert_almost_equal(np.rad2deg(np.arccos(np.dot(eigvecs[2], [0, 0, 1]))),
                            0)

    # Another small streamline
    sl = [[1,0,0], [2,0,0], [3,0,0]]
    sl_tensors = life.sl_tensors(sl, evals=evals)

    for t in sl_tensors:
        eigvals = la.eigvals(t)
        idx = np.argsort(eigvals)[::-1]
        # The eigenvalues are the same:
        npt.assert_array_almost_equal(eigvals[idx], evals)
        # This one has no rotations - all tensors are simply the canonical:
        eigvecs = la.eig(sl_tensors[0])[1][idx]
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[0], [1, 0, 0]))), 0)
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[1], [0, 1, 0]))), 0)
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[2], [0, 0, 1]))), 0)


def test_sl_signal():
    data_file, bval_file, bvec_file = get_data('small_64D')
    gtab = dpg.gradient_table(bval_file, bvec_file)
    evals = [0.0015, 0.0005, 0.0005]
    sl1 = [[[1,2,3], [4,5,3], [5,6,3], [6,7,3]],
          [[1,2,3], [4,5,3], [5,6,3]]]

    sig1 = [life.sl_signal(s, gtab, evals) for s in sl1]

    sl2 = [[[1,2,3], [4,5,3], [5,6,3], [6,7,3]]]

    sig2 = [life.sl_signal(s, gtab, evals) for s in sl2]

    npt.assert_array_equal(sl2[0], sl1[0])


def test_voxel2fiber():
    sl = [[[1,2,3], [4,5,3], [5,6,3], [6,7,3]],
          [[1,2,3], [4,5,3], [5,6,3]]]
    affine = np.eye(4)
    v2f, v2fn = life.voxel2fiber(sl, False, affine)
    npt.assert_array_equal(v2f, np.array([[1, 1], [1, 1], [1,  1],[1, 0]]))
    npt.assert_array_equal(v2fn, np.array([[ 0, 1, 2, 3], [ 0, 1, 2, np.nan]]))


def test_FiberModel_init():
    # Get some small amount of data:
    data_file, bval_file, bvec_file = get_data('small_64D')
    data_ni = nib.load(data_file)
    data = data_ni.get_data()
    data_aff = data_ni.get_affine()
    bvals, bvecs = (np.load(f) for f in (bval_file, bvec_file))
    gtab = dpg.gradient_table(bvals, bvecs)
    FM = life.FiberModel(gtab)

    sl = [[[1,2,3], [4,5,3], [5,6,3], [6,7,3]],
          [[1,2,3], [4,5,3], [5,6,3]]]

    affine = np.eye(4)

    fiber_matrix, iso_matrix, vox_coords = FM.model_setup(sl, affine)
    npt.assert_array_equal(np.array(vox_coords),
                    np.array([[1,2,3], [4, 5, 3], [5, 6, 3], [6, 7, 3]]))

    npt.assert_equal(fiber_matrix.shape, (len(vox_coords)*64, len(sl)))
    npt.assert_equal(iso_matrix.shape, (len(vox_coords)*64, len(vox_coords)))


def test_FiberFit():
    data_file, bval_file, bvec_file = get_data('small_64D')
    data_ni = nib.load(data_file)
    data = data_ni.get_data()
    data_aff = data_ni.get_affine()
    bvals, bvecs = (np.load(f) for f in (bval_file, bvec_file))
    gtab = dpg.gradient_table(bvals, bvecs)
    FM1 = life.FiberModel(gtab)

    sl = [[[1,2,3], [4,5,3], [5,6,3], [6,7,3]],
          [[1,2,3], [4,5,3], [5,6,3]]]
    affine = np.eye(4)
    fiber_matrix, iso_matrix, vox_coord = FM1.model_setup(sl, affine)

    evals = [0.0015, 0.0005, 0.0005]

    #w = [0.5, 0.5]
    #sig = [w[i] * life.sl_signal(sl[i], gtab, evals) for i in range(len(sl))]
    #data =

