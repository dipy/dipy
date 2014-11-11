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
from dipy.data import get_data, get_sphere
import dipy.core.optimize as opt

import numpy as np
import numpy.testing as npt


def test_streamline_gradients():
    streamline = [[1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10]]
    grads = np.array([[3, 3, 3], [2, 2, 2], [2, 2, 2], [3, 3, 3]])
    npt.assert_array_equal(life.streamline_gradients(streamline), grads)


def test_streamline_tensors():
    # Small streamline
    streamline = [[1, 2, 3], [4, 5, 3], [5, 6, 3]]
    # Non-default eigenvalues:
    evals = [0.0012, 0.0006, 0.0004]
    streamline_tensors = life.streamline_tensors(streamline, evals=evals)
    # Get the resulting eigenvalues:
    eigvals = la.eigvals(streamline_tensors[0])
    # la.eigvals returns things in a strange order, so reorder them:
    idx = np.argsort(eigvals)[::-1]
    # The eigenvalues are the same:
    npt.assert_array_almost_equal(eigvals[idx], evals)
    eigvecs = la.eig(streamline_tensors[0])[1][idx]
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
    streamline = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    streamline_tensors = life.streamline_tensors(streamline, evals=evals)

    for t in streamline_tensors:
        eigvals = la.eigvals(t)
        idx = np.argsort(eigvals)[::-1]
        # The eigenvalues are the same:
        npt.assert_array_almost_equal(eigvals[idx], evals)
        # This one has no rotations - all tensors are simply the canonical:
        eigvecs = la.eig(streamline_tensors[0])[1][idx]
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[0], [1, 0, 0]))), 0)
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[1], [0, 1, 0]))), 0)
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[2], [0, 0, 1]))), 0)


def test_streamline_signal():
    data_file, bval_file, bvec_file = get_data('small_64D')
    gtab = dpg.gradient_table(bval_file, bvec_file)
    evals = [0.0015, 0.0005, 0.0005]
    streamline1 = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
           [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]

    sig1 = [life.streamline_signal(s, gtab, evals) for s in streamline1]

    streamline2 = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]]]

    sig2 = [life.streamline_signal(s, gtab, evals) for s in streamline2]

    npt.assert_array_equal(streamline2[0], streamline1[0])


def test_voxel2streamline():
    streamline = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
          [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]
    affine = np.eye(4)
    v2f, v2fn = life.voxel2streamline(streamline, False, affine)
    npt.assert_array_equal(v2f, np.array([[1, 1], [1, 1], [1,  1], [1, 0]]))
    npt.assert_array_equal(v2fn, np.array([[0, 1, 2, 3], [0, 1, 2, np.nan]]))


def test_FiberModel_init():
    # Get some small amount of data:
    data_file, bval_file, bvec_file = get_data('small_64D')
    data_ni = nib.load(data_file)
    data = data_ni.get_data()
    data_aff = data_ni.get_affine()
    bvals, bvecs = (np.load(f) for f in (bval_file, bvec_file))
    gtab = dpg.gradient_table(bvals, bvecs)
    FM = life.FiberModel(gtab)

    streamline = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
          [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]

    affine = np.eye(4)

    for sphere in [None, False, get_sphere('symmetric362')]:
        fiber_matrix, vox_coords = FM.setup(streamline, affine, sphere=sphere)
        npt.assert_array_equal(np.array(vox_coords),
                               np.array([[1, 2, 3], [4, 5, 3],
                                         [5, 6, 3], [6, 7, 3]]))

        npt.assert_equal(fiber_matrix.shape, (len(vox_coords)*64, len(streamline)))


def test_FiberFit():
    data_file, bval_file, bvec_file = get_data('small_64D')
    data_ni = nib.load(data_file)
    data = data_ni.get_data()
    data_aff = data_ni.get_affine()
    bvals, bvecs = (np.load(f) for f in (bval_file, bvec_file))
    gtab = dpg.gradient_table(bvals, bvecs)
    FM = life.FiberModel(gtab)
    evals = [0.0015, 0.0005, 0.0005]

    streamline = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
          [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]

    fiber_matrix, vox_coords = FM.setup(streamline, None, evals)

    w = np.array([0.5, 0.5])
    sig = opt.spdot(fiber_matrix, w) + 1.0  # Add some isotropic stuff
    S0 = data[..., gtab.b0s_mask]
    rel_sig = data[..., ~gtab.b0s_mask]/data[..., gtab.b0s_mask]
    this_data = np.zeros((10, 10, 10, 64))
    this_data[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]] =\
        (sig.reshape((4, 64)) *
         S0[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]])

    # Grab some realistic S0 values:
    this_data = np.concatenate([data[..., gtab.b0s_mask], this_data], -1)

    fit = FM.fit(this_data, streamline)
    npt.assert_almost_equal(fit.predict()[1],
                            fit.data[1], decimal=-1)

    # Predict with an input GradientTable
    npt.assert_almost_equal(fit.predict(gtab)[1],
                            fit.data[1], decimal=-1)

    npt.assert_almost_equal(
        this_data[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]],
        fit.data)
