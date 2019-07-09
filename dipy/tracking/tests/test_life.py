import os
import os.path as op

import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec
import scipy.sparse as sps
import scipy.linalg as la

import nibabel as nib

import dipy.tracking.life as life
import dipy.tracking.eudx as edx
import dipy.core.sphere as dps
import dipy.core.gradients as dpg
import dipy.data as dpd
import dipy.core.optimize as opt
import dipy.core.ndindex as nd
import dipy.core.gradients as grad
import dipy.reconst.dti as dti
from dipy.io.gradients import read_bvals_bvecs

THIS_DIR = op.dirname(__file__)

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
    npt.assert_array_almost_equal(streamline_tensors[0],
                                  np.array([[0.0009, 0.0003, 0.],
                                            [0.0003, 0.0009, 0.],
                                            [0., 0., 0.0004]]))

    # Get the eigenvalues/eigenvectors:
    eigvals, eigvecs = la.eig(streamline_tensors[0])
    eigvecs = eigvecs[np.argsort(eigvals)[::-1]]
    eigvals = eigvals[np.argsort(eigvals)[::-1]]

    npt.assert_array_almost_equal(eigvals,
                                  np.array([0.0012, 0.0006, 0.0004]))

    npt.assert_array_almost_equal(eigvecs[0],
                                  np.array([0.70710678, -0.70710678, 0.]))
    # Another small streamline
    streamline = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    streamline_tensors = life.streamline_tensors(streamline, evals=evals)

    for t in streamline_tensors:
        eigvals, eigvecs = la.eig(t)
        eigvecs = eigvecs[np.argsort(eigvals)[::-1]]
        # This one has no rotations - all tensors are simply the canonical:
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[0], [1, 0, 0]))), 0)
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[1], [0, 1, 0]))), 0)
        npt.assert_almost_equal(np.rad2deg(np.arccos(
            np.dot(eigvecs[2], [0, 0, 1]))), 0)


def test_streamline_signal():
    data_file, bval_file, bvec_file = dpd.get_fnames('small_64D')
    gtab = dpg.gradient_table(bval_file, bvec_file)
    evals = [0.0015, 0.0005, 0.0005]
    streamline1 = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
                   [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]

    [life.streamline_signal(s, gtab, evals) for s in streamline1]

    streamline2 = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]]]

    [life.streamline_signal(s, gtab, evals) for s in streamline2]

    npt.assert_array_equal(streamline2[0], streamline1[0])


def test_voxel2streamline():
    streamline = [[[1.1, 2.4, 2.9], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
                  [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]
    affine = np.eye(4)
    v2f, v2fn = life.voxel2streamline(streamline, False, affine)
    npt.assert_equal(v2f, {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0]})
    npt.assert_equal(v2fn, {0: {0: [0], 1: [1], 2: [2], 3: [3]},
                            1: {0: [0], 1: [1], 2: [2]}})
    affine = np.array([[0.9, 0, 0, 10],
                       [0, 0.9, 0, -100],
                       [0, 0, 0.9, 2],
                       [0, 0, 0, 1]])

    xform_sl = life.transform_streamlines(streamline, np.linalg.inv(affine))
    v2f, v2fn = life.voxel2streamline(xform_sl, False, affine)
    npt.assert_equal(v2f, {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0]})
    npt.assert_equal(v2fn, {0: {0: [0], 1: [1], 2: [2], 3: [3]},
                            1: {0: [0], 1: [1], 2: [2]}})


def test_FiberModel_init():
    # Get some small amount of data:
    data_file, bval_file, bvec_file = dpd.get_fnames('small_64D')
    data_ni = nib.load(data_file)
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = dpg.gradient_table(bvals, bvecs)
    FM = life.FiberModel(gtab)

    streamline = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
                  [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]

    affine = np.eye(4)

    for sphere in [None, False, dpd.get_sphere('symmetric362')]:
        fiber_matrix, vox_coords = FM.setup(streamline, affine, sphere=sphere)
        npt.assert_array_equal(np.array(vox_coords),
                               np.array([[1, 2, 3], [4, 5, 3],
                                         [5, 6, 3], [6, 7, 3]]))

        npt.assert_equal(fiber_matrix.shape, (len(vox_coords) * 64,
                                              len(streamline)))


def test_FiberFit():
    data_file, bval_file, bvec_file = dpd.get_fnames('small_64D')
    data_ni = nib.load(data_file)
    data = data_ni.get_data()
    data_aff = data_ni.affine
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = dpg.gradient_table(bvals, bvecs)
    FM = life.FiberModel(gtab)
    evals = [0.0015, 0.0005, 0.0005]

    streamline = [[[1, 2, 3], [4, 5, 3], [5, 6, 3], [6, 7, 3]],
                  [[1, 2, 3], [4, 5, 3], [5, 6, 3]]]

    fiber_matrix, vox_coords = FM.setup(streamline, None, evals)

    w = np.array([0.5, 0.5])
    sig = opt.spdot(fiber_matrix, w) + 1.0  # Add some isotropic stuff
    S0 = data[..., gtab.b0s_mask]
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

def test_fit_data():
    fdata, fbval, fbvec = dpd.get_fnames('small_25')
    gtab = grad.gradient_table(fbval, fbvec)
    ni_data = nib.load(fdata)
    data = ni_data.get_data()
    dtmodel = dti.TensorModel(gtab)
    dtfit = dtmodel.fit(data)
    sphere = dpd.get_sphere()
    peak_idx = dti.quantize_evecs(dtfit.evecs, sphere.vertices)
    eu = edx.EuDX(dtfit.fa.astype('f8'), peak_idx,
                  seeds=list(nd.ndindex(data.shape[:-1])),
                  odf_vertices=sphere.vertices, a_low=0)
    tensor_streamlines = [streamline for streamline in eu]
    life_model = life.FiberModel(gtab)
    life_fit = life_model.fit(data, tensor_streamlines)
    model_error = life_fit.predict() - life_fit.data
    model_rmse = np.sqrt(np.mean(model_error ** 2, -1))
    matlab_rmse, matlab_weights = dpd.matlab_life_results()
    # Lower error than the matlab implementation for these data:
    npt.assert_(np.median(model_rmse) < np.median(matlab_rmse))
    # And a moderate correlation with the Matlab implementation weights:
    npt.assert_(np.corrcoef(matlab_weights, life_fit.beta)[0, 1] > 0.6)
