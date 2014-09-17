import os

import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec
import scipy.io as sio

import nibabel as nib

import dipy.tracking.life as life
import dipy.core.sphere as dps
import dipy.core.gradients as dpg
from dipy.data import get_data


def test_sl_gradients():
    sl = [[1,2,3], [4,5,6], [5,6,7]]
    grads = np.array([[3,3,3], [2,2,2], [1,1,1]])
    npt.assert_array_equal(life.sl_gradients(sl), grads)


def test_sl_tensors():
    # This is the response function, canonical tensor evals:
    evals = [1.5, 0.5, 0.5]
    # This streamline doesn't move on the z dimension
    sl = [[1,2,3], [4,5,3], [5,6,3]]
    # So the resulting tensor for each of the gradients should be simply a
    # rotation of the original canonical tensor onto 45 degrees in the x,y plain
    sl_tensors = life.sl_tensors(sl)


def test_sl_signal():
    data_file, bval_file, bvec_file = get_data('small_64D')
    gtab = dpg.gradient_table(bval_file, bvec_file)
    sl = [[1,2,3], [4,5,3], [5,6,3], [6,7,3]]
    evals = [0.0015, 0.0005, 0.0005]
    sig = life.sl_signal(sl, gtab, evals)

    sl = [[[1,2,3], [4,5,3], [5,6,3], [6,7,3]],
          [[1,2,3], [4,5,3], [5,6,3]]]
    sig = [life.sl_signal(s, gtab, evals) for s in sl]


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
    FM1 = life.FiberModel(gtab)

    sl = [[[1,2,3], [4,5,3], [5,6,3], [6,7,3]],
          [[1,2,3], [4,5,3], [5,6,3]]]
    affine = np.eye(4)

    mat = FM1.model_matrix(sl, affine)
