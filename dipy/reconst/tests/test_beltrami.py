""" Testing standalone beltrami fucntions """

from __future__ import division, print_function, absolute_import

import numpy as np
import dipy.reconst.dti as dti
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)
from nose.tools import assert_raises
from dipy.sims.voxel import (multi_tensor, single_tensor,
                             all_tensor_evecs, multi_tensor_dki)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_data

import dipy.reconst.beltrami as blt


# Loading bvals and bvecs
fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

# The Beltrami framework requires the b-values to be in the scale of unity,
# (i.e. b = 1000 becomes b = 1).
# The estimated diffusivities will have the the units of micrometers^2/milisecond
bvals = bvals / 1000
gtab = gradient_table(bvals, bvecs)

# Defining the tissue compartment evals of a multivoxel DWI (8 voxels),
# each row is a set of evals for a given voxel
evals_t = np.array([[1.7, 0.3, 0.3],
                    [1.3, 0.5, 0.5],
                    [1.8, 0.2, 0.2],
                    [1.7, 0.4, 0.4],
                    [0.3, 1.7, 0.3],
                    [0.5, 1.3, 0.5],
                    [0.2, 1.8, 0.2],
                    [0.4, 1.7, 0.4]])

# Defining the Free Water compartment evals, FW has diffusivity 3,
# the FW evals are the same for all voxels
evals_w = np.array([3, 3, 3])

# Defining the FW fractions for each voxel
fw = np.array([0.33, 0.42, 0.46, 0.51, 0.006, 0.71, 0.90, 0.0])

# Creating the linearized DWI and filling with the simulated signal
DWI = np.zeros((8, len(bvals)))

for i, evals_i in enumerate(evals_t):
    mevals = np.stack((evals_i, evals_w), axis=0)
    angles = [(90, 0), (90, 0)]
    fractions = [(1 - fw[i])*100, fw[i]*100]
    signal, sticks = multi_tensor(gtab, mevals, S0=100, angles=angles,
                                  fractions=fractions, snr=None)
    DWI[i, :] = signal

# Reshaping to 3D volume
DWI = np.reshape(DWI, (2, 2, 2, len(bvals)))

# Padding the DWI with ones along the x and y (row and column) directions,
# to simulate background voxels to be masked
data = np.ones((4, 4, 2, len(bvals)))
data[1:3, 1:3, ...] = DWI
mask = np.zeros(data.shape[:-1], dtype=bool)
mask[1:3, 1:3, ...] = True

# Estimating an initial guess from the data by applying standard DTI,
# using and 'OLS' fit
tenmodel = dti.TensorModel(gtab, fit_method='OLS')
tenfit = tenmodel.fit(data, mask=mask)
qform = tenfit.quadratic_form

# Storing only the independent componets of the quadratic form
Dxx = qform[..., 0, 0]
Dyy = qform[..., 1, 1]
Dzz = qform[..., 2, 2]
Dxy = qform[..., 0, 1]
Dxz = qform[..., 0, 2]
Dyz = qform[..., 1, 2]

D = np.stack((Dxx, Dyy, Dzz, Dxy, Dxz, Dyz), axis=3)


def test_x_manifold():
    # allocating the output array
    out = np.zeros(D.shape)

    # computing Iwasawa using the function x_manifold
    blt.x_manifold(D, mask, out)
    X_func = out[mask, :]

    # manually computing the Iwasawa coordinates for voxels inside mask
    # (according to definition)
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = np.rollaxis(D[mask, :], axis=1)
    X1 = Dxx
    X2 = Dyy - Dxy**2 / Dxx
    X3 = ((Dxx * Dxy**2 + Dxz * (Dyy * Dxz - 2 * Dxy * Dyz) +
          Dxx * (Dyz**2 - Dyy * Dzz)) / (Dxy**2 - Dxx * Dyy))
    X4 = Dxy / Dxx 
    X5 = Dxz / Dxx
    X6 = (Dxx * Dyz - Dxy * Dxz) / (Dxx * Dyy - Dxy**2)
    X_man = np.stack((X1, X2, X3, X4, X5, X6), axis=1)

    # check if the computations match
    assert_array_almost_equal(X_func, X_man)


def test_d_manifold():
    # allocating the output array
    out = np.zeros(D.shape)

    # computing Iwasawa using the function x_manifold
    blt.x_manifold(D, mask, out)
    this_X = out

    # converting the Iwasawa coordinates back to diffusion components
    blt.d_manifold(this_X, mask, out)
    this_D = out

    # check if the newly computed D (from converting X) matches
    # the original D (before converting to X)
    assert_array_almost_equal(this_D, D, decimal=5)
