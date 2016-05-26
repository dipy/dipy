""" Testing DTI

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
import numpy.testing as npt
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)
import nibabel as nib

import scipy.optimize as opt

from dipy.reconst import ivim as ivim
from dipy.reconst.ivim import ivim_function
from dipy.data import get_data, dsi_voxels, get_sphere
import dipy.core.gradients as grad
from dipy.sims.voxel import single_tensor
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import (gradient_table, GradientTable,
                                 gradient_table_from_bvals_bvecs,
                                 reorient_bvecs)
from dipy.sims.voxel import multi_tensor
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere


def test_nlls_fit():
    """
    Test the implementation of NLLS
    """
    bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                      120., 140., 160., 180., 200., 220., 240.,
                      260., 280., 300., 350., 400., ])
    N = len(bvals)
    bvecs = get_bvecs(N)
    gtab = gradient_table(bvals, bvecs.T)

    f, D_star, D = 0.06, 0.0072, 0.00097

    mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
    # This gives an isotropic signal

    signal = multi_tensor(gtab, mevals, snr=None, fractions=[
                          f * 100, 100 * (1 - f)])
    data = signal[0]
    S0 = data[0]
    ivim_model = ivim.IvimModel(gtab)
    ivim_fit = ivim_model.fit(data)

    S0_est, f_est, D_star_est, D_est = ivim_fit.model_params
    est_signal = ivim.ivim_function(bvals,
                                    S0_est,
                                    f_est,
                                    D_star_est,
                                    D_est)

    assert_equal(est_signal.shape, data.shape)
    assert_array_almost_equal(est_signal, data)
    assert_array_almost_equal(ivim_fit.model_params, [S0, f, D_star, D])


def get_bvecs(N):
    """Generate bvectors for N bvalues"""
    theta = np.pi * np.random.rand(N)
    phi = 2 * np.pi * np.random.rand(N)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    vertices = hsph_updated.vertices
    return vertices
