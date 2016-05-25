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


def test_nlls_fit():
    """
    Test the implementation of NLLS
    """
    fimg, fbvals, fbvecs = get_data('small_101D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    f, D_star, D = 90, 0.0015, 0.0008

    mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
    # This gives an isotropic signal

    signal = multi_tensor(gtab, mevals, snr=None, fractions=[f, 100 - f])
    data = signal[0]

    ivim_model = ivim.IvimModel(gtab)
    ivim_fit = ivim_model.fit(data)

    S0, f_est, D_star_est, D_est = ivim_fit.model_params
    est_signal = ivim.ivim_function(bvals,
                                    S0_est,
                                    f_est,
                                    D_star_est,
                                    D_est)

    assert_equal(est_signal.shape, data.shape[:-1])
    assert_array_almost_equal(est_signal, data)
    assert_array_almost_equal(ivim_fit.model_params, [S0, f, D_star, D])
