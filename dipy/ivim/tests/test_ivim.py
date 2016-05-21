""" Testing IVIM """

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import dipy.ivim as ivim
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)
from nose.tools import assert_raises
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.data import get_data
from dipy.ivim import (ivim_prediction)

from dipy.core.sphere import Sphere

from dipy.core.geometry import perpendicular_directions


def generate_gtab():
    bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                      120., 140., 160., 180., 200., 250., 300., 350.])
    N = len(bvals)
    theta = np.pi * np.random.rand(N)
    phi = 2 * np.pi * np.random.rand(N)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    bvecs = hsph_updated.vertices
    gtab = gradient_table(bvals, bvecs.T)
    return gtab


def generate_ivim_params(num_params):
    # Generate some parameters points taking arbitary parameters
    S0 = 100
    f = 0.15 * np.random.rand(num_params)
    D = 0.00018 * np.random.rand(num_params)
    D_star = 0.028 * np.random.rand(num_params)
    ivim_params = np.column_stack((f, D, D_star))
    return ivim_params


def generate_data(ivim_params, gtab):
    return ivim_prediction(ivim_params, gtab)

gtab = generate_gtab()
ivim_params = generate_ivim_params(5)
data = generate_data(ivim_params, gtab)


def test_ivim_fits():
    """ IVIM fits are tested on noise free data """
    # One stage fitting
    ivimM = ivim.IvimModel(gtab, fit_method="one_stage_fit")
    ivimF = ivimM.fit(generated_data)

    assert_array_almost_equal(ivimF.model_params, ivim_params)

    # testing multi-voxels
    # ivimF_multi = dkiM.fit(DWI)
    # assert_array_almost_equal(dkiF_multi.model_params, multi_params)


def test_ivim_predict():
    ivimM = ivim.IvimModel(gtab)
    pred = ivimM.predict(ivim_params, gtab, S0=100)

    assert_array_almost_equal(pred, data)

    # just to check that it works with more than one voxel:
    # pred_multi = ivimM.predict(multi_params, S0=100)
    # assert_array_almost_equal(pred_multi, IVIM)

    # check the function predict of the DiffusionKurtosisFit object
    # ivimF = ivimM.fit(IVIM)
    # pred_multi = ivimF.predict(gtab_2s, S0=100)
    # assert_array_almost_equal(pred_multi, DWI)


def test_ivim_errors():

    # first error of IVIM module is if a unknown fit method is given
    assert_raises(ValueError, ivim.IvimModel, gtab,
                  fit_method="JOANA")

    # second error of DKI module is if a min_signal is defined as negative
    assert_raises(ValueError, ivim.IvimModel, gtab,
                  min_signal=-1)
    # try case with correct min_signal
    ivimM = ivim.IvimModel(gtab, min_signal=1)
    ivimF = ivimM.fit(data)
    assert_array_almost_equal(ivimF.model_params, multi_params)

    # third error is if a given mask do not have same shape as data

    # test a correct mask

    # test a incorrect mask
