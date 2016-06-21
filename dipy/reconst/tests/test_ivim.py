""" Testing IVIM

"""
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal)

from dipy.reconst.ivim import ivim_function, IvimModel
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import multi_tensor
from dipy.core.sphere import disperse_charges, HemiSphere


def test_fit_minimize():
    """
    Test the implementation of the fitting with minimize
    """
    bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                      120., 140., 160., 180., 200., 220., 240.,
                      260., 280., 300., 350., 400., 500., 600.,
                      700., 800., 900., 1000.])
    N = len(bvals)
    bvecs = get_bvecs(N)
    gtab = gradient_table(bvals, bvecs.T)

    S0, f, D_star, D = 1.0, 0.2052, 0.00473, 0.00066

    mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
    # This gives an isotropic signal

    signal = multi_tensor(gtab, mevals, snr=None, S0=S0, fractions=[
                          f * 100, 100 * (1 - f)])
    data = np.array([signal[0], ])

    ivim_model = IvimModel(gtab)
    ivim_fit = ivim_model.fit(data, routine="minimize")

    est_signal = np.array([ivim_function(ivim_fit.model_params[0], bvals), ])

    assert_array_equal(est_signal.shape, data.shape)
    assert_array_almost_equal(est_signal, data)
    assert_array_almost_equal(ivim_fit.model_params[0], [S0, f, D_star, D])


def test_fit_leastsq():
    """
    Test the implementation of the fitting with leastsq
    """
    bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                      120., 140., 160., 180., 200., 220., 240.,
                      260., 280., 300., 350., 400., 500., 600.,
                      700., 800., 900., 1000.])
    N = len(bvals)
    bvecs = get_bvecs(N)
    gtab = gradient_table(bvals, bvecs.T)

    S0, f, D_star, D = 1.0, 0.2052, 0.00473, 0.00066

    mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
    # This gives an isotropic signal

    signal = multi_tensor(gtab, mevals, snr=None, S0=S0, fractions=[
                          f * 100, 100 * (1 - f)])
    data = signal[0]
    ivim_model = IvimModel(gtab)
    ivim_fit = ivim_model.fit(data, routine="leastsq")

    est_signal = ivim_function(ivim_fit.model_params, bvals)

    assert_array_equal(est_signal.shape, data.shape)
    assert_array_almost_equal(est_signal, data)
    assert_array_almost_equal(ivim_fit.model_params, [S0, f, D_star, D])


def test_multivoxel():
    """Test fitting with multivoxel data"""
    bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                      120., 140., 160., 180., 200., 220., 240.,
                      260., 280., 300., 350., 400., 500., 600.,
                      700., 800., 900., 1000.])
    N = len(bvals)
    bvecs = get_bvecs(N)
    gtab = gradient_table(bvals, bvecs.T)
    params = [[1.0, 0.2052, 0.00473, 0.00066], [1.0, 0.18, 0.00555, 0.0007]]

    data = generate_multivoxel_data(gtab, params)
    ivim_model = IvimModel(gtab)

    ivim_fit = ivim_model.fit(data)
    est_signal = generate_multivoxel_data(gtab, ivim_fit.model_params)

    assert_array_equal(est_signal.shape, data.shape)
    assert_array_almost_equal(est_signal, data)
    assert_array_almost_equal(ivim_fit.model_params, params)


def get_bvecs(N):
    """Generate bvectors for N bvalues"""
    theta = np.pi * np.random.rand(N)
    phi = 2 * np.pi * np.random.rand(N)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    vertices = hsph_updated.vertices
    return vertices


def generate_multivoxel_data(gtab, params):
    """Generate multivoxel data for testing"""
    data = []
    for parameters in params:
        S0, f, D_star, D = parameters
        signal = ivim_function(parameters, gtab.bvals)
        data.append(signal)
    return np.array(data)


def test_two_stage():
    """Test the two stage fitting routine"""
    bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                      120., 140., 160., 180., 200., 220., 240.,
                      260., 280., 300., 350., 400., 500., 600.,
                      700., 800., 900., 1000.])
    N = len(bvals)
    bvecs = get_bvecs(N)
    gtab = gradient_table(bvals, bvecs.T)

    S0, f, D_star, D = 1.0, 0.2052, 0.00473, 0.00066

    mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
    # This gives an isotropic signal

    signal = multi_tensor(gtab, mevals, snr=None, S0=S0, fractions=[
                          f * 100, 100 * (1 - f)])
    data = signal[0]
    ivim_model = IvimModel(gtab)

    ivim_fit = ivim_model.fit(data, fit_method="two_stage", routine='minimize')

    est_signal = ivim_function(ivim_fit.model_params, bvals)

    assert_array_equal(est_signal.shape, data.shape)
    assert_array_almost_equal(est_signal, data)
    assert_array_almost_equal(ivim_fit.model_params, [S0, f, D_star, D])


def test_predict():
    """
    Test model prediction API
    """
    bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                      120., 140., 160., 180., 200., 220., 240.,
                      260., 280., 300., 350., 400., 500., 600.,
                      700., 800., 900., 1000.])
    N = len(bvals)
    bvecs = get_bvecs(N)
    gtab = gradient_table(bvals, bvecs.T)

    S0, f, D_star, D = 1.0, 0.2052, 0.00473, 0.00066

    mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
    # This gives an isotropic signal

    signal = multi_tensor(gtab, mevals, snr=None, S0=S0, fractions=[
                          f * 100, 100 * (1 - f)])
    data = signal[0]
    ivim_model = IvimModel(gtab)
    ivim_fit = ivim_model.fit(data, routine="leastsq")

    p = ivim_fit.predict(gtab)
    assert_array_equal(p.shape, data.shape)
    assert_array_almost_equal(p, data)
