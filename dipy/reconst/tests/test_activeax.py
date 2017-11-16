# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from numpy.testing import assert_equal, assert_almost_equal
from scipy.optimize import least_squares
from dipy.data import get_data
import dipy.reconst.mix as mix
import dipy.reconst.activeax_fast as activeax_fast
from scipy.optimize import differential_evolution
from dipy.reconst.recspeed import S1, S2, S2_new, activeax_cost_one

maxiter = 1  # The maximum number of generations, genetic
#        algorithm 11 default
xtol = 1e-3

gamma = 2.675987 * 10 ** 8
D_intra = 0.6 * 10 ** 3
fname, fscanner = get_data('ActiveAx_synth_2d')
params = np.loadtxt(fscanner)
G = params[:, 3] / 10 ** 6  # gradient strength
D_iso = 2 * 10 ** 3
G2 = G*G
am = np.array([1.84118307861360])

fname, fscanner = get_data('ActiveAx_synth_2d')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data = img.get_data()
affine = img.affine
bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
te = params[:, 6]
gamma = 2.675987 * 10 ** 8
bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
bvals = bvals
print(bvals * 10 ** 6)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)
L = gtab.bvals * D_intra
signal = np.array(data[0, 0, 0])

signal_param = mix.make_signal_param(signal, gtab.bvals, gtab.bvecs,
                                     G, small_delta, big_delta)


def norm_meas_Aax(signal):

    """
    normalizing the signal based on the b0 values of each shell
    """
    y = signal
    y01 = (y[0] + y[1] + y[2]) / 3
    y02 = (y[93] + y[94] + y[95]) / 3
    y03 = (y[186] + y[187] + y[188]) / 3
    y04 = (y[279] + y[280] + y[281]) / 3
    y1 = y[0:93] / y01
    y2 = y[93:186] / y02
    y3 = y[186:279] / y03
    y4 = y[279:372] / y04
    f = np.concatenate((y1, y2, y3, y4))
    return f


"""
normalizing the signal based on the b0 values of each shell
"""
signal = norm_meas_Aax(signal)

fit_method = 'MIX'
activeax_model = activeax_fast.ActiveAxModel(gtab, fit_method=fit_method)


def test_activax_exvivo_model():

    x = np.array([0.5, 0.5, 10, 0.8])
    x_fe = np.array([0.44623926,  0.2855913,  0.15918695,  2.68329756,
                     2.89085876, 3.40398589,  0.10898249])

    x1, x2 = activeax_model.x_to_xs(x)
    yhat_zeppelin = np.zeros(gtab.small_delta.shape[0])
    S2(x2, gtab.bvals, gtab.bvecs, yhat_zeppelin)

    yhat_ball = activeax_model.S3()
    print(yhat_zeppelin.shape)
    assert_equal(yhat_zeppelin.shape, (372,))
    assert_almost_equal(yhat_zeppelin[6], 2.23215830075)

    assert_almost_equal(yhat_ball[25], 26.3995293508)

#    sigma = 0.05 # offset gaussian constant (not really needed)
    phi = activeax_model.Phi(x)
    print(phi.shape)
    assert_equal(phi.shape, (372, 4))

#    activeax_cost_one(phi, data[0, 0, 0])

#    error_one = mix.activeax_cost_one(phi, data[0, 0, 0])

#    assert_almost_equal(error_one, 0.00037568453377497451)
    yhat2_zeppelin = np.zeros(gtab.small_delta.shape[0])
    S2_new(x_fe, gtab.bvals, gtab.bvecs, yhat2_zeppelin)
    x = x_fe[3:6]

    yhat2_cylinder = np.zeros(gtab.small_delta.shape[0])

    S1(x1, am, gtab.bvecs, gtab.bvals, gtab.small_delta,
       gtab.big_delta, G2, L, yhat2_cylinder)
    yhat2_ball = yhat_ball
    yhat2_dot = activeax_model.S4()

    assert_almost_equal(yhat2_zeppelin[6], 3.0918316084156565)
    assert_almost_equal(yhat2_cylinder[6], 0.56056900197243242)
    assert_almost_equal(yhat2_ball[6], 26.399529350756438)
    assert_almost_equal(yhat2_dot[6], 0)

    exp_phi = activeax_model.Phi2(x_fe)

    assert_almost_equal(exp_phi[6, 0], 0.57088413721552789)
    assert_almost_equal(exp_phi[6, 1], 0.04541868892016445)
    assert_almost_equal(exp_phi[6, 2], 3.426337015990705e-12)
    assert_almost_equal(exp_phi[6, 3], 1)


def test_estimate_x_and_f():
    x_fe = np.array([0.44623926,  0.2855913,  0.15918695,  2.68329756,
                     2.89085876, 3.40398589,  0.10898249])
    x_fe = np.squeeze(x_fe)
    cost = activeax_model.nlls_cost(x_fe, signal)

    """
    assert_array_equal()
    [0.00039828375771280502]
    """
    return cost


def test_activax_exvivo_estimate():
    x = np.array([0.5, 0.5, 10, 0.8])
#    sigma = 0.05 # offset gaussian constant (not really needed)

    phi = activeax_model.Phi(x, gtab)
    fe = activeax_model.cvx_fit(np.array(signal), phi)

    """
    assert_array_equal()
    [[ 0.04266318]
     [ 0.58784575]
     [ 0.21049456]
     [ 0.15899651]]
    """
    return fe


def test_final():
    x_fe = np.array([0.44623926,  0.2855913,  0.15918695,  2.68329756,
                     2.89085876, 3.40398589,  0.10898249])
    bounds = ([0.01, 0.01,  0.01, 0.01, 0.01, 0.1, 0.01], [0.9,  0.9,  0.9,
              np.pi, np.pi, 11, 0.9])

    res = least_squares(activeax_model.nlls_cost, x_fe, bounds=(bounds),
                        xtol=xtol, args=(signal,))

    """
    assert_array_equal()
    [[ 0.44553814,  0.28593944,  0.15950191,  2.6832976 ,  2.89085801,
        3.40282779,  0.10902618]]
    """
    return res

fit_method = 'MIX'
activeax_model = activeax_fast.ActiveAxModel(gtab, fit_method=fit_method)

signal = double(signal)

activeax_fit = activeax_model.fit(signal)

bounds = [(0.01, np.pi), (0.01, np.pi), (0.1, 11), (0.1, 0.8)]


res_one = differential_evolution(activeax_model.stoc_search_cost, bounds,
                                 maxiter=maxiter, args=(signal,))
