# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from numpy.testing import assert_equal, assert_almost_equal
from scipy.optimize import least_squares
from dipy.data import get_data
import dipy.reconst.mix as mix

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
signal = np.array(data[0, 0, 0])

signal_param = mix.make_signal_param(signal, bvals, bvecs, G, small_delta,
                                     big_delta)


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


def test_activax_exvivo_model():

    x = np.array([0.5, 0.5, 10, 0.8])
    x_fe = np.array([0.44623926,  0.2855913,  0.15918695,  2.68329756,
                     2.89085876, 3.40398589,  0.10898249])
#    sigma = 0.05
    L1, summ, summ_rows, gper, L2, yhat_cylinder, \
        yhat_zeppelin, yhat_ball, yhat_one = mix.activax_exvivo_compartments(
            data, x, bvals, bvecs, G, small_delta, big_delta,
            gamma=gamma, D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
            debug=True)

    assert_almost_equal(L1[3], 3.96735278865)
    assert_almost_equal(L1[5], 5.54561445859)

    assert_equal(summ.shape, (372, 60))
    assert_almost_equal(summ[3, 3], 2.15633236279e-07)
    assert_almost_equal(summ[20, 2], 1.3498305300e-06)

    summ_rows = np.sum(summ, axis=1)

    print('summ_rows')
    print(summ_rows[20])
    print(summ_rows[2])

    assert_equal(summ_rows.shape, (372,))

    print(summ[3, 3])

    assert_equal(gper.shape, (372,))
    assert_almost_equal(gper[39], 0.004856614915350943)
    assert_almost_equal(gper[45], 0.60240287122917402)

    print(L2.shape)
    print(L2[5])

    assert_equal(L2.shape, (372,))
    assert_almost_equal(L2[5], 1.33006433687)

    print(yhat_zeppelin.shape)
    assert_equal(yhat_zeppelin.shape, (372,))
    assert_almost_equal(yhat_zeppelin[6], 2.23215830075)

    assert_almost_equal(yhat_ball[25], 26.3995293508)

#    sigma = 0.05 # offset gaussian constant (not really needed)
    phi = mix.activax_exvivo_model(x, bvals, bvecs, G, small_delta, big_delta)
    print(phi.shape)
    assert_equal(phi.shape, (372, 4))
    error_one = mix.activeax_cost_one(phi, data[0, 0, 0])

    assert_almost_equal(error_one, 0.00037568453377497451)

    yhat2_cylinder, yhat2_zeppelin, yhat2_ball, yhat2_dot = \
        mix.activax_exvivo_compartments2(x_fe, bvals, bvecs, G, small_delta,
                                         big_delta, gamma=gamma,
                                         D_intra=0.6 * 10 ** 3,
                                         D_iso=2 * 10 ** 3, debug=False)
    assert_almost_equal(yhat2_zeppelin[6], 3.0918316084156565)
    assert_almost_equal(yhat2_cylinder[6], 0.56056900197243242)
    assert_almost_equal(yhat2_ball[6], 26.399529350756438)
    assert_almost_equal(yhat2_dot[6], 0)

    exp_phi = mix.activax_exvivo_model2(x_fe, bvals, bvecs, G, small_delta,
                                        big_delta, gamma=gamma,
                                        D_intra=0.6 * 10 ** 3,
                                        D_iso=2 * 10 ** 3,
                                        debug=False)

    assert_almost_equal(exp_phi[6, 0], 0.57088413721552789)
    assert_almost_equal(exp_phi[6, 1], 0.04541868892016445)
    assert_almost_equal(exp_phi[6, 2], 3.426337015990705e-12)
    assert_almost_equal(exp_phi[6, 3], 1)


def test_nls_cost_func():
    x_fe = np.array([0.44623926,  0.2855913,  0.15918695,  2.68329756,
                     2.89085876, 3.40398589,  0.10898249])
    x_fe = np.squeeze(x_fe)
    cost = mix.nls_cost_func(x_fe, signal_param)
    """
    assert_array_equal()
    [0.00039828375771280502]
    """
    return cost


def test_activax_exvivo_estimate():
    x = np.array([0.5, 0.5, 10, 0.8])
#    sigma = 0.05 # offset gaussian constant (not really needed)
    phi = mix.activax_exvivo_model(x, bvals, bvecs, G,
                                   small_delta, big_delta)
    fe = mix.estimate_f(np.array(data[0, 0, 0]), phi)

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
    res = least_squares(mix.nls_cost_func, x_fe, bounds=(bounds),
                        args=(signal_param,))

    """
    assert_array_equal()
    [[ 0.44553814,  0.28593944,  0.15950191,  2.6832976 ,  2.89085801,
        3.40282779,  0.10902618]]
    """
    return res


def test_dif_evol():
    res_one = mix.dif_evol(signal, bvals, bvecs, G, small_delta, big_delta)

    """
    assert_array_equal()
    [[ 2.68329953,  2.89083525,  3.40510099,  0.6076389 ]]
    """
    return res_one
