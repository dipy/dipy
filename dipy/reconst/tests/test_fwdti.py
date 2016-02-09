""" Testing Free Water Elimination Model """

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
from dipy.reconst.fwdti import fwdti_prediction
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)
from nose.tools import assert_raises
from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)
from dipy.sims.voxel import (multi_tensor, single_tensor, _check_directions,
                             all_tensor_evecs)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_data

fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

# FW model requires multishell data
bvals_2s = np.concatenate((bvals, bvals * 1.5), axis=0)
bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
gtab_2s = gradient_table(bvals_2s, bvecs_2s)

# Simulation a typical DT and DW signal for no water contamination
dt = np.array([0.0017, 0, 0.0003, 0, 0, 0.0003])
evals, evecs = decompose_tensor(from_lower_triangular(dt))
S_tissue = single_tensor(gtab_2s, S0=100, evals=evals, evecs=evecs,
                         snr=None)
dm = dti.TensorModel(gtab_2s, 'WLS')
dtifit = dm.fit(S_tissue)
FAdti = dtifit.fa

# Simulation of 8 voxels tested
DWI = np.zeros((2, 2, 2, len(gtab_2s.bvals)))
FAref = np.zeros((2, 2, 2))
# Diffusion of tissue and water compartments are constant for all voxel
mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
# volume fractions
GTF = np.array([[[0.06, 0.71], [0.33, 0.93]],
                [[0., 0,], [0., 0.]]])
# model_params ground truth (to be fill)
model_params_mv = np.zeros((2, 2, 2, 14))         
for i in range(2):
    for j in range(2):
        gtf = GTF[0, i, j]
        S, p = multi_tensor(gtab_2s, mevals, S0=100,
                            angles=[(90, 0), (90, 0)],
                            fractions=[(1-gtf) * 100, gtf*100], snr=None)
        DWI[0, i, j] = S
        FAref[0, i, j] = FAdti
        R = all_tensor_evecs(p[0])
        R = R.reshape((9))
        model_params_mv[0, i, j] = np.concatenate(([0.0017, 0.0003, 0.0003],
                                                   R, [gtf, 100]), axis=0)


def test_fwdti_singlevoxel():
    # Simulation when water contamination is added
    gtf = 0.50  #ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f

    assert_almost_equal(FAdti, FAfwe, decimal=3)
    assert_almost_equal(Ffwe, gtf, decimal=3)

    # Test non-linear fit
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS')
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f

    assert_almost_equal(FAdti, FAfwe)
    assert_almost_equal(Ffwe, gtf)
    
    # Test non-linear fit, when no first quess is given
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', fw_params=None)
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f

    assert_almost_equal(FAdti, FAfwe)
    assert_almost_equal(Ffwe, gtf)


def test_fwdti_precision():
    # Simulation when water contamination is added
    gtf = 0.63416  #ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS', piterations=5)
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f

    assert_almost_equal(Ffwe, gtf, decimal=5)
    assert_almost_equal(FAdti, FAfwe, decimal=5)


def test_fwdti_multi_voxel():
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS')
    fwefit = fwdm.fit(DWI)
    Ffwe = fwefit.f

    assert_array_almost_equal(Ffwe, GTF, decimal=2)
    

def test_fwdti_predictions():
    # single voxel case
    # test funtion
    gtf = 0.50  #ground truth volume fraction
    S0=100
    angles = [(90, 0), (90, 0)]
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=S0,
                                  angles=angles,
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    R = all_tensor_evecs(peaks[0])
    R = R.reshape((9))
    model_params = np.concatenate(([0.0017, 0.0003, 0.0003], R, [gtf, 100]),
                                  axis=0)
    S_pred1 = fwdti_prediction(model_params, gtab_2s)
    assert_array_almost_equal(S_pred1, S_conta)

    # Testing in model class
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    S_pred2 = fwdm.predict(model_params)
    assert_array_almost_equal(S_pred2, S_conta)

    # Testing in fit class
    fwefit = fwdm.fit(S_conta)
    # Adjust simulations according to model parameters (note here testing the
    # robustness of fit is not the objective)
    mevals_ad = np.array([fwefit.model_params[0:3], [0.003, 0.003, 0.003]])
    angles_ad = fwefit.model_params[3:-2].reshape(3, 3)
    gtf_ad = fwefit.model_params[-2]
    S0 = fwefit.model_params[-1]
    S_conta_ad, peaks = multi_tensor(gtab_2s, mevals_ad, S0=S0,
                                     angles=angles_ad,
                                     fractions=[(1-gtf_ad) * 100, gtf_ad*100],
                                     snr=None)
    S_pred3 = fwefit.predict(gtab_2s)
    assert_array_almost_equal(S_pred3, S_conta_ad, decimal=5)

    # Multi voxel simulation
    S_pred1 = fwdti_prediction(model_params_mv, gtab_2s)
    assert_array_almost_equal(S_pred1, DWI)
    S_pred2 = fwdm.predict(model_params_mv)
    assert_array_almost_equal(S_pred2, DWI)


def test_fwdti_errors():
    # 1st error - if a unknown fit method is given to the FWTM
    assert_raises(ValueError, fwdti.FreeWaterTensorModel, gtab_2s,
                  fit_method="pKT")
    # 2nd error - if min_signal is negative
    assert_raises(ValueError, fwdti.FreeWaterTensorModel, gtab_2s,
                  min_signal=-1)
    # 3rd error - if incorrect mask is given
    fwdtiM = fwdti.FreeWaterTensorModel(gtab_2s)
    incorrect_mask = np.array([[True, True, False], [True, False, False]])
    assert_raises(ValueError, fwdtiM.fit, DWI, mask=incorrect_mask)

    # Testing the correct usage
    fwdtiM = fwdti.FreeWaterTensorModel(gtab_2s, min_signal=1)
    correct_mask = np.zeros((2, 2, 2))
    correct_mask[0, :, :] = 1;
    correct_mask = correct_mask > 0
    fwdtiF = fwdtiM.fit(DWI, mask=correct_mask)
    assert_array_almost_equal(fwdtiF.fa, FAref)
    assert_array_almost_equal(fwdtiF.f, GTF)

    # 4th error - if a sigma is selected by no value of sigma is given for
    # in the non-linear approach to performe restore
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', weighting='sigma')
    assert_raises(ValueError, fwdm.fit, DWI)


def test_fwdti_restore():
    # Restore have to work well even in nonproblematic cases
    # Simulate a signal corrupted by free water diffusion contamination
    gtf = 0.50  #ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', weighting='sigma',
                                      sigma = 4)
    fwdtiF = fwdm.fit(S_conta)
    assert_array_almost_equal(fwdtiF.fa, FAdti)
    assert_array_almost_equal(fwdtiF.f, gtf)
    fwdm2 = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', weighting='gmm')
    fwdtiF2 = fwdm2.fit(S_conta)
    assert_array_almost_equal(fwdtiF2.fa, FAdti)
    assert_array_almost_equal(fwdtiF2.f, gtf)
