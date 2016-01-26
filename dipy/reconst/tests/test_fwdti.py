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
from dipy.sims.voxel import (multi_tensor, single_tensor)
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


def test_fwdti_singlevoxel():
    # Simulation when water contamination is added
    gtf = 0.50  #ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(0, 0), (0, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f

    assert_almost_equal(FAdti, FAfwe, decimal=3)
    assert_almost_equal(Ffwe, gtf, decimal=3)


def test_fwdti_precision():
    # Simulation when water contamination is added
    gtf = 0.63416  #ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(0, 0), (0, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS', piterations=5)
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f

    assert_almost_equal(Ffwe, gtf, decimal=5)
    assert_almost_equal(FAdti, FAfwe, decimal=5)


def test_fwdti_multi_voxel():
    # 8 voxels tested
    DWI = np.zeros((2, 2, 2, len(gtab_2s.bvals)))
    Fref = np.zeros((2, 2, 2))
    FAref = np.zeros((2, 2, 2))
    # Diffusion of tissue and water compartments are constant for all voxel
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])

    # volume fractions are random
    for i in range(2):
        for j in range(2):
            gtf = random.uniform(0, 1)
            S, p = multi_tensor(gtab_2s, mevals, S0=100,
                                angles=[(0, 0), (0, 0)],
                                fractions=[(1-gtf) * 100, gtf*100], snr=None)
            DWI[i, j, 0] = S
            Fref[i, j, 0] = gtf
            FAref[i, j, 0] = FAdti

    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    fwefit = fwdm.fit(DWI)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f

    assert_array_almost_equal(Ffwe, Fref, decimal=1)
    

def test_fwdti_predictions():
    # single voxel case
    gtf = 0.50  #ground truth volume fraction
    S0=100
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=S0,
                                  angles=[(0, 0), (0, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    fwefit = fwdm.fit(S_conta)

    S_pred1 = fwdti_prediction(fwefit.model_params, gtab_2s, S0)
    assert_array_almost_equal(S_pred1, S_conta, decimal=1)