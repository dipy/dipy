""" Testing Free Water Elimination Model """

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
from dipy.reconst.fwdti import fwdti_prediction
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)
from nose.tools import assert_raises
from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)
from dipy.reconst.fwdti import (lower_triangular_to_cholesky,
                                cholesky_to_lower_triangular)
from dipy.sims.voxel import (multi_tensor, single_tensor,
                             all_tensor_evecs, multi_tensor_dki)
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
