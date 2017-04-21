""" Testing Mean Spherical DKI (MDKI) """

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import dipy.reconst.dki as dki
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)
from nose.tools import assert_raises
from dipy.sims.voxel import multi_tensor_dki
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import (gradient_table, unique_bvals, round_bvals)
from dipy.data import get_data
from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)
import dipy.reconst.mdki as mdki

from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.core.geometry import (sphere2cart, perpendicular_directions)

fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
bvals = round_bvals(bvals)
gtab = gradient_table(bvals, bvecs)

# 2 shells for techniques that requires multishell data
bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
gtab_2s = gradient_table(bvals_2s, bvecs_2s)

# Simulation 1. Spherical kurtosis tensor - MK and MD from the MDKI model
# should be equa to the MK and MD of the DKI tensor for cases of
# spherical kurtosis tensors
Di = 0.00099
De = 0.00226
mevals_sph = np.array([[Di, Di, Di], [De, De, De]])
f = 0.5
frac_sph = [f * 100, (1.0 - f) * 100]
signal_sph, dt_sph, kt_sph = multi_tensor_dki(gtab_2s, mevals_sph, S0=100,
                                              fractions=frac_sph,
                                              snr=None)
# Compute ground truth values
MDgt = f*Di + (1-f)*De
MKgt = 3 * f * (1-f) * ((Di-De) / MDgt) ** 2
params = np.array([MDgt, MKgt])
msignal_sph = np.zeros(3)
msignal_sph[0] = signal_sph[0]
msignal_sph[1] = signal_sph[1]
msignal_sph[2] = signal_sph[100]

# Simulation 2. Multi-voxel simulations
DWI = np.zeros((2, 2, 2, len(gtab_2s.bvals)))
MDWI = np.zeros((2, 2, 2, 3))
MDgt_multi = np.zeros((2, 2, 2))
MKgt_multi = np.zeros((2, 2, 2))
S0gt_multi = np.zeros((2, 2, 2))
params_multi = np.zeros((2, 2, 2, 2))

for i in range(2):
    for j in range(2):
        for k in range(1):  # Only one k to have some zero voxels
            f = random.uniform(0.0, 0.1)
            frac = [f * 100, (1.0 - f) * 100]
            signal_i, dt_i, kt_i = multi_tensor_dki(gtab_2s, mevals_sph,
                                                    S0=100, fractions=frac,
                                                    snr=None)
            DWI[i, j, k] = signal_i
            md_i = f*Di + (1-f)*De
            mk_i = 3 * f * (1-f) * ((Di-De) / md_i) ** 2
            MDgt_multi[i, j, k] = md_i
            MKgt_multi[i, j, k] = mk_i
            S0gt_multi = 100
            params_multi[i, j, k, 0] = md_i
            params_multi[i, j, k, 1] = mk_i
            MDWI[i, j, k, 0] = signal_i[0] 
            MDWI[i, j, k, 1] = signal_i[1] 
            MDWI[i, j, k, 2] = signal_i[100] 
            


def test_dki_predict():
    dkiM = mdki.MeanDiffusionKurtosisModel(gtab_2s)


def test_mdki_statistics():
    # tests if MD and MK are equal to expected values of a spherical
    # tensors
    a = 0


def test_dki_errors():
    # first error raises if MeanDiffusionKurtosisModel is called for
    # data will only one non-zero b-value
    assert_raises(ValueError, dki.DiffusionKurtosisModel, gtab)

    # second error raises if negative signal is given to MeanDiffusionKurtosis
    # model
    assert_raises(ValueError, dki.DiffusionKurtosisModel, gtab_2s,
                  min_signal=-1)

    # third error raises if wrong mask is given to fit
    mask_wrong = np.ones((2, 3, 1))
    mdki_model = mdki.MeanDiffusionKurtosisModel(gtab_2s)
    assert_raises(ValueError, mdki_model.fit, DWI, mask=mask_wrong)
    # try case with correct min_signal
    # dkiM = dki.DiffusionKurtosisModel(gtab_2s, min_signal=1)
    # dkiF = dkiM.fit(DWI)
    # assert_array_almost_equal(dkiF.model_params, multi_params)

    # third error is if a given mask do not have same shape as data
    # dkiM = dki.DiffusionKurtosisModel(gtab_2s)

    # test a correct mask
    # dkiF = dkiM.fit(DWI)
    # mask_correct = dkiF.fa > 0
    # mask_correct[1, 1] = False
    # multi_params[1, 1] = np.zeros(27)
    # mask_not_correct = np.array([[True, True, False], [True, False, False]])
    # dkiF = dkiM.fit(DWI, mask=mask_correct)
    # assert_array_almost_equal(dkiF.model_params, multi_params)
    # test a incorrect mask
    # assert_raises(ValueError, dkiM.fit, DWI, mask=mask_not_correct)

    # error if data with only one non zero b-value is given
    # assert_raises(ValueError, dki.DiffusionKurtosisModel, gtab)


def test_design_matrix():
    ub = unique_bvals(bvals_2s)
    D = mdki.design_matrix(ub)
    Dgt = np.ones((3, 3))
    Dgt[:, 0] = -ub
    Dgt[:, 1] = 1.0/6 * ub ** 2
    assert_array_almost_equal(D, Dgt)


def test_msignal():
    # Multi-voxel case
    ms, ng = mdki.mean_signal_bvalue(DWI, gtab_2s)
    assert_array_almost_equal(ms, MDWI)
    assert_array_almost_equal(ng, np.array([2, 64, 64]))

    # Single-voxel case
    ms, ng = mdki.mean_signal_bvalue(signal_sph, gtab_2s)
    assert_array_almost_equal(ng, np.array([2, 64, 64]))
    assert_array_almost_equal(ms, msignal_sph)
