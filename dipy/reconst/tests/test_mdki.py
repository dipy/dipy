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
import dipy.reconst.mdki as mdki

fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
bvals = round_bvals(bvals)
gtab = gradient_table(bvals, bvecs)

# 2 shells for techniques that requires multishell data
bvals_3s = np.concatenate((bvals, bvals*1.5, bvals * 2), axis=0)
bvecs_3s = np.concatenate((bvecs, bvecs, bvecs), axis=0)
gtab_3s = gradient_table(bvals_3s, bvecs_3s)

# Simulation 1. Spherical kurtosis tensor - MK and MD from the MDKI model
# should be equa to the MK and MD of the DKI tensor for cases of
# spherical kurtosis tensors
Di = 0.00099
De = 0.00226
mevals_sph = np.array([[Di, Di, Di], [De, De, De]])
f = 0.5
frac_sph = [f * 100, (1.0 - f) * 100]
signal_sph, dt_sph, kt_sph = multi_tensor_dki(gtab_3s, mevals_sph, S0=100,
                                              fractions=frac_sph,
                                              snr=None)
# Compute ground truth values
MDgt = f*Di + (1-f)*De
MKgt = 3 * f * (1-f) * ((Di-De) / MDgt) ** 2
params_single = np.array([MDgt, MKgt])
msignal_sph = np.zeros(4)
msignal_sph[0] = signal_sph[0]
msignal_sph[1] = signal_sph[1]
msignal_sph[2] = signal_sph[100]
msignal_sph[3] = signal_sph[180]

# Simulation 2. Multi-voxel simulations
DWI = np.zeros((2, 2, 2, len(gtab_3s.bvals)))
MDWI = np.zeros((2, 2, 2, 4))
MDgt_multi = np.zeros((2, 2, 2))
MKgt_multi = np.zeros((2, 2, 2))
S0gt_multi = np.zeros((2, 2, 2))
params_multi = np.zeros((2, 2, 2, 2))

for i in range(2):
    for j in range(2):
        for k in range(1):  # Only one k to have some zero voxels
            f = random.uniform(0.0, 0.1)
            frac = [f * 100, (1.0 - f) * 100]
            signal_i, dt_i, kt_i = multi_tensor_dki(gtab_3s, mevals_sph,
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
            MDWI[i, j, k, 3] = signal_i[180]


def test_dki_predict():
    dkiM = mdki.MeanDiffusionKurtosisModel(gtab_3s)

    # single voxel
    pred = dkiM.predict(params_single, S0=100)
    assert_array_almost_equal(pred, signal_sph)

    # multi-voxel
    pred = dkiM.predict(params_multi, S0=100)
    assert_array_almost_equal(pred[:, :, 0, :], DWI[:, :, 0, :])

    # check the function predict of the DiffusionKurtosisFit object
    dkiF = dkiM.fit(signal_sph)
    pred_single = dkiF.predict(gtab_3s, S0=100)
    assert_array_almost_equal(pred_single, signal_sph)
    dkiF = dkiM.fit(DWI)
    pred_multi = dkiF.predict(gtab_3s, S0=100)
    assert_array_almost_equal(pred_multi[:, :, 0, :], DWI[:, :, 0, :])

    # No S0
    dkiF = dkiM.fit(signal_sph)
    pred_single = dkiF.predict(gtab_3s)
    assert_array_almost_equal(100 * pred_single, signal_sph)
    dkiF = dkiM.fit(DWI)
    pred_multi = dkiF.predict(gtab_3s)
    assert_array_almost_equal(100 * pred_multi[:, :, 0, :], DWI[:, :, 0, :])

    # SO volume
    dkiF = dkiM.fit(DWI)
    pred_multi = dkiF.predict(gtab_3s, 100 * np.ones(DWI.shape[:-1]))
    assert_array_almost_equal(pred_multi[:, :, 0, :], DWI[:, :, 0, :])


def test_dki_errors():
    # first error raises if MeanDiffusionKurtosisModel is called for
    # data will only one non-zero b-value
    assert_raises(ValueError, dki.DiffusionKurtosisModel, gtab)

    # second error raises if negative signal is given to MeanDiffusionKurtosis
    # model
    assert_raises(ValueError, dki.DiffusionKurtosisModel, gtab_3s,
                  min_signal=-1)

    # third error raises if wrong mask is given to fit
    mask_wrong = np.ones((2, 3, 1))
    mdki_model = mdki.MeanDiffusionKurtosisModel(gtab_3s)
    assert_raises(ValueError, mdki_model.fit, DWI, mask=mask_wrong)


def test_design_matrix():
    ub = unique_bvals(bvals_3s)
    D = mdki.design_matrix(ub)
    Dgt = np.ones((4, 3))
    Dgt[:, 0] = -ub
    Dgt[:, 1] = 1.0/6 * ub ** 2
    assert_array_almost_equal(D, Dgt)


def test_msignal():
    # Multi-voxel case
    ms, ng = mdki.mean_signal_bvalue(DWI, gtab_3s)
    assert_array_almost_equal(ms, MDWI)
    assert_array_almost_equal(ng, np.array([3, 64, 64, 64]))

    # Single-voxel case
    ms, ng = mdki.mean_signal_bvalue(signal_sph, gtab_3s)
    assert_array_almost_equal(ng, np.array([3, 64, 64, 64]))
    assert_array_almost_equal(ms, msignal_sph)


def test_mdki_statistics():
    # tests if MD and MK are equal to expected values of a spherical
    # tensors

    # Multi-tensors
    ub = unique_bvals(bvals_3s)
    design_matrix = mdki.design_matrix(ub)
    msignal, ng = mdki.mean_signal_bvalue(DWI, gtab_3s, bmag=None)
    params = mdki.wls_fit_mdki(design_matrix, msignal, ng)
    assert_array_almost_equal(params[..., 1], MKgt_multi)
    assert_array_almost_equal(params[..., 0], MDgt_multi)

    mdkiM = mdki.MeanDiffusionKurtosisModel(gtab_3s)
    mdkiF = mdkiM.fit(DWI)
    mk = mdkiF.mk
    md = mdkiF.md
    assert_array_almost_equal(MKgt_multi, mk)
    assert_array_almost_equal(MDgt_multi, md)

    # Single-tensors
    mdkiF = mdkiM.fit(signal_sph)
    mk = mdkiF.mk
    md = mdkiF.md
    assert_array_almost_equal(MKgt, mk)
    assert_array_almost_equal(MDgt, md)

    # Test with given mask
    mdkiF = mdkiM.fit(DWI, mask=np.ones(DWI.shape[:-1]))
    mk = mdkiF.mk
    md = mdkiF.md
    assert_array_almost_equal(MKgt_multi, mk)
    assert_array_almost_equal(MDgt_multi, md)
