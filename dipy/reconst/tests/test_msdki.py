""" Testing Mean Signal DKI (MSDKI) """

import numpy as np
import random
from numpy.testing import (assert_array_almost_equal, assert_raises,
                           assert_almost_equal, assert_)
from dipy.sims.voxel import (single_tensor, multi_tensor_dki)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import (gradient_table, unique_bvals_magnitude,
                                 round_bvals)
from dipy.data import get_fnames
import dipy.reconst.msdki as msdki
from dipy.reconst.msdki import (msk_from_awf, awf_from_msk)

gtab_3s, signal_sph, msignal_sph, DWI, MDWI = None, None, None, None, None
gtab, params_single, params_multi, MKgt_multi = None, None, None, None
bvals_3s, MDgt_multi, S0gt_multi, MKgt, MDgt = None, None, None, None, None
bvecs = None


def setup_module():
    global gtab_3s, signal_sph, msignal_sph, DWI, MDWI
    global bvecs, gtab, params_single, params_multi, MKgt_multi
    global bvals_3s, MDgt_multi, S0gt_multi, MKgt, MDgt

    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    bvals = round_bvals(bvals)
    gtab = gradient_table(bvals, bvecs)

    # 2 shells for techniques that requires multishell data
    bvals_3s = np.concatenate((bvals, bvals*1.5, bvals * 2), axis=0)
    bvecs_3s = np.concatenate((bvecs, bvecs, bvecs), axis=0)
    gtab_3s = gradient_table(bvals_3s, bvecs_3s)

    # Simulation 1. Spherical kurtosis tensor - MSK and MSD from the MSDKI
    # model should be equal to the MK and MD of the DKI tensor for cases of
    # spherical kurtosis tensors
    Di = 0.00099
    De = 0.00226
    mevals_sph = np.array([[Di, Di, Di], [De, De, De]])
    f = 0.5
    frac_sph = [f * 100, (1.0 - f) * 100]
    signal_sph, dt_sph, kt_sph = multi_tensor_dki(gtab_3s, mevals_sph, S0=100,
                                                  fractions=frac_sph, snr=None)
    # Compute ground truth values
    MDgt = f * Di + (1 - f) * De
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
                f = random.uniform(0.0, 1)
                frac = [f * 100, (1.0 - f) * 100]
                signal_i, dt_i, kt_i = multi_tensor_dki(gtab_3s, mevals_sph,
                                                        S0=100, fractions=frac,
                                                        snr=None)
                DWI[i, j, k] = signal_i
                md_i = f*Di + (1-f)*De
                mk_i = 3 * f * (1-f) * ((Di-De) / md_i) ** 2
                MDgt_multi[i, j, k] = md_i
                MKgt_multi[i, j, k] = mk_i
                S0gt_multi[i, j, k] = 100
                params_multi[i, j, k, 0] = md_i
                params_multi[i, j, k, 1] = mk_i
                MDWI[i, j, k, 0] = signal_i[0]
                MDWI[i, j, k, 1] = signal_i[1]
                MDWI[i, j, k, 2] = signal_i[100]
                MDWI[i, j, k, 3] = signal_i[180]


def teardown_module():
    global gtab_3s, signal_sph, msignal_sph, DWI, MDWI
    global bvecs, gtab, params_single, params_multi, MKgt_multi
    global bvals_3s, MDgt_multi, S0gt_multi, MKgt, MDgt

    gtab_3s, signal_sph, msignal_sph, DWI, MDWI = None, None, None, None, None
    gtab, params_single, params_multi, MKgt_multi = None, None, None, None
    bvals_3s, MDgt_multi, S0gt_multi, MKgt, MDgt = None, None, None, None, None
    bvecs = None


def test_msdki_predict():
    dkiM = msdki.MeanDiffusionKurtosisModel(gtab_3s)

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


def test_errors():
    # first error raises if MeanDiffusionKurtosisModel is called for
    # data will only one non-zero b-value
    assert_raises(ValueError, msdki.MeanDiffusionKurtosisModel, gtab)

    # second error raises if negative signal is given to MeanDiffusionKurtosis
    # model
    assert_raises(ValueError, msdki.MeanDiffusionKurtosisModel, gtab_3s,
                  min_signal=-1)

    # third error raises if wrong mask is given to fit
    mask_wrong = np.ones((2, 3, 1))
    msdki_model = msdki.MeanDiffusionKurtosisModel(gtab_3s)
    assert_raises(ValueError, msdki_model.fit, DWI, mask=mask_wrong)

    # fourth error raises if an given index point to more dimensions that data
    # does not contain

    # define auxiliary function for the assert raises
    def aux_test_fun(ob, ind):
        met = ob[ind].msk
        return met

    mdkiF = msdki_model.fit(DWI)
    assert_raises(IndexError, aux_test_fun, mdkiF, (0, 0, 0, 0))
    # checking if aux_test_fun runs fine
    met = aux_test_fun(mdkiF, (0, 0, 0))
    assert_array_almost_equal(MKgt_multi[0, 0, 0], met)

    # Fifth error rises if wrong mask is given to awf_from_msk
    assert_raises(ValueError, awf_from_msk, MKgt_multi, mask=mask_wrong)


def test_design_matrix():
    ub = unique_bvals_magnitude(bvals_3s)
    D = msdki.design_matrix(ub)
    Dgt = np.ones((4, 3))
    Dgt[:, 0] = -ub
    Dgt[:, 1] = 1.0/6 * ub ** 2
    assert_array_almost_equal(D, Dgt)


def test_msignal():
    # Multi-voxel case
    ms, ng = msdki.mean_signal_bvalue(DWI, gtab_3s)
    assert_array_almost_equal(ms, MDWI)
    assert_array_almost_equal(ng, np.array([3, 64, 64, 64]))

    # Single-voxel case
    ms, ng = msdki.mean_signal_bvalue(signal_sph, gtab_3s)
    assert_array_almost_equal(ng, np.array([3, 64, 64, 64]))
    assert_array_almost_equal(ms, msignal_sph)


def test_msdki_statistics():
    # tests if MD and MK are equal to expected values of a spherical
    # tensors

    # Multi-tensors
    ub = unique_bvals_magnitude(bvals_3s)
    design_matrix = msdki.design_matrix(ub)
    msignal, ng = msdki.mean_signal_bvalue(DWI, gtab_3s, bmag=None)
    params = msdki.wls_fit_msdki(design_matrix, msignal, ng)
    assert_array_almost_equal(params[..., 1], MKgt_multi)
    assert_array_almost_equal(params[..., 0], MDgt_multi)

    mdkiM = msdki.MeanDiffusionKurtosisModel(gtab_3s)
    mdkiF = mdkiM.fit(DWI)
    mk = mdkiF.msk
    md = mdkiF.msd
    assert_array_almost_equal(MKgt_multi, mk)
    assert_array_almost_equal(MDgt_multi, md)

    # Single-tensors
    mdkiF = mdkiM.fit(signal_sph)
    mk = mdkiF.msk
    md = mdkiF.msd
    assert_array_almost_equal(MKgt, mk)
    assert_array_almost_equal(MDgt, md)

    # Test with given mask
    mask = np.ones(DWI.shape[:-1])
    v = (0, 0, 0)
    mask[1, 1, 1] = 0
    mdkiF = mdkiM.fit(DWI, mask=mask)
    mk = mdkiF.msk
    md = mdkiF.msd
    assert_array_almost_equal(MKgt_multi, mk)
    assert_array_almost_equal(MDgt_multi, md)
    assert_array_almost_equal(MKgt_multi[v], mdkiF[v].msk)  # tuple case
    assert_array_almost_equal(MDgt_multi[v], mdkiF[v].msd)  # tuple case
    assert_array_almost_equal(MKgt_multi[0], mdkiF[0].msk)  # not tuple case
    assert_array_almost_equal(MDgt_multi[0], mdkiF[0].msd)  # not tuple case

    # Test returned S0
    mdkiM = msdki.MeanDiffusionKurtosisModel(gtab_3s, return_S0_hat=True)
    mdkiF = mdkiM.fit(DWI)
    assert_array_almost_equal(S0gt_multi, mdkiF.S0_hat)
    assert_array_almost_equal(MKgt_multi[v], mdkiF[v].msk)


def test_kurtosis_to_smt2_conversion():
    # 1. Check conversion of smt2 awf to kurtosis
    # When awf = 0 kurtosis was to be 0
    awf0 = 0
    kexp0 = 0
    kest0 = msk_from_awf(awf0)
    assert_almost_equal(kest0, kexp0)

    # When awf = 1 kurtosis was to be 2.4
    awf1 = 1
    kexp1 = 2.4
    kest1 = msk_from_awf(awf1)
    assert_almost_equal(kest1, kexp1)

    # Check the inversion of msk_from_awf
    awf_test_array = np.linspace(0, 1, 100)
    k_exp = msk_from_awf(awf_test_array)
    awf_from_k = awf_from_msk(k_exp)
    assert_array_almost_equal(awf_from_k, awf_test_array)

    # Check the awf_from_msk estimates when kurtosis is out of expected
    # interval ranges - note that under SMT2 assumption MSK is never lower
    # than 0 and never higher than 2.4. Since SMT2 assumptions are commonly not
    # met kurtosis can be out of this expected range. So, if MSK is lower than
    # 0, f is set to 0 (avoiding negative f). On the other hand, if MSK is
    # higher than 2.4, f is set to the maximum value of 1.
    assert_array_almost_equal(awf_from_msk(np.array([-0.1, 2.5])),
                              np.array([0., 1.]))

    # if msk = np.nan, function outputs awf=np.nan
    assert_(np.isnan(awf_from_msk(np.array(np.nan))))


def test_smt2_metrics():
    # Just checking if parameters can be retrieved from MSDKI's fit class obj

    # Based on the multi-voxel simulations above (computes gt for SMT2 params)
    AWFgt = awf_from_msk(MKgt_multi)
    DIgt = 3 * MDgt_multi / (1 + 2 * (1 - AWFgt) ** 2)
    # General microscopic anisotropy estimation (Eq 8 Henriques et al MRM 2019)
    RDe = DIgt * (1 - AWFgt)  # tortuosity assumption
    VarD = 2/9 * (AWFgt * DIgt ** 2 + (1 - AWFgt) * (DIgt - RDe) ** 2)
    MD = (AWFgt * DIgt + (1 - AWFgt) * (DIgt + 2 * RDe)) / 3
    uFAgt = np.sqrt(3 / 2 * VarD[MD > 0] / (VarD[MD > 0] + MD[MD > 0] ** 2))

    mdkiM = msdki.MeanDiffusionKurtosisModel(gtab_3s)
    mdkiF = mdkiM.fit(DWI)
    assert_array_almost_equal(mdkiF.smt2f, AWFgt)
    assert_array_almost_equal(mdkiF.smt2di, DIgt)
    assert_array_almost_equal(mdkiF.smt2uFA[MD > 0], uFAgt)

    # Check if awf_from_msk when mask is given
    mask = MKgt_multi > 0
    AWF = awf_from_msk(MKgt_multi, mask)
    assert_array_almost_equal(AWF, AWFgt)


def test_smt2_specific_cases():
    mdkiM = msdki.MeanDiffusionKurtosisModel(gtab_3s)

    # Check smt2 is specific cases with known g.t:
    # 1) Intrinsic diffusion is equal MSD for single Gaussian isotropic
    #     diffusion (i.e. awf=0)
    sig_gaussian = single_tensor(gtab_3s, evals=np.array([2e-3, 2e-3, 2e-3]))
    mdkiF = mdkiM.fit(sig_gaussian)
    assert_almost_equal(mdkiF.msk, 0.0)
    assert_almost_equal(mdkiF.msd, 2.0e-3)
    assert_almost_equal(mdkiF.smt2f, 0)
    assert_almost_equal(mdkiF.smt2di, 2.0e-3)

    # 2) Intrinsic diffusion is equal to MSD/3 for single powder-averaged stick
    #    compartment
    Da = 2.0e-3
    mevals = np.zeros((64, 3))
    mevals[:, 0] = Da
    fracs = np.ones(64) * 100 / 64
    signal_pa, dt_sph, kt_sph = multi_tensor_dki(gtab_3s, mevals,
                                                 angles=bvecs[1:, :],
                                                 fractions=fracs, snr=None)
    mdkiF = mdkiM.fit(signal_pa)
    # decimal is set to 1 because of finite number of directions for powder
    # average calculation
    assert_almost_equal(mdkiF.msk, 2.4, decimal=1)
    assert_almost_equal(mdkiF.msd * 1000, Da/3 * 1000, decimal=1)
    assert_almost_equal(mdkiF.smt2f, 1, decimal=1)
    assert_almost_equal(mdkiF.smt2di, mdkiF.msd * 3, decimal=1)
