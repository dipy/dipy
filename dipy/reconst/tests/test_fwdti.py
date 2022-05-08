""" Testing Free Water Elimination Model """

import numpy as np
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
from dipy.reconst.fwdti import fwdti_prediction
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_raises)
from dipy.reconst.dti import (from_lower_triangular, decompose_tensor,
                              fractional_anisotropy)
from dipy.reconst.fwdti import (lower_triangular_to_cholesky,
                                cholesky_to_lower_triangular,
                                nls_fit_tensor, wls_fit_tensor)
from dipy.sims.voxel import (multi_tensor, single_tensor,
                             all_tensor_evecs, multi_tensor_dki)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames


def setup_module():
    """Module-level setup"""
    global gtab, gtab_2s, mevals, model_params_mv
    global DWI, FAref, GTF, MDref, FAdti, MDdti
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    # FW model requires multishell data
    bvals_2s = np.concatenate((bvals, bvals * 1.5), axis=0)
    bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
    gtab_2s = gradient_table(bvals_2s, bvecs_2s)

    # Simulation a typical DT and DW signal for no water contamination
    S0 = np.array(100)
    dt = np.array([0.0017, 0, 0.0003, 0, 0, 0.0003])
    evals, evecs = decompose_tensor(from_lower_triangular(dt))
    S_tissue = single_tensor(gtab_2s, S0=100, evals=evals, evecs=evecs,
                             snr=None)
    dm = dti.TensorModel(gtab_2s, 'WLS')
    dtifit = dm.fit(S_tissue)
    FAdti = dtifit.fa
    MDdti = dtifit.md
    dtiparams = dtifit.model_params

    # Simulation of 8 voxels tested
    DWI = np.zeros((2, 2, 2, len(gtab_2s.bvals)))
    FAref = np.zeros((2, 2, 2))
    MDref = np.zeros((2, 2, 2))
    # Diffusion of tissue and water compartments are constant for all voxel
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    # volume fractions
    GTF = np.array([[[0.06, 0.71], [0.33, 0.91]],
                    [[0., 0.], [0., 0.]]])
    # S0 multivoxel
    S0m = 100 * np.ones((2, 2, 2))
    # model_params ground truth (to be fill)
    model_params_mv = np.zeros((2, 2, 2, 13))
    for i in range(2):
        for j in range(2):
            gtf = GTF[0, i, j]
            S, p = multi_tensor(gtab_2s, mevals, S0=100,
                                angles=[(90, 0), (90, 0)],
                                fractions=[(1-gtf) * 100, gtf*100], snr=None)
            DWI[0, i, j] = S
            FAref[0, i, j] = FAdti
            MDref[0, i, j] = MDdti
            R = all_tensor_evecs(p[0])
            R = R.reshape(9)
            model_params_mv[0, i, j] = \
                np.concatenate(([0.0017, 0.0003, 0.0003], R, [gtf]), axis=0)


def test_fwdti_singlevoxel():
    # Simulation when water contamination is added
    gtf = 0.44444  # ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f
    MDfwe = fwefit.md

    assert_almost_equal(FAdti, FAfwe, decimal=3)
    assert_almost_equal(Ffwe, gtf, decimal=3)
    assert_almost_equal(MDfwe, MDdti, decimal=3)

    # Test non-linear fit
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', cholesky=False)
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f
    MDfwe = fwefit.md

    assert_almost_equal(FAdti, FAfwe)
    assert_almost_equal(Ffwe, gtf)
    assert_almost_equal(MDfwe, MDdti)

    # Test cholesky
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', cholesky=True)
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f
    MDfwe = fwefit.md

    assert_almost_equal(FAdti, FAfwe)
    assert_almost_equal(Ffwe, gtf)
    assert_almost_equal(MDfwe, MDfwe)


def test_fwdti_precision():
    # Simulation when water contamination is added
    gtf = 0.63416  # ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS', piterations=5)
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f
    MDfwe = fwefit.md

    assert_almost_equal(FAdti, FAfwe, decimal=5)
    assert_almost_equal(Ffwe, gtf, decimal=5)
    assert_almost_equal(MDfwe, MDdti, decimal=5)


def test_fwdti_multi_voxel():
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', cholesky=False)
    fwefit = fwdm.fit(DWI)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f
    MDfwe = fwefit.md

    assert_almost_equal(FAfwe, FAref)
    assert_almost_equal(Ffwe, GTF)
    assert_almost_equal(MDfwe, MDref)

    # Test Cholesky
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', cholesky=True)
    fwefit = fwdm.fit(DWI)
    FAfwe = fwefit.fa
    Ffwe = fwefit.f
    MDfwe = fwefit.md

    assert_almost_equal(FAfwe, FAref)
    assert_almost_equal(Ffwe, GTF)
    assert_almost_equal(MDfwe, MDref)


def test_fwdti_predictions():
    # single voxel case
    gtf = 0.50  # ground truth volume fraction
    angles = [(90, 0), (90, 0)]
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=angles,
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    R = all_tensor_evecs(peaks[0])
    R = R.reshape(9)
    model_params = np.concatenate(([0.0017, 0.0003, 0.0003], R, [gtf]),
                                  axis=0)
    S_pred1 = fwdti_prediction(model_params, gtab_2s, S0=100)
    assert_array_almost_equal(S_pred1, S_conta)

    # Testing in model class
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s)
    S_pred2 = fwdm.predict(model_params, S0=100)
    assert_array_almost_equal(S_pred2, S_conta)

    # Testing in fit class
    fwefit = fwdm.fit(S_conta)
    S_pred3 = fwefit.predict(gtab_2s, S0=100)
    assert_array_almost_equal(S_pred3, S_conta, decimal=5)

    # Multi voxel simulation
    S_pred1 = fwdti_prediction(model_params_mv, gtab_2s, S0=100)  # function
    assert_array_almost_equal(S_pred1, DWI)
    S_pred2 = fwdm.predict(model_params_mv, S0=100)  # Model class
    assert_array_almost_equal(S_pred2, DWI)
    fwefit = fwdm.fit(DWI)  # Fit class
    S_pred3 = fwefit.predict(gtab_2s, S0=100)
    assert_array_almost_equal(S_pred3, DWI)


def test_fwdti_errors():
    # 1st error - if a unknown fit method is given to the FWTM
    assert_raises(ValueError, fwdti.FreeWaterTensorModel, gtab_2s,
                  fit_method="pKT")
    # 2nd error - if incorrect mask is given
    fwdtiM = fwdti.FreeWaterTensorModel(gtab_2s)
    incorrect_mask = np.array([[True, True, False], [True, False, False]])
    assert_raises(ValueError, fwdtiM.fit, DWI, mask=incorrect_mask)

    # 3rd error - if data with only one non zero b-value is given
    assert_raises(ValueError, fwdti.FreeWaterTensorModel, gtab)

    # Testing the correct usage
    fwdtiM = fwdti.FreeWaterTensorModel(gtab_2s, min_signal=1)
    correct_mask = np.zeros((2, 2, 2))
    correct_mask[0, :, :] = 1
    correct_mask = correct_mask > 0
    fwdtiF = fwdtiM.fit(DWI, mask=correct_mask)
    assert_array_almost_equal(fwdtiF.fa, FAref)
    assert_array_almost_equal(fwdtiF.f, GTF)

    # 4th error - if a sigma is selected by no value of sigma is given
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', weighting='sigma')
    assert_raises(ValueError, fwdm.fit, DWI)


def test_fwdti_restore():
    # Restore has to work well even in nonproblematic cases
    # Simulate a signal corrupted by free water diffusion contamination
    gtf = 0.50  # ground truth volume fraction
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', weighting='sigma',
                                      sigma=4)
    fwdtiF = fwdm.fit(S_conta)
    assert_array_almost_equal(fwdtiF.fa, FAdti)
    assert_array_almost_equal(fwdtiF.f, gtf)
    fwdm2 = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', weighting='gmm')
    fwdtiF2 = fwdm2.fit(S_conta)
    assert_array_almost_equal(fwdtiF2.fa, FAdti)
    assert_array_almost_equal(fwdtiF2.f, gtf)


def test_cholesky_functions():
    S, dt, kt = multi_tensor_dki(gtab, mevals, S0=100,
                                 angles=[(45., 45.), (45., 45.)],
                                 fractions=[80, 20])
    R = lower_triangular_to_cholesky(dt)
    tensor = cholesky_to_lower_triangular(R)
    assert_array_almost_equal(dt, tensor)


def test_fwdti_jac_multi_voxel():
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    fwdm.fit(DWI[0, :, :])

    # no f transform
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', f_transform=False,
                                      jac=True)
    fwefit = fwdm.fit(DWI[0, :, :])
    Ffwe = fwefit.f
    assert_array_almost_equal(Ffwe, GTF[0, :])

    # with f transform
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS', f_transform=True,
                                      jac=True)
    fwefit = fwdm.fit(DWI[0, :, :])
    Ffwe = fwefit.f
    assert_array_almost_equal(Ffwe, GTF[0, :])


def test_standalone_functions():
    # WLS procedure
    params = wls_fit_tensor(gtab_2s, DWI)
    assert_array_almost_equal(params[..., 12], GTF)
    fa = fractional_anisotropy(params[..., :3])
    assert_array_almost_equal(fa, FAref)

    # NLS procedure
    params = nls_fit_tensor(gtab_2s, DWI)
    assert_array_almost_equal(params[..., 12], GTF)
    fa = fractional_anisotropy(params[..., :3])
    assert_array_almost_equal(fa, FAref)


def test_md_regularization():
    # single voxel
    gtf = 0.97  # for this ground truth value, md is larger than 2.7e-3
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS')
    fwefit = fwdm.fit(S_conta)

    assert_array_almost_equal(fwefit.fa, 0.0)
    assert_array_almost_equal(fwefit.md, 0.0)
    assert_array_almost_equal(fwefit.f, 1.0)

    # multi voxel
    DWI[0, 1, 1] = S_conta
    GTF[0, 1, 1] = 1
    FAref[0, 1, 1] = 0
    MDref[0, 1, 1] = 0

    fwefit = fwdm.fit(DWI)
    assert_array_almost_equal(fwefit.fa, FAref)
    assert_array_almost_equal(fwefit.md, MDref)
    assert_array_almost_equal(fwefit.f, GTF)


def test_negative_s0():
    # single voxel
    gtf = 0.55
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(90, 0), (90, 0)],
                                  fractions=[(1-gtf) * 100, gtf*100], snr=None)
    S_conta[gtab_2s.bvals == 0] = -100
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'NLS')
    fwefit = fwdm.fit(S_conta)
    assert_array_almost_equal(fwefit.fa, 0.0)
    assert_array_almost_equal(fwefit.md, 0.0)
    assert_array_almost_equal(fwefit.f, 0.0)

    # multi voxel
    DWI[0, 0, 1, gtab_2s.bvals == 0] = -100
    GTF[0, 0, 1] = 0
    FAref[0, 0, 1] = 0
    MDref[0, 0, 1] = 0
    fwefit = fwdm.fit(DWI)
    assert_array_almost_equal(fwefit.fa, FAref)
    assert_array_almost_equal(fwefit.md, MDref)
    assert_array_almost_equal(fwefit.f, GTF)
