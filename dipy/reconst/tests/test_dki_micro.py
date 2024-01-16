""" Testing DKI microstructure """

import numpy as np
import random
import dipy.reconst.dki_micro as dki_micro
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_, assert_raises, assert_allclose)
from dipy.sims.voxel import (multi_tensor_dki, _check_directions, multi_tensor)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.reconst.dti import (eig_from_lo_tri)

from dipy.data import default_sphere, get_sphere

gtab_2s, DWIsim, DWIsim_all_taylor = None, None, None
FIE, RDI, ADI, ADE, Tor, RDE = None, None, None, None, None, None


def setup_module():
    global gtab_2s, DWIsim, DWIsim_all_taylor, FIE, RDI, ADI, ADE, Tor, RDE

    fimg, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    # 2 shells for techniques that requires multishell data
    bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
    bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
    gtab_2s = gradient_table(bvals_2s, bvecs_2s)

    # single fiber simulate (which is the assumption of our model)
    FIE = np.array([[[0.30, 0.32], [0.74, 0.51]],
                    [[0.47, 0.21], [0.80, 0.63]]])
    RDI = np.zeros((2, 2, 2))
    ADI = np.array([[[1e-3, 1.3e-3], [0.8e-3, 1e-3]],
                    [[0.9e-3, 0.99e-3], [0.89e-3, 1.1e-3]]])
    ADE = np.array([[[2.2e-3, 2.3e-3], [2.8e-3, 2.1e-3]],
                    [[1.9e-3, 2.5e-3], [1.89e-3, 2.1e-3]]])
    Tor = np.array([[[2.6, 2.4], [2.8, 2.1]],
                    [[2.9, 2.5], [2.7, 2.3]]])
    RDE = ADE / Tor

    # prepare simulation:
    DWIsim = np.zeros((2, 2, 2, gtab_2s.bvals.size))

    # Diffusion microstructural model assumes that signal does not have Taylor
    # approximation components larger than the fourth order. Thus parameter
    # estimates are only equal to the ground truth values of the simulation
    # if signals taylor components larger than the fourth order are removed.
    # Signal without this taylor components can be generated using the
    # multi_tensor_dki simulations. Therefore we used this function to test the
    # expected estimates of the model.

    DWIsim_all_taylor = np.zeros((2, 2, 2, gtab_2s.bvals.size))

    # Signal with all taylor components can be simulated using the function
    # multi_tensor. Generating this signals will be useful to test the
    # prediction procedures of DKI-based microstructural model.

    for i in range(2):
        for j in range(2):
            for k in range(2):
                ADi = ADI[i, j, k]
                RDi = RDI[i, j, k]
                ADe = ADE[i, j, k]
                RDe = RDE[i, j, k]
                fie = FIE[i, j, k]
                mevals = np.array([[ADi, RDi, RDi], [ADe, RDe, RDe]])
                frac = [fie*100, (1 - fie)*100]
                theta = random.uniform(0, 180)
                phi = random.uniform(0, 320)
                angles = [(theta, phi), (theta, phi)]
                signal, dt, kt = multi_tensor_dki(gtab_2s, mevals,
                                                  angles=angles,
                                                  fractions=frac, snr=None)
                DWIsim[i, j, k, :] = signal
                signal, sticks = multi_tensor(gtab_2s, mevals, angles=angles,
                                              fractions=frac, snr=None)
                DWIsim_all_taylor[i, j, k, :] = signal


def teardown_module():
    global gtab_2s, DWIsim, DWIsim_all_taylor, FIE, RDI, ADI, ADE, Tor, RDE
    gtab_2s, DWIsim, DWIsim_all_taylor = None, None, None
    FIE, RDI, ADI, ADE, Tor, RDE = None, None, None, None, None, None


def test_single_fiber_model():
    # single fiber simulate (which is the assumption of our model)
    fie = 0.49
    ADi = 0.00099
    ADe = 0.00226
    RDi = 0
    RDe = 0.00087

    # prepare simulation:
    theta = random.uniform(0, 180)
    phi = random.uniform(0, 320)
    angles = [(theta, phi), (theta, phi)]
    mevals = np.array([[ADi, RDi, RDi], [ADe, RDe, RDe]])
    frac = [fie*100, (1 - fie)*100]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)
    # DKI fit
    dkiM = dki_micro.DiffusionKurtosisModel(gtab_2s, fit_method="WLS")
    dkiF = dkiM.fit(signal)

    # Axonal Water Fraction
    AWF = dki_micro.axonal_water_fraction(dkiF.model_params, default_sphere,
                                          mask=None, gtol=1e-5)
    assert_almost_equal(AWF, fie)

    # Extra-cellular and intra-cellular components
    edt, idt = dki_micro.diffusion_components(dkiF.model_params,
                                              default_sphere)
    EDT = eig_from_lo_tri(edt)
    IDT = eig_from_lo_tri(idt)

    # check eigenvalues
    assert_array_almost_equal(EDT[0:3], np.array([ADe, RDe, RDe]))
    assert_array_almost_equal(IDT[0:3], np.array([ADi, RDi, RDi]))
    # first eigenvalue should be the direction of the fibers
    fiber_direction = _check_directions([(theta, phi)])
    f_norm = abs(np.dot(fiber_direction, np.array((EDT[3], EDT[6], EDT[9]))))
    assert_almost_equal(f_norm, 1.)
    f_norm = abs(np.dot(fiber_direction, np.array((IDT[3], IDT[6], IDT[9]))))
    assert_almost_equal(f_norm, 1.)

    # Test model and fit objects
    wmtiM = dki_micro.KurtosisMicrostructureModel(gtab_2s, fit_method="WLS")
    wmtiF = wmtiM.fit(signal)
    assert_allclose(wmtiF.awf, AWF, rtol=1e-6)
    assert_array_almost_equal(wmtiF.hindered_evals,
                              np.array([ADe, RDe, RDe]))
    assert_array_almost_equal(wmtiF.restricted_evals,
                              np.array([ADi, RDi, RDi]))
    assert_almost_equal(wmtiF.hindered_ad, ADe)
    assert_almost_equal(wmtiF.hindered_rd, RDe)
    assert_almost_equal(wmtiF.axonal_diffusivity, ADi)
    assert_almost_equal(wmtiF.tortuosity, ADe/RDe, decimal=4)

    # Test diffusion_components when a kurtosis tensors is associated with
    # negative kurtosis values. E.g of this cases is given below:
    dkiparams = np.array([1.67135726e-03, 5.03651205e-04, 9.35365328e-05,
                          -7.11167583e-01, 6.23186820e-01, -3.25390313e-01,
                          -1.75247376e-02, -4.78415563e-01, -8.77958674e-01,
                          7.02804064e-01, 6.18673368e-01, -3.51154825e-01,
                          2.18384153, -2.76378153e-02, 2.22893297,
                          -2.68306546e-01, -1.28411610, -1.56557645e-01,
                          -1.80850619e-01, -8.33152110e-01, -3.62410766e-01,
                          1.57775442e-01, 8.73775381e-01, 2.77188975e-01,
                          -3.67415502e-02, -1.56330984e-01, -1.62295407e-02])
    edt, idt = dki_micro.diffusion_components(dkiparams)
    assert_(np.all(np.isfinite(edt)))


def test_wmti_model_multi_voxel():
    # DKI fit
    dkiM = dki_micro.DiffusionKurtosisModel(gtab_2s, fit_method="WLS")
    dkiF = dkiM.fit(DWIsim)

    # Axonal Water Fraction
    sphere = get_sphere()
    AWF = dki_micro.axonal_water_fraction(dkiF.model_params, sphere, mask=None,
                                          gtol=1e-5)
    assert_almost_equal(AWF, FIE)

    # Extra-cellular and intra-cellular components
    edt, idt = dki_micro.diffusion_components(dkiF.model_params, sphere)
    EDT = eig_from_lo_tri(edt)
    IDT = eig_from_lo_tri(idt)

    # check eigenvalues
    assert_array_almost_equal(EDT[..., 0], ADE, decimal=3)
    assert_array_almost_equal(EDT[..., 1], RDE, decimal=3)
    assert_array_almost_equal(EDT[..., 2], RDE, decimal=3)
    assert_array_almost_equal(IDT[..., 0], ADI, decimal=3)
    assert_array_almost_equal(IDT[..., 1], RDI, decimal=3)
    assert_array_almost_equal(IDT[..., 2], RDI, decimal=3)

    # Test methods performance when a signal with all zeros is present
    FIEc = FIE.copy()
    RDIc = RDI.copy()
    ADIc = ADI.copy()
    ADEc = ADE.copy()
    Torc = Tor.copy()
    RDEc = RDE.copy()
    DWIsimc = DWIsim.copy()

    FIEc[0, 0, 0] = 0
    RDIc[0, 0, 0] = 0
    ADIc[0, 0, 0] = 0
    ADEc[0, 0, 0] = 0
    Torc[0, 0, 0] = 0
    RDEc[0, 0, 0] = 0
    DWIsimc[0, 0, 0, :] = 0
    mask = np.ones((2, 2, 2))
    mask[0, 0, 0] = 0

    dkiF = dkiM.fit(DWIsimc)
    awf = dki_micro.axonal_water_fraction(dkiF.model_params, sphere,
                                          gtol=1e-5)
    assert_almost_equal(awf, FIEc)

    # Extra-cellular and intra-cellular components
    edt, idt = dki_micro.diffusion_components(dkiF.model_params, sphere,
                                              awf=awf)
    EDT = eig_from_lo_tri(edt)
    IDT = eig_from_lo_tri(idt)
    assert_array_almost_equal(EDT[..., 0], ADEc, decimal=3)
    assert_array_almost_equal(EDT[..., 1], RDEc, decimal=3)
    assert_array_almost_equal(EDT[..., 2], RDEc, decimal=3)
    assert_array_almost_equal(IDT[..., 0], ADIc, decimal=3)
    assert_array_almost_equal(IDT[..., 1], RDIc, decimal=3)
    assert_array_almost_equal(IDT[..., 2], RDIc, decimal=3)

    # Check when mask is given
    dkiF = dkiM.fit(DWIsim)
    awf = dki_micro.axonal_water_fraction(dkiF.model_params, sphere,
                                          gtol=1e-5, mask=mask)
    assert_almost_equal(awf, FIEc, decimal=3)

    # Extra-cellular and intra-cellular components
    edt, idt = dki_micro.diffusion_components(dkiF.model_params, sphere,
                                              awf=awf, mask=mask)
    EDT = eig_from_lo_tri(edt)
    IDT = eig_from_lo_tri(idt)
    assert_array_almost_equal(EDT[..., 0], ADEc, decimal=3)
    assert_array_almost_equal(EDT[..., 1], RDEc, decimal=3)
    assert_array_almost_equal(EDT[..., 2], RDEc, decimal=3)
    assert_array_almost_equal(IDT[..., 0], ADIc, decimal=3)
    assert_array_almost_equal(IDT[..., 1], RDIc, decimal=3)
    assert_array_almost_equal(IDT[..., 2], RDIc, decimal=3)

    # Check class object
    wmtiM = dki_micro.KurtosisMicrostructureModel(gtab_2s, fit_method="WLS")
    wmtiF = wmtiM.fit(DWIsim, mask=mask)
    assert_almost_equal(wmtiF.awf, FIEc, decimal=3)
    assert_almost_equal(wmtiF.axonal_diffusivity, ADIc, decimal=3)
    assert_almost_equal(wmtiF.hindered_ad, ADEc, decimal=3)
    assert_almost_equal(wmtiF.hindered_rd, RDEc, decimal=3)
    assert_almost_equal(wmtiF.tortuosity, Torc, decimal=3)


def test_dki_micro_predict_single_voxel():
    # single fiber simulate (which is the assumption of our model)
    fie = 0.49
    ADi = 0.00099
    ADe = 0.00226
    RDi = 0
    RDe = 0.00087

    # prepare simulation:
    theta = random.uniform(0, 180)
    phi = random.uniform(0, 320)
    angles = [(theta, phi), (theta, phi)]
    mevals = np.array([[ADi, RDi, RDi], [ADe, RDe, RDe]])
    frac = [fie*100, (1 - fie)*100]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)
    signal_gt, da = multi_tensor(gtab_2s, mevals, angles=angles,
                                 fractions=frac, snr=None)

    # Defined DKI microstrutural model
    dkiM = dki_micro.KurtosisMicrostructureModel(gtab_2s)

    # Fit single voxel signal
    dkiF = dkiM.fit(signal)

    # Check predict of KurtosisMicrostruturalModel
    pred = dkiM.predict(dkiF.model_params)
    assert_array_almost_equal(pred, signal_gt, decimal=4)

    pred = dkiM.predict(dkiF.model_params, S0=100)
    assert_array_almost_equal(pred, signal_gt * 100, decimal=4)

    # Check predict of KurtosisMicrostruturalFit
    pred = dkiF.predict(gtab_2s, S0=100)
    assert_array_almost_equal(pred, signal_gt * 100, decimal=4)


def test_dki_micro_predict_multi_voxel():
    dkiM = dki_micro.KurtosisMicrostructureModel(gtab_2s)
    dkiF = dkiM.fit(DWIsim)

    # Check predict of KurtosisMicrostruturalModel
    pred = dkiM.predict(dkiF.model_params)
    assert_array_almost_equal(pred, DWIsim_all_taylor, decimal=3)

    pred = dkiM.predict(dkiF.model_params, S0=100)
    assert_array_almost_equal(pred, DWIsim_all_taylor * 100, decimal=3)

    # Check predict of KurtosisMicrostruturalFit
    pred = dkiF.predict(gtab_2s, S0=100)
    assert_array_almost_equal(pred, DWIsim_all_taylor * 100, decimal=3)


def _help_test_awf_only(dkimicrofit, string):
    exec(string)


def test_dki_micro_awf_only():
    dkiM = dki_micro.KurtosisMicrostructureModel(gtab_2s)
    dkiF = dkiM.fit(DWIsim, awf_only=True)
    awf = dkiF.awf
    assert_almost_equal(awf, FIE, decimal=3)

    # assert_raises(dkiF.hindered_evals)
    assert_raises(ValueError, _help_test_awf_only, dkiF,
                  'dkimicrofit.hindered_evals')
    assert_raises(ValueError, _help_test_awf_only, dkiF,
                  'dkimicrofit.restricted_evals')
    assert_raises(ValueError, _help_test_awf_only, dkiF,
                  'dkimicrofit.axonal_diffusivity')
    assert_raises(ValueError, _help_test_awf_only, dkiF,
                  'dkimicrofit.hindered_ad')
    assert_raises(ValueError, _help_test_awf_only, dkiF,
                  'dkimicrofit.hindered_rd')
    assert_raises(ValueError, _help_test_awf_only, dkiF,
                  'dkimicrofit.tortuosity')


def additional_tortuosity_tests():
    # Test tortuosity when rd is zero
    # single voxel
    t = dki_micro.tortuosity(1.7e-3, 0.0)
    assert_almost_equal(t, 0.0)

    # multi-voxel
    RDEc = RDE.copy()
    Torc = Tor.copy()
    RDEc[1, 1, 1] = 0.0
    Torc[1, 1, 1] = 0.0
    t = dki_micro.tortuosity(ADE, RDEc)
    assert_almost_equal(Torc, t)
