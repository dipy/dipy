""" Testing DKI microstructure """

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import dipy.reconst.dki_micro as dki_micro
from numpy.testing import (assert_array_almost_equal, assert_almost_equal)
from dipy.sims.voxel import (multi_tensor_dki, _check_directions)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_data
from dipy.reconst.dti import (eig_from_lo_tri)

from dipy.data import get_sphere

fimg, fbvals, fbvecs = get_data('small_64D')
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
DWIsim = np.zeros((2., 2., 2., gtab_2s.bvals.size))

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
    sphere = get_sphere('symmetric724')
    AWF = dki_micro.axonal_water_fraction(dkiF.model_params, sphere, mask=None,
                                          gtol=1e-5)
    assert_almost_equal(AWF, fie)

    # Extra-cellular and intra-cellular components
    edt, idt = dki_micro.diffusion_components(dkiF.model_params, sphere)
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
    assert_almost_equal(wmtiF.awf, AWF)
    assert_almost_equal(wmtiF.axonal_diffusivity, ADi)
    assert_array_almost_equal(wmtiF.hindered_evals,
                              np.array([ADe, RDe, RDe]))
    assert_array_almost_equal(wmtiF.restricted_evals,
                              np.array([ADi, RDi, RDi]))
    assert_almost_equal(wmtiF.hindered_ad, ADe)
    assert_almost_equal(wmtiF.hindered_rd, RDe)
    assert_almost_equal(wmtiF.tortuosity, ADe/RDe)


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
    assert_array_almost_equal(EDT[..., 0], ADE)
    assert_array_almost_equal(EDT[..., 1], RDE)
    assert_array_almost_equal(EDT[..., 2], RDE)
    assert_array_almost_equal(IDT[..., 0], ADI)
    assert_array_almost_equal(IDT[..., 1], RDI)
    assert_array_almost_equal(IDT[..., 2], RDI)

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
    mask = np.ones((2., 2., 2.))
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
    assert_array_almost_equal(EDT[..., 0], ADEc)
    assert_array_almost_equal(EDT[..., 1], RDEc)
    assert_array_almost_equal(EDT[..., 2], RDEc)
    assert_array_almost_equal(IDT[..., 0], ADIc)
    assert_array_almost_equal(IDT[..., 1], RDIc)
    assert_array_almost_equal(IDT[..., 2], RDIc)

    # Check when mask is given
    dkiF = dkiM.fit(DWIsim)
    awf = dki_micro.axonal_water_fraction(dkiF.model_params, sphere,
                                          gtol=1e-5, mask=mask)
    assert_almost_equal(awf, FIEc)

    # Extra-cellular and intra-cellular components
    edt, idt = dki_micro.diffusion_components(dkiF.model_params, sphere,
                                              awf=awf, mask=mask)
    EDT = eig_from_lo_tri(edt)
    IDT = eig_from_lo_tri(idt)
    assert_array_almost_equal(EDT[..., 0], ADEc)
    assert_array_almost_equal(EDT[..., 1], RDEc)
    assert_array_almost_equal(EDT[..., 2], RDEc)
    assert_array_almost_equal(IDT[..., 0], ADIc)
    assert_array_almost_equal(IDT[..., 1], RDIc)
    assert_array_almost_equal(IDT[..., 2], RDIc)

    # Check class object
    wmtiM = dki_micro.KurtosisMicrostructureModel(gtab_2s, fit_method="WLS")
    wmtiF = wmtiM.fit(DWIsim, mask=mask)
    assert_almost_equal(wmtiF.awf, FIEc)
    assert_almost_equal(wmtiF.axonal_diffusivity, ADIc)
    assert_almost_equal(wmtiF.hindered_ad, ADEc)
    assert_almost_equal(wmtiF.hindered_rd, RDEc)
    assert_almost_equal(wmtiF.tortuosity, Torc)


def test_dki_micro_predict():
    dkiM = dki_micro.KurtosisMicrostructureModel(gtab_2s)
    dkiF = dkiM.fit(DWIsim)

    # Check predict of KurtosisMicrostruturalModel
    pred = dkiM.predict(dkiF.model_params)
    assert_array_almost_equal(pred, DWIsim)

    pred = dkiM.predict(dkiF.model_params, S0=100)
    assert_array_almost_equal(pred, DWIsim * 100)

    # Check predict of KurtosisMicrostruturalFit
    pred = dkiF.predict(gtab_2s, S0=100)
    assert_array_almost_equal(pred, DWIsim * 100)


def test_dki_micro_awf_only():
    dkiM = dki_micro.KurtosisMicrostructureModel(gtab_2s)
    dkiF = dkiM.fit(DWIsim, awf_only=True)
    awf = dki_micro.axonal_water_fraction(dkiF.model_params)
    assert_almost_equal(awf, FIE)

