""" Testing DKI """

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import dipy.reconst.dki as dki
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)
from nose.tools import assert_raises
from dipy.sims.voxel import multi_tensor_dki
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_data
from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)
from dipy.reconst.dki import (mean_kurtosis, carlson_rf,  carlson_rd,
                              axial_kurtosis, radial_kurtosis, _positive_evals)

from dipy.core.sphere import Sphere

from dipy.core.geometry import perpendicular_directions

fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

# 2 shells for techniques that requires multishell data
bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
gtab_2s = gradient_table(bvals_2s, bvecs_2s)

# Simulation 1. signals of two crossing fibers are simulated
mevals_cross = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                         [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
angles_cross = [(80, 10), (80, 10), (20, 30), (20, 30)]
fie = 0.49
frac_cross = [fie*50, (1-fie) * 50, fie*50, (1-fie) * 50]
# Noise free simulates
signal_cross, dt_cross, kt_cross = multi_tensor_dki(gtab_2s, mevals_cross,
                                                    S0=100,
                                                    angles=angles_cross,
                                                    fractions=frac_cross,
                                                    snr=None)
evals_cross, evecs_cross = decompose_tensor(from_lower_triangular(dt_cross))
crossing_ref = np.concatenate((evals_cross, evecs_cross[0], evecs_cross[1],
                               evecs_cross[2], kt_cross), axis=0)

# Simulation 2. Spherical kurtosis tensor.- for white matter, this can be a
# biological implaussible scenario, however this simulation is usefull for
# testing the estimation of directional apparent kurtosis and the mean
# kurtosis, since its directional and mean kurtosis ground truth are a constant
# which can be easly mathematicaly calculated.
Di = 0.00099
De = 0.00226
mevals_sph = np.array([[Di, Di, Di], [De, De, De]])
frac_sph = [50, 50]
signal_sph, dt_sph, kt_sph = multi_tensor_dki(gtab_2s, mevals_sph, S0=100,
                                              fractions=frac_sph,
                                              snr=None)
evals_sph, evecs_sph = decompose_tensor(from_lower_triangular(dt_sph))
params_sph = np.concatenate((evals_sph, evecs_sph[0], evecs_sph[1],
                             evecs_sph[2], kt_sph), axis=0)
# Compute ground truth - since KT is spherical, appparent kurtosic coeficient
# for all gradient directions and mean kurtosis have to be equal to Kref_sph.
f = 0.5
Dg = f*Di + (1-f)*De
Kref_sphere = 3 * f * (1-f) * ((Di-De) / Dg) ** 2

# Simulation 3. Multi-voxel simulations - dataset of four voxels is simulated.
# Since the objective of this simulation is to see if procedures are able to
# work with multi-dimentional data all voxels contains the same crossing signal
# produced in simulation 1.

DWI = np.zeros((2, 2, 1, len(gtab_2s.bvals)))
DWI[0, 0, 0] = DWI[0, 1, 0] = DWI[1, 0, 0] = DWI[1, 1, 0] = signal_cross
multi_params = np.zeros((2, 2, 1, 27))
multi_params[0, 0, 0] = multi_params[0, 1, 0] = crossing_ref
multi_params[1, 0, 0] = multi_params[1, 1, 0] = crossing_ref


def test_positive_evals():
    # Tested evals
    L1 = np.array([[1e-3, 1e-3, 2e-3], [0, 1e-3, 0]])
    L2 = np.array([[3e-3, 0, 2e-3], [1e-3, 1e-3, 0]])
    L3 = np.array([[4e-3, 1e-4, 0], [0, 1e-3, 0]])
    # only the first voxels have all eigenvalues larger than zero, thus:
    expected_ind = np.array([[True, False, False], [False, True, False]],
                            dtype=bool)
    # test function _positive_evals
    ind = _positive_evals(L1, L2, L3)
    assert_array_equal(ind, expected_ind)


def test_split_dki_param():
    dkiM = dki.DiffusionKurtosisModel(gtab_2s, fit_method="OLS")
    dkiF = dkiM.fit(DWI)
    evals, evecs, kt = dki.split_dki_param(dkiF.model_params)

    assert_array_almost_equal(evals, dkiF.evals)
    assert_array_almost_equal(evecs, dkiF.evecs)
    assert_array_almost_equal(kt, dkiF.kt)


def test_dki_fits():
    """ DKI fits are tested on noise free crossing fiber simulates """

    # OLS fitting
    dkiM = dki.DiffusionKurtosisModel(gtab_2s, fit_method="OLS")
    dkiF = dkiM.fit(signal_cross)

    assert_array_almost_equal(dkiF.model_params, crossing_ref)

    # WLS fitting
    dki_wlsM = dki.DiffusionKurtosisModel(gtab_2s, fit_method="WLS")
    dki_wlsF = dki_wlsM.fit(signal_cross)

    assert_array_almost_equal(dki_wlsF.model_params, crossing_ref)

    # testing multi-voxels
    dkiF_multi = dkiM.fit(DWI)
    assert_array_almost_equal(dkiF_multi.model_params, multi_params)

    dkiF_multi = dki_wlsM.fit(DWI)
    assert_array_almost_equal(dkiF_multi.model_params, multi_params)


def test_apparent_kurtosis_coef():
    """ Apparent kurtosis coeficients are tested for a spherical kurtosis
    tensor """

    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    AKC = dki.apparent_kurtosis_coef(params_sph, sph)

    # check all direction
    for d in range(len(gtab.bvecs[gtab.bvals > 0])):
        assert_array_almost_equal(AKC[d], Kref_sphere)


def test_dki_predict():
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    pred = dkiM.predict(crossing_ref, S0=100)

    assert_array_almost_equal(pred, signal_cross)

    # just to check that it works with more than one voxel:
    pred_multi = dkiM.predict(multi_params, S0=100)
    assert_array_almost_equal(pred_multi, DWI)

    # check the function predict of the DiffusionKurtosisFit object
    dkiF = dkiM.fit(DWI)
    pred_multi = dkiF.predict(gtab_2s, S0=100)
    assert_array_almost_equal(pred_multi, DWI)

    dkiF = dkiM.fit(pred_multi)
    pred_from_fit = dkiF.predict(dkiM.gtab, S0=100)
    assert_array_almost_equal(pred_from_fit, DWI)


def test_carlson_rf():
    # Define inputs that we know the outputs from:
    # Carlson, B.C., 1994. Numerical computation of real or complex
    # elliptic integrals. arXiv:math/9409227 [math.CA]

    # Real values (test in 2D format)
    x = np.array([[1.0, 0.5], [2.0, 2.0]])
    y = np.array([[2.0, 1.0], [3.0, 3.0]])
    z = np.array([[0.0, 0.0], [4.0, 4.0]])

    # Defene reference outputs
    RF_ref = np.array([[1.3110287771461, 1.8540746773014],
                       [0.58408284167715, 0.58408284167715]])

    # Compute integrals
    RF = carlson_rf(x, y, z)

    # Compare
    assert_array_almost_equal(RF, RF_ref)

    # Complex values
    x = np.array([1j, 1j - 1, 1j, 1j - 1])
    y = np.array([-1j, 1j, -1j, 1j])
    z = np.array([0.0, 0.0, 2, 1 - 1j])

    # Defene reference outputs
    RF_ref = np.array([1.8540746773014, 0.79612586584234 - 1.2138566698365j,
                       1.0441445654064, 0.93912050218619 - 0.53296252018635j])
    # Compute integrals
    RF = carlson_rf(x, y, z, errtol=3e-5)

    # Compare
    assert_array_almost_equal(RF, RF_ref)


def test_carlson_rd():

    # Define inputs that we know the outputs from:
    # Carlson, B.C., 1994. Numerical computation of real or complex
    # elliptic integrals. arXiv:math/9409227 [math.CA]

    # Real values
    x = np.array([0.0, 2.0])
    y = np.array([2.0, 3.0])
    z = np.array([1.0, 4.0])

    # Defene reference outputs
    RD_ref = np.array([1.7972103521034, 0.16510527294261])

    # Compute integrals
    RD = carlson_rd(x, y, z, errtol=1e-5)

    # Compare
    assert_array_almost_equal(RD, RD_ref)

    # Complex values (testing in 2D format)
    x = np.array([[1j, 0.0], [0.0, -2 - 1j]])
    y = np.array([[-1j, 1j], [1j-1, -1j]])
    z = np.array([[2.0, -1j], [1j, -1 + 1j]])

    # Defene reference outputs
    RD_ref = np.array([[0.65933854154220, 1.2708196271910 + 2.7811120159521j],
                       [-1.8577235439239 - 0.96193450888839j,
                        1.8249027393704 - 1.2218475784827j]])
    # Compute integrals
    RD = carlson_rd(x, y, z, errtol=1e-5)

    # Compare
    assert_array_almost_equal(RD, RD_ref)


def test_Wrotate_single_fiber():

    # Rotate the kurtosis tensor of single fiber simulate to the diffusion
    # tensor diagonal and check that is equal to the kurtosis tensor of the
    # same single fiber simulated directly to the x-axis

    # Define single fiber simulate
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    fie = 0.49
    frac = [fie*100, (1 - fie)*100]

    # simulate single fiber not aligned to the x-axis
    theta = random.uniform(0, 180)
    phi = random.uniform(0, 320)
    angles = [(theta, phi), (theta, phi)]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    evals, evecs = decompose_tensor(from_lower_triangular(dt))

    kt_rotated = dki.Wrotate(kt, evecs)
    # Now coordinate system has the DT diagonal aligned to the x-axis

    # Reference simulation in which DT diagonal is directly aligned to the
    # x-axis
    angles = (90, 0), (90, 0)
    signal, dt_ref, kt_ref = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                              fractions=frac, snr=None)

    assert_array_almost_equal(kt_rotated, kt_ref)


def test_Wrotate_crossing_fibers():
    # Test 2 - simulate crossing fibers intersecting at 70 degrees.
    # In this case, diffusion tensor principal eigenvector will be aligned in
    # the middle of the crossing fibers. Thus, after rotating the kurtosis
    # tensor, this will be equal to a kurtosis tensor simulate of crossing
    # fibers both deviating 35 degrees from the x-axis. Moreover, we know that
    # crossing fibers will be aligned to the x-y plane, because the smaller
    # diffusion eigenvalue, perpendicular to both crossings fibers, will be
    # aligned to the z-axis.

    # Simulate the crossing fiber
    angles = [(90, 30), (90, 30), (20, 30), (20, 30)]
    fie = 0.49
    frac = [fie*50, (1-fie) * 50, fie*50, (1-fie) * 50]
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                       [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])

    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    evals, evecs = decompose_tensor(from_lower_triangular(dt))

    kt_rotated = dki.Wrotate(kt, evecs)
    # Now coordinate system has diffusion tensor diagonal aligned to the x-axis

    # Simulate the reference kurtosis tensor
    angles = [(90, 35), (90, 35), (90, -35), (90, -35)]

    signal, dt, kt_ref = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                          fractions=frac, snr=None)

    # Compare rotated with the reference
    assert_array_almost_equal(kt_rotated, kt_ref)


def test_Wcons():

    # Construct the 4D kurtosis tensor manualy from the crossing fiber kt
    # simulate
    Wfit = np.zeros([3, 3, 3, 3])

    # Wxxxx
    Wfit[0, 0, 0, 0] = kt_cross[0]

    # Wyyyy
    Wfit[1, 1, 1, 1] = kt_cross[1]

    # Wzzzz
    Wfit[2, 2, 2, 2] = kt_cross[2]

    # Wxxxy
    Wfit[0, 0, 0, 1] = Wfit[0, 0, 1, 0] = Wfit[0, 1, 0, 0] = kt_cross[3]
    Wfit[1, 0, 0, 0] = kt_cross[3]

    # Wxxxz
    Wfit[0, 0, 0, 2] = Wfit[0, 0, 2, 0] = Wfit[0, 2, 0, 0] = kt_cross[4]
    Wfit[2, 0, 0, 0] = kt_cross[4]

    # Wxyyy
    Wfit[0, 1, 1, 1] = Wfit[1, 0, 1, 1] = Wfit[1, 1, 1, 0] = kt_cross[5]
    Wfit[1, 1, 0, 1] = kt_cross[5]

    # Wxxxz
    Wfit[1, 1, 1, 2] = Wfit[1, 2, 1, 1] = Wfit[2, 1, 1, 1] = kt_cross[6]
    Wfit[1, 1, 2, 1] = kt_cross[6]

    # Wxzzz
    Wfit[0, 2, 2, 2] = Wfit[2, 2, 2, 0] = Wfit[2, 0, 2, 2] = kt_cross[7]
    Wfit[2, 2, 0, 2] = kt_cross[7]

    # Wyzzz
    Wfit[1, 2, 2, 2] = Wfit[2, 2, 2, 1] = Wfit[2, 1, 2, 2] = kt_cross[8]
    Wfit[2, 2, 1, 2] = kt_cross[8]

    # Wxxyy
    Wfit[0, 0, 1, 1] = Wfit[0, 1, 0, 1] = Wfit[0, 1, 1, 0] = kt_cross[9]
    Wfit[1, 0, 0, 1] = Wfit[1, 0, 1, 0] = Wfit[1, 1, 0, 0] = kt_cross[9]

    # Wxxzz
    Wfit[0, 0, 2, 2] = Wfit[0, 2, 0, 2] = Wfit[0, 2, 2, 0] = kt_cross[10]
    Wfit[2, 0, 0, 2] = Wfit[2, 0, 2, 0] = Wfit[2, 2, 0, 0] = kt_cross[10]

    # Wyyzz
    Wfit[1, 1, 2, 2] = Wfit[1, 2, 1, 2] = Wfit[1, 2, 2, 1] = kt_cross[11]
    Wfit[2, 1, 1, 2] = Wfit[2, 2, 1, 1] = Wfit[2, 1, 2, 1] = kt_cross[11]

    # Wxxyz
    Wfit[0, 0, 1, 2] = Wfit[0, 0, 2, 1] = Wfit[0, 1, 0, 2] = kt_cross[12]
    Wfit[0, 1, 2, 0] = Wfit[0, 2, 0, 1] = Wfit[0, 2, 1, 0] = kt_cross[12]
    Wfit[1, 0, 0, 2] = Wfit[1, 0, 2, 0] = Wfit[1, 2, 0, 0] = kt_cross[12]
    Wfit[2, 0, 0, 1] = Wfit[2, 0, 1, 0] = Wfit[2, 1, 0, 0] = kt_cross[12]

    # Wxyyz
    Wfit[0, 1, 1, 2] = Wfit[0, 1, 2, 1] = Wfit[0, 2, 1, 1] = kt_cross[13]
    Wfit[1, 0, 1, 2] = Wfit[1, 1, 0, 2] = Wfit[1, 1, 2, 0] = kt_cross[13]
    Wfit[1, 2, 0, 1] = Wfit[1, 2, 1, 0] = Wfit[2, 0, 1, 1] = kt_cross[13]
    Wfit[2, 1, 0, 1] = Wfit[2, 1, 1, 0] = Wfit[1, 0, 2, 1] = kt_cross[13]

    # Wxyzz
    Wfit[0, 1, 2, 2] = Wfit[0, 2, 1, 2] = Wfit[0, 2, 2, 1] = kt_cross[14]
    Wfit[1, 0, 2, 2] = Wfit[1, 2, 0, 2] = Wfit[1, 2, 2, 0] = kt_cross[14]
    Wfit[2, 0, 1, 2] = Wfit[2, 0, 2, 1] = Wfit[2, 1, 0, 2] = kt_cross[14]
    Wfit[2, 1, 2, 0] = Wfit[2, 2, 0, 1] = Wfit[2, 2, 1, 0] = kt_cross[14]

    # Function to be tested
    W4D = dki.Wcons(kt_cross)

    Wfit = Wfit.reshape(-1)
    W4D = W4D.reshape(-1)

    assert_array_almost_equal(W4D, Wfit)


def test_spherical_dki_statistics():
    # tests if MK, AK and RK are equal to expected values of a spherical
    # kurtosis tensor

    # Define multi voxel spherical kurtosis simulations
    MParam = np.zeros((2, 2, 2, 27))
    MParam[0, 0, 0] = MParam[0, 0, 1] = MParam[0, 1, 0] = params_sph
    MParam[0, 1, 1] = MParam[1, 1, 0] = params_sph
    # MParam[1, 1, 1], MParam[1, 0, 0], and MParam[1, 0, 1] remains zero

    MRef = np.zeros((2, 2, 2))
    MRef[0, 0, 0] = MRef[0, 0, 1] = MRef[0, 1, 0] = Kref_sphere
    MRef[0, 1, 1] = MRef[1, 1, 0] = Kref_sphere
    MRef[1, 1, 1] = MRef[1, 0, 0] = MRef[1, 0, 1] = 0

    # Mean kurtosis analytical solution
    MK_multi = mean_kurtosis(MParam)
    assert_array_almost_equal(MK_multi, MRef)

    # radial kurtosis analytical solution
    RK_multi = radial_kurtosis(MParam)
    assert_array_almost_equal(RK_multi, MRef)

    # axial kurtosis analytical solution
    AK_multi = axial_kurtosis(MParam)
    assert_array_almost_equal(AK_multi, MRef)


def test_compare_MK_method():
    # tests if analytical solution of MK is equal to the average of directional
    # kurtosis sampled from a sphere

    # DKI Model fitting
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    dkiF = dkiM.fit(signal_cross)

    # MK analytical solution
    MK_as = dkiF.mk()

    # MK numerical method
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    MK_nm = np.mean(dki.apparent_kurtosis_coef(dkiF.model_params, sph),
                    axis=-1)

    assert_array_almost_equal(MK_as, MK_nm, decimal=1)


def test_single_voxel_DKI_stats():
    # tests if AK and RK are equal to expected values for a single fiber
    # simulate randomly oriented
    ADi = 0.00099
    ADe = 0.00226
    RDi = 0
    RDe = 0.00087
    # Reference values
    AD = fie*ADi + (1-fie)*ADe
    AK = 3 * fie * (1-fie) * ((ADi-ADe) / AD) ** 2
    RD = fie*RDi + (1-fie)*RDe
    RK = 3 * fie * (1-fie) * ((RDi-RDe) / RD) ** 2
    ref_vals = np.array([AD, AK, RD, RK])

    # simulate fiber randomly oriented
    theta = random.uniform(0, 180)
    phi = random.uniform(0, 320)
    angles = [(theta, phi), (theta, phi)]
    mevals = np.array([[ADi, RDi, RDi], [ADe, RDe, RDe]])
    frac = [fie*100, (1-fie)*100]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, S0=100, angles=angles,
                                      fractions=frac, snr=None)
    evals, evecs = decompose_tensor(from_lower_triangular(dt))
    dki_par = np.concatenate((evals, evecs[0], evecs[1], evecs[2], kt), axis=0)

    # Estimates using dki functions
    ADe1 = dki.axial_diffusivity(evals)
    RDe1 = dki.radial_diffusivity(evals)
    AKe1 = axial_kurtosis(dki_par)
    RKe1 = radial_kurtosis(dki_par)
    e1_vals = np.array([ADe1, AKe1, RDe1, RKe1])
    assert_array_almost_equal(e1_vals, ref_vals)

    # Estimates using the kurtosis class object
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    dkiF = dkiM.fit(signal)
    e2_vals = np.array([dkiF.ad, dkiF.ak(), dkiF.rd, dkiF.rk()])
    assert_array_almost_equal(e2_vals, ref_vals)

    # test MK (note this test correspond to the MK singularity L2==L3)
    MK_as = dkiF.mk()
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    MK_nm = np.mean(dkiF.akc(sph))

    assert_array_almost_equal(MK_as, MK_nm, decimal=1)


def test_compare_RK_methods():
    # tests if analytical solution of RK is equal to the perpendicular kurtosis
    # relative to the first diffusion axis

    # DKI Model fitting
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    dkiF = dkiM.fit(signal_cross)

    # MK analytical solution
    RK_as = dkiF.rk()

    # MK numerical method
    evecs = dkiF.evecs
    p_dir = perpendicular_directions(evecs[:, 0], num=30, half=True)
    ver = Sphere(xyz=p_dir)
    RK_nm = np.mean(dki.apparent_kurtosis_coef(dkiF.model_params, ver),
                    axis=-1)

    assert_array_almost_equal(RK_as, RK_nm)


def test_MK_singularities():
    # To test MK in case that analytical solution was a singularity not covered
    # by other tests

    dkiM = dki.DiffusionKurtosisModel(gtab_2s)

    # test singularity L1 == L2 - this is the case of a prolate diffusion
    # tensor for crossing fibers at 90 degrees
    angles_all = np.array([[(90, 0), (90, 0), (0, 0), (0, 0)],
                           [(89.9, 0), (89.9, 0), (0, 0), (0, 0)]])
    for angles_90 in angles_all:
        s_90, dt_90, kt_90 = multi_tensor_dki(gtab_2s, mevals_cross, S0=100,
                                              angles=angles_90,
                                              fractions=frac_cross, snr=None)
        dkiF = dkiM.fit(s_90)
        MK = dkiF.mk()

        sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])

        MK_nm = np.mean(dkiF.akc(sph))

        assert_almost_equal(MK, MK_nm, decimal=2)

        # test singularity L1 == L3 and L1 != L2
        # since L1 is defined as the larger eigenvalue and L3 the smallest
        # eigenvalue, this singularity teoretically will never be called,
        # because for L1 == L3, L2 have also to be  = L1 and L2.
        # Nevertheless, I decided to include this test since this singularity
        # is revelant for cases that eigenvalues are not ordered

        # artificially revert the eigenvalue and eigenvector order
        dki_params = dkiF.model_params.copy()
        dki_params[1] = dkiF.model_params[2]
        dki_params[2] = dkiF.model_params[1]
        dki_params[4] = dkiF.model_params[5]
        dki_params[5] = dkiF.model_params[4]
        dki_params[7] = dkiF.model_params[8]
        dki_params[8] = dkiF.model_params[7]
        dki_params[10] = dkiF.model_params[11]
        dki_params[11] = dkiF.model_params[10]

        MK = dki.mean_kurtosis(dki_params)
        MK_nm = np.mean(dki.apparent_kurtosis_coef(dki_params, sph))

        assert_almost_equal(MK, MK_nm, decimal=2)


def test_dki_errors():

    # first error of DKI module is if a unknown fit method is given
    assert_raises(ValueError, dki.DiffusionKurtosisModel, gtab_2s,
                  fit_method="JOANA")

    # second error of DKI module is if a min_signal is defined as negative
    assert_raises(ValueError, dki.DiffusionKurtosisModel, gtab_2s,
                  min_signal=-1)
    # try case with correct min_signal
    dkiM = dki.DiffusionKurtosisModel(gtab_2s, min_signal=1)
    dkiF = dkiM.fit(DWI)
    assert_array_almost_equal(dkiF.model_params, multi_params)

    # third error is if a given mask do not have same shape as data
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)

    # test a correct mask
    dkiF = dkiM.fit(DWI)
    mask_correct = dkiF.fa > 0
    mask_correct[1, 1] = False
    multi_params[1, 1] = np.zeros(27)
    mask_not_correct = np.array([[True, True, False], [True, False, False]])
    dkiF = dkiM.fit(DWI, mask=mask_correct)
    assert_array_almost_equal(dkiF.model_params, multi_params)
    # test a incorrect mask
    assert_raises(ValueError, dkiM.fit, DWI, mask=mask_not_correct)
