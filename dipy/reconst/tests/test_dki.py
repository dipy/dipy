""" Testing DKI """

from __future__ import division, print_function, absolute_import

import numpy as np

from nose.tools import assert_almost_equal

from numpy.testing import assert_array_almost_equal

from dipy.sims.voxel import multi_tensor_dki

import dipy.reconst.dki as dki

from dipy.io.gradients import read_bvals_bvecs

from dipy.core.gradients import gradient_table

from dipy.data import get_data

from dipy.data import get_sphere

from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)

from dipy.core.sphere import Sphere


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

evals_cross, evecs = decompose_tensor(from_lower_triangular(dt_cross))
crossing_ref = np.concatenate((evals_cross, evecs[0], evecs[1], evecs[2],
                               kt_cross), axis=0)

# Simulation 2. Spherical kurtosis tensor.- for white matter, this can be a
# biological implaussible scenario, however this simulation is usefull for
# testing the estimation of directional apparent kurtosis and the mean
# kurtosis, since its directional and mean kurtosis ground truth are a constant
# which can be easly mathematicaly calculated

# simulate a spherical kurtosis tensor
Di = 0.00099
De = 0.00226
mevals_sph = np.array([[Di, Di, Di], [De, De, De]])
frac_sph = [50, 50]
signal_sph, dt_sph, kt_sph = multi_tensor_dki(gtab_2s, mevals_sph, S0=100,
                                              fractions=frac_sph,
                                              snr=None)
evals_sph, evecs = decompose_tensor(from_lower_triangular(dt_sph))
params_sph = np.concatenate((evals_sph, evecs[0], evecs[1], evecs[2], kt_sph),
                            axis=0)

# Compute ground truth. Since KT is spherical, appparent kurtosic coeficient
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


def test_diffusion_kurtosis_odf():
    # Comparison of our vectorized implementation of the DKI-ODF using the
    # symmetry of the diffusion kurtosis tensor with the implementation on the
    # non-vectorized format directly presented in Eq. 5 of the article:
    #     Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015). Exploring
    #     the 3D geometry of the diffusion kurtosis tensor - Impact on the
    #     development of robust tractography procedures and novel biomarkers,
    #     NeuroImage 111: 85-99

    # define parameters
    alpha = 4
    sphere = get_sphere('symmetric362')
    V = sphere.vertices

    # Compute the dki-odf using the helper function to process single voxel
    dipy_odf = dki.diffusion_kurtosis_odf(crossing_ref, sphere, alpha=alpha)

    # reference ODF for a single voxel simulate
    MD = (dt_cross[0] + dt_cross[2] + dt_cross[5]) / 3
    U = np.linalg.pinv(from_lower_triangular(dt_cross)) * MD
    W = dki.Wcons(kt_cross)
    ODFref = np.zeros(len(V))
    for i in range(len(V)):
        ODFref[i] = _dki_odf_non_vectorized(V[i], W, U, alpha)

    assert_array_almost_equal(dipy_odf, ODFref)

    # test function for multi-voxel data
    dipy_odf = dki.diffusion_kurtosis_odf(multi_params, sphere, alpha=alpha)
    multi_ODFref = np.zeros((2, 2, 1, len(V)))
    multi_ODFref[0, 0, 0] = multi_ODFref[0, 1, 0] = ODFref
    multi_ODFref[1, 0, 0] = multi_ODFref[1, 1, 0] = ODFref
    assert_array_almost_equal(dipy_odf, multi_ODFref)

    # test dki class
    dkimodel = dki.DiffusionKurtosisModel(gtab_2s)
    dkifit = dkimodel.fit(DWI)
    dipy_odf = dkifit.dki_odf(sphere, alpha=alpha)
    assert_array_almost_equal(dipy_odf, multi_ODFref)


def _dki_odf_non_vectorized(n, W, U, a):
    """ Helper function to test Dipy implementation of diffusion_kurtosis_odf.

    This function is analogous to dipy's helper function _dki_odf_core from
    module dipy.reconst.dki, however here function is implemented in the format
    presented in Eq.5 of the article:

        Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015). Exploring
        the 3D geometry of the diffusion kurtosis tensor - Impact on the
        development of robust tractography procedures and novel biomarkers,
        NeuroImage 111: 85-99

    To a detailed information of inputs see the helper function_dki_odf_core
    """
    # Compute elements of matrix V
    Un = np.dot(U, n)
    nUn = np.dot(n, Un)
    V00 = Un[0]**2 / nUn
    V11 = Un[1]**2 / nUn
    V22 = Un[2]**2 / nUn
    V01 = Un[0]*Un[1] / nUn
    V02 = Un[0]*Un[2] / nUn
    V12 = Un[1]*Un[2] / nUn
    V = from_lower_triangular(np.array([V00, V01, V11, V02, V12, V22]))

    # diffusion ODF
    ODFg = (1./nUn) ** ((a + 1.)/2.)

    # Compute the summatory term of reference Eq.5
    SW = 0
    xyz = [0, 1, 2]
    for i in xyz:
        for j in xyz:
           for k in xyz:
               for l in xyz:
                   SW = SW + W[i, j, k, l] * (3*U[i, j]*U[k, l] - \
                                              6*(a + 1)*U[i, j]*V[k, l] + \
                                              (a + 1)*(a + 3)*V[i, j]*V[k, l])

    # return the total ODF
    return ODFg * (1. + 1/24.*SW)


def test_dki_directions():
    # define parameters
    alpha = 4
    sphere = get_sphere('symmetric362')

    pam = dki.dki_directions(crossing_ref, sphere, alpha=alpha,
                             relative_peak_threshold=0.1,
                             min_separation_angle=20, mask=None,
                             return_odf=True, normalize_peaks=False, npeaks=3,
                             gtol=None)

    # Check if detected two fiber directions
    Ndetect_peaks = 0.0
    for ps in pam.peak_dirs: 
        v_norm = np.linalg.norm(ps)
        Ndetect_peaks = Ndetect_peaks + v_norm

    assert_almost_equal(Ndetect_peaks, 2.)
    
    # Check if convergence is working propertly
    pam = dki.dki_directions(crossing_ref, sphere, alpha=alpha,
                             relative_peak_threshold=0.1,
                             min_separation_angle=20, mask=None,
                             return_odf=True, normalize_peaks=False, npeaks=3)

    # Since two fiber have the same weight their values have to be equal.
    assert_almost_equal(pam.peak_values[0], pam.peak_values[0])
    
    # Check if dki fiber direction estimate is working on dki class instance
    dkimodel = dki.DiffusionKurtosisModel(gtab_2s)
    dkifit = dkimodel.fit(DWI)
    dirs = dkifit.dki_directions(sphere, alpha=alpha,
                                 relative_peak_threshold=0.1,
                                 min_separation_angle=20, mask=None)
    assert_almost_equal(dirs.peak_values[0, 0, 0, 0], pam.peak_values[0])
