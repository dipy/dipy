""" Testing DKI """

from __future__ import division, print_function, absolute_import

import numpy as np

from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)

from dipy.sims.voxel import multi_tensor_dki

import dipy.reconst.dti as dti

import dipy.reconst.dki as dki

from dipy.reconst.dki import (mean_kurtosis, carlson_rf,  split_dki_param)

from dipy.io.gradients import read_bvals_bvecs

from dipy.core.gradients import gradient_table

from dipy.data import get_data

from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)

from dipy.core.sphere import Sphere


fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

"""Simulation 1 - Two crossing fibers"""

# 2 shells for techniques that requires multishell data
bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
gtab_2s = gradient_table(bvals_2s, bvecs_2s)

# Signals of two crossing fibers are simulated
mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                   [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
angles = [(80, 10), (80, 10), (20, 30), (20, 30)]
fie = 0.49
frac = [fie*50, (1-fie) * 50, fie*50, (1-fie) * 50]

# Noise free simulates
signal_crossing, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                           fractions=frac, snr=None)

evals, evecs = decompose_tensor(from_lower_triangular(dt))
crossing_ref = np.concatenate((evals, evecs[0], evecs[1], evecs[2], kt),
                              axis=0)

""" Simulation 2 - Spherical kurtosis tensor. this is biological implaussible
for white matter, however we know the ground truth properties of these kind of
simulations"""

# simulate a spherical kurtosis tensor
Di = 0.00099
De = 0.00226
mevals_sph = np.array([[Di, Di, Di], [De, De, De]])
frac_sph = [50, 50]
signal_spherical, dt_sph, kt_sph = multi_tensor_dki(gtab_2s, mevals_sph,
                                                    fractions=frac_sph,
                                                    snr=None)

evals, evecs = decompose_tensor(from_lower_triangular(dt_sph))
spherical_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                   kt_sph), axis=0)

# Since KT is spherical, in all directions AKC and MK should be
f = 0.5
Dg = f*Di + (1-f)*De
Kref_sphere = 3 * f * (1-f) * ((Di-De) / Dg) ** 2

""" Simulation 3 - Multi-voxel simulations - 4 voxels are simulated """
DWI = np.zeros((2, 2, 1, len(gtab_2s.bvals)))
DWI[0, 0, 0] = DWI[0, 1, 0] = DWI[1, 0, 0] = DWI[1, 1, 0] = signal_crossing

multi_params = np.zeros((2, 2, 1, 27))
multi_params[0, 0, 0] = multi_params[0, 1, 0] = crossing_ref
multi_params[1, 0, 0] = multi_params[1, 1, 0] = crossing_ref


def test_dki_fits():
    """DKI fits are tested on noise free simulates"""

    # OLS fitting
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    dkiF = dkiM.fit(signal_crossing)

    assert_array_almost_equal(dkiF.model_params, crossing_ref)

    # WLS fitting
    dki_wlsM = dki.DiffusionKurtosisModel(gtab_2s, fit_method="WLS_DKI")
    dki_wlsF = dki_wlsM.fit(signal_crossing)

    assert_array_almost_equal(dki_wlsF.model_params, crossing_ref)

    # WLS fitting addaption of Maurizios WLS implementation
    dki_params0 = wls_fit_dki(dki_wlsM.design_matrix, signal_crossing)
    out_shape = signal_crossing.shape[:-1] + (-1, )
    dki_params = dki_params0.reshape(out_shape)

    assert_array_almost_equal(dki_params, crossing_ref)

    # testing multi-voxels
    dkiF_multi = dkiM.fit(DWI)
    assert_array_almost_equal(dkiF_multi.model_params, multi_params)
    
    dkiF_multi = dki_wlsM.fit(DWI)
    assert_array_almost_equal(dkiF_multi.model_params, multi_params)


def test_apparent_kurtosis_coef():
    # Run apparent_kurtosis_coef function
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    AKC = dki.apparent_kurtosis_coef(spherical_params, sph)

    # check all direction
    for d in range(len(gtab.bvecs[gtab.bvals > 0])):
        assert_array_almost_equal(AKC[d], Kref_sphere)


def test_Wrotate_single_fiber():

    # Rotate the kurtosis tensor of single fiber simulate to the diffusion
    # tensor diagonal and check that is equal to the kurtosis tensor of the
    # same single fiber simulated directly to the x-axis

    # Define single fiber simulate 
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    fie = 0.49
    frac = [fie*100, (1 - fie)*100]

    # simulate single fiber not aligned to the x-axis
    angles = [(45, 0), (45, 0)]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    evals, evecs = decompose_tensor(from_lower_triangular(dt))

    kt_rotated = dki.Wrotate(kt, evecs)  # Now coordinate system has diffusion
                                         # tensor diagonal aligned with the
                                         # x-axis

    # Reference simulation which is simulated directly aligned to the x-axis
    angles = (90, 0), (90, 0)
    signal, dt_ref, kt_ref = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                              fractions=frac, snr=None)

    assert_array_almost_equal(kt_rotated, kt_ref)


def test_Wrotate_crossing_fibers():
    
    # Test 2 - simulate crossing fibers intersecting at 70 degrees.
    # In this case diffusion tensor principal eigenvector will be aligned in the
    # middle of the crossing fibers. Thus haver rotating the kurtosis tensor,
    # this will be equal to a kurtosis tensor simulate of crossing fibers both
    # deviating 35 degrees from the x-axis. Crossing fibers will be aligned to
    # the x-y plane because smaller eigenvalue prependicular to crossings
    # is aligned to the z-axis.
    
    # Simulate the crossing fiber
    angles = [(90, 30), (90, 30), (20, 30), (20, 30)]
    fie = 0.49
    frac = [fie*50, (1-fie) * 50, fie*50, (1-fie) * 50]
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                       [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])

    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    evals, evecs = decompose_tensor(from_lower_triangular(dt))

    kt_rotated = dki.Wrotate(kt, evecs)  # Now coordinate system has diffusion
                                         # tensor diagonal aligned with the
                                         # x-axis

    # Simulate the reference kurtosis tensor
    angles = [(90, 35), (90, 35), (90, -35), (90, -35)]

    signal, dt, kt_ref = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                          fractions=frac, snr=None)

    assert_array_almost_equal(kt_rotated, kt_ref)


def test_Wcons():

    # Construct the 4D kurtosis tensor manualy from the crossing fiber kt
    # simulate
    Wfit = np.zeros([3, 3, 3, 3])

    # Wxxxx
    Wfit[0, 0, 0, 0] = kt[0]

    # Wyyyy
    Wfit[1, 1, 1, 1] = kt[1]

    # Wzzzz
    Wfit[2, 2, 2, 2] = kt[2]

    # Wxxxy
    Wfit[0, 0, 0, 1] = Wfit[0, 0, 1, 0] = Wfit[0, 1, 0, 0] = kt[3]
    Wfit[1, 0, 0, 0] = kt[3]

    # Wxxxz
    Wfit[0, 0, 0, 2] = Wfit[0, 0, 2, 0] = Wfit[0, 2, 0, 0] = kt[4]
    Wfit[2, 0, 0, 0] = kt[4]

    # Wxyyy
    Wfit[0, 1, 1, 1] = Wfit[1, 0, 1, 1] = Wfit[1, 1, 1, 0] = kt[5]
    Wfit[1, 1, 0, 1] = kt[5]

    # Wxxxz
    Wfit[1, 1, 1, 2] = Wfit[1, 2, 1, 1] = Wfit[2, 1, 1, 1] = kt[6]
    Wfit[1, 1, 2, 1] = kt[6]    

    # Wxzzz
    Wfit[0, 2, 2, 2] = Wfit[2, 2, 2, 0] = Wfit[2, 0, 2, 2] = kt[7]
    Wfit[2, 2, 0, 2] = kt[7]

    # Wyzzz
    Wfit[1, 2, 2, 2] = Wfit[2, 2, 2, 1] = Wfit[2, 1, 2, 2] = kt[8]
    Wfit[2, 2, 1, 2] = kt[8]

    # Wxxyy
    Wfit[0, 0, 1, 1] = Wfit[0, 1, 0, 1] = Wfit[0, 1, 1, 0] = kt[9]
    Wfit[1, 0, 0, 1] = Wfit[1, 0, 1, 0] = Wfit[1, 1, 0, 0] = kt[9]

    # Wxxzz
    Wfit[0, 0, 2, 2] = Wfit[0, 2, 0, 2] = Wfit[0, 2, 2, 0] = kt[10]
    Wfit[2, 0, 0, 2] = Wfit[2, 0, 2, 0] = Wfit[2, 2, 0, 0] = kt[10]

    # Wyyzz
    Wfit[1, 1, 2, 2] = Wfit[1, 2, 1, 2] = Wfit[1, 2, 2, 1] = kt[11]
    Wfit[2, 1, 1, 2] = Wfit[2, 2, 1, 1] = Wfit[2, 1, 2, 1] = kt[11]

    # Wxxyz
    Wfit[0, 0, 1, 2] = Wfit[0, 0, 2, 1] = Wfit[0, 1, 0, 2] = kt[12]
    Wfit[0, 1, 2, 0] = Wfit[0, 2, 0, 1] = Wfit[0, 2, 1, 0] = kt[12]
    Wfit[1, 0, 0, 2] = Wfit[1, 0, 2, 0] = Wfit[1, 2, 0, 0] = kt[12]
    Wfit[2, 0, 0, 1] = Wfit[2, 0, 1, 0] = Wfit[2, 1, 0, 0] = kt[12]

    # Wxyyz
    Wfit[0, 1, 1, 2] = Wfit[0, 1, 2, 1] = Wfit[0, 2, 1, 1] = kt[13]
    Wfit[1, 0, 1, 2] = Wfit[1, 1, 0, 2] = Wfit[1, 1, 2, 0] = kt[13]
    Wfit[1, 2, 0, 1] = Wfit[1, 2, 1, 0] = Wfit[2, 0, 1, 1] = kt[13]
    Wfit[2, 1, 0, 1] = Wfit[2, 1, 1, 0] = Wfit[1, 0, 2, 1] = kt[13]

    # Wxyzz
    Wfit[0, 1, 2, 2] = Wfit[0, 2, 1, 2] = Wfit[0, 2, 2, 1] = kt[14]
    Wfit[1, 0, 2, 2] = Wfit[1, 2, 0, 2] = Wfit[1, 2, 2, 0] = kt[14]
    Wfit[2, 0, 1, 2] = Wfit[2, 0, 2, 1] = Wfit[2, 1, 0, 2] = kt[14]
    Wfit[2, 1, 2, 0] = Wfit[2, 2, 0, 1] = Wfit[2, 2, 1, 0] = kt[14]

    # Function to be tested
    Wcons = dki.Wcons(kt)

    Wfit = Wfit.reshape(-1)
    Wcons = Wcons.reshape(-1)

    assert_array_almost_equal(Wcons, Wfit)


def test_MK():
    """ tests MK solutions are equal to an expected values for a spherical
    kurtosis tensor"""

    # OLS fitting
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    dkiF = dkiM.fit(signal_spherical)

    # MK numerical method
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    MK_nm = mean_kurtosis(dkiF.model_params, sph)

    assert_almost_equal(Kref_sphere, MK_nm)
    
    # MK analytical solution
    MK_as = mean_kurtosis(dkiF.model_params)

    assert_almost_equal(Kref_sphere, MK_as)
    
    # multi spherical simulations
    MParam = np.zeros((2, 2, 2, 27))
    MParam[0, 0, 0] = MParam[0, 0, 1] = MParam[0, 1, 0] = spherical_params
    MParam[0, 1, 1] = MParam[1, 1, 0] = spherical_params
    # MParam[1, 1, 1], MParam[1, 0, 0], and MParam[1, 0, 1] remains zero
    MRef = np.zeros((2, 2, 2))
    MRef[0, 0, 0] = MRef[0, 0, 1] = MRef[0, 1, 0] = Kref_sphere
    MRef[0, 1, 1] = MRef[1, 1, 0] = Kref_sphere
    MRef[1, 1, 1] = MRef[1, 0, 0] = MRef[1, 0, 1] = float('nan')

    MK_multi = mean_kurtosis(MParam)
    assert_array_almost_equal(MK_multi, MRef)

    MK_multi = mean_kurtosis(MParam, sph)
    assert_array_almost_equal(MK_multi, MRef)


def test_compare_MK_method():
    """ tests if analytical solution of MK is equal to the exact solution"""

    # OLS fitting
    dkiM = dki.DiffusionKurtosisModel(gtab_2s)
    dkiF = dkiM.fit(signal_crossing)

    # MK analytical solution
    MK_as = dkiF.mk

    # MK numerical method
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    MK_nm = mean_kurtosis(dkiF.model_params, sph)

    assert_array_almost_equal(MK_as, MK_nm, decimal=1)


def test_carlson_rf():
    
    # Define inputs that we know the outputs from:
    # Carlson, B.C., 1994. Numerical computation of real or complex
    # elliptic integrals. arXiv:math/9409227 [math.CA]
    
    # Real values
    x = np.array([1.0, 0.5, 2.0])
    y = np.array([2.0, 1.0, 3.0])
    z = np.array([0.0, 0.0, 4.0])
    
    # Defene reference outputs
    RF_ref = np.array([1.3110287771461, 1.8540746773014, 0.58408284167715])
    
    # Compute integrals
    RF =  carlson_rf(x, y, z)

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
    RF =  carlson_rf(x, y, z, errtol=3e-5)

    # Compare
    assert_array_almost_equal(RF, RF_ref)


def wls_fit_dki(design_matrix, data, min_signal=1):
    r"""
    Adaption of the WLS fit implemented by Maurizio with faster all voxel
    lopping, with new output format (all important KT elements saved).
    """

    tol = 1e-6
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    # dki_params = np.empty((len(data_flat), 6, 3))
    # new line:
    dki_params = np.empty((len(data_flat), 27))
    min_diffusivity = tol / -design_matrix.min()

    ols_fit = _ols_fit_matrix(design_matrix)

    # for param, sig in zip(dki_params, data_flat):
    #     param[0], param[1:4], param[4], param[5] = _wls_iter(ols_fit,
    #     design_matrix, sig, min_signal, min_diffusivity)
    # new line:
    for vox in range(len(data_flat)):
        dki_params[vox] = _wls_iter(ols_fit, design_matrix, data_flat[vox],
                                    min_signal, min_diffusivity)

    # dki_params.shape=data.shape[:-1]+(18,)
    # dki_params=dki_params
    return dki_params


def _ols_fit_matrix(design_matrix):
    """
    (implemented by Maurizio)
    Helper function to calculate the ordinary least squares (OLS)
    fit as a matrix multiplication. Mainly used to calculate WLS weights. Can
    be used to calculate regression coefficients in OLS but not recommended.

    See Also:
    ---------
    wls_fit_tensor, ols_fit_tensor

    Example:
    --------
    ols_fit = _ols_fit_matrix(design_mat)
    ols_data = np.dot(ols_fit, data)
    """

    U, S, V = np.linalg.svd(design_matrix, False)
    return np.dot(U, U.T)


def _wls_iter(ols_fit, design_matrix, sig, min_signal, min_diffusivity):
    ''' Helper function used by wls_fit_tensor.
    '''
    sig = np.maximum(sig, min_signal)  # throw out zero signals
    log_s = np.log(sig)
    w = np.exp(np.dot(ols_fit, log_s))
    result = np.dot(np.linalg.pinv(design_matrix * w[:, None]), w * log_s)
    D = result[:6]
    # tensor=from_lower_triangular(D)
    # new line
    evals, evecs = decompose_tensor(from_lower_triangular(D),
                                    min_diffusivity=min_diffusivity)

    # MeanD_square=((tensor[0,0]+tensor[1,1]+tensor[2,2])/3.)**2
    # new_line:
    MeanD_square = (evals.mean(0))**2
    K_tensor_elements = result[6:21] / MeanD_square

    # new line:
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 K_tensor_elements), axis=0)

    out_shape = sig.shape[:-1] + (-1, )
    dki_params = dki_params.reshape(out_shape)

    # return decompose_tensors(tensor, K_tensor_elements,
    #                          min_diffusivity=min_diffusivity)
    # line line:
    return dki_params
