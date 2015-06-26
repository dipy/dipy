"""
Testing DKI
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)

from dipy.sims.voxel import multi_tensor_dki

import dipy.reconst.dti as dti

import dipy.reconst.dki as dki

from dipy.reconst.dki import mean_kurtosis

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
frac = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]

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
mevals = np.array([[Di, Di, Di], [De, De, De]])
frac = [50, 50]
signal_spherical, dt, kt = multi_tensor_dki(gtab_2s, mevals, fractions=frac,
                                            snr=None)
evals, evecs = decompose_tensor(from_lower_triangular(dt))
spherical_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2], kt),
                                   axis=0)

# Since KT is spherical, in all directions AKC and MK should be
f = 0.5
Dg = f*Di + (1-f)*De
Kref_sphere = 3 * f * (1-f) * ((Di-De) / Dg) ** 2


def test_dki_fits():
    """DKI fits are tested on noise free simulates"""

    # OLS fitting
    dkiM = dki.DKIModel(gtab_2s)
    dkiF = dkiM.fit(signal_crossing)

    assert_array_almost_equal(dkiF.model_params, crossing_ref)

    # WLS fitting
    dki_wlsM = dki.DKIModel(gtab_2s, fit_method="WLS_DKI")
    dki_wlsF = dki_wlsM.fit(signal_crossing)

    assert_array_almost_equal(dki_wlsF.model_params, crossing_ref)

    # WLS fitting addaption of Maurizios WLS implementation
    dki_params0 = wls_fit_dki(dki_wlsM.design_matrix, signal_crossing)
    out_shape = signal_crossing.shape[:-1] + (-1, )
    dki_params = dki_params0.reshape(out_shape)

    assert_array_almost_equal(dki_params, crossing_ref)


def test_apparent_kurtosis_coef():
    # Run apparent_kurtosis_coef function
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    AKC = dki.apparent_kurtosis_coef(spherical_params, sph)

    # check all direction
    for d in range(len(gtab.bvecs[gtab.bvals > 0])):
        assert_array_almost_equal(AKC[d], Kref_sphere)


def test_Wrotate():

    # Rotate the kurtosis tensor of single fiber simulate to the diffusion
    # tensor diagonal and check that is equal to the kurtosis tensor of the
    # same single fiber simulated directly to the x-axis

    # Define single voxel simulate 
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    frac = [fie*100, (1 - fie)*100]

    # simulate it not aligned to the x-axis
    angles = [(45, 0), (45, 0)]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    evals, evecs = decompose_tensor(from_lower_triangular(dt))

    kt_rotated = dki.Wrotate(kt, evecs)  # Now coordinate system has diffusion
                                         # tensor diagnol in x-axis

    # Reference simulate directly aligned to the x-axis
    angles = (90, 0), (90, 0)
    signal, dt_ref, kt_ref = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                              fractions=frac, snr=None)

    assert_array_almost_equal(kt_rotated, kt_ref)


def test_Wcons():

    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    Wfit = np.zeros([3, 3, 3, 3])

    Wfit[0,0,0,0] = kt[0]

    Wfit[1,1,1,1] = kt[1]

    Wfit[2,2,2,2] = kt[2]

    Wfit[0,0,0,1] = Wfit[0,0,1,0] = Wfit[0,1,0,0] = Wfit[1,0,0,0] = kt[3]

    Wfit[0,0,0,2] = Wfit[0,0,2,0] = Wfit[0,2,0,0] = Wfit[2,0,0,0] = kt[4]

    Wfit[0,1,1,1] = Wfit[1,0,1,1] = Wfit[1,1,1,0] = Wfit[1,1,0,1] = kt[5]

    Wfit[1,1,1,2] = Wfit[1,2,1,1] = Wfit[2,1,1,1] = Wfit[1,1,2,1] = kt[6]    

    Wfit[0,2,2,2] = Wfit[2,2,2,0] = Wfit[2,0,2,2] = Wfit[2,2,0,2] = kt[7]

    Wfit[1,2,2,2] = Wfit[2,2,2,1] = Wfit[2,1,2,2] = Wfit[2,2,1,2] = kt[8]

    Wfit[0,0,1,1] = Wfit[0,1,0,1] = Wfit[0,1,1,0] = Wfit[1,0,0,1] = kt[9]
    Wfit[1,0,1,0] = Wfit[1,1,0,0] = kt[9]

    Wfit[0,0,2,2] = Wfit[0,2,0,2] = Wfit[0,2,2,0] = Wfit[2,0,0,2] = kt[10]
    Wfit[2,0,2,0] = Wfit[2,2,0,0] = kt[10]

    Wfit[1,1,2,2] = Wfit[1,2,1,2] = Wfit[1,2,2,1] = Wfit[2,1,1,2] = kt[11]
    Wfit[2,2,1,1] = Wfit[2,1,2,1] = kt[11]

    Wfit[0,0,1,2] = Wfit[0,0,2,1] = Wfit[0,1,0,2] = Wfit[0,1,2,0] = kt[12]
    Wfit[0,2,0,1] = Wfit[0,2,1,0] = Wfit[1,0,0,2] = Wfit[1,0,2,0] = kt[12]
    Wfit[1,2,0,0] = Wfit[2,0,0,1] = Wfit[2,0,1,0] = Wfit[2,1,0,0] = kt[12]

    Wfit[0,1,1,2] = Wfit[0,1,2,1] = Wfit[0,2,1,1] = Wfit[1,0,1,2] = kt[13]
    Wfit[1,1,0,2] = Wfit[1,1,2,0] = Wfit[1,2,0,1] = Wfit[1,2,1,0] = kt[13]
    Wfit[2,0,1,1] = Wfit[2,1,0,1] = Wfit[2,1,1,0] = Wfit[1,0,2,1] = kt[13]

    Wfit[0,1,2,2] = Wfit[0,2,1,2] = Wfit[0,2,2,1] = Wfit[1,0,2,2] = kt[14]
    Wfit[1,2,0,2] = Wfit[1,2,2,0] = Wfit[2,0,1,2] = Wfit[2,0,2,1] = kt[14]
    Wfit[2,1,0,2] = Wfit[2,1,2,0] = Wfit[2,2,0,1] = Wfit[2,2,1,0] = kt[14]

    Wcons = dki.Wcons(kt)

    Wfit = Wfit.reshape(-1)
    Wcons = Wcons.reshape(-1)

    assert_array_almost_equal(Wcons, Wfit)


def test_MK():
    """ tests MK solutions are equal to an expected values for a spherical
    kurtosis tensor"""

    # OLS fitting
    dkiM = dki.DKIModel(gtab_2s)
    dkiF = dkiM.fit(signal_spherical)

    # MK numerical method
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    MK_nm = mean_kurtosis(dkiF.model_params, sph)

    assert_almost_equal(Kref_sphere, MK_nm)
    
    """
    # MK analytical solution
    MK_as = mean_kurtosis(dkiF.model_params)

    assert_almost_equal(Kref_sphere, MK_as)
    """


def test_compare_MK_method():
    """ tests if analytical solution of MK is equal to the exact solution"""
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    # OLS fitting
    dkiM = dki.DKIModel(gtab_2s)
    dkiF = dkiM.fit(signal_crossing)

    # MK analytical solution
    MK_as = dkiF.mk

    # MK numerical method
    sph = Sphere(xyz=gtab.bvecs[gtab.bvals > 0])
    MK_nm = mean_kurtosis(dkiF.model_params, sph)

    assert_almost_equal(MK_as, MK_nm, delta=0.1)


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
