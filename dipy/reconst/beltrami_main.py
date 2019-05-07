"""
Implements the main gradient descent function to estimate
the Free Water parameter from single-shell diffusion data.
"""

from __future__ import division
import numpy as np
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import design_matrix
import beltrami as blt  # importing the functions from beltrami.py


def get_atten(data, gtab):
    """
    Preprocessing, get S0 and Ak

    Parameters
    ----------
    data : (X, Y, Z, K) ndarray
        Diffusion data acquired for K directions.
    gtab : (K, 7)
        Gradients table class instance.
    
    Returns
    -------
    S0 : (X, Y, Z) ndarray
        Non diffusion-weighted volume (bval = 0).
    Ak : (X, Y, Z, K) ndarray
        Normalized attenuations (Ak = data / S0).
    bvals : (K) ndarray
        Vector containing the bvalues.
    bvecs : (K, 3) ndarray
        Normalized gradient directions.

    Notes
    -----
    If multiple S0 volumes are present, the mean volume is returned,
    the bvals and bvecs returned are cropped to exclude the S0 volumes.
    """

    ind = gtab.b0s_mask
    bvals = gtab.bvals[~ind]
    bvecs = gtab.bvecs[~ind, :]
    bval = bvals[0]

    # getting S0 and Sk
    ind = gtab.b0s_mask
    S0s = data[..., ind]
    S0 = np.mean(S0s, axis=-1)[..., np.newaxis]
    S0[S0 < 0.0001] = 0.0001
    Sk = data[..., ~ind]

    # getting Ak
    D_min = 0.01
    D_max = 5
    A_min = np.exp(-bval * D_max)
    A_max = np.exp(-bval * D_min)
    Ak = Sk / S0
    Ak = np.clip(Ak, A_min, A_max)

    return (S0, Ak, bvals, bvecs)


def initialize(S0, Ak, bvals, bvecs, Diso, lambda_min, lambda_max):
    """
    Initializes the diffusion tensor field and tissue volume fraction.

    Parameters
    ----------
    S0 : (X, Y, Z) ndarray
        Non-diffusion weighted volume (bval = 0).
    Ak : (X, Y, Z, K) ndarray
        Normalized attenuations.
    bvals : (K) ndarray
        Vector containig the bvalues, exluding the b0.
    bvecs : (K, 3) ndarray
        Gradient directions, excluding the direction for S0.
    Diso : float
        Diffusion constant of isotropic Free Water.
    lambda_min : float
        Minimum expected diffusion constant in tissue.
    lambda_max : float
        Maximum expected diffusion constant in tissua.

    Returns
    -------
    fmin : (X, Y, Z) ndarra
        Lower limit of the allowed tissue volume fraction (1 - fw).
    fmax : (X, Y, Z) ndarray
        Upper limit of the allowed tissue volume fraction (1 - fw).
    f0 : (X, Y, Z) ndarray
        Initial guess of the tissue volume fraction (1 - fw0)
    D0 : (X, Y, Z, 6) ndarray
        Initial guess of the diffusion tensor in lower triangular order:
        Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
    H : (6, K) ndarray
        Transposed design matrix.
    
    Notes
    -----
    1) The initial diffusion tensor field is estimated with standard DTI.
    2) The initial tissue volume fraction is estimated from the inital
       Mean Diffusivity map.
    3) The lower and upper limits (fmin and fmax) of the tissue volume fraction
       are computed from the initial eigenvalues of the diffusion tensor and
       expected tissue diffusivities lambda_min and lambda_max.
    
    Special thanks to Mr.Ofer Pasternak for clarifying some details for this
    initialization.

    """
    # getting new gtab
    bval = np.mean(bvals)
    # bvals = bval * np.ones(bvecs.shape[0])
    bvals = np.insert(bvals, 0, 0)
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    gtab = gradient_table(bvals, bvecs)
    # getting initial MD and evals
    x, y, z, k = Ak.shape
    data = np.zeros((x, y, z, k + 1))
    data[..., 0] = 1
    data[..., 1:] = Ak
    model = dti.TensorModel(gtab)
    fit = model.fit(data)
    model = dti.TensorModel(gtab)
    fit = model.fit(data)
    MD = fit.md
    evals = fit.evals
    # getiing fmin and fmax
    Aw = np.exp(-bval * Diso)
    Amin = np.exp(-bval * lambda_max)  # min expected attenuation in tissue
    Amax = np.exp(-bval * lambda_min)  # max expected attenuation in tissue
    fmin = (np.exp(-bval * evals[..., 2]) - Aw) / (Amax - Aw)
    fmax = (np.exp(-bval * evals[..., 0]) - Aw) / (Amin - Aw)
    fmin[fmin < 0] = 0.0001
    fmin[fmin > 1] = 1 - 0.0001
    fmax[fmax < 0] = 0.0001
    fmax[fmax > 1] = 1 - 0.0001
    # fmin[...] = 0
    # fmax[...] = 1
    # getting f0
    base_MD = 0.6 # theoretical value of MD in tissue, this might be tweaked
    f0 = (np.exp(-bval * MD) - Aw) / (np.exp(-bval * base_MD) - Aw)
    bad_f0s = np.logical_or(f0 < fmin, f0 > fmax)
    f0[bad_f0s] = (fmax[bad_f0s] + fmin[bad_f0s]) / 2
    # corrected tissue attenuation
    Cw = (1 - f0) * Aw
    At = (Ak - Cw[..., np.newaxis]) / f0[..., np.newaxis]
    # At = np.clip(At, Amin, Amax)
    # initializing new tensor D0
    data[..., 0] = 1
    data[..., 1:] = At
    model = dti.TensorModel(gtab, fit_method='OLS')
    fit = model.fit(data)
    # getting unique components of D0
    # qform = fit.quadratic_form
    # rows = [0, 1, 2, 0, 0, 1]
    # cols = [0, 1, 2, 1, 2, 2]
    # D0 = qform[..., rows, cols]
    D0 = fit.lower_triangular()

    H = design_matrix(gtab)
    H = H[1:, :-1]
    H = -1 * H.T

    return (fmin, fmax, f0, D0, H)
