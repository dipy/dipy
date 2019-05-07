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


def gradient_descent(data, gtab, zooms, maxiter, dt, mask, metric='affine',
                     alpha1=1, alpha2=1, beta=1, Diso=3, lmin=0.01, lmax=2.5):
    """
    Perfoms the Beltrami minimization algorithm to estimate the Free Water
    fraction from single-shell diffusion data.

    Parameters
    ----------
    data : (X, Y, Z, K) ndarray
        Raw diffusion data
    gtab : gradients table class instance
    zooms : (3) ndarray
        Voxel dimensions (dx, dy, dz)
    maxiter : int
        Maximum number of allowed iterations
    dt : float
        Learning rate/step
    mask : (X, Y, Z) boolean array
        Boolean mask that marks indices of the data that
        should be processed
    metric : string
        Type of metric tensor used for smoothing the diffusion manifold:
        'affine' or 'euclidean'
    alpha1 : float
        Weight of the Fidelity term
    alpha2 : float
        Weight of the Beltrami/Civita terms
    beta : float
        Ratio that controls how isotropic is the regularization of thr manifold
    Diso : float
        Diffusion constant of isotropic Free Water
    lmin : float
        Minimum expected diffusion constant in tissue
    lmax : float
        Maximum expected difusion constant in tissue

    Returns
    -------   
    fmin : (X, Y, Z) ndarray
        Lower limit of the allowed tissue volume fraction (1 - fw).
    fmax : (X, Y, Z) ndarray
        Upper limit of the allowed tissue volume fraction (1 - fw).
    f0 : (X, Y, Z) ndarray
        Initial guess of the tissue volume fraction (1 - fw0)
    D0 : (X, Y, Z, 6) ndarray
        Initial guess of the tiddue diffusion tensor in lower triangular order:
        Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
    fn : (X, Y, Z) ndarray
        Final estimate for the tissue volume fraction (1 - fw)
    Dn : (X, Y, Z, 6) ndarray
        Final estimate for the tissue diffusion tensor in lower triangular order:
        Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
    bad_g : (X, Y, Z) boolean array
        Marks the indices of the voxels where the determinant of the metric tensor g
        was found to be unstable during minimization, and where the Free Water
        estimated may be invalid.
    
    Notes
    -----
    1) Before constructing the gtab and passing it to this fucntion,
       the b-values should be converted to the units of
       milisecond/micrometer^-2. This is done for numerical reasons.

    2) Although this module is called Free Water estimation, the result
       returned in the form of tissue fraction: f = 1 - fw, this was done
       to maintain consistency with the equations of the original paper
       (put reference!!!)    

    3) The use of the affine metric over the euclidean takes more time to
       process, because more complex operations are involved
    """

    # preprocessing
    mask = mask.astype(bool)
    zooms = np.array(zooms) / np.min(zooms)
    S0, Ak, bvals, bvecs = get_atten(data, gtab)

    # initialization
    fmin, fmax, f0, D0, H = initialize(S0, Ak, bvals, bvecs, Diso, lmin, lmax)

    # masks for spatial derivatives
    nx, ny, nz = D0.shape[:-1]
    ind_fx = np.append(np.arange(1, nx), nx-1)
    ind_fy = np.append(np.arange(1, ny), ny-1)
    ind_fz = np.append(np.arange(1, nz), nz-1)
    ind_bx = np.append(0, np.arange(nx-1))
    ind_by = np.append(0, np.arange(ny-1))
    ind_bz = np.append(0, np.arange(nz-1))

    mask_fx = np.logical_and(mask[ind_fx, :, :], mask)
    mask_fy = np.logical_and(mask[:, ind_fy, :], mask)
    mask_fz = np.logical_and(mask[:, :, ind_fz], mask)

    mask_f = np.logical_and(mask_fx, mask_fy)
    np.logical_and(mask_f, mask_fz, out=mask_f)

    mask_bx = np.logical_and(mask[ind_bx, :, :], mask_f)
    mask_by = np.logical_and(mask[:, ind_by, :], mask_f)
    mask_bz = np.logical_and(mask[:, :, ind_bz], mask_f)

    # Iwasawa parameterization
    fn = np.copy(f0)
    Dn = np.copy(D0)
    if metric == 'affine' :
       Xn = np.zeros(Dn.shape)
       Xn[mask, :] = blt.x_manifold(Dn[mask, :])
    
    # allocating increment matrices
    dB = np.zeros(Dn.shape)
    dC = np.zeros(Dn.shape)
    df = np.zeros(f0.shape)
    dF = np.zeros(Dn.shape)


    for i in np.arange(maxiter):

        if i == maxiter // 2:
            alpha2 = 0
        
        if metric == 'affine':
            # compute beltrami and civita increments
            bad_g, g, dB[...], dC[...] = blt.beltrami_affine(Xn, mask, mask_fx, mask_fy,
                                                             mask_fz, mask_bx, mask_by,
                                                             mask_bz, zooms, beta)
            
            # compute fidelity and f increments
            df[...], dF[...] = blt.fidelity_affine(Ak, Xn, Dn, fn, g, bvals, H, mask, Diso)

            # update
            dB[bad_g, :] = 0  # no update where the metric g is unstable
            dC[bad_g, :] = 0
            df[bad_g] = 0
            dF[bad_g, :] = 0

            csf = df > 0.8 # no update for csf voxels
            dB[csf, :] = 0
            dC[csf, :] = 0
            df[csf] = 0
            dF[csf, :] = 0

            Xn = Xn + dt * (alpha1 * dF + alpha2 * (dB + dC))

            # compute new tensor
            Dn[mask, :] = blt.d_manifold(Xn[mask, :])
        
        elif metric == 'euclidean':
            bad_g, g, dB[...] = blt.beltrami_euclidean(Dn, mask, mask_fx, mask_fy,
                                                       mask_fz, mask_bx, mask_by,
                                                       mask_bz, zooms, beta)

            # compute fidelity and f increments
            df[...], dF[...] = blt.fidelity_euclidean(Ak, Dn, fn, g, bvals, H, mask, Diso)

            # update
            dB[bad_g, :] = 0  # no update where the metric g is unstable
            df[bad_g] = 0
            dF[bad_g, :] = 0

            csf = df > 0.8 # no update for csf voxels
            dB[csf, :] = 0
            df[csf] = 0
            dF[csf, :] = 0

            Dn = Dn + dt * (alpha1 * dF + alpha2 * dB) 

        # update tissue volume fraction
        fn = fn + dt * df

        # correct bad f values
        fn = np.clip(fn, fmin, fmax)

        # reset increments
        dB[...] = 0
        dC[...] = 0
        df[...] = 0
        dF[...] = 0

    Dn = Dn * 10**-3
    return (fmin, fmax, f0, D0, fn, Dn, bad_g)
