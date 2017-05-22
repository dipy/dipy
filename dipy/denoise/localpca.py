import numpy as np
import scipy as sp
import scipy.linalg as sla
import scipy.special as sps
from scipy.interpolate import interp1d
from warnings import warn

from dipy.core.ndindex import ndindex

# Try to get the SVD through direct API to lapack:
try:
    from scipy.linalg.lapack import sgesvd as svd
    svd_args = [1, 0]
# If you have an older version of scipy, we fall back
# on the standard scipy SVD API:
except ImportError:
    from scipy.linalg import svd
    svd_args = [False]


def inv_eta(phi):
    """
    The inverse of eta, which corrects for Rician bias at the end of the
    denoising process (see eqs. 4,5,6 in [Manjon13]_).

    Parameters
    ----------
    phi : float
       signal to noise ratio

    Returns
    -------
    inv_eta : float
        The inverse of the eta function
    """
    phi_sq = phi ** 2
    inv_eta1 = np.log(np.sqrt(np.pi/2)) - phi_sq / 2
    inv_eta2 = 2 * np.log((((1 + (phi_sq / 2)) * sps.iv(0, phi_sq/4)) +
                          ((phi_sq / 2) * sps.iv(1, (phi_sq/4)))))
    return np.exp(inv_eta1 + inv_eta2)


def get_eta(start=0, stop=37.4, step=0.01):
    """
    Get's an eta function: an interpolated lookup table between SNR
    and bias corrected.
    """
    x = np.arange(start, stop, step)
    y = inv_eta(x)
    return interp1d(y, x)


def eta(phi, f=None):
    """
    A lookup table for correction of Rician bias after denoising (see eqs.
    4,5,6 in [Manjon13]_).

    Parameters
    ----------
    phi : float
    """
    if phi < 1.25:
        return 0
    else:
        return f(phi)


def localpca(arr, sigma, patch_radius=2, tau_factor=2.3, correct_bias=False):
    r"""Local PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.
    sigma : float or 3D array
        Standard deviation of the noise estimated from the data.
    patch_radius : int, optional
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 2 (denoise in blocks of 5x5x5 voxels).
    tau_factor : float, optional
        Thresholding of PCA eigenvalues is done by nulling out eigenvalues that
        are smaller than:

        .. math ::

                \tau = (\tau_{factor} \sigma)^2

        Default: 2.3, based on the results described in [Manjon13]_.

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data.

    References
    ----------
    .. [Manjon13] Manjon JV, Coupe P, Concha L, Buades A, Collins DL (2013)
                  Diffusion Weighted Image Denoising Using Overcomplete Local
                  PCA. PLoS ONE 8(9): e73021.
                  https://doi.org/10.1371/journal.pone.0073021
    """
    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.",
                         arr.shape)

    patch_size = 2 * patch_radius + 1

    if patch_size ** 3 < arr.shape[-1]:
        e_s = "You asked for PCA denoising with a "
        e_s += "patch_radius of {0} ".format(patch_radius)
        e_s += "for data with {0} directions. ".format(arr.shape[-1])
        e_s += "This would result in an ill-conditioned PCA matrix. "
        e_s += "Please increase the patch_radius."
        raise ValueError(e_s)

    if isinstance(sigma, np.ndarray):
        if not sigma.shape == arr.shape[:-1]:
            e_s = "You provided a sigma array with a shape"
            e_s += "{0} for data with".format(sigma.shape)
            e_s += "shape {0}. Please provide a sigma array".format(arr.shape)
            e_s += " that matches the spatial dimensions of the data."
            raise ValueError(e_s)

    tau = np.ones(arr.shape[:-1]) * ((tau_factor * sigma) ** 2)
    # declare arrays for theta and thetax
    theta = np.zeros(arr.shape, dtype=np.float64)
    thetax = np.zeros(arr.shape, dtype=np.float64)

    # loop around and find the 3D patch for each direction at each pixel
    for k in range(patch_radius, arr.shape[2] - patch_radius):
        for j in range(patch_radius, arr.shape[1] - patch_radius):
            for i in range(patch_radius, arr.shape[0] - patch_radius):
                # Shorthand for indexing variables:
                ix1 = i - patch_radius
                ix2 = i + patch_radius + 1
                jx1 = j - patch_radius
                jx2 = j + patch_radius + 1
                kx1 = k - patch_radius
                kx2 = k + patch_radius + 1

                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(
                                patch_size ** 3, arr.shape[-1])
                # compute the mean and normalize
                M = np.mean(X, axis=0)
                X = X - M
                # PCA using an SVD
                U, S, Vt = svd(X, *svd_args)[:3]
                # Items in S are the eigenvalues, but in ascending order
                # We invert the order (=> descending), square and normalize
                # \lambda_i = s_i^2 / n
                d = S[::-1] ** 2 / X.shape[0]
                # Rows of Vt are eigenvectors, but also in ascending eigenvalue
                # order:
                W = Vt[::-1].T
                # Threshold by tau in W, this replaces \hat{D} in the Manjon et
                # al. (2013) formulation:
                W[:, d < tau[i, j, k]] = 0
                # This is equations 1 and 2 in Manjon 2013 (\hat{D} is subsumed
                # in the above nulling of eigenvectors corresponding to
                # eigenvalues smaller than tau:
                Xest = X.dot(W).dot(W.T) + M
                Xest = Xest.reshape(patch_size,
                                    patch_size,
                                    patch_size, arr.shape[-1])
                # This is equation 3 in Manjon 2013:
                theta[ix1:ix2, jx1:jx2, kx1:kx2] += 1.0 / (1.0 + np.sum(d > 0))
                thetax[ix1:ix2, jx1:jx2, kx1:kx2] += Xest / (1 + np.sum(d > 0))

    denoised_arr = thetax / theta

    if correct_bias:
        func = get_eta()
        # Prepare to iterate over the entire thing:
        index = ndindex(denoised_arr.shape)
        if isinstance(sigma, np.ndarray):
            for idx in index:
                denoised_arr[idx] = (sigma[idx] /
                                     func(denoised_arr[idx]))
        else:
            for idx in index:
                denoised_arr[idx] = (sigma /
                                     func(denoised_arr[idx]))

    return denoised_arr.astype(arr.dtype)
