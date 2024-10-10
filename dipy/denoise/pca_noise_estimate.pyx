"""
================================
PCA Based Local Noise Estimation
================================

"""

import numpy as np
import scipy.special as sps
from scipy import ndimage
cimport cython
cimport numpy as cnp
from warnings import warn


# Try to get the SVD through direct API to lapack:
try:
    from scipy.linalg.lapack import sgesvd as svd
    svd_args = [1, 0]
# If you have an older version of scipy, we fall back
# on the standard scipy SVD API:
except ImportError:
    from scipy.linalg import svd
    svd_args = [False]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def pca_noise_estimate(data, gtab, patch_radius=1, correct_bias=True,
                       smooth=2, images_as_samples=False):
    """PCA based local noise estimation.

    Parameters
    ----------
    data : 4D array
        the input dMRI data. The first 3 dimension must have size >= 2 *
        patch_radius + 1 or size = 1.
    gtab : gradient table object
        gradient information for the data gives us the bvals and bvecs of
        diffusion data, which is needed here to select between the noise
        estimation methods.
    patch_radius : int
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 1 (estimate noise in blocks of 3x3x3 voxels).
    correct_bias : bool
        Whether to correct for bias due to Rician noise. This is an
        implementation of equation 8 in :footcite:p:`Manjon2013`.
    smooth : int
        Radius of a Gaussian smoothing filter to apply to the noise estimate
        before returning. Default: 2.
    images_as_samples : bool, optional
        Whether to use images as rows (samples) for PCA (algorithm in
        :footcite:p:`Manjon2013`) or to use images as columns (features).

    Returns
    -------
    sigma_corr: 3D array
        The local noise standard deviation estimate.

    Notes
    -----
    In :footcite:p:`Manjon2013`, images are used as samples, so voxels are
    features, therefore eigenvectors are image-shaped. However,
    :footcite:t:`Manjon2013` is not clear on how to use these eigenvectors
    to determine the noise level, so here eigenvalues (variance over samples
    explained by eigenvectors) are used to scale the eigenvectors. Use
    images_as_samples=True to use this algorithm. Alternatively, voxels can
    be used as samples using images_as_samples=False. This is not the
    canonical algorithm of :footcite:t:`Manjon2013`.

    References
    ----------
    .. footbibliography::
    """
    # first identify the number of the b0 images
    K = gtab.b0s_mask[gtab.b0s_mask].size

    if K > 1:
        # If multiple b0 values then use MUBE noise estimate
        data0 = data[..., gtab.b0s_mask]
        sibe = False

    else:
        # if only one b0 value then SIBE noise estimate
        data0 = data[..., ~gtab.b0s_mask]
        sibe = True

    if patch_radius < 1:
        warn("Minimum patch radius must be 1, setting to 1", UserWarning)
        patch_radius = 1

    data0 = data0.astype(np.float64)
    cdef:
        # We need to be explicit because of boundscheck = False
        cnp.npy_intp dsm = np.min(data0.shape[0:3])
        cnp.npy_intp n0 = data0.shape[0]
        cnp.npy_intp n1 = data0.shape[1]
        cnp.npy_intp n2 = data0.shape[2]
        cnp.npy_intp n3 = data0.shape[3]
        cnp.npy_intp nsamples = n0 * n1 * n2
        cnp.npy_intp i, j, k, i0, j0, k0, l0
        cnp.npy_intp prx = patch_radius if n0 > 1 else 0
        cnp.npy_intp pry = patch_radius if n1 > 1 else 0
        cnp.npy_intp prz = patch_radius if n2 > 1 else 0
        cnp.npy_intp norm = (2 * prx + 1) * (2 * pry + 1) * (2 * prz + 1)
        double sum_reg, temp1
        double[:, :, :] I = np.zeros((n0, n1, n2))

    # check dimensions of data
    if (dsm != 1) and (dsm < 2 * patch_radius + 1):
        raise ValueError("Array 'data' is incorrect shape")

    if images_as_samples:
        X = data0.reshape(nsamples, n3).T
    else:
        X = data0.reshape(nsamples, n3)

    # Demean:
    M = np.mean(X, axis=0)
    X = X - M
    U, S, Vt = svd(X, *svd_args)[:3]
    # Rows of Vt are the eigenvectors, in ascending eigenvalue order:
    W = Vt.T

    if images_as_samples:
        W = W.astype('double')
        # #vox(features) >> # img(samples), last eigval zero (X is centered)
        idx = n3 - 2  # use second-to-last eigvec
        V = W[:, idx].reshape(n0, n1, n2)

        # ref [1]_ method is ambiguous on how to use image-shaped eigvec
        # since eigvec is normalized, used eigval=variance for scale
        I = V * S[idx]
    else:
        # Project into the data space
        V = X.dot(W)

        # Grab the column corresponding to the smallest eigen-vector/-value:
        # #vox(samples) >> #img(features), last eigenvector is meaningful
        I = V[:, -1].reshape(n0, n1, n2)

    del V, W, X, U, S, Vt

    cdef:
        double[:, :, :] count = np.zeros((n0, n1, n2))
        double[:, :, :] mean = np.zeros((n0, n1, n2))
        double[:, :, :] sigma_sq = np.zeros((n0, n1, n2))
        double[:, :, :, :] data0temp = data0

    with nogil:
        for i in range(prx, n0 - prx):
            for j in range(pry, n1 - pry):
                for k in range(prz, n2 - prz):
                    sum_reg = 0
                    temp1 = 0
                    for i0 in range(-prx, prx + 1):
                        for j0 in range(-pry, pry + 1):
                            for k0 in range(-prz, prz + 1):
                                sum_reg += I[i + i0, j + j0, k + k0] / norm
                                for l0 in range(n3):
                                    temp1 += (data0temp[i + i0, j + j0, k + k0, l0])\
                                             / (norm * n3)

                    for i0 in range(-prx, prx + 1):
                        for j0 in range(-pry, pry + 1):
                            for k0 in range(-prz, prz + 1):
                                sigma_sq[i + i0, j + j0, k + k0] += (
                                    I[i + i0, j + j0, k + k0] - sum_reg) ** 2
                                mean[i + i0, j + j0, k + k0] += temp1
                                count[i + i0, j + j0, k + k0] += 1

    sigma_sq = np.divide(sigma_sq, count)

    # find the SNR and make the correction for bias due to Rician noise:
    if correct_bias:
        mean = np.divide(mean, count)
        snr = np.divide(mean, np.sqrt(sigma_sq))
        snr_sq = (snr ** 2)
        # xi is practically equal to 1 above 37.4, and we overflow, raising
        # warnings and creating ot-a-numbers.
        # Instead, we will replace these values with 1 below
        with np.errstate(over='ignore', invalid='ignore'):
            xi = (2 + snr_sq - (np.pi / 8) * np.exp(-snr_sq / 2) *
                  ((2 + snr_sq) * sps.iv(0, snr_sq / 4) +
                  snr_sq * sps.iv(1, snr_sq / 4)) ** 2).astype(float)
        xi[snr > 37.4] = 1
        sigma_corr = sigma_sq / xi
        sigma_corr[np.isnan(sigma_corr)] = 0
    else:
        sigma_corr = sigma_sq

    if smooth is not None:
        sigma_corr = ndimage.gaussian_filter(sigma_corr, smooth)

    return np.sqrt(sigma_corr)
