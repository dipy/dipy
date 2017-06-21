"""
================================
PCA Based Local Noise Estimation
================================

"""

import numpy as np
import nibabel as nib
import scipy.special as sps
from scipy import ndimage
cimport cython
cimport numpy as cnp

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
                       smooth=2):
    """ PCA based local noise estimation.

    Parameters
    ----------
    data: 4D array
        the input dMRI data.

    gtab: gradient table object
      gradient information for the data gives us the bvals and bvecs of
      diffusion data, which is needed here to select between the noise
      estimation methods.
    patch_radius : int
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 1 (estimate noise in blocks of 3x3x3 voxels).
    correct_bias : bool
      Whether to correct for bias due to Rician noise. This is an implementation
      of equation 8 in [1]_.

    smooth : int
      Radius of a Gaussian smoothing filter to apply to the noise estimate
      before returning. Default: 2.

    Returns
    -------
    sigma_corr: 3D array
        The local noise standard deviation estimate.

    References
    ----------
    .. [1] Manjon JV, Coupe P, Concha L, Buades A, Collins DL "Diffusion
           Weighted Image Denoising Using Overcomplete Local PCA". PLoS ONE
           8(9): e73021. doi:10.1371/journal.pone.0073021.
    """
    # first identify the number of the b0 images
    K = gtab.b0s_mask[gtab.b0s_mask].size

    if(K > 1):
        # If multiple b0 values then use MUBE noise estimate
        data0 = data[..., gtab.b0s_mask]
        sibe = False

    else:
        # if only one b0 value then SIBE noise estimate
        data0 = data[..., ~gtab.b0s_mask]
        sibe = True

    data0 = data0.astype(np.float64)
    cdef:
        cnp.npy_intp n0 = data0.shape[0]
        cnp.npy_intp n1 = data0.shape[1]
        cnp.npy_intp n2 = data0.shape[2]
        cnp.npy_intp n3 = data0.shape[3]
        cnp.npy_intp nsamples = n0 * n1 * n2
        cnp.npy_intp i, j, k, i0, j0, k0, l0
        cnp.npy_intp pr = patch_radius
        cnp.npy_intp  patch_size = 2 * pr + 1
        double norm = patch_size ** 3
        double sum_reg, temp1
        double[:, :, :] I = np.zeros((n0, n1, n2))

    X = data0.reshape(nsamples, n3)
    # Demean:
    M = np.mean(X, axis=0)
    X = X - M
    U, S, Vt = svd(X, *svd_args)[:3]
    # Rows of Vt are the eigenvectors, in ascending eigenvalue order:
    W = Vt.T
    # Project into the data space
    V = X.dot(W)

    # Grab the column corresponding to the smallest eigen-vector/-value:
    I = V[:, -1].reshape(n0, n1, n2)
    del V, W, X, U, S, Vt

    cdef:
      double[:, :, :] count = np.zeros((n0, n1, n2))
      double[:, :, :] mean = np.zeros((n0, n1, n2))
      double[:, :, :] sigma_sq = np.zeros((n0, n1, n2))
      double[:, :, :, :] data0temp = data0



    with nogil:
        for i in range(pr, n0 - pr):
            for j in range(pr, n1 - pr):
                for k in range(pr, n2 - pr):
                    sum_reg = 0
                    temp1 = 0
                    for i0 in range(-pr, pr + 1):
                        for j0 in range(-pr, pr + 1):
                            for k0 in range(-pr, pr + 1):
                                sum_reg += I[i + i0, j + j0, k + k0] / norm
                                for l0 in range(n3):
                                    temp1 += (data0temp[i + i0, j+ j0, k + k0, l0]) / (norm * n3)

                    for i0 in range(-pr, pr + 1):
                        for j0 in range(-pr, pr + 1):
                            for k0 in range(-pr, pr + 1):
                                sigma_sq[i + i0, j +j0, k + k0] += (
                                    I[i + i0, j + j0, k + k0] - sum_reg) ** 2
                                mean[i + i0, j + j0, k + k0] += temp1
                                count[i + i0, j +j0, k + k0] += 1

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
                  ((2 + snr_sq) * sps.iv(0, (snr_sq) / 4) +
                  (snr_sq) * sps.iv(1, (snr_sq) / 4)) ** 2).astype(float)
      xi[snr > 37.4] = 1
      sigma_corr = sigma_sq / xi
      sigma_corr[np.isnan(sigma_corr)] = 0
    else:
      sigma_corr = sigma_sq

    if smooth is not None:
      sigma_corr = ndimage.gaussian_filter(sigma_corr, smooth)

    return np.sqrt(sigma_corr)
