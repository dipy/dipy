"""
===========================================
Estimate Noise Levels for LocalPCA Datasets
===========================================
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
def pca_noise_estimate(data, gtab, correct_bias=False, smooth=None):
    """Noise estimation for local PCA denoising.

    Parameters
    ----------
    data: 4D array
        the input dMR data.

    gtab: gradient table object
        gradient information for the data gives us the bvals and bvecs of
        diffusion data, which is needed here to select between the noise
        estimation methods.

    correct_bias : bool
      Whether to correct for bias due to Rician noise. This is an
      implementation of equation 8 in [1]_.

    smooth : int
        Radius of a Gaussian smoothing filter to apply to the noise estimate
        before returning. Default: no smoothing.


    Returns
    -------
    sigma_corr: 3D array
        The local noise variance estimate

    References
    ----------
    .. [1] Manjon JV, Coupe P, Concha L, Buades A, Collins DL "Diffusion
           Weighted Image Denoising Using Overcomplete Local PCA". PLoS ONE
           8(9): e73021. doi:10.1371/journal.pone.0073021
    """
    if len(data.shape) != 4:
      raise ValueError("Data must be 4 dimensional", data.shape)

    # first identify the number of  b0 images
    K = gtab.b0s_mask[gtab.b0s_mask].size

    if(K > 1):
        # If multiple b0 values then MUBE Noise Estimate
        data0 = data[..., gtab.b0s_mask]

    else:
        # if only one b0 value then SIBE Noise Estimate
        data0 = data[..., ~gtab.b0s_mask]

    cdef:
        cnp.npy_intp n0 = data0.shape[0]
        cnp.npy_intp n1 = data0.shape[1]
        cnp.npy_intp n2 = data0.shape[2]
        cnp.npy_intp n3 = data0.shape[3]
        cnp.npy_intp nsamples = n0 * n1 * n2
        cnp.npy_intp i, j, k, i0, j0, k0, l0
        double sum_reg, temp1
        double[:, :, :] I = np.zeros((n0, n1, n2))
        double[:, :, :] count = np.zeros((n0, n1, n2))
        double[:, :, :] mean = np.zeros((n0, n1, n2))
        double[:, :, :] sigma_sq = np.zeros((n0, n1, n2))
        double[:, :, :, :] data0temp = data0

    X = data0.reshape(n0 * n1 * n2, n3)

    # Demean the data:
    X = X - np.mean(X, axis=0)

    # PCA using the SVD:
    U, S, Vt = svd(X, *svd_args)[:3]
    # Items in S are the eigenvalues, but in ascending order
    # We invert the order (=> descending), square and normalize
    # \lambda_i = s_i^2 / n
    d = S[::-1] ** 2 / X.shape[0]
    # Rows of Vt are eigenvectors, but also in ascending eigenvalue
    # order:
    W = Vt[::-1].T
    d[d < 0] = 0
    d_new = d[d != 0]
    # we want eigen vectors of XX^T so we do
    V = X.dot(W)
    print(d.shape)
    print(d_new.shape)
    print(V.shape)
    print((V[:, d.shape[0] - d_new.shape[0]]).shape)
    print(n0)
    print(n1)
    print(n2)
    I = np.reshape(V[:, d.shape[0] - d_new.shape[0]], (n0, n1, n2))
    print(I.shape)

    with nogil:
        for i in range(1, n0 - 1):
            for j in range(1, n1 - 1):
                for k in range(1, n2 - 1):
                    sum_reg = 0
                    temp1 = 0
                    for i0 in range(-1, 2):
                        for j0 in range(-1, 2):
                            for k0 in range(-1, 2):
                                sum_reg += I[i + i0, j + j0, k + k0] / 27.0
                                for l0 in range(n3):
                                    temp1 += (
                                    data0temp[i + i0, j+ j0, k + k0, l0] / (27.0 * n3))

                    for i0 in range(-1, 2):
                        for j0 in range(-1, 2):
                            for k0 in range(-1, 2):
                                sigma_sq[i + i0, j +j0, k + k0] = (
                                  sigma_sq[i + i0, j +j0, k + k0] +
                                  (I[i + i0, j + j0, k + k0] - sum_reg) ** 2)
                                mean[i + i0, j + j0, k + k0] += temp1
                                count[i + i0, j +j0, k + k0] += 1

    sigma_sq = np.divide(sigma_sq, count)

    # Compute the local mean of the data
    mean = np.divide(mean, count)

    if correct_bias:
      # find the SNR and make the correction (equation 8 in Manjon 2013)
      snr = np.divide(mean, np.sqrt(sigma_sq))
      snr_sq = snr ** 2
      zeta = (
        2 + snr_sq -
        (np.pi / 8) * np.exp(-snr_sq / 2) *
        ((2 + snr_sq) * sps.iv(0, (snr_sq) / 4) +
         (snr_sq) * sps.iv(1, (snr_sq) / 4)) ** 2)
      sigma_sq = np.divide(sigma_sq, zeta)

    # Avoid NaNs:
    #sigma_sq[np.isnan(sigma_sq)] = 0

    if smooth is not None:
      # smoothing by lpf
      sigma_sq = ndimage.gaussian_filter(sigma_sq, smooth)

    sigma_sq[np.isnan(sigma_sq)] = 0

    return np.sqrt(sigma_sq)
