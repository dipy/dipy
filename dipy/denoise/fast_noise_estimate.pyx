"""
===========================================
Estimate Noise Levels for LocalPCA Datasets
===========================================

"""

import numpy as np
import nibabel as nib
import scipy as sp
from scipy import ndimage
cimport cython
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fast_noise_estimate(data, gtab):
    """ Noise estimation for local PCA denoising

    Parameters
    ----------
    data: 4D array
        the input dMR data

    gtab: gradient table object
        gradient information for the data gives us the
        bvals and bvecs of diffusion data, which is needed
        here to select between the noise estimation
        methods

    Returns
    -------
    sigma_corr: 3D array
        The local noise variance estimate

    References
    ----------
    [1]: Manjon JV, Coupe P, Concha L, Buades A, Collins DL
        "Diffusion Weighted Image Denoising Using Overcomplete
         Local PCA"
         

    """
    
    # first identify the number of the b0 images
    K = gtab.b0s_mask[gtab.b0s_mask].size

    if(K > 1):
        # If multiple b0 values then
        data0 = data[..., gtab.b0s_mask]

    else:
        # if only one b0 value then
        # SIBE Noise Estimate
        data0 = data[..., ~gtab.b0s_mask]
        # MUBE Noise Estimate

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
        double[:, :, :] sigma = np.zeros((n0, n1, n2))
        double[:, :, :, :] data0temp = data0

    X = data0.reshape(
        data0.shape[0] *
        data0.shape[1] *
        data0.shape[2],
        data0.shape[3])
    # X = NxK
    # Now to subtract the mean
    M = np.mean(X, axis=0)
    X = X - np.array([M, ] * X.shape[0],
                                      dtype=np.float64)
    C = np.transpose(X).dot(X)
    C = C / M.shape[0]
    # Do PCA and find the lowest principal component
    [d, W] = np.linalg.eigh(C)
    d[d < 0] = 0
    d_new = d[d != 0]
    # we want eigen vectors of XX^T so we do
    V = X.dot(W)

    I = V[:, d.shape[0] - d_new.shape[0]].reshape(
        data0.shape[0],
        data0.shape[1],
        data0.shape[2])
                
    with nogil:
        for i in range(1, n0 - 1):
            for j in range(1, n1 - 1):
                for k in range(1, n2 - 1):
                    sum_reg = 0
                    temp1 = 0
                    for i0 in range(-1,2):
                        for j0 in range(-1,2):
                            for k0 in range(-1,2):
                                sum_reg += I[i + i0, j + j0, k + k0] / 27.0
                                for l0 in range(n3):
                                    temp1 += (data0temp[i + i0, j+ j0, k + k0, l0]) / (27.0 * n3)

                    for i0 in range(-1,2):
                        for j0 in range(-1,2):
                            for k0 in range(-1,2):
                                sigma[i + i0, j +j0, k + k0] += (I[i + i0, j + j0, k + k0] - sum_reg)**2
                                mean[i + i0, j + j0, k + k0] += temp1
                                count[i + i0, j +j0, k + k0] += 1             

    sigma = np.divide(sigma, count)
    
    # Compute the local mean of for the data
    mean = np.divide(mean, count)
    snr = np.zeros_like(I).astype(np.float64)
    sigma_corr = np.zeros_like(data).astype(np.float64)

    # find the SNR and make the correction
    # SNR Correction

    snr = np.divide(mean, np.sqrt(sigma))
    eta = 2 + snr**2 - (np.pi / 8) * np.exp(-0.5 * (snr**2)) * ((2 + snr**2) * sp.special.iv(
        0, 0.25 * (snr**2)) + (snr**2) * sp.special.iv(1, 0.25 * (snr**2)))**2
    sigma_corr = sigma / eta
    # smoothing by lpf
    sigma_corr[np.isnan(sigma_corr)] = 0
    sigma_corrr = ndimage.gaussian_filter(sigma_corr, 3)
    return sigma_corrr
