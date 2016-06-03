"""
===========================================
Estimate Noise Levels for LocalPCA Datasets
===========================================

"""

import numpy as np
import nibabel as nib
import scipy as sp
from scipy import ndimage


def estimate_sigma_localpca(data, gtab):
    # first identify the number of the b0 images
    K = gtab.b0s_mask[gtab.b0s_mask].size

    if(K > 1):
        print("MUBE Estimate")
        # If multiple b0 values then
        data0 = data[..., gtab.b0s_mask]

    else:
        print("SIBE Estimate")
        # if only one b0 value then
        # SIBE Noise Estimate
        data0 = data[..., ~gtab.b0s_mask]
        # MUBE Noise Estimate

    # Form their matrix
    X = data0.reshape(
        data0.shape[0] *
        data0.shape[1] *
        data0.shape[2],
        data0.shape[3])
    # X = NxK
    # Now to subtract the mean
    M = np.mean(X, axis=0)
    X = X - M
    C = np.transpose(X).dot(X)
    C = C / M.shape[0]
    # Do PCA and find the lowest principal component
    [d, W] = np.linalg.eigh(C)
    d[d < 0] = 0
    d_new = d[d != 0]
    # we want eigen vectors of XX^T so we do
    V = X.dot(W)
    # unit normalize the clolumns of V
    # V = V / np.linalg.norm(V, ord=2, axis=0)
    # As the W is sorted in increasing eignevalue order, so is V
    # choose the smallest principle component
    I = V[:, d.shape[0] - d_new.shape[0]].reshape(
        data.shape[0],
        data.shape[1],
        data.shape[2])

    sigma = np.zeros(I.shape, dtype=np.float64)
    count = np.zeros_like(I)
    mean = np.zeros_like(I).astype(np.float64)

    # for i in range(1, I.shape[0] - 1):
    #     print(i)
    #     for j in range(1, I.shape[1] - 1):
    #         for k in range(1, I.shape[2] - 1):

    #             sigma[i, j, k] += np.var(I[i - 1:i + 2,
    #                                        j - 1:j + 2,
    #                                        k - 1:k + 2])
    #             temp = data[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2, :]
    #             mean[
    #                 i, j, k] = np.mean(temp)

    for i in range(1, I.shape[0] - 1):
        for j in range(1, I.shape[1] - 1):
            for k in range(1, I.shape[2] - 1):

                temp = I[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]
                temp = (temp - np.mean(temp)) * (temp - np.mean(temp))
                sigma[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2] += temp
                temp = data[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2, :]
                mean[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2] += np.mean(temp)
                count[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2] += 1

    sigma = sigma / count

    # Compute the local mean of for the data
    mean = mean / count
    snr = np.zeros_like(I).astype(np.float64)
    sigma_corr = np.zeros_like(data).astype(np.float64)

    # find the SNR and make the correction
    # SNR Correction
    print("Noise estimation done without correction")
    # for l in range(data.shape[3]):
    snr = mean / np.sqrt(sigma)
    eta = 2 + snr**2 - (np.pi / 8) * np.exp(-0.5 * (snr**2)) * ((2 + snr**2) * sp.special.iv(
        0, 0.25 * (snr**2)) + (snr**2) * sp.special.iv(1, 0.25 * (snr**2)))**2
    sigma_corr = sigma / eta
    print("Noise estimate corrected for rician noise")
    # smoothing by lpf
    sigma_corr[np.isnan(sigma_corr)] = 0
    # sigma_corr_mean = np.mean(sigma_corr, axis=3)
    sigma_corrr = ndimage.gaussian_filter(sigma_corr, 3)
    sigmar = ndimage.gaussian_filter(sigma, 3)
    print("Noise estimation finished")
    return [sigmar, sigma_corrr]
