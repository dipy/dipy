#!python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2019,
#
# Developer : M. Okan Irfanoglu (irfanoglumo@mail.nih.gov)
#             Anh Thai (15thai@cua.edu, thaias@nih.gov)
# -------------------------------------------------------------------------
# Adapted implementation of the random matrix theory procedure suggested by:
#
# [Veraart16] Veraart J, Fieremans E, Novikov DS (2016)
#             Diffusion MRI noise mapping using random matrix theory.
#             Magnetic resonance in Medicine 76(5), p1582-1593.
#             https://doi.org/10.1002/mrm.26059
# -------------------------------------------------------------------------

import numpy as np
from math import floor, sqrt
from scipy.linalg import eigh


def randommatrix_lpca(arr, patch_size=0, out_dtype=None):
    """Local PCA-based denoising of diffusion datasets.

        Parameters
        ----------
        arr : 4D array
            Array of data to be denoised. The dimensions are (X, Y, Z, N),
            where N are the diffusion gradient directions.
        patch_size : int, optional
            The diameter of the local patch to be taken around each voxel (in
            voxels). The radius will be half of this value. If not provided,
            the default will be automatically computed as:

            .. math ::
                    patch_extent = max(5,|lfloor N^{1/3}| rfloor)

        out_dtype : str or dtype, optional
            The dtype for the output array. Default: output has the same
            dtype as the input.

        Returns
        -------
        denoised_arr : 4D array
            This is the denoised array of the same size as that of
            the input data, clipped to non-negative values
        noise_arr : 3D array
            Voxelwise standard deviation of the noise estimated
            from the data.
        sigma : float
            Mean value of noise standard deviations over all voxels
            (mean of noise_arr).

        References
        ----------
        .. [Veraart16] Veraart J, Fieremans E, Novikov DS (2016)
                      Diffusion MRI noise mapping using random matrix theory.
                      Magnetic resonance in Medicine 76(5), p1582-1593.
                      https://doi.org/10.1002/mrm.26059
        """

    if out_dtype is None:
        out_dtype = arr.dtype

    # We retain float64 precision, if the input is in this precision:
    if arr.dtype == np.float64:
        calc_dtype = np.float64
    # Otherwise, we'll calculate things in float32 (saving memory)
    else:
        calc_dtype = np.float32

    if not arr.ndim == 4:
        raise ValueError("Noise reducing can only be performed on 4D arrays.",
                         arr.shape)

    # Denoising array dimension varibles
    nVols = arr.shape[-1]
    if patch_size <= 0:
        patch_size = max(5, nVols ** (1 / 3))

    if patch_size % 2 == 0:
        patch_size -= 1
    patch_radius = patch_size // 2
    m = arr.shape[-1]
    n = patch_size ** 3
    r = m if (m < n) else n

    arr = arr.astype(calc_dtype)
    noise_arr = np.zeros(arr.shape[:-1])
    denoised_arr = np.zeros(arr.shape)

    pad_width = ((patch_radius, patch_radius),
                 (patch_radius, patch_radius),
                 (patch_radius, patch_radius),
                 (0, 0))
    padded_arr = np.pad(arr,
                        pad_width=pad_width,
                        mode='constant',
                        constant_values=0)

    for k in range(0, arr.shape[2]):
        for j in range(0, arr.shape[1]):
            for i in range(0, arr.shape[0]):
                Y = padded_arr[i: i + 2 * patch_radius + 1,
                               j:j + 2 * patch_radius + 1,
                               k:k + 2 * patch_radius + 1,
                               :]
                X = (Y.reshape(n, m)).transpose()

                if m <= n:
                    XtX = X.dot(np.transpose(X))
                else:
                    XtX = np.transpose(X).dot(X)

                # Computing Eigenvalues and EigenVector
                eigenVal, eigenVec = eigh(XtX)

                # Sorting eigen value in descending order
                eigenVal[::-1].sort()  # Descending Sort
                eigenVal = eigenVal / n

                # Looking for the non-positive eigen value index
                p = eigenVal.shape[0]
                for t in range(eigenVal.shape[0]):
                    if eigenVal[t] <= 0:
                        p = t
                        break

                cum_eigenVal = np.zeros(eigenVal.shape)

                for t in range(p - 2, -1, -1):
                    cum_eigenVal[t] = cum_eigenVal[t + 1] + eigenVal[t + 1]

                # Finding p_hat
                p_hat = 0
                for t in range(p - 1):
                    gamma = (p - t - 1) / n
                    sigma_hat = (eigenVal[t + 1] - eigenVal[p - 1]) / \
                                (4 * sqrt(gamma))
                    RHS = (p - t - 1) * sigma_hat
                    if cum_eigenVal[t] >= RHS:
                        p_hat = t
                        break

                if p_hat == p - 1:
                    sigma2 = 0
                else:
                    sigma2 = cum_eigenVal[p_hat] / (p - p_hat - 1)

                eigenVal[0:r - p_hat-1] = 0
                eigenVal[r - p_hat-1:] = 1

                # Reconstructing denoised image
                if (p_hat != (p - 1)):
                    if m <= n:
                        nvals = eigenVec.dot(np.diag(eigenVal))\
                            .dot(eigenVec.transpose()).dot(X[:, n // 2])
                    else:
                        nvals = np.dot(X,
                                       np.dot(eigenVec,
                                              np.dot(np.diag(eigenVal),
                                                     eigenVec.transpose()[:, n // 2]))
                                       )
                else:
                    nvals = X[:, n // 2]

                denoised_arr[i, j, k, :] = nvals
                noise_arr[i, j, k] = sqrt(sigma2)

    sigma = np.mean(noise_arr[np.nonzero(noise_arr)])
    return denoised_arr.astype(out_dtype), noise_arr.astype(out_dtype), sigma
