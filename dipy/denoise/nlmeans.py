from __future__ import division, print_function

import numpy as np
from dipy.denoise.denspeed import nlmeans_3d
from scipy.ndimage.filters import convolve


def nlmeans(arr, sigma, mask=None, patch_radius=1, block_radius=5, rician=True):
    """ Non-local means for denoising 3D and 4D images

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.
    """

    if arr.ndim == 3:

        return nlmeans_3d(arr, mask, sigma,
                          patch_radius, block_radius,
                          rician).astype(arr.dtype)

    if arr.ndim == 4:

        denoised_arr = np.zeros_like(arr)
        sigma_arr = np.ones(arr.shape[-1], dtype=np.float32) * sigma

        for i in range(arr.shape[-1]):
            sigma = sigma_arr[i]
            denoised_arr[..., i] = nlmeans_3d(arr[..., i],
                                              mask,
                                              sigma,
                                              patch_radius,
                                              block_radius,
                                              rician).astype(arr.dtype)

        return denoised_arr


def estimate_sigma(arr):
    """Standard deviation estimation from local patches

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    Returns
    -------
    sigma : ndarray
        standard deviation of the noise, one estimation per volume.
    """
    k = np.zeros((3, 3, 3), dtype=np.int8)

    k[0, 1, 1] = 1
    k[2, 1, 1] = 1
    k[1, 0, 1] = 1
    k[1, 2, 1] = 1
    k[1, 1, 0] = 1
    k[1, 1, 2] = 1

    if arr.ndim == 3:
        sigma = np.zeros(1, dtype=np.float32)
        arr = arr[..., None]
    elif arr.ndim == 4:
        sigma = np.zeros(arr.shape[-1], dtype=np.float32)
    else:
        raise ValueError("Array shape is not supported!", arr.shape)

    for i in range(sigma.size):
        mean_block = np.sqrt(6/7) * (arr[..., i] - 1/6 * convolve(arr[..., i], k))
        mask = mean_block.astype(np.bool)
        sigma[i] = np.sqrt(np.mean(mean_block[mask]**2))

    return sigma
