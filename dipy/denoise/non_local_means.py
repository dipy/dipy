from __future__ import division, print_function

import numpy as np
from dipy.denoise.nlmeans_block import nlmeans_block


def non_local_means(arr, sigma, mask=None, patch_radius=1, block_radius=5,
                    rician=True):
    r""" Non-local means for denoising 3D and 4D images, using
        blockwise averaging approach

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : double
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

    References
    ----------

    .. [Coupe08] P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C.
                 Barillot, An Optimized Blockwise Non Local Means Denoising
                 Filter for 3D Magnetic Resonance Images, IEEE Transactions on
                 Medical Imaging, 27(4):425-441, 2008
    """
    if not np.isscalar(sigma) and not sigma.shape == (1, ):
        raise ValueError("Sigma input needs to be of type double", sigma)
    if mask is None and arr.ndim > 2:
        mask = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.int32)
    else:
        mask = mask.astype(np.int32)

    if mask.ndim != 3:
        raise ValueError('mask needs to be a 3D ndarray', mask.shape)

    denoised_arr = np.zeros_like(arr, dtype=np.float32)

    # the cython part expects an array of double
    arr = arr.astype(np.float64)
    # it also does not recognize bool dtype
    rician = int(rician)

    if arr.ndim == 3:
        denoised_arr[:] = nlmeans_block(arr,
                                        mask,
                                        patch_radius,
                                        block_radius,
                                        sigma,
                                        rician)
    elif arr.ndim == 4:
            for i in range(arr.shape[-1]):
                denoised_arr[..., i] = nlmeans_block(arr[..., i],
                                                     mask,
                                                     patch_radius,
                                                     block_radius,
                                                     sigma,
                                                     rician)
    else:
        raise ValueError("Only 3D or 4D array are supported!", arr.shape)

    return np.asarray(denoised_arr)
