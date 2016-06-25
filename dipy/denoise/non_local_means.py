from __future__ import division, print_function

import numpy as np
from dipy.denoise.nlmeans_block import nlmeans_block


def non_local_means(arr, sigma, mask=None, patch_radius=1, block_radius=5,
                    rician=True):
    """ Non-local means for denoising 3D and 4D images, using
        blockwise averaging approach

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.
    num_threads : int
        Number of threads. If None (default) then all available threads
        will be used (all CPU cores).

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.

    References
    ----------

    [1] P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot,
        "An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic
        Resonance Images"
        IEEE Transactions on Medical Imaging, 27(4):425-441, 2008

    [2] Pierrick Coupe, Jose Manjon, Montserrat Robles, Louis Collins.
        "Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising"
        IET Image Processing, Institution of Engineering and Technology, 2011

    """
    if mask is None:
        mask = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]), dtype='f8')
    else:
        mask = np.ascontiguousarray(mask, dtype='f8')

    if mask.ndim != 3:
        raise ValueError('mask needs to be a 3D ndarray', mask.shape)

    if arr.ndim == 3:
        sigma = np.ones(arr.shape, dtype=np.float64) * sigma
        return np.array(nlmeans_block(
                np.double(arr),
                mask,
                patch_radius,
                block_radius,
                sigma[0,0,0],
                np.int(rician))).astype(arr.dtype)
    elif arr.ndim == 4:
        denoised_arr = np.zeros_like(arr)

        if isinstance(sigma, np.ndarray) and sigma.ndim == 3:
            sigma = (np.ones(arr.shape, dtype=np.float64) *
                     sigma[..., np.newaxis])
        else:
            sigma = np.ones(arr.shape, dtype=np.float64) * sigma

        for i in range(arr.shape[-1]):
            denoised_arr[..., i] = np.array(nlmeans_block(np.double(
                arr[..., i]), mask, patch_radius, block_radius, sigma[0, 0, 0, 0],
                np.int(rician))).astype(arr.dtype)

        return denoised_arr

    else:
        raise ValueError("Only 3D or 4D array are supported!", arr.shape)
