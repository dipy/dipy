from numbers import Number

import numpy as np

from dipy.denoise.nlmeans_block import nlmeans_block
from dipy.testing.decorators import warning_for_keywords


@warning_for_keywords()
def non_local_means(
    arr, sigma, *, mask=None, patch_radius=1, block_radius=5, rician=True
):
    r"""Non-local means for denoising 3D and 4D images, using blockwise
    averaging approach.

    See :footcite:p:`Coupe2008` and :footcite:p:`Coupe2012` for further details
    about the method.

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be denoised
    mask : 3D ndarray
        Mask on data where the non-local means will be applied.
    sigma : float or ndarray
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
    .. footbibliography::

    """
    if isinstance(sigma, np.ndarray) and sigma.size == 1:
        sigma = sigma.item()
    if isinstance(sigma, np.ndarray):
        if arr.ndim == 3:
            raise ValueError("sigma should be a scalar for 3D data", sigma)
        if not np.issubdtype(sigma.dtype, np.number):
            raise ValueError("sigma should be an array of floats", sigma)
        if arr.ndim == 4 and sigma.ndim != 1:
            raise ValueError("sigma should be a 1D array for 4D data", sigma)
        if arr.ndim == 4 and sigma.shape[0] != arr.shape[-1]:
            raise ValueError(
                "sigma should have the same length as the last "
                "dimension of arr for 4D data",
                sigma,
            )
    else:
        if not isinstance(sigma, Number):
            raise ValueError("sigma should be a float", sigma)
        # if sigma is a scalar and arr is 4D, we assume the same sigma for all
        if arr.ndim == 4:
            sigma = np.array([sigma] * arr.shape[-1])

    if mask is None and arr.ndim > 2:
        mask = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]), dtype="f8")
    else:
        mask = np.ascontiguousarray(mask, dtype="f8")

    if mask.ndim != 3:
        raise ValueError("mask needs to be a 3D ndarray", mask.shape)

    if arr.ndim == 3:
        return np.array(
            nlmeans_block(
                np.double(arr), mask, patch_radius, block_radius, sigma, int(rician)
            )
        ).astype(arr.dtype)
    elif arr.ndim == 4:
        denoised_arr = np.zeros_like(arr)
        for i in range(arr.shape[-1]):
            denoised_arr[..., i] = np.array(
                nlmeans_block(
                    np.double(arr[..., i]),
                    mask,
                    patch_radius,
                    block_radius,
                    sigma[i],
                    int(rician),
                )
            ).astype(arr.dtype)

        return denoised_arr

    else:
        raise ValueError("Only 3D or 4D array are supported!", arr.shape)
