from numbers import Number

import numpy as np

from dipy.denoise.denspeed import nlmeans_3d
from dipy.testing.decorators import warning_for_keywords


@warning_for_keywords()
def nlmeans(
    arr,
    sigma,
    *,
    mask=None,
    patch_radius=1,
    block_radius=None,
    rician=True,
    num_threads=None,
    method="blockwise",
):
    r"""
    Non-local means denoising for 3D and 4D images with selectable algorithms.

    This implementation provides two different algorithms for non-local means denoising:
    the classic approach and an improved blockwise algorithm. The blockwise method
    offers better coordinate handling, memory efficiency, and performance through
    advanced parallelization and statistical pre-filtering.

    See :footcite:p:`Descoteaux2008a` for further details about the classic method.
    See :footcite:p:`Coupe2008` and :footcite:p:`Coupe2012` for further details
    about the blockwise method.

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be denoised. For 3D arrays, shape should be (height, width, depth).
        For 4D arrays, shape should be (height, width, depth, volumes) where the last
        dimension represents different volumes (e.g., DWI directions).
    sigma : float or 1D ndarray
        Standard deviation of the noise estimated from the data. For 3D arrays,
        this should be a scalar. For 4D arrays, this can be either a scalar (same noise
        level for all volumes) or a 1D array with length equal to the number of volumes.
    mask : 3D ndarray, optional
        Binary mask indicating which voxels to process. Should have shape
        (height, width, depth). Voxels with mask value 0 are set to 0 in output.
        If None, all voxels are processed.
    patch_radius : int, optional
        Radius for similarity search neighborhoods. Patches of size
        (2*patch_radius + 1)³ around each voxel are compared to find similar
        structures.
    block_radius : int, optional
        Radius for weighted averaging blocks. Each block has size
        (2*block_radius + 1)³. Larger blocks provide more smoothing but
        increase computational cost. If None, defaults are:

        - method='classic': 5 (11*11*11 blocks)
        - method='blockwise': 2 (5*5*5 blocks)
    rician : bool, optional
        If True, assumes Rician noise model (appropriate for magnitude MRI data).
        If False, assumes Gaussian noise model.
    num_threads : int, optional
        Number of OpenMP threads to use for parallel processing. If None,
        uses all available CPU threads. Set to 1 to disable parallel processing.
    method : str, optional
        Algorithm method to use:

        - 'blockwise': Improved algorithm with better coordinate handling,
          memory efficiency, and statistical pre-filtering
        - 'classic': Original algorithm with traditional implementation

        .. versionadded:: 1.12.0

    Returns
    -------
    denoised_arr : ndarray
        The denoised array with the same shape and dtype as the input ``arr``.
        Values are clipped to non-negative range for Rician noise model.

    Notes
    -----
    Due to coordinate bug fixes in the blockwise method, equivalent denoising
    quality may require different parameters between methods:
    - Classic patch_radius=3 ≈ Blockwise patch_radius=2
    - Block_radius can be smaller for blockwise due to efficiency improvements

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.denoise.nlmeans import nlmeans
    >>> # Create synthetic noisy data
    >>> clean_data = np.random.rand(50, 50, 50) * 100
    >>> noisy_data = clean_data + np.random.randn(50, 50, 50) * 10
    >>> # Denoise a 3D image with default blockwise method
    >>> denoised_data = nlmeans(noisy_data, sigma=10.0)
    >>> # Use classic method for compatibility:
    >>> denoised_classic = nlmeans(noisy_data, sigma=10.0, method='classic')
    >>> # Denoise 4D DWI data with different noise levels per volume:
    >>> dwi_data = np.random.rand(64, 64, 40, 30) * 1000  # 30 DWI directions
    >>> noise_levels = np.linspace(50, 100, 30)  # Varying noise
    >>> denoised_dwi = nlmeans(dwi_data, sigma=noise_levels)
    """
    if block_radius is None:
        if method == "classic":
            block_radius = 5
        elif method == "blockwise":
            block_radius = 2
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'classic' or 'blockwise'."
            )

    if isinstance(sigma, np.ndarray) and sigma.size == 1:
        sigma = sigma.item()
    if isinstance(sigma, np.ndarray):
        if not np.issubdtype(sigma.dtype, np.number):
            raise ValueError("sigma should be an array of floats", sigma)

        # Validation depends on both array dimensions and method
        if arr.ndim == 3:
            if method == "classic":
                # Classic method for 3D: sigma should be 3D array or scalar
                if sigma.ndim not in [0, 3] and not (
                    sigma.ndim == 3 and sigma.shape == arr.shape
                ):
                    raise ValueError(
                        "For classic method with 3D data, sigma should be scalar or"
                        f" 3D array, got shape {sigma.shape}"
                    )
            elif method == "blockwise":
                # Blockwise method: more flexible, accepts scalar, 1D, or 3D
                if sigma.ndim > 3:
                    raise ValueError(
                        "For blockwise method, sigma should be at most 3D, got "
                        f"shape {sigma.shape}"
                    )
        elif arr.ndim == 4:
            if sigma.ndim != 1:
                raise ValueError("sigma should be a 1D array for 4D data", sigma)
            if sigma.shape[0] != arr.shape[-1]:
                raise ValueError(
                    "sigma should have the same length as the last "
                    "dimension of arr for 4D data",
                    sigma,
                )
    else:
        if not isinstance(sigma, Number):
            raise ValueError("sigma should be a float", sigma)
        if arr.ndim == 4:
            sigma = np.array([sigma] * arr.shape[-1])

    if mask is None and arr.ndim > 2:
        mask = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float64)
    else:
        mask = np.ascontiguousarray(mask, dtype=np.float64)

    if mask.ndim != 3:
        raise ValueError("mask needs to be a 3D ndarray", mask.shape)

    if arr.ndim == 3:
        if method == "classic":
            # Classic method needs 3D sigma array
            if not isinstance(sigma, np.ndarray):
                sigma_3d = np.full(arr.shape, sigma, dtype="f8")
            else:
                # Ensure sigma is 3D and same shape as arr
                if sigma.shape != arr.shape:
                    if sigma.ndim == 0 or (sigma.ndim == 1 and len(sigma) == 1):
                        sigma_3d = np.full(arr.shape, float(sigma), dtype="f8")
                    else:
                        raise ValueError(
                            f"sigma shape {sigma.shape} incompatible "
                            f"with arr shape {arr.shape}"
                        )
                else:
                    sigma_3d = np.ascontiguousarray(sigma, dtype="f8")
        else:
            # For blockwise method, sigma can be scalar or array
            sigma_3d = sigma

        result = nlmeans_3d(
            np.ascontiguousarray(arr, dtype=np.float64),
            mask,
            sigma_3d,
            patch_radius,
            block_radius,
            rician,
            num_threads,
            method,
        )
        return np.asarray(result, dtype=arr.dtype)

    elif arr.ndim == 4:
        denoised_arr = np.zeros_like(arr)
        for i in range(arr.shape[-1]):
            # For classic method, we need to convert scalar sigma to 3D array
            if method == "classic":
                if isinstance(sigma, np.ndarray):
                    sigma_vol = np.full(arr[..., i].shape, sigma[i], dtype="f8")
                else:
                    sigma_vol = np.full(arr[..., i].shape, sigma, dtype="f8")
            else:
                # For blockwise method, use scalar sigma
                if isinstance(sigma, np.ndarray):
                    sigma_vol = sigma[i]
                else:
                    sigma_vol = sigma

            result = nlmeans_3d(
                np.ascontiguousarray(arr[..., i], dtype=np.float64),
                mask,
                sigma_vol,
                patch_radius,
                block_radius,
                rician,
                num_threads,
                method,
            )
            denoised_arr[..., i] = np.asarray(result, dtype=arr.dtype)

        return denoised_arr

    else:
        raise ValueError("Only 3D or 4D arrays are supported!", arr.shape)
