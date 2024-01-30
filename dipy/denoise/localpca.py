import copy
from warnings import warn

import numpy as np
from scipy.linalg.lapack import dgesvd as svd
from scipy.linalg import eigh

from dipy.denoise.pca_noise_estimate import pca_noise_estimate


def dimensionality_problem_message(arr, num_samples, spr):
    """Message about the number of samples being smaller than one less the
    dimensionality of the data to be denoised.

    Parameters
    ----------
    arr : ndarray
        Data to be denoised.
    num_samples : int
        Number of samples.
    spr : int
        Suggested patch radius.
.
    Returns
    -------
    str
        Message.
    """

    return (
        f"Number of samples {arr.shape[-1]} - 1 < Dimensionality "
        f"{num_samples}. This might have a performance impact. Increase p"
        f"atch_radius to {spr} to avoid this."
    )


def _pca_classifier(L, nvoxels):
    """ Classifies which PCA eigenvalues are related to noise and estimates the
    noise variance

    Parameters
    ----------
    L : array (n,)
        Array containing the PCA eigenvalues in ascending order.
    nvoxels : int
        Number of voxels used to compute L

    Returns
    -------
    var : float
        Estimation of the noise variance
    ncomps : int
        Number of eigenvalues related to noise

    Notes
    -----
    This is based on the algorithm described in [1]_.

    References
    ----------
    .. [1] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
           Fieremans E, 2016. Denoising of Diffusion MRI using random matrix
           theory. Neuroimage 142:394-406.
           doi: 10.1016/j.neuroimage.2016.08.016
    """

    # if num_samples - 1 (to correct for mean subtraction) is less than number
    # of features, discard the zero eigenvalues
    if L.size > nvoxels - 1:
        L = L[-(nvoxels - 1):]

    # Note that the condition expressed in the while-loop is expressed in terms
    # of the variance of equation (12), not equation (11) as in [1]_. Also,
    # this code implements ascending eigenvalues, unlike [1]_.
    var = np.mean(L)
    c = L.size - 1
    r = L[c] - L[0] - 4 * np.sqrt((c + 1.0) / nvoxels) * var
    while r > 0:
        var = np.mean(L[:c])
        c = c - 1
        r = L[c] - L[0] - 4 * np.sqrt((c + 1.0) / nvoxels) * var
    ncomps = c + 1

    return var, ncomps


def create_patch_radius_arr(arr, pr):
    """Create the patch radius array from the data to be denoised and the patch
    radius.

    Parameters
    ----------
    arr : ndarray
        Data to be denoised.
    pr : int or ndarray
        Patch radius.

    Returns
    -------
    patch_radius : ndarray
        Number of samples.
    """

    patch_radius = copy.deepcopy(pr)

    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    if len(patch_radius) != 3:
        raise ValueError("patch_radius should have length 3")
    else:
        patch_radius = np.asarray(patch_radius).astype(int)
    patch_radius[arr.shape[0:3] == np.ones(3)] = 0  # account for dim of size 1

    return patch_radius


def compute_patch_size(patch_radius):
    """Compute patch size from the patch radius.

    it is twice the radius plus one.

    Parameters
    ----------
    patch_radius : ndarray
        Patch radius.

    Returns
    -------
    int
        Number of samples.
    """

    return 2 * patch_radius + 1


def compute_num_samples(patch_size):
    """Compute the number of samples as the dot product of the elements.

    Parameters
    ----------
    patch_size : ndarray
        Patch size.

    Returns
    -------
    int
        Number of samples.
    """

    return np.prod(patch_size)


def compute_suggested_patch_radius(arr, patch_size):
    """Compute the suggested patch radius.

    Parameters
    ----------
    arr : ndarray
        Data to be denoised.
    patch_size : ndarray
        Patch size.

    Returns
    -------
    int
        Suggested patch radius.
    """

    tmp = np.sum(patch_size == 1)  # count spatial dimensions with size 1
    if tmp == 0:
        root = np.ceil(arr.shape[-1] ** (1. / 3))  # 3D
    if tmp == 1:
        root = np.ceil(arr.shape[-1] ** (1. / 2))  # 2D
    if tmp == 2:
        root = arr.shape[-1]  # 1D
    root = root + 1 if (root % 2) == 0 else root  # make odd

    return int((root - 1) / 2)


def genpca(arr, sigma=None, mask=None, patch_radius=2, pca_method='eig',
           tau_factor=None, return_sigma=False, out_dtype=None,
           suppress_warning=False):
    r"""General function to perform PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions. The first 3 dimension must have
        size >= 2 * patch_radius + 1 or size = 1.
    sigma : float or 3D array (optional)
        Standard deviation of the noise estimated from the data. If no sigma
        is given, this will be estimated based on random matrix theory
        [1]_,[2]_
    mask : 3D boolean array (optional)
        A mask with voxels that are true inside the brain and false outside of
        it. The function denoises within the true part and returns zeros
        outside of those voxels.
    patch_radius : int or 1D array (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). E.g. patch_radius=2 gives 5x5x5 patches.
    pca_method : 'eig' or 'svd' (optional)
        Use either eigenvalue decomposition (eig) or singular value
        decomposition (svd) for principal component analysis. The default
        method is 'eig' which is faster. However, occasionally 'svd' might be
        more accurate.
    tau_factor : float (optional)
        Thresholding of PCA eigenvalues is done by nulling out eigenvalues that
        are smaller than:

        .. math ::

                \tau = (\tau_{factor} \sigma)^2

        \tau_{factor} can be set to a predefined values (e.g. \tau_{factor} =
        2.3 [3]_), or automatically calculated using random matrix theory
        (in case that \tau_{factor} is set to None).
    return_sigma : bool (optional)
        If true, the Standard deviation of the noise will be returned.
    out_dtype : str or dtype (optional)
        The dtype for the output array. Default: output has the same dtype as
        the input.
    suppress_warning : bool (optional)
        If true, suppress warning caused by patch_size < arr.shape[-1].

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values.

    References
    ----------
    .. [1] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
           Fieremans E, 2016. Denoising of Diffusion MRI using random matrix
           theory. Neuroimage 142:394-406.
           doi: 10.1016/j.neuroimage.2016.08.016
    .. [2] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
           mapping using random matrix theory. Magnetic Resonance in Medicine.
           doi: 10.1002/mrm.26059.
    .. [3] Manjon JV, Coupe P, Concha L, Buades A, Collins DL (2013)
           Diffusion Weighted Image Denoising Using Overcomplete Local
           PCA. PLoS ONE 8(9): e73021.
           https://doi.org/10.1371/journal.pone.0073021
    """
    if mask is None:
        # If mask is not specified, use the whole volume
        mask = np.ones_like(arr, dtype=bool)[..., 0]

    if out_dtype is None:
        out_dtype = arr.dtype

    # We retain float64 precision, iff the input is in this precision:
    if arr.dtype == np.float64:
        calc_dtype = np.float64
    # Otherwise, we'll calculate things in float32 (saving memory)
    else:
        calc_dtype = np.float32

    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.",
                         arr.shape)

    if pca_method.lower() == 'svd':
        is_svd = True
    elif pca_method.lower() == 'eig':
        is_svd = False
    else:
        raise ValueError("pca_method should be either 'eig' or 'svd'")

    patch_radius_arr = create_patch_radius_arr(arr, patch_radius)
    patch_size = compute_patch_size(patch_radius_arr)

    ash = arr.shape[0:3]
    if np.any((ash != np.ones(3)) * (ash < patch_size)):
        raise ValueError("Array 'arr' is incorrect shape")

    num_samples = compute_num_samples(patch_size)
    if num_samples == 1:
        raise ValueError("Cannot have only 1 sample,\
                          please increase patch_radius.")
    # account for mean subtraction by testing #samples - 1
    if (num_samples - 1) < arr.shape[-1] and not suppress_warning:
        spr = compute_suggested_patch_radius(arr, patch_size)
        warn(dimensionality_problem_message(arr, num_samples, spr),
             UserWarning)

    if isinstance(sigma, np.ndarray):
        var = sigma ** 2
        if not sigma.shape == arr.shape[:-1]:
            e_s = "You provided a sigma array with a shape"
            e_s += "{0} for data with".format(sigma.shape)
            e_s += "shape {0}. Please provide a sigma array".format(arr.shape)
            e_s += " that matches the spatial dimensions of the data."
            raise ValueError(e_s)
    elif isinstance(sigma, (int, float)):
        var = sigma ** 2 * np.ones(arr.shape[:-1])

    dim = arr.shape[-1]
    if tau_factor is None:
        tau_factor = 1 + np.sqrt(dim / num_samples)

    theta = np.zeros(arr.shape, dtype=calc_dtype)
    thetax = np.zeros(arr.shape, dtype=calc_dtype)

    if return_sigma is True and sigma is None:
        var = np.zeros(arr.shape[:-1], dtype=calc_dtype)
        thetavar = np.zeros(arr.shape[:-1], dtype=calc_dtype)

    # loop around and find the 3D patch for each direction at each pixel
    for k in range(patch_radius_arr[2], arr.shape[2] - patch_radius_arr[2]):
        for j in range(patch_radius_arr[1],
                       arr.shape[1] - patch_radius_arr[1]):
            for i in range(
                    patch_radius_arr[0], arr.shape[0] - patch_radius_arr[0]):
                # Shorthand for indexing variables:
                if not mask[i, j, k]:
                    continue
                ix1 = i - patch_radius_arr[0]
                ix2 = i + patch_radius_arr[0] + 1
                jx1 = j - patch_radius_arr[1]
                jx2 = j + patch_radius_arr[1] + 1
                kx1 = k - patch_radius_arr[2]
                kx2 = k + patch_radius_arr[2] + 1

                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(
                                num_samples, dim)
                # compute the mean
                M = np.mean(X, axis=0)
                # Upcast the dtype for precision in the SVD
                X = X - M

                if is_svd:
                    # PCA using an SVD
                    svd_args = [1, 0]
                    U, S, Vt = svd(X, *svd_args)[:3]
                    # Items in S are the eigenvalues, but in ascending order
                    # We invert the order (=> descending), square and normalize
                    # \lambda_i = s_i^2 / n
                    d = S[::-1] ** 2 / X.shape[0]
                    # Rows of Vt are eigenvectors, but also in ascending
                    # eigenvalue order:
                    W = Vt[::-1].T

                else:
                    # PCA using an Eigenvalue decomposition
                    C = np.transpose(X).dot(X)
                    C = C / X.shape[0]
                    [d, W] = eigh(C)

                if sigma is None:
                    # Random matrix theory
                    this_var, _ = _pca_classifier(d, num_samples)
                else:
                    # Predefined variance
                    this_var = var[i, j, k]

                # Threshold by tau:
                tau = tau_factor ** 2 * this_var

                # Update ncomps according to tau_factor
                ncomps = np.sum(d < tau)
                W[:, :ncomps] = 0

                # This is equations 1 and 2 in Manjon 2013:
                Xest = X.dot(W).dot(W.T) + M
                Xest = Xest.reshape(patch_size[0],
                                    patch_size[1],
                                    patch_size[2], dim)
                # This is equation 3 in Manjon 2013:
                this_theta = 1.0 / (1.0 + dim - ncomps)
                theta[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                thetax[ix1:ix2, jx1:jx2, kx1:kx2] += Xest * this_theta
                if return_sigma is True and sigma is None:
                    var[ix1:ix2, jx1:jx2, kx1:kx2] += this_var * this_theta
                    thetavar[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta

    denoised_arr = thetax / theta
    denoised_arr.clip(min=0, out=denoised_arr)
    denoised_arr[mask == 0] = 0
    if return_sigma is True:
        if sigma is None:
            var = var / thetavar
            var[mask == 0] = 0
            return denoised_arr.astype(out_dtype), np.sqrt(var)
        else:
            return denoised_arr.astype(out_dtype), sigma
    else:
        return denoised_arr.astype(out_dtype)


def localpca(arr, sigma=None, mask=None, patch_radius=2, gtab=None,
             patch_radius_sigma=1, pca_method='eig', tau_factor=2.3,
             return_sigma=False, correct_bias=True, out_dtype=None,
             suppress_warning=False):
    r""" Performs local PCA denoising according to Manjon et al. [1]_.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.
    sigma : float or 3D array (optional)
        Standard deviation of the noise estimated from the data. If not given,
        calculate using method in [1]_.
    mask : 3D boolean array (optional)
        A mask with voxels that are true inside the brain and false outside of
        it. The function denoises within the true part and returns zeros
        outside of those voxels.
    patch_radius : int or 1D array (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). E.g. patch_radius=2 gives 5x5x5 patches.
    gtab: gradient table object (optional if sigma is provided)
        gradient information for the data gives us the bvals and bvecs of
        diffusion data, which is needed to calculate noise level if sigma is
        not provided.
    patch_radius_sigma : int (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels) for estimating sigma. E.g. patch_radius_sigma=2 gives
        5x5x5 patches.
    pca_method : 'eig' or 'svd' (optional)
        Use either eigenvalue decomposition (eig) or singular value
        decomposition (svd) for principal component analysis. The default
        method is 'eig' which is faster. However, occasionally 'svd' might be
        more accurate.
    tau_factor : float (optional)
        Thresholding of PCA eigenvalues is done by nulling out eigenvalues that
        are smaller than:

        .. math ::

                \tau = (\tau_{factor} \sigma)^2

        \tau_{factor} can be change to adjust the relationship between the
        noise standard deviation and the threshold \tau. If \tau_{factor} is
        set to None, it will be automatically calculated using the
        Marcenko-Pastur distribution [2]_. Default: 2.3 according to [1]_.
    return_sigma : bool (optional)
        If true, a noise standard deviation estimate based on the
        Marcenko-Pastur distribution is returned [2]_.
    correct_bias : bool (optional)
        Whether to correct for bias due to Rician noise. This is an
        implementation of equation 8 in [1]_.
    out_dtype : str or dtype (optional)
        The dtype for the output array. Default: output has the same dtype as
        the input.
    suppress_warning : bool (optional)
        If true, suppress warning caused by patch_size < arr.shape[-1].

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values

    References
    ----------
    .. [1] Manjon JV, Coupe P, Concha L, Buades A, Collins DL (2013)
           Diffusion Weighted Image Denoising Using Overcomplete Local
           PCA. PLoS ONE 8(9): e73021.
           https://doi.org/10.1371/journal.pone.0073021
    .. [2] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
           Fieremans E, 2016. Denoising of Diffusion MRI using random matrix
           theory. Neuroimage 142:394-406.
           doi: 10.1016/j.neuroimage.2016.08.016
    """
    # check gtab is given, if sigma is not given
    if sigma is None and gtab is None:
        raise ValueError("gtab must be provided if sigma is not given")

    # calculate sigma
    if sigma is None:
        sigma = pca_noise_estimate(arr, gtab,
                                   correct_bias=correct_bias,
                                   patch_radius=patch_radius_sigma,
                                   images_as_samples=True)

    return genpca(arr, sigma=sigma, mask=mask, patch_radius=patch_radius,
                  pca_method=pca_method, tau_factor=tau_factor,
                  return_sigma=return_sigma, out_dtype=out_dtype,
                  suppress_warning=suppress_warning)


def mppca(arr, mask=None, patch_radius=2, pca_method='eig',
          return_sigma=False, out_dtype=None, suppress_warning=False):
    r"""Performs PCA-based denoising using the Marcenko-Pastur
    distribution [1]_.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.
    mask : 3D boolean array (optional)
        A mask with voxels that are true inside the brain and false outside of
        it. The function denoises within the true part and returns zeros
        outside of those voxels.
    patch_radius : int or 1D array (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). E.g. patch_radius=2 gives 5x5x5 patches.
    pca_method : 'eig' or 'svd' (optional)
        Use either eigenvalue decomposition (eig) or singular value
        decomposition (svd) for principal component analysis. The default
        method is 'eig' which is faster. However, occasionally 'svd' might be
        more accurate.
    return_sigma : bool (optional)
        If true, a noise standard deviation estimate based on the
        Marcenko-Pastur distribution is returned [2]_.
    out_dtype : str or dtype (optional)
        The dtype for the output array. Default: output has the same dtype as
        the input.
    suppress_warning : bool (optional)
        If true, suppress warning caused by patch_size < arr.shape[-1].

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values
    sigma : 3D array (when return_sigma=True)
        Estimate of the spatial varying standard deviation of the noise

    References
    ----------
    .. [1] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
           Fieremans E, 2016. Denoising of Diffusion MRI using random matrix
           theory. Neuroimage 142:394-406.
           doi: 10.1016/j.neuroimage.2016.08.016
    .. [2] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
           mapping using random matrix theory. Magnetic Resonance in Medicine.
           doi: 10.1002/mrm.26059.
    """
    return genpca(arr, sigma=None, mask=mask, patch_radius=patch_radius,
                  pca_method=pca_method, tau_factor=None,
                  return_sigma=return_sigma, out_dtype=out_dtype,
                  suppress_warning=suppress_warning)
