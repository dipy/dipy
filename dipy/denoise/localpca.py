import numpy as np
import scipy as sp
import scipy.linalg as sla
from warnings import warn

# Try to get the SVD through direct API to lapack:
try:
    from scipy.linalg.lapack import sgesvd as svd
    svd_args = [1, 0]
# If you have an older version of scipy, we fall back
# on the standard scipy SVD API:
except ImportError:
    from scipy.linalg import svd
    svd_args = [False]


def localpca(arr, sigma, patch_radius=2):
    r"""Local PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the different diffusion gradient directions.
    sigma : float or 3D array
        Standard deviation of the noise estimated from the data.
    patch_radius : int
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 2 (denoise in blocks of 5x5x5 voxels).

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data.

    References
    ----------
    .. [Manjon13] Manjon JV, Coupe P, Concha L, Buades A, Collins DL (2013)
                  Diffusion Weighted Image Denoising Using Overcomplete Local
                  PCA. PLoS ONE 8(9): e73021.
                  https://doi.org/10.1371/journal.pone.0073021
    """
    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.",
                         arr.shape)

    patch_size = 2 * patch_radius + 1

    if patch_size ** 3 < arr.shape[-1]:
        e_s = "You asked for PCA denoising with a patch_radius of {0} "
        e_s += "for data with {1} directions. This would result in an "
        e_s += "ill-conditioned PCA matrix. Please increase the patch_radius."
        e_s.format(patch_radius, arr.shape[-1])
        raise ValueError(e_s)

    tau = 2.3 * 2.3 * sigma * sigma
    if isinstance(sigma, np.ndarray):
        if not sigma.shape == arr.shape[:-1]:
            e_s = "You provided a sigma array with a shape {0} for data with "
            e_s = "shape {1}. Please provide a sigma array that matches the "
            e_s = "spatial dimensions of the data."
            e_s.format(sigma.shape, arr.shape)
            raise ValueError(e_s)
        sigma = (np.ones(arr.shape, dtype=np.float64) *
                 sigma[..., np.newaxis])
    tau = np.ones(arr.shape[:-1], dtype=np.float64) * tau

    # loop around and find the 3D patch for each direction at each pixel

    # declare arrays for theta and thetax
    theta = np.zeros(arr.shape, dtype=np.float64)
    thetax = np.zeros(arr.shape, dtype=np.float64)

    for k in range(patch_radius, arr.shape[2] - patch_radius):
        for j in range(patch_radius, arr.shape[1] - patch_radius):
            for i in range(patch_radius, arr.shape[0] - patch_radius):

                X = np.zeros(
                    (patch_size * patch_size * patch_size, arr.shape[3]))
                M = np.zeros(arr.shape[3])

                temp = arr[i - patch_radius: i + patch_radius + 1,
                           j - patch_radius: j + patch_radius + 1,
                           k - patch_radius: k + patch_radius + 1, :]
                X = temp.reshape(
                    patch_size *
                    patch_size *
                    patch_size,
                    arr.shape[-1])
                # compute the mean and normalize
                M = np.mean(X, axis=0)
                X = X - M
                # PCA. There are two ways to do this:
                # # 1. Compute the covariance matrix C = X_transpose X
                # C = np.dot(X.T, X)
                # C = C / X.shape[0]
                # # and then decompose it:
                # d, W = sla.eigh(C)

                # 2. Alternatively, calculate the eigenvalues and
                #  eigenvectors of the covariance matrix through an SVD:
                U, S, Vt = svd(X, *svd_args)[:3]
                # Items in S are the eigenvalues, but in ascending order
                # We invert the order (=> descending), square and normalize
                # \lambda_i = s_i^2 / n
                d = S[::-1] ** 2 / X.shape[0]
                # Rows of Vt are eigenvectors, but also in ascending eigenvalue
                # order:
                W = Vt[::-1].T
                d[d < tau[i, j, k]] = 0
                W_hat = np.zeros_like(W)
                W_hat[:, d > 0] = W[:, d > 0]
                Y = X.dot(W_hat)
                X_est = Y.dot(np.transpose(W_hat))

                # theta value
                temp = X_est + \
                    np.array([M, ] * X_est.shape[0], dtype=np.float64)
                temp = temp.reshape(
                    patch_size, patch_size, patch_size, arr.shape[-1])
                # Also update the estimate matrix which is X_est * theta

                theta[i - patch_radius: i + patch_radius + 1,
                      j - patch_radius: j + patch_radius + 1,
                      k - patch_radius: k + patch_radius + 1,
                      :] = theta[i - patch_radius: i + patch_radius + 1,
                                 j - patch_radius: j + patch_radius + 1,
                                 k - patch_radius: k + patch_radius + 1,
                                 :] + 1.0 / (1.0 + np.linalg.norm(d,
                                                                  ord=0))

                thetax[i - patch_radius: i + patch_radius + 1,
                       j - patch_radius: j + patch_radius + 1,
                       k - patch_radius: k + patch_radius + 1,
                       :] = thetax[i - patch_radius: i + patch_radius + 1,
                                   j - patch_radius: j + patch_radius + 1,
                                   k - patch_radius: k + patch_radius + 1,
                                   :] + temp / (1 + np.linalg.norm(d,
                                                                   ord=0))
    denoised_arr = thetax / theta
    return denoised_arr.astype(arr.dtype)
