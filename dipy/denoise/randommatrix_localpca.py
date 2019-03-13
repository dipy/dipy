import numpy as np
cimport numpy as cnp
cimport cython

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp

from cython.parallel import prange
from libc.stdlib cimport malloc, free
from multiprocessing import cpu_count


try:
    from scipy.linalg.lapack import dgesvd as svd
    svd_args = [1, 0]
    # If you have an older version of scipy, we fall back
    # on the standard scipy SVD API:
except ImportError:
    from scipy.linalg import svd
    svd_args = [False]
from scipy.linalg import eigh


def randommatrix_localpca(arr, patch_extent=0, out_dtype=None):
    r"""Local PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.   
    patch_extent : int, optional
        The diameter of the local patch to be taken around each voxel (in
        voxels). The radius will be half of this value. If not provided,
        the default will be automatically computed as:

        .. math ::

                patch_extent = max(5,\lfloor N^{1/3} \rfloor)

    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values
    noise_arr : 3D array
        Voxelwise standard deviation of the noise estimated from the data.
    sigma : float
        Mean value of noise standard deviations over all voxels (mean of 
        noise_arr).

    References    
    ----------
    .. [Veraart16] Veraart J, Fieremans E, Novikov DS (2016)
                  Diffusion MRI noise mapping using random matrix theory.
                  Magnetic resonance in Medicine 76(5), p1582-1593.
                  https://doi.org/10.1002/mrm.26059
    """

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

    if patch_extent <= 0:
        Nvols = arr.shape[:-1]
        patch_extent = max(5,Nvols ** (1. / 3.)) 
  
    path_radius=int(math.floor(patch_extent/2.))
  
    noise_arr= np.zeros([arr.shape[0],[arr.shape[1],[arr.shape[2]], dtype=calc_dtype)

        



    # loop around and find the 3D patch for each direction at each pixel
    for k in range(patch_radius, arr.shape[2] - patch_radius):
        for j in range(patch_radius, arr.shape[1] - patch_radius):
            for i in range(patch_radius, arr.shape[0] - patch_radius):
                # Shorthand for indexing variables:
                if not mask[i, j, k]:
                    continue
                ix1 = i - patch_radius
                ix2 = i + patch_radius + 1
                jx1 = j - patch_radius
                jx2 = j + patch_radius + 1
                kx1 = k - patch_radius
                kx2 = k + patch_radius + 1

                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(
                                patch_size ** 3, arr.shape[-1])
                # compute the mean and normalize
                M = np.mean(X, axis=0)
                # Upcast the dtype for precision in the SVD
                X = X - M

                if is_svd:
                    # PCA using an SVD
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
                    [d, W] = eigh(C, turbo=True)

                # Threshold by tau:
                W[:, d < tau] = 0
                # This is equations 1 and 2 in Manjon 2013:
                Xest = X.dot(W).dot(W.T) + M
                Xest = Xest.reshape(patch_size,
                                    patch_size,
                                    patch_size, arr.shape[-1])
                # This is equation 3 in Manjon 2013:
                this_theta = 1.0 / (1.0 + np.sum(d > 0))
                theta[ix1:ix2, jx1:jx2, kx1:kx2] += this_theta
                thetax[ix1:ix2, jx1:jx2, kx1:kx2] += Xest * this_theta

    denoised_arr = thetax / theta
    denoised_arr.clip(min=0, out=denoised_arr)
    denoised_arr[~mask] = 0
    return denoised_arr.astype(out_dtype)
