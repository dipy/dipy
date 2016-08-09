import numpy as np
import scipy as sp
from dipy.denoise.fast_lpca import fast_lpca


def localpca(arr, sigma, patch_radius=1):
    r"""
    Local PCA Based Denoising of Diffusion Datasets

    Parameters
    ----------
    arr : A 4D array which is to be denoised
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : The radius of the local patch to
                be taken around each voxel

    Returns
    -------

    denoised_arr : 4D array
        this is the denoised array of the same size as that of
        arr (input data)

    References
    ----------
    .. [Manjon13] Manjon JV, Coupe P, Concha L, Buades A, Collins DL
                  "Diffusion Weighted Image Denoising Using Overcomplete Local PCA"
                  PLOS 2013

    """

    if arr.ndim == 4:
        # warning for memory considerations
        size = 2 * arr.shape[1] * arr.shape[2] * arr.shape[3] * arr.shape[3]
        memory_use = 8.0 * size / (1024.0 * 1024.0)
        if memory_use > 2000:
            print("Memory Usage", memory_use, "mb")
            print("Use the slower version of local PCA to avoid problems")
            call = raw_input("Do you want to continue? y/n :")
            if call == 'y' or call == 'Y':
                if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

                    sigma = np.array(sigma, dtype=np.float64)
                else:
                    sigma = np.ones(
                        (arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float64) * sigma

                denoised_arr = np.array(fast_lpca(arr.astype(np.float64),
                            patch_radius,sigma))
                denoised_arr = np.abs(denoised_arr)
                
                return denoised_arr.astype(arr.dtype)

            else:
                raise ValueError("Aborted!")
        else:
            if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

                sigma = np.array(sigma, dtype=np.float64)
            else:
                sigma = np.ones(
                    (arr.shape[0],
                     arr.shape[1],
                        arr.shape[2]),
                    dtype=np.float64) * sigma

            denoised_arr = np.array(fast_lpca(arr.astype(np.float64),
                            patch_radius,sigma))
            denoised_arr = np.abs(denoised_arr)
            
            return denoised_arr.astype(arr.dtype)

    else:
        raise ValueError("Only 4D array are supported!", arr.shape)
