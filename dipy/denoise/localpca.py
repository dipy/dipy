import numpy as np
import scipy as sp
from dipy.denoise.fast_lpca import fast_lpca

def localpca(arr, sigma, patch_radius=1):

    """
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
    [1] Manjon JV, Coupe P, Concha L, Buades A, Collins DL
        "Diffusion Weighted Image Denoising Using Overcomplete Local PCA"
        PLOS 2013
    
    """

    if arr.ndim == 4:
        # warning for memory considerations
        size = 2 * arr.shape[1] * arr.shape[2] * arr.shape[3] * arr.shape[3]
        memory_use = 8.0 * size / (1024.0 * 1024.0)
        if memory_use > 2000:
            print("Memory Usage", memory_use , "mb")
            print("Use the slower version of local PCA to avoid problems")
            call = raw_input("Do you want to continue? y/n :")
            if call == 'y' or call == 'Y':
                if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

                    sigma = (np.ones(arr.shape, dtype=np.float64)
                             * sigma[..., np.newaxis])            
                else:
                    sigma = np.ones(arr.shape, dtype=np.float64) * sigma
                
                denoised_arr = np.array(fast_lpca(arr.astype(np.float64), patch_radius, sigma))
                denoised_arr = np.abs(denoised_arr)
                # # phi = np.linspace(0,15,1000)
                # # # # we need to find the index of the closest value of arr/sigma from the dataset
                # # eta_phi = np.sqrt(np.pi/2) * np.exp(-0.5 * phi**2) * (((1 + 0.5 * phi**2) * sp.special.iv(0,0.25 * phi**2) + (0.5 * phi**2) * sp.special.iv(1,0.25 * phi**2))**2)
                # # # # eta_phi = eta_phi[1:200]
                # # corrected_arr = np.zeros_like(denoised_arr)
                # # phi = np.abs(denoised_arr / np.sqrt(sigma))
                # # phi[np.isnan(phi)] = 0
                # # opt_diff = np.abs(phi - eta_phi[0])
                # # for i in range(eta_phi.size):
                # #     print(i)
                # #     if(i!=0):
                # #         new_diff = np.abs(phi - eta_phi[i])
                # #         corrected_arr[new_diff < opt_diff] = i
                # #         opt_diff[new_diff<opt_diff] = new_diff[new_diff<opt_diff]

                # # corrected_arr = np.sqrt(sigma) * corrected_arr * 15.0/1000.0
                return denoised_arr.astype(arr.dtype)

            else:
                raise ValueError("Aborted!")
        else:
            if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

                sigma = (np.ones(arr.shape, dtype=np.float64)
                         * sigma[..., np.newaxis])            
            else:
                sigma = np.ones(arr.shape, dtype=np.float64) * sigma
                
            denoised_arr = np.array(fast_lpca(arr.astype(np.float64), patch_radius, sigma))
            denoised_arr = np.abs(denoised_arr)
            # # phi = np.linspace(0,15,1000)
            # # # # we need to find the index of the closest value of arr/sigma from the dataset
            # # eta_phi = np.sqrt(np.pi/2) * np.exp(-0.5 * phi**2) * (((1 + 0.5 * phi**2) * sp.special.iv(0,0.25 * phi**2) + (0.5 * phi**2) * sp.special.iv(1,0.25 * phi**2))**2)
            # # # # eta_phi = eta_phi[1:200]
            # # corrected_arr = np.zeros_like(denoised_arr)
            # # phi = np.abs(denoised_arr / np.sqrt(sigma))
            # # phi[np.isnan(phi)] = 0
            # # opt_diff = np.abs(phi - eta_phi[0])
            # # for i in range(eta_phi.size):
            # #     print(i)
            # #     if(i!=0):
            # #         new_diff = np.abs(phi - eta_phi[i])
            # #         corrected_arr[new_diff < opt_diff] = i
            # #         opt_diff[new_diff<opt_diff] = new_diff[new_diff<opt_diff]

            # # corrected_arr = np.sqrt(sigma) * corrected_arr * 15.0/1000.0
            return denoised_arr.astype(arr.dtype)

    else:
        raise ValueError("Only 4D array are supported!", arr.shape)
