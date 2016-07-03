import numpy as np
import scipy as sp


def localpca_slow(arr, sigma, patch_radius=1, rician=True):
    """Local PCA Based Denoising of Diffusion Datasets

    Parameters
    ----------
    arr : A 4D array which is to be denoised
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : The radius of the local patch to
                be taken around each voxel
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------

    denoised_arr : 4D array
        this is the denoised array of the same size as that of
        arr (input data)

    References
    ----------
    Diffusion Weighted Image Denoising Using Overcomplete Local PCA
    Manjon JV, Coupe P, Concha L, Buades A, Collins DL

    """

    if arr.ndim == 4:

        tou = 2.3 * 2.3 * sigma

        if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

            sigma = (np.ones(arr.shape, dtype=np.float64)
                     * sigma[..., np.newaxis])
            tou = (np.ones(arr.shape, dtype=np.float64) * tou[..., np.newaxis])
        else:
            sigma = np.ones(arr.shape, dtype=np.float64) * sigma
            tou = np.ones(arr.shape, dtype=np.float64) * tou

        # loop around and find the 3D patch for each direction at each pixel

        # declare arrays for theta and thetax
        theta = np.zeros(arr.shape, dtype=np.float64)
        thetax = np.zeros(arr.shape, dtype=np.float64)

        patch_size = 2 * patch_radius + 1

        for k in range(patch_radius, arr.shape[2] - patch_radius, 1):
            for j in range(patch_radius, arr.shape[1] - patch_radius, 1):
                for i in range(patch_radius, arr.shape[0] - patch_radius, 1):

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
                        arr.shape[3])
                    # compute the mean and normalize
                    M = np.mean(X, axis=0)
                    X = X - np.array([M, ] * X.shape[0],
                                     dtype=np.float64)

                    # Compute the covariance matrix C = X_transpose X
                    C = np.transpose(X).dot(X)
                    C = C / X.shape[0]
                    # compute EVD of the covariance matrix of X get the matrices W and D, hence get matrix Y = XW
                    # Threshold matrix D and then compute X_est = YW_transpose
                    # D_est
                    [d, W] = np.linalg.eigh(C)

                    d[d < tou[i, j, k, :]] = 0
                    
                    W_hat = W * 0
                    W_hat[:, d > 0] = W[:, d > 0]
                    Y = X.dot(W_hat)
                    X_est = Y.dot(np.transpose(W_hat))
                    # When the block covers each pixel identify it into the
                    # label matrix theta

                    # generate a theta matrix for patch around i,j,k and store it's
                    # theta value
                    temp = X_est + \
                        np.array([M, ] * X_est.shape[0], dtype=np.float64)
                    temp = temp.reshape(
                        patch_size, patch_size, patch_size, arr.shape[3])
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

        # the final denoised without rician adaptation
        denoised_arr = thetax / theta
        # phi = np.linspace(0,15,1000)
        # # # we need to find the index of the closest value of arr/sigma from the dataset
        # eta_phi = np.sqrt(np.pi/2) * np.exp(-0.5 * phi**2) * (((1 + 0.5 * phi**2) * sp.special.iv(0,0.25 * phi**2) + (0.5 * phi**2) * sp.special.iv(1,0.25 * phi**2))**2)
        # # # eta_phi = eta_phi[1:200]
        # corrected_arr = np.zeros_like(denoised_arr)
        # phi = np.abs(denoised_arr / np.sqrt(sigma))
        # phi[np.isnan(phi)] = 0
        # opt_diff = np.abs(phi - eta_phi[0])
        # for i in range(eta_phi.size):
        #     print(i)
        #     if(i!=0):
        #         new_diff = np.abs(phi - eta_phi[i])
        #         corrected_arr[new_diff < opt_diff] = i
        #         opt_diff[new_diff<opt_diff] = new_diff[new_diff<opt_diff]

        # corrected_arr = np.sqrt(sigma) * corrected_arr * 15.0/1000.0
        return denoised_arr.astype(arr.dtype)

    else:
        raise ValueError("Only 4D array are supported!", arr.shape)
