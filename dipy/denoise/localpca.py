import numpy as np
import scipy as sp

def localpca(arr, sigma, patch_radius=1, tou=0, rician=True):
    '''
    Local PCA Based Denoising of Diffusion Datasets

    References
    ----------
    Diffusion Weighted Image Denoising Using Overcomplete Local PCA
    Manjon JV, Coupe P, Concha L, Buades A, Collins DL

    Parameters
    ----------
    arr : A 4D array which is to be denoised
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : The radius of the local patch to
                be taken around each voxel
    tou : float or 3D array
        threshold parameter for diagonal matrix thresholding,
        default value = (2.3 * sigma * sigma)
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------

    denoised_arr : 4D array
        this is the denoised array of the same size as that of
        arr (input data) 

    '''

    # Read the array and fix the dimensions

    if arr.ndim == 4:
        sigma = np.zeros(arr[...,0].shape,dtype = np.float64)

        if tou == 0:
            tou = 2.3 * 2.3 * sigma 

        if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

            sigma = (np.ones(arr.shape, dtype=np.float64) * sigma[..., np.newaxis])
            tou = (np.ones(arr.shape, dtype=np.float64) * tou[..., np.newaxis])
        else:
            sigma = np.ones(arr.shape, dtype=np.float64) * sigma
            tou = np.ones(arr.shape, dtype=np.float64) * tou    
        
        # loop around and find the 3D patch for each direction at each pixel
        
        # declare arrays for theta and thetax
        theta = np.zeros(arr.shape,dtype = np.float64)
        thetax = np.zeros(arr.shape,dtype = np.float64)

        patch_size = 2 * patch_radius + 1

        for k in range(patch_radius, arr.shape[2] - patch_radius, 1):
            print(k)
            for j in range(patch_radius, arr.shape[1] - patch_radius, 1):
                for i in range(patch_radius, arr.shape[0] - patch_radius , 1):
                    
                    X = np.zeros((patch_size * patch_size * patch_size, arr.shape[3]))
                    M = np.zeros(arr.shape[3])

                    temp = arr[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                         k - patch_radius : k + patch_radius + 1,:]
                    X = temp.reshape(patch_size * patch_size * patch_size, arr.shape[3])
                    # compute the mean and normalize
                    M = np.mean(X,axis=1)
                    X = X - np.array([M,]*X.shape[1],dtype=np.float64).transpose()
                        
                    # using the PCA trick
                    # Compute the covariance matrix C = X_transpose X
                    C = np.transpose(X).dot(X)
                    C = C/arr.shape[3]
                    # compute EVD of the covariance matrix of X get the matrices W and D, hence get matrix Y = XW
                    # Threshold matrix D and then compute X_est = YW_transpose D_est
                    [d,W] = np.linalg.eigh(C)
                    # for sigma estimate we perform the median estimation

                    # find the median of the standard deviation of the eigenvalues
                    # median_sqrt = np.median(np.sqrt(d[d > alpha]))
                    # if(np.isnan(median_sqrt)):
                    #     median_sqrt = 0
                    # # Chop of the positive eigenvalues whose standard deviation is more that 2 times that of the above quantity
                    # # Take the remaining eigenvalues and estimate sigma
                    # sigma[i,j,k] = beta * beta * np.median(d[np.sqrt(d[d > alpha]) < 2 * median_sqrt])
                    # if(np.isnan(sigma[i,j,k])):
                    #     sigma[i,j,k] = 0

                    d[d < sigma[i,j,k]] = 0
                    d[d>0] = 1
                    D_hat = np.diag(d)
                    
                    Y = X.dot(W)
                    # When the block covers each pixel identify it into the label matrix theta
                    X_est = Y.dot(np.transpose(W))
                    X_est = X_est.dot(D_hat)

                    temp = X_est + np.array([M,]*X_est.shape[1], dtype = np.float64).transpose()
                    temp = temp.reshape(patch_size, patch_size, patch_size, arr.shape[3])
                    # Also update the estimate matrix which is X_est * theta
                        
                    theta[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                         k - patch_radius : k + patch_radius + 1 ,:] = theta[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                         k - patch_radius : k + patch_radius + 1 ,:] + 1/(1 + np.linalg.norm(d,ord=0))

                    thetax[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                         k - patch_radius : k + patch_radius + 1 ,:] = thetax[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                         k - patch_radius : k + patch_radius + 1 ,:] + temp / (1 + np.linalg.norm(d,ord=0))


        # the final denoised without rician adaptation
        denoised_arr = thetax / theta

        # Compute the lookup table
        phi = np.linspace(0,15,1000)
        # we need to find the index of the closest value of arr/sigma from the dataset
        eta_phi = np.sqrt(np.pi/2) * np.exp(-0.5 * phi**2) * (((1 + 0.5 * phi**2) * sp.special.iv(0,0.25 * phi**2) + (0.5 * phi**2) * sp.special.iv(1,0.25 * phi**2))**2)
        corrected_arr = np.zeros_like(denoised_arr)
        y = denoised_arr / np.sqrt(sigma)
        opt_diff = np.abs(y - eta_phi[0])
        for i in range(eta_phi.size):
            print(i)
            if(i!=0):
                new_diff = np.abs(y - eta_phi[i])
                corrected_arr[new_diff < opt_diff] = i
                opt_diff[new_diff<opt_diff] = new_diff[new_diff<opt_diff]

        corrected_arr = np.sqrt(sigma) * corrected_arr * 15.0/1000.0
        # After estimation pass it through a function ~ rician adaptation

        return corrected_arr

    else:
        raise ValueError("Only 4D array are supported!", arr.shape)
