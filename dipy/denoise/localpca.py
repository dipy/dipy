import numpy as np

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

        if tou == 0:
            tou = 2.3 * sigma * sigma

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

                    for l in range(0, arr.shape[3], 1):
                        
                        # create the matrix X and normalize it
                        temp = arr[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                             k - patch_radius : k + patch_radius + 1,l]
                        temp = temp.reshape(patch_size * patch_size * patch_size)
                        X[:,l] = temp
                        # compute the mean and normalize
                        M[l] = np.mean(X[:,l])
                        X[:,l] = (X[:,l] - M[l])

                    # Compute the covariance matrix C = X_transpose X
                    C = np.transpose(X).dot(X)
                    C = C/arr.shape[3]
                    # compute EVD of the covariance matrix of X get the matrices W and D, hence get matrix Y = XW
                    # Threshold matrix D and then compute X_est = YW_transpose D_est
                    [d,W] = np.linalg.eigh(C)
                    d[d < tou[i][j][k][0]] = 0
                    D_hat = np.diag(d)
                    Y = X.dot(W)
                    # When the block covers each pixel identify it into the label matrix theta
                    X_est = Y.dot(np.transpose(W))
                    X_est = X_est.dot(D_hat)

                    for l in range(0,arr.shape[3],1):
                        # generate a theta matrix for patch around i,j,k and store it's theta value
                        # Also update the estimate matrix which is X_est * theta
                        temp = X_est[:,l] + M[l]
                        temp = temp.reshape(patch_size , patch_size , patch_size)
                        theta[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                             k - patch_radius : k + patch_radius + 1 ,l] = theta[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                             k - patch_radius : k + patch_radius + 1 ,l] + 1/(1 + np.linalg.norm(d,ord=0))

                        thetax[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                             k - patch_radius : k + patch_radius + 1 ,l] = thetax[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                             k - patch_radius : k + patch_radius + 1 ,l] + temp / (1 + np.linalg.norm(d,ord=0))

        # the final denoised without rician adaptation
        denoised_arr = thetax / theta
        
        # After estimation pass it through a function ~ rician adaptation

        return denoised_arr

    else:
        raise ValueError("Only 4D array are supported!", arr.shape)
    



