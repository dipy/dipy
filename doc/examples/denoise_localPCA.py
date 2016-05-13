"""
===============================
Denoise images using Local PCA 
===============================

"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.localPCA_denoise import localPCA_denoise
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell

fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()

data = img.get_data()
affine = img.get_affine()

# currently just taking the small patch of the data to preserv time

data = data[0:50,0:50,0:50,0:10]
sigma = estimate_sigma(data, N=4)
arr = data
tou = 0
patch_radius = 1
# out = localPCA_denoise(data,sigma)
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

        for k in range(patch_radius, arr.shape[2] - patch_radius , 1):
            print k
            for j in range(patch_radius, arr.shape[1] - patch_radius , 1):
                for i in range(patch_radius, arr.shape[0] - patch_radius , 1):
                    
                    X = np.zeros((patch_size * patch_size * patch_size, arr.shape[3]))
                    M = np.zeros(arr.shape[3])
                    SD = np.zeros(arr.shape[3])
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
                        temp = (X_est[:,l] + M[l])
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
        # denoised_arr = rician_adaptation(denoised_arr,sigma)

        # return denoised_arr
before = data[:,:,19,1]
after = denoised_arr[:,:,19,1]
difference = np.abs(after.astype('f8') - before.astype('f8'))
fig, ax = plt.subplots(1, 3)
ax[0].imshow(before, cmap='gray', origin='lower')
ax[0].set_title('before')
ax[1].imshow(after, cmap='gray', origin='lower')
ax[1].set_title('after')
ax[2].imshow(difference, cmap='gray', origin='lower')
ax[2].set_title('difference')

for i in range(3):
    ax[i].set_axis_off()

plt.show()
