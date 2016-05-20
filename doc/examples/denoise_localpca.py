"""
===============================
Denoise images using Local PCA 
===============================

"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.localpca import localpca
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.noise_estimate_localpca import estimate_sigma_localpca
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()

# fetch_stanford_hardi()
# img, gtab = read_stanford_hardi()


data = img.get_data()
affine = img.get_affine()

# currently just taking the small patch of the data to preserv time

data = data[:,:,10:32,:]

[sigma,sigma_c] = estimate_sigma_localpca(data,gtab)
# identify the b0 images from the dataset

# first identify the number of the b0 images
arr = data
tou = 0
patch_radius = 1

if arr.ndim == 4:

        if tou == 0:
            tou = 2.3 * 2.3 * sigma * sigma

        if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

            sigma = (np.ones(arr.shape, dtype=np.float64) * sigma[..., np.newaxis])
            tou = (np.ones(arr.shape, dtype=np.float64) * tou[..., np.newaxis])
        else:
            sigma = np.ones(arr.shape, dtype=np.float64) * sigma
            tou = np.ones(arr.shape, dtype=np.float64) * tou    
        
        # loop around and find the 3D patch for each direction at each pixel
        
        # declare arrays for theta and thetax
        theta = np.zeros(arr.shape, dtype = np.float64)
        thetax = np.zeros(arr.shape, dtype = np.float64)

        patch_size = 2 * patch_radius + 1

        for k in range(patch_radius, arr.shape[2] - patch_radius , 1):
            print k
            for j in range(patch_radius, arr.shape[1] - patch_radius , 1):
                for i in range(patch_radius, arr.shape[0] - patch_radius , 1):
                    
                    X = np.zeros((arr.shape[3], patch_size * patch_size * patch_size))
                    M = np.zeros(arr.shape[3])

                    temp = arr[i - patch_radius : i + patch_radius + 1,j - patch_radius : j + patch_radius + 1,
                         k - patch_radius : k + patch_radius + 1,:]
                    temp = temp.reshape(patch_size * patch_size * patch_size, arr.shape[3])
                    X = temp.transpose()
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
                    d[d < tou[i][j][k][0]] = 0
                    D_hat = np.diag(d)
                    # Y = X.dot(W)
                    # # When the block covers each pixel identify it into the label matrix theta
                    # X_est = Y.dot(np.transpose(W))
                    X_est = X.dot(D_hat)
                    
                    # generate a theta matrix for patch around i,j,k and store it's theta value
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
        
        # After estimation pass it through a function ~ rician adaptation
        


        # return denoised_arr
before = data[:,:,18,1]
after = denoised_arr[:,:,18,1]
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
