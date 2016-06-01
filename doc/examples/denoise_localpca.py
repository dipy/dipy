"""
===============================
Denoise images using Local PCA
===============================

"""

import numpy as np
import scipy as sp
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.localpca import localpca
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.denoise.noise_estimate_localpca import estimate_sigma_localpca
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

# fetch_taiwan_ntu_dsi()
# img,gtab = read_taiwan_ntu_dsi()

# fetch_sherbrooke_3shell()
# img, gtab = read_sherbrooke_3shell()

# fetch_stanford_hardi()
# img, gtab = read_stanford_hardi()

img = nib.load('/Users/Riddhish/Documents/GSOC/DIPY/data/test.nii')
den_img = nib.load(
    '/Users/Riddhish/Documents/GSOC/DIPY/data/test_denoised_rician.nii')
b1, b2 = read_bvals_bvecs('/Users/Riddhish/Documents/GSOC/DIPY/data/test.bval',
                          '/Users/Riddhish/Documents/GSOC/DIPY/data/test.bvec')
gtab = gradient_table(b1, b2)
data = np.array(img.get_data())
den_data = np.array(den_img.get_data())
affine = img.get_affine()

# currently just taking the small patch of the data to preserv time

data = np.array(data[20:90, 20:90, 10:15, :])
den_data = np.array(den_data[20:90, 20:90, 10:15, :])
mini = np.min(data)
maxi = np.max(data)
data = (data - mini) * 255.0 / maxi
[sigma, sigma_c] = estimate_sigma_localpca(data, gtab)
# identify the b0 images from the dataset
sigma = sigma_c
# first identify the number of the b0 images
arr = data
tou = 0
patch_radius = 1

# beta = 1.29
# alpha = 1.01e-6
t = time()
if arr.ndim == 4:

    if tou == 0:
        tou = 2.3 * 2.3 * sigma

    if isinstance(sigma, np.ndarray) and sigma.ndim == 3:

        sigma = (np.ones(arr.shape, dtype=np.float64) * sigma[..., np.newaxis])
        tou = (np.ones(arr.shape, dtype=np.float64) * tou[..., np.newaxis])
    else:
        sigma = np.ones(arr.shape, dtype=np.float64) * sigma
        tou = np.ones(arr.shape, dtype=np.float64) * tou

    # loop around and find the 3D patch for each direction at each pixel
    numeig = np.zeros(arr[..., 0].shape, dtype=np.float64)
    # sigma = np.zeros(arr[...,0].shape,dtype = np.float64)

    # declare arrays for theta and thetax
    theta = np.zeros(arr.shape, dtype=np.float64)
    thetax = np.zeros(arr.shape, dtype=np.float64)

    patch_size = 2 * patch_radius + 1

    for k in range(patch_radius, arr.shape[2] - patch_radius, 1):
        print k
        for j in range(patch_radius, arr.shape[1] - patch_radius, 1):
            for i in range(patch_radius, arr.shape[0] - patch_radius, 1):

                X = np.zeros(
                    (patch_size * patch_size * patch_size, arr.shape[3]))
                M = np.zeros(arr.shape[3])

                temp = arr[
                    i -
                    patch_radius: i +
                    patch_radius +
                    1,
                    j -
                    patch_radius: j +
                    patch_radius +
                    1,
                    k -
                    patch_radius: k +
                    patch_radius +
                    1,
                    :]
                X = temp.reshape(
                    patch_size *
                    patch_size *
                    patch_size,
                    arr.shape[3])
                # compute the mean and normalize
                M = np.mean(X, axis=1)
                X = X - np.array([M, ] * X.shape[1],
                                 dtype=np.float64).transpose()

                # using the PCA trick
                # Compute the covariance matrix C = X_transpose X
                C = np.transpose(X).dot(X)
                C = C / arr.shape[3]
                # compute EVD of the covariance matrix of X get the matrices W and D, hence get matrix Y = XW
                # Threshold matrix D and then compute X_est = YW_transpose
                # D_est
                [d, W] = np.linalg.eig(C)

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

                d[d < tou[i, j, k, :]] = 0
                d[d > 0] = 1
                numeig[i, j, k] = np.sum(d)
                D_hat = np.diag(d)

                Y = X.dot(W)
                # # When the block covers each pixel identify it into the label matrix theta
                X_est = Y.dot(np.transpose(W))
                X_est = X_est.dot(D_hat)

                # generate a theta matrix for patch around i,j,k and store it's
                # theta value
                temp = X_est + \
                    np.array([M, ] * X_est.shape[1], dtype=np.float64).transpose()
                temp = temp.reshape(
                    patch_size, patch_size, patch_size, arr.shape[3])
                # Also update the estimate matrix which is X_est * theta

                theta[i - patch_radius: i + patch_radius + 1,
                      j - patch_radius: j + patch_radius + 1,
                      k - patch_radius: k + patch_radius + 1,
                      :] = theta[i - patch_radius: i + patch_radius + 1,
                                 j - patch_radius: j + patch_radius + 1,
                                 k - patch_radius: k + patch_radius + 1,
                                 :] + 1 / (1 + np.linalg.norm(d,
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
    # # we need to find the index of the closest value of arr/sigma from the dataset
    # eta_phi = np.sqrt(np.pi/2) * np.exp(-0.5 * phi**2) * (((1 + 0.5 * phi**2) * sp.special.iv(0,0.25 * phi**2) + (0.5 * phi**2) * sp.special.iv(1,0.25 * phi**2))**2)
    # # eta_phi = eta_phi[1:200]
    # corrected_arr = np.zeros_like(denoised_arr)
    # phi = denoised_arr / np.sqrt(sigma)
    # phi[np.isnan(phi)] = 0
    # opt_diff = np.abs(phi - eta_phi[0])
    # for i in range(eta_phi.size):
    #     print(i)
    #     if(i!=0):
    #         new_diff = np.abs(phi - eta_phi[i])
    #         corrected_arr[new_diff < opt_diff] = i
    #         opt_diff[new_diff<opt_diff] = new_diff[new_diff<opt_diff]

    # corrected_arr = np.sqrt(sigma) * corrected_arr * 15.0/1000.0


# before = data[:,:,0,1]
# after = denoised_arr[:,:,0,1]
# difference1 = np.abs(after.astype('f8') - before.astype('f8'))
# difference = corrected_arr[:,:,0,1]
# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(before, cmap='gray', origin='lower')
# ax[0].set_title('before')
# ax[1].imshow(after, cmap='gray', origin='lower')
# ax[1].set_title('without rician correction')
# ax[2].imshow(difference1, cmap='gray', origin='lower')
# ax[2].set_title('difference')

# for i in range(3):
#     ax[i].set_axis_off()
print("time taken", -t + time())
denoised_arr = denoised_arr * maxi / 255 + mini
data = data * maxi / 255 + mini
orig = data[:, :, 2, 19]
rmse = np.sum(np.abs(denoised_arr[:,
                                  :,
                                  2,
                                  :] - den_data[:,
                                                :,
                                                2,
                                                :])) / np.sum(np.abs(den_data[:,
                                                                              :,
                                                                              2,
                                                                              :]))
print("RMSE between python and matlab output", rmse)
den_matlab = den_data[:, :, 2, 19]
den_python = denoised_arr[:, :, 2, 19]
diff_matlab = np.abs(orig.astype('f8') - den_matlab.astype('f8'))
diff_python = np.abs(orig.astype('f8') - den_python.astype('f8'))
fig, ax = plt.subplots(2, 4)
ax[0, 0].imshow(orig, cmap='gray', origin='lower')
ax[0, 0].set_title('Original')
ax[0, 1].imshow(den_matlab, cmap='gray', origin='lower')
ax[0, 1].set_title('Matlab Output')
ax[0, 2].imshow(diff_matlab, cmap='gray', origin='lower')
ax[0, 2].set_title('Matlab Residual')
ax[1, 0].imshow(orig, cmap='gray', origin='lower')
ax[1, 0].set_title('Original')
ax[1, 1].imshow(den_python, cmap='gray', origin='lower')
ax[1, 1].set_title('Python Output')
ax[1, 2].imshow(diff_python, cmap='gray', origin='lower')
ax[1, 2].set_title('Python Residual')
ax[1, 3].imshow(numeig[:, :, 2], cmap='jet', origin='lower')
ax[1, 3].set_title('Retained eigenvalues')

# nib.save(nib.Nifti1Image(denoised_arr, affine), '/Users/Riddhish/Documents/GSOC/DIPY/data/no_overcomplete_mean.nii')
plt.show()
