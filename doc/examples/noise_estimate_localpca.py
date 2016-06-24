"""
===========================================
Estimate Noise Levels for LocalPCA Datasets
===========================================

"""

import numpy as np
import nibabel as nib
import scipy as sp
import matplotlib.pyplot as plt
from time import time
from scipy import ndimage
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell
from dipy.data import fetch_stanford_hardi, read_stanford_hardi


# fetch_sherbrooke_3shell()
# img, gtab = read_sherbrooke_3shell()

# fetch_stanford_hardi()
# img, gtab = read_stanford_hardi()

img = nib.load('/Users/Riddhish/Documents/GSOC/DIPY/data/test.nii')
den_img = nib.load(
    '/Users/Riddhish/Documents/GSOC/DIPY/data/test_denoised_gaussian.nii')
b1, b2 = read_bvals_bvecs('/Users/Riddhish/Documents/GSOC/DIPY/data/test.bval',
                          '/Users/Riddhish/Documents/GSOC/DIPY/data/test.bvec')
gtab = gradient_table(b1, b2)
data = np.array(img.get_data())
den_data = np.array(den_img.get_data())
affine = img.get_affine()

# currently just taking the small patch of the data to preserv time

# data = np.array(data[20:90,20:90,10:15,:])
# identify the b0 images from the dataset

# first identify the number of the b0 images
K = gtab.b0s_mask[gtab.b0s_mask].size

if(K > 1):
    print("MUBE Estimate")
    # If multiple b0 values then
    data0 = data[..., gtab.b0s_mask]

else:
    print("SIBE Estimate")
    # if only one b0 value then
    # SIBE Noise Estimate
    data0 = data[..., ~gtab.b0s_mask]
    # MUBE Noise Estimate

# Form their matrix
X = data0.reshape(
    data0.shape[0] *
    data0.shape[1] *
    data0.shape[2],
    data0.shape[3])
# X = NxK
# Now to subtract the mean
M = np.mean(X, axis=0)
X = X - M
C = np.transpose(X).dot(X)
# C = C/data0.shape[3]
# Do PCA and find the lowest principal component
[d, W] = np.linalg.eigh(C)
d[d < 0] = 0
d_new = d[d != 0]
# we want eigen vectors of XX^T so we do
V = X.dot(W)
# unit normalize the clolumns of V
# V = V / np.linalg.norm(V, ord=2, axis=0)
# As the W is sorted in increasing eignevalue order, so is V
# choose the smallest principle component
I = V[
    :,
    d.shape[0] -
    d_new.shape[0]].reshape(
        data.shape[0],
        data.shape[1],
    data.shape[2])

sigma = np.zeros(I.shape, dtype=np.float64)
count = np.zeros_like(data)
mean = np.zeros_like(data).astype(np.float64)

# Compute the local noise variance of the I image
for i in range(1, I.shape[0] - 1):
    print(i)
    for j in range(1, I.shape[1] - 1):
        for k in range(1, I.shape[2] - 1):

            sigma[i, j, k] += np.var(I[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
            temp = data[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2, :]
            mean[
                i, j, k, :] += np.mean(np.mean(np.mean(temp, axis=0), axis=0), axis=0)


snr = np.zeros_like(I).astype(np.float64)
sigma_corr = np.zeros_like(data).astype(np.float64)

# find the SNR and make the correction
# SNR Correction

for l in range(data.shape[3]):
    print(l)
    snr = mean[..., l] / np.sqrt(sigma)
    eta = 2 + snr**2 - (np.pi / 8) * np.exp(-0.5 * (snr**2)) * ((2 + snr**2) * \
                        sp.special.iv(0, 0.25 * (snr**2)) + (snr**2) * sp.special.iv(1, 0.25 * (snr**2)))**2
    sigma_corr[..., l] = sigma / eta

# smoothing by lpf
sigma_corr[np.isnan(sigma_corr)] = 0
sigma_corr_mean = np.mean(sigma_corr, axis=3)
sigma_corrr = ndimage.gaussian_filter(sigma_corr_mean, 3)
sigmar = ndimage.gaussian_filter(sigma, 3)


# display
fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(data[:, :, 0, 13], cmap='gray', origin='lower')
ax[0, 0].set_title('Original')
ax[0, 1].imshow(sigma[:, :, 2], cmap='gray', origin='lower')
ax[0, 1].set_title('sigma by PCA')
ax[0, 2].imshow(sigma_corr_mean[:, :, 2], cmap='gray', origin='lower')
ax[0, 2].set_title('after rescaling')
ax[1, 0].imshow(sigmar[:, :, 2], cmap='gray', origin='lower')
ax[1, 0].set_title('sigma by PCA + LPF')
ax[1, 1].imshow(sigma_corrr[:, :, 2], cmap='gray', origin='lower')
ax[1, 1].set_title('after rescaling + LPF')
ax[1, 2].imshow(I[:, :, 0], cmap='gray', origin='lower')
ax[1, 2].set_title('The least significant PC image')
# for i in range(3):
#     ax[i].set_axis_off()

plt.show()
