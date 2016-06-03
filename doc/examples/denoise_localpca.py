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

data = np.array(data[20:100, 20:100, 10:15, :])
den_data = np.array(den_data[20:100, 20:100, 10:15, :])

[sigma, sigma_c] = estimate_sigma_localpca(data, gtab)
# identify the b0 images from the dataset
sigma = sigma_c
t = time()

# perform the local PCA denoising
denoised_arr = localpca(data,sigma)

print("time taken", -t + time())
orig = data[:, :,2, 10]
rmse = np.sum(np.abs(denoised_arr[:,:,:,:] - 
    den_data[:,:,:,:])) / np.sum(np.abs(den_data[:,:,:,:]))
print("RMSE between python and matlab output", rmse)
den_matlab = den_data[:, :, 2, 10]
den_python = denoised_arr[:, :, 2, 10]
diff_matlab = np.abs(orig.astype('f8') - den_matlab.astype('f8'))
diff_python = np.abs(orig.astype('f8') - den_python.astype('f8'))
fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(orig, cmap='gray', origin='lower')
ax[0, 0].set_title('Original')
ax[0, 1].imshow(den_matlab, cmap='gray', origin='lower')
ax[0, 1].set_title('Matlab Output')
ax[0, 2].imshow(diff_matlab, cmap='gray', origin='lower')
ax[0, 2].set_title('Matlab Residual')
ax[1,0].imshow(orig, cmap='gray', origin='lower')
ax[1,0].set_title('Original')
ax[1,1].imshow(den_python, cmap='gray', origin='lower')
ax[1,1].set_title('Python Output')
ax[1,2].imshow(diff_python, cmap='gray', origin='lower')
ax[1,2].set_title('Python Residual')

# nib.save(nib.Nifti1Image(denoised_arr, affine), '/Users/Riddhish/Documents/GSOC/DIPY/data/final.nii')
plt.show()
