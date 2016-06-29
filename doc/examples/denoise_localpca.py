"""
===============================
Denoise images using Local PCA
===============================

Using the local PCA based denoising for diffusion
images [Manjon2013]_ we can state of the art 
results. The advantage of local PCA over the other denoising
methods is that it takes into the account the directional
information of the diffusion data as well.

Let's load the necessary modules
"""

import numpy as np
import scipy as sp
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.localpca import localpca
from dipy.denoise.fast_noise_estimate import fast_noise_estimate
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

"""
Load one of the datasets, it has 21 gradients and 1 b0 image
"""

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

# data = np.array(data[20:100, 20:100, 10:15, :])
# den_data = np.array(den_data[20:100, 20:100, 10:15, :])

"""
We use a special noise estimation method for getting the sigma
to be used in local PCA algorithm. This is also proposed in [Manjon2013]_
It takes both data and the gradient table object as inputs and returns an 
estimate of local noise standard deviation as a 3D array

"""
t = time()
t1 = time()
sigma = np.array(fast_noise_estimate(data.astype(np.float64), gtab))
print("Sigma estimation time", time() - t)
# identify the b0 images from the dataset
t = time()

"""
Perform the localPCA using the function localpca.

The localpca algorithm [Manjon2013]_ takes into account for the directional 
information in the diffusion MR data. It performs PCA on local 4D patch and 
then thresholds it using the local variance estimate done by noise estimation
function, then performing PCA reconstruction on it gives us the deniosed 
estimate.
"""
# perform the local PCA denoising
denoised_arr_fast = localpca(data, sigma = sigma)

print("Time taken for local PCA", -t + time())

print("Total time", time() - t1)
# print(np.array(eigenf))
orig = data[:, :,2, 10]
rmse = np.sum(np.abs(denoised_arr_fast[:,:,:,:] - 
    den_data[:,:,:,:])) / np.sum(np.abs(den_data[:,:,:,:]))
print("RMSE between python and matlab output", rmse)
den_matlab = den_data[:, :, 2, 10]
den_python = denoised_arr_fast[:, :, 2, 10]

"""
Let us plot the axial slice of the k= 10 direction, and the residuals
"""
diff_matlab = np.abs(orig.astype('f8') - den_matlab.astype('f8'))
diff_python = np.abs(orig.astype('f8') - den_python.astype('f8'))
fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(orig, cmap='gray', origin='lower' , interpolation='none')
ax[0, 0].set_title('Original')
ax[0, 1].imshow(den_matlab, cmap='gray', origin='lower',interpolation='none')
ax[0, 1].set_title('Matlab Output')
ax[0, 2].imshow(diff_matlab, cmap='gray', origin='lower', interpolation='none')
ax[0, 2].set_title('Matlab Residual')
ax[1,0].imshow(orig, cmap='gray', origin='lower', interpolation='none')
ax[1,0].set_title('Original')
ax[1,1].imshow(den_python, cmap='gray', origin='lower', interpolation='none')
ax[1,1].set_title('Python Output')
ax[1,2].imshow(diff_python, cmap='gray', origin='lower', interpolation='none')
ax[1,2].set_title('Python Residual')

"""
Save the denoised output in nifty file format
"""
nib.save(nib.Nifti1Image(denoised_arr_fast, affine), '/Users/Riddhish/Documents/GSOC/DIPY/data/final.nii')
plt.show()

"""
.. [Manjon2013] Manjon JV, Coupe P, Concha L, Buades A, Collins DL
        "Diffusion Weighted Image Denoising Using Overcomplete
         Local PCA" 2013

.. include:: ../links_names.inc

"""