"""
===============================
Denoise images using Local PCA
===============================

Using the local PCA based denoising for diffusion
images [Manjon2013]_ we can denoise diffusion images significantly. This
algorithm is effective as it takes into account the directional information in
diffusion data.

The basic idea behind local PCA based diffusion denoising can
be explained in the following three basic steps

* First we estimate the local noise variance at each voxel

* Then we apply PCA at local patches around each voxels over the gradient
  directions

* Threshold the eigenvalues based on the local estimate of sigma and then do
  the PCA reconstruction

Let's load the necessary modules
"""

import numpy as np
import scipy as sp
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.localpca_slow import localpca_slow
from dipy.denoise.fast_noise_estimate import fast_noise_estimate
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell

"""
Load one of the datasets, it has 63 gradients and 1 b0 image
"""

fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()

data = np.array(img.get_data())
affine = img.get_affine()

print("Input Volume", data.shape)

"""
We use a special noise estimation method ``fast_noise_estimate`` for getting
the sigma to be used in local PCA algorithm. It takes both data and the
gradient table object as input and returns an estimate of local noise standard
deviation as a 3D array.
"""

t = time()

sigma = np.array(fast_noise_estimate(data.astype(np.float64), gtab))
print("Sigma estimation time", time() - t)

"""
Perform the localPCA using the function localpca.

The localpca algorithm takes into account for the directional
information in the diffusion MR data. It performs PCA on local 4D patch and
then thresholds it using the local variance estimate done by noise estimation
function, then performing PCA reconstruction on it gives us the deniosed
estimate.
"""

t = time()

denoised_arr = localpca_slow(data, sigma=sigma, patch_radius=2)

print("Time taken for local PCA (slow)", -t + time())

"""
Let us plot the axial slice of the original and denoised data.
We visualize all the slices (22 in total)
"""

sli = data.shape[2] / 2
gra = data.shape[3] / 2
orig = data[:, :, sli, gra]
den = denoised_arr[:, :, sli, gra]
diff = np.abs(orig.astype('f8') - den.astype('f8'))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(orig, cmap='gray', origin='lower', interpolation='none')
ax[0].set_title('Original')
ax[0].set_axis_off()
ax[1].imshow(den, cmap='gray', origin='lower', interpolation='none')
ax[1].set_title('Denoised Output')
ax[1].set_axis_off()
ax[2].imshow(diff, cmap='gray', origin='lower', interpolation='none')
ax[2].set_title('Residual')
ax[2].set_axis_off()
plt.savefig('denoised_localpca.png', bbox_inches='tight')

print("The result saved in denoised_localpca.png")

"""
.. figure:: denoised_localpca.png
   :align: center

   **Showing the middle axial slice of the local PCA denoised output**.
"""

nib.save(nib.Nifti1Image(denoised_arr,
                         affine), 'denoised_localpca.nii.gz')

print("Entire denoised data saved in denoised_localpca.nii.gz")

"""
.. [Manjon2013] Manjon JV, Coupe P, Concha L, Buades A, Collins DL
   "Diffusion Weighted Image Denoising Using Overcomplete Local PCA" PLOS 2013

.. include:: ../links_names.inc
"""
