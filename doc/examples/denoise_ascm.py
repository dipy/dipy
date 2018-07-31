"""
==============================================================
Denoise images using Adaptive Soft Coefficient Matching (ASCM)
==============================================================

The adaptive soft coefficient matching (ASCM) as described in [Coupe11]_ is a
improved extension of non-local means (NLMEANS) denoising. ASCM gives a better
denoised images from two standard non-local means denoised versions of the
original data with different degrees sharpness. Here, one denoised input is
more "smooth" than the other (the easiest way to achieve this denoising is use
``non_local_means`` with two different patch radii).

ASCM involves these basic steps

* Computes wavelet decomposition of the noisy as well as denoised inputs

* Combines the wavelets for the output image in a way that it takes it's
  smoothness (low frequency components) from the input with larger smoothing,
  and the sharp features (high frequency components) from the input with
  less smoothing.

This way ASCM gives us a well denoised output while preserving the sharpness
of the image features.

Let us load the necessary modules
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.data import (fetch_sherbrooke_3shell,
                       read_sherbrooke_3shell)
from dipy.denoise.noise_estimate import estimate_sigma
from time import time
from dipy.denoise.non_local_means import non_local_means
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching

"""
Choose one of the data from the datasets in dipy_
"""

fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()

data = img.get_data()
affine = img.affine

mask = data[..., 0] > 80
data = data[..., 1]

print("vol size", data.shape)

t = time()

"""
In order to generate the two pre-denoised versions of the data we will use the
``non_local_means`` denoising. For ``non_local_means`` first we need to
estimate the standard deviation of the noise. We use N=4 since the Sherbrooke
dataset was acquired on a 1.5T Siemens scanner with a 4 array head coil.
"""

sigma = estimate_sigma(data, N=4)

"""
For the denoised version of the original data which preserves sharper features,
we perform non-local means with smaller patch size.
"""

den_small = non_local_means(
    data,
    sigma=sigma,
    mask=mask,
    patch_radius=1,
    block_radius=1,
    rician=True)

"""
For the denoised version of the original data that implies more smoothing, we
perform non-local means with larger patch size.
"""

den_large = non_local_means(
    data,
    sigma=sigma,
    mask=mask,
    patch_radius=2,
    block_radius=1,
    rician=True)

"""
Now we perform the adaptive soft coefficient matching. Empirically we set the
adaptive parameter in ascm to be the average of the local noise variance,
in this case the sigma itself.
"""

den_final = adaptive_soft_matching(data, den_small, den_large, sigma[0])

print("total time", time() - t)

"""
To access the quality of this denoising procedure, we plot the an axial slice
of the original data, it's denoised output and residuals.
"""

axial_middle = data.shape[2] // 2

original = data[:, :, axial_middle].T
final_output = den_final[:, :, axial_middle].T
difference = np.abs(final_output.astype('f8') - original.astype('f8'))
difference[~mask[:, :, axial_middle].T] = 0

fig, ax = plt.subplots(1, 3)
ax[0].imshow(original, cmap='gray', origin='lower')
ax[0].set_title('Original')
ax[1].imshow(final_output, cmap='gray', origin='lower')
ax[1].set_title('ASCM output')
ax[2].imshow(difference, cmap='gray', origin='lower')
ax[2].set_title('Residual')
for i in range(3):
    ax[i].set_axis_off()

plt.savefig('denoised_ascm.png', bbox_inches='tight')

print("The ascm result saved in denoised_ascm.png")

"""
.. figure:: denoised_ascm.png
   :align: center

   Showing the axial slice without (left) and with (middle) ASCM denoising.
"""

"""
From the above figure we can see that the residual is really uniform in nature
which dictates that ASCM denoises the data while preserving the sharpness of
the features.
"""

nib.save(nib.Nifti1Image(den_final, affine), 'denoised_ascm.nii.gz')

print("Saving the entire denoised output in denoised_ascm.nii.gz")

"""
For comparison propose we also plot the outputs of the ``non_local_means``
(both with the larger as well as with the smaller patch radius) with the ASCM
output.
"""

fig, ax = plt.subplots(1, 4)
ax[0].imshow(original, cmap='gray', origin='lower')
ax[0].set_title('Original')
ax[1].imshow(den_small[..., axial_middle].T, cmap='gray', origin='lower',
             interpolation='none')
ax[1].set_title('NLMEANS small')
ax[2].imshow(den_large[..., axial_middle].T, cmap='gray', origin='lower',
             interpolation='none')
ax[2].set_title('NLMEANS large')
ax[3].imshow(final_output, cmap='gray', origin='lower', interpolation='none')
ax[3].set_title('ASCM ')
for i in range(4):
    ax[i].set_axis_off()

plt.savefig('ascm_comparison.png', bbox_inches='tight')

print("The comparison result saved in ascm_comparison.png")

"""
.. figure:: ascm_comparison.png
   :align: center

   Comparing outputs of the NLMEANS and ASCM.
"""

"""
From the above figure, we can observe that the information of two pre-denoised
versions of the raw data, ASCM outperforms standard non-local means in
supressing noise and preserving feature sharpness.

References
----------

..  [Coupe11] Pierrick Coupe, Jose Manjon, Montserrat Robles, Louis Collins.
    Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising.
    IET Image Processing, Institution of Engineering and Technology,
    2011. <00645538>

.. include:: ../links_names.inc

"""
