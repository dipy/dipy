"""
==============================================
Denoise images using Non-Local Means (NLMEANS)
==============================================

Using the non-local means filter [1]_ ,[2]_ you can denoise 3D or 4D images and
boost the SNR of your datasets. You can also decide between modeling the noise
as Gaussian or Rician (default).

"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell


fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()

data = img.get_data()
affine = img.get_affine()

mask = data[..., 0] > 80

# We select only one volume for the example to run quickly.
data = data[..., 1]

print("vol size", data.shape)

t = time()

"""
In order to call ``nlmeans`` first you need to estimate the standard deviation
of the noise. We use N=4 since the Sherbrooke dataset was acquired on a 1.5T
Siemens scanner with a 4 array head coil.
"""

sigma = estimate_sigma(data, N=4)

"""
Compare results by both the approaches to nlmeans averaging
1) Blockwise averaging [1]
2) Voxelwise averaging (default) [2]
"""

den = nlmeans(
    data,
    sigma=sigma,
    mask=None,
    patch_radius=1,
    block_radius=1,
    rician=True,
    avg_type='blockwise')
den_v = nlmeans(
    data,
    sigma=sigma,
    mask=None,
    patch_radius=1,
    block_radius=1,
    rician=True,
    avg_type='voxelwise')

print("total time", time() - t)

axial_middle = data.shape[2] / 2

before = data[:, :, axial_middle].T
after = den[:, :, axial_middle].T
after_v = den_v[:, :, axial_middle].T
difference = np.abs(after.astype('f8') - before.astype('f8'))
difference_v = np.abs(after_v.astype('f8') - before.astype('f8'))
difference[~mask[:, :, axial_middle].T] = 0
difference_v[~mask[:, :, axial_middle].T] = 0

fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(before, cmap='gray', origin='lower')
ax[0, 0].set_title('before')
ax[0, 1].imshow(after, cmap='gray', origin='lower')
ax[0, 1].set_title('after (blockwise)')
ax[0, 2].imshow(difference, cmap='gray', origin='lower')
ax[0, 2].set_title('difference (blockwise)')
ax[1, 0].imshow(before, cmap='gray', origin='lower')
ax[1, 0].set_title('before')
ax[1, 1].imshow(after_v, cmap='gray', origin='lower')
ax[1, 1].set_title('after (voxelwise)')
ax[1, 2].imshow(difference_v, cmap='gray', origin='lower')
ax[1, 2].set_title('difference (voxelwise)')


plt.show()
plt.savefig('denoised.png', bbox_inches='tight')

"""
.. figure:: denoised.png
   :align: center

   **Showing the middle axial slice without (left) and with (right) NLMEANS denoising**.
"""

nib.save(nib.Nifti1Image(den, affine), 'denoised_blockwise.nii.gz')
nib.save(nib.Nifti1Image(den_v, affine), 'denoised_voxelwise.nii.gz')

"""

References
----------

.. [1] "Impact of Rician Adapted Non-Local Means Filtering on HARDI"
	Descoteaux, Maxim and Wiest-Daessle`, Nicolas and Prima, Sylvain and Barillot,
	Christian and Deriche, Rachid
	MICCAI – 2008

.. [2] P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot,
   "An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic
   Resonance Images", IEEE Transactions on Medical Imaging, 27(4):425-441, 2008.

.. include:: ../links_names.inc
"""
