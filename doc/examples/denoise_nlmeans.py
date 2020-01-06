"""
==============================================
Denoise images using Non-Local Means (NLMEANS)
==============================================

Using the non-local means filter [Coupe08]_ and [Coupe11]_ and  you can denoise
3D or 4D images and boost the SNR of your datasets. You can also decide between
modeling the noise as Gaussian or Rician (default).

"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import get_fnames
from dipy.io.image import load_nifti


dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames('sherbrooke_3shell')
data, affine = load_nifti(dwi_fname)

mask = data[..., 0] > 80

# We select only one volume for the example to run quickly.
data = data[..., 1]

print("vol size", data.shape)

# lets create a noisy data with Gaussian data

"""
In order to call ``non_local_means`` first you need to estimate the standard
deviation of the noise. We use N=4 since the Sherbrooke dataset was acquired
on a 1.5T Siemens scanner with a 4 array head coil.
"""

sigma = estimate_sigma(data, N=4)

t = time()

"""
Calling the main function ``non_local_means``
"""

t = time()

den = nlmeans(data, sigma=sigma, mask=mask, patch_radius=1,
              block_radius=1, rician=True)

print("total time", time() - t)
"""
Let us plot the axial slice of the denoised output
"""

axial_middle = data.shape[2] // 2

before = data[:, :, axial_middle].T
after = den[:, :, axial_middle].T

difference = np.abs(after.astype(np.float64) - before.astype(np.float64))

difference[~mask[:, :, axial_middle].T] = 0


fig, ax = plt.subplots(1, 3)
ax[0].imshow(before, cmap='gray', origin='lower')
ax[0].set_title('before')
ax[1].imshow(after, cmap='gray', origin='lower')
ax[1].set_title('after')
ax[2].imshow(difference, cmap='gray', origin='lower')
ax[2].set_title('difference')

plt.savefig('denoised.png', bbox_inches='tight')


"""
.. figure:: denoised.png
   :align: center

   **Showing axial slice before (left) and after (right) NLMEANS denoising**
"""

nib.save(nib.Nifti1Image(den, affine), 'denoised.nii.gz')

"""
An improved version of non-local means denoising is adaptive soft coefficient
matching, please refer to :ref:`example_denoise_ascm` for more details.

References
----------

.. [Coupe08] P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot,
   "An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic
   Resonance Images", IEEE Transactions on Medical Imaging, 27(4):425-441, 2008

.. [Coupe11] Pierrick Coupe, Jose Manjon, Montserrat Robles, Louis Collins.
    "Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising"
    IET Image Processing, Institution of Engineering and Technology, 2011

.. include:: ../links_names.inc
"""
