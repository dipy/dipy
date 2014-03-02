"""
==============================================
Denoise images using Non-Local Means (NLMEANS)
==============================================

Using the non-local means filter [Coupe2008]_ you can denoise 3D or 4D images and
boost the SNR of your datasets. You can also decide between modeling the noise
as Gaussian or Rician (default).

"""

import numpy as np
from time import time
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell


fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()

data = img.get_data()
aff = img.get_affine()

mask = data[..., 0] > 80

data = data[..., 0]

print("vol size", data.shape)

t = time()

"""
In order to call ``nlmeans`` first you need to estimate the standard deviation
of the noise.
"""

sigma = np.std(data[~mask])

den = nlmeans(data, sigma=sigma, mask=mask)

print("total time", time() - t)
print("vol size", den.shape)

import matplotlib.pyplot as plt

axial_middle = data.shape[2] / 2

before = data[:, :, axial_middle].T
after = den[:, :, axial_middle].T
difference = np.abs(after.astype('f8') - before.astype('f8'))
difference[~mask[:, :, axial_middle].T] = 0

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
plt.savefig('denoised_S0.png', bbox_inches='tight')

"""
.. figure:: denoised_S0.png
   :align: center

   **Showing the middle axial slice without (left) and with (right) NLMEANS denoising**.
"""

nib.save(nib.Nifti1Image(den, aff), 'denoised.nii.gz')

"""

.. [Coupe2008] P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot,
   "An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic
   Resonance Images", IEEE Transactions on Medical Imaging, 27(4):425-441, 2008.

.. include:: ../links_names.inc


"""
