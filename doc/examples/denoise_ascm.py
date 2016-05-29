"""
==============================================================
Denoise images using Adaptive Soft Coefficient Matching (ASCM)
==============================================================

Using the non-local means based adaptive denoising [1]_ you can denoise 3D or 
4D images and boost the SNR of your datasets.

"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.data import fetch_sherbrooke_3shell
from dipy.data import read_sherbrooke_3shell
from dipy.denoise.noise_estimate import estimate_sigma
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.ascm import ascm

fetch_sherbrooke_3shell()
img,gtab = read_sherbrooke_3shell()

data = img.get_data()
affine = img.get_affine()

mask = data[..., 0] > 80
data = data[...,1]

print("vol size", data.shape)

t = time()

"""
In order to call ``nlmeans`` first you need to estimate the standard deviation
of the noise. We use N=4 since the Sherbrooke dataset was acquired on a 1.5T
Siemens scanner with a 4 array head coil. We will use the blockwise approach 
of nlmeans with two different sizes which is required for adaptive soft coeficient
matching.
"""

sigma = estimate_sigma(data, N=4)

# Smaller block denoising: More sharp less denoised
den_small = nlmeans(data, sigma=sigma, mask=None, patch_radius = 1, block_radius = 1,rician = True, type='blockwise')
# Larger block denoising: Less sharp and more denoised
den_large = nlmeans(data, sigma=sigma, mask=None, patch_radius = 2, block_radius = 1,rician = True, type='blockwise')

# Now perform the adaptive soft coefficient matching
"""
Empirically we set the parameter h in ascm to be the average of the local 
noise variance, sigma itself here in this case
"""
den_final = np.array(ascm(data, den_small, den_large, sigma[0]))

print("total time", time() - t)

axial_middle = data.shape[2] / 2

original = data[:, :, axial_middle].T
final_output = den_final[:,:,axial_middle].T
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

plt.show()
plt.savefig('denoised.png', bbox_inches='tight')

nib.save(nib.Nifti1Image(den_final, affine), 'denoised_ascm.nii.gz')

"""
References
----------

..  [1] : Pierrick Coupe, Jose Manjon, Montserrat Robles, Louis Collins. 
    Adaptive Multiresolution Non-Local Means Filter for 3D MR Image Denoising. 
    IET Image Processing, Institution of Engineering and Technology, 
    2011. <hal-00645538>

"""