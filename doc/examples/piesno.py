"""
=============================
Noise estimation using PIESNO
=============================

Using the Probabilistic Identification and Estimation of Noise (PIESNO) [Koay2009]_
one can detect the standard deviation of the noise from
diffusion-weighted imaging (DWI). PIESNO also works with multiple channel
DWI datasets that are acquired from N array coils for both SENSE and
GRAPPA reconstructions.

The PIESNO paper [Koay2009]_ works in two steps. 1) First, it finds voxels that are most likely
background voxels. Intuitively, these voxels have very similar
diffusion-weighted intensities (up to some noise) in the fourth dimension
of the DWI dataset. White matter, gray matter or CSF voxels have diffusion intensities
that vary quite a lot across different directions.
2) From these estimated background voxels and
the input number of coils N, PIESNO finds what sigma each Gaussian
from each of the N coils would have generated the observed Rician (N=1)
or non-central Chi (N>1) distributed noise profile in the DWI datasets.
[Koay2009]_ gives all the details.

PIESNO makes an important assumption: the
Gaussian noise standard deviation is assumed to be uniform. The noise
is uniform across multiple slice locations or across multiple images of the same location.

"""

import nibabel as nib
import numpy as np
from dipy.denoise.noise_estimate import piesno
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell


fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()
data = img.get_data()

"""

Now that we have fetched a dataset, we must call PIESNO with the right number
of coils used to acquire this dataset. It is also important to know what
was the parallel reconstruction algorithm used.
Here, the data comes from a GRAPPA reconstruction from
a 12-elements head coil available on the Tim Trio Siemens, for which
the 12 coils elements are combined into 4 groups of 3 coils elements
each. The signal is received through 4 distinct groups of receiver channels,
yielding N = 4. Had we used a GE acquisition, we would have used N=1 even
if multiple channel coils are used because GE uses a SENSE reconstruction,
which has a Rician noise nature and thus N is always 1.

As a convenience, we will estimate the noise for the whole volume in one go,
but it is also possible ot get a slice by slice estimation of the noise if
it is more desirable through the piesno_3D function.
"""

sigma, mask = piesno(data, N=4, return_mask=True)

axial = data[:, :, data.shape[2] / 2, 0].T
axial_piesno = mask[:, :, data.shape[2] / 2].T

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].imshow(axial, cmap='gray', origin='lower')
ax[0].set_title('Axial slice of the b=0 data')
ax[1].imshow(axial_piesno, cmap='gray', origin='lower')
ax[1].set_title('Background voxels from the data')
for a in ax:
    a.set_axis_off()

# Uncomment the coming line if you want a window to show the images.
#plt.show()
plt.savefig('piesno.png', bbox_inches='tight')

"""
.. figure:: piesno.png
   :align: center

   **Showing the mid axial slice of the b=0 image (left) and estimated background voxels (right) used to estimate the noise standard deviation**.
"""

nib.save(nib.Nifti1Image(mask, img.get_affine(), img.get_header()),
         'mask_piesno.nii.gz')

print('The noise standard deviation is sigma= ', sigma)
print('The std of the background is =', np.std(data[mask[...,None].astype(np.bool)]))

"""

Here, we obtained a noise standard deviation of 7.26. For comparison, a simple
standard deviation of all voxels in the estimated mask (as done in the previous
example :ref:`example_snr_in_cc`) gives a value of 6.1.

"""

"""

.. [Koay2009] Koay C.G., E. Ozarslan, C. Pierpaoli. Probabilistic Identification and Estimation of Noise (PIESNO): A self-consistent approach and its applications in MRI. JMR, 199(1):94-103, 2009.

.. include:: ../links_names.inc


"""
