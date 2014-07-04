"""
=============================
Noise estimation using PIESNO
=============================

Using PIESNO [Koay2009]_ one can detect the noise's standard deviation from
diffusion-weighted imaging (DWI). PIESNO also works from multiple channel
DWI datasets that are acquired from N array coils for both SENSE and
GRAPPA reconstructions.

The PIESNO paper [Koay2009]_ is mathetically quite involved.
PIESNO works in two steps. 1) First, it finds voxels that are most likely
background voxels. Intuitively, these voxels have very similar
diffusion-weighted intensities (up to some noise) in the fourth dimension
of the DWI dataset, as opposed to tissue voxels that have diffuison intensities
that vary quite a lot across different directions.
2) From these esimated background voxels and
the input number of coils N, PIESNO finds what sigma each Gaussian distributed
image profile from each of the N coils would have generated the observed Rician (N=1)
or non-central Chi (N>1) distributed noise profile in the DWI datasets.
[Koay2009]_ gives all the glory details.

PIESNO makes an important assumption: the
Gaussian noise standard deviation is assumed to be uniform either
across multiple slice locations or across multiple images of the same location,
e.g., if the readout bandwidth is maintained at the same level for all the images.

"""

import nibabel as nib
from dipy.denoise.noise_estimate import piesno
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell


fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()
data = img.get_data()

"""

Now that we have fetched a dataset, we must call PIESNO with right number N
of coil used to acquire this dataset. It is also important to know what
was the parallel reconstruction algorithm used.
Here, the data comes from a GRAPPA reconstruction from
a 12-element head coil available on the Tim Trio Siemens, for which
the 12 coil elements are combined into 4 groups of 3 coil elements
each. These groups are received through 4 distinct receiver channels,
yielding N = 4. Had we used a GE acquisition, we would have used N=1 even
if multiple channel coils are used because GE uses a SENSE reconstruction,
which has a Rician noise nature and thus N is always 1.

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
for i in range(2):
    ax[i].set_axis_off()

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


"""

Here, we obtained a noise standard deviation of 7.26.

"""

"""

.. [Koay2009] Koay C.G., E. Ozarslan, C. Pierpaoli. Probabilistic Identification and Estimation of Noise (PIESNO): A self-consistent approach and its applications in MRI. JMR, 199(1):94-103, 2009.

.. include:: ../links_names.inc


"""
