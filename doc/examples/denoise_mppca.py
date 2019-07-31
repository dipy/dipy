"""
==========================================================
Denoise images using the Marcenko and Pastur PCA algorithm
==========================================================

The local PCA based denoising algorithm is an effective denoising
method because it exploits the redundancy across the diffusion-weighted images
[Manjon2013]_, [Veraart2016a]_. This algorithm has been shown to provide an
optimal compromise between noise suppression and loss of anatomical information
for different techniques such as DTI [Manjon2013]_, spherical deconvolution
[Veraart2016a] and DKI [Henri2018]_.

The basic idea behind local PCA based diffusion denoising is to remove the
data's principal components mostly related to noise. The classification of
the principal components can be performed based on prior noise variance
estimates and empirical thresholds [Manjon2013]_ or based on random matrix
theory [Veraa2016a]. In addition to noise suppression, local PCA can be used
to estimate the noise variance [Veraa2016b].

In the following example, we show how to denoise diffusion MRI images and
estimate the noise variance using the local PCA algorithm.

Let's load the necessary modules
"""

# load general modules
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time

# load main localpca function
from dipy.denoise.localpca import localpca

# load functions to fetch data for this example
from dipy.data import (fetch_cfin_multib, read_cfin_dwi)

# load other dipy's functions that will be used for auxiliar analysis
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import dipy.reconst.dki as dki

"""
For this example, we use fetch to download a multi-shell dataset which was
kindly provided by Hansen and Jespersen (more details about the data are
provided in their paper [Hansen2016]_). The total size of the downloaded data
is 192 MBytes, however you only need to fetch it once.
"""

fetch_cfin_multib()

img, gtab = read_cfin_dwi()

data = img.get_data()

affine = img.affine

"""
For the sake of simplicity, we only select two non-zero b-values of this
extensive dataset.
"""

bvals = gtab.bvals

bvecs = gtab.bvecs

sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)

data = data[..., sel_b]

gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

print(data.shape)

"""
As one can see from its shape, the selected data contains a total of 67
volumes of images corresponding to all the diffusion gradient directions
of the selected b-values.

Local PCA denoising can be performed by running the following command:
""" 

t = time()

denoised_arr = localpca(data, patch_radius=2)

print("Time taken for local PCA ", -t + time())

"""
Internally, the ``localpca`` algorithm locally denoises the 4D data using
a 3D sliding window which is defined by the variable patch_radius (number of
comprising voxels around the center of the window). In total, this window
should comprise a larger number of voxels than the number of diffusion-weighted
volumes. Since our data has a total of 67 volumes, the patch_radius is set to
2 which corresponds to a 5x5x5 sliding window comprising a total of 125 voxels.

To assess the performance of the ``localpca`` algorithm, an axial slice of the
original, denoised data, and their difference are ploted below:
"""

sli = data.shape[2] // 2
gra = data.shape[3] - 1
orig = data[:, :, sli, gra]
den = denoised_arr[:, :, sli, gra]
rms_diff = np.sqrt((orig - den) ** 2)

fig1, ax = plt.subplots(1, 3, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(orig.T, cmap='gray', interpolation='none',
                  origin='lower')
ax.flat[0].set_title('Original')
ax.flat[1].imshow(den.T, cmap='gray', interpolation='none',
                  origin='lower')
ax.flat[1].set_title('Debiused Output')
ax.flat[2].imshow(rms_diff.T, cmap='gray', interpolation='none',
                  origin='lower')
ax.flat[2].set_title('denoised_localpca.png', bbox_inches='tight')

print("The result saved in denoised_localpca.png")

"""
.. figure:: denoised_localpca.png
   :align: center

   Showing the middle axial slice of the local PCA denoised output.
"""

nib.save(nib.Nifti1Image(denoised_arr,
                         affine), 'denoised_localpca.nii.gz')

print("Entire denoised data saved in denoised_localpca.nii.gz")

"""


"""

import dipy.reconst.dki as dki
dkimodel = dki.DiffusionKurtosisModel(gtab)

maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)

dki_orig = dkimodel.fit(data, mask=mask)
dki_den = dkimodel.fit(denoised_arr, mask=mask)

FA_orig = dki_orig.fa
FA_den = dki_den.fa
MD_orig = dki_orig.md
MD_den = dki_den.md
MK_orig = dki_orig.mk(0, 3)
MK_den = dki_den.mk(0, 3)


fig1, ax = plt.subplots(2, 3, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.03)

ax.flat[0].imshow(MD_orig[:, :, sli].T, cmap='gray', vmin=0, vmax=2.0e-3,
                  origin='lower')
ax.flat[0].set_title('MD (DKI)')
ax.flat[1].imshow(FA_orig[:, :, sli].T, cmap='gray', vmin=0, vmax=0.7,
                  origin='lower')
ax.flat[1].set_title('FA (DKI)')
ax.flat[2].imshow(MK_orig[:, :, sli].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[2].set_title('AD (DKI)')
ax.flat[3].imshow(MD_den[:, :, sli].T, cmap='gray', vmin=0, vmax=2.0e-3,
                  origin='lower')
ax.flat[3].set_title('MD (DKI)')
ax.flat[4].imshow(FA_den[:, :, sli].T, cmap='gray', vmin=0, vmax=0.7,
                  origin='lower')
ax.flat[4].set_title('FA (DKI)')
ax.flat[5].imshow(MK_den[:, :, sli].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[5].set_title('AD (DKI)')
plt.show()
fig1.savefig('Diffusion_tensor_measures_from_DTI_and_DKI.png')

"""
References
----------

.. [Manjon2013] Manjon JV, Coupe P, Concha L, Buades A, Collins DL "Diffusion
                Weighted Image Denoising Using Overcomplete Local PCA" (2013).
                PLoS ONE 8(9): e73021. doi:10.1371/journal.pone.0073021.

.. [Veraa2016a] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
                mapping using random matrix theory. Magnetic Resonance in
                Medicine. doi: 10.1002/mrm.26059.

.. [Henri2018] Henriques, R., 2018. Advanced Methods for Diffusion MRI Data
               Analysis and their Application to the Healthy Ageing Brain
               (Doctoral thesis). https://doi.org/10.17863/CAM.29356

.. [Veraa2016b] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
                Fieremans E, 2016. Denoising of Diffusion MRI using random
                matrix theory. Neuroimage 142:394-406.
                doi: 10.1016/j.neuroimage.2016.08.016

.. include:: ../links_names.inc
"""
