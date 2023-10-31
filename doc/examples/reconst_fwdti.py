"""
=============================================================================
Using the free water elimination model to remove DTI free water contamination
=============================================================================

As shown previously (see
:ref:`sphx_glr_examples_built_reconstruction_reconst_dti.py`), the diffusion
tensor model is a simple way to characterize diffusion anisotropy. However,
in regions near the ventricles and parenchyma, anisotropy can be
underestimated by partial volume effects of the cerebral spinal fluid (CSF).
This free water contamination can particularly corrupt Diffusion Tensor
Imaging analysis of microstructural changes when different groups of subjects
show different brain morphology (e.g. brain ventricle enlargement associated
with brain tissue atrophy that occurs in several brain pathologies and aging).

A way to remove this free water influences is to expand the DTI model to take
into account an extra compartment representing the contributions of free water
diffusion [Pasternak2009]_. The expression of the expanded DTI model is shown
below:

.. math::

    S(\mathbf{g}, b) = S_0(1-f)e^{-b\mathbf{g}^T \mathbf{D}
    \mathbf{g}}+S_0fe^{-b D_{iso}}

where $\mathbf{g}$ and $b$ are diffusion gradient direction and weighted (more
information see :ref:`sphx_glr_examples_built_reconstruction_reconst_dti.py`),
$S(\mathbf{g}, b)$ is thebdiffusion-weighted signal measured, $S_0$ is the
signal in a measurement with no diffusion weighting, $\mathbf{D}$ is the
diffusion tensor, $f$ the volume fraction of the free water component, and
$D_{iso}$ is the isotropic value of the free water diffusion (normally set to
$3.0 \times 10^{-3} mm^{2}s^{-1}$).

In this example, we show how to process a diffusion weighting dataset using an
adapted version of the free water elimination proposed by [Hoy2014]_.

The full details of Dipy's free water DTI implementation was published in
[Henriques2017]_. Please cite this work if you use this algorithm.

Let's start by importing the relevant modules:
"""

import numpy as np
import dipy.reconst.fwdti as fwdti
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
from dipy.data import fetch_hbn
import os.path as op
import nibabel as nib
from dipy.core.gradients import gradient_table

###############################################################################
# Without spatial constrains the free water elimination model cannot be solved
# in data acquired from one non-zero b-value [Hoy2014]_. Therefore, here we
# download a dataset that was acquired with multiple b-values.

data_path = fetch_hbn(["NDARAA948VFH"])[1]
dwi_path = op.join(
       data_path, "derivatives", "qsiprep", "sub-NDARAA948VFH",
       "ses-HBNsiteRU", "dwi")

img = nib.load(op.join(
       dwi_path,
       "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.nii.gz"))

gtab = gradient_table(
       op.join(dwi_path,
"sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.bval"),
       op.join(dwi_path,
"sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.bvec"))

data = np.asarray(img.dataobj)

###############################################################################
# The free water DTI model can take some minutes to process the full data set.
# Thus, we use a brain mask that was calculated during pre-processing, to
# remove the background of the image and avoid unnecessary calculations.

mask_img = nib.load(
       op.join(dwi_path,
"sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-brain_mask.nii.gz"))

###############################################################################
# Moreover, for illustration purposes we process only one slice of the data.

mask = mask_img.get_fdata()

data_small = data[:, :, 50:51]
mask_small = mask[:, :, 50:51]

###############################################################################
# The free water elimination model fit can then be initialized by instantiating
# a FreeWaterTensorModel class object:

fwdtimodel = fwdti.FreeWaterTensorModel(gtab)

###############################################################################
# The data can then be fitted using the ``fit`` function of the defined model
# object:

fwdtifit = fwdtimodel.fit(data_small, mask=mask_small)

###############################################################################
# This 2-steps procedure will create a FreeWaterTensorFit object which contains
# all the diffusion tensor statistics free for free water contamination. Below
# we extract the fractional anisotropy (FA) and the mean diffusivity (MD) of
# the free water diffusion tensor.

FA = fwdtifit.fa
MD = fwdtifit.md

###############################################################################
# For comparison we also compute the same standard measures processed by the
# standard DTI model

dtimodel = dti.TensorModel(gtab)

dtifit = dtimodel.fit(data_small, mask=mask_small)

dti_FA = dtifit.fa
dti_MD = dtifit.md

###############################################################################
# Below the FA values for both free water elimination DTI model and standard
# DTI model are plotted in panels A and B, while the respective MD values are
# plotted in panels D and E. For a better visualization of the effect of the
# free water correction, the differences between these two metrics are shown
# in panels C and E. In addition to the standard diffusion statistics, the
# estimated volume fraction of the free water contamination is shown on
# panel G.

fig1, ax = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(FA[:, :, 0].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('A) fwDTI FA')
ax.flat[1].imshow(dti_FA[:, :, 0].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[1].set_title('B) standard DTI FA')

FAdiff = abs(FA[:, :, 0] - dti_FA[:, :, 0])
ax.flat[2].imshow(FAdiff.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.flat[2].set_title('C) FA difference')

ax.flat[3].axis('off')

ax.flat[4].imshow(MD[:, :, 0].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[4].set_title('D) fwDTI MD')
ax.flat[5].imshow(dti_MD[:, :, 0].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[5].set_title('E) standard DTI MD')

MDdiff = abs(MD[:, :, 0] - dti_MD[:, :, 0])
ax.flat[6].imshow(MDdiff.T, origin='lower', cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[6].set_title('F) MD difference')

F = fwdtifit.f

ax.flat[7].imshow(F[:, :, 0].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[7].set_title('G) free water volume')

plt.show()
fig1.savefig('In_vivo_free_water_DTI_and_standard_DTI_measures.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# In vivo diffusion measures obtain from the free water DTI and standard
# DTI. The values of Fractional Anisotropy for the free water DTI model and
# standard DTI model and their difference are shown in the upper panels (A-C),
# while respective MD values are shown in the lower panels (D-F). In addition
# the free water volume fraction estimated from the fwDTI model is shown in
# panel G.
#
#
# From the figure, one can observe that the free water elimination model
# produces in general higher values of FA and lower values of MD than the
# standard DTI model. These differences in FA and MD estimation are expected
# due to the suppression of the free water isotropic diffusion components.
# Unexpected high amplitudes of FA are however observed in the periventricular
# gray matter. This is a known artefact of regions associated to voxels with
# high water volume fraction (i.e. voxels containing basically CSF). We are
# able to remove this problematic voxels by excluding all FA values
# associated with measured volume fractions above a reasonable threshold
# of 0.7:

FA[F > 0.7] = 0
dti_FA[F > 0.7] = 0

###############################################################################
# Above we reproduce the plots of the in vivo FA from the two DTI fits and
# where we can see that the inflated FA values were practically removed:

fig1, ax = plt.subplots(1, 3, figsize=(9, 3),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(FA[:, :, 0].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('A) fwDTI FA')
ax.flat[1].imshow(dti_FA[:, :, 0].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[1].set_title('B) standard DTI FA')

FAdiff = abs(FA[:, :, 0] - dti_FA[:, :, 0])
ax.flat[2].imshow(FAdiff.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.flat[2].set_title('C) FA difference')

plt.show()
fig1.savefig('In_vivo_free_water_DTI_and_standard_DTI_corrected.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# In vivo FA measures obtain from the free water DTI (A) and standard
# DTI (B) and their difference (C). Problematic inflated FA values of the
# images were removed by dismissing voxels above a volume fraction threshold
# of 0.7.
#
#
# References
# ----------
# .. [Pasternak2009] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., Assaf,
#    Y., 2009. Free water elimination and mapping from diffusion MRI. Magn.
#    Reson. Med. 62(3): 717-30. doi: 10.1002/mrm.22055.
#
# .. [Hoy2014] Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014.
#    Optimization of a free water elimination two-compartmental model for
#    diffusion tensor imaging. NeuroImage 103, 323-333. doi:
#    10.1016/j.neuroimage.2014.09.053
#
# .. [Henriques2017] Henriques, R.N., Rokem, A., Garyfallidis, E., St-Jean, S.,
#    Peterson E.T., Correia, M.M., 2017. [Re] Optimization of a free water
#    elimination two-compartment model for diffusion tensor imaging.
#    ReScience volume 3, issue 1, article number 2
