"""
==============================================================
Crossing invariant fiber response function with FORECAST model
==============================================================

We show how to obtain a voxel specific response function in the form of
axially symmetric tensor and the fODF using the FORECAST model from
[Anderson2005]_ , [Kaden2016]_ and [Zucchelli2017]_.

First import the necessary modules:
"""

import numpy as np
import matplotlib.pyplot as plt
from dipy.reconst.forecast import ForecastModel
from dipy.viz import actor, window
from dipy.data import fetch_hbn, get_sphere
import nibabel as nib
import os.path as op
from dipy.core.gradients import gradient_table

###############################################################################
# Download and read the data for this tutorial. Our implementation of FORECAST
# requires multi-shell `data.fetch_hbn()` provides data that was acquired using
# b-values of 1000 and 2000 as part of the Healthy Brain Network study
# [Alexander2017]_ and was preprocessed and quality controlled in the HBN-POD2
# dataset [RichieHalford2022]_.

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

mask_img = nib.load(
       op.join(dwi_path,
"sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-brain_mask.nii.gz"))

brain_mask = mask_img.get_fdata()

###############################################################################
# Let us consider only a single slice for the FORECAST fitting

data_small = data[:, :, 50:51]
mask_small = brain_mask[:, :, 50:51]

###############################################################################
# Instantiate the FORECAST Model.
#
# "sh_order_max" is the spherical harmonics order (l) used for the fODF.
#
# dec_alg is the spherical deconvolution algorithm used for the FORECAST basis
# fitting, in this case we used the Constrained Spherical Deconvolution (CSD)
# algorithm.

fm = ForecastModel(gtab, sh_order_max=6, dec_alg='CSD')

###############################################################################
# Fit the FORECAST to the data

f_fit = fm.fit(data_small, mask_small)

###############################################################################
# Calculate the crossing invariant tensor indices [Kaden2016]_ : the parallel
# diffusivity, the perpendicular diffusivity, the fractional anisotropy and
# the mean diffusivity.

d_par = f_fit.dpar
d_perp = f_fit.dperp
fa = f_fit.fractional_anisotropy()
md = f_fit.mean_diffusivity()

###############################################################################
# Show the indices and save them in FORECAST_indices.png.

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title='parallel diffusivity')
ax1.set_axis_off()
ind = ax1.imshow(d_par[:, :, 0].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
ax2 = fig.add_subplot(2, 2, 2, title='perpendicular diffusivity')
ax2.set_axis_off()
ind = ax2.imshow(d_perp[:, :, 0].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
ax3 = fig.add_subplot(2, 2, 3, title='fractional anisotropy')
ax3.set_axis_off()
ind = ax3.imshow(fa[:, :, 0].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
ax4 = fig.add_subplot(2, 2, 4, title='mean diffusivity')
ax4.set_axis_off()
ind = ax4.imshow(md[:, :, 0].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
plt.savefig('FORECAST_indices.png', dpi=300, bbox_inches='tight')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# FORECAST scalar indices.
#
#
# Load an ODF reconstruction sphere

sphere = get_sphere('repulsion724')

###############################################################################
# Compute the fODFs.

odf = f_fit.odf(sphere)
print('fODF.shape (%d, %d, %d, %d)' % odf.shape)

###############################################################################
# Display a part of the fODFs

odf_actor = actor.odf_slicer(odf[30:60, 30:60, :], sphere=sphere,
                             colormap='plasma', scale=0.6)
scene = window.Scene()
scene.add(odf_actor)
window.record(scene, out_path='fODFs.png', size=(600, 600), magnification=4)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Fiber Orientation Distribution Functions, in a small ROI of the brain.
#
#
# References
# ----------
#
# .. [Anderson2005] Anderson A. W., "Measurement of Fiber Orientation
#        Distributions Using High Angular Resolution Diffusion Imaging",
#        Magnetic Resonance in Medicine, 2005.
#
# .. [Kaden2016] Kaden E. et al., "Quantitative Mapping of the Per-Axon
#        Diffusion Coefficients in Brain White Matter", Magnetic Resonance
#        in Medicine, 2016.
#
# .. [Zucchelli2017] Zucchelli E. et al., "A generalized SMT-based framework
#        for Diffusion MRI microstructural model estimation", MICCAI Workshop
#        on Computational DIFFUSION MRI (CDMRI), 2017.
#
# .. [Alexander2017] Alexander LM, Escalera J, Ai L, et al. An open resource
#        for transdiagnostic research in pediatric mental health and learning
#        disorders. Sci Data. 2017;4:170181.
#
# .. [RichieHalford2022] Richie-Halford A, Cieslak M, Ai L, et al. An
#        analysis-ready and quality controlled resource for pediatric brain
#        white-matter research. Scientific Data. 2022;9(1):1-27.
