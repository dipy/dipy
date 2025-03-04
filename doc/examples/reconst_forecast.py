"""
==============================================================
Crossing invariant fiber response function with FORECAST model
==============================================================

We show how to obtain a voxel specific response function in the form of
axially symmetric tensor and the fODF using the FORECAST model from
:footcite:p:`Anderson2005`, :footcite:p:`Kaden2016a` and
:footcite:p:`Zucchelli2017`.

First import the necessary modules:
"""

import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import fetch_hbn, get_sphere
from dipy.reconst.forecast import ForecastModel
from dipy.viz import actor, window

###############################################################################
# Download and read the data for this tutorial. Our implementation of FORECAST
# requires multi-shell `data.fetch_hbn()` provides data that was acquired using
# b-values of 1000 and 2000 as part of the Healthy Brain Network study
# :footcite:p:`Alexander2017` and was preprocessed and quality controlled in the
# HBN-POD2 dataset :footcite:p:`RichieHalford2022`.

data_path = fetch_hbn(["NDARAA948VFH"])[1]
dwi_path = op.join(
    data_path, "derivatives", "qsiprep", "sub-NDARAA948VFH", "ses-HBNsiteRU", "dwi"
)

img = nib.load(
    op.join(
        dwi_path,
        "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.nii.gz",
    )
)

gtab = gradient_table(
    op.join(
        dwi_path,
        "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.bval",
    ),
    bvecs=op.join(
        dwi_path,
        "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.bvec",
    ),
)

data = np.asarray(img.dataobj)

mask_img = nib.load(
    op.join(
        dwi_path,
        "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-brain_mask.nii.gz",
    )
)

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

fm = ForecastModel(gtab, sh_order_max=6, dec_alg="CSD")

###############################################################################
# Fit the FORECAST to the data

f_fit = fm.fit(data_small, mask=mask_small)

###############################################################################
# Calculate the crossing invariant tensor indices :footcite:p:`Kaden2016a`: the
# parallel diffusivity, the perpendicular diffusivity, the fractional anisotropy
# and the mean diffusivity.

d_par = f_fit.dpar
d_perp = f_fit.dperp
fa = f_fit.fractional_anisotropy()
md = f_fit.mean_diffusivity()

###############################################################################
# Show the indices and save them in FORECAST_indices.png.

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title="parallel diffusivity")
ax1.set_axis_off()
ind = ax1.imshow(
    d_par[:, :, 0].T, interpolation="nearest", origin="lower", cmap=plt.cm.gray
)
plt.colorbar(ind, shrink=0.6)
ax2 = fig.add_subplot(2, 2, 2, title="perpendicular diffusivity")
ax2.set_axis_off()
ind = ax2.imshow(
    d_perp[:, :, 0].T, interpolation="nearest", origin="lower", cmap=plt.cm.gray
)
plt.colorbar(ind, shrink=0.6)
ax3 = fig.add_subplot(2, 2, 3, title="fractional anisotropy")
ax3.set_axis_off()
ind = ax3.imshow(
    fa[:, :, 0].T, interpolation="nearest", origin="lower", cmap=plt.cm.gray
)
plt.colorbar(ind, shrink=0.6)
ax4 = fig.add_subplot(2, 2, 4, title="mean diffusivity")
ax4.set_axis_off()
ind = ax4.imshow(
    md[:, :, 0].T, interpolation="nearest", origin="lower", cmap=plt.cm.gray
)
plt.colorbar(ind, shrink=0.6)
plt.savefig("FORECAST_indices.png", dpi=300, bbox_inches="tight")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# FORECAST scalar indices.
#
#
# Load an ODF reconstruction sphere

sphere = get_sphere(name="repulsion724")

###############################################################################
# Compute the fODFs.

odf = f_fit.odf(sphere)
print(f"fODF.shape {odf.shape}")

###############################################################################
# Display a part of the fODFs

odf_actor = actor.odf_slicer(
    odf[30:60, 30:60, :], sphere=sphere, colormap="plasma", scale=0.6
)
scene = window.Scene()
scene.add(odf_actor)
window.record(scene=scene, out_path="fODFs.png", size=(600, 600), magnification=4)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Fiber Orientation Distribution Functions, in a small ROI of the brain.
#
#
# References
# ----------
#
# .. footbibliography::
#
