"""
==================================================
Local reconstruction using the Histological ResDNN
==================================================

A data-driven approach to modeling the non-linear mapping between observed
DW-MRI signals and ground truth structures using sequential deep neural network
regression with residual block deep neural network (ResDNN)
:footcite:p:`Nath2018`, :footcite:p:`Nath2019`.

Training was performed on two 3-D histology datasets of squirrel monkey brains
and validated on a third. A second validation was performed on HCP datasets.
"""

import os

import numpy as np
import scipy.ndimage as ndi

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.nn.histo_resdnn import HistoResDNN
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window

# Disable oneDNN warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

###############################################################################
# This ResDNN model requires single-shell data with one or more b0s, the data
# is fetched and a gradient table is constructed from bvals/bvecs.

# Fetch DWI and GTAB
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
data, affine = load_nifti(hardi_fname)
data = np.squeeze(data)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

###############################################################################
# To accelerate computation, a brain mask must be computed. The resulting mask
# is saved for visual inspection.

mean_b0 = data[..., gtab.b0s_mask]
mean_b0 = np.mean(mean_b0, axis=-1)
_, mask = median_otsu(mean_b0)

mask_labeled, _ = ndi.label(mask)
unique, count = np.unique(mask_labeled, return_counts=True)
val = unique[np.argmax(count[1:]) + 1]
mask[mask_labeled != val] = 0

save_nifti("mask.nii.gz", mask.astype(np.uint8), affine)

###############################################################################
# Using a ResDNN for sh_order_max 8 (default) and load the appropriate weights.
# Fit the data and save the resulting fODF.

resdnn_model = HistoResDNN(verbose=True)
resdnn_model.fetch_default_weights()
predicted_sh = resdnn_model.predict(data, gtab, mask=mask)

save_nifti("predicted_sh.nii.gz", predicted_sh, affine)

###############################################################################
# Preparing the scene using FURY. The mid-coronal slice is cropped around the
# corpus callosum, so that the fODF should be aligned with its curvature and a
# single-fiber population should be visible. The coronal slice lies in the
# x-z plane, so we rotate it about the x axis into the x-y plane faced by the
# default camera. The background image is transposed accordingly.

slice_index = mask.shape[1] // 2
roi = (slice(21, 62), slice(slice_index, slice_index + 1), slice(38, 68))

coronal_to_xy = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]
)

fod_spheres = actor.odf_slicer(
    sh_coeffs=predicted_sh[roi],
    basis_type=resdnn_model.basis_type,
    mask=mask[roi],
    affine=coronal_to_xy,
    scale=0.6,
    visibility=(False, False, True),
)

mean_sh = np.mean(predicted_sh[roi], axis=-1).swapaxes(1, 2)
background_img = actor.data_slicer(
    mean_sh, opacity=0.5, interpolation="nearest", visibility=(False, False, True)
)

window.show([fod_spheres, background_img], title="pred_fODF")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Visualization of the predicted fODF.
#
#
#
# References
# ----------
#
# .. footbibliography::
#
