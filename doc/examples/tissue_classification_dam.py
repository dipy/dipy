"""
=====================================================
Tissue Classification using Diffusion MRI with DAM
=====================================================

This example demonstrates tissue classification of white matter (WM) and gray matter
(GM) from multi-shell diffusion MRI data using the Directional Average Maps (DAM)
proposed by :footcite:p:`Cheng2020`. DAM uses the diffusion properties of the tissue to
classify the voxels into WM and GM by fitting a polynomial model to the diffusion
signal. The process involves preprocessing steps including skull-stripping with
median otsu, denoising with Patch2Self, and then perform tissue classification.

Let's start by loading the necessary modules:
"""

import matplotlib.pyplot as plt

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.denoise.patch2self import patch2self
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import dam_classifier
from dipy.viz.plotting import image_mosaic

###############################################################################
# First we fetch the diffusion image, bvalues and bvectors from the cfin dataset.
fraw, fbval, fbvec, t1_fname = get_fnames(name="cfin_multib")

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs=bvecs)

###############################################################################
# After loading the diffusion data, we can apply the median_otsu algorithm to
# skull-strip the data and obtain a binary mask. We can then use the mask to
# denoise the data using the Patch2Self algorithm.
b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
denoised_arr = patch2self(b0_mask, bvals=bvals, b0_denoising=False)

###############################################################################
# Now we can use the DAM classifier to classify the voxels into WM and GM.
# The DAM classifier requires the denoised data, the bvalues, and the mask.
# The DAM classifier returns the WM and GM masks.
# It is important to note that the DAM classifier is a threshold-based classifier
# and the threshold values can be adjusted based on the data. The `wm_threshold`
# parameter controls the sensitivity of the classifier.
# For data like HCP, threshold of 0.5 proves to be a good choice. For data like
# cfin, higher threshold values like 0.7 or 0.8 are more suitable.
wm_mask, gm_mask = dam_classifier(denoised_arr, bvals, wm_threshold=0.7)

###############################################################################
# Now we can visualize the WM and GM masks.

images = [
    data[:, :, data.shape[2] // 2, 0],  # DWI (b0)
    wm_mask[:, :, data.shape[2] // 2],  # White Matter Mask
    gm_mask[:, :, data.shape[2] // 2],  # Grey Matter Mask
]

ax_labels = ["DWI (b0)", "White Matter Mask", "Grey Matter Mask"]
ax_kwargs = [{"cmap": "gray"} for _ in images]

fig, ax = image_mosaic(
    images, ax_labels=ax_labels, ax_kwargs=ax_kwargs, figsize=(20, 5)
)
plt.subplots_adjust(wspace=2.0)
fig.savefig("tissue_classification_dam.png")
plt.show()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Original B0 image (left), White matter (center) and gray matter (right) are
# binary masks as obtained from DAM.
#
#
# References
# ----------
#
# .. footbibliography::
