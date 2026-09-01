"""
=======================
SPECTRA Bundle Profiles
=======================

This example demonstrates Spatial Inference for Tractometry (SPECTRA),
a tractometry framework that characterizes spatial variation within white
matter bundles.

SPECTRA extends along-tract (1D) profiling from Bundle Analytics (BUAN)
to a 2D parameterization that captures both along-tract and radial variation
across bundle cross-sections, defined directly on atlas streamlines.
Streamline points are assigned to an along-tract segment and a radial bin
relative to the atlas bundle, allowing scalar measures to be summarized
within each spatial bin.

See :footcite:p:`Feng2026SPECTRA` for further details about the method.
"""

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.stats.analysis import spectra_assignment_map, spectra_profile

###############################################################################
# Load the data
# -------------
#
# We use the same arcuate fasciculus example dataset that contains
# the original subject bundle, the subject bundle transformed to atlas space,
# a model bundle, and an FA image.

af_orig_file, af_mni_file, af_model_file, fa_file = get_fnames(
    name="buan_bundle_profiles"
)

sft_orig = load_tractogram(af_orig_file, reference="same", bbox_valid_check=False)
orig_bundle = sft_orig.streamlines

sft_mni = load_tractogram(af_mni_file, reference="same", bbox_valid_check=False)
bundle = sft_mni.streamlines

sft_model = load_tractogram(af_model_file, reference="same", bbox_valid_check=False)
model_bundle = sft_model.streamlines

fa_img = nib.load(fa_file)
fa = fa_img.get_fdata()
affine = fa_img.affine


###############################################################################
# SPECTRA spatial assignment
# --------------------------
#
# SPECTRA defines a two-dimensional spatial grid using a model bundle. The
# first dimension represents position along the tract and the second
# represents radial position across the tract.
#
# Here, the arcuate fasciculus is divided into 20 along-tract segments and
# five radial bins.

n_segments = 20
n_radial = 5

s_index, r_index, dist, valid_mask, counts = spectra_assignment_map(
    bundle,
    model_bundle,
    n_segments=n_segments,
    n_radial=n_radial,
)

print("Number of along-tract segments:", n_segments)
print("Number of radial bins:", n_radial)
print("Grid shape:", counts.shape)


###############################################################################
# The output ``s_index`` gives the along-tract assignment of each streamline
# point, while ``r_index`` gives its radial assignment. Together, these define
# the spatial location of each point in the SPECTRA grid.
#
# ``counts`` contains the number of streamline points assigned to each
# spatial bin. Empty bins are masked so that they appear white in the
# visualization.

counts_masked = np.ma.masked_where(counts == 0, counts)

cmap = plt.colormaps["viridis"].copy()
cmap.set_bad("white")

fig, ax = plt.subplots()

im = ax.imshow(
    counts_masked.T,
    origin="lower",
    aspect="auto",
    cmap=cmap,
)

ax.set_xlabel("Along-tract segment")
ax.set_ylabel("Radial bin")
ax.set_title("SPECTRA streamline point distribution")

fig.colorbar(im, ax=ax, label="Number of points")

plt.show()


###############################################################################
# Compute a SPECTRA FA profile
# ----------------------------
#
# The spatial assignments can be used to summarize a scalar microstructural
# measure within each SPECTRA bin. Here, we calculate a two-dimensional FA
# profile.
#
# The subject bundle in common space is used for spatial assignment, while
# ``orig_bundle`` is used to sample FA in the original image space similar to
# BUAN tractometry.

profile = spectra_profile(
    model_bundle,
    bundle,
    orig_bundle,
    fa,
    affine,
    n_segments=n_segments,
    n_radial=n_radial,
)

print("SPECTRA profile shape:", profile.shape)


###############################################################################
# Visualize the SPECTRA profile
# -----------------------------
#
# The resulting profile has shape ``(n_segments, n_radial)``. Each value
# represents the mean FA within a spatial location of the bundle.
#
# Spatial bins without a valid FA value are masked and displayed in white.

profile_masked = np.ma.masked_invalid(profile)

cmap = plt.colormaps["viridis"].copy()
cmap.set_bad("white")

fig, ax = plt.subplots()

im = ax.imshow(
    profile_masked.T,
    origin="lower",
    aspect="auto",
    cmap=cmap,
)

ax.set_xlabel("Along-tract segment")
ax.set_ylabel("Radial bin")
ax.set_title("SPECTRA FA profile")

fig.colorbar(im, ax=ax, label="Fractional anisotropy")

plt.show()


###############################################################################
# SPECTRA therefore preserves the conventional along-tract organization used
# in tractometry while additionally characterizing variation across the bundle.
# The same framework can be used with other scalar measures such as MD, RD,
# or AD, provided that the corresponding metric image and affine are supplied.


###############################################################################
# References
# ----------
#
# .. footbibliography::
