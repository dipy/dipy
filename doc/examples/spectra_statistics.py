"""
==========================================
SPECTRA Statistics for Group Comparisons
==========================================

This example shows how to test for group differences in a two-dimensional
SPECTRA bundle profile, and how to correct for multiple comparisons with the
hierarchical false discovery rate (hFDR) procedure of SPECTRA.

SPECTRA profiles describe a measure on a grid of along-tract segments and
radial bins, see
:ref:`sphx_glr_examples_built_streamline_analysis_spectra_bundle_profiles.py`. A group
study therefore involves a large number of correlated tests, one per grid cell.
SPECTRA estimates the spatial correlation of the residuals to define blocks of
approximately one resolution element (RESEL), tests the blocks first, and then
tests the cells only within the significant blocks. This improves statistical
power compared to correcting all the cells with the standard
Benjamini-Hochberg procedure.

See :footcite:p:`Feng2026SPECTRA` for further details about the method.

The statistics use the SPECTRA grid and profiles of
:class:`dipy.stats.analysis.SpectraGrid`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from dipy.stats.analysis import (
    SpectraGrid,
    estimate_length_scales,
    fdr_correction,
    fit_grid_lmm,
    hierarchical_fdr,
)

###############################################################################
# Build the SPECTRA grid and a profile
# ------------------------------------
#
# As in the SPECTRA bundle profiles example, we use the arcuate fasciculus of
# one subject (in native space and in atlas space), the model bundle and an FA
# image. The grid is built once from the model bundle, and is then used for
# all the subjects of a study, so that their profiles can be compared cell by
# cell.

af_orig_file, af_mni_file, af_model_file, fa_file = get_fnames(
    name="buan_bundle_profiles"
)

sft_orig = load_tractogram(af_orig_file, reference="same", bbox_valid_check=False)
orig_bundle = sft_orig.streamlines

sft_mni = load_tractogram(af_mni_file, reference="same", bbox_valid_check=False)
bundle = sft_mni.streamlines

sft_model = load_tractogram(af_model_file, reference="same", bbox_valid_check=False)
model_bundle = sft_model.streamlines

fa, affine = load_nifti(fa_file)

grid = SpectraGrid(model_bundle, n_segments=20, n_radial=5)
fa_profile = grid.profile(bundle, orig_bundle, fa, affine)

print("Grid shape:", fa_profile.shape)
print(f"Segment length: {grid.segment_length_actual:.1f} mm")
print(f"Radial bin width: {grid.radial_length_actual:.1f} mm")

###############################################################################
# Simulate a group study
# ----------------------
#
# The example dataset has a single subject. To show the statistics, we
# simulate 120 subjects from three acquisition sites, half of them patients.
# Each simulated profile is the FA profile above plus a spatially smooth
# noise, and the patients have a lower FA in part of the bundle. In a real
# study, the profile of each subject is computed with ``grid.profile`` from
# the bundle and the FA map of that subject.
#
# The cells without streamline points are NaN in ``grid.profile``, and some
# cells are also missing in some of the subjects. Missing cells must stay NaN:
# they are then dropped from the analysis. Setting them to 0 would treat them
# as measurements and bias the statistics.

rng = np.random.default_rng(1)
n_subjects = 120

data = pd.DataFrame(
    {
        "group": np.repeat(["patient", "control"], n_subjects // 2),
        "age": rng.normal(45, 12, n_subjects),
        "site": np.tile(["A", "B", "C"], n_subjects // 3),
    }
)

noise = rng.standard_normal((n_subjects, grid.n_segments, grid.n_radial))
noise = gaussian_filter(noise, sigma=(0, 1.0, 0.5)) * 0.08
profiles = fa_profile + noise

is_patient = (data["group"] == "patient").to_numpy()
profiles[is_patient, 11:17, 1:3] -= 0.05

missing = rng.random(profiles.shape) < 0.15
profiles[missing] = np.nan

###############################################################################
# Fit a linear mixed model in each cell
# -------------------------------------
#
# In each cell, the profile is regressed on the group and the age, with a
# random intercept for the acquisition site. Models are fitted on the subjects
# with a valid measure in the cell, and cells with too few subjects are
# skipped. As in SPECTRA, small isolated regions of cells are not fitted, and
# a cell is only tested if it has a fitted neighbor along the tract and across
# the radial bins. We also fit the reduced model, without the group, since its
# residuals are used to estimate the spatial correlation.

fit = fit_grid_lmm(
    profiles,
    data,
    "group",
    covariates=["age"],
    groups="site",
    continuous=["age"],
    case_group="patient",
    ref_group="control",
    fit_reduced=True,
)

print("Number of tested cells:", fit["mask"].sum(), "of", fit["mask"].size)

###############################################################################
# Estimate the spatial correlation
# --------------------------------
#
# The correlation length scales along the segments and across the radial bins
# are obtained by fitting a Matern correlation to the residuals of the reduced
# model. The grid gives the segment length and the radial bin width, so that
# the length scales are in millimeters. They set the size of a RESEL.

l_s, l_r = estimate_length_scales(fit["residuals_reduced"], grid)
print(f"Length scales: l_s={l_s:.1f} mm, l_r={l_r:.1f} mm")

###############################################################################
# Hierarchical FDR
# ----------------
#
# Stage 1 tests blocks of cells of about one RESEL, at ``alpha_block``.
# Stage 2 tests the cells of the rejected blocks. The cells with an adjusted
# p-value below ``alpha_cell`` are significant. Each bundle is given with its
# grid, its statistics and its length scales. Several bundles can be passed at
# once, Stage 1 is then applied to the blocks of all the bundles.

bundles = {"AF_L": {"grid": grid, "t": fit["t"], "p": fit["p"], "l_s": l_s, "l_r": l_r}}
p_hfdr, blocks = hierarchical_fdr(bundles, alpha_block=0.05)

n_rejected = sum(block["significant"] for block in blocks)
print(f"Stage 1: {n_rejected} of {len(blocks)} blocks rejected")

alpha_cell = 0.15
significant = p_hfdr["AF_L"] < alpha_cell
print(f"Stage 2: {significant.sum()} cells significant at alpha_cell={alpha_cell}")

###############################################################################
# Changing ``alpha_cell`` only changes the threshold, so there is no need to
# run the procedure again. For comparison, we also correct all the cells with
# the Benjamini-Hochberg procedure.

print(f"Stage 2: {(p_hfdr['AF_L'] < 0.25).sum()} cells significant at 0.25")

p_bh = fdr_correction(fit["p"])
print(f"Benjamini-Hochberg: {(p_bh < 0.05).sum()} cells significant at 0.05")

###############################################################################
# Visualize the results
# ---------------------
#
# We show the statistic of each cell, and the cells that are significant with
# the hierarchical procedure.

t_map = np.ma.masked_invalid(np.where(fit["mask"], fit["t"], np.nan))
significant_t = np.ma.masked_where(~significant, t_map)

cmap = plt.colormaps["coolwarm"].copy()
cmap.set_bad("white")
limit = np.abs(t_map).max()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, values, title in zip(
    axes, [t_map, significant_t], ["t-statistic", "Significant cells (hFDR)"]
):
    im = ax.imshow(
        values.T, origin="lower", aspect="auto", cmap=cmap, vmin=-limit, vmax=limit
    )
    ax.set_xlabel("Along-tract segment")
    ax.set_title(title)
axes[0].set_ylabel("Radial bin")
fig.colorbar(im, ax=axes, label="t")

# plt.show()
plt.savefig("spectra_statistics.png")

###############################################################################
# The hierarchical procedure recovers the simulated region while limiting the
# false discoveries elsewhere.


###############################################################################
# References
# ----------
#
# .. footbibliography::
