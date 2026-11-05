"""
==========================================
SPECTRA Statistics for Group Comparisons
==========================================

This example shows how to test for group differences in a two-dimensional
SPECTRA bundle profile, and how to correct for multiple comparisons with the
hierarchical false discovery rate (hFDR) procedure of SPECTRA.

SPECTRA profiles describe a measure (here FA) on a grid of along-tract segments
and radial bins, see
:ref:`sphx_glr_examples_built_streamline_analysis_spectra_bundle_profiles.py`.
A group study therefore means one statistical test per grid cell. This raises
two problems:

* Many tests give false positives: with 100 tests of a measure that does not
  differ between the groups, we expect about 5 of them to be significant at
  p < 0.05.
* The tests are not independent: neighbouring cells sample nearby tissue, so
  their values tend to vary together. The usual Benjamini-Hochberg (BH)
  correction treats every cell on its own and ignores this.

SPECTRA estimates how far this correlation extends, groups the cells into
blocks of about one resolution element (RESEL), tests the blocks first, and
then tests the cells only inside the blocks that are significant.

The steps of this example are:

1. build the SPECTRA grid and a profile for each subject (here simulated),
2. fit a linear mixed model in each cell,
3. estimate the spatial correlation of the residuals,
4. correct for multiple comparisons with the hierarchical FDR, and compare it
   with BH, since we know where the true effect is.

See :footcite:p:`Feng2026SPECTRA` for further details about the method.
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
    matern_correlation,
)

###############################################################################
# Build the SPECTRA grid and a profile
# ------------------------------------
#
# As in the SPECTRA bundle profiles example, we use the arcuate fasciculus of
# one subject (in native space and in atlas space), the model bundle and an FA
# image.
#
# The grid is built once, from the model bundle, and then used for all the
# subjects. A cell therefore covers the same part of the bundle in every
# subject, which is what makes a cell-by-cell comparison possible. We ask for
# cells of 5 mm, along the tract and across it. Smaller cells would describe the
# bundle in more detail, but each cell would contain fewer streamline points, so
# its value would be noisier, and there would be more cells to test and to
# correct for.

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

grid = SpectraGrid(model_bundle, segment_length=5.0, radial_length=5.0)
fa_profile = grid.profile(bundle, orig_bundle, fa, affine)
has_data = np.isfinite(fa_profile)

print("Grid shape:", fa_profile.shape)
print(f"Segment length: {grid.segment_length_actual:.1f} mm")
print(f"Radial bin width: {grid.radial_length_actual:.1f} mm")
print("Cells with data:", has_data.sum(), "of", fa_profile.size)

###############################################################################
# The grid is a rectangle of 26 segments by 9 radial bins, and the bundle only
# fills part of it: it is thick in the middle of the tract and thin near its
# ends, and the cells with too few streamline points are masked. The cells
# outside of the bundle have no data. They are NaN, not 0. A missing measure
# must stay missing: a 0 would be read as a real FA of 0 and would bias the
# statistics, so the functions below leave the NaN cells out.

###############################################################################
# Simulate a group study
# ----------------------
#
# The example dataset has a single subject, so we simulate a study. In a real
# study, the profile of each subject is computed with ``grid.profile`` from the
# bundle and the FA map of that subject.
#
# We simulate 120 subjects from three acquisition sites, half of them patients.
# The profile of each subject is the FA profile above plus noise. The noise is
# smooth along and across the bundle (``gaussian_filter``), because in real data
# the values of neighbouring cells vary together. This is the spatial
# correlation that SPECTRA estimates in the next steps. With independent noise,
# the cells would be independent tests and there would be nothing to model.
#
# The patients have a lower FA in one part of the bundle. We simulate it as a
# smooth patch: the FA is lowered by 0.08 at its center (segment 14, radial bin
# 4), which is about 13% of the FA there, and the decrease goes down to 0 at the
# border of the patch, as in a focal lesion. A smooth patch is closer to a real
# effect than a block with sharp edges, and its border is hard to detect. This
# is the true effect, and we will check the tests against it. Finally, 15% of the
# values are set to NaN at random: in a real study, a cell is often missing for
# some subjects.

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

segment, radial = np.meshgrid(
    np.arange(grid.n_segments), np.arange(grid.n_radial), indexing="ij"
)
squared_distance = ((segment - 14) / 6) ** 2 + ((radial - 4) / 3) ** 2
effect = 0.08 * np.clip(1 - squared_distance, 0, None)
in_effect = (effect > 0) & has_data
is_patient = (data["group"] == "patient").to_numpy()
profiles[is_patient] -= effect

missing = rng.random(profiles.shape) < 0.15
profiles[missing] = np.nan

print("Cells with an effect:", in_effect.sum())
print("  of which at least half of the maximum:", (in_effect & (effect >= 0.04)).sum())

###############################################################################
# Before any test, we look at the data. The first map is the bundle itself: the
# FA of the subject in each cell of the grid, with the cells outside of the
# bundle in white. The second map is the simulated effect alone, the FA that the
# patients lose in each cell. The third map is what we would see in the data:
# the mean FA of the patients minus the mean FA of the controls. The black line
# outlines the cells with an effect.

patient_mean = np.ma.masked_invalid(profiles[is_patient]).mean(axis=0)
control_mean = np.ma.masked_invalid(profiles[~is_patient]).mean(axis=0)
observed = patient_mean - control_mean

cmap_fa = plt.colormaps["viridis"].copy()
cmap_fa.set_bad("white")
cmap_diff = plt.colormaps["coolwarm"].copy()
cmap_diff.set_bad("white")

maps = [
    ("FA of the subject (the bundle)", np.ma.masked_invalid(fa_profile), cmap_fa, "FA"),
    (
        "Simulated effect",
        np.ma.masked_where(~has_data, -effect),
        cmap_diff,
        "FA change",
    ),
    ("Observed: patients - controls", observed, cmap_diff, "FA difference"),
]

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, (title, values, cmap, label) in zip(axes, maps):
    if cmap is cmap_diff:
        limit = 0.09
        im = ax.imshow(values.T, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
    else:
        im = ax.imshow(values.T, origin="lower", cmap=cmap)
    ax.contour(in_effect.T, levels=[0.5], colors="k", linewidths=1)
    ax.set_title(title)
    ax.set_ylabel("Radial bin")
    fig.colorbar(im, ax=ax, label=label, fraction=0.025, pad=0.02)
axes[-1].set_xlabel("Along-tract segment")
fig.tight_layout()

# plt.show()
plt.savefig("spectra_statistics_data.png")

###############################################################################
# The first map shows the shape of the bundle: the cells with data form a band,
# thick in the middle and thin at both ends. The second map shows where we put
# the effect, a patch inside the thick part of the bundle. The third map shows
# the same patch in the data, but with noise: the patients are lower than the
# controls around the center of the patch, and the cells at its border, where the
# true effect is small, cannot be told from the noise by eye.
#
# The effect is easy to see in the middle of the patch because it is strong and
# the groups are large. With a weaker effect or fewer subjects, the map alone
# could not tell the effect from noise. The map also ignores the age and the site
# of the subjects, and it does not say how likely such a difference is by chance.
# This is what the tests are for.

###############################################################################
# Fit a linear mixed model in each cell
# -------------------------------------
#
# In each cell, the FA is regressed on the group and the age, with a random
# intercept for the acquisition site::
#
#     FA ~ group + age + (1 | site)
#
# The group is the effect of interest. The age is a covariate, since FA changes
# with age and the subjects have different ages. The random intercept absorbs
# the shift that all the subjects of a site share, for example because of the
# scanner, without using up one parameter for each site.
#
# The profile values are z-scored before the fit, so the group coefficient
# ``beta`` is in standard deviations of the FA. The cells that are NaN for a
# subject are left out of the model of that cell, and a cell with too few
# subjects is not fitted. As in SPECTRA, small isolated regions of cells are not
# fitted, and a cell is only tested if it has a fitted neighbor along the tract
# and across the radial bins. We also fit the reduced model, without the group,
# since we need its residuals in the next step.

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

tested = fit["mask"]
print("Cells with data:", has_data.sum())
print("Tested cells:", tested.sum())
print("Cells with an effect that are tested:", (tested & in_effect).sum())
print(f"Cells with an effect: mean t = {fit['t'][tested & in_effect].mean():.1f}")
print(
    f"Cells without effect: mean |t| = {np.abs(fit['t'][tested & ~in_effect]).mean():.1f}"
)

###############################################################################
# Out of the cells with data, most are tested: the others have no fitted
# neighbors on both sides, mainly at the thin ends of the bundle. All the cells
# with an effect are tested.
#
# In the cells with an effect, the mean t is negative: the patients are lower
# than the controls. It is an average over the whole patch, including its weak
# border. In the cells without effect, the mean absolute t is close to the 0.8
# that noise alone gives, so those cells behave as expected when there is no
# effect.

###############################################################################
# Estimate the spatial correlation
# --------------------------------
#
# The tests are not independent, so we measure how far the correlation of
# neighbouring cells extends. A Matern correlation ``exp(-d / l)`` is fitted to
# the correlation of the residuals, separately along the tract and across the
# radial bins. This gives a length scale ``l`` in millimeters (the grid gives
# the segment length and the radial bin width): the larger it is, the farther
# two cells vary together.
#
# SPECTRA uses the residuals of the reduced model, which is the model without
# the group. It is the model of the data when there is no group difference, and
# the hierarchical FDR needs the correlation that the cells have in that case.
# The price is that a real effect stays in these residuals and makes them
# slightly smoother. We can see how much by comparing with the residuals of the
# full model.

l_s, l_r = estimate_length_scales(fit["residuals_reduced"], grid)
print(f"Length scales: l_s={l_s:.1f} mm, l_r={l_r:.1f} mm")

l_s_full, l_r_full = estimate_length_scales(fit["residuals"], grid)
print(f"With the full model: l_s={l_s_full:.1f} mm, l_r={l_r_full:.1f} mm")

along = matern_correlation(grid.segment_length_actual, l_s)
across = matern_correlation(grid.radial_length_actual, l_r)
print(f"Correlation of two neighbouring cells: {along:.2f} along, {across:.2f} across")

###############################################################################
# Along the tract, the length scale is about 3 segments, and two neighbouring
# segments have a correlation of about 0.7. Across the radial bins, the length
# scale is smaller than one bin, and the correlation of neighbouring bins is
# only about 0.3. The noise is smoother along the tract than across it, as we
# simulated.
#
# The residuals of the full model give shorter length scales, by about 2 mm
# along the tract and 0.4 mm across. So the effect that stays in the reduced
# residuals does make them a little smoother, but the length scales stay close.
#
# These length scales set the size of a RESEL. SPECTRA takes the sides of a
# block as 2.355 times the length scale, the factor that links the standard
# deviation of a Gaussian to its full width at half maximum. Here a block is
# about 7 segments by 2 radial bins, so up to 14 cells.

###############################################################################
# Hierarchical FDR
# ----------------
#
# Stage 1 combines the p-values of the cells of each block into one p-value for
# the block, and corrects the blocks with the BH procedure at ``alpha_block``.
# Stage 2 looks only inside the blocks that were rejected: it computes the
# p-value of each cell given that its block was selected, and corrects these
# p-values within the block. Each bundle is given with its grid, its statistics
# and its length scales. Several bundles can be corrected together: Stage 1
# then uses the blocks of all the bundles.

bundles = {"AF_L": {"grid": grid, "t": fit["t"], "p": fit["p"], "l_s": l_s, "l_r": l_r}}
p_hfdr, blocks = hierarchical_fdr(bundles, alpha_block=0.05)

n_rejected = sum(block["significant"] for block in blocks)
print(f"Stage 1: {n_rejected} of {len(blocks)} blocks rejected")
for i, block in enumerate(blocks):
    if block["significant"]:
        segments, radial_bins = np.array(block["cells"]).T
        print(
            f"  Block {i}: {len(block['cells']):2d} cells, "
            f"{in_effect[segments, radial_bins].sum():2d} with an effect"
        )

###############################################################################
# The blocks that are rejected are the ones that cover the patch. The cells with
# an adjusted p-value below ``alpha_cell`` are significant. We take
# ``alpha_cell=0.15``, a level reported by SPECTRA. It looks lenient, but it does
# not play the same role as 0.15 in the BH procedure. It is only applied to the
# cells of blocks that already passed Stage 1 at 0.05, and their p-values are
# conditional on that selection. We compare it with BH over all the cells, at the
# usual 0.05 and at the same 0.15, using our knowledge of the true effect.

alpha_cell = 0.15
p_bh = fdr_correction(fit["p"])


def count(significant):
    """Cells with an effect found, and cells without effect found."""
    return (significant & in_effect).sum(), (significant & ~in_effect).sum()


results = {
    f"hFDR, alpha_cell={alpha_cell}": p_hfdr["AF_L"] < alpha_cell,
    "BH, level 0.05": p_bh < 0.05,
    "BH, level 0.15": p_bh < 0.15,
}
print(f"{'':22s} {'cells with effect found':>24s} {'cells without effect':>22s}")
for name, significant in results.items():
    found, false = count(significant)
    print(f"{name:22s} {found:>17d} of {in_effect.sum()} {false:>22d}")

###############################################################################
# All the methods find most of the cells with an effect, and miss the cells at
# the border of the patch, where the effect is too small to detect with this
# noise. They differ in how they trade detections against false discoveries. BH
# at 0.05 finds two more cells than the hierarchical FDR (23 and 21). BH at the
# same 0.15 level finds 26, but it also adds 2 cells that have no effect, which
# the hierarchical FDR does not. So the hierarchical FDR is not more sensitive in
# this simulation, but its 0.15 does not bring false discoveries. One simulation
# cannot say which method is better in general, it only shows how they behave
# when the truth is known.
#
# To check that 0.15 is not lenient, we shuffle the group labels. The profiles
# are the same, but the labels now carry no information, so no cell should be
# significant.

shuffled = data.copy()
shuffled["group"] = rng.permutation(data["group"].to_numpy())
fit_null = fit_grid_lmm(
    profiles,
    shuffled,
    "group",
    covariates=["age"],
    groups="site",
    continuous=["age"],
    case_group="patient",
    ref_group="control",
    fit_reduced=True,
)
l_s_null, l_r_null = estimate_length_scales(fit_null["residuals_reduced"], grid)
null_bundle = {
    "AF_L": {
        "grid": grid,
        "t": fit_null["t"],
        "p": fit_null["p"],
        "l_s": l_s_null,
        "l_r": l_r_null,
    }
}
p_hfdr_null, _ = hierarchical_fdr(null_bundle)
p_bh_null = fdr_correction(fit_null["p"])
print("Shuffled labels, significant cells:")
print("  hFDR:", (p_hfdr_null["AF_L"] < alpha_cell).sum())
print(
    "  BH at 0.05:", (p_bh_null < 0.05).sum(), "| BH at 0.15:", (p_bh_null < 0.15).sum()
)

###############################################################################
# No cell is significant with any method. A single shuffle is only an
# illustration: the false discovery rate is a property of many repetitions, and
# a few false cells can appear in some of them.

###############################################################################
# Visualize the results
# ---------------------
#
# We show the t-statistic of each tested cell, then the cells that are
# significant with the hierarchical FDR and with BH at 0.05. The black line
# outlines the cells with an effect, so a significant cell outside of it is a
# false discovery.

t_map = np.ma.masked_invalid(np.where(tested, fit["t"], np.nan))
panels = {
    "t-statistic of the tested cells": t_map,
    "Significant with the hierarchical FDR (alpha_cell=0.15)": np.ma.masked_where(
        ~(p_hfdr["AF_L"] < alpha_cell), t_map
    ),
    "Significant with BH (level 0.05)": np.ma.masked_where(~(p_bh < 0.05), t_map),
}

cmap = plt.colormaps["coolwarm"].copy()
cmap.set_bad("white")
limit = np.abs(t_map).max()

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, (title, values) in zip(axes, panels.items()):
    im = ax.imshow(values.T, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
    ax.contour(in_effect.T, levels=[0.5], colors="k", linewidths=1)
    ax.set_title(title)
    ax.set_ylabel("Radial bin")
    fig.colorbar(im, ax=ax, label="t", fraction=0.025, pad=0.02)
axes[-1].set_xlabel("Along-tract segment")
fig.tight_layout()

# plt.show()
plt.savefig("spectra_statistics.png")

###############################################################################
# The t-statistic map has a patch of large negative values at the place of the
# effect, and small values elsewhere. The significant cells of the hierarchical
# FDR fill the middle of the patch and stay inside the outline. BH at 0.05
# covers the patch a little more.

###############################################################################
# Command line
# ------------
#
# The same analysis can be run from the command line, in this order:
#
# 1. ``dipy_spectra_profile``: once for each subject. It computes the SPECTRA
#    profiles of the bundles of that subject for each metric map (step 1).
# 2. ``dipy_spectra_lmm``: once for each bundle, when all the subjects have
#    their profiles. It fits the models of step 2 and estimates the length
#    scales of step 3. It is separate from the next one so that the bundles can
#    be fitted in parallel jobs.
# 3. ``dipy_spectra_fdr``: once, on the results of all the bundles. It corrects
#    them together with the hierarchical FDR or BH (step 4) and writes the
#    tables of the significant cells.
#
# ``dipy_spectra_hfdr_external`` is used instead of the last two when the
# statistics of each cell were computed by another software, for example a mixed
# model in R. It reads a table of statistics and residuals and applies the
# hierarchical FDR to them.
#
# .. code-block:: bash
#
#     dipy_spectra_profile atlas/ bundles/ native_bundles/ metrics/ \
#         --out_dir profiles/sub-01
#     dipy_spectra_lmm metadata.csv profiles/ atlas/ --target_col dx \
#         --target_case AD --target_ref CN --out_dir fits
#     dipy_spectra_fdr fits/ atlas/ --out_dir results

###############################################################################
# References
# ----------
#
# .. footbibliography::
#
