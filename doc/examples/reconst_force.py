"""
==================================================================================
Reconstruction with FORCE (FORward matching for Complex microstructure Estimation)
==================================================================================

FORCE :footcite:p:`Shah2025` is a forward-modeling paradigm that reframes how
diffusion MRI data are analyzed. Instead of inverting the measured signal,
FORCE simulates a large library of biologically plausible intra-voxel fiber
configurations and tissue compositions. It then identifies the best-matching
library entry for each voxel by operating directly in signal space.

The key steps are:

1. **Simulate** a large library of tissue configurations and their predicted
   diffusion signals.
2. **Index** the library using a fast inner-product search index.
3. **Match** each measured voxel signal to its nearest neighbor(s) in the
   library.
4. **Read off** microstructure parameters (FA, MD, WM/GM/CSF fractions,
   fiber count, dispersion, neurite density, …) from the matched entries.

Because FORCE never fits a parametric model to each voxel independently, it
scales gracefully to arbitrary acquisition protocols and can be run in
parallel across CPU cores or Ray workers.

Let us start by importing the relevant modules.
"""

import numpy as np

###############################################################################
# ``dipy.io`` handles data loading; ``dipy.data`` provides the Stanford HARDI
# dataset that ships with DIPY.
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import save_pam
from dipy.viz.plotting import image_mosaic

###############################################################################
# Download (or locate in cache) the Stanford HARDI dataset.  The first call
# fetches ~87 MB from the internet; subsequent calls reuse the local copy.

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

print(f"data shape: {data.shape}")

###############################################################################
# Create a brain mask so that we only fit voxels inside the brain.

from dipy.segment.mask import median_otsu

_, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=4, numpass=4)

print(f"mask shape: {mask.shape}, brain voxels: {mask.sum()}")

###############################################################################
# Instantiate the FORCE model.  At construction time we specify matching
# parameters; the simulation library is created separately via ``generate()``.
#
# * ``n_neighbors`` — number of library entries to retrieve per voxel query.
# * ``penalty`` — fiber-complexity penalty subtracted from the match score,
#   scaled by the number of fibers. Larger values bias the match towards
#   simpler configurations.
# * ``use_posterior`` — when ``True``, parameters are averaged over the
#   ``n_neighbors`` nearest entries weighted by a softmax posterior; when
#   ``False`` (default) only the single best-match entry is used.
# * ``posterior_beta`` — softmax temperature of that posterior. Larger values
#   concentrate the weights on the closest matches.
# * ``compute_odf`` — when ``True``, a posterior ODF is assembled per voxel and
#   exposed as ``fit.odf``.
#
# The posterior over the ``n_neighbors`` matches is always computed, even when
# ``use_posterior=False``. That flag only decides where the *point estimates*
# come from: the posterior mean, or the single best match. The uncertainty and
# ambiguity maps shown further below are derived from the posterior either way.
# The one exception is ``fit.entropy``, the entropy of the posterior weights,
# which is only meaningful in posterior mode and is ``NaN`` otherwise.

from dipy.reconst.force import FORCEModel, force_peaks

model = FORCEModel(
    gtab,
    n_neighbors=50,
    use_posterior=True,
    verbose=True,
)

###############################################################################
# Generate the simulation library.  For this tutorial, we use 500,000 simulations.
#
# When ``use_cache=True`` (the default), FORCE stores the generated library in
# ``~/.dipy/force_simulations/`` and reloads it automatically on subsequent
# runs with identical parameters, skipping regeneration entirely.

model.generate(
    num_simulations=500000,
    num_cpus=-1,
    verbose=True,
    use_cache=False,
)

###############################################################################
# Fit the model to the data.
#
# ``model.fit()`` runs serially by default. The ``ray`` engine is considerably
# faster, as it parallelises the matching and post-processing across worker
# processes instead of a single one. To use it, pass ``engine="ray"`` (and
# optionally ``n_jobs=<N>``)::
#
#     fit = model.fit(data, mask=mask, engine="ray", n_jobs=-1)
#
# The ``ray`` engine requires Ray to be installed (``pip install ray``); if it
# is not available the fit falls back to serial execution.

fit = model.fit(data, mask=mask, n_jobs=-1, verbose=True)

###############################################################################
# The ``fit`` object is a ``MultiVoxelFit`` container.  Its attributes are
# 3-D arrays with the same spatial shape as ``data[..., 0]``.  Masked voxels
# contain zeros.

fa_map = fit.fa
md_map = fit.md
rd_map = fit.rd
wm_fraction = fit.wm_fraction
gm_fraction = fit.gm_fraction
csf_fraction = fit.csf_fraction
num_fibers = fit.num_fibers
dispersion = fit.dispersion
nd = fit.nd
uncertainty = fit.uncertainty
ambiguity = fit.ambiguity

print(f"FA  — min: {fa_map[mask].min():.3f}  max: {fa_map[mask].max():.3f}")
print(f"MD  — min: {md_map[mask].min():.6f}  max: {md_map[mask].max():.6f}")

###############################################################################
# To save the peaks generated from the FORCE directly, we need to call the force_peaks
# function on the fitted object.  This will return a
# PeaksAndMetrics object containing the peak directions, values, and indices, which can
# be saved to disk using save_pam.
peaks = force_peaks(fit)

###############################################################################
# Now lets import the save_pam function and save the peaks to disk as a .pam5 file.
# The affine is needed to ensure that the peaks are correctly aligned with the original
# data.

save_pam("force_peaks.pam5", peaks, affine=affine)


###############################################################################
# Save selected maps to disk as NIfTI files.

save_nifti("force_fa.nii.gz", fa_map.astype(np.float32), affine)
save_nifti("force_md.nii.gz", md_map.astype(np.float32), affine)
save_nifti("force_wm_fraction.nii.gz", wm_fraction.astype(np.float32), affine)
save_nifti("force_num_fibers.nii.gz", num_fibers.astype(np.float32), affine)
save_nifti("force_gm_fraction.nii.gz", gm_fraction.astype(np.float32), affine)
save_nifti("force_csf_fraction.nii.gz", csf_fraction.astype(np.float32), affine)
save_nifti("force_uncertainty.nii.gz", uncertainty.astype(np.float32), affine)

###############################################################################
# Visualize a representative axial slice.

mid_z = (fa_map.shape[2] // 2) - 2


def _slice(arr):
    return np.rot90(arr[:, :, mid_z])


# Each map family gets its own figure and its own colormap: DTI -> viridis,
# NODDI-like -> inferno, and tissue partial volume estimates -> gray.
image_mosaic(
    [_slice(fa_map), _slice(md_map), _slice(rd_map)],
    ax_labels=["FA", "MD", "RD"],
    ax_kwargs=[
        {"cmap": "viridis", "vmin": 0, "vmax": 1},
        {"cmap": "viridis"},
        {"cmap": "viridis"},
    ],
    figsize=(15, 5),
    filename="force_maps_dti.png",
)


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# DTI maps for an axial slice of the Stanford HARDI dataset.

image_mosaic(
    [_slice(nd), _slice(dispersion)],
    ax_labels=["NDI", "ODI"],
    ax_kwargs=[{"cmap": "inferno", "vmin": 0, "vmax": 1}, {"cmap": "inferno"}],
    figsize=(10, 5),
    filename="force_maps_noddi.png",
)


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# NODDI-like maps: neurite density index and orientation dispersion index.

image_mosaic(
    [
        _slice(wm_fraction),
        _slice(gm_fraction),
        _slice(csf_fraction),
        _slice(num_fibers),
    ],
    ax_labels=["WM fraction", "GM fraction", "CSF fraction", "Number of fibers"],
    ax_kwargs=[
        {"cmap": "gray", "vmin": 0, "vmax": 1},
        {"cmap": "gray", "vmin": 0, "vmax": 1},
        {"cmap": "gray", "vmin": 0, "vmax": 1},
        {"cmap": "viridis", "vmin": 0, "vmax": 3},
    ],
    figsize=(20, 5),
    filename="force_maps_tissue.png",
)


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Tissue partial volume estimates (WM/GM/CSF fractions, *gray*) and the number
# of fiber populations per voxel.
#
# FORCE also reports, for each microstructure parameter, an **uncertainty** map
# (spread of the posterior) and an **ambiguity** map (multi-modality of the
# posterior), both normalised to the prior range. They are available per
# parameter as ``fit.uncertainty_<param>`` and ``fit.ambiguity_<param>``, for
# example ``fit.uncertainty_fa`` or ``fit.ambiguity_wm_fraction``, alongside
# the voxel-level ``fit.uncertainty`` and ``fit.ambiguity``. As noted above,
# these come from the posterior regardless of ``use_posterior``. Below we show
# them for the NODDI parameters NDI and ODI.

image_mosaic(
    [
        _slice(fit.uncertainty_nd),
        _slice(fit.ambiguity_nd),
        _slice(fit.uncertainty_dispersion),
        _slice(fit.ambiguity_dispersion),
    ],
    ax_labels=["NDI uncertainty", "NDI ambiguity", "ODI uncertainty", "ODI ambiguity"],
    ax_kwargs=[{"cmap": "hot"}] * 4,
    figsize=(20, 5),
    filename="force_uncertainty_ambiguity.png",
)


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Per-microstructure uncertainty (left) and ambiguity (right) maps for NDI and
# ODI.
#
# Using a different signal model
# ------------------------------
#
# FORCE can also be used with different tissue models. Matching only sees the
# library's signal matrix, so it does not depend on how those signals were
# made. Its speed is set by the library size and the number of gradient
# directions, and stays the same when the model changes.
#
# There are two ways to substitute your own model. The direct one is to edit
# the signal generators in ``dipy/sims/_force_core.pyx`` and rebuild; the
# library then carries your signals and everything downstream works unchanged.
# The other avoids a rebuild: generate the library yourself and hand it to the
# model::
#
#     model = FORCEModel(gtab, simulations=my_simulations)
#
# In both cases the per-voxel parameters are read straight off the matched
# entry, so the dictionary has to carry a value per simulation for each
# parameter you want back: ``signals`` and ``labels`` at minimum, plus
# ``num_fibers``, ``dispersion``, ``nd``, ``wm_fraction``, ``gm_fraction``,
# ``csf_fraction``, ``fa``, ``md``, ``rd`` and ``fraction_array``. Optional
# extras (micro-FA, the DKI metrics, ``odfs``) are picked up when present.
# Whatever you supply is what the fit reports back.
#
# References
# ----------
#
# .. footbibliography::
#
