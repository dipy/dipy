"""
==================================================================================
Reconstruction with FORCE (FORward modeling for Complex microstructure Estimation)
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
# * ``use_posterior`` — when ``True``, parameters are averaged over the
#   ``n_neighbors`` nearest entries weighted by a softmax posterior; when
#   ``False`` (default) only the single best-match entry is used.

from dipy.reconst.force import FORCEModel, force_peaks

model = FORCEModel(
    gtab,
    n_neighbors=50,
    use_posterior=False,
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
# For serial execution (one CPU), simply call ``model.fit()``.  To exploit
# multiple cores, pass ``engine="ray"`` and ``n_jobs=<N>``.  The
# ``@multi_voxel_fit`` decorator handles chunking, masking, and result
# assembly automatically.

fit = model.fit(data, mask=mask, n_jobs=-1, verbose=True)

###############################################################################
# The ``fit`` object is a ``MultiVoxelFit`` container.  Its attributes are
# 3-D arrays with the same spatial shape as ``data[..., 0]``.  Masked voxels
# contain zeros.

fa_map = fit.fa
md_map = fit.md
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

import matplotlib.pyplot as plt

mid_z = (fa_map.shape[2] // 2) - 5

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

panels = [
    (axes[0, 0], fa_map, "FA", "gray", 0, 1),
    (axes[0, 1], md_map, "MD", "hot", None, None),
    (axes[0, 2], wm_fraction, "WM Fraction", "gray", 0, 1),
    (axes[1, 0], num_fibers, "Num Fibers", "viridis", 0, 3),
    (axes[1, 1], gm_fraction, "GM Fraction", "gray", 0, 1),
    (axes[1, 2], csf_fraction, "CSF Fraction", "Blues", 0, 1),
]

for ax, arr, title, cmap, vmin, vmax in panels:
    kwargs = {"cmap": cmap}
    if vmin is not None:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    im = ax.imshow(np.rot90(arr[:, :, mid_z]), **kwargs)
    ax.set_title(f"{title} (slice {mid_z})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("force_maps.png", dpi=150, bbox_inches="tight")
# plt.show()


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# FORCE microstructure maps for an axial slice of the Stanford HARDI dataset.
# From left to right, top to bottom: FA, MD, WM fraction, number of fibers,
# GM fraction, CSF fraction.
#
# References
# ----------
#
# .. footbibliography::
#
