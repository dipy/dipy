"""
================================================
Reconstruction with ODF-Fingerprinting (ODF-FP)
================================================

ODF-FP :footcite:p:`Baete2019,Filipiak2022` reconstructs the diffusion
orientation distribution function (ODF) by *matching* rather than fitting.
Instead of inverting a parametric model in every voxel, it compares the
measured ODF against a precomputed dictionary of ODF "fingerprints" that were
simulated from a multi-compartment biophysical model, and assigns to the voxel
the microstructure of the best-matching fingerprint.

The key steps are:

1. **Simulate** a dictionary of ODF fingerprints (and their microstructure
   parameters) from a multi-compartment model, reconstructed with GQI.
2. **Reconstruct** the measured ODF of every voxel with the same GQI model.
3. **Align** each ODF so that its main peak points to the pole and **match** it
   to the most similar dictionary fingerprint by penalized cosine similarity.
4. **Read off** the matched fingerprint's ODF, peak directions and
   microstructure (number of fibers, free-water fraction, ...).

Because the alignment rotation is shared across voxels and the matching runs on
whole batches in parallel, ODF-FP scales gracefully to large volumes.

Let us start by importing the relevant modules.
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.odffp import OdffpDictionary, OdffpModel, odffp_peaks
from dipy.segment.mask import median_otsu

###############################################################################
# Download (or locate in cache) the Stanford HARDI dataset and load the data,
# the affine and the gradient table.

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

print(f"data shape: {data.shape}")

###############################################################################
# Compute a brain mask with ``median_otsu`` and keep a single axial slice.

_, brain_mask = median_otsu(data, vol_idx=range(10, 50), median_radius=4, numpass=4)

z = data.shape[2] // 2
dataslice = data[:, :, z : z + 1]
mask = brain_mask[:, :, z : z + 1]

print(f"brain voxels in slice: {int(mask.sum())}")

###############################################################################
# Build the dictionary of ODF fingerprints. ``OdffpDictionary`` simulates the
# diffusion signal of random multi-compartment configurations on a symmetric
# sphere and reconstructs their ODFs with GQI. In practice the dictionary holds
# on the order of one million fingerprints, which is the default; generating it
# takes about a minute and a few gigabytes of memory.

sphere = get_sphere(name="repulsion724")

odf_dict = OdffpDictionary(gtab, sphere=sphere)
odf_dict.generate()

print(f"dictionary fingerprints: {odf_dict.odf.shape[1]}")

###############################################################################
# Instantiate the model and fit the slice. ``fit`` is decorated with
# ``@multi_voxel_fit(batched=True)``, so the voxels are matched in batches and
# the result is a ``MultiVoxelFit`` whose attributes are spatial maps. For
# multi-core execution, pass ``engine="ray"`` and ``n_jobs=<N>`` to ``fit``.

model = OdffpModel(gtab, odf_dict)
fit = model.fit(dataslice, mask=mask)

###############################################################################
# As with FORCE, the peaks are obtained by passing the fit to ``odffp_peaks``,
# which returns a ``PeaksAndMetrics`` object holding the peak directions,
# values and indices, the GFA and QA maps, with the matched ODFs stored as SH
# coefficients. The stored ODFs and the peak values follow the quantitative
# anisotropy convention: the isotropic floor of every ODF is removed and the
# volume is scaled so that the largest peak is 1, so that the size of an ODF
# reflects its anisotropy. It is written to a PAM5 file with ``save_pam``.

from dipy.io.peaks import save_pam

peaks = odffp_peaks(fit)
save_pam("odffp_peaks.pam5", peaks, affine=affine)

print(f"SH coefficients shape: {peaks.shm_coeff.shape}")

###############################################################################
# The matched ODFs can be rendered directly from these SH coefficients with the
# Skyline viewer. The same ``odffp_peaks.pam5`` can also be opened
# interactively with ``dipy_skyline odffp_peaks.pam5``.

from dipy.viz.skyline.app import skyline

skyline(
    visualizer_type="stealth",
    sh_coeffs=[(peaks.shm_coeff, affine, "odf")],
    out_stealth_png="odffp_odfs.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# ODF-FP matched ODFs of the axial slice, rendered from the SH coefficients of
# the peaks object.

###############################################################################
# Each voxel also inherits the microstructure of its matched fingerprint. For
# instance, the free-water fraction follows directly from the matched
# dictionary entry, while the quantitative anisotropy of the main peak reflects
# the amplitude of the matched ODF above its isotropic floor.

free_water = np.zeros(mask.shape)
free_water[mask] = fit.compartment_volume[mask][:, 0]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(np.rot90(free_water[:, :, 0]), cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Free-water fraction")
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(np.rot90(peaks.qa[:, :, 0, 0]), cmap="gray", vmin=0)
axes[1].set_title("QA of the main peak")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("odffp_microstructure.png", dpi=150, bbox_inches="tight")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# ODF-FP microstructure maps read off the matched fingerprints: the free-water
# fraction (left) and the quantitative anisotropy of the main peak (right).
#
# References
# ----------
#
# .. footbibliography::
#
