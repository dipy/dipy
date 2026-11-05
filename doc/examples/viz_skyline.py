"""
====================================================
Interactive Visualization with DIPY Skyline (Python)
====================================================

This tutorial shows how to use DIPY Skyline programmatically from Python to
visualize brain images, tractograms, peak directions and ODFs. Skyline is
DIPY's GPU-based interactive viewer (built on FURY >= 2.1.0) and the successor
of Horizon :footcite:p:`Garyfallidis2019`. The ``horizon()`` function is
deprecated and forwards to Skyline.

Skyline can also be launched from the command line with ``dipy_skyline`` (see
:ref:`viz_flow`). Here we call it directly from a script, which allows more
flexible and reproducible visualization workflows.

What is Skyline?
----------------

Skyline is DIPY's interactive visualization system that allows you to:

- Visualize and interact with tractograms (streamlines)
- Cluster streamlines using QuickBundlesX
- Display anatomical images (T1, T2, FA, etc.) as slices
- Visualize peak directions
- **Render ODF glyphs from spherical harmonics (SH) coefficients (new
  compared to Horizon)**
- Display ROIs and brain surfaces (cortical meshes)
- **Keep slices synchronized across images, peaks and ODFs (new compared to
  Horizon)**
- Save offscreen snapshots of your visualizations

Getting Started
---------------

First, let's import the necessary modules and check that we have FURY
installed (required for 3D visualization):

"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.reconst.dti import TensorModel
from dipy.reconst.force import FORCEModel, force_peaks
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import deterministic_tracking
from dipy.utils.logging import logger
from dipy.viz.skyline.app import skyline

###############################################################################
# Basic Example: Visualizing a Brain Image
# =========================================
#
# Let's start with the simplest use case: visualizing a 3D brain image.
# We load the Stanford HARDI dataset and visualize its b0 volume (first 3D
# volume of the 4D data).

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)

# Extract the first b0 image (this dataset has multiple b0 volumes)
b0_data = data[..., 0]

###############################################################################
# Images are provided to Skyline as tuples ``(data, affine, name)``, where
# ``name`` is the label displayed in the sidebar. The data is always placed in
# world coordinates through ``affine``, so no extra flag is needed to choose
# between voxel and world coordinates.

_skyline = skyline(
    images=[(b0_data, affine, "b0")],
    out_stealth_png="skyline_basic_image.png",
)

###############################################################################
# Visualizing Tractograms with Clustering
# ========================================
#
# Skyline can cluster streamlines with QuickBundlesX, which makes it easier to
# explore large tractograms.

# Create FA map and white matter mask for tractography
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

# Create white matter mask first
_, white_matter_mask = median_otsu(data, vol_idx=range(10, 50), numpass=3)

logger.info("Computing tensor model...")
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=white_matter_mask)
fa_data = tenfit.fa

logger.info("Generating streamlines...")

# Create seeds
seeds = utils.seeds_from_mask(white_matter_mask, affine, density=[2, 2, 2])
# Use a subset of seeds for faster demo execution
np.random.shuffle(seeds)
seeds = seeds[:5000]

# Create stopping criterion from FA
stopping_criterion = ThresholdStoppingCriterion(fa_data, 0.2)

# Perform tractography using the modern functional tracker
# Fit the FORCE model to get posterior ODFs, stored as SH coefficients, for
# deterministic tracking :footcite:p:`Shah2026`
logger.info("Computing FORCE model...")
force_model = FORCEModel(
    gtab,
    n_neighbors=50,
    use_posterior=True,
    posterior_beta=2000,
    compute_odf=True,
)
force_model.generate(num_simulations=500000, num_cpus=-1)
force_fit = force_model.fit(data, mask=white_matter_mask)
force_pam = force_peaks(force_fit)

# Generate streamlines with built-in length filtering
# min_len and max_len filter during tracking (no post-processing needed)
streamline_generator = deterministic_tracking(
    seeds,
    stopping_criterion,
    affine,
    step_size=0.5,
    min_len=10,  # Minimum length in mm (filters short streamlines)
    max_len=500,  # Maximum length in mm
    sh=force_pam.shm_coeff,
    max_angle=30.0,
    sphere=default_sphere,
    nbr_threads=0,  # 0 = auto-detect cores; set to >0 for explicit threading
    return_all=False,
)

streamlines = Streamlines(streamline_generator)
logger.info(f"Generated {len(streamlines)} streamlines (min_len=10mm, max_len=500mm)")

# Create StatefulTractogram
# Using the image object directly ensures proper coordinate system handling
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)

###############################################################################
# Now visualize with clustering enabled. Tractograms are provided as tuples
# ``(sft, name)``. We also overlay the tractogram on FA for anatomical context.
#
# Clusters with fewer than ``cluster_size_thr`` streamlines, or with a mean
# length below ``cluster_length_thr`` mm, are hidden. Both thresholds can also
# be changed live in the tractogram panel. Horizon's upper bounds
# (``length_lt`` and ``clusters_lt``) have no Skyline counterpart.

_skyline = skyline(
    images=[(fa_data, affine, "FA")],
    tractograms=[(sft, "FORCE tractogram")],
    is_light_version=True,
    is_cluster=True,
    cluster_thr=15.0,
    cluster_size_thr=10,
    cluster_length_thr=30,
    out_stealth_png="skyline_tractography.png",
)

###############################################################################
# Visualizing Peak Directions
# ============================
#
# Peak directions are provided as tuples
# ``(peak_dirs, affine, name, peak_values)``. ``peak_values`` sets the length
# of each peak line, in voxel units. Raw tensor ODF values are not on a voxel
# scale, so we normalize them by the strongest peak of each voxel: first peaks
# get unit length and secondary peaks are shorter. The line length can also be
# adjusted in the peaks panel. Here we compute peaks from the tensor model and
# display them over FA.

logger.info("Computing peaks...")

# Compute peaks from the tensor model using the white matter mask
dti_peaks = peaks_from_model(
    tenmodel,
    data,
    default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=white_matter_mask,
)

relative_peak_values = dti_peaks.peak_values / np.maximum(
    dti_peaks.peak_values[..., :1], np.finfo(float).eps
)

_skyline = skyline(
    images=[(fa_data, affine, "FA")],
    peaks=[(dti_peaks.peak_dirs, affine, "DTI peaks", relative_peak_values)],
    out_stealth_png="skyline_dti_peaks.png",
)

###############################################################################
# Peaks saved to a ``.pam5`` file can be opened the same way from the command
# line with ``dipy_skyline file.pam5``.

###############################################################################
# Visualizing ODFs
# =================
#
# Horizon could only draw peaks. Skyline renders full ODF glyphs on the GPU
# directly from SH coefficients. We reuse the FORCE posterior ODFs computed for
# tracking above, stored as SH coefficients of order 8.
#
# SH coefficients are provided as tuples ``(coeffs, affine, name, basis)``;
# the name and basis are optional. FORCE coefficients use DIPY's legacy
# ``descoteaux07`` basis, which differs from the ``sh_basis`` default, so we
# declare it in the fourth tuple element. The glyph scale and opacity can be
# changed in the ODF panel.

_skyline = skyline(
    images=[(fa_data, affine, "FA")],
    sh_coeffs=[(force_pam.shm_coeff, affine, "FORCE ODFs", "descoteaux07_legacy")],
    out_stealth_png="skyline_force_odfs.png",
)

###############################################################################
# Synchronized Slices
# ====================
#
# Every image, peaks and ODF layer has a **Synchronize Slices** toggle, which
# is on by default. Moving a slice slider on one synchronized layer moves every
# other synchronized layer.
#
# Slice positions are exchanged in world coordinates through each layer's
# affine. FA and ODFs, or images of different resolutions, therefore stay
# aligned on the same anatomical plane.
#
# - The global slice controls in the sidebar move all synchronized layers at
#   once. Global axis visibility toggles apply to every slice layer, whether or
#   not it is synchronized.
# - Switching synchronization off on one layer freezes it. For example, FA can
#   stay on one axial slice while the ODFs or peaks are scrubbed through
#   others.
#
# Horizon had no per-layer synchronization.

###############################################################################
# Summary
# =======
#
# This tutorial covered:
#
# 1. Basic image visualization with Skyline
# 2. Visualizing tractography with clustering and cluster thresholds
# 3. Visualizing peak directions
# 4. Visualizing ODF glyphs from SH coefficients
# 5. Synchronizing slices across layers
#
# Key Parameters Summary
# ----------------------
#
# - ``visualizer_type``: Window kind: ``"standalone"``, ``"gui"``,
#   ``"jupyter"`` or ``"stealth"`` (offscreen)
# - ``images``: List of tuples ``(data, affine, name)``
# - ``tractograms``: List of tuples ``(sft, name)``
# - ``peaks``: List of tuples ``(peak_dirs, affine, name, peak_values)``
# - ``sh_coeffs``: List of tuples ``(coeffs, affine, name, basis_type)``; the
#   name and basis type are optional
# - ``sh_basis``: SH basis convention for ODF files and unlabelled arrays
# - ``rois``: List of tuples ``(roi, affine, name)``
# - ``surfaces``: List of tuples ``(vertices, faces, name)``
# - ``is_cluster``: Enable QuickBundlesX clustering (bool)
# - ``cluster_thr``: Distance threshold for clustering in mm (float)
# - ``cluster_size_thr``: Hide clusters with fewer streamlines (int)
# - ``cluster_length_thr``: Hide clusters with a shorter mean length in mm
#   (float)
# - ``tract_colors``: Tractogram coloring: ``"direction"``, ``"random"`` or an
#   RGB(A) tuple
# - ``bg_color``: Background color as an RGB tuple in ``[0, 1]``
# - ``glass_brain``: Render surfaces as a transparent glass brain (bool)
# - ``rgb``: Force (True) or disable (False) RGB mode for 4D images
# - ``out_stealth_png``: Output filename for the offscreen snapshot (str)
# - ``out_dir``: Directory for the offscreen snapshot (str or Path)
#
# For more details, see the command-line interface tutorial at
# :ref:`viz_flow`.
#
# References
# ----------
#
# .. footbibliography::
