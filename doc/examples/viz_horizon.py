"""
====================================================
Interactive Visualization with DIPY Horizon (Python)
====================================================

This tutorial demonstrates how to use DIPY Horizon programmatically in Python
code for interactive visualization of tractography data, brain images,
surfaces, and peak directions. Horizon is a powerful interactive medical
visualization tool introduced in :footcite:t:`Garyfallidis2019`.

While Horizon can be used from the command line (see :ref:`viz_flow`), this
example shows how to use it directly in Python scripts for more flexible
and programmatic visualization workflows.

What is Horizon?
----------------

Horizon is DIPY's interactive visualization system that allows you to:

- Visualize and interact with tractograms (streamlines)
- Display anatomical images (T1, T2, FA, etc.) as slices
- Cluster streamlines using QuickBundlesX
- Visualize peak directions and spherical harmonics
- Display brain surfaces (cortical meshes)
- Combine multiple data types in a single view
- Save snapshots of your visualizations

Getting Started
---------------

First, let's import the necessary modules and check that we have FURY
installed (required for 3D visualization):

"""

import sys

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    auto_response_ssst,
)
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import deterministic_tracking
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.viz import has_fury
from dipy.viz.horizon.app import Horizon

fury, has_fury, setup_module = optional_package("fury")

###############################################################################
# Interactive Mode Control
# =========================
#
# Set this flag to control whether to open interactive windows or save images.
# - True: Opens interactive 3D windows (close them manually to continue)
# - False: Saves PNG screenshots automatically (faster, good for documentation)

interactive = False  # Change this to True for interactive exploration!

###############################################################################

# Check if FURY is available
if not has_fury:
    logger.error("FURY is required for Horizon. Install it with: pip install fury")
    sys.exit(1)

from dipy.viz import horizon

###############################################################################
# Basic Example: Visualizing a Brain Image
# =========================================
#
# Let's start with the simplest use case: visualizing a 3D brain image.
# We'll load a T1-weighted image and display it using Horizon.

# Load example data
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)

# Extract b0 image (first volume)
b0_data = data[..., 0]

###############################################################################
# To visualize an image with Horizon, we need to provide it as a tuple
# containing (data, affine, optional_filename). The affine matrix defines
# the spatial orientation of the image.


# Prepare image data for Horizon
images = [(b0_data, affine, "b0.nii.gz")]

# Create visualization
horizon(
    images=images,
    world_coords=True,  # Use world coordinates (not voxel coordinates)
    interactive=interactive,  # Controlled by flag at top of script
    out_png="horizon_basic_image.png",
)

if not interactive:
    logger.info("Saved: horizon_basic_image.png")

###############################################################################
# Visualizing Multiple Images
# ============================
#
# Horizon can display multiple images simultaneously. This is useful for
# comparing different contrasts or viewing multiple subjects.

# Create FA map for comparison
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

logger.info("Computing tensor model...")
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data)
fa_data = tenfit.fa

# Visualize both b0 and FA
images = [
    (b0_data, affine, "b0.nii.gz"),
    (fa_data, affine, "fa.nii.gz"),
]

horizon(
    images=images,
    world_coords=True,
    interactive=interactive,
    out_png="horizon_multiple_images.png",
)

if not interactive:
    logger.info("Saved: horizon_multiple_images.png")

###############################################################################
# Visualizing Tractograms with Clustering
# ========================================
#
# One of Horizon's most powerful features is interactive tractography
# visualization with automatic clustering using QuickBundlesX.

logger.info("Generating streamlines...")

# Create a seed mask from white matter
# median_otsu returns a 4D masked data array, we need to extract a 3D mask
_, white_matter_mask = median_otsu(data, vol_idx=range(10, 50), numpass=3)

# Create seeds
seeds = utils.seeds_from_mask(white_matter_mask, affine, density=[2, 2, 2])

# we'll use a subset of seeds
np.random.shuffle(seeds)
seeds = seeds[:5000]  # Use 5000 seeds for demo

# Create stopping criterion from FA
stopping_criterion = ThresholdStoppingCriterion(fa_data, 0.2)

# Perform tractography using the modern functional tracker
# Build a CSD model to get SH coefficients for deterministic tracking
logger.info("Computing CSD model...")
response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=fa_data > 0.2)

# Generate streamlines with built-in length filtering
# min_len and max_len filter during tracking (no post-processing needed)
streamline_generator = deterministic_tracking(
    seeds,
    stopping_criterion,
    affine,
    step_size=0.5,
    min_len=10,  # Minimum length in mm (filters short streamlines)
    max_len=500,  # Maximum length in mm
    sh=csd_fit.shm_coeff,
    max_angle=30.0,
    sphere=default_sphere,
    nbr_threads=0,  # 0 = auto-detect cores; set to >0 for explicit threading
    return_all=False,
)

streamlines = Streamlines(streamline_generator)
logger.info(f"Generated {len(streamlines)} streamlines (min_len=10mm, max_len=500mm)")

# Create StatefulTractogram (required for Horizon)
# Using the image object directly ensures proper coordinate system handling
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)

###########################################################################
# Now visualize with clustering enabled. Clustering groups similar
# streamlines together, making it easier to explore large tractography
# datasets.

tractograms = [sft]

horizon(
    tractograms=tractograms,
    cluster=True,  # Enable QuickBundlesX clustering
    cluster_thr=15.0,  # Distance threshold in mm (smaller = more clusters)
    random_colors=False,  # Use consistent colors
    world_coords=True,
    interactive=interactive,
    out_png="horizon_tractography_clustered.png",
)

if not interactive:
    logger.info("Saved: horizon_tractography_clustered.png")

###############################################################################
# Combining Tractograms and Images
# =================================
#
# Horizon really shines when you combine multiple data types. Let's visualize
# streamlines overlaid on anatomical images.

# Visualize tractography with FA background
horizon(
    tractograms=tractograms,
    images=[(fa_data, affine, "fa.nii.gz")],
    cluster=True,
    cluster_thr=15.0,
    world_coords=True,
    interactive=interactive,
    out_png="horizon_combined.png",
)

if not interactive:
    logger.info("Saved: horizon_combined.png")

###############################################################################
# Filtering Streamlines by Length
# ================================
#
# Horizon allows you to filter clusters by various criteria, including
# streamline length.


# Show only streamlines longer than 30mm and shorter than 150mm
horizon(
    tractograms=tractograms,
    images=[(fa_data, affine, "fa.nii.gz")],
    cluster=True,
    cluster_thr=15.0,
    length_gt=30,  # Greater than 30mm
    length_lt=150,  # Less than 150mm
    world_coords=True,
    interactive=interactive,
    out_png="horizon_filtered_length.png",
)

if not interactive:
    logger.info("Saved: horizon_filtered_length.png")

###############################################################################
# Filtering by Cluster Size
# ==========================
#
# You can also filter based on the number of streamlines in each cluster.

# Show only clusters with at least 10 streamlines
horizon(
    tractograms=tractograms,
    images=[(fa_data, affine, "fa.nii.gz")],
    cluster=True,
    cluster_thr=15.0,
    clusters_gt=10,  # Clusters with > 10 streamlines
    world_coords=True,
    interactive=interactive,
    out_png="horizon_filtered_clusters.png",
)

if not interactive:
    logger.info("Saved: horizon_filtered_clusters.png")

# ##############################################################################
# Visualizing Peak Directions
# ============================

# .. note::
#     Peak visualization with Horizon requires peaks data with affine
#     information. The current implementation of `peaks_from_model` doesn't
#     include affine data in the returned `PeaksAndMetrics` object.
#     For peak visualization, you can manually add the affine to the peaks
#     object or save the peaks to disk and load them with the affine matrix.

# Example (commented out due to missing affine attribute):


# Create a simple mask for peak computation
mask = fa_data > 0.2

# Compute peaks from the tensor model
dti_peaks = peaks_from_model(
    tenmodel,
    data,
    default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=mask,
)

# Add affine to peaks for visualization
dti_peaks.affine = affine

# Visualize peaks with FA background
horizon(
    pams=[dti_peaks],
    images=[(fa_data, affine, "fa.nii.gz")],
    world_coords=True,
    interactive=interactive,
    out_png="horizon_dti_peaks.png",
)

if not interactive:
    logger.info("Saved: horizon_dti_peaks.png")

###############################################################################
# Interactive Mode
# ================
#
# All the examples above use `interactive=False` to automatically save images.
# For actual data exploration, set `interactive=True` to open an interactive
# window where you can:
#
# - Left click: select clusters
# - E: expand clusters
# - R: collapse all clusters
# - H: hide unselected clusters
# - I: invert selection
# - A: select all clusters
# - S: save selected streamlines to file
# - Y: open new window
# - O: hide/show control panel
#
# Mouse controls:
#
# - Shift + drag: move camera
# - Ctrl/Cmd + drag: rotate camera
# - Ctrl/Cmd + R: reset camera
#
# Note: The window will stay open until you close it manually.

###############################################################################
# Customizing Visualization
# ==========================
#
# Horizon provides many customization options:

horizon(
    tractograms=tractograms,
    images=[(fa_data, affine, "fa.nii.gz")],
    cluster=True,
    cluster_thr=10.0,  # Smaller threshold = more clusters
    random_colors="tracts",  # Random colors for each tractogram
    bg_color=(1, 1, 1),  # White background
    world_coords=True,
    interactive=interactive,
    out_png="horizon_customized.png",
)

if not interactive:
    logger.info("Saved: horizon_customized.png")

###############################################################################
# Programmatic Control: Using the Horizon Class
# ==============================================
#
# For advanced use cases, you can use the Horizon class directly instead of
# the convenience function. This gives you more control:

# Create Horizon instance
hz = Horizon(
    tractograms=tractograms,
    images=[(fa_data, affine, "fa.nii.gz")],
    cluster=True,
    cluster_thr=15.0,
    world_coords=True,
    interactive=interactive,
    out_png="horizon_class.png",
)

# Build the visualization
hz.build_scene()

# You can access the ShowManager for further customization
# show_m = hz.build_show(show_m=None)

if not interactive:
    logger.info("Saved: horizon_class.png")


###############################################################################
# Summary
# =======
#
# This tutorial covered:
#
# 1. Basic image visualization with Horizon
# 2. Displaying multiple images
# 3. Visualizing tractography with clustering
# 4. Combining tractograms and images
# 5. Filtering by length and cluster size
# 6. Visualizing peak directions (see code comments)
# 7. Interactive exploration with keyboard/mouse controls
# 8. Customization options (colors, thresholds, backgrounds)
# 9. Using the Horizon class directly for advanced control
#
# Key Parameters Summary
# ----------------------
#
# - `tractograms`: List of StatefulTractogram objects
# - `images`: List of tuples (data, affine, filename)
# - `pams`: List of PeakAndMetrics objects
# - `surfaces`: List of tuples (vertices, faces, filename)
# - `cluster`: Enable QuickBundlesX clustering (bool)
# - `cluster_thr`: Distance threshold for clustering in mm (float)
# - `length_gt/length_lt`: Filter streamlines by length (float)
# - `clusters_gt/clusters_lt`: Filter by cluster size (int)
# - `world_coords`: Use world coordinates vs voxel coordinates (bool)
# - `interactive`: Enable interactive window (bool)
# - `out_png`: Output filename for snapshot (str)
# - `bg_color`: Background color as RGB tuple (tuple)
# - `roi_images`: Display images as ROI contours (bool)
# - `random_colors`: Use random colors for tractograms/ROIs (str or bool)
#
# For more details, see the Horizon documentation and the command-line
# interface tutorial at :ref:`viz_flow`.
#
# See :footcite:p:`Garyfallidis2019` for further details about Horizon.
