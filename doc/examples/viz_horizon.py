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

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.viz import has_fury

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
    print("FURY is required for Horizon. Install it with: pip install fury")
else:
    from dipy.viz import horizon

###############################################################################
# Basic Example: Visualizing a Brain Image
# =========================================
#
# Let's start with the simplest use case: visualizing a 3D brain image.
# We'll load a T1-weighted image and display it using Horizon.

# Load example data
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
data, affine = load_nifti(hardi_fname)

# Extract b0 image (first volume)
b0_data = data[..., 0]

###############################################################################
# To visualize an image with Horizon, we need to provide it as a tuple
# containing (data, affine, optional_filename). The affine matrix defines
# the spatial orientation of the image.

if has_fury:
    # Prepare image data for Horizon
    # Format: list of tuples [(data, affine, filename)]
    images = [(b0_data, affine, "b0.nii.gz")]

    # Create visualization
    horizon(
        images=images,
        world_coords=True,  # Use world coordinates (not voxel coordinates)
        interactive=interactive,  # Controlled by flag at top of script
        out_png="horizon_basic_image.png",
    )

    if not interactive:
        print("Saved: horizon_basic_image.png")

###############################################################################
# Visualizing Multiple Images
# ============================
#
# Horizon can display multiple images simultaneously. This is useful for
# comparing different contrasts or viewing multiple subjects.

if has_fury:
    # Create FA map for comparison
    from dipy.io.gradients import read_bvals_bvecs

    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs=bvecs)

    # Compute DTI model
    print("Computing tensor model...")
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
        print("Saved: horizon_multiple_images.png")

###############################################################################
# Visualizing Tractograms with Clustering
# ========================================
#
# One of Horizon's most powerful features is interactive tractography
# visualization with automatic clustering using QuickBundlesX.

if has_fury:
    print("Generating streamlines...")

    # Create a seed mask from white matter
    # median_otsu returns a 4D masked data array, we need to extract a 3D mask
    _, white_matter_mask = median_otsu(data, vol_idx=range(10, 50), numpass=3)

    # Create seeds
    seeds = utils.seeds_from_mask(
        white_matter_mask, affine, density=[2, 2, 2]
    )

    # we'll use a subset of seeds
    np.random.shuffle(seeds)
    seeds = seeds[:5000]  # Use 5000 seeds for demo

    # Create stopping criterion from FA
    stopping_criterion = ThresholdStoppingCriterion(fa_data, 0.2)

    # Perform tractography using EuDX (simple and fast for demo)
    from dipy.direction import DeterministicMaximumDirectionGetter
    from dipy.io.gradients import read_bvals_bvecs

    # Get tensor model peaks
    peak_indices = tenfit.directions

    # Create a simple direction getter
    dg = DeterministicMaximumDirectionGetter.from_pmf(
        tenfit.odf(get_sphere(name="repulsion724")),
        max_angle=30.0,
        sphere=get_sphere(name="repulsion724"),
    )

    # Generate streamlines
    streamline_generator = LocalTracking(
        dg, stopping_criterion, seeds, affine, step_size=0.5
    )

    streamlines = Streamlines(streamline_generator)
    print(f"Generated {len(streamlines)} streamlines")

    # Keep only streamlines with reasonable length
    streamlines = [s for s in streamlines if len(s) > 10]
    print(f"Kept {len(streamlines)} streamlines after filtering")

    # Create StatefulTractogram (required for Horizon)
    # This ensures coordinate systems are properly handled
    from dipy.io.utils import create_nifti_header

    nii_header = create_nifti_header(affine, data.shape[:3], data.shape[:3])
    sft = StatefulTractogram(streamlines, nii_header, Space.RASMM)

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
        print("Saved: horizon_tractography_clustered.png")

###############################################################################
# Combining Tractograms and Images
# =================================
#
# Horizon really shines when you combine multiple data types. Let's visualize
# streamlines overlaid on anatomical images.

if has_fury:
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
        print("Saved: horizon_combined.png")

###############################################################################
# Filtering Streamlines by Length
# ================================
#
# Horizon allows you to filter clusters by various criteria, including
# streamline length.

if has_fury:
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
        print("Saved: horizon_filtered_length.png")

###############################################################################
# Filtering by Cluster Size
# ==========================
#
# You can also filter based on the number of streamlines in each cluster.

if has_fury:
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
        print("Saved: horizon_filtered_clusters.png")

###############################################################################
# Note: Advanced Peak Directions Visualization
# =============================================
#
# Horizon can also visualize peak directions from models like CSD using the
# `pams` parameter. This accepts PeakAndMetrics objects from
# `dipy.direction.peaks_from_model`. This is useful for quality control and
# understanding fiber orientations. Here's a quick example:
#
# .. code-block:: python
#
#     from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
#     from dipy.direction import peaks_from_model
#
#     # Fit CSD model and compute peaks
#     response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
#     csd_model = ConstrainedSphericalDeconvModel(gtab, response)
#     csd_peaks = peaks_from_model(csd_model, data, sphere, mask=mask)
#
#     # Visualize peaks with Horizon
#     horizon(
#         pams=[csd_peaks],
#         images=[(fa_data, affine, "fa.nii.gz")],
#         world_coords=True,
#         interactive=True,
#     )

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
# To enable interactive mode, uncomment the code below:
#
# This will open an interactive 3D window where you can:
# - Rotate, zoom, and pan the view
# - Select/deselect clusters by clicking
# - Use keyboard shortcuts to manipulate the display
#
# Uncomment these lines to try interactive mode:
#
# if has_fury:
#     horizon(
#         tractograms=tractograms,
#         images=[(fa_data, affine, "fa.nii.gz")],
#         cluster=True,
#         cluster_thr=15.0,
#         world_coords=True,
#         interactive=True,  # This enables the interactive window!
#     )
#
# Note: The window will stay open until you close it manually.

###############################################################################
# Customizing Visualization
# ==========================
#
# Horizon provides many customization options:

if has_fury:
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
        print("Saved: horizon_customized.png")

###############################################################################
# Programmatic Control: Using the Horizon Class
# ==============================================
#
# For advanced use cases, you can use the Horizon class directly instead of
# the convenience function. This gives you more control:

if has_fury:
    from dipy.viz.horizon.app import Horizon

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
        print("Saved: horizon_class.png")

###############################################################################
# Using DIPY Horizon in Python code
# =====================================
#
# Here's a complete, minimal example you can copy and run interactively.
# Save this as a separate script and run it to explore Horizon interactively:
#
# .. code-block:: python
#
#     from dipy.data import get_fnames
#     from dipy.io.image import load_nifti
#     from dipy.io.stateful_tractogram import Space, StatefulTractogram
#     from dipy.io.streamline import load_tractogram
#     from dipy.viz import horizon
#
#     # Load some example data
#     hardi_fname = get_fnames(name='stanford_hardi')[0]
#     data, affine = load_nifti(hardi_fname)
#     b0_data = data[..., 0]
#
#     # For tractograms, you can load your own .trk, .tck, or .trx file:
#     # sft = load_tractogram('your_tractogram.trk', 'same')
#     #
#     # Then visualize interactively:
#     # horizon(
#     #     tractograms=[sft],
#     #     images=[(b0_data, affine, 'b0.nii.gz')],
#     #     cluster=True,
#     #     cluster_thr=15.0,
#     #     interactive=True,  # This opens the interactive window!
#     # )
#     #
#     # Or just visualize the image:
#     horizon(
#         images=[(b0_data, affine, 'b0.nii.gz')],
#         interactive=True,
#     )
#
# The interactive window allows you to:
#
# - **Rotate**: Click and drag
# - **Zoom**: Mouse wheel or pinch
# - **Pan**: Shift + drag
# - **Reset view**: Ctrl/Cmd + R
# - **Toggle control panel**: Press O
# - **Take screenshot**: Use the camera button in the UI
#
# When visualizing tractography with clustering:
#
# - **Select clusters**: Left click on centroid
# - **Expand cluster**: Press E
# - **Hide unselected**: Press H
# - **Select all**: Press A
# - **Invert selection**: Press I
# - **Collapse all**: Press R
# - **Save selection**: Press S

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
# References
# ----------
#
# .. footbibliography::
